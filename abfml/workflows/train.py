"""Prepare data, initialize or restore models, train, and save standard model files."""

import hashlib
import json
import math
import platform
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from abfml.config import load_config
from abfml.core.registry import build_model, model_spec
from abfml.data import ASEReader, AtomicDataset, create_dataloader
from abfml.builtin.statistics import energy_shifts
from abfml.core.neighbor import configure_cache, resolve_builder
from abfml.serialization import load_model, save_model
from abfml.logger import LossLog, get_logger, format_summary, format_timing
from abfml.training import (
    build_optimizer,
    build_scheduler,
    calculate_loss,
    fit,
    load_checkpoint,
    restore_state,
    save_checkpoint,
)

logger = get_logger("abfml.workflow")
PARITY_LIMIT = 500_000


def _mark_stage(timing, stage):
    """Use non-overlapping stages; accumulate output intervals and synchronize CUDA only at stage
    boundaries.
    """
    device = timing["device"]
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    now = time.perf_counter()
    timing["durations"][timing["stage"]] += now - timing["started"]
    timing["stage"], timing["started"] = stage, now


def run(config_path, *, resume=None, fine_tune=None, epochs=None):
    """Train additional epochs and restore the caller's default floating dtype."""
    timing = {"started": time.perf_counter(), "stage": "Data & initialization", "device": None,
              "durations": dict.fromkeys(("Data & initialization", "Training & validation",
                                          "Final evaluation", "Result output"), 0.)}
    previous_dtype = torch.get_default_dtype()
    try:
        return _train(Path(config_path), resume=resume, fine_tune=fine_tune, epochs=epochs, timing=timing)
    finally:
        torch.set_default_dtype(previous_dtype)
        _mark_stage(timing, None)
        logger.info("Runtime breakdown:\n%s", format_timing(timing["durations"]))
        elapsed = sum(timing["durations"].values())
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("Total runtime: %02d:%02d:%02d (%.2f s)",
                    hours, minutes, seconds, elapsed)


def _train(config_path, *, resume, fine_tune, epochs, timing):
    if resume is not None and fine_tune is not None:
        raise ValueError("resume and fine_tune cannot be used together")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    total_epochs = config.training.epochs if epochs is None else int(epochs)
    if total_epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_default_dtype(getattr(torch, config.common.dtype))
    torch.manual_seed(config.common.seed)
    random.seed(config.common.seed)
    np.random.seed(config.common.seed)
    resume = resume or config.output.checkpoint.resume
    if resume is not None and fine_tune is not None:
        raise ValueError("fine_tune conflicts with configured resume")
    source = fine_tune or resume
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        saved = checkpoint["config"]
        if tuple(raw.get("model_modules", ())) != tuple(saved.get("model_modules", ())):
            raise ValueError("Resume configuration differs: model_modules; use fine-tune for changes")
        for key in ("common", "model", "data", "loss"):
            if raw[key] != saved[key]:
                raise ValueError(f"Resume configuration differs: {key}; use fine-tune for changes")
        for key in ("optimizer", "scheduler"):
            if raw["training"].get(key) != saved["training"].get(key):
                raise ValueError(f"Resume configuration differs: training.{key}")
        previous_monitor = saved["output"].get("checkpoint", {}).get("monitor", "valid_loss")
        if config.output.checkpoint.monitor != previous_monitor:
            raise ValueError("Resume configuration differs: output.checkpoint.monitor")

    def read(files):
        return AtomicDataset(ASEReader(files, file_format=config.data.format, dtype=getattr(torch, config.common.dtype)))

    if config.data.split:
        dataset = read(config.data.files)
        train_len = round(len(dataset) * config.data.split.train)
        train_set, valid_set = random_split(dataset, [train_len, len(dataset) - train_len],
            generator=torch.Generator().manual_seed(config.data.split.seed))
        indices = {"train": train_set.indices, "valid": valid_set.indices}
        files = config.data.files
    else:
        train_set, valid_set = read(config.data.train_files), read(config.data.valid_files)
        indices = {"train": list(range(len(train_set))), "valid": list(range(len(valid_set)))}
        files = config.data.train_files + config.data.valid_files
    if not len(train_set) or not len(valid_set):
        raise ValueError("Training and validation datasets must both be non-empty")
    data_state = {"indices": indices, "files": []}
    for path in files:
        with open(path, "rb") as handle:
            data_state["files"].append({"path": str(Path(path).resolve()),
                "sha256": hashlib.file_digest(handle, "sha256").hexdigest()})
    if checkpoint and checkpoint["data_state"] != data_state:
        raise ValueError("Resume dataset content or split differs from the checkpoint")

    builder = model_spec(config.model_name).builder
    provider = configure_cache(
        resolve_builder(builder, cutoff=config.model.cutoff), config.data.loader.neighbor_cache)
    options = config.data.loader.to_kwargs()
    train_loader = create_dataloader(train_set, provider, **options)
    valid_loader = create_dataloader(valid_set, provider, **{**options, "shuffle": False, "drop_last": False})
    if len(train_loader) == 0:
        raise ValueError("drop_last removed every training batch")

    device = config.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if source:
        model = load_model(source, device=device, dtype=getattr(torch, config.common.dtype))
        source_payload = torch.load(source, map_location="cpu", weights_only=True)
        if source_payload["config"]["model"] != raw["model"] or tuple(model.type_map) != config.common.type_map:
            raise ValueError("Loaded model architecture or element order differs from configuration")
    else:
        kwargs = config.model_kwargs()
        required = getattr(builder, "required_statistics", ())
        unknown = set(required) - {"energy_shift", "std_mean", "energy_zero", "q_scaler", "acsf_calibration"}
        if unknown:
            raise ValueError(f"Unsupported model statistics: {sorted(unknown)}")
        if "energy_shift" in required:
            kwargs["energy_shift"] = energy_shifts(train_set, list(config.common.type_map))
        if "std_mean" in required:
            kwargs["std_mean"] = builder.measure_normalization(train_set, provider=provider, **kwargs)
        model = build_model(config.model_name, **kwargs)
        if "acsf_calibration" in required:
            model.initialize_statistics(train_set, provider=provider)
        if "energy_zero" in required and kwargs.get("output_bias") is None:
            model.calibrate_energy(train_set)
        if "q_scaler" in required:
            model.calibrate_scaler(train_set, provider=provider)
        model.to(device)

    optimizer = build_optimizer(model, **config.training.optimizer.to_kwargs())
    scheduler = build_scheduler(optimizer, **config.training.scheduler.to_kwargs()) if config.training.scheduler else None
    initial_epoch, initial_step, best_metric = 0, 0, None
    if checkpoint:
        initial_epoch, initial_step = restore_state(checkpoint, model, optimizer, scheduler)
        best_metric = checkpoint["best_metric"]
    # Log actual model information after initialization/restoration, without dumping weight values.
    parameters = list(model.named_parameters())
    total = sum(value.numel() for _, value in parameters)
    trainable = sum(value.numel() for _, value in parameters if value.requires_grad)
    logger.info("Model: %s (%s)", config.model_name, type(model).__name__)
    logger.info("Model structure:\n%s", model)
    logger.info("Parameters: total=%s, trainable=%s, frozen=%s",
                f"{total:,}", f"{trainable:,}", f"{total - trainable:,}")
    floating = next((value for value in model.parameters() if value.is_floating_point()), None)
    if floating is None:
        floating = next((value for value in model.buffers() if value.is_floating_point()), None)
    actual_device = floating.device if floating is not None else torch.device(device)
    actual_dtype = floating.dtype if floating is not None else getattr(torch, config.common.dtype)
    timing["device"] = actual_device
    cpu_name = platform.processor() or platform.machine()
    # processor() is often empty on Linux; prefer the kernel-provided CPU model.
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    cpu_name = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
    logger.info("Hardware: OS=%s %s, arch=%s", platform.system(), platform.release(), platform.machine())
    logger.info("CPU: %s; logical CPUs=%s; PyTorch threads=%d, interop threads=%d",
                cpu_name, os.cpu_count(), torch.get_num_threads(), torch.get_num_interop_threads())
    # Visible logical CPUs differ from scheduler allocations; affinity helps interpret cluster performance.
    if hasattr(os, "sched_getaffinity"):
        try:
            logger.info("CPU affinity: %d logical CPUs", len(os.sched_getaffinity(0)))
        except OSError:
            pass
    logger.info("Runtime: Python=%s, PyTorch=%s", platform.python_version(), torch.__version__)
    logger.info("Compute: device=%s, dtype=%s", actual_device, actual_dtype)
    if actual_device.type == "cuda":
        gpu = torch.cuda.get_device_properties(actual_device)
        logger.info("GPU: %s; VRAM=%.2f GiB; capability=%d.%d; PyTorch CUDA=%s",
                    gpu.name, gpu.total_memory / (1024 ** 3), gpu.major, gpu.minor, torch.version.cuda)
    logger.info("Dataset: train=%d, valid=%d; device=%s, dtype=%s; start_epoch=%d, start_step=%d",
                len(train_set), len(valid_set), device, config.common.dtype,
                initial_epoch, initial_step)

    loss_config = config.loss.to_dict()

    def loss_fn(output, batch):
        ratio = 1.0
        schedule = config.training.scheduler
        if schedule and schedule.start_lr > schedule.limit_lr:
            ratio = (optimizer.param_groups[0]["lr"] - schedule.limit_lr) / (schedule.start_lr - schedule.limit_lr)
            ratio = min(1., max(0., ratio))
        return calculate_loss(output, batch, loss_config, lr_ratio=ratio)

    def validation_loss(output, batch):
        # Fix validation weights so best metrics are comparable across epochs.
        return calculate_loss(output, batch, loss_config, lr_ratio=0.0)

    out = Path(config.output.dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "data.json").write_text(json.dumps(data_state, indent=2), encoding="utf-8")
    loss_path = out / "loss.out"
    loss_log = LossLog(loss_path, append=checkpoint is not None and loss_path.exists())

    def epoch_callback(epoch, metrics, step):
        nonlocal best_metric
        improved = False
        ck = config.output.checkpoint
        if ck.keep_best:
            current = metrics[ck.monitor]
            if not math.isfinite(current):
                raise ValueError("Checkpoint monitor is not finite")
            improved = best_metric is None or current < best_metric
            if improved:
                best_metric = current
        state = dict(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                     step=step, config=raw, metrics=metrics, best_metric=best_metric, data_state=data_state)
        save_checkpoint(out / "last.pt", **state)
        if epoch % ck.save_every == 0:
            save_checkpoint(out / f"epoch_{epoch:04d}.pt", **state)
        if improved:
            save_checkpoint(out / "best.pt", **state)
        loss_log.append(metrics)

    _mark_stage(timing, "Training & validation")
    fit(model, train_loader, optimizer, loss_fn, epochs=total_epochs, valid_dataloader=valid_loader,
        validation_loss=validation_loss, scheduler=scheduler, gradient_clip_norm=config.training.gradient_clip_norm,
        print_freq=config.training.print_freq, initial_epoch=initial_epoch, initial_step=initial_step,
        epoch_callback=epoch_callback)
    _mark_stage(timing, "Result output")
    # The standard model stores the final epoch; best.pt retains the optimal validation checkpoint.
    model.eval()
    save_model(out / config.output.model_filename, model, raw)
    logger.info("Training completed; standard model saved to %s", out / config.output.model_filename)
    from abfml.tools import plot_loss
    plot_loss(loss_path, output=out, log=True)
    logger.info("Training curves saved to %s", out / "loss.png")
    _mark_stage(timing, "Final evaluation")
    from abfml.workflows.evaluate import evaluate_dataset
    best_path = out / "best.pt"
    if config.output.checkpoint.keep_best and best_path.is_file():
        summary_model = load_model(best_path, device=device, dtype=actual_dtype)
        summary_name = f"Best model: best.pt (monitor={config.output.checkpoint.monitor})"
    else:
        summary_model = model
        summary_name = "Final model (best checkpoint unavailable)"
    draw_parity = len(train_set) + len(valid_set) < PARITY_LIMIT
    grouped, reports = {}, {}
    for name, subset in (("Train", train_set), ("Valid", valid_set)):
        logger.info("Evaluating %s on %d %s structures", summary_name, len(subset), name)
        grouped[name] = {} if draw_parity else None
        reports[name] = evaluate_dataset(summary_model, subset, pairs=grouped[name],
                                        batch_size=config.data.loader.batch_size, device=device,
                                        dtype=actual_dtype, provider=provider)
    _mark_stage(timing, "Result output")
    if draw_parity:
        from abfml.tools import plot_predictions
        plot_predictions(grouped, output=out)
        logger.info("Train/valid prediction panel saved to %s", out / "parity.png")
    else:
        logger.info("Training prediction plots skipped: %d structures >= %d",
                    len(train_set) + len(valid_set), PARITY_LIMIT)
    logger.info("%s\n%s", summary_name, format_summary(reports))
    logger.info("Relative error: 100 * RMSE / mean(abs(reference)); zero mean: N/A")
    return model
