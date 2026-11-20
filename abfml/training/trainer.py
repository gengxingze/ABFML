"""Training/validation loops and their error accumulators."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput, collect_outputs
from abfml.core.schema import LABELS, normalize_label
from abfml.logger import format_progress as _format_progress
from abfml.logger import get_logger


logger = get_logger("abfml.training")


def _error_statistics(
    output: ModelOutput,
    batch: tuple[ModelInput, dict[str, torch.Tensor]],
    selected=None,
) -> dict[str, tuple[float, int]]:
    """Return squared-error sums and scalar counts for matching predictions and labels."""
    statistics: dict[str, tuple[float, int]] = {}
    inputs, labels = batch
    predictions = collect_outputs(output)
    for name in labels:
        if selected is not None and name not in selected:
            continue
        prediction = predictions.get(name)
        label = labels.get(name)
        if prediction is None or label is None:
            continue
        if name == "energy" and prediction.ndim == 0 and label.ndim == 1:
            label = label.sum()
        if prediction.shape != label.shape:
            raise ValueError(
                f"Metric '{name}' requires matching shapes, got "
                f"{prediction.shape} and {label.shape}"
            )
        residual = prediction.detach() - label
        if residual.numel() == 0:
            continue
        statistics[name] = (float(residual.square().sum()), residual.numel())

        atom_counts = inputs.atom_counts.to(
            dtype=residual.dtype, device=residual.device
        )
        spec = LABELS.get(name)
        if spec is not None and spec.per_atom:
            per_atom = normalize_label(name, residual, atom_counts)
            statistics[spec.per_atom] = (float(per_atom.square().sum()), per_atom.numel())
    return statistics


def _rmse_metrics(statistics: dict[str, tuple[float, int]]) -> dict[str, float]:
    return {
        name + "_rmse": math.sqrt(squared_sum / count)
        for name, (squared_sum, count) in statistics.items()
    }


class _RunningMetrics:
    """Shared batch-loss and component-error accumulator for training and validation."""

    def __init__(self) -> None:
        self.batch_count = 0
        self.loss_sum = 0.0
        self.component_sums: dict[str, float] = {}
        self.error_sums: dict[str, float] = {}
        self.error_counts: dict[str, int] = {}
        self.total_time = 0.0

    def update(
        self,
        loss: torch.Tensor,
        components: dict[str, torch.Tensor],
        errors: dict[str, tuple[float, int]],
        elapsed: float,
    ) -> None:
        self.batch_count += 1
        self.loss_sum += float(loss.detach())
        for name, value in components.items():
            self.component_sums[name] = (
                self.component_sums.get(name, 0.0) + float(value.detach())
            )
        for name, (squared_sum, count) in errors.items():
            self.error_sums[name] = self.error_sums.get(name, 0.0) + squared_sum
            self.error_counts[name] = self.error_counts.get(name, 0) + count
        self.total_time += elapsed

    def result(self) -> dict[str, float]:
        if self.batch_count == 0:
            raise ValueError("Cannot compute metrics for an empty data_load")
        result = {
            "loss": self.loss_sum / self.batch_count,
            "time": self.total_time,
        }
        for name, value in self.component_sums.items():
            result["loss_" + name] = value / self.batch_count
        result.update(_rmse_metrics({
            name: (squared_sum, self.error_counts[name])
            for name, squared_sum in self.error_sums.items()
        }))
        return result


def train_loop(
    data_load: Iterable[tuple[ModelInput, dict[str, torch.Tensor]]],
    model: AtomicModel,
    optimizer: Optimizer,
    iters_step: int,
    loss_fn: Callable[
        [ModelOutput, tuple[ModelInput, dict[str, torch.Tensor]]],
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ],
    *,
    scheduler: LRScheduler | None = None,
    gradient_clip_norm: float | None = None,
    print_freq: int = 1,
) -> tuple[dict[str, float], int]:
    """Train one epoch and return losses, per-label RMSE, and updated global step.
    
    loss_fn returns total and component losses from outputs and labels. Losses are arithmetic
    means over batches; actual RMSE accumulates squared errors across all epoch components,
    independent of batch size. Extended labels follow the same rule.
    """
    if iters_step < 0:
        raise ValueError("iters_step must be non-negative")
    if print_freq < 1:
        raise ValueError("print_freq must be positive")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    parameter = next(model.parameters(), None)
    if parameter is None or not parameter.is_floating_point():
        raise ValueError("Training requires at least one floating-point model parameter")
    dtype = parameter.dtype
    device = parameter.device

    model.train()
    running = _RunningMetrics()
    current_step = iters_step

    for iters, raw_batch in enumerate(data_load):
        time_start = time.perf_counter()
        batch = (raw_batch[0].to(device=device, dtype=dtype),
                     {name: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
                      for name, value in raw_batch[1].items()})
        optimizer.zero_grad()
        output = model(batch[0])
        loss, components = loss_fn(output, batch)
        error_statistics = _error_statistics(output, batch, selected=components)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Training loss is not finite")
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        current_step = iters_step + iters + 1

        running.update(
            loss,
            components,
            error_statistics,
            time.perf_counter() - time_start,
        )
        if iters % print_freq == 0:
            logger.info(
                "| Train Iters:%6d , loss: %.2e%s , lr: %.2e",
                current_step,
                float(loss.detach()),
                _format_progress(_rmse_metrics(error_statistics)),
                optimizer.param_groups[0]["lr"],
            )

    if running.batch_count == 0:
        raise ValueError("Cannot train with an empty data_load")
    return running.result(), current_step


def _tensor_options(model: AtomicModel) -> tuple[torch.dtype, torch.device]:
    """Infer computation dtype and device from parameters or buffers."""
    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.is_floating_point():
        return parameter.dtype, parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None and buffer.is_floating_point():
        return buffer.dtype, buffer.device
    return torch.get_default_dtype(), torch.device("cpu")


def valid_loop(
    data_load: Iterable[tuple[ModelInput, dict[str, torch.Tensor]]],
    model: AtomicModel,
    loss_fn: Callable[
        [ModelOutput, tuple[ModelInput, dict[str, torch.Tensor]]],
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ],
    *,
    print_freq: int = 1,
) -> dict[str, float]:
    """Evaluate losses and actual RMSE for matching labels without updating parameters.
    
    Keep autograd enabled to derive forces and virial from energy, without loss backward or
    optimizer steps. Average losses over batches and aggregate RMSE over all scalar components.
    """
    if print_freq < 1:
        raise ValueError("print_freq must be positive")
    dtype, device = _tensor_options(model)

    model.eval()
    running = _RunningMetrics()

    with torch.enable_grad():
        for iters, raw_batch in enumerate(data_load):
            time_start = time.perf_counter()
            batch = (raw_batch[0].to(device=device, dtype=dtype),
                     {name: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
                      for name, value in raw_batch[1].items()})
            output = model(batch[0])
            loss, components = loss_fn(output, batch)
            error_statistics = _error_statistics(output, batch, selected=components)
            running.update(
                loss,
                components,
                error_statistics,
                time.perf_counter() - time_start,
            )

            if iters % print_freq == 0:
                logger.info(
                    "| Valid Iters:%6d , loss: %.2e%s",
                    iters + 1,
                    float(loss.detach()),
                    _format_progress(_rmse_metrics(error_statistics)),
                )

    if running.batch_count == 0:
        raise ValueError("Cannot evaluate with an empty data_load")
    return running.result()


def fit(
    model: AtomicModel,
    train_dataloader: Iterable[tuple[ModelInput, dict[str, torch.Tensor]]],
    optimizer: Optimizer,
    loss_fn: Callable[
        [ModelOutput, tuple[ModelInput, dict[str, torch.Tensor]]],
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ],
    *,
    epochs: int,
    valid_dataloader: Iterable[tuple[ModelInput, dict[str, torch.Tensor]]] | None = None,
    scheduler: LRScheduler | None = None,
    gradient_clip_norm: float | None = None,
    print_freq: int = 1,
    validation_loss: Callable | None = None,
    initial_epoch: int = 0,
    initial_step: int = 0,
    epoch_callback: Callable[[int, dict[str, float | int], int], None] | None = None,
) -> tuple[list[dict[str, float | int]], int]:
    """Train additional epochs and return metric history and final optimization step.
    
    initial_epoch and initial_step support resumption. After each epoch's logging,
    epoch_callback receives (epoch, metrics, step) for checkpoint saving. This function does not
    write files, export models, or parse configuration.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if initial_epoch < 0 or initial_step < 0:
        raise ValueError("initial_epoch and initial_step must be non-negative")

    history: list[dict[str, float | int]] = []
    step = initial_step
    for epoch_offset in range(1, epochs + 1):
        epoch = initial_epoch + epoch_offset
        train_metrics, step = train_loop(
            train_dataloader,
            model,
            optimizer,
            step,
            loss_fn,
            scheduler=scheduler,
            gradient_clip_norm=gradient_clip_norm,
            print_freq=print_freq,
        )
        metrics: dict[str, float | int] = {
            "epoch": int(epoch),
            **{f"train_{name}": value for name, value in train_metrics.items()},
        }
        if valid_dataloader is not None:
            valid_metrics = valid_loop(
                valid_dataloader, model, validation_loss or loss_fn, print_freq=print_freq
            )
            metrics.update(
                {f"valid_{name}": value for name, value in valid_metrics.items()}
            )
        metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])

        history.append(metrics)
        validation = ""
        if valid_dataloader is not None:
            validation = (
                f" , valid loss: {metrics['valid_loss']:.2e}"
                + _format_progress(valid_metrics, prefix="valid ")
            )
        logger.info(
            "| Epoch:%3d , train loss: %.2e%s%s , time: %.1fs",
            epoch,
            train_metrics["loss"],
            _format_progress(train_metrics, prefix="train "),
            validation,
            train_metrics["time"],
        )
        if epoch_callback is not None:
            epoch_callback(epoch, metrics, step)
    return history, step
