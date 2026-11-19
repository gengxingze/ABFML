"""Independent evaluation with component-wise MAE/RMSE and per-structure predictions."""

import math
from time import perf_counter
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
from ase.io.extxyz import output_column_format

from abfml.core.types import collect_outputs
from abfml.core.schema import LABELS, label_unit, normalize_label
from abfml.data import ASEReader, AtomicDataset, create_dataloader
from abfml.serialization import load_model
from abfml.logger import get_logger, format_evaluation
from abfml.data.reader import to_ase

logger = get_logger("abfml.evaluation")

def _frame_values(values, index, start, end):
    result = {}
    for name, value in values.items():
        spec = LABELS.get(name)
        if spec is None or spec.source is not None:
            continue
        if spec.scope == "structure":
            result[name] = value[index].detach().cpu().numpy()
        elif spec.scope in {"atom", "center"}:
            result[name] = value[start:end].detach().cpu().numpy()
    return result


def _write_frame(handle, sample, references, predictions):
    """Use distinct reference/prediction prefixes while preserving geometry, shapes, and double
    precision.
    """
    atoms = to_ase(sample)
    shapes = {}
    for prefix, values in (("ref", references), ("pred", predictions)):
        for name, value in values.items():
            key = f"{prefix}_{name}"
            shapes[key] = list(value.shape)
            if LABELS[name].scope in {"atom", "center"}:
                atoms.new_array(key, value.reshape(len(atoms), -1) if value.ndim > 2 else value)
            else:
                atoms.info[key] = value.item() if value.ndim == 0 else value
    atoms.info["abfml_shapes"] = shapes
    arrays = {"symbols": np.asarray(atoms.get_chemical_symbols()), "positions": atoms.positions,
              **{key: value for key, value in atoms.arrays.items() if key not in {"numbers", "positions"}}}
    # Use ASE's standard header generator with 17 significant digits for atomic floating columns.
    header, _, _, fmt = output_column_format(atoms, list(arrays), arrays)
    fmt = fmt.replace("%16.8f", "%.16e")
    handle.write(f"{len(atoms)}\n{header}\n")
    for index in range(len(atoms)):
        row = tuple(item for value in arrays.values() for item in np.asarray(value[index]).ravel())
        handle.write(fmt % row)


def evaluate(source, files, *, output_dir=None, batch_size=1, device="cpu", dtype=None,
             labels=None, file_format=None):
    """Evaluate independently of training configuration; explicit labels must exist in every batch.
    
    Without labels, collect matching references and predictions and report scalar counts. Report
    total and per-atom energy errors separately; forces use Cartesian components.
    """
    started = perf_counter()
    model = load_model(source, device=device, dtype=dtype)
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else (dtype or torch.float64)
    logger.info("Evaluation | Model: %s | Device: %s | Precision: %s | Batch size: %d",
                type(model).__name__, device, dtype, batch_size)
    logger.info("Input: %s", files)
    dataset = AtomicDataset(ASEReader(files, file_format=file_format, dtype=dtype))
    logger.info("Dataset: %d structure(s) | Labels: %s", len(dataset),
                ", ".join(labels) if labels is not None else "available reference/prediction pairs")
    report = evaluate_dataset(model, dataset, output_dir=output_dir, batch_size=batch_size,
                              device=device, dtype=dtype, labels=labels)
    if report["metrics"]:
        logger.info("Evaluation results:\n%s", format_evaluation(report))
        logger.info("Rel. (%) = 100 * RMSE / mean(abs(reference)); Count counts scalar components")
    else:
        logger.info("Prediction only: no matched reference labels; error metrics unavailable")
    report["elapsed_seconds"] = perf_counter() - started
    logger.info("Evaluation completed | Structures: %d | Total time: %.2f s",
                report["num_structures"], report["elapsed_seconds"])
    return report


def evaluate_dataset(model, dataset, *, output_dir=None, batch_size=1, device="cpu",
                     dtype=None, labels=None, provider=None, pairs=None):
    """Optionally collect pairs in memory; standalone evaluation saves one extxyz and one panel
    figure.
    """
    model.eval()
    if dtype is None:
        floating = next((value for value in (*model.parameters(), *model.buffers())
                         if value.is_floating_point()), None)
        dtype = floating.dtype if floating is not None else torch.float64
    if not len(dataset):
        raise ValueError("Cannot evaluate an empty dataset")
    loader = create_dataloader(dataset, provider, model=model, batch_size=batch_size,
                               shuffle=False, drop_last=False)
    out = Path(output_dir) if output_dir is not None else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    statistics = {}
    offset = 0

    def accumulate(name, residual, reference):
        value = residual.detach().double()
        target = reference.detach().double()
        if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(target).all()):
            raise ValueError(f"Non-finite values in {name}")
        absolute, squared, reference_absolute, count = statistics.get(name, (0., 0., 0., 0))
        statistics[name] = (absolute + value.abs().sum().item(),
                            squared + value.square().sum().item(),
                            reference_absolute + target.abs().sum().item(), count + value.numel())

    from abfml.tools.parity import collect_pairs, plot_predictions
    collected = pairs if pairs is not None else ({} if out else None)
    context = (out / "predictions.extxyz").open("w", encoding="utf-8") if out else nullcontext(None)
    with context as handle, torch.enable_grad():
        for raw_inputs, references in loader:
            inputs = raw_inputs.to(device=device, dtype=dtype)
            predictions = {name: value.detach().cpu() for name, value in
                           collect_outputs(model(inputs)).items()}
            for name, value in predictions.items():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Non-finite predictions: {name}")
            for name in labels if labels is not None else references.keys():
                if name not in predictions or name not in references:
                    if labels is not None:
                        raise ValueError(f"Requested label '{name}' lacks prediction or reference")
                    continue
                prediction, reference = predictions[name], references[name]
                if prediction.shape != reference.shape:
                    raise ValueError(f"Evaluation shape mismatch: {name}")
                residual = prediction - reference
                accumulate(name, residual, reference)
                counts = inputs.atom_counts.cpu().to(dtype)
                spec = LABELS.get(name)
                if spec is not None and spec.per_atom:
                    accumulate(spec.per_atom, normalize_label(name, residual, counts),
                               normalize_label(name, reference, counts))
            if handle or collected is not None:
                for i in range(inputs.num_structures):
                    start, end = int(inputs.atom_ptr[i]), int(inputs.atom_ptr[i + 1])
                    frame_ref = _frame_values(references, i, start, end)
                    frame_pred = _frame_values(predictions, i, start, end)
                    if labels is not None:
                        frame_ref = {name: value for name, value in frame_ref.items() if name in labels}
                    if collected is not None:
                        collect_pairs(collected, frame_ref, frame_pred, end-start)
                    if handle:
                        _write_frame(handle, dataset[offset+i], frame_ref, frame_pred)
            offset += inputs.num_structures
    report = {"num_structures": len(dataset), "metrics": {
        name: {"mae": absolute / count, "rmse": math.sqrt(squared / count),
               "reference_mean_abs": reference_absolute / count,
               "relative_error_percent": (100. * math.sqrt(squared / count) / (reference_absolute / count)
                                          if reference_absolute > 0 else None),
               "count": count, "unit": label_unit(name)}
        for name, (absolute, squared, reference_absolute, count) in statistics.items() if count
    }}
    if out:
        logger.info("Predictions saved to %s", (out / "predictions.extxyz").resolve())
        if collected:
            plot_predictions({"Evaluation": collected}, output=out)
            logger.info("Prediction panel saved to %s", (out / "parity.png").resolve())
        else:
            logger.info("Prediction plots skipped: no matched supported reference labels")
    return report
