"""Dynamic loss weights and name-matched MSE/RMSE supervision."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from abfml.core.types import ModelInput, ModelOutput, collect_outputs
from abfml.core.schema import normalize_label


def calculate_weight(
    start_weights: Mapping[str, float],
    limit_weights: Mapping[str, float],
    lr_ratio: float,
) -> dict[str, float]:
    """Linearly interpolate start and limit weights using lr_ratio.
    
    lr_ratio=1 returns start_weights; lr_ratio=0 returns limit_weights. Both mappings must
    contain exactly the same supervised label names.
    """
    if not 0.0 <= lr_ratio <= 1.0:
        raise ValueError("lr_ratio must be in [0, 1]")
    if set(start_weights) != set(limit_weights) or not start_weights:
        raise ValueError("Start and limit weights must contain the same non-empty terms")
    result: dict[str, float] = {}
    for name, start_weight in start_weights.items():
        limit_weight = limit_weights[name]
        if (
            not math.isfinite(start_weight)
            or not math.isfinite(limit_weight)
            or start_weight < 0
            or limit_weight < 0
        ):
            raise ValueError("Loss weights must be finite and non-negative")
        result[name] = (
            limit_weight * (1.0 - lr_ratio) + start_weight * lr_ratio
        )
    return result


def calculate_loss(
    output: ModelOutput,
    batch: tuple[ModelInput, dict[str, torch.Tensor]],
    config: Mapping[str, object],
    *,
    lr_ratio: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute name-matched supervision with one error metric and dynamic weights.
    
    config contains metric, label, start_weight, and limit_weight. The last three lists align by
    position; lr_ratio interpolates between endpoint weights. Standard outputs and properties
    share label-based lookup, so new labels require no branches here. Normalize total energy and
    virial by each structure's atom count before MSE/RMSE; forces and atomic energies already
    use atomic components. Extended labels keep their original scales. Return total weighted
    loss and unweighted component errors.
    """
    labels = config.get("label")
    start_values = config.get("start_weight")
    limit_values = config.get("limit_weight")
    if (
        not isinstance(labels, (list, tuple))
        or not labels
        or not all(isinstance(name, str) for name in labels)
    ):
        raise ValueError("loss.label must be a non-empty list of names")
    if len(set(labels)) != len(labels):
        raise ValueError("loss.label must not contain duplicate names")
    if not isinstance(start_values, (list, tuple)) or not isinstance(
        limit_values, (list, tuple)
    ):
        raise ValueError("start_weight and limit_weight must be lists")
    if len(labels) != len(start_values) or len(labels) != len(limit_values):
        raise ValueError("label, start_weight and limit_weight must have equal lengths")

    metric = str(config.get("metric", "mse")).lower()
    if metric not in {"mse", "rmse"}:
        raise ValueError(f"Unsupported loss metric: {metric}")
    try:
        start_list = [float(value) for value in start_values]
        limit_list = [float(value) for value in limit_values]
    except (TypeError, ValueError) as error:
        raise ValueError("Loss weights must be numeric") from error
    weights = calculate_weight(
        dict(zip(labels, start_list)),
        dict(zip(labels, limit_list)),
        lr_ratio,
    )

    inputs, references = batch
    predictions = collect_outputs(output)

    total: torch.Tensor | None = None
    components: dict[str, torch.Tensor] = {}
    for name in labels:
        prediction = predictions.get(name)
        label = references.get(name)
        if weights[name] == 0:
            continue
        if prediction is None or label is None:
            raise ValueError(f"Configured loss term '{name}' is missing a prediction or label")
        # A total-energy-only model may return one batch scalar; sum its reference labels accordingly.
        if name == "energy" and prediction.ndim == 0 and label.ndim == 1:
            label = label.sum()
        if prediction.shape != label.shape:
            raise ValueError(
                f"Loss '{name}' requires matching shapes, got "
                f"{prediction.shape} and {label.shape}"
            )
        residual = prediction - label
        residual = normalize_label(name, residual, inputs.atom_counts.to(dtype=residual.dtype, device=residual.device))
        mse = residual.square().mean()
        value = mse if metric == "mse" else torch.sqrt(mse + 1.0e-24)
        components[name] = value
        if total is None:
            total = prediction.new_zeros(())
        total = total + weights[name] * value

    if total is None:
        raise ValueError("No configured loss term has both a prediction and a label")
    return total, components
