"""Atomic checkpoint saving, loading, and training-state restoration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import random
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from abfml.core.model import AtomicModel

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: AtomicModel,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    step: int,
    config: Mapping[str, object],
    metrics: Mapping[str, float],
    best_metric: float | None,
    data_state: dict | None = None,
) -> None:
    """Save complete training state through a temporary file and atomic replacement."""
    from abfml.serialization import module_files

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "abfml.checkpoint",
        "version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "step": int(step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "config": dict(config),
        "model_module_files": module_files(config),
        "metrics": dict(metrics),
        "best_metric": best_metric,
        "data_state": data_state,
        "rng": {
            "python": random.getstate(),
            "numpy": (np.random.get_state()[0], np.random.get_state()[1].tolist(),
                      *np.random.get_state()[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict:
    """Load and validate checkpoint version and training-progress fields."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    if payload.get("format") != "abfml.checkpoint" or payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {payload.get('version')}"
        )
    required = {"epoch", "step", "model_state", "optimizer_state", "config"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    if payload["epoch"] < 0 or payload["step"] < 0:
        raise ValueError("Checkpoint epoch and step must be non-negative")
    return payload


def restore_state(
    checkpoint: Mapping[str, object],
    model: AtomicModel,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> tuple[int, int]:
    """Restore training state and return completed (epoch, global_step)."""
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler_state = checkpoint.get("scheduler_state")
    if (scheduler is None) != (scheduler_state is None):
        raise ValueError("Checkpoint scheduler does not match the training configuration")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    numpy_state = rng["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng["cuda"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return int(checkpoint["epoch"]), int(checkpoint["step"])
