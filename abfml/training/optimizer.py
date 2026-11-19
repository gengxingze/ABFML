"""Named optimizer/scheduler construction and learning rates indexed by global optimization step."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR


def adjust_lr(
    start_lr: float,
    limit_lr: float,
    decay_rate: float,
    decay_step: int,
    iters_step: int,
    *,
    type_name: str = "exp",
) -> float:
    """Compute continuous or staircase exponential learning-rate decay with a lower bound.
    
    exp: start_lr * decay_rate ** (iters_step / decay_step).
    exp_decay: start_lr * decay_rate ** (iters_step // decay_step).
    """
    if start_lr <= 0:
        raise ValueError("start_lr must be positive")
    if limit_lr < 0 or limit_lr > start_lr:
        raise ValueError("limit_lr must be in [0, start_lr]")
    if not 0 < decay_rate <= 1:
        raise ValueError("decay_rate must be in (0, 1]")
    if decay_step < 1 or iters_step < 0:
        raise ValueError("decay_step must be positive and iters_step non-negative")

    if type_name == "exp":
        exponent = iters_step / decay_step
    elif type_name == "exp_decay":
        exponent = iters_step // decay_step
    else:
        raise ValueError(f"Unsupported learning rate type: {type_name}")
    learning_rate = start_lr * decay_rate ** exponent
    return max(learning_rate, limit_lr)


_OPTIMIZER_FACTORIES: dict[str, Callable[..., Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def build_optimizer(
    model: nn.Module,
    name: str = "adam",
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> Optimizer:
    """Build an optimizer by case-insensitive name for requires_grad=True parameters.
    
    Pass remaining keyword arguments unchanged to the torch.optim class, such as Adam betas and
    eps.
    """
    factory = _OPTIMIZER_FACTORIES.get(str(name).lower())
    if factory is None:
        raise ValueError(
            f"Unsupported optimizer '{name}'. "
            f"Available: {sorted(_OPTIMIZER_FACTORIES)}"
        )
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a positive finite number")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be a non-negative finite number")

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("No trainable parameters: cannot build an optimizer")
    return factory(parameters, lr=learning_rate, weight_decay=weight_decay, **kwargs)


def build_scheduler(
    optimizer: Optimizer,
    name: str | None = None,
    *,
    start_lr: float | None = None,
    limit_lr: float = 0.0,
    decay_rate: float = 0.96,
    decay_step: int = 100,
    last_epoch: int = -1,
    **kwargs: Any,
) -> LRScheduler | None:
    """Build a scheduler that advances once per optimization step.
    
    None, an empty string, or 'none' selects a constant rate. Schedulers use optimization steps,
    matching train_loop's per-batch scheduler.step(). last_epoch follows PyTorch conventions: -1
    starts fresh; resumption supplies completed steps.
    
    exponential/exp: continuous exponential decay; exp_decay: staircase exponential decay; step:
    multiply by gamma every step_size steps, bounded by limit_lr; cosine: anneal to limit_lr
    over t_max steps.
    """
    if name is None or str(name).strip().lower() in {"", "none"}:
        return None
    key = str(name).strip().lower()

    initial_lr = float(optimizer.param_groups[0]["lr"])
    if start_lr is None:
        start_lr = initial_lr
    if start_lr <= 0:
        raise ValueError("start_lr must be positive")
    if limit_lr < 0 or limit_lr > start_lr:
        raise ValueError("limit_lr must be in [0, start_lr]")

    if key in {"exp", "exponential", "exp_decay"}:
        stepped = key == "exp_decay"

        def exponential(step: int) -> float:
            return (
                adjust_lr(
                    start_lr,
                    limit_lr,
                    decay_rate,
                    decay_step,
                    step,
                    type_name="exp_decay" if stepped else "exp",
                )
                / initial_lr
            )

        return LambdaLR(optimizer, lr_lambda=exponential, last_epoch=last_epoch)

    if key == "step":
        step_size = int(kwargs.get("step_size", decay_step))
        gamma = float(kwargs.get("gamma", decay_rate))
        if step_size < 1:
            raise ValueError("step_size must be positive")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        floor = limit_lr / initial_lr

        def stepwise(step: int) -> float:
            return max(gamma ** (step // step_size), floor)

        return LambdaLR(optimizer, lr_lambda=stepwise, last_epoch=last_epoch)

    if key == "cosine":
        t_max = int(kwargs.get("t_max", kwargs.get("decay_step", decay_step)))
        if t_max < 1:
            raise ValueError("t_max must be positive")
        floor = limit_lr / initial_lr

        def cosine(step: int) -> float:
            progress = min(step, t_max) / t_max
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=cosine, last_epoch=last_epoch)

    raise ValueError(f"Unsupported scheduler '{name}'")
