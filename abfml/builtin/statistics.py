"""General training statistics for species energy baselines."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from abfml.core.structure import AtomicSample


class EnergyShiftStatistics(nn.Module):
    """Estimate species energy shifts by least squares on elemental compositions.
    
    For structure b, E_b=sum_t N_bt*shift_t; accumulate normal equations across structures.
    finalize() returns the constructor's list[float]. Rank-deficient compositions use a least-
    squares solution, so callers should still ensure sufficient independent compositions.
    """

    def __init__(self, type_map: list[int]) -> None:
        super().__init__()
        if not type_map or len(set(type_map)) != len(type_map) or min(type_map) < 0:
            raise ValueError("type_map must contain unique non-negative atomic numbers")
        self.type_map = list(type_map)
        num_types = len(type_map)
        self.register_buffer("normal_matrix", torch.zeros(num_types, num_types, dtype=torch.float64))
        self.register_buffer("rhs", torch.zeros(num_types, dtype=torch.float64))
        self.register_buffer("num_samples", torch.zeros((), dtype=torch.long))
        type_lut = torch.full((max(type_map) + 1,), -1, dtype=torch.long)
        type_lut[torch.as_tensor(type_map, dtype=torch.long)] = torch.arange(num_types)
        self.register_buffer("type_lut", type_lut)

    @torch.no_grad()
    def update(self, atomic_numbers: torch.Tensor, energy: torch.Tensor) -> None:
        """Accumulate one structure's elemental composition and scalar total energy."""
        numbers = atomic_numbers.detach().to(
            device=self.type_lut.device,
            dtype=torch.long,
        )
        value = energy.detach().to(device=self.rhs.device, dtype=torch.float64)
        if numbers.ndim != 1 or numbers.numel() == 0:
            raise ValueError("atomic_numbers must be a non-empty [N] tensor")
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError("energy must be a finite scalar")
        if int(numbers.min()) < 0 or int(numbers.max()) >= self.type_lut.numel():
            raise ValueError("Structure contains an element absent from type_map")
        type_ids = self.type_lut[numbers]
        if bool((type_ids < 0).any()):
            raise ValueError("Structure contains an element absent from type_map")

        composition = torch.bincount(type_ids, minlength=len(self.type_map)).to(torch.float64)
        self.normal_matrix.add_(composition.unsqueeze(1) @ composition.unsqueeze(0))
        self.rhs.add_(composition * value)
        self.num_samples.add_(1)

    @torch.no_grad()
    def finalize(self) -> list[float]:
        if int(self.num_samples) == 0:
            raise ValueError("Cannot compute energy shifts from an empty dataset")
        solution = torch.linalg.lstsq(
            self.normal_matrix,
            self.rhs.unsqueeze(1),
        ).solution.reshape(-1)
        return [float(value) for value in solution.cpu()]


def energy_shifts(
    structures: Iterable[AtomicSample],
    type_map: list[int],
) -> list[float]:
    """Accumulate species energy baselines from training structures, each requiring an energy
    label.
    """
    statistics = EnergyShiftStatistics(type_map)
    for structure in structures:
        if structure.energy is None:
            raise ValueError("Every structure must have an energy label")
        statistics.update(structure.atomic_numbers, structure.energy)
    return statistics.finalize()
