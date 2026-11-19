"""Minimal external-model example: a harmonic pair potential.

Load this module through model_modules without modifying ABFML. A bidirectional neighbor graph
assigns half of each pair energy to each directed edge:

    E = sum_edges k/4 * (r - r0)^2

This file demonstrates the complete extension contract: model class, configuration parser, and
register_model binding.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from abfml.config import ConfigError, as_float, check_unknown
from abfml.core.model import AtomicModel
from abfml.core.registry import ModelSpec, register_model
from abfml.core.types import ModelInput, ModelOutput
from abfml.builtin.functions import segment_sum


class HarmonicPair(AtomicModel):
    """Pairwise harmonic potential with 1/2 credit on each directed edge."""

    capabilities = ("energy", "atomic_energies", "forces", "virial")

    def __init__(self, type_map: list[int], cutoff: float,
                 stiffness: float, equilibrium: float):
        super().__init__(type_map=type_map, cutoff=cutoff)
        self.stiffness = nn.Parameter(torch.tensor(float(stiffness)))
        self.equilibrium = nn.Parameter(torch.tensor(float(equilibrium)))

    def compute(self, inputs: ModelInput) -> ModelOutput:
        _, edge_ptr, _, edge_vectors = inputs.require_neighbors()
        distances = edge_vectors.square().sum(dim=-1).clamp(min=1.0e-24).sqrt()
        edge_energies = 0.25 * self.stiffness * (distances - self.equilibrium) ** 2
        atomic_energies = segment_sum(edge_energies, edge_ptr)
        return ModelOutput(
            energy=atomic_energies.sum(),
            atomic_energies=atomic_energies,
        )


@dataclass(frozen=True)
class HarmonicPairConfig:
    type_map: tuple[int, ...]
    cutoff: float
    stiffness: float
    equilibrium: float

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, type_map: tuple[int, ...], path: str = "model"
    ) -> "HarmonicPairConfig":
        check_unknown(raw, {"name", "cutoff", "stiffness", "equilibrium"}, path)
        cutoff = as_float(raw.get("cutoff"), f"{path}.cutoff")
        if cutoff <= 0:
            raise ConfigError(f"{path}.cutoff", "must be positive")
        stiffness = as_float(raw.get("stiffness"), f"{path}.stiffness")
        equilibrium = as_float(raw.get("equilibrium"), f"{path}.equilibrium")
        if equilibrium <= 0:
            raise ConfigError(f"{path}.equilibrium", "must be positive")
        return cls(
            type_map=type_map,
            cutoff=cutoff,
            stiffness=stiffness,
            equilibrium=equilibrium,
        )

    def model_kwargs(self) -> dict:
        return {
            "type_map": list(self.type_map),
            "cutoff": self.cutoff,
            "stiffness": self.stiffness,
            "equilibrium": self.equilibrium,
        }


register_model("harmonic_pair", ModelSpec(HarmonicPair, HarmonicPairConfig))
