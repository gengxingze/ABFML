"""Lennard-Jones 12/6 force field on canonical CSR neighbors."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from abfml.config import ConfigError, as_float, check_unknown
from abfml.core.registry import ModelSpec, register_model
from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput
from abfml.builtin.functions import safe_reciprocal, segment_sum


class LennardJones(AtomicModel):
    """Pairwise 12/6 potential with 1/2 credit on each directed edge."""

    capabilities = ("energy", "atomic_energies", "forces", "virial")

    def __init__(self, type_map: list[int], cutoff: float, epsilon: float, sigma: float):
        super().__init__(type_map=type_map, cutoff=cutoff)
        self.epsilon = nn.Parameter(torch.tensor(float(epsilon)))
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))

    def compute(self, inputs: ModelInput) -> ModelOutput:
        _, edge_ptr, _, edge_vectors = inputs.require_neighbors()
        distances = edge_vectors.square().sum(dim=-1).clamp(min=1.0e-24).sqrt()
        valid = distances > 1.0e-12
        reciprocal = safe_reciprocal(distances, valid)
        scaled = self.sigma * reciprocal
        edge_energies = 2.0 * self.epsilon * (scaled.pow(12) - scaled.pow(6))
        atomic_energies = segment_sum(edge_energies, edge_ptr)
        return ModelOutput(
            energy=atomic_energies.sum(),
            atomic_energies=atomic_energies,
        )


# ---------------------------------------------------------------------------
# Keep model declarations and configuration parsing in the same file.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LennardJonesConfig:
    type_map: tuple[int, ...]
    cutoff: float
    epsilon: float
    sigma: float

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, type_map: tuple[int, ...], path: str = "model"
    ) -> "LennardJonesConfig":
        check_unknown(raw, {"name", "cutoff", "epsilon", "sigma"}, path)
        cutoff = as_float(raw.get("cutoff"), f"{path}.cutoff")
        if cutoff <= 0:
            raise ConfigError(f"{path}.cutoff", "must be positive")
        epsilon = as_float(raw.get("epsilon"), f"{path}.epsilon")
        sigma = as_float(raw.get("sigma"), f"{path}.sigma")
        if sigma <= 0:
            raise ConfigError(f"{path}.sigma", "must be positive")
        return cls(
            type_map=type_map,
            cutoff=cutoff,
            epsilon=epsilon,
            sigma=sigma,
        )

    def model_kwargs(self) -> dict:
        return {
            "type_map": list(self.type_map),
            "cutoff": self.cutoff,
            "epsilon": self.epsilon,
            "sigma": self.sigma,
        }


# Bind model and configuration here; the registry remains implementation-independent.
register_model("lennard_jones", ModelSpec(LennardJones, LennardJonesConfig))
