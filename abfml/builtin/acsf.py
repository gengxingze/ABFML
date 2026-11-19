"""ACSF atomic potentials with three exponential symmetry functions, species normalization, and
independent species networks.

Radial functions use Gaussian windows. Narrow angular terms include r_jk; wide terms do not.
Angular sums use distinct neighbor slots j<k and unordered species-pair channels. Basis
parameters and training statistics are buffers; fitting-network parameters are trainable.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations_with_replacement

import torch
from torch import nn
from torch.nn import functional as F

from abfml.config import (
    ConfigError,
    as_float,
    as_int,
    as_str,
    check_unknown,
    parse_network,
)
from abfml.core.registry import ModelSpec, register_model
from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput
from abfml.core.neighbor import to_dense
from abfml.builtin.networks import SpeciesFittingNetwork
from abfml.builtin.functions import type_lookup


class ACSFFeatureStatistics(nn.Module):
    """Accumulate sample-variance statistics for ACSF descriptors across batches.
    
    Statistics are computed per species and feature using float64. finalize() returns [std, avg]
    for direct model initialization.
    """

    def __init__(
        self,
        num_types: int,
        num_features: int,
    ) -> None:
        super().__init__()
        if num_types < 1 or num_features < 1:
            raise ValueError("num_types and num_features must be positive")
        self.num_types = int(num_types)
        self.num_features = int(num_features)
        self.register_buffer("count", torch.zeros(num_types, 1, dtype=torch.float64))
        self.register_buffer("sum", torch.zeros(num_types, num_features, dtype=torch.float64))
        self.register_buffer("sum_squares", torch.zeros(num_types, num_features, dtype=torch.float64))

    @torch.no_grad()
    def update(self, features: torch.Tensor, center_type_ids: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.num_features:
            raise ValueError("features must have shape [N, num_features]")
        if center_type_ids.shape != (features.shape[0],):
            raise ValueError("center_type_ids must have shape [N]")
        if features.shape[0] == 0:
            return
        values = features.detach().to(dtype=torch.float64)
        types = center_type_ids.detach().to(dtype=torch.long)
        if bool(((types < 0) | (types >= self.num_types)).any()):
            raise ValueError("center_type_ids contains an invalid species index")
        weights = F.one_hot(types, self.num_types).to(dtype=torch.float64)
        batch_count = weights.sum(dim=0, keepdim=True).transpose(0, 1)
        self.count.add_(batch_count)
        self.sum.add_(weights.transpose(0, 1) @ values)
        self.sum_squares.add_(weights.transpose(0, 1) @ values.square())

    @torch.no_grad()
    def finalize(self) -> list[torch.Tensor]:
        sample_count = self.count
        if bool((sample_count <= 1).any()):
            missing = (sample_count <= 1).reshape(-1).nonzero().reshape(-1)
            raise ValueError(
                "At least two samples are required for every species; "
                f"insufficient species indices: {missing.tolist()}"
            )
        mean = self.sum / sample_count
        variance = (
            (self.sum_squares - self.sum.square() / sample_count) / (sample_count - 1.0)
        ).clamp(min=0.0)
        std = variance.sqrt()
        return [std, mean]


def _cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Cosine cutoff: one at zero distance and zero at or beyond the cutoff."""
    return torch.where(distances < cutoff,
                       .5 * (torch.cos(math.pi * distances / cutoff) + 1.),
                       torch.zeros_like(distances))


@dataclass(frozen=True)
class RadialBasis:
    """One radial basis function; eta has inverse-length-squared units and rs has length units."""

    eta: float
    rs: float

    def __post_init__(self):
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be positive and finite")
        if not math.isfinite(self.rs) or self.rs < 0:
            raise ValueError("rs must be non-negative and finite")


@dataclass(frozen=True)
class AngularBasis(RadialBasis):
    """One angular basis function; integer zeta stabilizes second derivatives at collinear
    geometries.
    """

    lambd: float
    zeta: int

    def __post_init__(self):
        super().__post_init__()
        if self.lambd not in {-1., 1.}:
            raise ValueError("lambda must be -1 or 1")
        if (isinstance(self.zeta, bool) or not math.isfinite(self.zeta)
                or self.zeta < 1 or int(self.zeta) != self.zeta):
            raise ValueError("zeta must be a positive integer")
        object.__setattr__(self, "zeta", int(self.zeta))


class ACSFRadialDescriptor(nn.Module):
    """Sum exp[-eta*(r-rs)^2]*fc(r) by neighbor species; return [C,T*K]."""

    def __init__(self, num_types: int, cutoff: float, basis: tuple[RadialBasis, ...]):
        super().__init__()
        self.num_types = num_types
        self.cutoff = cutoff
        self.output_dim = num_types * len(basis)
        self.register_buffer("eta", torch.tensor([b.eta for b in basis]))
        self.register_buffer("rs", torch.tensor([b.rs for b in basis]))

    def forward(self, types, vectors, valid):
        distances = vectors.square().sum(-1).clamp(min=1.e-24).sqrt()
        values = torch.exp(-self.eta * (distances.unsqueeze(-1) - self.rs).square())
        values = values * (_cosine_cutoff(distances, self.cutoff) * valid).unsqueeze(-1)
        species = F.one_hot(types, self.num_types).to(vectors.dtype)
        return torch.matmul(species.transpose(1, 2), values).reshape(vectors.shape[0], self.output_dim)


class ACSFAngularDescriptor(nn.Module):
    """Sum angular terms by unordered species pairs; include_rjk=True selects narrow terms.
    
    Neighbor slots use j<k. Different periodic images of the same atom remain distinct
    neighbors. Channels are ordered by species pair, then basis function.
    """

    def __init__(self, num_types: int, cutoff: float, basis: tuple[AngularBasis, ...], *, include_rjk: bool):
        super().__init__()
        self.cutoff = cutoff
        self.include_rjk = include_rjk
        pairs = tuple(combinations_with_replacement(range(num_types), 2))
        self.num_pairs = len(pairs)
        self.output_dim = self.num_pairs * len(basis)
        pair_lut = torch.empty(num_types, num_types, dtype=torch.long)
        for index, (a, b) in enumerate(pairs):
            pair_lut[a, b] = pair_lut[b, a] = index
        self.register_buffer("pair_lut", pair_lut)
        self.register_buffer("eta", torch.tensor([b.eta for b in basis]))
        self.register_buffer("rs", torch.tensor([b.rs for b in basis]))
        self.register_buffer("lambd", torch.tensor([b.lambd for b in basis]))
        self.register_buffer("zeta", torch.tensor([b.zeta for b in basis], dtype=torch.long))

    def forward(self, types, vectors, valid):
        centers, width, _ = vectors.shape
        j, k = torch.triu_indices(width, width, offset=1, device=vectors.device)
        vj, vk = vectors[:, j], vectors[:, k]
        rj = vj.square().sum(-1).clamp(min=1.e-24).sqrt()
        rk = vk.square().sum(-1).clamp(min=1.e-24).sqrt()
        cosine = ((vj * vk).sum(-1) / (rj * rk)).clamp(-1., 1.)
        angular = 2. ** (1 - self.zeta) * (
            1. + self.lambd * cosine.unsqueeze(-1)
        ).clamp(min=0.).pow(self.zeta)
        exponent = (rj.unsqueeze(-1) - self.rs).square() + (rk.unsqueeze(-1) - self.rs).square()
        envelope = _cosine_cutoff(rj, self.cutoff) * _cosine_cutoff(rk, self.cutoff)
        if self.include_rjk:
            rjk = (vj - vk).square().sum(-1).clamp(min=1.e-24).sqrt()
            exponent = exponent + (rjk.unsqueeze(-1) - self.rs).square()
            envelope = envelope * _cosine_cutoff(rjk, self.cutoff)
        values = angular * torch.exp(-self.eta * exponent)
        values = values * (envelope * valid[:, j] * valid[:, k]).unsqueeze(-1)
        pair_ids = self.pair_lut[types[:, j], types[:, k]]
        output = vectors.new_zeros(centers, self.num_pairs, self.eta.numel())
        output = output.scatter_add(1, pair_ids.unsqueeze(-1).expand_as(values), values)
        return output.reshape(centers, self.output_dim)


@dataclass(frozen=True)
class ACSFBlockConfig:
    """A complete descriptor block without parallel parameter lists or a redundant num_basis."""

    type: str
    basis: tuple[RadialBasis | AngularBasis, ...]

    @classmethod
    def from_dict(cls, raw, *, path):
        if not isinstance(raw, Mapping):
            raise ConfigError(path, "must be an object")
        check_unknown(raw, {"type", "basis"}, path)
        kind = as_str(raw.get("type"), f"{path}.type")
        if kind not in {"radial_gaussian", "angular_narrow", "angular_wide"}:
            raise ConfigError(f"{path}.type", "expected radial_gaussian, angular_narrow or angular_wide")
        records = raw.get("basis")
        if not isinstance(records, (list, tuple)) or not records:
            raise ConfigError(f"{path}.basis", "must be a non-empty list")
        basis = []
        for index, record in enumerate(records):
            location = f"{path}.basis[{index}]"
            if not isinstance(record, Mapping):
                raise ConfigError(location, "must be an object")
            angular = kind != "radial_gaussian"
            check_unknown(record, {"eta", "rs", "lambda", "zeta"} if angular else {"eta", "rs"}, location)
            eta = as_float(record.get("eta"), f"{location}.eta")
            rs = as_float(record.get("rs"), f"{location}.rs")
            try:
                if angular:
                    lambd = as_float(record.get("lambda"), f"{location}.lambda")
                    zeta = as_float(record.get("zeta"), f"{location}.zeta")
                    if not math.isfinite(zeta):
                        raise ValueError("zeta must be finite")
                    basis.append(AngularBasis(eta, rs, lambd, zeta))
                else:
                    basis.append(RadialBasis(eta, rs))
            except ValueError as error:
                raise ConfigError(location, str(error)) from error
        return cls(kind, tuple(basis))


class ACSFDescriptor(nn.Module):
    """Convert CSR geometry into concatenated ACSFs with shared dimension and feature metadata."""

    def __init__(self, type_map, cutoff, max_neighbors, blocks):
        super().__init__()
        if not math.isfinite(cutoff) or cutoff <= 0 or max_neighbors < 1:
            raise ValueError("cutoff must be positive and finite; max_neighbors must be positive")
        if not blocks:
            raise ValueError("ACSF requires at least one descriptor block")
        self.type_map = tuple(type_map)
        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.register_buffer("type_lut", type_lookup(list(type_map)))
        self.blocks = nn.ModuleList()
        layout = []
        for block_index, block in enumerate(blocks):
            if block.type == "radial_gaussian":
                module = ACSFRadialDescriptor(len(type_map), cutoff, block.basis)
                channels = [(number,) for number in type_map]
            elif block.type in {"angular_narrow", "angular_wide"}:
                module = ACSFAngularDescriptor(len(type_map), cutoff, block.basis,
                                             include_rjk=block.type == "angular_narrow")
                channels = list(combinations_with_replacement(type_map, 2))
            else:
                raise ValueError(f"Unsupported ACSF block: {block.type}")
            self.blocks.append(module)
            for elements in channels:
                for basis_index, basis in enumerate(block.basis):
                    layout.append({"block": block_index, "type": block.type, "elements": elements,
                                   "basis": basis_index, "eta": basis.eta, "rs": basis.rs,
                                   **({"lambda": basis.lambd, "zeta": basis.zeta}
                                      if isinstance(basis, AngularBasis) else {})})
        self.output_dim = sum(block.output_dim for block in self.blocks)
        self.feature_layout = tuple(layout)

    def forward(self, inputs: ModelInput):
        """Return raw descriptors [C,D] and center species indices [C]."""
        centers, ptr, neighbors, vectors = inputs.require_neighbors()
        numbers = inputs.atomic_numbers
        if bool(((numbers < 1) | (numbers >= self.type_lut.numel())).any()):
            raise ValueError("Atomic number is absent from type_map")
        atom_types = self.type_lut[numbers]
        if bool((atom_types < 0).any()):
            raise ValueError("Atomic number is absent from type_map")
        indices, dense_vectors, valid = to_dense(ptr, neighbors, vectors, self.max_neighbors)
        types = atom_types[indices.clamp(min=0)]
        features = torch.cat([block(types, dense_vectors, valid) for block in self.blocks], dim=-1)
        # Keep a zero-gradient path for empty graphs so shared mechanics returns zero isolated-atom forces.
        return features + vectors.sum() * 0., atom_types[centers]


class ACSFFeatureNormalizer(nn.Module):
    """Normalize by species and feature; nearly constant channels are centered without amplifying
    noise.
    """

    def __init__(self, num_types, num_features, epsilon=1.e-8):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("feature_std", torch.ones(num_types, num_features))
        self.register_buffer("feature_avg", torch.zeros(num_types, num_features))

    @torch.no_grad()
    def set_statistics(self, std, avg):
        if std.shape != self.feature_std.shape or avg.shape != std.shape:
            raise ValueError("ACSF statistics must have shape [T,F]")
        if not bool(torch.isfinite(std).all() and torch.isfinite(avg).all()) or bool((std < 0).any()):
            raise ValueError("ACSF statistics must be finite with non-negative std")
        self.feature_std.copy_(std)
        self.feature_avg.copy_(avg)

    def forward(self, features, center_type_ids):
        std = self.feature_std[center_type_ids]
        scale = torch.where(std > self.epsilon, std, torch.ones_like(std))
        return (features - self.feature_avg[center_type_ids]) / scale


class ACSFModel(AtomicModel):
    """Combine descriptors, normalization, and fitting networks to predict corrections to species
    energy baselines.
    """

    required_statistics = ("acsf_calibration",)
    capabilities = ("energy", "atomic_energies", "forces", "virial")

    def __init__(self, descriptor: ACSFDescriptor, hidden_layers, activation="tanh", bias=True, residual=False):
        super().__init__(type_map=list(descriptor.type_map), cutoff=descriptor.cutoff)
        self.descriptor = descriptor
        self.normalizer = ACSFFeatureNormalizer(len(self.type_map), descriptor.output_dim)
        self.fitting = SpeciesFittingNetwork(len(self.type_map), [descriptor.output_dim, *hidden_layers],
                                            activation, bias, residual, [0.] * len(self.type_map))
        # Store the baseline as a buffer; zero-initialized output biases learn residual energies.
        nn.init.zeros_(self.fitting.layers[-1].bias)
        self.register_buffer("energy_reference", torch.zeros(len(self.type_map)))

    @torch.no_grad()
    def initialize_statistics(self, structures, provider):
        """Collect training statistics with the existing descriptor, without temporary networks or
        random-number consumption.
        """
        from abfml.data.collate import StructureCollator
        from abfml.builtin.statistics import EnergyShiftStatistics

        statistics = ACSFFeatureStatistics(len(self.type_map), self.descriptor.output_dim)
        energy_statistics = EnergyShiftStatistics(self.type_map)
        collator = StructureCollator(provider)
        for structure in structures:
            if structure.energy is None:
                raise ValueError("ACSF calibration requires energy labels")
            inputs, _ = collator([structure])
            features, types = self.descriptor(inputs.to(device=self.energy_reference.device,
                                                        dtype=self.energy_reference.dtype))
            statistics.update(features.cpu(), types.cpu())
            energy_statistics.update(structure.atomic_numbers, structure.energy)
        std, avg = statistics.finalize()
        self.normalizer.set_statistics(std, avg)
        self.energy_reference.copy_(self.energy_reference.new_tensor(energy_statistics.finalize()))

    def compute(self, inputs):
        features, types = self.descriptor(inputs)
        descriptors = self.normalizer(features, types)
        correction = self.fitting(descriptors, types).reshape(-1)
        atomic_energies = self.energy_reference[types] + correction
        return ModelOutput(energy=atomic_energies.sum(), atomic_energies=atomic_energies,
                           properties={"descriptors": descriptors})


@dataclass(frozen=True)
class ACSFConfig:
    """ACSF configuration with explicit basis functions, neighbor capacity, and descriptor
    hyperparameters.
    """

    type_map: tuple[int, ...]
    cutoff: float
    max_neighbors: int
    blocks: tuple[ACSFBlockConfig, ...]
    hidden_layers: tuple[int, ...]
    activation: str
    bias: bool
    residual: bool

    @classmethod
    def from_dict(cls, raw, *, type_map, path="model"):
        check_unknown(raw, {"name", "descriptor", "fitting"}, path)
        descriptor = raw.get("descriptor")
        location = f"{path}.descriptor"
        if not isinstance(descriptor, Mapping):
            raise ConfigError(location, "must be an object")
        check_unknown(descriptor, {"cutoff", "cutoff_function", "max_neighbors", "blocks"}, location)
        cutoff = as_float(descriptor.get("cutoff"), f"{location}.cutoff")
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ConfigError(f"{location}.cutoff", "must be positive and finite")
        if descriptor.get("cutoff_function", "cos") != "cos":
            raise ConfigError(f"{location}.cutoff_function", "only cos is supported")
        max_neighbors = as_int(descriptor.get("max_neighbors", 100), f"{location}.max_neighbors")
        if max_neighbors < 1:
            raise ConfigError(f"{location}.max_neighbors", "must be positive")
        blocks = descriptor.get("blocks")
        if not isinstance(blocks, (list, tuple)) or not blocks:
            raise ConfigError(f"{location}.blocks", "must be a non-empty list")
        parsed = tuple(ACSFBlockConfig.from_dict(block, path=f"{location}.blocks[{i}]")
                       for i, block in enumerate(blocks))
        fitting = raw.get("fitting")
        if not isinstance(fitting, Mapping):
            raise ConfigError(f"{path}.fitting", "must be an object")
        hidden_layers, activation, bias, residual = parse_network(fitting, path=f"{path}.fitting")
        return cls(tuple(type_map), cutoff, max_neighbors, parsed, hidden_layers, activation, bias, residual)

    def model_kwargs(self):
        """Build a descriptor-only module and pass current parameter names directly to the model."""
        return {"descriptor": ACSFDescriptor(self.type_map, self.cutoff, self.max_neighbors, self.blocks),
                "hidden_layers": self.hidden_layers, "activation": self.activation,
                "bias": self.bias, "residual": self.residual}


# Bind model and configuration here; the registry remains implementation-independent.
register_model("acsf", ModelSpec(ACSFModel, ACSFConfig))
