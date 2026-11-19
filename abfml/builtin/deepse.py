"""Deep Potential Smooth Edition (DeepPot-SE) models.

Pipeline: CSR neighbors -> dense coordinates -> smoothing and normalization -> embedding ->
rotation-invariant descriptors -> fitting -> atomic energies.

se_e2_a combines distance embeddings with full relative coordinates into H*M2 descriptors;
se_e2_r uses radial embeddings; se_e3 embeds angles between neighbor vectors.

The radial variant preserves ABFML's convention D_i=(sum_j G_ij)^2/Nc^2. This differs from sum_j
G_ij/Nc in DeePMD-kit documentation. Weight interoperability requires identifying the target
version and formula rather than replacing this convention silently.

References: DeePMD-kit se_e2_a/se_e2_r/se_e3 documentation; L. Zhang et al., Phys. Rev. Lett.
120, 143001 (2018).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from abfml.config import (
    ConfigError,
    as_float,
    as_int,
    check_unknown,
    parse_network,
)
from abfml.core.registry import ModelSpec, register_model
from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput
from abfml.builtin.networks import ResidualMLP
from abfml.builtin.functions import cutoff_function as smooth_fun, type_lookup, safe_reciprocal
from abfml.builtin.functions import SMOOTH_FUNCTIONS


class DeepPotSECoordinateStatistics(nn.Module):
    """Collect coordinate-matrix standard deviations and means using original ABFML conventions.
    
    Compute radial mean/std and one shared Cartesian-component std per structure, then average
    across structures. Fixed-width zero padding participates in these historical statistics.
    """

    def __init__(self, num_types: int, coordinate_size: int) -> None:
        super().__init__()
        if num_types < 1 or coordinate_size not in {1, 4}:
            raise ValueError("DeepPot-SE statistics require positive types and 1 or 4 coordinates")
        self.num_types = int(num_types)
        self.coordinate_size = int(coordinate_size)
        shape = (self.num_types, self.coordinate_size)
        self.register_buffer("std_sum", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("mean_sum", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("count", torch.zeros(self.num_types, dtype=torch.float64))

    @torch.no_grad()
    def update(self, r_matrix: torch.Tensor, center_type_ids: torch.Tensor) -> None:
        """Accumulate one structure's coordinate matrix with shape [C,M,1] or [C,M,4]."""
        if (
            r_matrix.ndim != 3
            or r_matrix.shape[-1] != self.coordinate_size
            or center_type_ids.shape != (r_matrix.shape[0],)
        ):
            raise ValueError("Invalid DeepPot-SE coordinate matrix or center type shape")
        values = r_matrix.detach().to(dtype=torch.float64)
        for type_id in range(self.num_types):
            selected = values[center_type_ids == type_id]
            if selected.shape[0] == 0:
                continue
            radial_std, radial_mean = torch.std_mean(selected[..., 0])
            self.std_sum[type_id, 0].add_(radial_std)
            self.mean_sum[type_id, 0].add_(radial_mean)
            if self.coordinate_size == 4:
                xyz_std, _ = torch.std_mean(selected[..., 1:])
                self.std_sum[type_id, 1:].add_(xyz_std)
            self.count[type_id].add_(1)

    @torch.no_grad()
    def finalize(self) -> list[torch.Tensor]:
        if bool((self.count == 0).any()):
            missing = (self.count == 0).nonzero().reshape(-1).tolist()
            raise ValueError(f"No DeepPot-SE coordinates were observed for species: {missing}")
        count = self.count.unsqueeze(-1)
        return [self.std_sum / count, self.mean_sum / count]


class _DeepPotSEBase(AtomicModel):
    """Shared DeepPot-SE parameters, networks, and neighbor preparation.
    
    There are T*T center-neighbor species embeddings and T center-species fitting networks. Each
    variant implements its own descriptor in compute(), keeping mathematical paths separate.
    Padding aligns tensors but is not a physical neighbor; padded coordinates and embeddings
    must remain zero.
    """

    required_statistics: tuple[str, ...] = ("energy_shift", "std_mean")
    capabilities = ("energy", "atomic_energies", "forces", "virial")
    coordinate_size: int = 4

    @classmethod
    def measure_normalization(cls, structures, provider, **model_kwargs) -> list[torch.Tensor]:
        """Compute training coordinate statistics when requested by the model workflow."""
        from abfml.core.neighbor import to_dense

        type_map = model_kwargs["type_map"]
        embedding_config = model_kwargs["embedding_config"]
        max_neighbors = int(model_kwargs["max_neighbors"])
        type_slots = model_kwargs.get("type_slots")
        type_lut = type_lookup(type_map)
        statistics = DeepPotSECoordinateStatistics(len(type_map), cls.coordinate_size)

        for structure in structures:
            neighbors = provider.build(structure)
            neighbor_type_ids = type_lut[
                structure.atomic_numbers[neighbors.neighbor_indices]
            ]
            _, r_vec, valid = to_dense(
                neighbors.edge_ptr,
                neighbors.neighbor_indices,
                neighbors.edge_vectors,
                max_neighbors,
                neighbor_type_ids=neighbor_type_ids,
                num_types=len(type_map),
                type_slots=type_slots,
            )
            center_type_ids = type_lut[
                structure.atomic_numbers[neighbors.center_indices]
            ]
            r = r_vec.square().sum(dim=-1).clamp(min=1.0e-24).sqrt()
            s = smooth_fun(
                embedding_config["smooth_fun"],
                r,
                float(embedding_config["R_min"]),
                float(embedding_config["R_max"]),
            )
            if cls.coordinate_size == 1:
                r_matrix = s.unsqueeze(-1)
            else:
                inv_r = safe_reciprocal(r, valid)
                r_matrix = torch.cat(
                    (s.unsqueeze(-1), r_vec * (s * inv_r).unsqueeze(-1)), dim=-1
                )
            statistics.update(r_matrix, center_type_ids)
        return statistics.finalize()

    def __init__(
        self,
        type_map: list[int],
        cutoff: float,
        max_neighbors: int,
        fitting_config: dict,
        embedding_config: dict,
        energy_shift: list[float],
        std_mean: Optional[list[torch.Tensor]],
        fitting_inputs: int,
        coordinate_size: int,
        type_slots: Optional[list[int]] = None,
    ):
        super().__init__(type_map=type_map, cutoff=cutoff)
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be positive")
        self.descriptor_dim = int(fitting_inputs)
        self.max_neighbors = int(max_neighbors)
        if type_slots is None:
            self.type_slots = [self.max_neighbors for _ in type_map]
        else:
            if len(type_slots) != len(type_map):
                raise ValueError("type_slots must have one entry per species")
            if any(width < 0 for width in type_slots):
                raise ValueError("type_slots must be non-negative")
            self.type_slots = type_slots
        self.total_neighbors = sum(self.type_slots)
        self.r_min = float(embedding_config["R_min"])
        self.r_max = float(embedding_config["R_max"])
        self.smooth_type = embedding_config["smooth_fun"]
        self.embedding_size = int(embedding_config["network_size"][-1])
        self.register_buffer("type_lut", type_lookup(type_map))

        self.embedding_net = nn.ModuleList()
        self.fitting_net = nn.ModuleList()
        for center_type in range(len(type_map)):
            for neighbor_type in range(len(type_map)):
                self.embedding_net.append(
                    ResidualMLP(
                        network_size=[1, *embedding_config["network_size"]],
                        activate_output=True,
                        activate=embedding_config["activate_function"],
                        bias=embedding_config["bias"],
                        resnet_dt=embedding_config["resnet_dt"],
                    )
                )
            self.fitting_net.append(
                ResidualMLP(
                    network_size=[fitting_inputs, *fitting_config["network_size"], 1],
                    activate_output=False,
                    shifted_output=True,
                    activate=fitting_config["activate_function"],
                    bias=fitting_config["bias"],
                    resnet_dt=fitting_config["resnet_dt"],
                    energy_shift=energy_shift[center_type],
                )
            )

        if std_mean is None:
            feature_std = torch.ones(len(type_map), coordinate_size)
            feature_avg = torch.zeros(len(type_map), coordinate_size)
        else:
            feature_std, feature_avg = std_mean
        self.register_buffer("feature_std", feature_std.detach().clone())
        self.register_buffer("feature_avg", feature_avg.detach().clone())

    def _prepare_neighbors(
        self,
        atomic_numbers: torch.Tensor,
        center_indices: torch.Tensor,
        edge_ptr: torch.Tensor,
        neighbor_indices: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prepare species-segmented fixed-width neighbors and shared radial quantities.
        
        Return r_vec, validity masks, neighbor/center species indices, safe inv_r=1/r, and
        smoothed s=fc(r). Species slots are concatenated with total width sum(type_slots).
        """
        edge_type_ids = self.type_lut[atomic_numbers[neighbor_indices]]
        dense_indices, r_vec, valid = self.to_dense(
            edge_ptr,
            neighbor_indices,
            edge_vectors,
            self.max_neighbors,
            neighbor_type_ids=edge_type_ids,
            type_slots=self.type_slots,
        )
        safe_indices = dense_indices.clamp(min=0)
        neighbor_types = atomic_numbers[safe_indices]
        neighbor_type_ids = self.type_lut[neighbor_types]
        center_type_ids = self.type_lut[atomic_numbers[center_indices]]
        if bool((center_type_ids < 0).any()):
            raise ValueError("Atomic number is absent from type_map")

        r = r_vec.square().sum(dim=-1).clamp(min=1.0e-24).sqrt()
        inv_r = safe_reciprocal(r, valid)
        s = smooth_fun(
            self.smooth_type,
            r,
            self.r_min,
            self.r_max,
        )
        return r_vec, valid, neighbor_type_ids, center_type_ids, inv_r, s

    def _normalize_coordinates(
        self,
        r_matrix: torch.Tensor,
        valid: torch.Tensor,
        center_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize coordinate matrices by center species and explicitly clear padding."""
        r_matrix = (
            r_matrix - self.feature_avg[center_type_ids].unsqueeze(1)
        ) / self.feature_std[center_type_ids].unsqueeze(1)
        return r_matrix * valid.unsqueeze(-1).to(r_matrix.dtype)

    def _embed_species(
        self,
        s_ij: torch.Tensor,
        neighbor_type_ids: torch.Tensor,
        valid: torch.Tensor,
        center_type: int,
    ) -> torch.Tensor:
        """Evaluate species-pair embedding networks in batches to obtain G_ij.
        
        s_ij and neighbor_type_ids have shape [C,M]. Each scalar maps to an H-dimensional
        vector, producing [C,M,H]. Padding and unknown species slots remain exactly zero for
        subsequent contractions and sums.
        """
        n_centers, max_neighbors = s_ij.shape
        flat_s_ij = s_ij.reshape(-1)
        flat_types = neighbor_type_ids.reshape(-1)
        flat_valid = valid.reshape(-1)
        g_ij = s_ij.new_zeros(
            n_centers * max_neighbors,
            self.embedding_size,
        )
        for network_index, embedding_net in enumerate(self.embedding_net):
            if network_index // len(self.type_map) == center_type:
                neighbor_type = network_index % len(self.type_map)
                selected = flat_valid & (flat_types == neighbor_type)
                if int(selected.sum()) > 0:
                    g_ij[selected] = embedding_net(
                        flat_s_ij[selected].unsqueeze(-1)
                    )
        return g_ij.reshape(n_centers, max_neighbors, self.embedding_size)


class DeepPotSE2A(_DeepPotSEBase):
    """se_e2_a: invariant descriptors from relative coordinates and distance embeddings.
    
    R=[C,M,4], G=[C,M,H], and D=(G^T R)(R^T G_<)/M^2. G_< retains the first M2 embedding
    channels, giving fitting width H*M2. M is the fixed neighbor-slot capacity.
    """

    def __init__(
        self,
        type_map: list[int],
        cutoff: float,
        max_neighbors: int,
        fitting_config: dict,
        embedding_config: dict,
        energy_shift: list[float],
        std_mean: Optional[list[torch.Tensor]] = None,
        type_slots: Optional[list[int]] = None,
    ):
        embedding_size = int(embedding_config["network_size"][-1])
        m2 = int(embedding_config["M2"])
        super().__init__(
            type_map,
            cutoff,
            max_neighbors,
            fitting_config,
            embedding_config,
            energy_shift,
            std_mean,
            type_slots=type_slots,
            fitting_inputs=embedding_size * m2,
            coordinate_size=4,
        )
        self.m2 = m2

    def compute(self, inputs: ModelInput) -> ModelOutput:
        """Compute D_i=(G^T R)(R^T G_<)/M^2 and atomic energies."""
        atomic_numbers = inputs.atomic_numbers
        center_indices, edge_ptr, neighbor_indices, edge_vectors = (
            inputs.require_neighbors()
        )
        (
            r_vec,
            valid,
            neighbor_type_ids,
            center_type_ids,
            inv_r,
            s,
        ) = self._prepare_neighbors(
            atomic_numbers,
            center_indices,
            edge_ptr,
            neighbor_indices,
            edge_vectors,
        )

        # R_ij=(s,s*x/r,s*y/r,s*z/r), shape [C,M,4].
        r_matrix = torch.cat(
            (
                s.unsqueeze(-1),
                r_vec * (s * inv_r).unsqueeze(-1),
            ),
            dim=-1,
        )
        r_matrix = self._normalize_coordinates(r_matrix, valid, center_type_ids)

        n_centers = center_indices.numel()
        e_i = r_vec.new_zeros(n_centers)
        descriptors = r_vec.new_zeros(n_centers, self.descriptor_dim)
        for center_type, fitting_net in enumerate(self.fitting_net):
            center_mask = center_type_ids == center_type
            selected_r = r_matrix[center_mask]
            selected_valid = valid[center_mask]
            selected_types = neighbor_type_ids[center_mask]
            n_selected = selected_r.shape[0]

            # g_ij=G(s_ij): species-pair networks map each distance scalar to an H-dimensional vector.
            g_ij = self._embed_species(
                selected_r[:, :, 0],
                selected_types,
                selected_valid,
                center_type,
            )
            # rT_g=A=R^T*G/M, shape [C,4,H]; D_i=A^T*A_<, where
            # A_< keeps the first M2 columns, giving D_i=(G^T*R)(R^T*G_<)/M^2.
            rT_g = torch.matmul(
                selected_r.transpose(1, 2),
                g_ij,
            ) / float(self.total_neighbors)
            d_i = torch.matmul(
                rT_g.transpose(1, 2),
                rT_g[:, :, : self.m2],
            ).reshape(n_selected, self.embedding_size * self.m2)
            descriptors[center_mask] = d_i
            e_i[center_mask] = fitting_net(d_i).reshape(-1)

        return ModelOutput(
            energy=e_i.sum(),
            atomic_energies=e_i,
            properties={"descriptors": descriptors},
        )


class DeepPotSE2R(_DeepPotSEBase):
    """se_e2_r: radial descriptors following the ABFML convention.
    
    G=[C,M,H] produces D=(sum_j G_ij)^2/M^2 with shape [C,H]. This differs from DeePMD-kit's
    sum_j G_ij/M; official se_e2_r weights are not assumed interchangeable.
    """

    coordinate_size: int = 1

    def __init__(
        self,
        type_map: list[int],
        cutoff: float,
        max_neighbors: int,
        fitting_config: dict,
        embedding_config: dict,
        energy_shift: list[float],
        std_mean: Optional[list[torch.Tensor]] = None,
        type_slots: Optional[list[int]] = None,
    ):
        super().__init__(
            type_map,
            cutoff,
            max_neighbors,
            fitting_config,
            embedding_config,
            energy_shift,
            std_mean,
            type_slots=type_slots,
            fitting_inputs=int(embedding_config["network_size"][-1]),
            coordinate_size=1,
        )

    def compute(self, inputs: ModelInput) -> ModelOutput:
        """Compute radial D_i=(sum_j G_ij)^2/M^2 and atomic energies."""
        atomic_numbers = inputs.atomic_numbers
        center_indices, edge_ptr, neighbor_indices, edge_vectors = (
            inputs.require_neighbors()
        )
        (
            r_vec,
            valid,
            neighbor_type_ids,
            center_type_ids,
            inv_r,
            s,
        ) = self._prepare_neighbors(
            atomic_numbers,
            center_indices,
            edge_ptr,
            neighbor_indices,
            edge_vectors,
        )
        del inv_r

        # The radial model retains only column s of the coordinate matrix, shape [C,M,1].
        r_matrix = self._normalize_coordinates(
            s.unsqueeze(-1),
            valid,
            center_type_ids,
        )

        n_centers = center_indices.numel()
        e_i = r_vec.new_zeros(n_centers)
        descriptors = r_vec.new_zeros(n_centers, self.descriptor_dim)
        for center_type, fitting_net in enumerate(self.fitting_net):
            center_mask = center_type_ids == center_type
            selected_r = r_matrix[center_mask]
            selected_valid = valid[center_mask]
            selected_types = neighbor_type_ids[center_mask]
            g_ij = self._embed_species(
                selected_r[:, :, 0],
                selected_types,
                selected_valid,
                center_type,
            )
            d_i = g_ij.sum(dim=1).square() / float(self.total_neighbors ** 2)
            descriptors[center_mask] = d_i
            e_i[center_mask] = fitting_net(d_i).reshape(-1)

        return ModelOutput(
            energy=e_i.sum(),
            atomic_energies=e_i,
            properties={"descriptors": descriptors},
        )


class DeepPotSE3(_DeepPotSEBase):
    """se_e3: three-body descriptors from neighbor-pair angle embeddings.
    
    For ordered pairs (j,k), form theta_ijk=s_ij*s_ik*cos(angle_ijk), sum theta*Embedding(theta)
    over j and k, and divide by M^2. The [C,H] output retains angles, not absolute orientation.
    """

    def __init__(
        self,
        type_map: list[int],
        cutoff: float,
        max_neighbors: int,
        fitting_config: dict,
        embedding_config: dict,
        energy_shift: list[float],
        std_mean: Optional[list[torch.Tensor]] = None,
        type_slots: Optional[list[int]] = None,
    ):
        super().__init__(
            type_map,
            cutoff,
            max_neighbors,
            fitting_config,
            embedding_config,
            energy_shift,
            std_mean,
            type_slots=type_slots,
            fitting_inputs=int(embedding_config["network_size"][-1]),
            coordinate_size=4,
        )

    def compute(self, inputs: ModelInput) -> ModelOutput:
        """Compute three-body descriptors and atomic energies from neighbor-pair angle embeddings."""
        atomic_numbers = inputs.atomic_numbers
        center_indices, edge_ptr, neighbor_indices, edge_vectors = (
            inputs.require_neighbors()
        )
        (
            r_vec,
            valid,
            neighbor_type_ids,
            center_type_ids,
            inv_r,
            s,
        ) = self._prepare_neighbors(
            atomic_numbers,
            center_indices,
            edge_ptr,
            neighbor_indices,
            edge_vectors,
        )
        r_matrix = torch.cat(
            (
                s.unsqueeze(-1),
                r_vec * (s * inv_r).unsqueeze(-1),
            ),
            dim=-1,
        )
        r_matrix = self._normalize_coordinates(r_matrix, valid, center_type_ids)

        n_centers = center_indices.numel()
        e_i = r_vec.new_zeros(n_centers)
        descriptors = r_vec.new_zeros(n_centers, self.descriptor_dim)
        for center_type, fitting_net in enumerate(self.fitting_net):
            center_mask = center_type_ids == center_type
            selected_r = r_matrix[center_mask]
            selected_valid = valid[center_mask]
            selected_types = neighbor_type_ids[center_mask]
            n_selected = selected_r.shape[0]

            # cos_theta=cos(theta_ijk)=r_ij·r_ik/(|r_ij|*|r_ik|)。
            cos_theta = torch.matmul(
                r_vec[center_mask],
                r_vec[center_mask].transpose(1, 2),
            )
            selected_inv_r = inv_r[center_mask]
            cos_theta = (
                cos_theta
                * selected_inv_r.unsqueeze(-1)
                * selected_inv_r.unsqueeze(-2)
            )
            # theta_ijk=s_ij*s_ik*cos_theta, shape [C_selected,M,M].
            # Both smoothing factors suppress invalid neighbors and contributions at the cutoff.
            theta = (
                selected_r[:, :, 0].unsqueeze(-1)
                * selected_r[:, :, 0].unsqueeze(-2)
                * cos_theta
            )
            d_i = selected_r.new_zeros(
                n_selected,
                self.embedding_size,
            )
            # For each fixed j, compute g_theta=Embedding(theta_jk), then
            # sum_k(theta_jk*g_theta) and aggregate over j with index_add.
            # This avoids the previous [C,M,M,H] intermediate tensor.
            for network_index, embedding_net in enumerate(self.embedding_net):
                if network_index // len(self.type_map) == center_type:
                    neighbor_type = network_index % len(self.type_map)
                    selected_neighbors = (
                        selected_valid & (selected_types == neighbor_type)
                    )
                    if int(selected_neighbors.sum()) > 0:
                        selected_theta = theta[selected_neighbors]
                        g_theta = embedding_net(selected_theta.unsqueeze(-1))
                        sum_k_theta_g = (
                            selected_theta.unsqueeze(-1) * g_theta
                        ).sum(dim=1)
                        center_slots = torch.arange(
                            n_selected,
                            device=theta.device,
                        ).unsqueeze(1).expand(-1, self.total_neighbors)
                        d_i = d_i.index_add(
                            0,
                            center_slots[selected_neighbors],
                            sum_k_theta_g,
                        )
            d_i = d_i / float(self.total_neighbors ** 2)
            descriptors[center_mask] = d_i
            e_i[center_mask] = fitting_net(d_i).reshape(-1)

        return ModelOutput(
            energy=e_i.sum(),
            atomic_energies=e_i,
            properties={"descriptors": descriptors},
        )


# ---------------------------------------------------------------------------
# Keep model declarations and configuration parsing in the same file.
# ---------------------------------------------------------------------------


def _parse_slots(value: object, path: str, num_types: int) -> tuple[int, ...]:
    """Normalize scalar or list type_slots into a tuple of length T."""
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ConfigError(path, "must be non-negative")
        return (value,) * num_types
    if isinstance(value, (list, tuple)):
        slots = tuple(as_int(item, f"{path}[{i}]") for i, item in enumerate(value))
        if len(slots) != num_types:
            raise ConfigError(
                path, f"must have one entry per element ({num_types}), got {len(slots)}"
            )
        if any(width < 0 for width in slots):
            raise ConfigError(path, "all slots must be non-negative")
        return slots
    raise ConfigError(path, "expected an integer or a list of integers")


def _parse_smooth(value: object, path: str) -> str:
    smooth = value if isinstance(value, str) else None
    if smooth not in SMOOTH_FUNCTIONS:
        raise ConfigError(path, f"unsupported smooth function: {value!r}")
    return smooth


@dataclass(frozen=True)
class _DeepPotSEConfig:
    """Shared configuration for the three DeepPot variants; se_e2_a adds m2."""

    type_map: tuple[int, ...]
    cutoff: float
    type_slots: tuple[int, ...]
    r_min: float
    smooth: str
    embedding_network_size: tuple[int, ...]
    embedding_activation: str
    embedding_bias: bool
    embedding_residual: bool
    fitting_hidden_layers: tuple[int, ...]
    fitting_activation: str
    fitting_bias: bool
    fitting_residual: bool

    @property
    def max_neighbors(self) -> int:
        """Total fixed-width neighbor capacity, determined by type_slots for segmented layouts."""
        return sum(self.type_slots)

    @staticmethod
    def _parse_common(
        raw: Mapping,
        *,
        type_map: tuple[int, ...],
        path: str,
    ) -> dict:
        check_unknown(
            raw,
            {"name", "cutoff", "type_slots", "descriptor", "fitting"},
            path,
        )
        cutoff = as_float(raw.get("cutoff"), f"{path}.cutoff")
        if cutoff <= 0:
            raise ConfigError(f"{path}.cutoff", "must be positive")
        type_slots = _parse_slots(
            raw.get("type_slots"), f"{path}.type_slots", len(type_map)
        )

        descriptor = raw.get("descriptor")
        if not isinstance(descriptor, Mapping):
            raise ConfigError(f"{path}.descriptor", "must be an object")
        check_unknown(descriptor, {"r_min", "smooth", "embedding", "m2"}, f"{path}.descriptor")
        r_min = as_float(descriptor.get("r_min"), f"{path}.descriptor.r_min")
        if not 0.0 <= r_min < cutoff:
            raise ConfigError(
                f"{path}.descriptor.r_min", "must satisfy 0 <= r_min < cutoff"
            )
        smooth = _parse_smooth(
            descriptor.get("smooth"), f"{path}.descriptor.smooth"
        )

        embedding_raw = descriptor.get("embedding")
        if not isinstance(embedding_raw, Mapping):
            raise ConfigError(f"{path}.descriptor.embedding", "must be an object")
        embedding_network_size, embedding_activation, embedding_bias, embedding_residual = (
            parse_network(
                embedding_raw, path=f"{path}.descriptor.embedding", layers_key="network_size"
            )
        )

        fitting_raw = raw.get("fitting")
        if not isinstance(fitting_raw, Mapping):
            raise ConfigError(f"{path}.fitting", "must be an object")
        fitting_hidden_layers, fitting_activation, fitting_bias, fitting_residual = (
            parse_network(fitting_raw, path=f"{path}.fitting")
        )

        return {
            "type_map": type_map,
            "cutoff": cutoff,
            "type_slots": type_slots,
            "r_min": r_min,
            "smooth": smooth,
            "embedding_network_size": embedding_network_size,
            "embedding_activation": embedding_activation,
            "embedding_bias": embedding_bias,
            "embedding_residual": embedding_residual,
            "fitting_hidden_layers": fitting_hidden_layers,
            "fitting_activation": fitting_activation,
            "fitting_bias": fitting_bias,
            "fitting_residual": fitting_residual,
        }

    def embedding_options(self) -> dict:
        """Embedding constructor arguments; network_size includes all widths, ending with the
        output width.
        """
        return {
            "R_min": self.r_min,
            "R_max": self.cutoff,
            "smooth_fun": self.smooth,
            "network_size": list(self.embedding_network_size),
            "activate_function": self.embedding_activation,
            "bias": self.embedding_bias,
            "resnet_dt": self.embedding_residual,
        }

    def fitting_options(self) -> dict:
        """Fitting-network constructor arguments."""
        return {
            "network_size": list(self.fitting_hidden_layers),
            "activate_function": self.fitting_activation,
            "bias": self.fitting_bias,
            "resnet_dt": self.fitting_residual,
        }

    def model_kwargs(self) -> dict:
        """Expand user hyperparameters into constructor arguments; the workflow supplies statistics
        separately.
        """
        return {
            "type_map": list(self.type_map),
            "cutoff": self.cutoff,
            "max_neighbors": self.max_neighbors,
            "fitting_config": self.fitting_options(),
            "embedding_config": self.embedding_options(),
            "type_slots": list(self.type_slots),
        }


@dataclass(frozen=True)
class DeepPotSE2AConfig(_DeepPotSEConfig):
    """se_e2_a requires m2 <= embedding_network_size[-1]."""

    m2: int = 0

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, type_map: tuple[int, ...], path: str = "model"
    ) -> "DeepPotSE2AConfig":
        common = cls._parse_common(raw, type_map=type_map, path=path)
        descriptor = raw["descriptor"]
        m2 = as_int(descriptor.get("m2"), f"{path}.descriptor.m2")
        if m2 < 1:
            raise ConfigError(f"{path}.descriptor.m2", "must be positive")
        if m2 > common["embedding_network_size"][-1]:
            raise ConfigError(
                f"{path}.descriptor.m2", "must not exceed descriptor.embedding.network_size[-1]"
            )
        return cls(m2=m2, **common)

    def embedding_options(self) -> dict:
        config = super().embedding_options()
        config["M2"] = self.m2
        return config


@dataclass(frozen=True)
class DeepPotSE2RConfig(_DeepPotSEConfig):
    """se_e2_r radial descriptors do not require m2."""

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, type_map: tuple[int, ...], path: str = "model"
    ) -> "DeepPotSE2RConfig":
        common = cls._parse_common(raw, type_map=type_map, path=path)
        if "m2" in raw["descriptor"]:
            raise ConfigError(f"{path}.descriptor.m2", "is not used by se_e2_r")
        return cls(**common)


@dataclass(frozen=True)
class DeepPotSE3Config(_DeepPotSEConfig):
    """se_e3 three-body descriptors do not require m2."""

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, type_map: tuple[int, ...], path: str = "model"
    ) -> "DeepPotSE3Config":
        common = cls._parse_common(raw, type_map=type_map, path=path)
        if "m2" in raw["descriptor"]:
            raise ConfigError(f"{path}.descriptor.m2", "is not used by se_e3")
        return cls(**common)


# Bind model and configuration here; the registry remains implementation-independent.
register_model("deep_pot_se_e2_a", ModelSpec(DeepPotSE2A, DeepPotSE2AConfig))
register_model("deep_pot_se_e2_r", ModelSpec(DeepPotSE2R, DeepPotSE2RConfig))
register_model("deep_pot_se_e3", ModelSpec(DeepPotSE3, DeepPotSE3Config))
