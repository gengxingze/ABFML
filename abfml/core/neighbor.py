"""Neighbor graphs, search backends, backend selection, and caching.

Components follow their dependency order:
- DenseNeighbors: fixed-width views produced by densification.
- CSRNeighbors: canonical directed graphs without padding.
- NeighborBuilder: abstract neighbor-search backend.
- ASENeighborBuilder: full directed searches with lazy ASE imports.
- CachedNeighborBuilder: neighbor caches for fixed structures.
- resolve_builder and configure_cache: backend and cache selection.

Additional backends, such as PhysicsNeMo, can implement NeighborBuilder.build.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Iterable
import hashlib

import torch

from abfml.core.structure import AtomicSample


@dataclass
class DenseNeighbors:
    """Fixed-width neighbor views for descriptor models."""

    neighbor_indices: torch.Tensor  # [C, M]
    edge_vectors: torch.Tensor  # [C, M, 3]
    mask: torch.Tensor  # [C, M]


@dataclass
class CSRNeighbors:
    """Canonical CSR representation of a directed neighbor graph.
    
    Center slot c maps to center_indices[c]; its edges occupy the half-open interval
    edge_ptr[c]:edge_ptr[c + 1]. Padding is introduced only by to_dense(). Builders operate on
    one AtomicSample; batching is handled by the caller.
    """

    center_indices: torch.Tensor  # [C], int64
    edge_ptr: torch.Tensor  # [C + 1], int64
    neighbor_indices: torch.Tensor  # [E], int64
    edge_vectors: torch.Tensor  # [E, 3]
    edge_shifts: torch.Tensor | None = None  # [E, 3], int64

    def __post_init__(self) -> None:
        self.center_indices = torch.as_tensor(self.center_indices, dtype=torch.long)
        self.edge_ptr = torch.as_tensor(self.edge_ptr, dtype=torch.long)
        self.neighbor_indices = torch.as_tensor(self.neighbor_indices, dtype=torch.long)
        self.edge_vectors = torch.as_tensor(self.edge_vectors)
        if self.edge_shifts is not None:
            self.edge_shifts = torch.as_tensor(self.edge_shifts, dtype=torch.long)
        self._validate()

    @property
    def num_centers(self) -> int:
        return int(self.center_indices.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.neighbor_indices.shape[0])

    @property
    def degrees(self) -> torch.Tensor:
        return self.edge_ptr[1:] - self.edge_ptr[:-1]

    @property
    def edge_index(self) -> torch.Tensor:
        """Return a GNN/COO edge-index view with shape ``[2, E]``."""
        center_per_edge = torch.repeat_interleave(self.center_indices, self.degrees)
        return torch.stack((center_per_edge, self.neighbor_indices), dim=0)

    def to_dense(
        self,
        max_neighbors: int = -1,
        neighbor_type_ids: torch.Tensor | None = None,
        num_types: int = 0,
        type_slots: list[int] | None = None,
    ) -> DenseNeighbors:
        """Convert CSR neighbors to a fixed-width dense view.
        
        Global width is M; species-segmented width is T*M or sum(type_slots). Padding uses index
        -1, zero vectors, and False masks. Edge order within each center and species is
        preserved.
        """
        return DenseNeighbors(*to_dense(
            self.edge_ptr, self.neighbor_indices, self.edge_vectors,
            max_neighbors, neighbor_type_ids, num_types, type_slots,
        ))

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "CSRNeighbors":
        target_dtype = self.edge_vectors.dtype if dtype is None else dtype
        return CSRNeighbors(
            center_indices=self.center_indices.to(device=device),
            edge_ptr=self.edge_ptr.to(device=device),
            neighbor_indices=self.neighbor_indices.to(device=device),
            edge_vectors=self.edge_vectors.to(device=device, dtype=target_dtype),
            edge_shifts=(
                None if self.edge_shifts is None else self.edge_shifts.to(device=device)
            ),
        )

    def _validate(self) -> None:
        n_edges = self.num_edges
        if self.center_indices.ndim != 1:
            raise ValueError("center_indices must have shape [C]")
        if self.edge_ptr.shape != (self.num_centers + 1,):
            raise ValueError("edge_ptr must have shape [C + 1]")
        if self.neighbor_indices.ndim != 1:
            raise ValueError("neighbor_indices must have shape [E]")
        if self.edge_vectors.shape != (n_edges, 3):
            raise ValueError(f"edge_vectors must have shape ({n_edges}, 3)")
        if not self.edge_vectors.is_floating_point():
            raise TypeError("edge_vectors must use a floating-point dtype")
        if self.edge_shifts is not None and self.edge_shifts.shape != (n_edges, 3):
            raise ValueError(f"edge_shifts must have shape ({n_edges}, 3)")
        if self.num_centers and int(self.center_indices.min().item()) < 0:
            raise ValueError("center_indices cannot contain negative indices")
        if int(self.edge_ptr[0].item()) != 0:
            raise ValueError("edge_ptr must start at zero")
        if bool((self.edge_ptr[1:] < self.edge_ptr[:-1]).any()):
            raise ValueError("edge_ptr must be non-decreasing")
        if int(self.edge_ptr[-1].item()) != n_edges:
            raise ValueError("edge_ptr[-1] must equal the number of edges")
        if n_edges and int(self.neighbor_indices.min().item()) < 0:
            raise ValueError("CSR neighbor_indices cannot contain padding or negative indices")

        index_device = self.center_indices.device
        if self.edge_ptr.device != index_device or self.neighbor_indices.device != index_device:
            raise ValueError("All neighbor index tensors must be on the same device")
        if self.edge_vectors.device != index_device:
            raise ValueError("Neighbor indices and vectors must be on the same device")
        if self.edge_shifts is not None and self.edge_shifts.device != index_device:
            raise ValueError("edge_shifts must be on the same device as the other tensors")


class NeighborBuilder(ABC):
    """Abstract backend for one structure's neighbor graph.
    
    All backends share a cutoff and implement build(). Backend-specific options, such as
    neighbor limits, belong to each implementation.
    """

    def __init__(self, cutoff: float) -> None:
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)

    @abstractmethod
    def build(self, sample: AtomicSample) -> CSRNeighbors:
        """Return the directed neighbor graph for one structure."""
        ...


def to_dense(
    edge_ptr: torch.Tensor,
    neighbor_indices: torch.Tensor,
    edge_vectors: torch.Tensor,
    max_neighbors: int = -1,
    neighbor_type_ids: torch.Tensor | None = None,
    num_types: int = 0,
    type_slots: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor-only densification shared by Python callers and exported models."""
    n_centers = edge_ptr.numel() - 1
    n_edges = neighbor_indices.numel()
    degrees = edge_ptr[1:] - edge_ptr[:-1]
    center_slots = torch.repeat_interleave(
        torch.arange(n_centers, device=neighbor_indices.device), degrees
    )

    if neighbor_type_ids is None:
        if type_slots is not None:
            raise ValueError("type_slots requires neighbor_type_ids")
        if max_neighbors < 0:
            max_neighbors = int(degrees.max()) if n_edges > 0 else 0
        if n_edges > 0 and int(degrees.max()) > max_neighbors:
            raise ValueError("max_neighbors is smaller than the largest CSR degree")
        local_slots = (
            torch.arange(n_edges, device=neighbor_indices.device)
            - edge_ptr[center_slots]
        )
        flat_slots = center_slots * max_neighbors + local_slots
        total_neighbors = max_neighbors

        dense_indices = torch.full(
            (n_centers, total_neighbors),
            -1,
            dtype=neighbor_indices.dtype,
            device=neighbor_indices.device,
        )
        dense_vectors = torch.zeros(
            (n_centers, total_neighbors, 3),
            dtype=edge_vectors.dtype,
            device=edge_vectors.device,
        )
        dense_mask = torch.zeros(
            (n_centers, total_neighbors),
            dtype=torch.bool,
            device=neighbor_indices.device,
        )
        dense_indices.reshape(-1).index_copy_(0, flat_slots, neighbor_indices)
        dense_vectors.reshape(-1, 3).index_copy_(0, flat_slots, edge_vectors)
        dense_mask.reshape(-1).index_fill_(0, flat_slots, True)
        return dense_indices, dense_vectors, dense_mask

    if neighbor_type_ids.shape != neighbor_indices.shape:
        raise ValueError("neighbor_type_ids must have shape [E]")
    if num_types < 1:
        raise ValueError("num_types must be positive in species-segmented mode")
    if n_edges > 0 and bool(
        ((neighbor_type_ids < 0) | (neighbor_type_ids >= num_types)).any()
    ):
        raise ValueError("neighbor_type_ids contains an unknown species")

    if type_slots is None:
        if max_neighbors < 1:
            raise ValueError(
                "Species-segmented mode requires positive max_neighbors "
                "when type_slots is absent"
            )
        slots = [max_neighbors for _ in range(num_types)]
    else:
        if len(type_slots) != num_types:
            raise ValueError("type_slots must have one entry per species")
        slots = type_slots

    total_neighbors = 0
    offsets = [0]
    for width in slots:
        if width < 0:
            raise ValueError("type_slots must be non-negative")
        total_neighbors += width
        offsets.append(total_neighbors)

    dense_indices = torch.full(
        (n_centers, total_neighbors),
        -1,
        dtype=neighbor_indices.dtype,
        device=neighbor_indices.device,
    )
    dense_vectors = torch.zeros(
        (n_centers, total_neighbors, 3),
        dtype=edge_vectors.dtype,
        device=edge_vectors.device,
    )
    dense_mask = torch.zeros(
        (n_centers, total_neighbors),
        dtype=torch.bool,
        device=neighbor_indices.device,
    )

    for type_id, width in enumerate(slots):
        source_edges = torch.nonzero(neighbor_type_ids == type_id).reshape(-1)
        type_centers = center_slots[source_edges]
        type_degrees = torch.bincount(type_centers, minlength=n_centers)
        if source_edges.numel() > 0 and int(type_degrees.max()) > width:
            raise ValueError("type_slots is smaller than a species CSR degree")
        type_ptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=edge_ptr.device),
                type_degrees.cumsum(dim=0),
            )
        )
        local_slots = (
            torch.arange(source_edges.numel(), device=neighbor_indices.device)
            - type_ptr[type_centers]
        )
        flat_slots = (
            type_centers * total_neighbors + offsets[type_id] + local_slots
        )
        dense_indices.reshape(-1).index_copy_(
            0, flat_slots, neighbor_indices[source_edges]
        )
        dense_vectors.reshape(-1, 3).index_copy_(
            0, flat_slots, edge_vectors[source_edges]
        )
        dense_mask.reshape(-1).index_fill_(0, flat_slots, True)
    return dense_indices, dense_vectors, dense_mask


class ASENeighborBuilder(NeighborBuilder):
    """Build structure neighbors using ASE NeighborList.
    
    External callers such as LAMMPS and future backends such as PhysicsNeMo may construct
    CSRNeighbors directly from their own arrays.
    """

    def __init__(
        self,
        cutoff: float,
        max_neighbors: int | None = None,
        overflow: str = "error",
    ) -> None:
        super().__init__(cutoff)
        if max_neighbors is not None and max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        if overflow not in {"error", "nearest"}:
            raise ValueError("overflow must be 'error' or 'nearest'")
        self.max_neighbors = max_neighbors
        self.overflow = overflow

    def build(self, sample: AtomicSample) -> CSRNeighbors:
        # Import ASE only when this backend is used; tensor and external-neighbor interfaces remain ASE-independent.
        import numpy as np
        from ase import Atoms
        from ase.neighborlist import NeighborList

        atoms = Atoms(
            numbers=sample.atomic_numbers.detach().cpu().numpy(),
            positions=sample.positions.detach().cpu().numpy(),
            cell=sample.cell.detach().cpu().numpy(),
            pbc=sample.pbc.detach().cpu().numpy(),
        )
        n_atoms = len(atoms)
        neighbor_list = NeighborList(
            [self.cutoff / 2.0] * n_atoms,
            skin=0.0,
            self_interaction=False,
            bothways=True,
            sorted=False,
        )
        neighbor_list.update(atoms)

        all_neighbors: list[np.ndarray] = []
        all_vectors: list[np.ndarray] = []
        all_shifts: list[np.ndarray] = []
        degrees: list[int] = []

        for center in range(n_atoms):
            indices, shifts = neighbor_list.get_neighbors(center)
            neighbor_positions = atoms.positions[indices] + np.dot(shifts, atoms.cell.array)
            vectors = neighbor_positions - atoms.positions[center]
            distances = np.linalg.norm(vectors, axis=1)

            if self.max_neighbors is not None and len(indices) > self.max_neighbors:
                if self.overflow == "error":
                    raise ValueError(
                        f"Neighbor overflow at atom {center}: found {len(indices)}, "
                        f"but max_neighbors is {self.max_neighbors}"
                    )
                selected = np.argsort(distances, kind="stable")[: self.max_neighbors]
                indices = indices[selected]
                shifts = shifts[selected]
                vectors = vectors[selected]

            degrees.append(len(indices))
            all_neighbors.append(indices.astype(np.int64, copy=False))
            all_vectors.append(vectors)
            all_shifts.append(shifts.astype(np.int64, copy=False))

        edge_ptr = np.concatenate(([0], np.cumsum(degrees, dtype=np.int64)))
        neighbor_indices = (
            np.concatenate(all_neighbors) if all_neighbors else np.empty(0, dtype=np.int64)
        )
        edge_vectors = (
            np.concatenate(all_vectors, axis=0)
            if all_vectors
            else np.empty((0, 3), dtype=np.float64)
        )
        edge_shifts = (
            np.concatenate(all_shifts) if all_shifts else np.empty((0, 3), dtype=np.int64)
        )

        return CSRNeighbors(
            center_indices=torch.arange(n_atoms, dtype=torch.long, device=sample.device),
            edge_ptr=torch.as_tensor(edge_ptr, dtype=torch.long, device=sample.device),
            neighbor_indices=torch.as_tensor(
                neighbor_indices, dtype=torch.long, device=sample.device
            ),
            edge_vectors=torch.as_tensor(
                edge_vectors, dtype=sample.dtype, device=sample.device
            ),
            edge_shifts=torch.as_tensor(
                edge_shifts, dtype=torch.long, device=sample.device
            ),
        )


def resolve_builder(model=None, provider: NeighborBuilder | None = None, *,
                    cutoff: float | None = None) -> NeighborBuilder | None:
    """Honor model neighbor declarations before selecting an external backend.
    
    model may be an instance or a builder class. Without a model, the explicit provider
    determines whether neighbors are built.
    """
    if model is None:
        return provider
    mode = getattr(model, "neighbor_mode", "external")
    if mode in ("internal", "none"):
        return None
    if mode != "external":
        raise ValueError(f"Unknown neighbor mode: {mode}")
    required_cutoff = cutoff if cutoff is not None else getattr(model, "cutoff", None)
    if provider is not None:
        if required_cutoff is not None and provider.cutoff < required_cutoff:
            raise ValueError("Neighbor cutoff is smaller than the model cutoff")
        return provider
    if required_cutoff is None:
        raise ValueError("External neighbors require a model cutoff or explicit provider")
    return ASENeighborBuilder(required_cutoff)


def _copy_graph(graph: CSRNeighbors) -> CSRNeighbors:
    return CSRNeighbors(
        center_indices=graph.center_indices.detach().clone(),
        edge_ptr=graph.edge_ptr.detach().clone(),
        neighbor_indices=graph.neighbor_indices.detach().clone(),
        edge_vectors=graph.edge_vectors.detach().clone(),
        edge_shifts=None if graph.edge_shifts is None else graph.edge_shifts.detach().clone(),
    )


class CachedNeighborBuilder(NeighborBuilder):
    """Optional unbounded CPU cache for small, mostly fixed datasets.
    
    Keys include geometry values and dtypes, independent of object identity and sample order.
    Geometry changes rebuild neighbors; label changes do not. Returned graphs are copies, so
    autograd and caller mutations cannot corrupt cached tensors. Configure the backend before
    use and call clear() after changing custom backend options. Workers keep independent cache
    copies; prewarming before worker startup avoids repeated graph construction across epochs.
    """

    def __init__(self, builder: NeighborBuilder) -> None:
        super().__init__(builder.cutoff)
        self.builder = builder
        self._cache: dict[bytes, CSRNeighbors] = {}

    def _key(self, sample: AtomicSample) -> bytes:
        digest = hashlib.sha256()
        digest.update(repr((self.builder.cutoff,
                            getattr(self.builder, "max_neighbors", None),
                            getattr(self.builder, "overflow", None))).encode())
        for tensor in (sample.atomic_numbers, sample.positions, sample.cell, sample.pbc):
            if tensor.device.type != "cpu":
                raise ValueError("Memory neighbor caching requires CPU dataset samples")
            tensor = tensor.detach().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        return digest.digest()

    def build(self, sample: AtomicSample) -> CSRNeighbors:
        key = self._key(sample)
        if key not in self._cache:
            self._cache[key] = _copy_graph(self.builder.build(sample))
        return _copy_graph(self._cache[key])

    def preload(self, samples: Iterable[AtomicSample]) -> None:
        for sample in samples:
            key = self._key(sample)
            if key not in self._cache:
                self._cache[key] = _copy_graph(self.builder.build(sample))

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


def configure_cache(provider, mode: str):
    if mode not in ("none", "memory"):
        raise ValueError("neighbor_cache must be 'none' or 'memory'")
    if provider is None or mode == "none" or isinstance(provider, CachedNeighborBuilder):
        return provider
    return CachedNeighborBuilder(provider)
