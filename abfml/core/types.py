"""Model geometry inputs, prediction outputs, and tensor collection."""

from dataclasses import dataclass, field, fields
import torch


@dataclass(repr=False, eq=False)
class ModelInput:
    """Geometry and extended physical inputs visible to models, without supervision labels.
    
    Atoms may represent full structures or local/ghost storage. Batches flatten atoms and
    concatenate CSR graphs; input completeness is the responsibility of the preparation layer.
    """

    atomic_numbers: torch.Tensor  # [N]，int64
    positions: torch.Tensor | None = None  # [N,3]
    cell: torch.Tensor | None = None  # [3,3] or [B,3,3].
    pbc: torch.Tensor | None = None  # [3] or [B,3], bool.

    center_indices: torch.Tensor | None = None  # [C]，int64
    edge_ptr: torch.Tensor | None = None  # [C+1]，int64
    neighbor_indices: torch.Tensor | None = None  # [E]，int64
    edge_vectors: torch.Tensor | None = None  # [E,3]，r_ij=R_j-R_i
    edge_shifts: torch.Tensor | None = None  # [E,3]，int64

    center_batch: torch.Tensor | None = None  # [C], structure index of each center.
    atom_ptr: torch.Tensor | None = None  # [B+1], atom offsets for each structure.
    num_structures: int = 1
    is_full_structure: bool = False

    atomic_virial: bool = False  # Whether to request center-assigned atomic virial for local centers.

    total_charge: torch.Tensor | None = None  # Scalar or [B].
    external_field: torch.Tensor | None = None
    properties: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("atomic_numbers", "center_indices", "edge_ptr", "neighbor_indices",
                     "edge_shifts", "center_batch", "atom_ptr"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, torch.as_tensor(value, dtype=torch.long))
        if self.pbc is not None:
            self.pbc = torch.as_tensor(self.pbc, dtype=torch.bool)

    @property
    def num_atoms(self) -> int:
        return self.atomic_numbers.numel()

    @property
    def atom_counts(self) -> torch.Tensor:
        if self.atom_ptr is None:
            return self.atomic_numbers.new_tensor([self.num_atoms])
        return self.atom_ptr[1:] - self.atom_ptr[:-1]

    def to(self, *, device=None, dtype=None) -> "ModelInput":
        """Move model inputs while preserving integer and boolean dtypes."""
        def convert(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value
        return ModelInput(**{item.name: convert(getattr(self, item.name)) for item in fields(self)})

    def require_neighbors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return non-optional CSR fields for callers and static type narrowing."""
        center_indices = self.center_indices
        edge_ptr = self.edge_ptr
        neighbor_indices = self.neighbor_indices
        edge_vectors = self.edge_vectors
        assert center_indices is not None
        assert edge_ptr is not None
        assert neighbor_indices is not None
        assert edge_vectors is not None
        return center_indices, edge_ptr, neighbor_indices, edge_vectors


@dataclass(repr=False, eq=False)
class ModelOutput:
    """Physical predictions allowing energy, forces, and other properties independently."""

    energy: torch.Tensor | None = None  # Scalar total energy or per-structure [B].
    atomic_energies: torch.Tensor | None = None  # [C], optional.
    forces: torch.Tensor | None = None  # [N,3]
    virial: torch.Tensor | None = None  # [3,3] or [B,3,3].
    atomic_virial: torch.Tensor | None = None  # [C,3,3], center-assigned.
    charges: torch.Tensor | None = None  # [N], per-atom charges in elementary-charge units.
    properties: dict[str, torch.Tensor] = field(default_factory=dict)


def collect_outputs(output: ModelOutput) -> dict[str, torch.Tensor]:
    """Collect standard outputs and extended properties for losses, metrics, and logging."""
    predictions = dict(output.properties)
    standard = {
        "energy": output.energy,
        "atomic_energies": output.atomic_energies,
        "forces": output.forces,
        "virial": output.virial,
        "charges": output.charges,
        "atomic_virial": output.atomic_virial,
    }
    for name, value in standard.items():
        if value is not None:
            if name in predictions:
                raise ValueError(f"Duplicate model output: {name}")
            predictions[name] = value
    return predictions
