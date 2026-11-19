"""Core atomic structure data for ABFML."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from abfml.core.schema import LABELS


_STANDARD_LABELS = tuple(name for name, spec in LABELS.items() if spec.sample_field)


@dataclass
class AtomicSample:
    """One atomic structure with standard and optional extended labels.
    
    Geometry and labels stay together to preserve alignment. Neighbors depend on model cutoffs
    and are stored separately. extra_labels accepts arbitrary tensor labels without modifying
    this class. Internal units are angstrom for positions, eV for energy, eV/angstrom for
    forces, and eV for virial.
    """

    atomic_numbers: torch.Tensor  # [N], int64
    positions: torch.Tensor  # [N, 3], floating point.
    cell: torch.Tensor  # [3, 3], floating point; each row is a lattice vector.
    pbc: torch.Tensor  # [3], bool
    energy: torch.Tensor | None = None  # Scalar.
    atomic_energies: torch.Tensor | None = None  # [N]
    forces: torch.Tensor | None = None  # [N, 3]
    virial: torch.Tensor | None = None  # [3, 3]
    extra_labels: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atomic_numbers = torch.as_tensor(self.atomic_numbers, dtype=torch.long)
        self.positions = torch.as_tensor(self.positions)
        self.cell = torch.as_tensor(self.cell)
        self.pbc = torch.as_tensor(self.pbc, dtype=torch.bool)

        for name in _STANDARD_LABELS:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, torch.as_tensor(value))
        self.extra_labels = {
            name: torch.as_tensor(value) for name, value in self.extra_labels.items()
        }

        self._validate()

    @property
    def num_atoms(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        return self.positions.dtype

    @property
    def label_names(self) -> tuple[str, ...]:
        standard = tuple(
            name for name in _STANDARD_LABELS if getattr(self, name) is not None
        )
        return standard + tuple(self.extra_labels)

    def get_label(self, name: str) -> torch.Tensor | None:
        """Read a standard or extended label; return None when absent."""
        if name in _STANDARD_LABELS:
            return getattr(self, name)
        return self.extra_labels.get(name)


    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "AtomicSample":
        """Return a device/dtype-converted object while preserving integer and boolean types."""
        target_dtype = self.dtype if dtype is None else dtype

        def convert(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.to(device=device, dtype=target_dtype)

        def convert_extra(value: torch.Tensor) -> torch.Tensor:
            if value.is_floating_point():
                return value.to(device=device, dtype=target_dtype)
            return value.to(device=device)

        return AtomicSample(
            atomic_numbers=self.atomic_numbers.to(device=device),
            positions=convert(self.positions),
            cell=convert(self.cell),
            pbc=self.pbc.to(device=device),
            energy=convert(self.energy),
            atomic_energies=convert(self.atomic_energies),
            forces=convert(self.forces),
            virial=convert(self.virial),
            extra_labels={
                name: convert_extra(value) for name, value in self.extra_labels.items()
            },
        )

    def _validate(self) -> None:
        n_atoms = self.num_atoms
        if self.atomic_numbers.ndim != 1:
            raise ValueError("atomic_numbers must have shape [N]")
        if self.positions.shape != (n_atoms, 3):
            raise ValueError(f"positions must have shape ({n_atoms}, 3)")
        if self.cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")
        if self.pbc.shape != (3,):
            raise ValueError("pbc must have shape (3,)")
        if not self.positions.is_floating_point() or not self.cell.is_floating_point():
            raise TypeError("positions and cell must use a floating-point dtype")
        if self.positions.dtype != self.cell.dtype:
            raise TypeError("positions and cell must have the same dtype")
        if self.positions.device != self.cell.device:
            raise ValueError("positions and cell must be on the same device")

        expected_shapes = {name: ((n_atoms,) if LABELS[name].scope in {"atom", "center"} else ())
                           + LABELS[name].shape for name in _STANDARD_LABELS}
        for name, expected_shape in expected_shapes.items():
            value = getattr(self, name)
            if value is not None and value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")

        extended_shapes = {name: ((n_atoms,) if spec.scope in {"atom", "center"} else ()) + spec.shape
                           for name, spec in LABELS.items() if not spec.sample_field and spec.source is None}
        for name, value in self.extra_labels.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Extended label names must be non-empty strings")
            if name in expected_shapes:
                raise ValueError(f"{name} is a standard label, not an extended label")
            if name in extended_shapes and value.shape != extended_shapes[name]:
                raise ValueError(
                    f"{name} must have shape {extended_shapes[name]}, got {value.shape}"
                )
            if value.device != self.device:
                raise ValueError(f"Extended label {name} must be on the structure device")
