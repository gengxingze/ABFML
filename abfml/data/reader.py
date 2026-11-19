"""Dataset reading and conversion of ASE geometry and labels."""

from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress
from ase.io import iread

from abfml.core.structure import AtomicSample


class ASEReader:
    """Reiterable reader parsing frames in file order without caching samples.
    
    index applies to each file independently. Only existing labels are read; conversion never
    invokes ASE calculators for new results.
    """

    def __init__(self, files: str | Path | Sequence[str | Path], *,
                 file_format: str | None = None, index: int | slice | str = ":",
                 dtype: torch.dtype = torch.float64) -> None:
        self.files = tuple(Path(p) for p in ([files] if isinstance(files, (str, Path)) else files))
        if not self.files:
            raise ValueError("files must not be empty")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Reader dtype must be float32 or float64")
        self.file_format = file_format
        self.index = str(index) if isinstance(index, int) else index
        self.dtype = dtype

    def __iter__(self) -> Iterator[AtomicSample]:
        for path in self.files:
            try:
                for atoms in iread(path, index=self.index, format=self.file_format):
                    yield from_ase(atoms, dtype=self.dtype)
            except Exception as error:
                error.add_note(f"While reading structures from {path}")
                raise


def from_ase(atoms: Atoms, dtype: torch.dtype = torch.float64) -> "AtomicSample":
    """Convert ASE Atoms without triggering new calculations."""
    results = atoms.calc.results if atoms.calc is not None else {}

    energy = results.get("energy")
    forces = results.get("forces")
    atomic_energies = results.get("energies")
    extra_labels: dict[str, torch.Tensor] = {}
    for result_name, label_name in (
        ("charges", "charges"),
        ("total_charge", "total_charge"),
        ("dipole", "dipole"),
    ):
        value = results.get(result_name)
        if value is not None:
            extra_labels[label_name] = torch.as_tensor(value, dtype=dtype)

    virial = None
    stress = results.get("stress")
    if stress is not None:
        stress_tensor = torch.as_tensor(stress, dtype=dtype)
        if stress_tensor.shape == (6,):
            stress_tensor = torch.as_tensor(
                voigt_6_to_full_3x3_stress(stress_tensor.numpy()),
                dtype=dtype,
            )
        if stress_tensor.shape != (3, 3):
            raise ValueError(f"ASE stress must have shape (6,) or (3, 3), got {stress_tensor.shape}")
        virial = -stress_tensor * float(atoms.get_volume())

    return AtomicSample(
        atomic_numbers=torch.as_tensor(atoms.numbers.copy(), dtype=torch.long),
        positions=torch.as_tensor(atoms.positions.copy(), dtype=dtype),
        cell=torch.as_tensor(atoms.cell.array.copy(), dtype=dtype),
        pbc=torch.as_tensor(atoms.pbc.copy(), dtype=torch.bool),
        energy=None if energy is None else torch.as_tensor(energy, dtype=dtype),
        atomic_energies=(
            None
            if atomic_energies is None
            else torch.as_tensor(atomic_energies, dtype=dtype)
        ),
        forces=None if forces is None else torch.as_tensor(forces, dtype=dtype),
        virial=virial,
        extra_labels=extra_labels,
    )


def to_ase(sample: AtomicSample) -> Atoms:
    """Convert geometry to ASE Atoms without reference labels."""
    return Atoms(
        numbers=sample.atomic_numbers.detach().cpu().numpy(),
        positions=sample.positions.detach().cpu().numpy(),
        cell=sample.cell.detach().cpu().numpy(),
        pbc=sample.pbc.detach().cpu().numpy(),
    )
