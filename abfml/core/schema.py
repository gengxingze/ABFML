"""Shared physical-label metadata, without unit conversion or model computation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelSpec:
    """shape describes one entity; scope determines the atom or center dimension."""

    scope: str
    shape: tuple[int, ...]
    unit: str
    per_atom: str | None = None
    display_unit: str | None = None
    display_scale: float = 1.
    sample_field: bool = False
    source: str | None = None


LABELS = {
    "energy": LabelSpec("structure", (), "eV", per_atom="energy_per_atom", sample_field=True),
    "atomic_energies": LabelSpec("center", (), "eV", sample_field=True),
    "forces": LabelSpec("atom", (3,), "eV/A", display_unit="meV/A", display_scale=1000., sample_field=True),
    "virial": LabelSpec("structure", (3, 3), "eV", per_atom="virial_per_atom", sample_field=True),
    "charges": LabelSpec("atom", (), "e"),
    "total_charge": LabelSpec("structure", (), "e"),
    "dipole": LabelSpec("structure", (3,), "e*A"),
    "born_effective_charges": LabelSpec("atom", (3, 3), "e"),
    "atomic_virial": LabelSpec("center", (3, 3), "eV"),
    "energy_per_atom": LabelSpec("structure", (), "eV/atom", display_unit="meV/atom",
                                 display_scale=1000., source="energy"),
    "virial_per_atom": LabelSpec("structure", (3, 3), "eV/atom", display_unit="meV/atom",
                                 display_scale=1000., source="virial"),
    "force_norm": LabelSpec("atom", (), "eV/A", source="forces"),
    "stress": LabelSpec("structure", (3, 3), "eV/A^3", source="virial"),
}


def label_unit(name: str, *, display=False) -> str:
    """Do not infer units for unknown labels; callers may explicitly declare their scopes."""
    spec = LABELS.get(name)
    if spec is None:
        return "label_units"
    return (spec.display_unit or spec.unit) if display else spec.unit


def normalize_label(name, value, atom_counts):
    """Normalize only structure labels declaring per_atom, preserving tensor shape."""
    spec = LABELS.get(name)
    if spec is None or spec.per_atom is None:
        return value
    if value.ndim == 0:
        return value / atom_counts.sum()
    return value / atom_counts.reshape((-1,) + (1,)*(value.ndim-1))
