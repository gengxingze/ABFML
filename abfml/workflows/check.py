"""Check MLFF symmetry and energy-derivative consistency on a structure."""

from time import perf_counter

import numpy as np
import torch

from abfml.interfaces.ase import ABFMLCalculator
from abfml.logger import get_logger

logger = get_logger("abfml.check")


def _test_structure(model):
    """Generate reproducible asymmetric periodic structures covering species and same-species
    permutations.
    """
    from ase import Atoms
    from ase.data import covalent_radii

    numbers = list(model.type_map or [])
    if not numbers:
        raise ValueError("Automatic generation requires type_map; provide a structure explicitly")
    spacing = 2.2 * float(np.median(covalent_radii[numbers]))
    if model.cutoff is not None:
        spacing = min(spacing, float(model.cutoff) * 0.4)
    count = max(8, 2 * len(numbers))
    side = int(np.ceil(count ** (1. / 3.)))
    grid = np.indices((side, side, side)).reshape(3, -1).T[:count].astype(float)
    positions = spacing * (grid + np.random.default_rng(0).uniform(-0.08, 0.08, (count, 3)))
    padding = max(float(model.cutoff or 0), spacing * 2)
    positions -= positions.min(axis=0)
    cell = positions.max(axis=0) + 2 * padding
    return Atoms(numbers=(numbers * count)[:count], positions=positions + padding,
                 cell=cell, pbc=True)


def check_model(source, atoms=None, *, device="cpu", index=0, step=1.e-5,
                atol=1.e-5, rtol=1.e-4):
    """Check finite-difference forces/stress and translation, rotation, and species permutation.
    
    If atoms is None, generate a structure without training data; index is retained for
    compatibility. Checks cover local behavior near this structure, not reliability on all
    geometries. Finite differences rebuild neighbors each time; cutoff discontinuities may cause
    failures.
    """
    if not np.isfinite([step, atol, rtol]).all() or step <= 0 or atol < 0 or rtol < 0:
        raise ValueError("step must be positive and tolerances finite and non-negative")
    started = perf_counter()
    generated = atoms is None
    # Finite differences are sensitive to rounding, so checks use float64.
    calculator = ABFMLCalculator(source, device=device, dtype=torch.float64)
    if generated:
        atoms = _test_structure(calculator.model)
    if len(atoms) == 0:
        raise ValueError("Physical checks require at least one atom")
    logger.info("Physical consistency check | Model: %s | Device: %s | Precision: %s",
                type(calculator.model).__name__, calculator.device, calculator.dtype)
    logger.info("Structure: %s | Formula: %s | Atoms: %d | PBC: %s",
                "automatically generated (seed=0)" if generated else "provided",
                atoms.get_chemical_formula(), len(atoms), atoms.pbc.tolist())
    logger.info("Finite difference step: %.1e | atol: %.1e | rtol: %.1e", step, atol, rtol)
    if not {"energy", "forces"}.issubset(calculator.implemented_properties):
        raise ValueError("Physical consistency checks require energy and forces")
    stress_enabled = ("stress" in calculator.implemented_properties and atoms.pbc.any()
                      and atoms.cell.rank == 3)

    def predict(structure):
        structure = structure.copy()
        structure.calc = calculator
        result = {"energy": np.asarray(structure.get_potential_energy()),
                  "forces": structure.get_forces()}
        if stress_enabled:
            result["stress"] = structure.get_stress(voigt=False)
        return result

    baseline = predict(atoms)
    if np.max(np.abs(baseline["forces"])) < 1.e-12:
        logger.warning("Baseline forces are near zero; derivative checks may be uninformative")
    results = {}

    def compare(name, actual, expected, unit):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = np.abs(actual - expected)
        results[name] = {"passed": bool(np.isfinite(actual).all() and
            np.all(error <= atol + rtol * np.abs(expected))),
            "max_abs_error": float(error.max()), "unit": unit}

    shifted = atoms.copy()
    shifted.positions += [0.371, -0.527, 0.193]
    translated = predict(shifted)
    compare("translation_energy", translated["energy"], baseline["energy"], "eV")
    compare("translation_forces", translated["forces"], baseline["forces"], "eV/angstrom")
    angle = 0.47
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    # Rotate around a second axis to test more than in-plane symmetry.
    tilt = np.array([[1., 0, 0], [0, np.cos(.31), -np.sin(.31)], [0, np.sin(.31), np.cos(.31)]])
    rotation = rotation @ tilt
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated.set_cell(atoms.cell.array @ rotation.T, scale_atoms=False)
    prediction = predict(rotated)
    compare("rotation_energy", prediction["energy"], baseline["energy"], "eV")
    compare("rotation_forces", prediction["forces"], baseline["forces"] @ rotation.T, "eV/angstrom")
    if stress_enabled:
        compare("rotation_stress", prediction["stress"], rotation @ baseline["stress"] @ rotation.T, "eV/angstrom^3")
    permutation = np.arange(len(atoms))
    for number in np.unique(atoms.numbers):
        indices = np.flatnonzero(atoms.numbers == number)
        permutation[indices] = indices[::-1]
    prediction = predict(atoms[permutation])
    compare("permutation_energy", prediction["energy"], baseline["energy"], "eV")
    compare("permutation_forces", prediction["forces"], baseline["forces"][permutation], "eV/angstrom")
    compare("net_force", baseline["forces"].sum(axis=0), np.zeros(3), "eV/angstrom")
    finite_forces = np.zeros((len(atoms), 3))
    for atom in range(len(atoms)):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, axis] += step
            minus.positions[atom, axis] -= step
            finite_forces[atom, axis] = -(predict(plus)["energy"] - predict(minus)["energy"]) / (2 * step)
    compare("force_finite_difference", baseline["forces"], finite_forces, "eV/angstrom")
    if stress_enabled:
        stress = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                strain = np.zeros((3, 3))
                strain[a, b] = step
                plus, minus = atoms.copy(), atoms.copy()
                plus.set_cell(atoms.cell.array @ (np.eye(3) + strain), scale_atoms=True)
                minus.set_cell(atoms.cell.array @ (np.eye(3) - strain), scale_atoms=True)
                stress[a, b] = (predict(plus)["energy"] - predict(minus)["energy"]) / (2 * step * atoms.get_volume())
        compare("stress_finite_difference", baseline["stress"], stress, "eV/angstrom^3")
    report = {"passed": all(result["passed"] for result in results.values()),
              "step": step, "atol": atol, "rtol": rtol, "checks": results,
              "skipped": [] if stress_enabled else ["stress: requires periodic full-rank cell and model virial"]}

    rows = [["Check", "Result", "Max abs. error", "Unit"]]
    for name, result in results.items():
        rows.append([name, "PASS" if result["passed"] else "FAIL",
                     f"{result['max_abs_error']:.2e}", result["unit"].replace("angstrom", "A")])
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows]
    lines.insert(1, "-+-".join("-" * width for width in widths))
    logger.info("Check results:\n%s", "\n".join(lines))
    for skipped in report["skipped"]:
        logger.info("Skipped: %s", skipped)
    report["elapsed_seconds"] = perf_counter() - started
    logger.info("Check completed | Result: %s | Passed: %d/%d | Total time: %.2f s",
                "PASS" if report["passed"] else "FAIL",
                sum(item["passed"] for item in results.values()), len(results), report["elapsed_seconds"])
    return report
