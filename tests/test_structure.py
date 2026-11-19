"""Tests and readable examples for Structure."""

from abfml.data.reader import from_ase, to_ase

import unittest
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from ase.io import read

from abfml.core.structure import AtomicSample


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY = ROOT / "examples" / "data" / "test.extxyz"


class StructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.atoms = read(TRAJECTORY, index=0)
        cls.structure = from_ase(cls.atoms)

    def test_from_ase_with_reference_labels(self) -> None:
        structure = self.structure
        print("\n[Structure] Structure read from a trajectory")
        print("Atoms:", structure.num_atoms)
        print("atomic_numbers[:8]:", structure.atomic_numbers[:8])
        print("positions[:2]:\n", structure.positions[:2])
        print("cell:\n", structure.cell)
        print("pbc:", structure.pbc)
        print("energy:", structure.energy)
        print("forces shape:", structure.forces.shape)
        print("forces[:2]:\n", structure.forces[:2])
        print("virial:\n", structure.virial)

        self.assertEqual(structure.num_atoms, len(self.atoms))
        self.assertEqual(structure.atomic_numbers.dtype, torch.long)
        self.assertEqual(structure.positions.shape, (len(self.atoms), 3))
        self.assertEqual(structure.cell.shape, (3, 3))
        self.assertEqual(structure.pbc.shape, (3,))
        self.assertIsNotNone(structure.energy)
        self.assertIsNotNone(structure.forces)
        self.assertIsNotNone(structure.virial)
        self.assertIsNone(structure.atomic_energies)

        torch.testing.assert_close(
            structure.energy,
            torch.tensor(self.atoms.get_potential_energy(), dtype=structure.dtype),
        )
        torch.testing.assert_close(
            structure.forces,
            torch.as_tensor(self.atoms.get_forces(), dtype=structure.dtype),
        )
        expected_virial = -torch.as_tensor(
            self.atoms.get_stress(voigt=False), dtype=structure.dtype
        ) * self.atoms.get_volume()
        torch.testing.assert_close(structure.virial, expected_virial)

    def test_unlabeled_structure(self) -> None:
        structure = from_ase(bulk("Cu", a=3.62))
        print("\n[Structure] Structure without supervision labels")
        print("Atoms:", structure.num_atoms)
        print("energy / forces / virial:", structure.energy, structure.forces, structure.virial)

        self.assertIsNone(structure.energy)
        self.assertIsNone(structure.atomic_energies)
        self.assertIsNone(structure.forces)
        self.assertIsNone(structure.virial)

    def test_ase_round_trip(self) -> None:
        round_trip = to_ase(self.structure)
        print("\n[Structure] ASE round-trip conversion")
        print("First atom position:", round_trip.positions[0])
        print("Cell volume:", round_trip.get_volume())

        np.testing.assert_allclose(round_trip.positions, self.atoms.positions)
        np.testing.assert_allclose(round_trip.cell.array, self.atoms.cell.array)
        np.testing.assert_array_equal(round_trip.numbers, self.atoms.numbers)
        np.testing.assert_array_equal(round_trip.pbc, self.atoms.pbc)

    def test_dtype_conversion(self) -> None:
        converted = self.structure.to(dtype=torch.float32)
        print("\n[Structure] dtype conversion")
        print("positions:", converted.positions.dtype)
        print("energy:", converted.energy.dtype)
        print("atomic_numbers:", converted.atomic_numbers.dtype)
        print("pbc:", converted.pbc.dtype)

        self.assertEqual(converted.positions.dtype, torch.float32)
        self.assertEqual(converted.cell.dtype, torch.float32)
        self.assertEqual(converted.energy.dtype, torch.float32)
        self.assertEqual(converted.forces.dtype, torch.float32)
        self.assertEqual(converted.virial.dtype, torch.float32)
        self.assertEqual(converted.atomic_numbers.dtype, torch.long)
        self.assertEqual(converted.pbc.dtype, torch.bool)

    def test_extended_labels_and_dtype_conversion(self) -> None:
        structure = AtomicSample(
            atomic_numbers=[8, 1],
            positions=torch.zeros(2, 3, dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64) * 10,
            pbc=[True, True, True],
            extra_labels={
                "charges": torch.tensor([-0.8, 0.8], dtype=torch.float64),
                "total_charge": torch.tensor(0.0, dtype=torch.float64),
                "dipole": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
                "born_effective_charges": torch.zeros(2, 3, 3, dtype=torch.float64),
                "atom_groups": torch.tensor([0, 1], dtype=torch.long),
            },
        )
        converted = structure.to(dtype=torch.float32)
        print("\n[Structure] Extended supervision labels")
        print("label_names:", structure.label_names)
        print("charges:", structure.get_label("charges"))
        print("dipole:", structure.get_label("dipole"))
        print("atom_groups dtype after .to():", converted.get_label("atom_groups").dtype)

        self.assertEqual(len(structure.label_names), 5)
        self.assertIsNone(structure.get_label("energy"))
        self.assertIsNone(structure.get_label("missing_label"))
        self.assertEqual(structure.get_label("charges").shape, (2,))
        self.assertEqual(structure.get_label("total_charge").shape, ())
        self.assertEqual(structure.get_label("dipole").shape, (3,))
        self.assertEqual(structure.get_label("born_effective_charges").shape, (2, 3, 3))
        for name in ("charges", "total_charge", "dipole", "born_effective_charges"):
            self.assertEqual(converted.get_label(name).dtype, torch.float32)
        self.assertEqual(converted.get_label("atom_groups").dtype, torch.long)
        torch.testing.assert_close(
            converted.get_label("charges"),
            structure.get_label("charges").to(torch.float32),
        )

    def test_ase_extended_label_mapping(self) -> None:
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-1.0,
            charges=np.array([-0.1, 0.1]),
            dipole=np.array([0.0, 0.0, 0.074]),
        )
        structure = from_ase(atoms)
        print("\n[Structure] ASE charge and dipole labels")
        print("energy:", structure.energy)
        print("charges:", structure.get_label("charges"))
        print("dipole:", structure.get_label("dipole"))

        torch.testing.assert_close(
            structure.get_label("charges"),
            torch.tensor([-0.1, 0.1], dtype=structure.dtype),
        )
        torch.testing.assert_close(
            structure.get_label("dipole"),
            torch.tensor([0.0, 0.0, 0.074], dtype=structure.dtype),
        )

    def test_extended_label_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "charges"):
            AtomicSample(
                atomic_numbers=[1, 1],
                positions=torch.zeros(2, 3),
                cell=torch.eye(3),
                pbc=[False, False, False],
                extra_labels={"charges": torch.zeros(3)},
            )
        with self.assertRaisesRegex(ValueError, "standard label"):
            AtomicSample(
                atomic_numbers=[1],
                positions=torch.zeros(1, 3),
                cell=torch.eye(3),
                pbc=[False, False, False],
                extra_labels={"energy": torch.tensor(1.0)},
            )

    def test_invalid_positions_are_rejected(self) -> None:
        print("\n[Structure] Invalid shape checks")
        with self.assertRaisesRegex(ValueError, "positions"):
            AtomicSample(
                atomic_numbers=[29, 29],
                positions=torch.zeros(3),
                cell=torch.eye(3, dtype=torch.float64),
                pbc=[True, True, True],
            )
        print("Passed: invalid positions shape is rejected")


if __name__ == "__main__":
    unittest.main()
