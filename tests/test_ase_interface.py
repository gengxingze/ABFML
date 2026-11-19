"""Tests for the ASE Calculator and standard model loading."""

from abfml.data.reader import from_ase

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from abfml.core.neighbor import ASENeighborBuilder
from abfml.core.structure import AtomicSample
from abfml.interfaces import ABFMLCalculator
from abfml.builtin.lennard_jones import LennardJones
from abfml.data import StructureCollator
from abfml.serialization import save_model


def make_atoms() -> Atoms:
    """Two-atom structure with a nonorthogonal cell and periodic-image neighbors."""
    return Atoms(
        numbers=[1, 1],
        positions=[[0.1, 0.2, 0.3], [1.4, 1.1, 0.8]],
        cell=[[4.0, 0.0, 0.0], [0.4, 4.2, 0.0], [0.2, 0.3, 4.4]],
        pbc=True,
    )


def direct_prediction(model, atoms: Atoms):
    structure = from_ase(atoms)
    neighbors = ASENeighborBuilder(model.cutoff).build(structure)
    with torch.enable_grad():
        return model(StructureCollator()([structure], [neighbors])[0])


class ASEInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LennardJones(
            type_map=[1], cutoff=4.0, epsilon=0.01, sigma=1.2
        ).to(dtype=torch.float64)
        self.model.eval()

    def assert_matches_direct_model(self, calculator: ABFMLCalculator) -> None:
        atoms = make_atoms()
        atoms.calc = calculator
        output = direct_prediction(self.model, atoms)
        energy = output.energy
        atomic_energies = output.atomic_energies
        forces = output.forces
        virial = output.virial

        np.testing.assert_allclose(
            atoms.get_potential_energy(),
            float(energy.detach()),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            atoms.get_potential_energies(),
            atomic_energies.detach().numpy(),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            atoms.get_forces(), forces.detach().numpy(), rtol=1.0e-10, atol=1.0e-12
        )
        expected_stress = -virial.detach().numpy().reshape(3, 3) / atoms.get_volume()
        calculated_stress = atoms.get_stress()
        self.assertEqual(calculated_stress.shape, (6,))
        np.testing.assert_allclose(
            calculated_stress[:3], np.diag(expected_stress), rtol=1.0e-10, atol=1.0e-12
        )
        np.testing.assert_allclose(
            calculated_stress[3:],
            [expected_stress[1, 2], expected_stress[0, 2], expected_stress[0, 1]],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_model_instance_matches_direct_forward(self) -> None:
        self.assert_matches_direct_model(ABFMLCalculator(self.model, device="cpu"))

    def test_standard_model_matches_direct_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(path, self.model, {"common": {"elements": [1], "dtype": "float64"},
                "model": {"name": "lennard_jones", "cutoff": 4., "epsilon": .01, "sigma": 1.2}})
            self.assert_matches_direct_model(ABFMLCalculator(path, device="cpu"))

    def test_nonperiodic_stress_is_rejected(self) -> None:
        atoms = make_atoms()
        atoms.pbc = False
        atoms.calc = ABFMLCalculator(self.model, device="cpu")
        with self.assertRaises(PropertyNotImplementedError):
            atoms.get_stress()


if __name__ == "__main__":
    unittest.main()
