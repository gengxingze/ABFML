"""AtomicModel external-neighbor, dense-view, and direct-output tests."""

from abfml.data.reader import from_ase

import unittest
from abfml.data import StructureCollator

import torch
from ase import Atoms

from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput
from abfml.core.neighbor import ASENeighborBuilder
from abfml.core.neighbor import CSRNeighbors
from abfml.core.structure import AtomicSample


class PairField(AtomicModel):
    """E=sum_edges |r_ij|^2/2 for analytic validation of shared mechanics."""

    def compute(self, inputs: ModelInput) -> ModelOutput:
        center_indices, edge_ptr, _, edge_vectors = inputs.require_neighbors()
        edge_energies = 0.5 * edge_vectors.square().sum(dim=-1)
        center_slots = torch.repeat_interleave(
            torch.arange(center_indices.numel(), device=center_indices.device),
            edge_ptr[1:] - edge_ptr[:-1],
        )
        atomic_energies = edge_vectors.new_zeros(center_indices.numel()).index_add(
            0, center_slots, edge_energies
        )
        return ModelOutput(
            energy=atomic_energies.sum(),
            atomic_energies=atomic_energies,
        )


class DensePairField(PairField):
    """Convert CSR to fixed-width dense neighbors inside compute()."""

    def compute(self, inputs: ModelInput) -> ModelOutput:
        _, edge_ptr, neighbor_indices, edge_vectors = inputs.require_neighbors()
        _, dense_vectors, dense_mask = self.to_dense(
            edge_ptr,
            neighbor_indices,
            edge_vectors,
            max_neighbors=2,
        )
        edge_energies = (
            0.5
            * dense_vectors.square().sum(dim=-1)
            * dense_mask.to(dense_vectors.dtype)
        )
        atomic_energies = edge_energies.sum(dim=-1)
        return ModelOutput(
            energy=atomic_energies.sum(),
            atomic_energies=atomic_energies,
        )


class DirectForceField(AtomicModel):
    """Direct-force models may return energy and extra properties without differentiable energy."""

    force_mode = "direct"

    def compute(self, inputs: ModelInput) -> ModelOutput:
        _, _, _, edge_vectors = inputs.require_neighbors()
        return ModelOutput(
            energy=edge_vectors.detach().sum(),
            forces=torch.full(
                (inputs.atomic_numbers.numel(), 3),
                7.0,
                dtype=edge_vectors.dtype,
                device=edge_vectors.device,
            ),
            properties={
                "charges": inputs.atomic_numbers.to(edge_vectors.dtype) * 0.1,
                "dipole": torch.tensor(
                    [1.0, 2.0, 3.0], dtype=edge_vectors.dtype
                ),
            },
        )


class InvalidDirectForceField(PairField):
    force_mode = "direct"

    def compute(self, inputs: ModelInput) -> ModelOutput:
        _, _, _, edge_vectors = inputs.require_neighbors()
        return ModelOutput(
            energy=edge_vectors.sum() * 0.0,
            forces=edge_vectors.new_zeros(inputs.atomic_numbers.numel(), 2),
        )


class AtomicModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = PairField(type_map=[1], cutoff=3.0).eval()
        self.atomic_numbers = torch.tensor([1, 1], dtype=torch.long)
        self.center_indices = torch.tensor([0, 1], dtype=torch.long)
        self.edge_ptr = torch.tensor([0, 1, 2], dtype=torch.long)
        self.neighbor_indices = torch.tensor([1, 0], dtype=torch.long)
        self.edge_vectors = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64
        )
        self.neighbors = CSRNeighbors(
            center_indices=self.center_indices,
            edge_ptr=self.edge_ptr,
            neighbor_indices=self.neighbor_indices,
            edge_vectors=self.edge_vectors,
            edge_shifts=torch.zeros(2, 3, dtype=torch.long),
        )

    def model_input(self, edge_vectors: torch.Tensor | None = None) -> ModelInput:
        if edge_vectors is None:
            edge_vectors = self.edge_vectors
        return ModelInput(
            atomic_numbers=self.atomic_numbers,
            center_indices=self.center_indices,
            edge_ptr=self.edge_ptr,
            neighbor_indices=self.neighbor_indices,
            edge_vectors=edge_vectors,
            edge_shifts=self.neighbors.edge_shifts,
        )

    def assert_forward_equal(self, actual: ModelOutput, expected: ModelOutput) -> None:
        for name in ("energy", "atomic_energies", "forces", "virial"):
            value = getattr(actual, name)
            reference = getattr(expected, name)
            self.assertEqual(value is None, reference is None)
            if value is not None:
                torch.testing.assert_close(value, reference)
        self.assertEqual(actual.properties.keys(), expected.properties.keys())
        for name in actual.properties:
            torch.testing.assert_close(actual.properties[name], expected.properties[name])

    def test_external_vector_energy_and_forces(self) -> None:
        inputs = self.model_input()
        output = self.model(inputs)
        print("\n[AtomicModel] External neighbor vectors")
        print("energy:", output.energy)
        print("atomic_energies:", output.atomic_energies)
        print("forces:\n", output.forces)
        print("virial:\n", output.virial)

        torch.testing.assert_close(output.energy, torch.tensor(1.0, dtype=torch.float64))
        torch.testing.assert_close(
            output.atomic_energies,
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )
        torch.testing.assert_close(
            output.forces,
            torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=torch.float64),
        )
        expected_virial = torch.zeros(3, 3, dtype=torch.float64)
        expected_virial[0, 0] = -2.0
        torch.testing.assert_close(output.virial, expected_virial)
        self.assertFalse(hasattr(inputs, "energy"))

    def test_external_neighbor_prediction_from_structure(self) -> None:
        atoms = Atoms(
            "H2",
            positions=[[0, 0, 0], [1, 0, 0]],
            cell=[10, 10, 10],
            pbc=False,
        )
        structure = from_ase(atoms)
        neighbors = ASENeighborBuilder(self.model.cutoff).build(structure)
        output = self.model(StructureCollator()([structure], [neighbors])[0])
        output.energy = output.energy.squeeze(0)
        output.virial = output.virial.squeeze(0)
        expected = self.model(self.model_input())
        self.assert_forward_equal(output, expected)

    def test_full_structure_input(self) -> None:
        atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
        structure = from_ase(atoms)
        inputs, _ = StructureCollator()([structure], [self.neighbors])
        self.assertTrue(inputs.is_full_structure)
        torch.testing.assert_close(inputs.positions, structure.positions)
        torch.testing.assert_close(inputs.cell[0], structure.cell)
        output = self.model(inputs)
        self.assertEqual(output.forces.shape, (2, 3))

    def test_direct_forces_bypass_autograd(self) -> None:
        output = DirectForceField(type_map=[1], cutoff=3.0).eval()(
            self.model_input()
        )
        print("\n[AtomicModel] Direct forces and extended properties")
        print("properties:", output.properties)

        self.assertIsNone(output.atomic_energies)
        self.assertIsNone(output.virial)
        torch.testing.assert_close(output.energy, self.edge_vectors.sum())
        torch.testing.assert_close(
            output.forces, torch.full((2, 3), 7.0, dtype=torch.float64)
        )
        torch.testing.assert_close(
            output.properties["charges"],
            torch.tensor([0.1, 0.1], dtype=torch.float64),
        )
        torch.testing.assert_close(
            output.properties["dipole"],
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        )

    def test_invalid_direct_force_shape_is_rejected(self) -> None:
        model = InvalidDirectForceField(type_map=[1], cutoff=3.0).eval()
        with self.assertRaisesRegex(ValueError, "forces must have shape"):
            model(self.model_input())

    def test_dense_feature_model_inside_compute(self) -> None:
        model = DensePairField(type_map=[1], cutoff=3.0).eval()
        dense_indices, dense_vectors, dense_mask = model.to_dense(
            self.edge_ptr,
            self.neighbor_indices,
            self.edge_vectors,
            max_neighbors=2,
        )
        print("\n[AtomicModel] Internal CSR -> dense conversion in descriptor models")
        print("dense_indices:\n", dense_indices)
        print("dense_mask:\n", dense_mask)
        print("dense_vectors shape:", dense_vectors.shape)

        torch.testing.assert_close(
            dense_indices,
            torch.tensor([[1, -1], [0, -1]], dtype=torch.long),
        )
        torch.testing.assert_close(
            dense_mask,
            torch.tensor([[True, False], [True, False]]),
        )
        expected = self.model(self.model_input())
        actual = model(self.model_input())
        self.assert_forward_equal(actual, expected)

    def test_csr_to_species_segmented_dense(self) -> None:
        model = PairField(type_map=[1, 8], cutoff=3.0).eval()
        edge_ptr = torch.tensor([0, 4, 6], dtype=torch.long)
        neighbor_indices = torch.tensor([0, 1, 2, 3, 3, 2], dtype=torch.long)
        neighbor_type_ids = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
        edge_vectors = torch.arange(18, dtype=torch.float64).reshape(6, 3)

        global_indices, _, _ = model.to_dense(
            edge_ptr, neighbor_indices, edge_vectors, max_neighbors=4
        )
        torch.testing.assert_close(
            global_indices,
            torch.tensor([[0, 1, 2, 3], [3, 2, -1, -1]]),
        )

        dense_indices, dense_vectors, dense_mask = model.to_dense(
            edge_ptr,
            neighbor_indices,
            edge_vectors,
            max_neighbors=3,
            neighbor_type_ids=neighbor_type_ids,
        )
        expected_indices = torch.tensor(
            [[0, 2, -1, 1, 3, -1], [2, -1, -1, 3, -1, -1]]
        )
        expected_mask = expected_indices >= 0
        source_edges = torch.tensor(
            [[0, 2, -1, 1, 3, -1], [5, -1, -1, 4, -1, -1]]
        )
        expected_vectors = torch.zeros(2, 6, 3, dtype=torch.float64)
        expected_vectors[expected_mask] = edge_vectors[source_edges[expected_mask]]
        torch.testing.assert_close(dense_indices, expected_indices)
        torch.testing.assert_close(dense_mask, expected_mask)
        torch.testing.assert_close(dense_vectors, expected_vectors)

        custom_indices, _, custom_mask = model.to_dense(
            edge_ptr,
            neighbor_indices,
            edge_vectors,
            max_neighbors=1,
            neighbor_type_ids=neighbor_type_ids,
            type_slots=[3, 2],
        )
        torch.testing.assert_close(
            custom_indices,
            torch.tensor([[0, 2, -1, 1, 3], [2, -1, -1, 3, -1]]),
        )
        torch.testing.assert_close(custom_mask, custom_indices >= 0)

        with self.assertRaisesRegex(ValueError, "species CSR degree"):
            model.to_dense(
                edge_ptr,
                neighbor_indices,
                edge_vectors,
                max_neighbors=1,
                neighbor_type_ids=neighbor_type_ids,
                type_slots=[1, 2],
            )

    def test_virial_matches_finite_strain_derivative(self) -> None:
        output = self.model(self.model_input())
        epsilon = 1.0e-5
        energy_plus = self.model(
            self.model_input(self.edge_vectors * (1.0 + epsilon))
        ).energy
        energy_minus = self.model(
            self.model_input(self.edge_vectors * (1.0 - epsilon))
        ).energy
        strain_derivative = (energy_plus - energy_minus) / (2.0 * epsilon)
        torch.testing.assert_close(
            -output.virial[0, 0],
            strain_derivative,
            rtol=1.0e-6,
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
