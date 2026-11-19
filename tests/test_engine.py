"""AtomicModel external-neighbor, position-only, and output-mode tests."""

import unittest

import torch

from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput


class ExternalPairModel(AtomicModel):
    """E=sum_e |r_e|^2/2 with optional atomic energies."""

    def compute(self, inputs: ModelInput) -> ModelOutput:
        center_indices = inputs.center_indices
        edge_ptr = inputs.edge_ptr
        edge_vectors = inputs.edge_vectors
        assert center_indices is not None
        assert edge_ptr is not None
        assert edge_vectors is not None
        center_per_edge = torch.repeat_interleave(
            center_indices,
            edge_ptr[1:] - edge_ptr[:-1],
        )
        edge_energies = 0.5 * edge_vectors.square().sum(dim=-1)
        atomic_energies = edge_vectors.new_zeros(center_indices.numel()).index_add(
            0, center_per_edge, edge_energies
        )
        return ModelOutput(
            energy=atomic_energies.sum(),
            atomic_energies=atomic_energies,
        )


class PositionOnlyModel(AtomicModel):
    """Compute each structure's E=R^2/2 without CSR inputs or atomic energies."""

    geometry_mode = "global"
    neighbor_mode = "none"

    def compute(self, inputs: ModelInput) -> ModelOutput:
        positions = inputs.positions
        assert positions is not None
        atom_contributions = 0.5 * positions.square().sum(dim=-1)
        if inputs.center_batch is None:
            return ModelOutput(energy=atom_contributions.sum())

        atom_ptr = inputs.atom_ptr
        assert atom_ptr is not None
        energies = []
        for batch_index in range(inputs.num_structures):
            start = int(atom_ptr[batch_index])
            end = int(atom_ptr[batch_index + 1])
            energies.append(atom_contributions[start:end].sum())
        return ModelOutput(energy=torch.stack(energies))


class DirectForceModel(AtomicModel):
    neighbor_mode = "none"
    force_mode = "direct"

    def compute(self, inputs: ModelInput) -> ModelOutput:
        return ModelOutput(
            forces=torch.ones(
                inputs.atomic_numbers.numel(), 3, dtype=torch.float64
            )
        )


class ChargeOnlyModel(AtomicModel):
    neighbor_mode = "none"
    force_mode = "none"
    capabilities = ("charges",)

    def compute(self, inputs: ModelInput) -> ModelOutput:
        return ModelOutput(
            charges=inputs.atomic_numbers.to(torch.float64) * 0.1,
            properties={"dipole": torch.tensor([1.0, 0.0, 0.0])},
        )


def position_input(n_atoms: int) -> ModelInput:
    positions = torch.zeros(n_atoms, 3, dtype=torch.float64)
    positions[:, 0] = torch.arange(n_atoms, dtype=torch.float64)
    return ModelInput(
        atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
        positions=positions,
    )


class AtomicModelEngineTest(unittest.TestCase):
    def test_external_variable_degree_forces_and_virial(self) -> None:
        inputs = ModelInput(
            atomic_numbers=torch.ones(3, dtype=torch.long),
            center_indices=torch.tensor([0, 1]),
            edge_ptr=torch.tensor([0, 2, 3]),
            neighbor_indices=torch.tensor([1, 2, 0]),
            edge_vectors=torch.tensor(
                [[1.0, 0.0, 0.0]] * 3, dtype=torch.float64
            ),
        )
        output = ExternalPairModel().eval()(inputs)

        torch.testing.assert_close(
            output.energy, torch.tensor(1.5, dtype=torch.float64)
        )
        torch.testing.assert_close(
            output.atomic_energies,
            torch.tensor([1.0, 0.5], dtype=torch.float64),
        )
        torch.testing.assert_close(
            output.forces,
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
        )
        self.assertEqual(output.virial.shape, (3, 3))
        self.assertEqual(float(output.virial[0, 0].detach()), -3.0)

    def test_external_batch_recovers_structure_energies(self) -> None:
        # Three-atom and five-atom structures each have one unit edge per center; CSR allows variable atom counts.
        inputs = ModelInput(
            atomic_numbers=torch.ones(8, dtype=torch.long),
            center_indices=torch.arange(8),
            edge_ptr=torch.arange(9),
            neighbor_indices=torch.tensor([1, 2, 1, 4, 5, 6, 7, 6]),
            edge_vectors=torch.tensor(
                [[1.0, 0.0, 0.0]] * 8, dtype=torch.float64
            ),
            center_batch=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]),
            atom_ptr=torch.tensor([0, 3, 8]),
            num_structures=2,
        )
        output = ExternalPairModel().eval()(inputs)
        torch.testing.assert_close(
            output.energy, torch.tensor([1.5, 2.5], dtype=torch.float64)
        )
        torch.testing.assert_close(
            output.virial[:, 0, 0],
            torch.tensor([-3.0, -5.0], dtype=torch.float64),
        )

    def test_position_only_single_and_variable_size_batch(self) -> None:
        model = PositionOnlyModel().eval()
        input0 = position_input(3)
        input1 = position_input(5)
        output0 = model(input0)
        output1 = model(input1)

        self.assertIsNone(output0.atomic_energies)
        torch.testing.assert_close(
            output0.energy, torch.tensor(2.5, dtype=torch.float64)
        )
        torch.testing.assert_close(
            output1.energy, torch.tensor(15.0, dtype=torch.float64)
        )
        torch.testing.assert_close(output0.forces, -input0.positions)
        self.assertEqual(float(output0.virial[0, 0].detach()), -5.0)

        positions = torch.cat([input0.positions, input1.positions])
        batch = ModelInput(
            atomic_numbers=torch.ones(8, dtype=torch.long),
            positions=positions,
            center_batch=torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]),
            atom_ptr=torch.tensor([0, 3, 8]),
            num_structures=2,
            is_full_structure=True,
        )
        output = model(batch)
        self.assertIsNone(output.atomic_energies)
        torch.testing.assert_close(
            output.energy, torch.tensor([2.5, 15.0], dtype=torch.float64)
        )
        torch.testing.assert_close(output.forces, -positions)
        torch.testing.assert_close(
            output.virial[:, 0, 0],
            torch.tensor([-5.0, -30.0], dtype=torch.float64),
        )
        print("\n[AtomicModel] Position-only, without CSR or atomic energies")
        print("Batch energies for 3/5 atoms:", output.energy)
        print("forces shape:", output.forces.shape)
        print("virials shape:", output.virial.shape)

    def test_direct_force_and_charge_only_models(self) -> None:
        inputs = position_input(3)
        direct = DirectForceModel().eval()(inputs)
        self.assertIsNone(direct.energy)
        torch.testing.assert_close(direct.forces, torch.ones(3, 3, dtype=torch.float64))

        charges = ChargeOnlyModel().eval()(inputs)
        self.assertIsNone(charges.energy)
        self.assertIsNone(charges.forces)
        self.assertEqual(charges.charges.shape, (3,))
        self.assertEqual(charges.properties["dipole"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
