"""Local/ghost CSR input contracts for external callers such as LAMMPS."""

import unittest

import torch

from abfml.core.types import ModelInput
from abfml.builtin.lennard_jones import LennardJones


class ExternalNeighborABITest(unittest.TestCase):
    def test_ghost_atoms_receive_forces_but_not_atomic_energies(self) -> None:
        # Atom 0 is the only local center; atoms 1 and 2 are ghost neighbors.
        atomic_numbers = torch.tensor([1, 1, 1], dtype=torch.int64)
        center_indices = torch.tensor([0], dtype=torch.int64)  # [N_local=1]
        edge_ptr = torch.tensor([0, 2], dtype=torch.int64)  # [N_local+1]
        neighbor_indices = torch.tensor([1, 2], dtype=torch.int64)  # [E=2]
        edge_vectors = torch.tensor(  # [E,3]，r_ij = R_j - R_i
            [[1.5, 0.0, 0.0], [0.0, 1.6, 0.0]], dtype=torch.float64
        )
        model = LennardJones(
            type_map=[1], cutoff=3.0, epsilon=0.01, sigma=1.0
        ).to(dtype=torch.float64)
        model.eval()

        with torch.enable_grad():
            output = model(ModelInput(
                atomic_numbers=atomic_numbers,
                center_indices=center_indices,
                edge_ptr=edge_ptr,
                neighbor_indices=neighbor_indices,
                edge_vectors=edge_vectors,
            ))

        self.assertEqual(output.energy.shape, ())
        self.assertEqual(output.atomic_energies.shape, (1,))
        self.assertEqual(output.forces.shape, (3, 3))
        self.assertEqual(output.virial.shape, (3, 3))
        torch.testing.assert_close(output.energy, output.atomic_energies.sum())
        torch.testing.assert_close(
            output.forces.sum(dim=0), torch.zeros(3, dtype=torch.float64)
        )
        self.assertGreater(float(output.forces[1].abs().sum()), 0.0)
        self.assertGreater(float(output.forces[2].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
