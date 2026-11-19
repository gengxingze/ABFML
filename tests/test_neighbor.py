"""Tests and readable examples for CSRNeighbors and ASENeighborBuilder."""

from abfml.data.reader import from_ase

import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from abfml.core.neighbor import ASENeighborBuilder
from abfml.core.neighbor import CSRNeighbors
from abfml.core.structure import AtomicSample


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY = ROOT / "examples" / "data" / "test.extxyz"


class NeighborTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        atoms = read(TRAJECTORY, index=0)
        cls.structure = from_ase(atoms)
        cls.neighbors = ASENeighborBuilder(cutoff=6.0).build(cls.structure)

    def test_csr_and_edge_view(self) -> None:
        neighbors = self.neighbors
        print("\n[CSRNeighbors] CSR and GNN edge views")
        print("Centers / edges:", neighbors.num_centers, neighbors.num_edges)
        print("center_indices[:8]:", neighbors.center_indices[:8])
        print("edge_ptr[:8]:", neighbors.edge_ptr[:8])
        print("degrees[:8]:", neighbors.degrees[:8])
        print("neighbor_indices[:10]:", neighbors.neighbor_indices[:10])
        print("edge_vectors[:3]:\n", neighbors.edge_vectors[:3])
        print("edge_shifts[:3]:\n", neighbors.edge_shifts[:3])
        print("edge_index[:, :5]:\n", neighbors.edge_index[:, :5])

        self.assertEqual(neighbors.num_centers, self.structure.num_atoms)
        self.assertGreater(neighbors.num_edges, 0)
        self.assertEqual(int(neighbors.edge_ptr[-1].item()), neighbors.num_edges)
        self.assertEqual(neighbors.edge_index.shape, (2, neighbors.num_edges))
        self.assertTrue(bool((neighbors.degrees > 0).all()))

        distances = torch.linalg.vector_norm(neighbors.edge_vectors, dim=-1)
        self.assertLessEqual(float(distances.max().item()), 6.0 + 1e-8)
        expected_centers = torch.repeat_interleave(
            neighbors.center_indices, neighbors.degrees
        )
        torch.testing.assert_close(neighbors.edge_index[0], expected_centers)
        torch.testing.assert_close(neighbors.edge_index[1], neighbors.neighbor_indices)

    def test_dense_view(self) -> None:
        neighbors = self.neighbors
        dense = neighbors.to_dense()
        print("\n[DenseNeighbors] Descriptor model view")
        print("neighbor_indices shape:", dense.neighbor_indices.shape)
        print("edge_vectors shape:", dense.edge_vectors.shape)
        print("mask shape:", dense.mask.shape)
        print("Valid edges:", int(dense.mask.sum().item()))
        print("First center neighbor indices[:12]:", dense.neighbor_indices[0, :12])
        print("First center mask[:12]:", dense.mask[0, :12])
        print("First center vectors[:3]:\n", dense.edge_vectors[0, :3])

        self.assertEqual(dense.neighbor_indices.shape[0], neighbors.num_centers)
        self.assertEqual(int(dense.mask.sum().item()), neighbors.num_edges)
        self.assertTrue(bool((dense.neighbor_indices[~dense.mask] == -1).all()))
        torch.testing.assert_close(dense.edge_vectors[dense.mask], neighbors.edge_vectors)

        capacity = int(neighbors.degrees.max()) + 2
        dense_limited = neighbors.to_dense(max_neighbors=capacity)
        print("Shape with explicit padded capacity:", dense_limited.neighbor_indices.shape)
        self.assertEqual(dense_limited.neighbor_indices.shape[1], capacity)
        self.assertEqual(int(dense_limited.mask.sum()), neighbors.num_edges)

    def test_external_neighbor_arrays(self) -> None:
        external = CSRNeighbors(
            center_indices=self.neighbors.center_indices,
            edge_ptr=self.neighbors.edge_ptr,
            neighbor_indices=self.neighbors.neighbor_indices,
            edge_vectors=self.neighbors.edge_vectors,
            edge_shifts=self.neighbors.edge_shifts,
        )
        print("\n[CSRNeighbors] Direct construction from external arrays")
        print("Centers / edges:", external.num_centers, external.num_edges)
        print("No ASE calls; edge_index matches the original arrays")

        torch.testing.assert_close(external.edge_index, self.neighbors.edge_index)
        torch.testing.assert_close(external.edge_vectors, self.neighbors.edge_vectors)

    def test_overflow_policies(self) -> None:
        print("\n[ASENeighborBuilder] Neighbor overflow policies")
        with self.assertRaises(ValueError) as context:
            ASENeighborBuilder(cutoff=6.0, max_neighbors=1).build(self.structure)
        print("Error policy output:", context.exception)
        self.assertIn("overflow", str(context.exception))

        nearest = ASENeighborBuilder(
            cutoff=6.0,
            max_neighbors=4,
            overflow="nearest",
        ).build(self.structure)
        print("Nearest policy: edges / maximum degree:", nearest.num_edges, int(nearest.degrees.max().item()))
        print("Nearest policy: degrees of first 8 centers:", nearest.degrees[:8])
        self.assertLessEqual(int(nearest.degrees.max().item()), 4)

    def test_periodic_self_images(self) -> None:
        atoms = Atoms(
            "Cu",
            positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3) * 2.0,
            pbc=[True, True, True],
        )
        neighbors = ASENeighborBuilder(cutoff=2.1).build(from_ase(atoms))
        distances = torch.linalg.vector_norm(neighbors.edge_vectors, dim=-1)

        print("\n[CSRNeighbors] Periodic self-images in a small unit cell")
        print("Centers / edges:", neighbors.num_centers, neighbors.num_edges)
        print("neighbor_indices:", neighbors.neighbor_indices)
        print("edge_shifts:\n", neighbors.edge_shifts)
        print("edge_vectors:\n", neighbors.edge_vectors)
        print(" distances:", distances)

        self.assertEqual(neighbors.num_edges, 6)
        self.assertTrue(bool((neighbors.neighbor_indices == 0).all()))
        self.assertTrue(bool((neighbors.edge_shifts != 0).any(dim=1).all()))
        torch.testing.assert_close(distances, torch.full_like(distances, 2.0))

    def test_empty_neighbors(self) -> None:
        empty = CSRNeighbors(
            center_indices=[],
            edge_ptr=[0],
            neighbor_indices=[],
            edge_vectors=torch.empty((0, 3), dtype=torch.float64),
        )
        print("\n[CSRNeighbors] Empty neighbor data")
        print("Centers / edges:", empty.num_centers, empty.num_edges)
        print("edge_index shape:", empty.edge_index.shape)
        print("dense mask shape:", empty.to_dense().mask.shape)

        self.assertEqual(empty.num_centers, 0)
        self.assertEqual(empty.num_edges, 0)
        self.assertEqual(empty.edge_index.shape, (2, 0))
        self.assertEqual(empty.to_dense().mask.shape, (0, 0))

    def test_invalid_csr_is_rejected(self) -> None:
        print("\n[CSRNeighbors] Invalid CSR checks")
        with self.assertRaisesRegex(ValueError, "edge_ptr"):
            CSRNeighbors(
                center_indices=[],
                edge_ptr=[1],
                neighbor_indices=[],
                edge_vectors=torch.empty((0, 3), dtype=torch.float64),
            )
        with self.assertRaisesRegex(ValueError, "negative"):
            CSRNeighbors(
                center_indices=[0],
                edge_ptr=[0, 1],
                neighbor_indices=[-1],
                edge_vectors=torch.zeros((1, 3), dtype=torch.float64),
            )
        print("Passed: invalid edge_ptr and negative neighbor indices are rejected")


if __name__ == "__main__":
    unittest.main()
