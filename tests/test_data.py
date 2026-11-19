"""Data reading, splitting, DataLoader collation, and training statistics tests."""

from abfml.data.reader import ASEReader

import unittest
from pathlib import Path

import torch
from ase.io import read
from torch.utils.data import DataLoader

from abfml.core.neighbor import ASENeighborBuilder
from abfml.core import AtomicSample, ModelInput
from abfml.data import StructureCollator, AtomicDataset, create_dataloader
from abfml.builtin.statistics import energy_shifts


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY = ROOT / "examples" / "data" / "test.extxyz"


def labeled_structure(numbers: list[int], energy: float) -> AtomicSample:
    n_atoms = len(numbers)
    positions = torch.zeros(n_atoms, 3, dtype=torch.float64)
    positions[:, 0] = torch.arange(n_atoms, dtype=torch.float64)
    return AtomicSample(
        atomic_numbers=numbers,
        positions=positions,
        cell=torch.eye(3, dtype=torch.float64) * 20,
        pbc=[False, False, False],
        energy=torch.tensor(energy, dtype=torch.float64),
    )


class DataReadAndDatasetTest(unittest.TestCase):
    def test_read_trajectory_and_dataset(self) -> None:
        frames = read(TRAJECTORY, index=":")
        structures = list(ASEReader(TRAJECTORY))
        dataset = AtomicDataset(structures)
        print("\n[data] ASE trajectory -> AtomicDataset")
        print("Frames:", len(dataset))
        print("Atoms in first frame:", dataset[0].num_atoms)
        print("Energy of first frame:", dataset[0].energy)

        self.assertEqual(len(dataset), len(frames))
        self.assertGreater(len(dataset), 1)
        self.assertEqual(dataset[0].num_atoms, len(frames[0]))
        self.assertIsNotNone(dataset[0].energy)
        self.assertIsNotNone(dataset[0].forces)
        subset = dataset.subset([2, 0])
        self.assertEqual(len(subset), 2)
        self.assertIs(subset[0], dataset[2])
        self.assertIs(subset[1], dataset[0])

    def test_multiple_files_single_frame_and_dtype(self) -> None:
        dataset = AtomicDataset(ASEReader([TRAJECTORY, TRAJECTORY], index=0, dtype=torch.float32))
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0].positions.dtype, torch.float32)
        torch.testing.assert_close(dataset[0].positions, dataset[1].positions)


class DataCollateTest(unittest.TestCase):
    def test_dataloader_builds_structure_batch(self) -> None:
        dataset = AtomicDataset(ASEReader(TRAJECTORY, index=slice(0, 2)))
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=StructureCollator(ASENeighborBuilder(cutoff=6.0)),
        )
        batch, labels = next(iter(loader))
        frames = read(TRAJECTORY, index=":2")
        num_atoms = sum(len(frame) for frame in frames)
        print("\n[data] DataLoader -> ModelInput and labels")
        print("Structures / atoms:", batch.num_structures, batch.num_atoms)
        print("atom_ptr:", batch.atom_ptr)
        print("Edges:", batch.neighbor_indices.numel())
        print("energy shape:", labels["energy"].shape)
        print("forces shape:", labels["forces"].shape)

        self.assertIsInstance(batch, ModelInput)
        self.assertEqual(batch.num_structures, 2)
        self.assertEqual(batch.num_atoms, num_atoms)
        self.assertGreater(batch.neighbor_indices.numel(), 0)
        self.assertEqual(labels["energy"].shape, (2,))
        self.assertEqual(labels["forces"].shape, (num_atoms, 3))
        self.assertEqual(batch.center_batch.shape, (num_atoms,))

    def test_dataloader_without_provider_preserves_geometry_and_labels(self) -> None:
        dataset = AtomicDataset(ASEReader(TRAJECTORY, index=slice(0, 2)))
        batch, labels = next(iter(create_dataloader(dataset, None, batch_size=2)))
        frames = read(TRAJECTORY, index=":2")
        counts = [len(frame) for frame in frames]
        num_atoms = sum(counts)

        self.assertIsNone(batch.edge_ptr)
        self.assertEqual(batch.center_batch.shape, (num_atoms,))
        self.assertEqual(batch.center_batch.dtype, torch.long)
        self.assertEqual(batch.atom_ptr.tolist(), [0, counts[0], num_atoms])
        self.assertEqual(batch.positions.shape, (num_atoms, 3))
        self.assertEqual(labels["energy"].shape, (2,))
        self.assertEqual(labels["forces"].shape, (num_atoms, 3))

    def test_collate_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            StructureCollator(ASENeighborBuilder(cutoff=3.0))([])


class EnergyShiftTest(unittest.TestCase):
    def test_least_squares_energy_shifts(self) -> None:
        structures = [
            labeled_structure([1, 1], -2.0),
            labeled_structure([8, 8], -4.0),
            labeled_structure([1, 1, 8, 8], -6.0),
        ]
        shifts = energy_shifts(structures, [1, 8])
        print("\n[data] Least-squares species energy baseline")
        print("H / O energy shifts:", shifts)
        self.assertEqual(len(shifts), 2)
        torch.testing.assert_close(
            torch.tensor(shifts, dtype=torch.float64),
            torch.tensor([-1.0, -2.0], dtype=torch.float64),
        )

    def test_energy_shift_validation(self) -> None:
        structure = labeled_structure([1], 0.0)
        structure.energy = None
        with self.assertRaisesRegex(ValueError, "energy label"):
            energy_shifts([structure], [1])
        with self.assertRaisesRegex(ValueError, "absent from type_map"):
            energy_shifts([labeled_structure([8], -2.0)], [1])


if __name__ == "__main__":
    unittest.main()
