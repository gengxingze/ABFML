"""DataLoader factory tests."""

import unittest

import torch
from torch.utils.data import DataLoader

from abfml.core.types import ModelInput
from abfml.core.neighbor import ASENeighborBuilder
from abfml.core.structure import AtomicSample
from abfml.data.dataset import AtomicDataset
from abfml.data.loader import create_dataloader


def make_dimer(distance: float) -> AtomicSample:
    return AtomicSample(
        atomic_numbers=torch.tensor([18, 18]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        cell=torch.eye(3) * 20.0,
        pbc=torch.zeros(3, dtype=torch.bool),
    )


class CreateDataLoaderTest(unittest.TestCase):
    def test_yields_structure_batches_with_requested_size(self) -> None:
        dataset = AtomicDataset(
            [make_dimer(2.0), make_dimer(2.5), make_dimer(3.0)]
        )
        loader = create_dataloader(
            dataset, ASENeighborBuilder(5.0), batch_size=2
        )
        self.assertIsInstance(loader, DataLoader)
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertIsInstance(batches[0][0], ModelInput)
        self.assertEqual(batches[0][0].num_structures, 2)
        self.assertEqual(batches[1][0].num_structures, 1)

    def test_drop_last_discards_partial_batch(self) -> None:
        dataset = AtomicDataset(
            [make_dimer(2.0), make_dimer(2.5), make_dimer(3.0)]
        )
        loader = create_dataloader(
            dataset,
            ASENeighborBuilder(5.0),
            batch_size=2,
            drop_last=True,
        )
        self.assertEqual(len(list(loader)), 1)

    def test_invalid_arguments_are_rejected(self) -> None:
        dataset = AtomicDataset([make_dimer(2.0)])
        with self.assertRaisesRegex(ValueError, "batch_size"):
            create_dataloader(dataset, ASENeighborBuilder(5.0), batch_size=0)
        with self.assertRaisesRegex(ValueError, "num_workers"):
            create_dataloader(dataset, ASENeighborBuilder(5.0), num_workers=-1)


if __name__ == "__main__":
    unittest.main()
