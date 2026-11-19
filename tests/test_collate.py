"""Collator index offsets, label scopes, and batched mechanics tests."""

import unittest
from dataclasses import replace
import torch

from abfml.core import AtomicModel, ModelInput, ModelOutput, CSRNeighbors, AtomicSample
from abfml.data import StructureCollator


class PairField(AtomicModel):
    def compute(self, inputs):
        centers, ptr, _, vectors = inputs.require_neighbors()
        slots = torch.repeat_interleave(torch.arange(len(centers)), ptr[1:] - ptr[:-1])
        atomic = vectors.new_zeros(len(centers)).index_add(0, slots, .5*vectors.square().sum(-1))
        return ModelOutput(energy=atomic.sum(), atomic_energies=atomic)


class TotalEnergyOnlyField(PairField):
    def compute(self, inputs):
        return ModelOutput(energy=.5*inputs.edge_vectors.square().sum())


class CollatorTest(unittest.TestCase):
    def setUp(self):
        self.samples, self.graphs = [], []
        for count, distance in [(2, 1.), (3, 2.)]:
            positions = torch.zeros(count, 3, dtype=torch.float64)
            positions[1, 0] = distance
            self.samples.append(AtomicSample(
                atomic_numbers=[1]*count, positions=positions,
                cell=torch.eye(3, dtype=torch.float64)*10, pbc=[False]*3,
                energy=torch.tensor(-distance, dtype=torch.float64),
                forces=torch.zeros(count, 3, dtype=torch.float64),
                extra_labels={"groups": torch.arange(count)},
            ))
            self.graphs.append(CSRNeighbors(center_indices=torch.arange(count),
                edge_ptr=[0, 1, 2] if count == 2 else [0, 1, 2, 2], neighbor_indices=[1, 0],
                edge_vectors=torch.tensor([[distance, 0, 0], [-distance, 0, 0]], dtype=torch.float64)))
        self.collator = StructureCollator(label_scopes={"groups": "atom"})
        self.model = PairField(type_map=[1], cutoff=3.).eval()

    def test_indices_labels_and_conversion(self):
        inputs, labels = self.collator(self.samples, self.graphs)
        self.assertIsInstance(inputs, ModelInput)
        self.assertEqual(inputs.atom_ptr.tolist(), [0, 2, 5])
        self.assertEqual(inputs.center_batch.tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(inputs.edge_ptr.tolist(), [0, 1, 2, 3, 4, 4])
        self.assertEqual(inputs.neighbor_indices.tolist(), [1, 0, 3, 2])
        self.assertEqual(labels['energy'].tolist(), [-1., -2.])
        self.assertEqual(labels['forces'].shape, (5, 3))
        self.assertEqual(labels['groups'].tolist(), [0, 1, 0, 1, 2])
        converted = inputs.to(dtype=torch.float32)
        self.assertEqual(converted.positions.dtype, torch.float32)
        self.assertEqual(converted.edge_ptr.dtype, torch.long)
        self.assertEqual(converted.pbc.dtype, torch.bool)

    def test_batch_matches_separate_predictions(self):
        output = self.model(self.collator(self.samples, self.graphs)[0])
        singles = [self.model(self.collator([s], [g])[0]) for s, g in zip(self.samples, self.graphs)]
        for name in ('energy', 'atomic_energies', 'forces', 'virial'):
            torch.testing.assert_close(getattr(output, name), torch.cat([getattr(s, name) for s in singles]))
        torch.testing.assert_close(output.energy, torch.tensor([1., 4.], dtype=torch.float64))
        torch.testing.assert_close(output.virial[:,0,0], torch.tensor([-2., -8.], dtype=torch.float64))

    def test_missing_labels_and_scope(self):
        with self.assertRaisesRegex(ValueError, 'every structure'):
            self.collator([self.samples[0], replace(self.samples[1], energy=None)], self.graphs)
        with self.assertRaisesRegex(ValueError, 'explicit label scope'):
            StructureCollator()(self.samples, self.graphs)
        with self.assertRaisesRegex(ValueError, 'every structure'):
            self.collator([self.samples[0], replace(self.samples[1], extra_labels={})], self.graphs)
        with self.assertRaisesRegex(ValueError, 'one entry per atom'):
            self.collator([replace(self.samples[0], extra_labels={"groups": torch.ones(1)}),
                           self.samples[1]], self.graphs)

    def test_empty_graph_and_single_batch_shape(self):
        sample = AtomicSample([1], torch.zeros(1, 3, dtype=torch.float64), torch.eye(3,dtype=torch.float64), [False]*3)
        graph = CSRNeighbors([0], [0,0], [], torch.empty(0,3,dtype=torch.float64))
        output = self.model(StructureCollator()([sample], [graph])[0])
        torch.testing.assert_close(output.energy, torch.zeros(1,dtype=torch.float64))
        torch.testing.assert_close(output.forces, torch.zeros(1,3,dtype=torch.float64))
        torch.testing.assert_close(output.virial, torch.zeros(1,3,3,dtype=torch.float64))

    def test_ambiguous_batch_energy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Multi-structure'):
            TotalEnergyOnlyField()(self.collator(self.samples,self.graphs)[0])
