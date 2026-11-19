"""Label metadata drives scopes, normalization, units, and evaluation serialization."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from abfml.core.schema import LABELS, LabelSpec, label_unit, normalize_label
from abfml.core.structure import AtomicSample
from abfml.data import StructureCollator
from abfml.logger import format_progress
from abfml.workflows.evaluate import _frame_values


class LabelSchemaTest(unittest.TestCase):
    def test_units_and_display_are_separate(self):
        self.assertEqual(label_unit('forces'), 'eV/A')
        self.assertEqual(label_unit('forces', display=True), 'meV/A')
        self.assertEqual(label_unit('virial'), 'eV')
        self.assertEqual(label_unit('missing'), 'label_units')
        self.assertEqual(LABELS['atomic_energies'].scope, 'center')
        self.assertIn('F_RMSE: 100.00 meV/A', format_progress({'forces_rmse': .1}))

    def test_normalization_and_gradients(self):
        counts = torch.tensor([2., 3.], dtype=torch.float64)
        energy = torch.tensor([4., 12.], dtype=torch.float64, requires_grad=True)
        normalized = normalize_label('energy', energy, counts)
        torch.testing.assert_close(normalized, torch.tensor([2., 4.], dtype=torch.float64))
        torch.testing.assert_close(torch.autograd.grad(normalized.sum(), energy)[0], 1./counts)
        tensors = torch.arange(18., dtype=torch.float64).reshape(2, 3, 3)
        torch.testing.assert_close(normalize_label('virial', tensors, counts), tensors/counts[:, None, None])
        self.assertIs(normalize_label('forces', tensors, counts), tensors)
        self.assertIs(normalize_label('custom', tensors, counts), tensors)

    def test_custom_metadata_controls_shapes_and_scope(self):
        spec = LabelSpec('atom', (2,), 'custom_unit')
        with patch.dict(LABELS, {'custom': spec}):
            self.assertEqual(label_unit('custom'), 'custom_unit')
            sample = AtomicSample(torch.tensor([1, 1]), torch.zeros(2, 3), torch.eye(3), torch.zeros(3).bool(),
                                  extra_labels={'custom': torch.ones(2, 2)})
            _, labels = StructureCollator()([sample, sample])
            self.assertEqual(labels['custom'].shape, (4, 2))
            frame = _frame_values(labels, 1, 2, 4)
            np.testing.assert_array_equal(frame['custom'], np.ones((2, 2)))
            with self.assertRaisesRegex(ValueError, 'custom has a fixed scope'):
                StructureCollator(label_scopes={'custom': 'structure'})([sample])
            with self.assertRaisesRegex(ValueError, 'custom must have shape'):
                AtomicSample(torch.tensor([1, 1]), torch.zeros(2, 3), torch.eye(3), torch.zeros(3).bool(),
                             extra_labels={'custom': torch.ones(2)})


if __name__ == '__main__':
    unittest.main()
