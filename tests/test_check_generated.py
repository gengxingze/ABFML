"""Generated structures cover species, contain valid neighbors, and support derivative checks."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from abfml.workflows.check import _test_structure, check_model
from tests.test_workflows import build_for_test, atoms_for_test


class GeneratedCheckTest(unittest.TestCase):
    def test_geometry(self):
        model = SimpleNamespace(type_map=[1, 8], cutoff=3.)
        atoms = _test_structure(model)
        np.testing.assert_array_equal(atoms.positions, _test_structure(model).positions)
        self.assertEqual(set(atoms.numbers), {1, 8})
        self.assertTrue(atoms.pbc.all())
        distances = atoms.get_all_distances(mic=True)
        distances += np.eye(len(atoms)) * 1.e6
        self.assertGreater(distances.min(), 0.)
        self.assertLess(distances.min(), model.cutoff)
        for number in model.type_map:
            self.assertGreaterEqual(np.count_nonzero(atoms.numbers == number), 2)

    def test_missing_types(self):
        with self.assertRaisesRegex(ValueError, 'type_map'):
            _test_structure(SimpleNamespace(type_map=None, cutoff=3.))

    def test_generated_check(self):
        model = build_for_test({'name': 'lennard_jones', 'cutoff': 3.,
                                'epsilon': .2, 'sigma': 1.})
        with self.assertLogs('abfml.check', level='INFO') as logs:
            report = check_model(model)
        self.assertTrue(report['passed'], report)
        self.assertIn('stress_finite_difference', report['checks'])
        self.assertIn('automatically generated', '\n'.join(logs.output))
        self.assertIn('Total time:', '\n'.join(logs.output))

    def test_evaluation_logs(self):
        from ase.io import write
        from abfml.interfaces import ABFMLCalculator
        from abfml.workflows.evaluate import evaluate

        model = build_for_test({'name': 'lennard_jones', 'cutoff': 3.,
                                'epsilon': .2, 'sigma': 1.})
        atoms = atoms_for_test()
        atoms.calc = ABFMLCalculator(model)
        atoms.get_potential_energy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.extxyz'
            write(path, atoms)
            with self.assertLogs('abfml.evaluation', level='INFO') as logs:
                report = evaluate(model, path)
        self.assertLess(report['metrics']['energy']['rmse'], 1.e-10)
        text = '\n'.join(logs.output)
        for expected in ('Precision:', 'Dataset:', 'Evaluation results:', 'MAE', 'RMSE', 'Total time:'):
            self.assertIn(expected, text)


if __name__ == '__main__':
    unittest.main()
