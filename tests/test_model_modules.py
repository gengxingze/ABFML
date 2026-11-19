"""Validate core dependencies and external models from configuration to inference in isolated
processes.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from abfml.config import load_modules
from abfml.config import ConfigError


ROOT = Path(__file__).resolve().parents[1]


class ModelModulesTest(unittest.TestCase):
    def run_python(self, arguments, *, cwd=ROOT, extra_path=None):
        env = os.environ.copy()
        paths = [str(ROOT)]
        if extra_path is not None:
            paths.insert(0, str(extra_path))
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        result = subprocess.run([sys.executable, *arguments], cwd=cwd, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_core_import_has_no_builtin_or_ase_dependency(self):
        self.run_python(["-c", textwrap.dedent('''
            import sys
            from abfml.core.registry import MODEL_REGISTRY
            assert not MODEL_REGISTRY.names
            assert not any(n == 'ase' or n.startswith('ase.') for n in sys.modules)
            assert 'abfml.builtin' not in sys.modules
            import abfml.builtin
            assert set(MODEL_REGISTRY.names) == {
                'acsf', 'deep_pot_se_e2_a', 'deep_pot_se_e2_r', 'deep_pot_se_e3', 'nep', 'lennard_jones'}
            import abfml.builtin
            assert len(MODEL_REGISTRY) == 6
        ''')])

    def test_config_parses_network_without_import_cycle(self):
        self.run_python(['-c', textwrap.dedent('''
            import sys
            from abfml.config import parse_network
            assert 'abfml.builtin' not in sys.modules
            assert parse_network({'hidden_layers': [4], 'activation': 'tanh'}, path='network')[1] == 'tanh'
            from abfml.builtin.networks import ACTIVATIONS
            from abfml.builtin.functions import SMOOTH_FUNCTIONS
            assert len(ACTIVATIONS) == 18
            assert len(SMOOTH_FUNCTIONS) == 11
        ''')])

    def test_invalid_and_missing_module_names_are_explained(self):
        for value in ("harmonic", ["../custom.py"], ["a..b"], [""], [3], ["a", "a"]):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "model_modules"):
                load_modules(value)
        with self.assertRaisesRegex(ConfigError, "model_modules\\[0\\].*cannot import"):
            load_modules(["abfml_missing_test_plugin"])

    def test_external_model_train_reload_evaluate_ase_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "harmonic.py").write_text(
                (ROOT / "examples/harmonic/harmonic.py").read_text(encoding="utf-8"), encoding="utf-8")
            self.run_python(["-c", textwrap.dedent('''
                import json
                from pathlib import Path
                from tests.test_workflows import training_fixture
                from abfml.config import load_config
                from abfml.workflows.train import run
                root = Path.cwd()
                raw, path = training_fixture(root)
                raw['model_modules'] = ['harmonic']
                raw['model'] = {'name': 'harmonic_pair', 'cutoff': 3.,
                                'stiffness': .2, 'equilibrium': 1.}
                path.write_text(json.dumps(raw), encoding='utf-8')
                assert load_config(path).model_modules == ('harmonic',)
                assert load_config(path).model_name == 'harmonic_pair'
                run(path, epochs=1)
            ''')], cwd=root, extra_path=root)
            self.run_python(["-m", "abfml", "evaluate", "run/model.pt", "data.extxyz",
                             "--output", "evaluation"], cwd=root, extra_path=root)
            self.assertTrue((root / "evaluation/predictions.extxyz").is_file())
            self.assertTrue((root / "evaluation/parity.png").is_file())
            self.run_python(["-c", textwrap.dedent('''
                import sys
                import torch
                from ase.io import read
                from abfml.serialization import load_model
                from abfml.interfaces import ABFMLCalculator
                from abfml.workflows.check import check_model
                from abfml.workflows.train import run
                assert 'harmonic' not in sys.modules
                payload = torch.load('run/model.pt', weights_only=True)
                assert payload['config']['model_modules'] == ['harmonic']
                model = load_model('run/model.pt')
                assert model.__class__.__module__ == 'harmonic'
                atoms = read('data.extxyz')
                atoms.calc = ABFMLCalculator('run/model.pt')
                assert atoms.get_forces().shape == (3, 3)
                assert check_model(model, atoms)['passed']
                run('input.json', resume='run/last.pt', epochs=1)
                checkpoint = torch.load('run/last.pt', weights_only=True)
                assert checkpoint['epoch'] == 2
            ''')], cwd=root, extra_path=root)


if __name__ == "__main__":
    unittest.main()
