"""End-to-end training, model-file, evaluation, physical-check, and CLI validation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write, read

import abfml.builtin  # Load builtin model registrations.
from abfml.core.registry import build_model, model_spec, parse_options
from abfml.workflows.check import check_model
from abfml.config import load_config
from abfml.workflows.evaluate import evaluate
from abfml.interfaces import ABFMLCalculator
from abfml.serialization import load_model, save_model
from abfml.workflows.train import run


def model_configs():
    fitting = {"hidden_layers": [4], "activation": "tanh", "bias": True, "residual": False}
    yield {"name": "lennard_jones", "cutoff": 3., "epsilon": .2, "sigma": 1.}
    yield {"name": "acsf", "descriptor": {"cutoff": 3., "max_neighbors": 20, "blocks": [
        {"type": "radial_gaussian", "basis": [{"eta": 1., "rs": 0.}, {"eta": 2., "rs": 1.}]},
        {"type": "angular_narrow", "basis": [{"eta": .2, "rs": 0., "lambda": 1., "zeta": 2}]},
        {"type": "angular_wide", "basis": [{"eta": .2, "rs": 0., "lambda": -1., "zeta": 1}]},
    ]}, "fitting": fitting}
    for name in ("deep_pot_se_e2_a", "deep_pot_se_e2_r", "deep_pot_se_e3"):
        descriptor = {"r_min": .5, "smooth": "poly2_r",
                      "embedding": {"network_size": [4, 4],
                                    "activation": "tanh", "bias": True, "residual": False}}
        if name == "deep_pot_se_e2_a":
            descriptor["m2"] = 2
        yield {"name": name, "cutoff": 3., "type_slots": [20], "descriptor": descriptor, "fitting": fitting}
    yield {"name": "nep", "cutoff_radial": 3., "cutoff_angular": 2.5,
           "n_max_radial": 2, "n_max_angular": 2, "basis_size_radial": 2,
           "basis_size_angular": 2, "l_max": [2, 0, 0, 0], "num_neurons": 4, "output_bias": 0.}


def build_for_test(config):
    kwargs = parse_options(config, type_map=(1,)).model_kwargs()
    if "energy_shift" in model_spec(config["name"]).builder.required_statistics:
        kwargs["energy_shift"] = [.13]
    return build_model(config["name"], **kwargs).double().eval()


def atoms_for_test():
    return Atoms('H3', positions=[[.2,.3,.1], [1.5,.6,.4], [.6,1.8,.7]],
                 cell=[[6.,0,0],[.4,6.2,0],[.1,.3,6.4]], pbc=True)


def training_fixture(root):
    config = {
        "config_version": 1, "common": {"elements": [1], "dtype": "float64", "seed": 73},
        "model": next(model_configs()),
        "data": {"files": ["data.extxyz"], "split": {"train": .5, "valid": .5, "seed": 5},
                 "loader": {"batch_size": 1, "shuffle": True, "neighbor_cache": "memory"}},
        "training": {"epochs": 2, "device": "cpu", "optimizer": {"name": "adam", "learning_rate": .001}},
        "loss": {"label": ["energy", "forces", "virial"], "start_weight": [1.,1.,1.], "limit_weight": [1.,1.,1.]},
        "output": {"dir": "run", "checkpoint": {"save_every": 1}},
    }
    teacher = build_for_test({**config['model'], 'epsilon': .3})
    frames = []
    for offset in [0., .1, .2, .3]:
        atoms = atoms_for_test()
        atoms.positions[1,0] += offset
        atoms.calc = ABFMLCalculator(teacher)
        energy, forces, stress = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        frames.append(atoms)
    write(root/'data.extxyz', frames)
    path = root/'input.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    return config, path


class WorkflowTest(unittest.TestCase):
    def test_stage_timing_is_exclusive(self):
        from abfml.workflows.train import _mark_stage

        names = ('Data & initialization', 'Training & validation', 'Final evaluation', 'Result output')
        for device in (torch.device('cpu'), torch.device('cuda:2')):
            with self.subTest(device=device):
                timing = {'started': 0., 'stage': names[0], 'device': device,
                          'durations': dict.fromkeys(names, 0.)}
                with patch('abfml.workflows.train.time.perf_counter', side_effect=[2., 7., 9., 12., 14.]), \
                        patch('abfml.workflows.train.torch.cuda.synchronize') as synchronize:
                    for stage in (names[1], names[3], names[2], names[3], None):
                        _mark_stage(timing, stage)
                self.assertEqual(list(timing['durations'].values()), [2., 5., 3., 4.])
                self.assertEqual(sum(timing['durations'].values()), 14.)
                self.assertEqual(synchronize.call_count, 5 if device.type == 'cuda' else 0)

    def test_best_model_summary_and_relative_error(self):
        from abfml.workflows.evaluate import evaluate_dataset
        from tests.test_losslog import metrics

        def staged_fit(model, *args, epoch_callback, **kwargs):
            for epoch, epsilon in ((1, .15), (2, .6)):
                with torch.no_grad():
                    model.epsilon.fill_(epsilon)
                epoch_callback(epoch, {**metrics(epoch), 'valid_loss': float(epoch)}, epoch)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, path = training_fixture(root)
            with patch('abfml.workflows.train.fit', side_effect=staged_fit), \
                    patch('abfml.workflows.evaluate.evaluate_dataset', wraps=evaluate_dataset) as evaluated, \
                    self.assertLogs('abfml.workflow', level='INFO') as captured:
                final = run(path)
            self.assertAlmostEqual(final.epsilon.item(), .6)
            self.assertAlmostEqual(load_model(root/'run/model.pt').epsilon.item(), .6)
            for call in evaluated.call_args_list:
                self.assertAlmostEqual(call.args[0].epsilon.item(), .15)
            text = '\n'.join(captured.output)
            self.assertIn('Best model: best.pt', text)
            self.assertIn('Train RMSE', text)
            self.assertIn('Valid Rel. (%)', text)
            self.assertIn('RMSE / mean(abs(reference))', text)
            self.assertIn('Runtime breakdown:', text)
            for name in ('Data & initialization', 'Training & validation', 'Final evaluation', 'Result output'):
                self.assertIn(name, text)
            self.assertIn('Share (%)', text)
            self.assertIn('Total runtime:', text)
            one = evaluate(root/'run/best.pt', root/'data.extxyz', batch_size=1)
            many = evaluate(root/'run/best.pt', root/'data.extxyz', batch_size=3)
            from abfml.data import ASEReader
            samples = list(ASEReader(root/'data.extxyz'))
            for name in ('energy_per_atom', 'forces', 'virial_per_atom'):
                base = name.removesuffix('_per_atom')
                targets = np.concatenate([(sample.get_label(base).numpy() / sample.num_atoms
                                           if name.endswith('_per_atom') else sample.get_label(base).numpy()).ravel()
                                          for sample in samples])
                expected = 100. * one['metrics'][name]['rmse'] / np.abs(targets).mean()
                self.assertAlmostEqual(one['metrics'][name]['mae'], .5*np.abs(targets).mean(), places=6)
                self.assertAlmostEqual(one['metrics'][name]['relative_error_percent'], expected, places=10)
                self.assertAlmostEqual(one['metrics'][name]['relative_error_percent'],
                                       many['metrics'][name]['relative_error_percent'], places=10)

    def test_zero_reference_relative_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atoms = atoms_for_test()
            atoms.calc = SinglePointCalculator(atoms, energy=0., forces=np.zeros((len(atoms), 3)))
            write(root/'zero.extxyz', atoms)
            report = evaluate(build_for_test(next(model_configs())), root/'zero.extxyz')
            for name in ('energy', 'energy_per_atom', 'forces'):
                self.assertIsNone(report['metrics'][name]['relative_error_percent'])
                self.assertGreater(report['metrics'][name]['rmse'], 0.)

    def test_automatic_plots_use_actual_splits(self):
        from abfml.data import ASEReader
        from abfml.workflows.evaluate import evaluate_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, path = training_fixture(root)
            # Include incomplete training tail batches in final parity evaluation.
            config['data']['split']['train'] = .75
            config['data']['split']['valid'] = .25
            config['data']['loader'].update(batch_size=2, drop_last=True)
            path.write_text(json.dumps(config), encoding='utf-8')
            with patch('abfml.workflows.evaluate.evaluate_dataset', wraps=evaluate_dataset) as evaluated:
                run(path, epochs=1)
            self.assertTrue((root/'run/loss.png').is_file())
            self.assertFalse((root/'run/loss.csv').exists())
            split = json.loads((root/'run/data.json').read_text())['indices']
            samples = list(ASEReader(root/'data.extxyz'))
            for name, call in zip(('train', 'valid'), evaluated.call_args_list):
                pairs = call.kwargs['pairs']['energy']
                self.assertEqual(len(pairs), len(split[name]))
                for pair, index in zip(pairs, split[name]):
                    self.assertAlmostEqual(pair[0].item(), samples[index].energy.item())
                self.assertNotIn('output_dir', call.kwargs)
                self.assertFalse((root/'run'/name).exists())
            self.assertTrue((root/'run/parity.png').is_file())
            self.assertFalse(list((root/'run').glob('*.extxyz')))
            self.assertFalse(list((root/'run').glob('*metrics*')))

    def test_large_training_skips_prediction_plots(self):
        from abfml.workflows.evaluate import evaluate_dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, path = training_fixture(root)
            with patch('abfml.workflows.train.PARITY_LIMIT', 4), \
                    patch('abfml.workflows.evaluate.evaluate_dataset', wraps=evaluate_dataset) as evaluate_mock:
                run(path, epochs=1)
            self.assertEqual(evaluate_mock.call_count, 2)
            self.assertTrue(all(call.kwargs['pairs'] is None for call in evaluate_mock.call_args_list))
            self.assertTrue((root/'run/loss.png').is_file())
            self.assertFalse((root/'run/train').exists())
            self.assertFalse((root/'run/parity.png').exists())

    def test_model_roundtrip_and_physical_checks_all_families(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            for config in model_configs():
                with self.subTest(model=config['name']):
                    model = build_for_test(config)
                    save_model(path, model, {"common": {"elements": [1], "dtype": "float64"}, "model": config})
                    loaded = load_model(path)
                    first, second = atoms_for_test(), atoms_for_test()
                    first.calc, second.calc = ABFMLCalculator(model), ABFMLCalculator(loaded)
                    np.testing.assert_allclose(first.get_potential_energy(), second.get_potential_energy(), rtol=1.e-10)
                    np.testing.assert_allclose(first.get_forces(), second.get_forces(), atol=1.e-10)
                    np.testing.assert_allclose(first.get_stress(), second.get_stress(), atol=1.e-10)
                    report = check_model(loaded, atoms_for_test(), atol=2.e-5)
                    self.assertTrue(report['passed'], report)

    def test_resume_matches_uninterrupted_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, path = training_fixture(root)
            full = run(path)
            config['output']['dir'] = 'split'
            path.write_text(json.dumps(config), encoding='utf-8')
            run(path, epochs=1)
            resumed = run(path, resume=root/'split/last.pt', epochs=1)
            for name in full.state_dict():
                torch.testing.assert_close(full.state_dict()[name], resumed.state_dict()[name], rtol=0., atol=0.)
            before = root/'run/model.pt'
            a = evaluate(before, root/'data.extxyz', batch_size=1, output_dir=root/'eval')
            b = evaluate(before, root/'data.extxyz', batch_size=3)
            for name in a['metrics']:
                self.assertAlmostEqual(a['metrics'][name]['rmse'], b['metrics'][name]['rmse'], places=12)
                self.assertAlmostEqual(a['metrics'][name]['mae'], b['metrics'][name]['mae'], places=12)
            self.assertEqual(a['metrics']['forces']['count'], 36)
            frames = read(root/'eval/predictions.extxyz', ':')
            self.assertEqual(len(frames), 4)
            self.assertIn('pred_forces', frames[0].arrays)
            self.assertIn('ref_forces', frames[0].arrays)
            self.assertEqual({file.name for file in (root/'eval').iterdir()}, {'predictions.extxyz', 'parity.png'})
            loss_out = root/'run/loss.out'
            self.assertTrue(loss_out.exists())
            loss_text = loss_out.read_text(encoding="utf-8")
            self.assertIn("T-energy", loss_text)
            loss_lines = loss_text.splitlines()
            loss_columns = next(
                line for line in loss_lines if line.startswith("# columns:")
            ).split()[2:]
            loss_data = [line.split() for line in loss_lines if line and not line.startswith("#")]
            self.assertNotEqual(loss_data[0][loss_columns.index("T-energy")], "NA")
            # Inference files remain loadable after moving them and deleting training data.
            (root/'data.extxyz').unlink()
            moved = root/'moved.pt'
            before.replace(moved)
            atoms = atoms_for_test()
            atoms.calc = ABFMLCalculator(moved)
            self.assertTrue(np.isfinite(atoms.get_forces()).all())

    def test_training_all_model_families_and_resume_skips_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, path = training_fixture(root)
            for model_config in list(model_configs())[1:]:
                with self.subTest(model=model_config['name']):
                    config['model'] = model_config
                    config['output']['dir'] = model_config['name']
                    path.write_text(json.dumps(config), encoding='utf-8')
                    model = run(path, epochs=1)
                    loaded = load_model(root/model_config['name']/'model.pt')
                    before, after = atoms_for_test(), atoms_for_test()
                    before.calc, after.calc = ABFMLCalculator(model), ABFMLCalculator(loaded)
                    np.testing.assert_allclose(before.get_potential_energy(), after.get_potential_energy(), atol=1.e-12)
                    np.testing.assert_allclose(before.get_forces(), after.get_forces(), atol=1.e-12)
                    saved = root/model_config['name']/'last.pt'
                    method = {'acsf': 'initialize_statistics', 'nep': 'calibrate_scaler'}.get(
                        model_config['name'], 'measure_normalization')
                    with patch.object(type(model), method, side_effect=AssertionError('Repeated statistics')):
                        run(path, resume=saved, epochs=1)

    def test_fine_tune_restarts_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, path = training_fixture(root)
            run(path, epochs=2)
            source = root/'run/model.pt'
            config['output']['dir'] = 'fine'
            config['training']['optimizer']['learning_rate'] = .0001
            path.write_text(json.dumps(config), encoding='utf-8')
            run(path, fine_tune=source, epochs=1)
            saved = torch.load(root/'fine/last.pt', weights_only=True)
            self.assertEqual(saved['epoch'], 1)
            self.assertEqual(saved['step'], 2)
            self.assertEqual(saved['optimizer_state']['param_groups'][0]['lr'], .0001)

    def test_unlabeled_evaluation_and_strict_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'unlabeled.extxyz'
            write(path, [atoms_for_test()])
            model = build_for_test(next(model_configs()))
            report = evaluate(model, path, output_dir=root/'eval')
            self.assertEqual(report['metrics'], {})
            frame = read(root/'eval/predictions.extxyz')
            self.assertIn('pred_forces', frame.arrays)
            self.assertFalse(any(key.startswith('ref_') for key in frame.info['abfml_shapes']))
            self.assertFalse((root/'eval/parity.png').exists())
            with self.assertRaisesRegex(ValueError, 'lacks prediction or reference'):
                evaluate(model, path, labels=['energy'])

    def test_resume_rejects_modified_data_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, path = training_fixture(root)
            run(path, epochs=1)
            checkpoint = root/'run/last.pt'
            config['model']['cutoff'] = 4.
            path.write_text(json.dumps(config), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                run(path, resume=checkpoint)
            config['model']['cutoff'] = 3.
            path.write_text(json.dumps(config), encoding='utf-8')
            with (root/'data.extxyz').open('a') as handle:
                handle.write('\n')
            with self.assertRaisesRegex(ValueError, 'dataset content'):
                run(path, resume=checkpoint)

    def test_ase_invalidates_results_and_rejects_unknown_species(self):
        atoms = atoms_for_test()
        model = build_for_test(next(model_configs()))
        atoms.calc = ABFMLCalculator(model)
        before = atoms.get_potential_energy()
        atoms.positions[1,0] += .1
        self.assertNotEqual(before, atoms.get_potential_energy())
        atoms.numbers[0] = 8
        with self.assertRaisesRegex(ValueError, 'type_map'):
            atoms.get_forces()

    def test_cli_train_evaluate_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, path = training_fixture(root)
            commands = [
                ['train', str(path), '--epochs', '1'],
                ['evaluate', str(root/'run/model.pt'), str(root/'data.extxyz'), '--output', str(root/'eval')],
                ['check', str(root/'run/model.pt'), str(root/'data.extxyz')],
            ]
            for arguments in commands:
                result = subprocess.run([sys.executable, '-m', 'abfml', *arguments], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            self.assertIn('PASS', result.stderr.decode(errors='replace'))


if __name__ == '__main__':
    unittest.main()
