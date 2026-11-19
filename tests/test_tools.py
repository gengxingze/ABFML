"""Verify Python and CLI analysis entry points generate figures and numerical outputs."""

import csv
import io
import json
from contextlib import chdir, redirect_stdout, redirect_stderr
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from abfml.serialization import save_model
from abfml.tools import (
    plot_descriptors, plot_labels, plot_loss, plot_neighbors,
    plot_predictions,
    read_pairs, export_pairs,
)
from tests.test_workflows import atoms_for_test, build_for_test, model_configs


class AnalysisTest(unittest.TestCase):
    def test_structure_descriptor_pooling(self):
        from abfml.cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = atoms_for_test()
            second = first[:2]
            second.positions[1, 0] += .2
            data = root/'data.extxyz'
            write(data, [first, second])
            config = list(model_configs())[1]
            model = root/'model.pt'
            save_model(model, build_for_test(config), {'common': {'elements': [1]}, 'model': config})
            atom_rows = plot_descriptors(model, {'test': data}, output=root/'atoms', export=True)
            with np.load(root/'atoms/descriptors.npz') as values:
                atomic = values['descriptors'].copy()
                self.assertEqual(values['level'].item(), 'atom')
            self.assertEqual(atom_rows[0]['points'], 5)
            expected = np.stack((atomic[:3].mean(axis=0), atomic[3:].mean(axis=0)))
            for batch_size in (1, 2):
                out = root/f'structures_{batch_size}'
                rows = plot_descriptors(model, {'test': data}, output=out, batch_size=batch_size,
                                        level='structure', export=True)
                self.assertEqual(rows[0]['points'], 2)
                self.assertEqual(rows[0]['centers'], 5)
                with np.load(out/'descriptors.npz') as values:
                    np.testing.assert_allclose(values['descriptors'], expected, atol=1.e-12)
                    self.assertEqual(values['pooling'].item(), 'mean')
                with (out/'pca.csv').open() as handle:
                    metadata = list(csv.DictReader(handle))
                self.assertEqual([int(row['centers']) for row in metadata], [3, 2])
                self.assertTrue(all(row['level'] == 'structure' for row in metadata))
            with redirect_stdout(io.StringIO()) as console:
                main(['plot', 'descriptors', '--model', str(model), '--dataset', f'test={data}',
                      '--level', 'structure', '--output', str(root/'cli')])
            self.assertIn('structure', console.getvalue())
            one = root/'one.extxyz'
            write(one, first)
            with self.assertRaisesRegex(ValueError, 'at least two finite structure-level'):
                plot_descriptors(model, {'test': one}, output=root/'one', level='structure')

    def test_console_supports_gbk(self):
        from abfml.tools.common import print_table

        buffer = io.BytesIO()
        terminal = io.TextIOWrapper(buffer, encoding='gbk')
        with redirect_stdout(terminal):
            print_table([{'unit': 'eV/Å'}], {'Cutoff (Å)': 'unit'})
        terminal.flush()
        self.assertIn('eV/A', buffer.getvalue().decode('gbk'))

    def test_default_outputs_and_console(self):
        from abfml.cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atoms = atoms_for_test()
            atoms.calc = SinglePointCalculator(atoms, energy=1.)
            data = root/'data.extxyz'
            write(data, atoms)
            config = list(model_configs())[1]
            model = root/'model.pt'
            save_model(model, build_for_test(config), {'common': {'elements': [1]}, 'model': config})
            for kind, arguments, filename in (
                ('labels', [str(data)], 'labels.png'),
                ('neighbors', [str(data), '--cutoff', '2', '3'], 'neighbors.png'),
                ('descriptors', ['--model', str(model), '--dataset', f'test={data}'], 'descriptors.png'),
            ):
                folder = root/kind
                folder.mkdir()
                console = io.StringIO()
                with chdir(folder), redirect_stdout(console):
                    main(['plot', kind, *arguments])
                self.assertEqual({path.name for path in folder.iterdir()}, {filename})
                self.assertIn('Figure:', console.getvalue())
                if kind == 'neighbors':
                    self.assertIn('Min', console.getvalue())
                    self.assertIn('Max', console.getvalue())
                with chdir(folder), redirect_stdout(io.StringIO()):
                    main(['plot', kind, *arguments, '--output'])
                self.assertTrue(list(folder.glob('*.csv')))

    def test_label_input_errors_identify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root/'broken.extxyz'
            invalid.write_text('\nProperties=species:S:1:pos:R:3 energy=1.0\nH 0 0 0\n')
            with self.assertRaisesRegex(ValueError, "Dataset 'broken': No structures read.*broken.extxyz"):
                plot_labels({'broken': invalid}, output=root/'output')
            unlabeled = root/'unlabeled.extxyz'
            write(unlabeled, atoms_for_test())
            with self.assertRaisesRegex(ValueError, "Dataset 'unlabeled': No supported labels.*unlabeled.extxyz"):
                plot_labels({'unlabeled': unlabeled}, output=root/'output')

    def test_unified_label_inputs(self):
        from abfml.cli import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atoms = atoms_for_test()
            atoms.calc = SinglePointCalculator(atoms, energy=1.)
            data = root/'data.extxyz'
            write(data, atoms)
            plot_labels([data, data], output=root/'merged', export=True)
            with (root/'merged/labels_summary.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row['dataset'] for row in rows}, {'dataset'})
            self.assertEqual(next(row['count'] for row in rows if row['label'] == 'energy'), '2')
            main(['plot', 'labels', '--dataset', f'Train={data}', '--dataset', f'Valid={data}',
                  '--dataset', f'Valid={data}', '--output', str(root/'grouped')])
            with (root/'grouped/labels_summary.csv').open() as handle:
                counts = {row['dataset']: row['count'] for row in csv.DictReader(handle)
                          if row['label'] == 'energy'}
            self.assertEqual(counts, {'Train': '1', 'Valid': '2'})
            main(['plot', 'labels', str(data), '--output', str(root/'single')])
            self.assertTrue((root/'single/labels.png').is_file())
            for arguments in (['labels', '--output', str(root/'invalid')],
                              ['labels', str(data), '--dataset', f'Train={data}',
                               '--output', str(root/'invalid')]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(['plot', *arguments])
                self.assertEqual(error.exception.code, 2)
            self.assertFalse((root/'invalid').exists())

    def test_train_valid_panels(self):
        from abfml.tools.common import plt

        with tempfile.TemporaryDirectory() as directory:
            groups = {'Train': {'forces': [(np.array([0., 1.]), np.array([.1, .9]))]},
                      'Valid': {'forces': [(np.array([0., 2.]), np.array([.2, 1.8]))]}}
            with patch.object(plt, 'close') as closed:
                plot_predictions(groups, output=directory)
            figure = closed.call_args.args[0]
            try:
                self.assertEqual(len(figure.axes), 1)
                self.assertEqual(len(figure.axes[0].collections), 2)
                self.assertEqual([text.get_text() for text in figure.legends[0].texts],
                                 ['Train', 'Valid'])
                self.assertEqual({path.name for path in Path(directory).iterdir()}, {'parity.png'})
            finally:
                plt.close(figure)

    def test_extxyz_precision_and_shapes(self):
        from abfml.workflows.evaluate import _write_frame
        from abfml.data.reader import from_ase

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'predictions.extxyz'
            sample = from_ase(atoms_for_test())
            force = np.arange(9, dtype=float).reshape(3, 3) / 7.123456789012345
            tensor = force + np.array([[0., .1, .2], [.3, .4, .5], [.6, .7, .8]])
            reference = {'forces': force, 'virial': tensor, 'energy': np.array(1.234567890123456)}
            prediction = {name: value + .123456789012345 for name, value in reference.items()}
            with path.open('w', encoding='utf-8') as handle:
                _write_frame(handle, sample, reference, prediction)
            values = read_pairs(path)
            for name in reference:
                np.testing.assert_allclose(values[name][0][0], reference[name].ravel(), rtol=1.e-14, atol=1.e-15)
                np.testing.assert_allclose(values[name][0][1], prediction[name].ravel(), rtol=1.e-14, atol=1.e-15)
            export_pairs(path, label='virial_per_atom', output=Path(directory)/'virial.csv')
            pairs = np.loadtxt(Path(directory)/'virial.csv', delimiter=',', skiprows=1)
            np.testing.assert_allclose(pairs[:, 0], tensor.ravel()/3, rtol=1.e-14)

    def test_parity_axes(self):
        from abfml.tools import parity

        saved = parity.save_figure
        checked = []

        def inspect_figure(figure, path, **kwargs):
            if path.name == "parity.png":
                axis = figure.axes[0]
                np.testing.assert_allclose(axis.get_xlim(), axis.get_ylim())
                self.assertGreater(axis.get_xlim()[1], axis.get_xlim()[0])
                self.assertEqual(axis.get_aspect(), 1.)
                self.assertTrue(all(spine.get_visible() for spine in axis.spines.values()))
                self.assertIn("RMSE", axis.texts[0].get_text())
                self.assertEqual(kwargs["dpi"], 300)
                checked.append(path.name)
            saved(figure, path, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "predictions.jsonl"
            source.write_text(json.dumps({"num_atoms": 1, "references": {"energy": 0.},
                                           "predictions": {"energy": 0.}}) + "\n")
            with patch.object(parity, "save_figure", side_effect=inspect_figure):
                plot_predictions(source, output=root / "plots")
            self.assertEqual(len(checked), 1)

    def test_grouped_curves(self):
        from abfml.tools.common import plt

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loss = root / "loss.out"
            loss.write_text("# columns: Epoch loss V-loss lr T-energy V-energy T-charges V-charges time\n"
                            "# units: - - - - meV/atom meV/atom e e s\n"
                            "1 2 3 .001 10 12 .2 NA 1\n2 1 2 .0009 8 9 .1 .15 2\n")
            with patch.object(plt, "close") as closed:
                plot_loss(loss, output=root / "plots", log=True)
            figure = closed.call_args.args[0]
            try:
                self.assertEqual(len(figure.axes), 4)
                energy = figure.axes[1]
                self.assertEqual([line.get_label() for line in energy.lines], ["Train", "Valid"])
                np.testing.assert_array_equal(energy.lines[1].get_ydata(), [12, 9])
                self.assertTrue(np.isnan(figure.axes[2].lines[1].get_ydata()[0]))
                self.assertFalse(any("Epoch time" in axis.get_title() for axis in figure.axes))
                self.assertTrue(all(spine.get_visible() for axis in figure.axes
                                    for spine in axis.spines.values()))
                self.assertFalse((root / "plots/loss.csv").exists())
                self.assertEqual({path.name for path in (root / "plots").iterdir()}, {"loss.png"})
            finally:
                plt.close(figure)

    def test_loss_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loss = root / "loss.out"
            loss.write_text("# columns: Epoch loss V-loss\n# units: - - -\n1 2 NA\n2 1 1.5\n")
            plot_loss(loss, output=root / "loss", log=True)
            self.assertTrue((root / "loss/loss.png").is_file())
            self.assertFalse((root / "loss/loss.csv").exists())
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"num_atoms": 2,
                "references": {"energy": 2.0, "forces": [[0., 0., 0.], [0., 0., 0.]]},
                "predictions": {"energy": 2.2, "forces": [[.1, 0., 0.], [-.1, 0., 0.]]}}) + "\n")
            plot_predictions(predictions, output=root / "parity")
            pairs = read_pairs(predictions)['energy_per_atom'][0]
            self.assertAlmostEqual(float(np.abs(pairs[1]-pairs[0]).mean()), .1)
            self.assertEqual({file.name for file in (root/'parity').iterdir()}, {'parity.png'})
            export_pairs(predictions, label='forces', output=root/'forces.csv')
            values = np.loadtxt(root/'forces.csv', delimiter=',', skiprows=1)
            self.assertEqual(values.shape, (6, 2))
            np.testing.assert_allclose(values[:, 1], [.1, 0, 0, -.1, 0, 0])

    def test_datasets_and_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atoms = atoms_for_test()
            atoms.calc = SinglePointCalculator(atoms, energy=1., forces=np.zeros((3, 3)))
            data = root / "data.extxyz"
            write(data, atoms)
            plot_labels(data, output=root / "labels")
            plot_labels({"train": data, "valid": [data]}, output=root / "splits", export=True)
            plot_neighbors(data, cutoffs=[3.], output=root / "neighbors", slots=1, export=True)
            self.assertTrue((root / "labels/labels.png").is_file())
            self.assertTrue((root / "splits/labels_summary.csv").is_file())
            with (root/'splits/labels_summary.csv').open() as handle:
                self.assertEqual({row['dataset'] for row in csv.DictReader(handle)}, {'train', 'valid'})
            with (root / "neighbors/neighbors_summary.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["overflow_fraction"]), 1.)
            config = list(model_configs())[1]
            model = build_for_test(config)
            path = root / "model.pt"
            save_model(path, model, {"common": {"elements": [1]}, "model": config})
            plot_descriptors(path, {"train": data}, output=root / "descriptors", batch_size=1, export=True)
            with np.load(root / "descriptors/descriptors.npz") as values:
                self.assertEqual(values["descriptors"].shape[0], 3)
            self.assertTrue((root / "descriptors/descriptors.png").is_file())

    def test_cli_entries(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; from abfml.cli import build_parser; build_parser(); "
            "assert 'matplotlib' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for arguments, expected in ((["abfml", "plot", "--help"], "descriptors"),
                                    (["abfml", "--help"], "plot")):
            result = subprocess.run([sys.executable, "-m", *arguments], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.jsonl"
            path.write_text(json.dumps({"num_atoms": 1, "references": {"energy": 1.},
                                        "predictions": {"energy": 1.1}}) + "\n")
            result = subprocess.run([sys.executable, "-m", "abfml", "plot", "predictions", str(path),
                                     "--output", str(root / "output")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "output/parity.png").is_file())
            result = subprocess.run([sys.executable, '-m', 'abfml', 'plot', 'pairs', str(path),
                                     '--label', 'energy_per_atom', '--output', str(root/'pairs.csv')],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root/'pairs.csv').read_text().splitlines()[0], 'reference,prediction')


if __name__ == "__main__":
    unittest.main()
