"""Test data preparation, cache lifetime, and model-declared neighbor policies."""

from abfml.data.reader import from_ase

import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from abfml.config import LoaderConfig
from abfml.config import ConfigError
from abfml.core.neighbor import ASENeighborBuilder
from abfml.core.structure import AtomicSample
from abfml.core.model import AtomicModel
from abfml.core.types import ModelOutput
from abfml.data import ASEReader, AtomicDataset, CachedNeighborBuilder, StructureCollator
from abfml.data.loader import create_dataloader


def sample(distance=1.0):
    return from_ase(Atoms('H2', positions=[[0, 0, 0], [distance, 0, 0]]))


class CountingBuilder(ASENeighborBuilder):
    def __init__(self):
        super().__init__(3.0)
        self.calls = 0

    def build(self, structure):
        self.calls += 1
        return super().build(structure)


class ParentOnlyBuilder(CountingBuilder):
    def build(self, structure):
        if torch.utils.data.get_worker_info() is not None:
            raise AssertionError('Worker rebuilt a preloaded graph')
        return super().build(structure)


class InternalModel:
    neighbor_mode = 'internal'


class NoNeighborModel:
    neighbor_mode = 'none'


class ExternalModel:
    neighbor_mode = 'external'
    cutoff = 3.0


class InternalEnergyModel(AtomicModel):
    neighbor_mode = 'internal'

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def compute(self, inputs):
        atomic = self.scale * inputs.positions.square().sum(dim=1)
        return ModelOutput(energy=atomic.sum(), atomic_energies=atomic)


class DataPipelineTest(unittest.TestCase):
    def test_reader_selection_labels_and_reiteration(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'{i}.extxyz' for i in range(2)]
            frames = [Atoms('H2', positions=[[0, 0, 0], [r, 0, 0]]) for r in (1, 2)]
            for i, atoms in enumerate(frames):
                atoms.calc = SinglePointCalculator(atoms, energy=float(i), forces=torch.zeros(2, 3).numpy())
            for path in paths:
                write(path, frames)
            reader = ASEReader(paths, index=-1, dtype=torch.float32)
            dataset = AtomicDataset(reader)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0].dtype, torch.float32)
            self.assertEqual(float(dataset[0].energy), 1.0)
            self.assertEqual(len(list(reader)), 2)
            self.assertEqual(len(list(ASEReader(paths[0], index=slice(0, 1)))), 1)
            for path in paths:
                path.unlink()
            self.assertEqual(dataset.subset([1, 0, 1])[2].num_atoms, 2)

    def test_cache_reused_across_epochs_and_batch_mutation(self):
        dataset = AtomicDataset([sample(1), sample(2)])
        provider = CountingBuilder()
        loader = create_dataloader(dataset, provider, neighbor_cache='memory', batch_size=2)
        self.assertEqual(provider.calls, 2)
        first = next(iter(loader))
        expected = first[0].edge_vectors.clone()
        first[0].edge_vectors.fill_(100)
        for _ in range(3):
            batch = next(iter(loader))
            torch.testing.assert_close(batch[0].edge_vectors, expected)
        self.assertEqual(provider.calls, 2)

    def test_uncached_builds_each_epoch(self):
        provider = CountingBuilder()
        loader = create_dataloader(AtomicDataset([sample()]), provider)
        list(loader)
        list(loader)
        self.assertEqual(provider.calls, 2)

    def test_cache_tracks_geometry_and_search_settings(self):
        provider = CountingBuilder()
        cache = CachedNeighborBuilder(provider)
        structure = sample()
        cache.build(structure)
        structure.energy = torch.tensor(42.)
        cache.build(structure)
        self.assertEqual(provider.calls, 1)
        structure.positions[1, 0] = 2
        cache.build(structure)
        structure.cell[0, 0] = 10
        cache.build(structure)
        structure.pbc[0] = True
        cache.build(structure)
        provider.cutoff = 1.5
        self.assertEqual(cache.build(structure).num_edges, 0)
        self.assertEqual(provider.calls, 5)
        cache.clear()
        self.assertEqual(len(cache), 0)

    def test_model_internal_and_none_skip_even_preloading(self):
        for model in (InternalModel(), NoNeighborModel()):
            provider = CountingBuilder()
            loader = create_dataloader(AtomicDataset([sample()]), provider,
                                       model=model, neighbor_cache='memory')
            self.assertIsNone(next(iter(loader))[0].edge_ptr)
            self.assertEqual(provider.calls, 0)

    def test_external_model_selects_backend(self):
        batch = next(iter(create_dataloader(AtomicDataset([sample()]), model=ExternalModel())))
        self.assertEqual(batch[0].neighbor_indices.numel(), 2)
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            create_dataloader(AtomicDataset([sample()]), ASENeighborBuilder(1), model=ExternalModel())

    def test_internal_model_predicts_multistructure_batch_without_graph(self):
        model = InternalEnergyModel()
        batch = next(iter(create_dataloader(AtomicDataset([sample(1), sample(2)]),
                                            model=model, batch_size=2, neighbor_cache='memory')))
        output = model(batch[0])
        torch.testing.assert_close(output.energy, torch.tensor([1., 4.], dtype=torch.float64))
        self.assertEqual(output.virial.shape, (2, 3, 3))
        output.forces.square().sum().backward()
        self.assertTrue(torch.isfinite(model.scale.grad))

    def test_preloaded_cache_survives_worker_restart(self):
        provider = ParentOnlyBuilder()
        loader = create_dataloader(AtomicDataset([sample(1), sample(2)]), provider,
                                   neighbor_cache='memory', num_workers=1, batch_size=2)
        for _ in range(2):
            self.assertEqual(next(iter(loader))[0].neighbor_indices.numel(), 4)
        self.assertEqual(provider.calls, 2)

    def test_cached_and_uncached_batches_match(self):
        dataset = AtomicDataset([sample(1), sample(2)])
        cached = next(iter(create_dataloader(dataset, CountingBuilder(), neighbor_cache='memory', batch_size=2)))
        plain = StructureCollator(CountingBuilder())(list(dataset))
        for name in ('center_indices', 'edge_ptr', 'neighbor_indices', 'edge_vectors', 'edge_shifts'):
            torch.testing.assert_close(getattr(cached[0], name), getattr(plain[0], name))

    def test_loader_configuration(self):
        config = LoaderConfig.from_dict({'batch_size': 2, 'neighbor_cache': 'memory'}, path='data.loader')
        self.assertEqual(config.to_kwargs()['neighbor_cache'], 'memory')
        with self.assertRaises(ConfigError):
            LoaderConfig.from_dict({'batch_size': 2, 'neighbor_cache': 'disk'}, path='data.loader')

    def test_workflow_cache_matches_uncached_training(self):
        from abfml.workflows.train import run
        original_dtype = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original_dtype)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for distance in (1.5, 1.6, 1.7, 1.8):
                atoms = Atoms('H2', positions=[[0, 0, 0], [distance, 0, 0]])
                atoms.calc = SinglePointCalculator(atoms, energy=-0.2)
                frames.append(atoms)
            write(root / 'data.extxyz', frames)
            config = {
                'config_version': 1,
                'common': {'elements': ['H'], 'dtype': 'float64', 'seed': 0},
                'model': {'name': 'lennard_jones', 'cutoff': 3., 'epsilon': 0.2, 'sigma': 1.},
                'data': {'files': ['data.extxyz'], 'split': {'train': 0.5, 'valid': 0.5},
                         'loader': {'batch_size': 2, 'shuffle': False}},
                'training': {'epochs': 2, 'device': 'cpu',
                             'optimizer': {'name': 'adam', 'learning_rate': 0.001}},
                'loss': {'label': ['energy'], 'start_weight': [1.], 'limit_weight': [1.]},
                'output': {'dir': 'run', 'checkpoint': {'save_every': 1},
                           'model_filename': 'model.pt'},
            }
            states = []
            # Two epochs build eight graphs plus four for final evaluation; caching reuses them.
            for mode, expected_calls in [('none', 12), ('memory', 4)]:
                config['data']['loader']['neighbor_cache'] = mode
                config['output']['dir'] = mode
                path = root / 'config.json'
                path.write_text(json.dumps(config), encoding='utf-8')
                provider = CountingBuilder()
                with patch('abfml.core.neighbor.ASENeighborBuilder', return_value=provider):
                    trained = run(path)
                self.assertEqual(provider.calls, expected_calls)
                self.assertTrue((root / mode / 'last.pt').exists())
                states.append(trained.state_dict())
            for name in states[0]:
                torch.testing.assert_close(states[0][name], states[1][name])


if __name__ == '__main__':
    unittest.main()
