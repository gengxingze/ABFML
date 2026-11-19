"""End-to-end tests for valid_loop and fit."""

import unittest

import torch

from abfml.data import StructureCollator
from abfml.core import AtomicSample
from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput
from abfml.core.neighbor import CSRNeighbors
from abfml.builtin.functions import segment_sum
from abfml.training import calculate_loss, fit, valid_loop


class HarmonicField(AtomicModel):
    """E = k/2 * sum_edges |r_ij|^2 for independent training/validation checks."""

    def __init__(self) -> None:
        super().__init__(type_map=[1], cutoff=5.0)
        self.spring = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

    def compute(self, inputs: ModelInput) -> ModelOutput:
        _, edge_ptr, _, edge_vectors = inputs.require_neighbors()
        edge_energies = 0.5 * self.spring * edge_vectors.square().sum(dim=-1)
        atomic_energies = segment_sum(edge_energies, edge_ptr)
        # Omit explicit forces to verify valid_loop derives mechanics from differentiable energy.
        return ModelOutput(
            energy=atomic_energies.sum(),
            atomic_energies=atomic_energies,
        )


def make_batch(target_spring: float = 2.0):
    sample = AtomicSample(
        atomic_numbers=[1, 1],
        positions=torch.tensor([[-1., 0., 0.], [1., 0., 0.]], dtype=torch.float64),
        cell=torch.eye(3, dtype=torch.float64)*10, pbc=[False]*3,
        energy=torch.tensor(4.*target_spring, dtype=torch.float64),
        forces=torch.tensor([[4.*target_spring, 0., 0.], [-4.*target_spring, 0., 0.]], dtype=torch.float64),
        extra_labels={"charges": torch.tensor([0.1, -0.1], dtype=torch.float64)},
    )
    neighbors = CSRNeighbors(center_indices=[0,1], edge_ptr=[0,1,2], neighbor_indices=[1,0],
        edge_vectors=torch.tensor([[2.,0.,0.],[-2.,0.,0.]], dtype=torch.float64))
    return StructureCollator()([sample], [neighbors])


def energy_loss(
    output: ModelOutput,
    batch: tuple,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return calculate_loss(
        output,
        batch,
        {
            "metric": "mse",
            "label": ["energy"],
            "start_weight": [1.0],
            "limit_weight": [1.0],
        },
    )


class ValidLoopTest(unittest.TestCase):
    def test_returns_metrics_without_updating_parameters(self) -> None:
        model = HarmonicField()
        before = float(model.spring.detach())
        metrics = valid_loop([make_batch(), make_batch()], model, energy_loss)
        self.assertEqual(float(model.spring.detach()), before)
        self.assertIn("loss", metrics)
        self.assertIn("loss_energy", metrics)
        self.assertGreaterEqual(metrics["time"], 0.0)

    def test_empty_loader_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            valid_loop([], HarmonicField(), energy_loss)


class FitTest(unittest.TestCase):
    def test_runs_epochs_and_returns_history(self) -> None:
        model = HarmonicField()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        history, step = fit(
            model,
            [make_batch()],
            optimizer,
            energy_loss,
            epochs=3,
        )
        self.assertEqual(len(history), 3)
        self.assertEqual(step, 3)
        self.assertIn("train_loss", history[0])
        self.assertIn("epoch", history[0])
        self.assertNotIn("valid_loss", history[0])

    def test_valid_dataloader_adds_valid_metrics(self) -> None:
        model = HarmonicField()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        history, _ = fit(
            model,
            [make_batch()],
            optimizer,
            energy_loss,
            epochs=1,
            valid_dataloader=[make_batch()],
        )
        self.assertIn("valid_loss", history[0])

    def test_invalid_epochs_are_rejected(self) -> None:
        model = HarmonicField()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaisesRegex(ValueError, "epochs"):
            fit(model, [make_batch()], optimizer, energy_loss, epochs=0)


if __name__ == "__main__":
    unittest.main()
