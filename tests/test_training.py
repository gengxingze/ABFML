"""Training loops, dynamic label matching, and optional error-metric tests."""

import unittest

import torch

from abfml.data import StructureCollator
from abfml.core import AtomicSample
from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput
from abfml.core.neighbor import CSRNeighbors
from abfml.logger import format_errors, format_metrics, format_progress, format_timing
from abfml.builtin.functions import segment_sum
from abfml.training import calculate_loss, train_loop


class HarmonicField(AtomicModel):
    """E = k/2 * sum_edges |r_ij|^2 for independent training-loop checks."""

    def __init__(self) -> None:
        super().__init__(type_map=[1], cutoff=5.0)
        self.spring = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

    def compute(self, inputs: ModelInput) -> ModelOutput:
        _, edge_ptr, _, edge_vectors = inputs.require_neighbors()
        edge_energies = 0.5 * self.spring * edge_vectors.square().sum(dim=-1)
        atomic_energies = segment_sum(edge_energies, edge_ptr)
        return ModelOutput(
            energy=atomic_energies.sum(),
            atomic_energies=atomic_energies,
            properties={
                "charges": self.spring * inputs.atomic_numbers.to(
                    dtype=edge_vectors.dtype
                )
            },
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


class TrainLoopTest(unittest.TestCase):
    def test_updates_parameter_metrics_and_global_step(self) -> None:
        torch.set_default_dtype(torch.float64)
        model = HarmonicField()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = make_batch()
        metrics, next_step = train_loop(
            [batch, batch],
            model,
            optimizer,
            7,
            energy_loss,
        )
        self.assertEqual(next_step, 9)
        self.assertNotEqual(float(model.spring.detach()), 0.5)
        self.assertIn("loss", metrics)
        self.assertIn("loss_energy", metrics)
        self.assertIn("energy_rmse", metrics)
        self.assertIn("forces_rmse", metrics)
        self.assertIn("charges_rmse", metrics)
        self.assertNotIn("forces_mae", metrics)
        self.assertGreaterEqual(metrics["time"], 0.0)

    def test_train_log_uses_mev_units(self) -> None:
        torch.set_default_dtype(torch.float64)
        model = HarmonicField()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertLogs("abfml.training", level="INFO") as captured:
            train_loop([make_batch()], model, optimizer, 0, energy_loss)
        text = "\n".join(captured.output)
        self.assertIn("E_RMSE:", text)
        self.assertIn("meV/atom", text)
        self.assertIn("F_RMSE:", text)
        self.assertIn("meV/A", text)
        self.assertNotIn("energy_per_atom_RMSE", text)

    def test_empty_loader_is_rejected(self) -> None:
        model = HarmonicField()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaisesRegex(ValueError, "empty"):
            train_loop([], model, optimizer, 0, energy_loss)


class LoggerFormattingTest(unittest.TestCase):
    def test_timing_percentages_and_total(self):
        text = format_timing({'Data & initialization': 2., 'Training & validation': 5.,
                              'Final evaluation': 1., 'Result output': 2.})
        rows = [line.split('|') for line in text.splitlines() if '|' in line]
        self.assertEqual(len(rows), 6)
        self.assertEqual([row[2].strip() for row in rows[1:]],
                         ['20.00', '50.00', '10.00', '20.00', '100.00'])
        self.assertEqual(rows[-1][1].strip(), '10.00')

    def test_formats_arbitrary_labels_and_metric_types(self) -> None:
        text = format_errors({
            "loss": 1.0,
            "loss_energy": 2.0,
            "energy_rmse": 0.1,
            "charges_mae": 0.2,
            "dipole_relative_error": 0.3,
        })
        self.assertIn("E_RMSE", text)
        self.assertIn("charges_MAE", text)
        self.assertIn("dipole_relative_ERROR", text)
        self.assertNotIn("loss_energy", text)
        self.assertNotIn("loss:", text)

    def test_generic_formatter_does_not_require_an_error_suffix(self) -> None:
        self.assertIn("quality: 5.000000e-01", format_metrics({"quality": 0.5}))

    def test_training_formatter_uses_mev_units_and_skips_total_energy(self) -> None:
        text = format_progress(
            {
                "energy_rmse": 2.106609e-02,
                "energy_per_atom_rmse": 1.64579e-04,
                "forces_rmse": 2.323948e-02,
                "virial_rmse": 1.283932e00,
                "virial_per_atom_rmse": 1.00307e-02,
                "charges_rmse": 1.25e-02,
            }
        )
        self.assertIn("E_RMSE: 0.16 meV/atom", text)
        self.assertIn("F_RMSE: 23.24 meV/A", text)
        self.assertIn("virial_RMSE: 10.03 meV/atom", text)
        self.assertIn("charges_RMSE: 1.25e-02 e", text)
        self.assertNotIn("energy_RMSE", text)
        self.assertNotIn("virial_rmse", text)

    def test_training_formatter_prefix(self) -> None:
        text = format_progress(
            {"energy_per_atom_rmse": 1.0e-4, "forces_rmse": 1.0e-2, "virial_per_atom_rmse": 1.0e-2},
            prefix="train ",
        )
        self.assertTrue(text.startswith(" , train E_RMSE:"))
        self.assertIn("train F_RMSE:", text)
        self.assertIn("train virial_RMSE:", text)


class CalculateLossTest(unittest.TestCase):
    def test_mse_rmse_dynamic_matching_and_extended_label(self) -> None:
        torch.set_default_dtype(torch.float64)
        output = ModelOutput(
            energy=torch.tensor([3.0], requires_grad=True),
            atomic_energies=None,
            forces=torch.zeros(2, 3, requires_grad=True),
            properties={
                "charges": torch.tensor([0.2, 0.4], requires_grad=True)
            },
        )
        loss, components = calculate_loss(
            output,
            make_batch(),
            {
                "metric": "mse",
                "label": ["energy", "charges"],
                "start_weight": [2.0, 0.5],
                "limit_weight": [2.0, 0.5],
            },
        )
        force_loss, force_components = calculate_loss(
            output,
            make_batch(),
            {
                "metric": "rmse",
                "label": ["forces"],
                "start_weight": [1.0],
                "limit_weight": [1.0],
            },
        )
        force_rmse = (64.0 / 3.0) ** 0.5
        self.assertAlmostEqual(float(components["energy"].detach()), 6.25, places=6)
        self.assertAlmostEqual(
            float(components["charges"].detach()), 0.13, places=6
        )
        self.assertAlmostEqual(float(loss.detach()), 12.565, places=6)
        self.assertAlmostEqual(
            float(force_components["forces"].detach()), force_rmse, places=6
        )
        self.assertAlmostEqual(float(force_loss.detach()), force_rmse, places=6)
        self.assertEqual(set(components), {"energy", "charges"})
        self.assertEqual(set(force_components), {"forces"})

    def test_dynamic_weights_follow_lr_ratio(self) -> None:
        output = ModelOutput(
            energy=torch.tensor([3.0], requires_grad=True),
            atomic_energies=None,
            forces=torch.zeros(2, 3),
        )
        loss, _ = calculate_loss(
            output,
            make_batch(),
            {
                "metric": "mse",
                "label": ["energy"],
                "start_weight": [2.0],
                "limit_weight": [4.0],
            },
            lr_ratio=0.5,
        )
        self.assertAlmostEqual(float(loss.detach()), 18.75)

    def test_no_matching_term_is_rejected(self) -> None:
        output = ModelOutput(
            energy=torch.tensor([3.0], requires_grad=True),
            atomic_energies=None,
            forces=torch.zeros(2, 3),
        )
        with self.assertRaisesRegex(ValueError, "Configured loss term"):
            calculate_loss(
                output,
                make_batch(),
                {
                    "metric": "mse",
                    "label": ["dipole"],
                    "start_weight": [1.0],
                    "limit_weight": [1.0],
                },
            )

    def test_parallel_lists_must_have_equal_lengths(self) -> None:
        output = ModelOutput(
            energy=torch.tensor([3.0], requires_grad=True),
            atomic_energies=None,
            forces=torch.zeros(2, 3),
        )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            calculate_loss(
                output,
                make_batch(),
                {
                    "metric": "mse",
                    "label": ["energy", "forces"],
                    "start_weight": [1.0],
                    "limit_weight": [1.0, 100.0],
                },
            )


if __name__ == "__main__":
    unittest.main()
