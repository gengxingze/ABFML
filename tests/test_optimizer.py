"""Optimizer and learning-rate scheduler construction tests."""

import unittest

import torch

from abfml.training.optimizer import build_optimizer, build_scheduler


class BuildOptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Linear(4, 2)

    def test_builds_registered_optimizer(self) -> None:
        optimizer = build_optimizer(self.model, "adam", learning_rate=0.01)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_unknown_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported optimizer"):
            build_optimizer(self.model, "rmsprop", learning_rate=0.01)

    def test_no_trainable_parameters_is_rejected(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        with self.assertRaisesRegex(ValueError, "trainable"):
            build_optimizer(self.model, "adam", learning_rate=0.01)


class BuildSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Linear(4, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.inputs = torch.randn(8, 4)

    def _optimizer_step(self) -> None:
        """Advance the optimizer before the scheduler, matching train_loop."""
        self.optimizer.zero_grad()
        self.model(self.inputs).sum().backward()
        self.optimizer.step()

    def test_none_and_none_name_return_none(self) -> None:
        self.assertIsNone(build_scheduler(self.optimizer))
        self.assertIsNone(build_scheduler(self.optimizer, "none", start_lr=0.01))

    def test_exponential_decays_per_step(self) -> None:
        scheduler = build_scheduler(
            self.optimizer,
            "exponential",
            start_lr=0.01,
            limit_lr=0.0,
            decay_rate=0.5,
            decay_step=10,
        )
        self.assertIsNotNone(scheduler)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], 0.01)

        self._optimizer_step()
        scheduler.step()  # First advance reaches global step 1.
        self.assertAlmostEqual(
            self.optimizer.param_groups[0]["lr"], 0.01 * 0.5 ** 0.1
        )
        self._optimizer_step()
        scheduler.step()  # Continue decay at step 2.
        self.assertAlmostEqual(
            self.optimizer.param_groups[0]["lr"], 0.01 * 0.5 ** 0.2
        )

    def test_unknown_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported scheduler"):
            build_scheduler(self.optimizer, "plateau", start_lr=0.01)


if __name__ == "__main__":
    unittest.main()
