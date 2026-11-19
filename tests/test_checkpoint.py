"""Checkpoint restoration and resumed epoch-count tests."""

import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_trainer import HarmonicField, energy_loss, make_batch
from abfml.training import fit, load_checkpoint, restore_state, save_checkpoint


class CheckpointTest(unittest.TestCase):
    def test_full_restore_and_weights_only_restore(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=3,
                step=12,
                config={"model": {"name": "demo"}},
                metrics={"train_loss": float(loss.detach())},
                best_metric=0.25,
            )
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["step"], 12)

            resumed = torch.nn.Linear(2, 1)
            resumed_optimizer = torch.optim.SGD(
                resumed.parameters(), lr=0.1, momentum=0.9
            )
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(
                resumed_optimizer, step_size=1, gamma=0.5
            )
            epoch, step = restore_state(
                checkpoint, resumed, resumed_optimizer, resumed_scheduler
            )
            self.assertEqual((epoch, step), (3, 12))
            torch.testing.assert_close(resumed.weight, model.weight)
            self.assertEqual(resumed_scheduler.last_epoch, scheduler.last_epoch)

            tuned = torch.nn.Linear(2, 1)
            tuned.load_state_dict(checkpoint["model_state"])
            torch.testing.assert_close(tuned.weight, model.weight)

    def test_fit_continues_epoch_numbers_and_reports_steps(self) -> None:
        model = HarmonicField()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        callbacks = []
        history, step = fit(
            model,
            [make_batch()],
            optimizer,
            energy_loss,
            epochs=2,
            initial_epoch=3,
            initial_step=7,
            epoch_callback=lambda epoch, metrics, step: callbacks.append((epoch, step)),
        )
        self.assertEqual([item["epoch"] for item in history], [4.0, 5.0])
        self.assertEqual(callbacks, [(4, 8), (5, 9)])
        self.assertEqual(step, 9)


if __name__ == "__main__":
    unittest.main()
