"""Dynamic loss-weight and learning-rate formula tests."""

import unittest

from abfml.training import adjust_lr, calculate_weight


class ScheduleTest(unittest.TestCase):
    def test_loss_weight_interpolation(self) -> None:
        start = {"energy": 1.0, "forces": 100.0}
        limit = {"energy": 10.0, "forces": 1.0}
        self.assertEqual(calculate_weight(start, limit, 1.0), start)
        self.assertEqual(calculate_weight(start, limit, 0.0), limit)
        weights = calculate_weight(start, limit, 0.8)
        self.assertAlmostEqual(weights["energy"], 2.8)
        self.assertAlmostEqual(weights["forces"], 80.2)

    def test_continuous_and_stepped_learning_rate(self) -> None:
        arguments = {
            "start_lr": 0.001,
            "limit_lr": 0.0001,
            "decay_rate": 0.5,
            "decay_step": 100,
        }
        self.assertAlmostEqual(adjust_lr(**arguments, iters_step=0), 0.001)
        self.assertAlmostEqual(adjust_lr(**arguments, iters_step=100), 0.0005)
        self.assertAlmostEqual(adjust_lr(**arguments, iters_step=200), 0.00025)
        self.assertAlmostEqual(
            adjust_lr(**arguments, iters_step=99, type_name="exp_decay"),
            0.001,
        )
        self.assertAlmostEqual(
            adjust_lr(**arguments, iters_step=100, type_name="exp_decay"),
            0.0005,
        )
        self.assertEqual(adjust_lr(**arguments, iters_step=10000), 0.0001)


if __name__ == "__main__":
    unittest.main()
