"""loss.out formatting, dynamic-column, and append tests."""

import tempfile
import unittest
from pathlib import Path

from abfml.logger import LossLog


def metrics(epoch: int, *, extra: bool = False) -> dict:
    values = {
        "epoch": epoch,
        "train_loss": 4.743e-3,
        "valid_loss": 3.907e-3,
        "learning_rate": 9.5e-4,
        "train_energy_per_atom_rmse": 1.952e-3,
        "train_forces_rmse": 6.689e-2,
        "train_virial_per_atom_rmse": 5.15e-2,
        "valid_energy_per_atom_rmse": 4.535e-4,
        "valid_forces_rmse": 6.064e-2,
        "valid_virial_per_atom_rmse": 4.8e-2,
        "train_time": 18.0,
        "valid_time": 1.0,
    }
    if extra:
        values["train_charges_rmse"] = 1.25e-2
        values["valid_charges_rmse"] = 1.1e-2
    return values


class LossLogTest(unittest.TestCase):
    def test_standard_columns_and_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loss.out"
            log = LossLog(path)
            log.append(metrics(1))

            text = path.read_text(encoding="utf-8")
            self.assertIn("# columns: Epoch loss V-loss lr T-energy", text)
            self.assertIn("meV/atom", text)
            data = [
                line for line in text.splitlines()
                if line and not line.startswith("#")
            ]
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0].split()[0], "1")
            self.assertEqual(data[0].split()[3], "9.500e-04")

    def test_extra_label_extends_header_and_pads_old_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loss.out"
            log = LossLog(path)
            log.append(metrics(1))
            log.append(metrics(2, extra=True))

            lines = path.read_text(encoding="utf-8").splitlines()
            columns = next(line for line in lines if line.startswith("# columns:")).split()[2:]
            self.assertIn("T-charges", columns)
            self.assertIn("V-charges", columns)
            self.assertLess(columns.index("T-charges"), columns.index("time"))
            units = next(line for line in lines if line.startswith("# units:")).split()[2:]
            self.assertEqual(units[columns.index("T-charges")], "e")

            data = [line.split() for line in lines if line and not line.startswith("#")]
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0][columns.index("T-charges")], "NA")
            self.assertEqual(data[1][columns.index("T-charges")], "1.250e-02")

    def test_resume_preserves_existing_rows_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loss.out"
            first = LossLog(path)
            first.append(metrics(1, extra=True))

            resumed = LossLog(path, append=True)
            resumed.append(metrics(2, extra=True))

            lines = path.read_text(encoding="utf-8").splitlines()
            data = [line.split() for line in lines if line and not line.startswith("#")]
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0][0], "1")
            self.assertEqual(data[1][0], "2")
            self.assertIn("T-charges", lines[2])
            columns = lines[2].split()[2:]
            units = lines[3].split()[2:]
            self.assertEqual(len(columns), len(set(columns)))
            self.assertEqual(len(columns), len(units))
            self.assertEqual(columns.count("T-force"), 1)

    def test_unknown_label_units_are_single_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loss.out"
            log = LossLog(path)
            log.append({**metrics(1), "train_custom_rmse": .1, "valid_custom_rmse": .2})
            lines = path.read_text(encoding="utf-8").splitlines()
            columns, units = lines[2].split()[2:], lines[3].split()[2:]
            self.assertEqual(len(columns), len(units))
            self.assertEqual(units[columns.index("T-custom")], "label_units")


if __name__ == "__main__":
    unittest.main()
