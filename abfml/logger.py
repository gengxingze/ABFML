"""Shared ABFML logging, metric formatting, and per-epoch loss records."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path
from abfml.core.schema import LABELS, label_unit

DEFAULT_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_METRIC_ALIASES = {"energy": "E", "forces": "F"}
_NON_ERROR_METRICS = {"epoch", "loss", "time"}

_BASE_COLUMNS = (
    "Epoch",
    "loss",
    "V-loss",
    "lr",
    "T-energy",
    "T-force",
    "T-virial",
    "V-energy",
    "V-force",
    "V-virial",
    "time",
)
_BASE_UNITS = (
    "-",
    "-",
    "-",
    "-",
    label_unit("energy_per_atom", display=True),
    label_unit("forces", display=True),
    label_unit("virial_per_atom", display=True),
    label_unit("energy_per_atom", display=True),
    label_unit("forces", display=True),
    label_unit("virial_per_atom", display=True),
    "s",
)
_STANDARD_LABELS = {"energy", "forces", "virial", "energy_per_atom", "virial_per_atom"}


def get_logger(name: str) -> logging.Logger:
    """Return a module logger without adding module-local handlers."""
    return logging.getLogger(name)


def format_timing(durations: Mapping[str, float]) -> str:
    """Format four wall-clock stages and shares using the same interval for the total."""
    total = sum(durations.values())
    rows = [["Stage", "Time (s)", "Share (%)"]]
    for name, seconds in durations.items():
        rows.append([name, f"{seconds:.2f}", f"{100. * seconds / total:.2f}" if total else "0.00"])
    rows.append(["Total", f"{total:.2f}", "100.00" if total else "0.00"])
    widths = [max(len(row[index]) for row in rows) for index in range(3)]
    lines = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows]
    separator = "-+-".join("-" * width for width in widths)
    lines.insert(1, separator)
    lines.insert(-1, separator)
    return "\n".join(lines)


def format_summary(reports: Mapping[str, Mapping]) -> str:
    """Format full-dataset metrics using training-log meV conventions for energy, forces, and
    virial.
    """
    metrics = {name: report["metrics"] for name, report in reports.items()}
    available = set().union(*(group.keys() for group in metrics.values()))
    selected = [name for name in available
                if not (name in {"energy", "virial"} and name + "_per_atom" in available)]
    standard = {"energy_per_atom": "Energy", "forces": "Forces", "virial_per_atom": "Virial"}
    order = {name: index for index, name in enumerate(standard)}
    header = ["Label", "Unit"]
    for group in metrics:
        header.extend((f"{group} RMSE", f"{group} Rel. (%)"))
    rows = [header]
    for name in sorted(selected, key=lambda name: (order.get(name, 99), name)):
        sample = next(group[name] for group in metrics.values() if name in group)
        label = standard.get(name, name)
        spec = LABELS.get(name)
        unit = label_unit(name, display=True) if spec else sample["unit"]
        scale = spec.display_scale if spec else 1.
        row = [label, unit]
        for group in metrics.values():
            value = group.get(name)
            if value is None:
                row.extend(("N/A", "N/A"))
            else:
                relative = value["relative_error_percent"]
                row.extend((f"{value['rmse'] * scale:.2f}", f"{relative:.2f}" if relative is not None else "N/A"))
        rows.append(row)
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    lines = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows]
    lines.insert(1, "-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def format_evaluation(report: Mapping) -> str:
    """Format independent evaluation metrics using meV conventions for energy, forces, and virial."""
    metrics = report["metrics"]
    available = set(metrics)
    selected = [name for name in available
                if not (name in {"energy", "virial"} and name + "_per_atom" in available)]
    standard = {"energy_per_atom": "Energy", "forces": "Forces", "virial_per_atom": "Virial"}
    order = {name: index for index, name in enumerate(standard)}
    header = ["Label", "Unit", "Count", "MAE", "RMSE", "Rel. (%)"]
    rows = [header]
    for name in sorted(selected, key=lambda name: (order.get(name, 99), name)):
        value = metrics[name]
        label = standard.get(name, name)
        spec = LABELS.get(name)
        # Windows GBK cannot encode angstrom symbols; use A in terminals and the symbol in figures.
        unit = (label_unit(name, display=True) if spec else value["unit"]).replace("Å", "A")
        scale = spec.display_scale if spec else 1.
        relative = value["relative_error_percent"]
        rows.append([label, unit, str(value["count"]), f"{value['mae'] * scale:.2f}",
                     f"{value['rmse'] * scale:.2f}",
                     f"{relative:.2f}" if relative is not None else "N/A"])
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    lines = [" | ".join(value.ljust(width) for value, width in zip(row, widths)) for row in rows]
    lines.insert(1, "-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def configure_logging(
    log_file: str | Path | None = None,
    *,
    level: int = logging.INFO,
) -> None:
    """Configure console logging once at the CLI entry point, optionally appending a UTF-8 file.
    
    Library users need not call this. Preserve existing root-logger configuration instead of
    adding duplicate handlers.
    """
    if logging.getLogger().handlers:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
        handlers=handlers,
    )


def _metric_label(name: str) -> str:
    """Convert label_metric into a log name without restricting labels or error metrics."""
    label, separator, metric = name.rpartition("_")
    if not separator:
        return _METRIC_ALIASES.get(name, name)
    return f"{_METRIC_ALIASES.get(label, label)}_{metric.upper()}"


def format_metrics(metrics: Mapping[str, float], *, prefix: str = "") -> str:
    """Format scalar metrics in caller-provided order."""
    return "".join(
        f" , {prefix}{_metric_label(name)}: {float(value):.6e}"
        for name, value in metrics.items()
    )


def format_errors(metrics: Mapping[str, float], *, prefix: str = "") -> str:
    """Format actual errors with labels and error types inferred from metric keys."""
    errors = {
        name: value
        for name, value in metrics.items()
        if name not in _NON_ERROR_METRICS and not name.startswith("loss_")
    }
    return format_metrics(errors, prefix=prefix)


def format_progress(metrics: Mapping[str, float], *, prefix: str = "") -> str:
    """Format training/validation physical errors using meV conventions.
    
    E_RMSE uses energy_per_atom_rmse, F_RMSE uses forces_rmse, and virial_RMSE uses
    virial_per_atom_rmse. Additional labels with references append dynamic <label>_RMSE entries.
    """
    parts: list[str] = []

    def add(name: str, value: object, label: str, specification: str) -> None:
        try:
            number = float(value) * LABELS[label].display_scale
        except (TypeError, ValueError):
            return
        if not math.isfinite(number):
            return
        parts.append(f" , {prefix}{name}: {number:{specification}} {label_unit(label, display=True)}")

    add("E_RMSE", metrics.get("energy_per_atom_rmse"), "energy_per_atom", ".2f")
    add("F_RMSE", metrics.get("forces_rmse"), "forces", ".2f")
    add("virial_RMSE", metrics.get("virial_per_atom_rmse"), "virial_per_atom", ".2f")

    standard = {
        "energy_rmse",
        "energy_per_atom_rmse",
        "forces_rmse",
        "virial_rmse",
        "virial_per_atom_rmse",
    }
    for name in sorted(metrics):
        if name in standard or not name.endswith("_rmse"):
            continue
        label = name[:-len("_rmse")]
        unit = label_unit(label)
        try:
            value = float(metrics[name])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        parts.append(f" , {prefix}{label}_RMSE: {value:.2e} {unit}")
    return "".join(parts)


class LossLog:
    """Write standard loss.out columns and dynamically extend additional labels.
    
    append=True preserves existing rows. New labels extend the header and fill historical cells
    with NA. Writes use a temporary file followed by replacement.
    """

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self._columns = list(_BASE_COLUMNS)
        self._units = list(_BASE_UNITS)
        self._extras: list[str] = []
        self._rows: list[dict[str, str]] = []
        if append and self.path.exists():
            self._read()
        self._write()

    def append(self, metrics: Mapping[str, object]) -> None:
        """Append one epoch's metrics."""
        extras = sorted(set(self._extras) | set(self._discover_extras(metrics)))
        desired = self._desired_columns(extras)
        if desired != self._columns:
            self._columns = desired
            self._units = self._desired_units(extras)
            self._extras = extras
            for row in self._rows:
                for column in self._columns:
                    row.setdefault(column, "NA")

        row = self._make_row(metrics, extras)
        self._rows.append(row)
        self._write()

    def _discover_extras(self, metrics: Mapping[str, object]) -> list[str]:
        labels: set[str] = set()
        for prefix in ("train_", "valid_"):
            for key in metrics:
                if not key.startswith(prefix) or not key.endswith("_rmse"):
                    continue
                label = key[len(prefix):-len("_rmse")]
                if label not in _STANDARD_LABELS:
                    labels.add(label)
        return sorted(labels)

    def _desired_columns(self, extras: list[str]) -> list[str]:
        extra_columns = [name for label in extras for name in (f"T-{label}", f"V-{label}")]
        return list(_BASE_COLUMNS[:-1]) + extra_columns + [_BASE_COLUMNS[-1]]

    def _desired_units(self, extras: list[str]) -> list[str]:
        extra_units = [
            label_unit(label)
            for label in extras
            for _ in (f"T-{label}", f"V-{label}")
        ]
        return list(_BASE_UNITS[:-1]) + extra_units + [_BASE_UNITS[-1]]

    def _make_row(self, metrics: Mapping[str, object], extras: list[str]) -> dict[str, str]:
        values: dict[str, str] = {
            "Epoch": self._epoch(metrics.get("epoch")),
            "loss": self._format(metrics.get("train_loss"), ".3e"),
            "V-loss": self._format(metrics.get("valid_loss"), ".3e"),
            "lr": self._format(metrics.get("learning_rate"), ".3e"),
            "T-energy": self._display(metrics.get("train_energy_per_atom_rmse"), "energy_per_atom"),
            "T-force": self._display(metrics.get("train_forces_rmse"), "forces"),
            "T-virial": self._display(metrics.get("train_virial_per_atom_rmse"), "virial_per_atom"),
            "V-energy": self._display(metrics.get("valid_energy_per_atom_rmse"), "energy_per_atom"),
            "V-force": self._display(metrics.get("valid_forces_rmse"), "forces"),
            "V-virial": self._display(metrics.get("valid_virial_per_atom_rmse"), "virial_per_atom"),
            "time": self._format(self._total_time(metrics), ".1f"),
        }
        for label in extras:
            values[f"T-{label}"] = self._format(metrics.get(f"train_{label}_rmse"), ".3e")
            values[f"V-{label}"] = self._format(metrics.get(f"valid_{label}_rmse"), ".3e")
        return {column: values.get(column, "NA") for column in self._columns}

    @staticmethod
    def _epoch(value: object) -> str:
        if value is None:
            return "NA"
        return str(int(float(value)))

    @staticmethod
    def _format(value: object, specification: str) -> str:
        if value is None:
            return "NA"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "NA"
        if not math.isfinite(number):
            return "NA"
        return format(number, specification)

    def _display(self, value: object, label: str) -> str:
        if value is None:
            return "NA"
        try:
            number = float(value) * LABELS[label].display_scale
        except (TypeError, ValueError):
            return "NA"
        if not math.isfinite(number):
            return "NA"
        return format(number, ".3f")

    @staticmethod
    def _total_time(metrics: Mapping[str, object]) -> float:
        train = float(metrics.get("train_time", 0.0) or 0.0)
        valid = float(metrics.get("valid_time", 0.0) or 0.0)
        return train + valid

    def _read(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        columns: list[str] | None = None
        units: list[str] | None = None
        rows: list[dict[str, str]] = []
        for line in text.splitlines():
            if line.startswith("# columns:"):
                columns = line[len("# columns:"):].split()
            elif line.startswith("# units:"):
                units = line[len("# units:"):].split()
            elif line and not line.startswith("#"):
                values = line.split()
                if columns is None:
                    raise ValueError(f"loss.out has data before the column header: {self.path}")
                if len(values) != len(columns):
                    raise ValueError(
                        f"loss.out row has {len(values)} values, expected {len(columns)}: {self.path}"
                    )
                rows.append(dict(zip(columns, values)))
        if columns is None:
            return
        self._columns = columns
        self._units = units if units and len(units) == len(columns) else [
            _BASE_UNITS[_BASE_COLUMNS.index(column)] if column in _BASE_COLUMNS
            else label_unit(column[2:])
            for column in columns
        ]
        extras: set[str] = set()
        for column in columns:
            if column.startswith("T-") and column not in _BASE_COLUMNS:
                extras.add(column[2:])
        self._extras = sorted(extras)
        self._rows = rows

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# ABFML loss.out",
            "# T=train, V=valid; E/F/V=RMSE; loss=weighted objective; time=epoch wall seconds",
            "# columns: " + " ".join(self._columns),
            "# units: " + " ".join(self._units),
        ]
        for row in self._rows:
            lines.append(" ".join(row.get(column, "NA") for column in self._columns))
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
