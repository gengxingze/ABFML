"""Application settings for common options, data, training, output, and model modules.

Shared validation and parsing live here; model-specific configuration lives with each model
implementation.
"""

from __future__ import annotations

import json
import math
from importlib import import_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ase.data import atomic_numbers as SYMBOL_TO_NUMBER

if TYPE_CHECKING:
    from abfml.core.registry import ModelConfig


class ConfigError(ValueError):
    """Configuration error with a field path such as model.descriptor.r_min."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def check_unknown(raw: Mapping, allowed: set[str], path: str) -> None:
    unknown = sorted(key for key in raw if key not in allowed)
    if unknown:
        raise ConfigError(path, f"unknown field(s): {unknown}")


def as_str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(path, f"expected a string, got {type(value).__name__}")
    return value


def as_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(path, f"expected a boolean, got {type(value).__name__}")
    return value


def as_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(path, f"expected an integer, got {type(value).__name__}")
    return value


def as_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(path, f"expected a number, got {type(value).__name__}")
    return float(value)


def int_list(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(path, "expected a list of integers")
    return tuple(as_int(item, f"{path}[{i}]") for i, item in enumerate(value))


def float_list(value: object, path: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(path, "expected a list of numbers")
    return tuple(as_float(item, f"{path}[{i}]") for i, item in enumerate(value))


def str_list(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(path, "expected a list of strings")
    return tuple(as_str(item, f"{path}[{i}]") for i, item in enumerate(value))


def resolve_path(value: str, base_dir: Path | None) -> str:
    """Resolve relative paths against the configuration directory and return absolute strings."""
    path = Path(value)
    if base_dir is not None and not path.is_absolute():
        return str((base_dir / path).resolve())
    return value


def _parse_widths(value: object, path: str) -> tuple[int, ...]:
    widths = int_list(value, path)
    if not widths:
        raise ConfigError(path, "must be a non-empty list")
    if any(width < 1 for width in widths):
        raise ConfigError(path, "all widths must be positive")
    return widths


def parse_network(
    raw: Mapping, *, path: str, layers_key: str = "hidden_layers"
) -> tuple[tuple[int, ...], str, bool, bool]:
    """Parse shared network fields into (widths, activation, bias, residual)."""
    # Load the construction mapping lazily to avoid circular builtin-model imports.
    from abfml.builtin.networks import ACTIVATIONS

    allowed = {layers_key, "activation", "bias", "residual"}
    check_unknown(raw, allowed, path)
    widths = _parse_widths(raw.get(layers_key), f"{path}.{layers_key}")
    activation = raw.get("activation", "tanh")
    if not isinstance(activation, str) or activation not in ACTIVATIONS:
        raise ConfigError(f"{path}.activation", f"unsupported activation: {activation}")
    bias = as_bool(raw.get("bias", True), f"{path}.bias")
    residual = as_bool(raw.get("residual", False), f"{path}.residual")
    return widths, str(activation), bias, residual


_DTYPES = {"float32", "float64"}


def _resolve_element(value: object, path: str) -> int:
    if isinstance(value, str):
        number = SYMBOL_TO_NUMBER.get(value)
        if number is None:
            raise ConfigError(path, f"unknown element symbol: {value}")
        return int(number)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    raise ConfigError(
        path, f"element must be a symbol or positive atomic number, got {value!r}"
    )


@dataclass(frozen=True)
class CommonConfig:
    """Global settings: user-ordered species, numerical precision, and random seed."""

    type_map: tuple[int, ...]
    dtype: str
    seed: int

    @classmethod
    def from_dict(cls, raw: Mapping, *, path: str = "common") -> "CommonConfig":
        check_unknown(raw, {"elements", "dtype", "seed"}, path)
        elements = raw.get("elements")
        if not isinstance(elements, (list, tuple)) or not elements:
            raise ConfigError(
                f"{path}.elements",
                "must be a non-empty list of element symbols or atomic numbers",
            )
        type_map = tuple(
            _resolve_element(element, f"{path}.elements[{index}]")
            for index, element in enumerate(elements)
        )
        if len(set(type_map)) != len(type_map):
            raise ConfigError(f"{path}.elements", "must not contain duplicate elements")

        dtype = raw.get("dtype", "float64")
        if dtype not in _DTYPES:
            raise ConfigError(f"{path}.dtype", f"must be one of {sorted(_DTYPES)}")
        seed = as_int(raw.get("seed", 0), f"{path}.seed")
        return cls(type_map=type_map, dtype=str(dtype), seed=seed)

    @property
    def num_types(self) -> int:
        return len(self.type_map)


# ---------------------------------------------------------------------------
# Data reading, splitting, and loading.
# ---------------------------------------------------------------------------


def _path_list(value: object, path: str) -> tuple[str, ...]:
    """Accept a string or list of strings and return a tuple."""
    if isinstance(value, str):
        return (value,)
    return str_list(value, path)


@dataclass(frozen=True)
class SplitConfig:
    """Random train/validation split fractions and an independent seed."""

    train: float
    valid: float
    seed: int

    @classmethod
    def from_dict(cls, raw: Mapping, *, path: str) -> "SplitConfig":
        check_unknown(raw, {"train", "valid", "seed"}, path)
        train = as_float(raw.get("train"), f"{path}.train")
        valid = as_float(raw.get("valid"), f"{path}.valid")
        seed = as_int(raw.get("seed", 0), f"{path}.seed")
        if train <= 0 or valid <= 0:
            raise ConfigError(path, "train and valid must be positive")
        if abs(train + valid - 1.0) > 1.0e-9:
            raise ConfigError(path, "train and valid must sum to 1")
        return cls(train=train, valid=valid, seed=seed)

    def fractions(self) -> tuple[float, float]:
        return (self.train, self.valid)


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    neighbor_cache: str = "none"

    @classmethod
    def from_dict(cls, raw: Mapping, *, path: str) -> "LoaderConfig":
        check_unknown(
            raw,
            {"batch_size", "shuffle", "num_workers", "pin_memory", "drop_last", "neighbor_cache"},
            path,
        )
        batch_size = as_int(raw.get("batch_size"), f"{path}.batch_size")
        if batch_size < 1:
            raise ConfigError(f"{path}.batch_size", "must be positive")
        num_workers = as_int(raw.get("num_workers", 0), f"{path}.num_workers")
        if num_workers < 0:
            raise ConfigError(f"{path}.num_workers", "must be non-negative")
        neighbor_cache = as_str(raw.get("neighbor_cache", "none"), f"{path}.neighbor_cache")
        if neighbor_cache not in {"none", "memory"}:
            raise ConfigError(f"{path}.neighbor_cache", "must be 'none' or 'memory'")
        return cls(
            batch_size=batch_size,
            shuffle=as_bool(raw.get("shuffle", True), f"{path}.shuffle"),
            num_workers=num_workers,
            pin_memory=as_bool(raw.get("pin_memory", False), f"{path}.pin_memory"),
            drop_last=as_bool(raw.get("drop_last", False), f"{path}.drop_last"),
            neighbor_cache=neighbor_cache,
        )

    def to_kwargs(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "neighbor_cache": self.neighbor_cache,
        }


@dataclass(frozen=True)
class DataConfig:
    format: str | None
    files: tuple[str, ...]
    split: SplitConfig | None
    train_files: tuple[str, ...]
    valid_files: tuple[str, ...]
    loader: LoaderConfig

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, path: str = "data", base_dir: Path | None = None
    ) -> "DataConfig":
        check_unknown(
            raw,
            {"format", "files", "split", "train_files", "valid_files", "loader"},
            path,
        )
        file_format = raw.get("format")
        if file_format is not None:
            file_format = as_str(file_format, f"{path}.format")

        loader_raw = raw.get("loader")
        if not isinstance(loader_raw, Mapping):
            raise ConfigError(f"{path}.loader", "must be an object")
        loader = LoaderConfig.from_dict(loader_raw, path=f"{path}.loader")

        files = raw.get("files")
        split = raw.get("split")
        train_files = raw.get("train_files")
        valid_files = raw.get("valid_files")

        has_split = files is not None or split is not None
        has_explicit = train_files is not None or valid_files is not None
        if has_split and has_explicit:
            raise ConfigError(
                path, "use either files+split or train_files/valid_files, not both"
            )

        if has_split:
            if files is None or split is None:
                raise ConfigError(path, "files and split must be provided together")
            resolved = tuple(
                resolve_path(item, base_dir)
                for item in _path_list(files, f"{path}.files")
            )
            split_config = SplitConfig.from_dict(split, path=f"{path}.split")
            return cls(file_format, resolved, split_config, (), (), loader)

        if has_explicit:
            if train_files is None or valid_files is None:
                raise ConfigError(
                    path, "train_files and valid_files must both be provided"
                )
            resolved_train = tuple(
                resolve_path(item, base_dir)
                for item in _path_list(train_files, f"{path}.train_files")
            )
            resolved_valid = tuple(
                resolve_path(item, base_dir)
                for item in _path_list(valid_files, f"{path}.valid_files")
            )
            return cls(file_format, (), None, resolved_train, resolved_valid, loader)

        raise ConfigError(path, "requires files+split or train_files/valid_files")


# ---------------------------------------------------------------------------
# Optimizers, learning-rate scheduling, training loops, and losses.
# ---------------------------------------------------------------------------

_OPTIMIZERS = {"adam", "adamw", "sgd"}
_SCHEDULERS = {"exponential", "exp", "exp_decay", "step", "cosine"}


def _freeze(value: object) -> object:
    """Recursively convert option lists to tuples and freeze dictionary values."""
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float = 0.0
    options: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping, *, path: str) -> "OptimizerConfig":
        if not isinstance(raw, Mapping):
            raise ConfigError(path, "must be an object")
        name = as_str(raw.get("name"), f"{path}.name").lower()
        if name not in _OPTIMIZERS:
            raise ConfigError(
                f"{path}.name", f"unsupported optimizer '{name}', available: {sorted(_OPTIMIZERS)}"
            )
        learning_rate = as_float(raw.get("learning_rate"), f"{path}.learning_rate")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ConfigError(f"{path}.learning_rate", "must be a positive finite number")
        weight_decay = as_float(raw.get("weight_decay", 0.0), f"{path}.weight_decay")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ConfigError(f"{path}.weight_decay", "must be a non-negative finite number")
        options = {
            key: _freeze(value)
            for key, value in raw.items()
            if key not in {"name", "learning_rate", "weight_decay"}
        }
        return cls(
            name=name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            options=options,
        )

    def to_kwargs(self) -> dict:
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            **self.options,
        }


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    start_lr: float
    limit_lr: float = 0.0
    decay_rate: float = 0.96
    decay_step: int = 100
    options: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping, *, path: str) -> "SchedulerConfig":
        if not isinstance(raw, Mapping):
            raise ConfigError(path, "must be an object")
        name = as_str(raw.get("name"), f"{path}.name").lower()
        if name not in _SCHEDULERS:
            raise ConfigError(
                f"{path}.name", f"unsupported scheduler '{name}', available: {sorted(_SCHEDULERS)}"
            )
        start_lr = as_float(raw.get("start_lr"), f"{path}.start_lr")
        if start_lr <= 0:
            raise ConfigError(f"{path}.start_lr", "must be positive")
        limit_lr = as_float(raw.get("limit_lr", 0.0), f"{path}.limit_lr")
        if limit_lr < 0 or limit_lr > start_lr:
            raise ConfigError(f"{path}.limit_lr", "must be in [0, start_lr]")
        decay_rate = as_float(raw.get("decay_rate", 0.96), f"{path}.decay_rate")
        if not 0 < decay_rate <= 1:
            raise ConfigError(f"{path}.decay_rate", "must be in (0, 1]")
        decay_step = as_int(raw.get("decay_step", 100), f"{path}.decay_step")
        if decay_step < 1:
            raise ConfigError(f"{path}.decay_step", "must be positive")
        options = {
            key: _freeze(value)
            for key, value in raw.items()
            if key not in {"name", "start_lr", "limit_lr", "decay_rate", "decay_step"}
        }
        return cls(
            name=name,
            start_lr=start_lr,
            limit_lr=limit_lr,
            decay_rate=decay_rate,
            decay_step=decay_step,
            options=options,
        )

    def to_kwargs(self) -> dict:
        return {
            "name": self.name,
            "start_lr": self.start_lr,
            "limit_lr": self.limit_lr,
            "decay_rate": self.decay_rate,
            "decay_step": self.decay_step,
            **self.options,
        }


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    epochs: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None
    gradient_clip_norm: float | None
    print_freq: int

    @classmethod
    def from_dict(cls, raw: Mapping, *, path: str = "training") -> "TrainingConfig":
        check_unknown(
            raw,
            {"device", "epochs", "optimizer", "scheduler", "gradient_clip_norm", "print_freq"},
            path,
        )
        device = as_str(raw.get("device", "auto"), f"{path}.device")
        epochs = as_int(raw.get("epochs"), f"{path}.epochs")
        if epochs < 1:
            raise ConfigError(f"{path}.epochs", "must be positive")

        optimizer_raw = raw.get("optimizer")
        if not isinstance(optimizer_raw, Mapping):
            raise ConfigError(f"{path}.optimizer", "must be an object")
        optimizer = OptimizerConfig.from_dict(optimizer_raw, path=f"{path}.optimizer")

        scheduler_raw = raw.get("scheduler")
        scheduler = (
            None
            if scheduler_raw is None
            else SchedulerConfig.from_dict(scheduler_raw, path=f"{path}.scheduler")
        )

        clip = raw.get("gradient_clip_norm")
        if clip is not None:
            clip = as_float(clip, f"{path}.gradient_clip_norm")
            if clip <= 0:
                raise ConfigError(f"{path}.gradient_clip_norm", "must be positive")

        print_freq = as_int(raw.get("print_freq", 1), f"{path}.print_freq")
        if print_freq < 1:
            raise ConfigError(f"{path}.print_freq", "must be positive")

        return cls(
            device=device,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_clip_norm=clip,
            print_freq=print_freq,
        )


@dataclass(frozen=True)
class LossConfig:
    metric: str
    label: tuple[str, ...]
    start_weight: tuple[float, ...]
    limit_weight: tuple[float, ...]

    @classmethod
    def from_dict(cls, raw: Mapping, *, path: str = "loss") -> "LossConfig":
        check_unknown(raw, {"metric", "label", "start_weight", "limit_weight"}, path)
        metric = as_str(raw.get("metric", "mse"), f"{path}.metric").lower()
        if metric not in {"mse", "rmse"}:
            raise ConfigError(f"{path}.metric", "must be 'mse' or 'rmse'")

        label = str_list(raw.get("label"), f"{path}.label")
        if not label:
            raise ConfigError(f"{path}.label", "must be a non-empty list")
        if len(set(label)) != len(label):
            raise ConfigError(f"{path}.label", "must not contain duplicate names")

        start_weight = float_list(raw.get("start_weight"), f"{path}.start_weight")
        limit_weight = float_list(raw.get("limit_weight"), f"{path}.limit_weight")
        if len(label) != len(start_weight) or len(label) != len(limit_weight):
            raise ConfigError(
                path, "label, start_weight and limit_weight must have equal lengths"
            )
        if any(
            not math.isfinite(weight) or weight < 0
            for weight in start_weight + limit_weight
        ):
            raise ConfigError(path, "loss weights must be finite and non-negative")

        return cls(metric=metric, label=label, start_weight=start_weight, limit_weight=limit_weight)

    def to_dict(self) -> dict:
        """Convert to the mapping expected by training/loss.py:calculate_loss."""
        return {
            "metric": self.metric,
            "label": list(self.label),
            "start_weight": list(self.start_weight),
            "limit_weight": list(self.limit_weight),
        }


# ---------------------------------------------------------------------------
# Output directories, checkpoints, and standard models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointConfig:
    save_every: int
    keep_best: bool
    monitor: str
    resume: str | None

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, path: str, base_dir: Path | None = None
    ) -> "CheckpointConfig":
        check_unknown(raw, {"save_every", "keep_best", "monitor", "resume"}, path)
        save_every = as_int(raw.get("save_every", 1), f"{path}.save_every")
        if save_every < 1:
            raise ConfigError(f"{path}.save_every", "must be positive")
        keep_best = as_bool(raw.get("keep_best", True), f"{path}.keep_best")
        monitor = as_str(raw.get("monitor", "valid_loss"), f"{path}.monitor")
        resume = raw.get("resume")
        if resume is not None:
            resume = resolve_path(as_str(resume, f"{path}.resume"), base_dir)
        return cls(save_every, keep_best, monitor, resume)


@dataclass(frozen=True)
class OutputConfig:
    dir: str
    checkpoint: CheckpointConfig
    model_filename: str

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, path: str = "output", base_dir: Path | None = None
    ) -> "OutputConfig":
        check_unknown(raw, {"dir", "checkpoint", "model_filename"}, path)
        directory = resolve_path(as_str(raw.get("dir"), f"{path}.dir"), base_dir)

        checkpoint_raw = raw.get("checkpoint", {})
        if not isinstance(checkpoint_raw, Mapping):
            raise ConfigError(f"{path}.checkpoint", "must be an object")
        checkpoint = CheckpointConfig.from_dict(
            checkpoint_raw, path=f"{path}.checkpoint", base_dir=base_dir
        )

        filename = as_str(raw.get("model_filename", "model.pt"), f"{path}.model_filename")
        reserved = {"last.pt", "best.pt", "config.json", "data.json", "loss.out",
                    "loss.png", "parity.png"}
        if (not filename or Path(filename).name != filename or filename in reserved
                or filename.startswith("epoch_") or filename in {".", ".."}):
            raise ConfigError(f"{path}.model_filename", "must be a plain filename distinct from training artifacts")
        return cls(dir=directory, checkpoint=checkpoint, model_filename=filename)


# ---------------------------------------------------------------------------
# Root configuration.
# ---------------------------------------------------------------------------

_TOP_LEVEL = {"config_version", "model_modules", "common", "model", "data", "training", "loss", "output"}


def load_modules(value=()) -> tuple[str, ...]:
    """Load builtin and configured extension modules, returning validated module names.
    
    Use normal Python imports without scanning directories. Callers such as load_config add the
    configuration directory to sys.path. Repeated imports use Python's cache and do not repeat
    registration.
    """
    modules = str_list(value, "model_modules")
    for index, name in enumerate(modules):
        if not name or any(not part.isidentifier() for part in name.split(".")):
            raise ConfigError(f"model_modules[{index}]", "expected a dotted Python module name")
    if len(set(modules)) != len(modules):
        raise ConfigError("model_modules", "duplicate module names")
    import_module("abfml.builtin")
    for index, name in enumerate(modules):
        try:
            import_module(name)
        except ImportError as error:
            raise ConfigError(f"model_modules[{index}]", f"cannot import '{name}': {error}") from error
    return modules


@dataclass(frozen=True)
class RootConfig:
    """A complete parsed training configuration."""

    config_version: int
    common: CommonConfig
    model_name: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    output: OutputConfig
    model_modules: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, base_dir: Path | None = None
    ) -> "RootConfig":
        unknown = sorted(key for key in raw if key not in _TOP_LEVEL)
        if unknown:
            raise ConfigError("config", f"unknown top-level field(s): {unknown}")

        config_version = as_int(raw.get("config_version"), "config_version")
        if config_version != 1:
            raise ConfigError(
                "config_version", f"unsupported version {config_version}, expected 1"
            )

        common_raw = raw.get("common")
        if not isinstance(common_raw, Mapping):
            raise ConfigError("common", "must be an object")
        common = CommonConfig.from_dict(common_raw)

        model_raw = raw.get("model")
        if not isinstance(model_raw, Mapping):
            raise ConfigError("model", "must be an object")
        model_name = as_str(model_raw.get("name"), "model.name")
        # Use the configuration directory as the import base, matching data paths,
        # so the console command can load colocated model extensions.
        if base_dir is not None:
            import sys

            if str(base_dir) not in sys.path:
                sys.path.insert(0, str(base_dir))
        model_modules = load_modules(raw.get("model_modules", ()))
        # Parse model options by registered name after extension modules register themselves.
        from abfml.core.registry import parse_options

        model = parse_options(model_raw, type_map=common.type_map)

        data_raw = raw.get("data")
        if not isinstance(data_raw, Mapping):
            raise ConfigError("data", "must be an object")
        data = DataConfig.from_dict(data_raw, base_dir=base_dir)

        training_raw = raw.get("training")
        if not isinstance(training_raw, Mapping):
            raise ConfigError("training", "must be an object")
        training = TrainingConfig.from_dict(training_raw)

        loss_raw = raw.get("loss")
        if not isinstance(loss_raw, Mapping):
            raise ConfigError("loss", "must be an object")
        loss = LossConfig.from_dict(loss_raw)

        output_raw = raw.get("output")
        if not isinstance(output_raw, Mapping):
            raise ConfigError("output", "must be an object")
        output = OutputConfig.from_dict(output_raw, base_dir=base_dir)

        return cls(
            config_version=config_version,
            common=common,
            model_name=model_name,
            model=model,
            data=data,
            training=training,
            loss=loss,
            output=output,
            model_modules=model_modules,
        )

    def model_kwargs(self) -> dict:
        """User constructor hyperparameters; the training workflow supplies statistics separately."""
        return self.model.model_kwargs()


def load_config(path: str | Path) -> RootConfig:
    """Read JSON and parse a typed root configuration.
    
    Resolve relative paths against the configuration directory. Preserve missing-file and JSON
    syntax exceptions; semantic field errors raise ConfigError.
    """
    path = Path(path)
    base_dir = path.parent.resolve()
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ConfigError("config", "top-level JSON must be an object")
    return RootConfig.from_dict(raw, base_dir=base_dir)


__all__ = [
    "ConfigError",
    "check_unknown",
    "as_str",
    "as_bool",
    "as_int",
    "as_float",
    "int_list",
    "float_list",
    "str_list",
    "resolve_path",
    "parse_network",
    "CheckpointConfig",
    "CommonConfig",
    "DataConfig",
    "LoaderConfig",
    "LossConfig",
    "OptimizerConfig",
    "OutputConfig",
    "RootConfig",
    "SchedulerConfig",
    "SplitConfig",
    "TrainingConfig",
    "load_config",
    "load_modules",
]
