"""Model construction registry, independent of forward computation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from abfml.core.model import AtomicModel


class ModelConfig(Protocol):
    """Shared model configuration contract: expose a cutoff and constructor arguments."""

    @property
    def cutoff(self) -> float: ...

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, type_map: tuple[int, ...], path: str = "model"
    ) -> ModelConfig: ...

    def model_kwargs(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ModelSpec:
    """A model builder and optional configuration parser registered under one name.
    
    Keeping both together avoids parallel mappings. Pure Python construction may omit config;
    JSON configuration requires a parser.
    """

    builder: Callable[..., AtomicModel]
    config: type[ModelConfig] | None = None

    def build(self, *args: Any, **kwargs: Any) -> AtomicModel:
        model = self.builder(*args, **kwargs)
        if not isinstance(model, AtomicModel):
            raise TypeError(
                f"Model builder returned {type(model).__name__}, expected AtomicModel"
            )
        return model


class ModelRegistry:
    """Explicit name-to-ModelSpec registry that rejects silent replacement by default."""

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def register(self, name: str, spec: ModelSpec) -> None:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("Model name must be a non-empty string without surrounding spaces")
        if not isinstance(spec, ModelSpec):
            raise TypeError("spec must be a ModelSpec")
        if name in self._specs:
            raise ValueError(f"Model name is already registered: {name}")
        self._specs[name] = spec

    def get(self, name: str) -> ModelSpec:
        if name not in self._specs:
            available = ", ".join(self.names) or "<empty>"
            raise KeyError(f"Unknown model '{name}'. Available models: {available}")
        return self._specs[name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> AtomicModel:
        """Build a registered model, passing arguments unchanged to its builder."""
        return self.get(name).build(*args, **kwargs)


MODEL_REGISTRY = ModelRegistry()


def register_model(name: str, spec: ModelSpec) -> None:
    """Register a model globally; reject duplicate names explicitly."""
    MODEL_REGISTRY.register(name, spec)


def model_spec(name: str) -> ModelSpec:
    return MODEL_REGISTRY.get(name)


def build_model(name: str, *args: Any, **kwargs: Any) -> AtomicModel:
    """Build a model through the global registry."""
    return MODEL_REGISTRY.create(name, *args, **kwargs)


def parse_options(
    raw: Mapping, *, type_map: tuple[int, ...]
) -> ModelConfig:
    """Parse model options using the registered configuration class."""
    # Import only for parsing; registration and construction do not depend on application configuration.
    from abfml.config import ConfigError, as_str

    if not isinstance(raw, Mapping):
        raise ConfigError("model", "must be an object")
    name = as_str(raw.get("name"), "model.name")
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.names))
        raise ConfigError("model.name", f"unknown model '{name}', available: {available}")
    config_class = MODEL_REGISTRY.get(name).config
    if config_class is None:
        raise ConfigError("model.name", f"model '{name}' has no configuration parser")
    return config_class.from_dict(raw, type_map=type_map, path="model")
