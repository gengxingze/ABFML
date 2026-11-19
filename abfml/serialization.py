"""Save and load standard model files without compilation or deployment export."""

import importlib.util
import sys
from pathlib import Path

import torch

from abfml.config import CommonConfig, ConfigError, load_modules
from abfml.core.registry import build_model, model_spec, parse_options
from abfml.core.model import AtomicModel


def module_files(config) -> dict[str, str]:
    """Record loaded extension-module absolute paths for fallback lookup during loading."""
    files: dict[str, str] = {}
    for name in config.get("model_modules", ()):
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        if path:
            files[name] = str(Path(path).resolve())
    return files


def _load_model_modules(config, module_files_map=None) -> None:
    """Import extension modules normally, falling back to recorded file paths when missing.
    
    Fallback applies only to modules unavailable through normal imports. Preserve the original
    ConfigError if recorded paths are unavailable, for example after moving machines.
    """
    modules = config.get("model_modules", ())
    try:
        load_modules(modules)
    except ConfigError:
        recorded = module_files_map or {}
        missing = [name for name in modules if name not in sys.modules]
        if not all(Path(recorded.get(name, "")).is_file() for name in missing):
            raise
        for name in missing:
            spec = importlib.util.spec_from_file_location(name, recorded[name])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)


def save_model(path, model: AtomicModel, config: dict) -> None:
    """Save inference configuration and state without training data paths, optimizers, or Python
    objects.
    """
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.get_default_dtype()
    payload = {
        "format": "abfml.model", "version": 1,
        "config": {"common": {**config["common"], "dtype": str(dtype).split('.')[-1]},
                   "model": config["model"],
                   "model_modules": list(config.get("model_modules", ()))},
        "model_module_files": module_files(config),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(source, *, device="cpu", dtype=None) -> AtomicModel:
    """Load a standard model or current-format checkpoint, restoring statistics from the state
    dictionary.
    """
    if isinstance(source, AtomicModel):
        model = source
    else:
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") not in {
            "abfml.model", "abfml.checkpoint"
        } or payload.get("version") != 1:
            raise ValueError("Unsupported ABFML model format")
        config = payload["config"]
        common = CommonConfig.from_dict(config["common"])
        _load_model_modules(config, payload.get("model_module_files"))
        model_config = parse_options(config["model"], type_map=common.type_map)
        kwargs = model_config.model_kwargs()
        required = getattr(model_spec(config["model"]["name"]).builder, "required_statistics", ())
        if "energy_shift" in required:
            kwargs["energy_shift"] = [0.] * len(common.type_map)
        if "std_mean" in required:
            kwargs["std_mean"] = None
        if "q_scaler" in required:
            kwargs["q_scaler"] = None
        model = build_model(config["model"]["name"], **kwargs)
        model.to(dtype=getattr(torch, common.dtype))
        model.load_state_dict(payload["model_state"], strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model
