"""MLFF analysis entry points; import plotting libraries only when needed."""

from importlib import import_module

_EXPORTS = {
    "plot_loss": "curves",
    "read_labels": "labels",
    "plot_labels": "labels",
    "plot_distributions": "labels",
    "plot_predictions": "parity",
    "export_pairs": "parity",
    "read_pairs": "parity",
    "plot_neighbors": "coordination",
    "plot_descriptors": "projection",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
