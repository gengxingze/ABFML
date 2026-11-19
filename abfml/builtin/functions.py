"""Shared tensor operations and mathematical functions for models."""

import math
import torch


def type_lookup(type_map: list[int]) -> torch.Tensor:
    """Map atomic numbers to contiguous model type indices."""
    if not type_map:
        raise ValueError("type_map cannot be empty")
    lut = torch.full((max(type_map) + 1,), -1, dtype=torch.long)
    lut[torch.tensor(type_map, dtype=torch.long)] = torch.arange(len(type_map))
    return lut


def segment_sum(edge_values: torch.Tensor, edge_ptr: torch.Tensor) -> torch.Tensor:
    """Sum one value per edge into center slots without assuming center IDs."""
    n_centers = edge_ptr.numel() - 1
    degrees = edge_ptr[1:] - edge_ptr[:-1]
    center_slots = torch.repeat_interleave(
        torch.arange(n_centers, device=edge_values.device),
        degrees,
    )
    return torch.zeros(
        n_centers,
        dtype=edge_values.dtype,
        device=edge_values.device,
    ).index_add(0, center_slots, edge_values)


def safe_reciprocal(distances: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return 1/r on valid nonzero edges and zero elsewhere."""
    nonzero = valid & (distances > 1.0e-12)
    safe_distances = torch.where(nonzero, distances, torch.ones_like(distances))
    return torch.where(nonzero, 1.0 / safe_distances, torch.zeros_like(distances))


def _cosine(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.cos(math.pi * x) + 0.5


def _tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(1.0 - x).pow(3)


def _exponential(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(1.0 - 1.0 / (1.0 - x.square()))


def _poly1(x: torch.Tensor) -> torch.Tensor:
    return (2.0 * x - 3.0) * x.square() + 1.0


def _poly2(x: torch.Tensor) -> torch.Tensor:
    return (-6.0 * x.square() + 15.0 * x - 10.0) * x.pow(3) + 1.0


def _poly3(x: torch.Tensor) -> torch.Tensor:
    return (20.0 * x.pow(3) - 70.0 * x.square() + 84.0 * x - 35.0) * x.pow(4) + 1.0


def _poly4(x: torch.Tensor) -> torch.Tensor:
    return (-70.0 * x.pow(4) + 315.0 * x.pow(3) - 540.0 * x.square()
            + 420.0 * x - 126.0) * x.pow(5) + 1.0


# Single source of names: base function, inner 1/r, transition 1/r, and zero-origin normalization.
# Configuration validation reads this mapping rather than maintaining a separate name list.
SMOOTH_FUNCTIONS = {
    "cos": (_cosine, False, False, True),
    "cos_r": (_cosine, True, True, False),
    "tanh_u": (_tanh, True, False, False),
    "exp": (_exponential, False, False, False),
    "poly1": (_poly1, False, False, False),
    "poly1_r": (_poly1, True, True, False),
    "poly2": (_poly2, False, False, False),
    "poly2_r": (_poly2, True, True, False),
    "poly3": (_poly3, False, False, False),
    "poly3_r": (_poly3, True, True, False),
    "poly4": (_poly4, False, False, False),
}


def cutoff_function(
    smooth_type: str,
    distances: torch.Tensor,
    r_inner: float,
    r_outer: float,
) -> torch.Tensor:
    """Return the configured radial cutoff, preserving ABFML conventions.

    ``*_r`` variants approach ``1/r`` below ``r_inner`` and multiply the
    switching region by ``1/r``. ``tanh_u`` has the same inner region but an
    unweighted tanh switching region. Values outside the support are zero.
    """
    if smooth_type not in SMOOTH_FUNCTIONS:
        raise ValueError("Undefined cutoff function: " + smooth_type)
    envelope, inner_reciprocal, transition_reciprocal, from_zero = SMOOTH_FUNCTIONS[smooth_type]

    values = torch.zeros_like(distances)
    if from_zero:
        x = distances / r_outer
    else:
        x = (distances - r_inner) / (r_outer - r_inner)

    if inner_reciprocal:
        inner = (distances > 1.0e-5) & (distances < r_inner)
        transition = (distances > r_inner) & (distances < r_outer)
        values[inner] = 1.0 / distances[inner]
        selected = envelope(x[transition])
        if transition_reciprocal:
            selected = selected / distances[transition]
        values[transition] = selected
    else:
        valid = (distances > 1.0e-5) & (distances < r_outer)
        values[valid] = envelope(x[valid])
    return values
