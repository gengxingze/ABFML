import torch
import torch.nn as nn
from typing import List, Tuple


class ActivationModule(nn.Module):
    def __init__(self, activation_name: str):
        super(ActivationModule, self).__init__()
        activation_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
            'softsign': nn.Softsign(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'prelu': nn.PReLU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'celu': nn.CELU(),
            'glu': nn.GLU(),
            'logsigmoid': nn.LogSigmoid(),
            'rrelu': nn.RReLU(),
            'hardshrink': nn.Hardshrink(),
            'hardtanh': nn.Hardtanh(),
            'softshrink': nn.Softshrink(),
            'tanhshrink': nn.Tanhshrink()
        }
        self.activation = activation_functions.get(activation_name, None)

    def forward(self, x):
        return self.activation(x)


@torch.jit.script
def smooth_fun(smooth_type: str, rij: torch.Tensor, r_inner: float, r_outer: float) -> torch.Tensor:
    fx = torch.zeros_like(rij, dtype=rij.dtype, device=rij.device)
    if smooth_type == 'cos':
        mask = (rij > 1e-5) & (rij < r_outer)
        fx[mask] = (0.5 * torch.cos(torch.pi * rij[mask] / r_outer) + 0.5)
    elif smooth_type == 'cos_r':
        fx[(rij > 1e-5) & (rij < r_inner)] = 1 / rij[(rij > 1e-5) & (rij < r_inner)]
        mask = (rij > r_inner) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = (0.5 * torch.cos(torch.pi * x) + 0.5) / rij[mask]
    elif smooth_type == 'tanh_u':
        fx[(rij > 1e-5) & (rij < r_inner)] = 1 / rij[(rij > 1e-5) & (rij < r_inner)]
        mask = (rij > r_inner) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = torch.tanh(1 - x) ** 3
    elif smooth_type == 'exp':
        mask = (rij > 1e-5) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = torch.exp(1-1/(1-torch.square(x)))
    elif smooth_type == 'poly1':
        mask = (rij > 1e-5) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = ((2 * x - 3) * (x ** 2) + 1)
    elif smooth_type == 'poly2':
        mask = (rij > 1e-5) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = ((-6 * x ** 2 + 15 * x - 10) * (x ** 3) + 1)
    elif smooth_type == 'poly3':
        mask = (rij > 1e-5) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = ((20 * x ** 3 - 70 * x ** 2 + 84 * x - 35) * (x ** 4) + 1)
    elif smooth_type == 'poly4':
        mask = (rij > 1e-5) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = ((- 70 * x ** 4 + 315 * x ** 3 - 540 * x ** 2 + 420 * x - 126) * (x ** 5) + 1)
    elif smooth_type == 'poly1_r':
        fx[(rij > 1e-5) & (rij < r_inner)] = 1 / rij[(rij > 1e-5) & (rij < r_inner)]
        mask = (rij > r_inner) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = ((2 * x - 3) * (x ** 2) + 1) / rij[mask]
    elif smooth_type == 'poly2_r':
        fx[(rij > 1e-5) & (rij < r_inner)] = 1 / rij[(rij > 1e-5) & (rij < r_inner)]
        mask = (rij > r_inner) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = ((-6 * x ** 2 + 15 * x - 10) * (x ** 3) + 1) / rij[mask]
    elif smooth_type == 'poly3_r':
        fx[(rij > 1e-5) & (rij < r_inner)] = 1 / rij[(rij > 1e-5) & (rij < r_inner)]
        mask = (rij > r_inner) & (rij < r_outer)
        x = (rij[mask] - r_inner) / (r_outer - r_inner)
        fx[mask] = ((20 * x ** 3 - 70 * x ** 2 + 84 * x - 35) * (x ** 4) + 1) / rij[mask]
    else:
        raise KeyError(f'Undefined smooth types {smooth_type}')
    return fx


@torch.jit.script
def polynomial_fun(fun_name: str, n: int, rij: torch.Tensor, r_inner: float, r_outer: float) -> torch.Tensor:
    shape = list(rij.shape[:-1]) + [n + 1]
    fx = torch.zeros(shape, dtype=rij.dtype, device=rij.device)
    if n < 2:
        raise ValueError('n must be greater than 2')
    if fun_name == 'chebyshev':
        mask = (rij > r_inner) & (rij < r_outer)
        x = 2 * (rij[mask] - r_inner) / (r_outer - r_inner) - 1
        fx[..., 0:1][mask] = 1
        fx[..., 1:2][mask] = x
        for i in range(1, n + 1):
            fx_temp_1 = (2 * fx[..., 1]).clone()
            fx_temp_2 = fx[..., i - 1].clone()
            fx_temp_3 = fx[..., i - 2].clone()
            fx[..., i] = fx_temp_1 * fx_temp_2 - fx_temp_3
    return fx


#@torch.jit.script
def derive_mechanics(
    grad_Ei: torch.Tensor,
    neighbor_vectors: torch.Tensor,
    neighbor_indices: torch.Tensor,
    n_ghost: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n_atoms, max_neighbors = neighbor_indices.shape
    dtype, device = neighbor_vectors.dtype, neighbor_vectors.device

    # Initialize force and per-atom virial tensors
    force = torch.zeros(batch, n_atoms + n_ghost, 3, dtype=dtype, device=device)
    force[:, :n_atoms, :] = grad_Ei.sum(dim=-2)  # Direct contribution from central atoms

    atomic_virial = torch.zeros(batch, n_atoms + n_ghost, 9, dtype=dtype, device=device)

    # Extract relative position vectors (exclude norm)
    rel_pos = neighbor_vectors[..., 1:]  # Shape: [batch, n_atoms, max_neighbors, 3]

    # Compute local virial contributions via outer product of dr and dE
    local_virials = torch.matmul(rel_pos.unsqueeze(-1), -1 * grad_Ei.unsqueeze(-2)).reshape(batch, n_atoms, max_neighbors, 9)

    # Replace invalid neighbor indices (-1) with 0
    neighbor_indices[neighbor_indices == -1] = 0

    for b in range(batch):
        # Flatten indices and contribution values
        neighbor_idx_b = neighbor_indices[b].view(-1).to(torch.int64)

        # Force scatter-add
        force_contrib_b = -1 * grad_Ei[b].view(-1, 3)
        index_f = neighbor_idx_b.unsqueeze(-1).expand(-1, 3)
        force[b] = force[b].scatter_add(0, index_f, force_contrib_b)

        # Virial scatter-add
        virial_contrib_b = local_virials[b].view(-1, 9)
        index_v = neighbor_idx_b.unsqueeze(-1).expand(-1, 9)
        atomic_virial[b] = atomic_virial[b].scatter_add(0, index_v, virial_contrib_b)

    # Compute total virial by summing over all atoms
    virial = atomic_virial.sum(dim=1)

    return force, virial, atomic_virial

