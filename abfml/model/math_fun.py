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
def calculate_distance_grad(Ri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, n_atoms, max_neighbor, _ = Ri.shape
    device = Ri.device
    dtype = Ri.dtype
    dRi = torch.zeros(batch, n_atoms, max_neighbor, 4, 3, dtype=dtype, device=device)
    rr = torch.zeros(batch, n_atoms, max_neighbor, dtype=dtype, device=device)
    mask = (Ri[..., 0] > 1e-5)
    rr[mask] = 1 / Ri[..., 0][mask]
    dRi[..., 0, 0] = Ri[..., 1] * rr
    dRi[..., 0, 1] = Ri[..., 2] * rr
    dRi[..., 0, 2] = Ri[..., 3] * rr
    dRi[..., 1, 0] = 1
    dRi[..., 2, 1] = 1
    dRi[..., 3, 2] = 1
    return Ri, dRi


@torch.jit.script
def compute_forces_and_virial(dE_Rid: torch.Tensor,
                              Rij: torch.Tensor,
                              Nij: torch.Tensor,
                              n_ghost: int,
                              dtype: torch.dtype,
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n_atoms, max_neighbor = Nij.shape

    # Initialize Force and Virial tensors
    Force = torch.zeros(batch, n_atoms + n_ghost, 3, dtype=dtype, device=device)
    Force[:, :n_atoms, :] = -1.0 * dE_Rid.sum(dim=-2)
    virial = torch.zeros(batch, n_atoms + n_ghost, 9, dtype=dtype, device=device)

    # Compute relative positions (rxyz) and virial contributions (virial_ij)
    rxyz = Rij[:, :, :, 1:]
    virial_ij = -1.0 * torch.matmul(rxyz.unsqueeze(-1), dE_Rid.unsqueeze(-2)).reshape(batch, n_atoms, max_neighbor, 9)

    # Here only the action force F_ij = -1 * sum(de_ij * dr_ij) is considered, and the reaction force
    # -F_ji = sum(de_ji * dr_ji) should also be considered and subtracted from.
    # Finally, F_ij = - 1*sum(de_ij * dr_ij) + sum(de_ji * dr_ji)
    # for bb in range(0, batch):
    #     for ii in range(0, n_atoms + n_ghost):
    #         Force[bb, ii] = Force[bb, ii] + dE_Rid[bb][Nij[bb] == ii].sum(dim=0)

    # Replace invalid neighbor indices with 0
    Nij[Nij == -1] = 0

    # Scatter-add for forces and virial
    for bb in range(batch):
        indices_f = Nij[bb].squeeze(dim=0).to(torch.int64).view(-1).unsqueeze(-1).expand(-1, 3)
        values_f = dE_Rid[bb].squeeze(dim=0).view(-1, 3)
        Force[bb] = Force[bb].scatter_add(0, indices_f, values_f).reshape(n_atoms + n_ghost, 3)

        indices_v = Nij[bb].squeeze(dim=0).to(torch.int64).view(-1).unsqueeze(-1).expand(-1, 9)
        values_v = virial_ij[bb].squeeze(dim=0).view(-1, 9)
        virial[bb] = virial[bb].scatter_add(0, indices_v, values_v).reshape(n_atoms + n_ghost, 9)

    # Sum Virial tensor contributions
    Virial = virial.sum(dim=-2)

    return Force, Virial, virial

