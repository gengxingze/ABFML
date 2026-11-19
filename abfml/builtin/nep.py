"""NEP4 descriptors, fitting networks, and atomic energy models.

Following GPUMD and Chapter 5 of Molecular Dynamics Simulation:
1. Compute Chebyshev radial bases f_k(r_ij) on CSR edges.
2. Form g_n(r_ij) using trainable species-pair coefficients c_nk^{IJ}.
3. Sum radial descriptors q_n^i=sum_j g_n(r_ij).
4. Aggregate real angular moments S_nlm using unit-direction bases.
5. Contract moments into three-, four-, and five-body invariants.
6. Multiply descriptors by q_scaler and apply species-specific tanh MLPs.

Implemented channels are radial q_n, three-body q_nl, four-body q_222/q_112, and five-body
q_1111. Nonzero q_123/q_233/q_134 settings raise errors; ZBL is not implemented. Multiple hidden
layers are a framework extension; network_size=[H] matches the standard GPUMD architecture.

GPUMD/TorchNEP conventions determine Chebyshev bases, real-basis component order, high-body
contraction coefficients, multiplicative scaling without centering, and shared output bias.

References: Z. Fan, Molecular Dynamics Simulation, Eqs. (5.7)-(5.11), (5.34)-(5.59),
(5.78)-(5.79); GPUMD NEP4 documentation; Z. Fan et al., Phys. Rev. B 104, 104309 (2021);
GPUMD/TorchNEP high-body contractions.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from abfml.config import ConfigError, as_float, as_int, int_list, check_unknown
from abfml.core.registry import ModelSpec, register_model
from abfml.core.model import AtomicModel
from abfml.core.types import ModelInput, ModelOutput
from abfml.builtin.networks import SpeciesLinear
from abfml.builtin.functions import type_lookup


# These constants follow GPUMD's unnormalized real-harmonic contraction coefficients;
# their order must match the basis components. C3B covers L=1..4 three-body terms;
# the remaining arrays cover q_222, q_1111, and q_112, respectively.
C3B = [
    0.238732414637843, 0.119366207318922, 0.119366207318922,
    0.099471839432435, 0.596831036594608, 0.596831036594608,
    0.149207759148652, 0.149207759148652,
    0.139260575205408, 0.104445431404056, 0.104445431404056,
    1.044454314040563, 1.044454314040563, 0.174075719006761,
    0.174075719006761, 0.011190581936149, 0.223811638722978,
    0.223811638722978, 0.111905819361489, 0.111905819361489,
    1.566681471060845, 1.566681471060845, 0.195835183882606,
    0.195835183882606,
]
C4B = [
    -0.007499480826664,
    -0.134990654879954,
    0.067495327439977,
    0.404971964639861,
    -0.809943929279723,
]
C5B = [0.026596810706114, 0.053193621412227, 0.026596810706114]
C4B2 = [
    0.027493550848847,
    0.164961305093080,
    -0.013746775424423,
    0.041240326273270,
    0.082480652546540,
]


class NepDescriptorRange(nn.Module):
    """Accumulate per-channel NEP descriptor minima and maxima across batches."""

    def __init__(self, descriptor_dim: int) -> None:
        super().__init__()
        if descriptor_dim < 1:
            raise ValueError("descriptor_dim must be positive")
        self.descriptor_dim = int(descriptor_dim)
        self.register_buffer("minimum", torch.full((descriptor_dim,), math.inf))
        self.register_buffer("maximum", torch.full((descriptor_dim,), -math.inf))
        self.register_buffer("num_samples", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, descriptors: torch.Tensor) -> None:
        if descriptors.ndim != 2 or descriptors.shape[1] != self.descriptor_dim:
            raise ValueError("descriptors must have shape [N, descriptor_dim]")
        if descriptors.shape[0] == 0:
            return
        values = descriptors.detach()
        self.minimum.copy_(torch.minimum(self.minimum, values.min(dim=0).values))
        self.maximum.copy_(torch.maximum(self.maximum, values.max(dim=0).values))
        self.num_samples.add_(values.shape[0])

    @torch.no_grad()
    def finalize(self) -> torch.Tensor:
        """Return q_scaler=1/(max-min), protecting degenerate ranges with 1e-12."""
        if int(self.num_samples) == 0:
            raise ValueError("Cannot compute q_scaler from an empty dataset")
        return 1.0 / (self.maximum - self.minimum).clamp(min=1.0e-12)


def radial_basis(
    distances: torch.Tensor,
    cutoff: float,
    basis_size: int,
) -> torch.Tensor:
    """Compute GPUMD NEP Chebyshev radial bases f_k(r).
    
    For r=distances and rc=cutoff, let x=2*(r/rc-1)^2-1. Use T_0=1, T_1=x, and T_(k+1)=2*x*T_k-
    T_(k-1). Return f_k=[T_k+1]*fc/2 with fc=[cos(pi*r/rc)+1]/2.
    
    There are basis_size+1 channels, k=0..basis_size, with shape distances.shape+[basis_size+1].
    Callers must filter edges outside the cutoff first.
    """
    ratio = distances / cutoff
    cutoff_values = 0.5 * (torch.cos(math.pi * ratio) + 1.0)
    x = 2.0 * (ratio - 1.0).square() - 1.0
    polynomials = [torch.ones_like(distances)]
    if basis_size >= 1:
        polynomials.append(x)
    for _order in range(2, basis_size + 1):
        polynomials.append(2.0 * x * polynomials[-1] - polynomials[-2])
    values = torch.stack(polynomials, dim=-1)
    return 0.5 * (values + 1.0) * cutoff_values.unsqueeze(-1)


def angular_basis(
    unit_vectors: torch.Tensor,
    l_max: int,
) -> torch.Tensor:
    """Compute real angular bases b_lm(r_hat), Eqs. (5.36)-(5.59).
    
    Input unit_vectors=r_hat=r_vec/r. Since b_lm(r_vec) is homogeneous of degree l,
    b_lm(r_hat)=b_lm(r_vec)/r^l. The unit-direction form avoids an additional explicit r^l
    factor when computing S_nlm in descriptor().
    
    Concatenate L blocks excluding L=0: L=1 gives (z,x,y); L=2 gives (3z^2-1,xz,yz,x^2-y^2,2xy);
    L=3/4 add seven/nine terms from Eqs. (5.44)-(5.59). Input is [E,3], output
    [E,(l_max+1)^2-1]. Component order and unnormalized coefficients must match C3B/C4B/C5B/C4B2
    to reproduce GPUMD/TorchNEP contractions.
    """
    if l_max < 1:
        return unit_vectors.new_zeros(unit_vectors.shape[0], 0)
    if l_max > 4:
        raise ValueError("This NEP implementation supports l_max <= 4")
    x = unit_vectors[:, 0]
    y = unit_vectors[:, 1]
    z = unit_vectors[:, 2]
    x2 = x * x
    y2 = y * y
    z2 = z * z
    x2my2 = x2 - y2
    components = [z, x, y]
    if l_max >= 2:
        components.extend([3.0 * z2 - 1.0, x * z, y * z, x2my2, 2.0 * x * y])
    if l_max >= 3:
        components.extend([
            (5.0 * z2 - 3.0) * z,
            (5.0 * z2 - 1.0) * x,
            (5.0 * z2 - 1.0) * y,
            x2my2 * z,
            2.0 * x * y * z,
            x * (x2 - 3.0 * y2),
            y * (3.0 * x2 - y2),
        ])
    if l_max >= 4:
        components.extend([
            (35.0 * z2 - 30.0) * z2 + 3.0,
            (7.0 * z2 - 3.0) * x * z,
            (7.0 * z2 - 3.0) * y * z,
            (7.0 * z2 - 1.0) * x2my2,
            (7.0 * z2 - 1.0) * 2.0 * x * y,
            z * x * (x2 - 3.0 * y2),
            z * y * (3.0 * x2 - y2),
            x2my2 * x2my2 - 4.0 * x2 * y2,
            4.0 * x * y * x2my2,
        ])
    return torch.stack(components, dim=-1)


class NepFittingNetwork(nn.Module):
    """NEP fitting networks selected by center species.
    
    Inputs descriptors=[C,D] and species=[C] produce atomic energies [C]. SpeciesLinear packs
    species parameters to avoid repeated small masked networks; each atom uses only its own
    species weights.
    
    network_size=[H] matches E_i=sum_h w_h^(1)*tanh(sum_d w_hd^(0)*q_d-b_h^(0)); additional
    widths extend this architecture. Hidden layers use tanh(xW-b). The output layer has no
    species bias; NepModel handles the shared output_bias.
    """

    def __init__(
        self,
        num_types: int,
        descriptor_dim: int,
        network_size: List[int],
    ):
        super().__init__()
        if num_types < 1 or descriptor_dim < 1:
            raise ValueError("num_types and descriptor_dim must be positive")
        if not network_size or any(width < 1 for width in network_size):
            raise ValueError("network_size must contain at least one positive width")
        self.network_size = [descriptor_dim] + network_size
        self.layers = nn.ModuleList()
        for index, width in enumerate(network_size):
            layer = SpeciesLinear(
                num_types,
                self.network_size[index],
                width,
                bias=True,
                bias_sign=-1.0,
            )
            layer.bias.data.zero_()
            self.layers.append(layer)
        self.output_layer = SpeciesLinear(
            num_types,
            network_size[-1],
            1,
            bias=False,
        )

    def forward(self, descriptors: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        outputs = descriptors
        for layer in self.layers:
            outputs = torch.tanh(layer(outputs, species))
        return self.output_layer(outputs, species).reshape(-1)


class NepDescriptorScaler(nn.Module):
    """Apply NEP's per-channel multiplicative descriptor scaling.
    
    q_scaler[d]=1/(q_max[d]-q_min[d]); forward computes q*q_scaler without subtracting a mean or
    minimum. Clamp range denominators to 1e-12 to avoid division by zero.
    """

    def __init__(self, descriptor_dim: int, q_scaler: Optional[torch.Tensor] = None):
        super().__init__()
        if q_scaler is None:
            q_scaler = torch.ones(descriptor_dim)
        if q_scaler.shape != (descriptor_dim,):
            raise ValueError("q_scaler must have shape [descriptor_dim]")
        self.register_buffer("q_scaler", q_scaler.detach().clone())

    def set_range(self, q_min: torch.Tensor, q_max: torch.Tensor) -> None:
        if q_min.shape != self.q_scaler.shape or q_max.shape != self.q_scaler.shape:
            raise ValueError("Descriptor ranges must match q_scaler")
        self.q_scaler.copy_(1.0 / (q_max - q_min).clamp(min=1.0e-12))

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return descriptors * self.q_scaler


class NepModel(AtomicModel):
    """Complete NEP atomic energy model with descriptors, scaling, and species fitting networks.
    
    Descriptors operate directly on CSR edges, aggregating with index_add without dense neighbor
    padding. For T species, N=n_max+1, and K=basis_size+1, coefficient shapes are
    [T,T,N_radial,K_radial] and [T,T,N_angular,K_angular]. Radial descriptors are [C,N_radial];
    three-body channels use (L,n) order and higher-body channels use n order. Total width is
    N_radial+N_angular*(L_max+enabled higher-body blocks).
    
    Block order is radial, three-body, optional q_222, optional q_1111, and optional q_112;
    q_scaler and fitting columns follow this order. compute() forms raw q, scales it, applies
    the center-species MLP, and subtracts a shared trainable output_bias. num_neurons=H and
    network_size=[H] are equivalent but mutually exclusive. ZBL and unsupported higher-body
    channels are not implemented.
    """

    required_statistics: tuple[str, ...] = ("q_scaler", "energy_zero")
    capabilities = ("energy", "atomic_energies", "forces", "virial")

    @torch.no_grad()
    def calibrate_energy(self, structures) -> torch.Tensor:
        """Initialize shared output bias from training mean atomic energy.
        
        With E_i=E_i^NN-output_bias, set output_bias=-sum_b(E_b)/sum_b(N_b). This aligns the
        energy zero without changing forces.
        """
        total_energy = 0.0
        total_atoms = 0
        for structure in structures:
            if structure.energy is None:
                raise ValueError("Energy-zero calibration requires energy labels")
            total_energy += float(structure.energy)
            total_atoms += structure.num_atoms
        if total_atoms == 0:
            raise ValueError("Cannot calibrate energy zero from an empty dataset")
        self.output_bias.fill_(-total_energy / total_atoms)
        return self.output_bias.detach()

    @torch.no_grad()
    def calibrate_scaler(self, structures, provider) -> torch.Tensor:
        """Calibrate descriptor ranges on training structures using initial radial coefficients.
        
        Descriptors depend on trainable coefficients, so calibration follows model construction
        rather than operating independently like DeepPot coordinate statistics. The [D] q_scaler
        is stored as a fixed buffer for training and serialization.
        """

        statistics = NepDescriptorRange(self.descriptor_dim)
        for structure in structures:
            neighbors = provider.build(structure)
            atom_types = self.type_lut[structure.atomic_numbers]
            descriptors = self.descriptor(
                atom_types,
                neighbors.center_indices,
                neighbors.edge_ptr,
                neighbors.neighbor_indices,
                neighbors.edge_vectors,
            )
            statistics.update(descriptors)
        q_scaler = statistics.finalize().to(
            device=self.scaler.q_scaler.device,
            dtype=self.scaler.q_scaler.dtype,
        )
        self.scaler.q_scaler.copy_(q_scaler)
        return self.scaler.q_scaler

    def __init__(
        self,
        type_map: List[int],
        cutoff_radial: float,
        cutoff_angular: float,
        n_max_radial: int,
        n_max_angular: int,
        basis_size_radial: int,
        basis_size_angular: int,
        l_max: List[int],
        num_neurons: Optional[int] = None,
        q_scaler: Optional[torch.Tensor] = None,
        output_bias: Optional[float] = None,
        network_size: Optional[List[int]] = None,
    ):
        super().__init__(type_map=type_map, cutoff=max(cutoff_radial, cutoff_angular))
        if network_size is not None and num_neurons is not None:
            raise ValueError("Specify either network_size or num_neurons, not both")
        if network_size is None:
            if num_neurons is None or num_neurons < 1:
                raise ValueError("num_neurons must be positive when network_size is absent")
            network_size = [num_neurons]
        self.register_buffer("type_lut", type_lookup(type_map))

        if cutoff_radial <= 0 or cutoff_angular <= 0:
            raise ValueError("NEP cutoffs must be positive")
        if n_max_radial < 0 or n_max_angular < 0:
            raise ValueError("n_max values must be non-negative")
        if basis_size_radial < 0 or basis_size_angular < 0:
            raise ValueError("basis_size values must be non-negative")
        if not l_max or l_max[0] < 0 or l_max[0] > 4:
            raise ValueError("l_max[0] must be between 0 and 4")
        if len(l_max) > 4 and any(value > 0 for value in l_max[4:]):
            raise ValueError("q_123, q_233, and q_134 are not implemented")
        self.num_types = len(type_map)
        self.cutoff_radial = float(cutoff_radial)
        self.cutoff_angular = float(cutoff_angular)
        self.n_max_radial = int(n_max_radial)
        self.n_max_angular = int(n_max_angular)
        self.basis_size_radial = int(basis_size_radial)
        self.basis_size_angular = int(basis_size_angular)
        self.l_max_3b = int(l_max[0])
        self.has_q_222 = len(l_max) > 1 and l_max[1] > 0
        self.has_q_1111 = len(l_max) > 2 and l_max[2] > 0
        self.has_q_112 = len(l_max) > 3 and l_max[3] > 0
        if (self.has_q_222 or self.has_q_112) and self.l_max_3b < 2:
            raise ValueError("q_222 and q_112 require l_max_3b >= 2")
        if self.has_q_1111 and self.l_max_3b < 1:
            raise ValueError("q_1111 requires l_max_3b >= 1")

        self.radial_dim = self.n_max_radial + 1
        self.angular_channels = self.n_max_angular + 1
        self.num_lm = (self.l_max_3b + 1) ** 2 - 1 if self.l_max_3b > 0 else 0
        higher_blocks = int(self.has_q_222) + int(self.has_q_1111) + int(self.has_q_112)
        self.descriptor_dim = (
            self.radial_dim
            + self.angular_channels * (self.l_max_3b + higher_blocks)
        )
        self.radial_coefficients = nn.Parameter(torch.empty(
            self.num_types,
            self.num_types,
            self.radial_dim,
            self.basis_size_radial + 1,
        ))
        self.angular_coefficients = nn.Parameter(torch.empty(
            self.num_types,
            self.num_types,
            self.angular_channels,
            self.basis_size_angular + 1,
        ))
        nn.init.normal_(self.radial_coefficients, std=0.1)
        nn.init.normal_(self.angular_coefficients, std=0.1)
        self.register_buffer("c3b", torch.tensor(C3B, dtype=torch.float64))
        self.register_buffer("c4b", torch.tensor(C4B, dtype=torch.float64))
        self.register_buffer("c5b", torch.tensor(C5B, dtype=torch.float64))
        self.register_buffer("c4b2", torch.tensor(C4B2, dtype=torch.float64))

        self.scaler = NepDescriptorScaler(self.descriptor_dim, q_scaler)
        self.fitting_network = NepFittingNetwork(
            self.num_types,
            self.descriptor_dim,
            network_size,
        )
        initial_bias = 0.0 if output_bias is None else float(output_bias)
        self.output_bias = nn.Parameter(torch.tensor(initial_bias))

    def descriptor(
        self,
        atom_types: torch.Tensor,
        center_indices: torch.Tensor,
        edge_ptr: torch.Tensor,
        neighbor_indices: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate edge_vectors=[E,3] into descriptors=[C,D].
        
        atom_types contains mapped species indices 0..T-1, not atomic numbers. edge_ptr assigns
        edges to centers; this method neither rebuilds neighbors nor applies PBC. Notation
        follows Chapter 5: f_k, g_n, b_lm, and s_nlm denote radial bases, radial functions, real
        angular bases, and angular moments.
        """
        n_centers = center_indices.numel()
        degrees = edge_ptr[1:] - edge_ptr[:-1]
        center_slots = torch.repeat_interleave(
            torch.arange(n_centers, device=edge_ptr.device),
            degrees,
        )
        squared_distances = edge_vectors.square().sum(dim=-1)
        distances = squared_distances.clamp(min=1.0e-24).sqrt()
        nonzero = squared_distances > 1.0e-24

        # Eq. (5.8): g_n(r_ij) = sum_k c_nk^(IJ) f_k(r_ij).
        # Filter radial edges first, then select coefficients by center species I and neighbor species J.
        radial_mask = nonzero & (distances < self.cutoff_radial)
        radial_slots = center_slots[radial_mask]
        radial_centers = center_indices[radial_slots]
        radial_neighbors = neighbor_indices[radial_mask]
        f_k_radial = radial_basis(
            distances[radial_mask],
            self.cutoff_radial,
            self.basis_size_radial,
        )
        radial_pair_types = atom_types[radial_centers] * self.num_types + atom_types[radial_neighbors]
        c_nk_radial = self.radial_coefficients.reshape(
            self.num_types * self.num_types,
            self.radial_dim,
            self.basis_size_radial + 1,
        )[radial_pair_types]
        g_n_radial = (c_nk_radial * f_k_radial.unsqueeze(1)).sum(dim=-1)
        # Eq. (5.7): q_n^i = sum_j g_n(r_ij).
        q_n_radial = edge_vectors.new_zeros(n_centers, self.radial_dim).index_add(
            0,
            radial_slots,
            g_n_radial,
        )
        parts = [q_n_radial]

        if self.l_max_3b > 0:
            # Angular and radial parts use independent cutoffs and trainable coefficients,
            # but share the radial form g_n(r_ij) = sum_k c_nk^(IJ) f_k(r_ij).
            angular_mask = nonzero & (distances < self.cutoff_angular)
            angular_slots = center_slots[angular_mask]
            angular_centers = center_indices[angular_slots]
            angular_neighbors = neighbor_indices[angular_mask]
            angular_vectors = edge_vectors[angular_mask]
            angular_distances = distances[angular_mask]
            f_k_angular = radial_basis(
                angular_distances,
                self.cutoff_angular,
                self.basis_size_angular,
            )
            angular_pair_types = atom_types[angular_centers] * self.num_types + atom_types[angular_neighbors]
            c_nk_angular = self.angular_coefficients.reshape(
                self.num_types * self.num_types,
                self.angular_channels,
                self.basis_size_angular + 1,
            )[angular_pair_types]
            g_n_angular = (c_nk_angular * f_k_angular.unsqueeze(1)).sum(dim=-1)

            # In Eqs. (5.36)-(5.59), b_lm is homogeneous of degree l. Evaluate it on
            # r_hat=r_vec/r, so
            # r^l*b_lm(r_hat)=b_lm(r_vec), without another explicit r^l multiplication.
            unit_vectors = angular_vectors / angular_distances.clamp(min=1.0e-12).unsqueeze(-1)
            b_lm = angular_basis(unit_vectors, self.l_max_3b)

            # Eq. (5.35): S_nlm^i = sum_j g_n(r_ij) r_ij^l b_lm(r_vec).
            # Form g_n*b_lm(r_hat) on each edge, then aggregate by center;
            # s_nlm=[C,N_angular,lm] represents S_nlm^i.
            edge_s_nlm = g_n_angular.unsqueeze(-1) * b_lm.unsqueeze(1)
            s_nlm = edge_vectors.new_zeros(
                n_centers,
                self.angular_channels,
                self.num_lm,
            ).index_add(0, angular_slots, edge_s_nlm)

            # Eq. (5.34): q_nl^i = sum_m C_lm (S_nlm^i)^2.
            # The m=0 term occurs once; paired real/imaginary components carry a factor of two.
            # GPUMD real-basis normalization coefficients are expanded into c3b.
            angular_parts: List[torch.Tensor] = []
            offset = 0
            for angular_order in range(1, self.l_max_3b + 1):
                width = 2 * angular_order + 1
                c_lm = self.c3b[offset:offset + width]
                squared_s_nlm = s_nlm[:, :, offset:offset + width].square()
                invariant = c_lm[0] * squared_s_nlm[:, :, 0]
                invariant = invariant + 2.0 * (
                    c_lm[1:] * squared_s_nlm[:, :, 1:]
                ).sum(dim=-1)
                angular_parts.append(invariant)
                offset += width
            three_body = torch.stack(angular_parts, dim=-1).transpose(1, 2).reshape(
                n_centers,
                self.angular_channels * self.l_max_3b,
            )
            parts.append(three_body)

            # Use the real-component indices of S_nlm from the reference, for example:
            # s10=S_n10、s11r=S_n11、s11i=S_n12、s20=S_n20。
            if self.has_q_1111 or self.has_q_112:
                s10, s11r, s11i = s_nlm[:, :, :3].unbind(-1)
            if self.has_q_222 or self.has_q_112:
                s20, s21r, s21i, s22r, s22i = s_nlm[:, :, 3:8].unbind(-1)
            if self.has_q_222:
                # Eq. (5.78): q_n222 contracts three L=2 moments S_n2m using Wigner 3j;
                # c222 stores fixed coefficients after expanding and combining real components.
                c222 = self.c4b
                parts.append(
                    c222[0] * s20.pow(3)
                    + c222[1] * s20 * (s21r.square() + s21i.square())
                    + c222[2] * s20 * (s22r.square() + s22i.square())
                    + c222[3] * s22r * (s21i.square() - s21r.square())
                    + c222[4] * s21r * s21i * s22i
                )
            if self.has_q_1111:
                # Eq. (5.79): let S0=S_n10^2 and S1=S_n11^2+S_n12^2, then
                # q_n1111 = 21/(80*pi^2)*S0^2
                #           + 21/(40*pi^2)*S0*S1 + 21/(80*pi^2)*S1^2。
                c1111 = self.c5b
                s0sq = s10.square()
                s1sq = s11r.square() + s11i.square()
                parts.append(c1111[0] * s0sq.square() + c1111[1] * s0sq * s1sq + c1111[2] * s1sq.square())
            if self.has_q_112:
                # NEP4 four-body q_n112 contracts two L=1 moments with one L=2 moment,
                # forming a rotational invariant under the general four-body definition in Eq. (5.74).
                c112 = self.c4b2
                parts.append(
                    c112[0] * s10.square() * s20
                    + c112[1] * s10 * (s11r * s21r + s11i * s21i)
                    + c112[2] * s20 * (s11r.square() + s11i.square())
                    + c112[3] * s22r * (s11r.square() - s11i.square())
                    + c112[4] * s11r * s11i * s22i
                )
        return torch.cat(parts, dim=-1)


    def set_scaler(self, q_min: torch.Tensor, q_max: torch.Tensor) -> None:
        """Update multiplicative scaling from per-channel descriptor ranges."""
        self.scaler.set_range(q_min, q_max)

    def compute(self, inputs: ModelInput) -> ModelOutput:
        """Compute NEP atomic energies from CSR edges and return scaled descriptors."""
        atomic_numbers = inputs.atomic_numbers
        center_indices, edge_ptr, neighbor_indices, edge_vectors = (
            inputs.require_neighbors()
        )
        atom_types = self.type_lut[atomic_numbers]
        if bool((atom_types < 0).any()):
            raise ValueError("Atomic number is absent from type_map")
        descriptors = self.descriptor(
            atom_types,
            center_indices,
            edge_ptr,
            neighbor_indices,
            edge_vectors,
        )
        descriptors = self.scaler(descriptors)
        center_types = atom_types[center_indices]
        atomic_energies = (
            self.fitting_network(descriptors, center_types) - self.output_bias
        )
        return ModelOutput(
            energy=atomic_energies.sum(),
            atomic_energies=atomic_energies,
            properties={"descriptors": descriptors},
        )


# ---------------------------------------------------------------------------
# Keep model declarations and configuration parsing in the same file.
# ---------------------------------------------------------------------------


def _parse_nonnegative(value: object, path: str) -> int:
    number = as_int(value, path)
    if number < 0:
        raise ConfigError(path, "must be non-negative")
    return number


@dataclass(frozen=True)
class NepConfig:
    type_map: tuple[int, ...]
    cutoff_radial: float
    cutoff_angular: float
    n_max_radial: int
    n_max_angular: int
    basis_size_radial: int
    basis_size_angular: int
    l_max: tuple[int, ...]
    network_size: tuple[int, ...]
    output_bias: float | None

    @property
    def cutoff(self) -> float:
        """Single cutoff used by neighbor construction."""
        return max(self.cutoff_radial, self.cutoff_angular)

    @classmethod
    def from_dict(
        cls, raw: Mapping, *, type_map: tuple[int, ...], path: str = "model"
    ) -> "NepConfig":
        check_unknown(
            raw,
            {
                "name", "cutoff_radial", "cutoff_angular", "n_max_radial",
                "n_max_angular", "basis_size_radial", "basis_size_angular",
                "l_max", "num_neurons", "hidden_layers", "output_bias",
            },
            path,
        )
        cutoff_radial = as_float(raw.get("cutoff_radial"), f"{path}.cutoff_radial")
        cutoff_angular = as_float(raw.get("cutoff_angular"), f"{path}.cutoff_angular")
        if cutoff_radial <= 0 or cutoff_angular <= 0:
            raise ConfigError(f"{path}", "both cutoffs must be positive")

        n_max_radial = _parse_nonnegative(raw.get("n_max_radial"), f"{path}.n_max_radial")
        n_max_angular = _parse_nonnegative(raw.get("n_max_angular"), f"{path}.n_max_angular")
        basis_size_radial = _parse_nonnegative(
            raw.get("basis_size_radial"), f"{path}.basis_size_radial"
        )
        basis_size_angular = _parse_nonnegative(
            raw.get("basis_size_angular"), f"{path}.basis_size_angular"
        )

        l_max = int_list(raw.get("l_max"), f"{path}.l_max")
        if not l_max:
            raise ConfigError(f"{path}.l_max", "must be a non-empty list")
        if not 0 <= l_max[0] <= 4:
            raise ConfigError(f"{path}.l_max", "l_max[0] must be between 0 and 4")
        if len(l_max) > 4:
            raise ConfigError(f"{path}.l_max", "at most 4 entries are supported")
        if any(value < 0 for value in l_max):
            raise ConfigError(f"{path}.l_max", "entries must be non-negative")

        num_neurons = raw.get("num_neurons")
        hidden_layers = raw.get("hidden_layers")
        if num_neurons is not None and hidden_layers is not None:
            raise ConfigError(
                f"{path}", "specify either num_neurons or hidden_layers, not both"
            )
        if hidden_layers is not None:
            widths = int_list(hidden_layers, f"{path}.hidden_layers")
            if not widths or any(width < 1 for width in widths):
                raise ConfigError(f"{path}.hidden_layers", "must be positive widths")
            network_size = widths
        elif num_neurons is not None:
            width = as_int(num_neurons, f"{path}.num_neurons")
            if width < 1:
                raise ConfigError(f"{path}.num_neurons", "must be positive")
            network_size = (width,)
        else:
            raise ConfigError(f"{path}", "requires num_neurons or hidden_layers")

        raw_output_bias = raw.get("output_bias")
        output_bias = (
            None
            if raw_output_bias is None
            else as_float(raw_output_bias, f"{path}.output_bias")
        )
        return cls(
            type_map=type_map,
            cutoff_radial=cutoff_radial,
            cutoff_angular=cutoff_angular,
            n_max_radial=n_max_radial,
            n_max_angular=n_max_angular,
            basis_size_radial=basis_size_radial,
            basis_size_angular=basis_size_angular,
            l_max=l_max,
            network_size=network_size,
            output_bias=output_bias,
        )

    def model_kwargs(self) -> dict:
        return {
            "type_map": list(self.type_map),
            "cutoff_radial": self.cutoff_radial,
            "cutoff_angular": self.cutoff_angular,
            "n_max_radial": self.n_max_radial,
            "n_max_angular": self.n_max_angular,
            "basis_size_radial": self.basis_size_radial,
            "basis_size_angular": self.basis_size_angular,
            "l_max": list(self.l_max),
            "network_size": list(self.network_size),
            "output_bias": self.output_bias,
        }


# Bind model and configuration here; the registry remains implementation-independent.
register_model("nep", ModelSpec(NepModel, NepConfig))
