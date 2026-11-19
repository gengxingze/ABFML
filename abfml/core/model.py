"""Atomic model contracts and shared mechanics."""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import nn
from abfml.core.types import ModelInput, ModelOutput
from abfml.core.neighbor import to_dense


class AtomicModel(nn.Module, ABC):
    """Base atomic model separating geometry requirements, neighbor sources, and force modes.
    
    ``energy`` may be a scalar total or per-structure ``[B]`` values. With atomic energies, a
    scalar batch total is split using ``center_batch``. Autograd differentiates the sum of model
    energies. In ``internal`` mode, ``compute()`` builds neighbors; the base class only
    dispatches coordinate derivatives.
    """

    geometry_mode: str = "local"       # "local" | "global"
    neighbor_mode: str = "external"    # "external" | "internal" | "none"
    force_mode: str = "autograd"       # "autograd" | "direct" | "none"
    capabilities: tuple[str, ...] = ("energy", "forces")
    cutoff: float | None = None
    required_statistics: tuple[str, ...] = ()

    def __init__(
        self,
        type_map: Optional[list[int]] = None,
        cutoff: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.type_map = None if type_map is None else list(type_map)
        if cutoff is not None:
            self.cutoff = float(cutoff)
        self.geometry_mode = str(self.geometry_mode)
        self.neighbor_mode = str(self.neighbor_mode)
        self.force_mode = str(self.force_mode)
        if self.geometry_mode not in {"local", "global"}:
            raise ValueError(f"Unknown geometry mode: {self.geometry_mode}")
        if self.neighbor_mode not in {"external", "internal", "none"}:
            raise ValueError(f"Unknown neighbor mode: {self.neighbor_mode}")
        if self.force_mode not in {"autograd", "direct", "none"}:
            raise ValueError(f"Unknown force mode: {self.force_mode}")
        if self.cutoff is not None and self.cutoff <= 0:
            raise ValueError("cutoff must be positive when provided")

    @abstractmethod
    def compute(self, inputs: ModelInput) -> ModelOutput:
        """Predict energy, forces, or other physical properties in a concrete model."""
        raise NotImplementedError

    def to_dense(
        self,
        edge_ptr: torch.Tensor,
        neighbor_indices: torch.Tensor,
        edge_vectors: torch.Tensor,
        max_neighbors: int = -1,
        neighbor_type_ids: Optional[torch.Tensor] = None,
        type_slots: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert CSR neighbors to a global or species-segmented dense view."""
        if neighbor_type_ids is not None and self.type_map is None:
            raise ValueError("type_map is required for species-segmented dense neighbors")
        return to_dense(
            edge_ptr,
            neighbor_indices,
            edge_vectors,
            max_neighbors,
            neighbor_type_ids=neighbor_type_ids,
            num_types=0 if self.type_map is None else len(self.type_map),
            type_slots=type_slots,
        )

    def _autograd_mechanics(
        self,
        inputs: ModelInput,
        outputs: ModelOutput,
    ) -> None:
        """Derive missing forces and virial from energy.
        
        The coordinate-based virial assumes no explicit cell dependence. Models with such
        dependence must return the complete virial from ``compute()``.
        """
        energy = outputs.energy
        assert energy is not None
        total_energy = energy.sum()
        if not total_energy.requires_grad:
            raise ValueError("Energy must be differentiable for autograd forces")

        center_batch = inputs.center_batch
        if self.neighbor_mode == "external":
            center_indices, edge_ptr, neighbor_indices, edge_vectors = (
                inputs.require_neighbors()
            )
            degrees = edge_ptr[1:] - edge_ptr[:-1]
            # [E]: map each edge to its actual center atom for force aggregation.
            center_per_edge = torch.repeat_interleave(center_indices, degrees)
            edge_grad = torch.autograd.grad(
                total_energy,
                edge_vectors,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            forces = edge_vectors.new_zeros((inputs.atomic_numbers.numel(), 3))
            forces = forces.index_add(0, center_per_edge, edge_grad)
            outputs.forces = forces.index_add(0, neighbor_indices, -edge_grad)

            if inputs.atomic_virial:
                if outputs.virial is not None:
                    raise ValueError("Center virial cannot partition a model-provided custom virial")
                edge_terms = -(edge_vectors.unsqueeze(-1) @ edge_grad.unsqueeze(-2))
                slots = torch.repeat_interleave(
                    torch.arange(center_indices.numel(), device=edge_ptr.device), degrees
                )
                outputs.atomic_virial = edge_terms.new_zeros(
                    (center_indices.numel(), 3, 3)
                ).index_add(0, slots, edge_terms)

            if outputs.virial is None:
                # [E,3,3]: W = -sum_e(r_e outer dE/dr_e), without volume normalization.
                edge_virials = -(edge_vectors.unsqueeze(-1) @ edge_grad.unsqueeze(-2))
                if center_batch is None:
                    outputs.virial = edge_virials.sum(dim=0)
                else:
                    # center_batch is indexed by center slot, not by actual atom index.
                    edge_batch = torch.repeat_interleave(center_batch, degrees)
                    outputs.virial = edge_virials.new_zeros(
                        (inputs.num_structures, 3, 3)
                    ).index_add(0, edge_batch, edge_virials)
            return

        positions = inputs.positions
        assert positions is not None
        position_grad = torch.autograd.grad(
            total_energy,
            positions,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]
        outputs.forces = -position_grad

        if outputs.virial is None:
            # [N,3,3]，W=-sum_i(R_i ⊗ dE/dR_i)。
            atom_virials = -(positions.unsqueeze(-1) @ position_grad.unsqueeze(-2))
            if center_batch is None:
                outputs.virial = atom_virials.sum(dim=0)
            else:
                atom_ptr = inputs.atom_ptr
                if atom_ptr is None:
                    raise ValueError("atom_ptr is required for batched virial")
                atom_batch = torch.repeat_interleave(
                    torch.arange(inputs.num_structures, device=positions.device),
                    atom_ptr[1:] - atom_ptr[:-1],
                )
                outputs.virial = atom_virials.new_zeros(
                    (inputs.num_structures, 3, 3)
                ).index_add(0, atom_batch, atom_virials)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        """Prepare differentiable inputs, run the model, and normalize physical outputs."""
        if self.force_mode == "autograd":
            if self.neighbor_mode == "external":
                if (
                    inputs.center_indices is None
                    or inputs.edge_ptr is None
                    or inputs.neighbor_indices is None
                    or inputs.edge_vectors is None
                ):
                    raise ValueError("External neighbors require complete CSR inputs")
                inputs.edge_vectors.requires_grad_(True)
            else:
                # Both internal and none differentiate coordinates; internal neighbor construction is handled elsewhere.
                if inputs.positions is None:
                    raise ValueError("positions are required for autograd forces")
                inputs.positions.requires_grad_(True)

        if self.type_map is not None:
            supported = inputs.atomic_numbers.new_tensor(self.type_map)
            if not bool(torch.isin(inputs.atomic_numbers, supported).all()):
                raise ValueError("Atomic number is absent from type_map")
        outputs = self.compute(inputs)

        energy = outputs.energy
        atomic_energies = outputs.atomic_energies
        n_structures = inputs.num_structures
        center_batch = inputs.center_batch

        if atomic_energies is not None and center_batch is not None:
            if atomic_energies.shape != center_batch.shape:
                raise ValueError(
                    "atomic_energies and center_batch must have the same shape"
                )

        if energy is not None:
            if energy.ndim == 0:
                if n_structures == 1:
                    outputs.energy = energy.reshape(1) if inputs.atom_ptr is not None else energy.reshape(())
                elif atomic_energies is not None:
                    if center_batch is None:
                        raise ValueError(
                            "center_batch is required to aggregate atomic energies"
                        )
                    outputs.energy = energy.new_zeros(n_structures).index_add(
                        0, center_batch, atomic_energies
                    )
                else:
                    raise ValueError("Multi-structure energy requires per-structure or atomic energies")
            elif energy.ndim == 1:
                if energy.shape != (n_structures,):
                    raise ValueError(
                        f"energy must have shape [{n_structures}], "
                        f"got {tuple(energy.shape)}"
                    )
            else:
                raise ValueError("energy must be a scalar or a 1D tensor")

        if self.force_mode == "autograd":
            if outputs.energy is None:
                raise ValueError("Energy is required for autograd force calculation")
            if outputs.forces is not None:
                raise ValueError(
                    "Autograd mode expects forces to be calculated by AtomicModel"
                )
            self._autograd_mechanics(inputs, outputs)
        elif self.force_mode == "direct" and outputs.forces is None:
            raise ValueError("Direct force model must return forces")

        if outputs.forces is not None:
            expected_shape = (inputs.atomic_numbers.numel(), 3)
            if outputs.forces.shape != expected_shape:
                raise ValueError(f"forces must have shape {expected_shape}")

        return outputs
