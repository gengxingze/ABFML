import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from abc import ABC
from typing import List, Optional, Tuple, Union, Dict
from abfml.core.model.math_fun import derive_mechanics
from abfml.param.param import Param


class FieldModel(nn.Module, ABC):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: Union[Dict[int, int],int]):
        """
        Base class for field-based machine learning potential models.

        Parameters
        ----------
        type_map : List[int]
            A list mapping atomic species to integer indices used in the model.
            For example, [0, 1, 2] might represent three different element types.

        cutoff : float
            Cutoff radius for neighbor searching.
            Only atoms within this distance will be considered as neighbors.

        neighbor : Union[Dict[int, int],int]
            Maximum number of neighbors for each atom type (or a global value).
            Used to define tensor sizes or guide model input structure.

        Example
        -------
        model = FieldModel(type_map=[0, 1], cutoff=6.0, neighbor=[1:100])
        """
        super(FieldModel, self).__init__()
        self.type_map = type_map
        self.neighbor = neighbor
        self.cutoff = cutoff

    def forward(self,
                element_map: torch.Tensor,
                central_atoms: torch.Tensor,
                neighbor_indices: torch.Tensor,
                neighbor_types: torch.Tensor,
                neighbor_vectors: torch.Tensor,
                n_ghost: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main forward pass computing physical quantities
        Args:
            element_map: Atomic numbers mapped to indices [batch, n_atoms]
            central_atoms: Central atom indices [batch, n_atoms]
            neighbor_indices: Neighbor atom indices [batch, n_atoms, max_nbr]
            neighbor_types: Neighbor atom types [batch, n_atoms, max_nbr]
            neighbor_vectors: Relative position vectors [batch, n_atoms, max_nbr, 4]
                              (distance, dx, dy, dz)
            n_ghost: Number of ghost atoms in the system

        Returns:
            Tuple containing:
            - Etot [batch, 1]
            - Ei [batch, n_atoms, 1]
            - force [batch, n_atoms, 3]
            - virial tensor [batch, 9]
            - atomic_virial [batch, n_atoms, 9]
        """
        # Extract dimensions and tensor properties
        batch, n_atoms, max_neighbor, _ = neighbor_vectors.shape
        device, dtype = neighbor_vectors.device, neighbor_vectors.dtype

        # Extract displacement vectors (dx, dy, dz) and enable autograd
        relative_positions = neighbor_vectors[..., 1:]
        relative_positions.requires_grad_(True)

        # Recalculate distance norm for generalization and recombine into input vector
        distance = torch.norm(relative_positions, dim=-1, keepdim=True)  # Shape: [batch, n_atoms, max_nbr, 1]
        neighbor_vector = torch.cat([distance, relative_positions], dim=-1)  # Shape: [batch, n_atoms, max_nbr, 4]

        # Call the user-defined field function to compute energy and optionally other properties
        physics_info = self.field(
            element_map=element_map,
            central_atoms=central_atoms,
            neighbor_indices=neighbor_indices,
            neighbor_types=neighbor_types,
            neighbor_vectors=neighbor_vector,
            n_ghost=n_ghost
        )

        # Required outputs
        Etot = physics_info['Etot']  # Total energy per configuration, shape: [batch, 1]
        Ei = physics_info['Ei']  # Per-atom energy, shape: [batch, n_atoms, 1]

        # Validate required outputs
        if Etot is None or not isinstance(Etot, torch.Tensor) or Etot.shape != (batch, 1):
            raise ValueError("physics_info['Etot'] must be a torch.Tensor of shape (batch, 1)")
        if Ei is None or not isinstance(Ei, torch.Tensor) or Ei.shape != (batch, n_atoms, 1):
            raise ValueError("physics_info['Ei'] must be a torch.Tensor of shape (batch, n_atoms, 1)")

        # If force is not provided, compute it via autograd
        if physics_info.get('Force') is None:
            # Create a gradient mask for backpropagation
            energy_mask: List[Optional[torch.Tensor]] = [torch.ones_like(Ei)]

            # Compute gradient of Ei with respect to atomic positions
            grad_Ei = torch.autograd.grad(
                outputs=[Ei],
                inputs=[relative_positions],
                grad_outputs=energy_mask,
                retain_graph=True,
                create_graph=True
            )[0]
            assert grad_Ei is not None

            # Compute force, virial tensor, and atomic-level virial from gradients
            force, virial, atomic_virial = derive_mechanics(
                grad_Ei=grad_Ei,
                neighbor_vectors=neighbor_vector,
                neighbor_indices=neighbor_indices,
                n_ghost=n_ghost
            )
        else:
            # If force is provided by field, validate its shape
            force = physics_info['Force']
            if not isinstance(force, torch.Tensor) or force.shape != (batch, n_atoms, 3):
                raise ValueError("physics_info['Force'] must be a torch.Tensor of shape (batch, n_atoms, 3)")

            # Optional: check for virial tensor
            virial = physics_info.get('virial')
            if virial is not None:
                if not isinstance(virial, torch.Tensor) or virial.shape != (batch, 3, 3):
                    raise ValueError("physics_info['virial'] must be a torch.Tensor of shape (batch, 3, 3)")
            else:
                virial = torch.zeros(batch, 3, 3, dtype=dtype, device=device)

            # Optional: check for atomic-level virial
            atomic_virial = physics_info.get('atomic_virial')
            if atomic_virial is not None:
                if not isinstance(atomic_virial, torch.Tensor) or atomic_virial.shape != (batch, n_atoms, 9):
                    raise ValueError(
                        "physics_info['atomic_virial'] must be a torch.Tensor of shape (batch, n_atoms, 9)")
            else:
                atomic_virial = torch.zeros(batch, n_atoms, 9, dtype=dtype, device=device)

        # Return all computed quantities
        return Etot, Ei, force, virial, atomic_virial

    def field(self,
              element_map: torch.Tensor,
              central_atoms: torch.Tensor,
              neighbor_indices: torch.Tensor,
              neighbor_types: torch.Tensor,
              neighbor_vectors: torch.Tensor,
              n_ghost: int) -> Dict[str, torch.Tensor]:
        """
        Abstract method to be implemented by subclasses. This method defines the core
        field calculation of the model, which computes atom-wise energies and optionally
        forces and virial tensors based on atomic environments.

        Parameters
        ----------
        element_map : torch.Tensor
            Tensor of shape [batch, n_atoms], indicating the atomic species (element index)
            for each atom in the batch.

        central_atoms : torch.Tensor
            Tensor of shape [batch, n_atoms], representing the type of central atoms.

        neighbor_indices : torch.Tensor
            Tensor of shape [batch, n_atoms, max_nbr], providing the indices of neighboring
            atoms for each central atom.

        neighbor_types : torch.Tensor
            Tensor of shape [batch, n_atoms, max_nbr], indicating the atomic species
            of neighboring atoms.

        neighbor_vectors : torch.Tensor
            Tensor of shape [batch, n_atoms, max_nbr, 4], where the last dimension represents
            (distance, dx, dy, dz), i.e., the norm and vector components of the relative positions.

        n_ghost : int
            Number of ghost atoms (used to pad or augment neighbor lists, typically for parallelism).

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the computed physical quantities:
            - 'Etot': Total energy, shape [batch, 1].
            - 'Ei': Atomic energy per atom, shape [batch, n_atoms, 1].
            - Optional: 'Force': Force on each atom, shape [batch, n_atoms, 3].
            - Optional: 'virial': Virial tensor, shape [batch, 9].
            - Optional: 'atomic_virial': virial Force on each atom, shape [batch, n_atoms, 9].

        Note
        ----
        This method must be implemented in subclasses of `FieldModel`.
        """
        return {'Etot': torch.tensor([0.0])}


class NormalModel:
    def __init__(self,
                 normal_data,
                 param_class: Param,
                 normal_rate: Union[float, str] = 'auto',
                 is_get_energy_shift: bool = False):
        self.param_class = param_class
        self.normal_loader = NormalModel.normal_data_loader(need_data=normal_data, normal_rate=normal_rate)
        self.is_get_energy_shift = is_get_energy_shift
        self.normal_data = tuple([])

    def initialize(self):
        normal_data = self.normal(normal_loader=self.normal_loader, param_class=self.param_class)

        if isinstance(normal_data, tuple):
            self.normal_data = normal_data
        else:
            self.normal_data = tuple([normal_data])

        if self.is_get_energy_shift:
            energy_shift = NormalModel.get_energy_shift(need_data=self.normal_loader, type_map=self.param_class.GlobalSet.type_map)
            self.normal_data = tuple([energy_shift]) + self.normal_data

    @staticmethod
    def normal_data_loader(need_data, normal_rate: Union[float, str]) -> DataLoader:
        total_image_num = len(need_data)
        total_indices = np.arange(total_image_num)
        if isinstance(normal_rate, float):
            if normal_rate <= 1.0:
                num = int(total_image_num * normal_rate + 1)
            else:
                raise Exception("rate")
        elif normal_rate == "auto":
            if total_image_num * 0.1 < 100:
                num = total_image_num
            else:
                num = int(total_image_num * 0.1 + 1)
        else:
            raise Exception("rate")
        np.random.shuffle(total_indices)
        normal_indices = total_indices[:num]
        normal_data = Subset(need_data, normal_indices)
        num_threads = torch.get_num_threads()
        num_worker = int(num_threads / 2)
        normal_data_loader = DataLoader(normal_data, batch_size=1, shuffle=True, num_workers=num_worker)

        return normal_data_loader

    @staticmethod
    def get_energy_shift(need_data, type_map: List[int]) -> List[float]:
        ntype = len(type_map)
        type_num = torch.zeros(ntype)
        energy_shift = [0.0] * ntype
        for i, image_batch in enumerate(need_data):
            central_atoms = image_batch["central_atoms"]
            element_types = image_batch["element_types"][0].to(torch.int64).tolist()
            for i_type, element in enumerate(element_types):
                mask = (central_atoms == element)
                indices = type_map.index(element)
                type_num[indices] += 1
                try:
                    energy = torch.mean(image_batch["energy"] / image_batch["n_atoms"]).item()
                    energy_shift[indices] = energy_shift[indices] + energy
                except KeyError:
                    try:
                        Ei = torch.mean(image_batch["atomic_energy"][mask]).item()
                        energy_shift[indices] = energy_shift[indices] + Ei
                    except KeyError:
                        energy_shift[indices] = energy_shift[indices] + np.random.uniform(-10.0, 0.0)

        type_num[type_num == 0] = 1
        for i, i_energy in enumerate(energy_shift):
            energy_shift[i] = (i_energy / type_num[i]).item()

        return energy_shift

    def normal(self, normal_loader, param_class):
        return None

