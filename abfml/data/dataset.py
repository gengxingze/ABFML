import torch
from torch.utils.data import Dataset


class ABFMLDataset(Dataset):
    def __init__(
        self,
        num_frames: int,
        num_atoms: int,
        element_types: torch.Tensor,
        central_atoms: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_types: torch.Tensor,
        neighbor_vectors: torch.Tensor,
        energy: torch.Tensor = None,
        atomic_energies: torch.Tensor = None,
        forces: torch.Tensor = None,
        virials: torch.Tensor = None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_atoms = num_atoms

        # Validate physical quantities
        physics_info = {
            "energy": (energy, (num_frames, 1)),
            "atomic_energies": (atomic_energies, (num_frames, num_atoms, 1)),
            "forces": (forces, (num_frames, num_atoms, 3)),
            "virials": (virials, (num_frames, 9)),
        }

        for name, (tensor, expected_shape) in physics_info.items():
            if tensor is not None:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"{name} must be a torch.Tensor, but got {type(tensor)}.")
                if tensor.shape != expected_shape:
                    raise ValueError(f"{name} must have shape {expected_shape}, but got {tensor.shape}.")

        self.energy = energy
        self.atomic_energies = atomic_energies
        self.forces = forces
        self.virials = virials

        # Validate neighbor-related tensors
        num_neighbors = neighbor_indices.shape[-1]
        neighbor_info = {
            "central_atoms": (central_atoms, (num_frames, num_atoms)),
            "neighbor_indices": (neighbor_indices, (num_frames, num_atoms, num_neighbors)),
            "neighbor_types": (neighbor_types, (num_frames, num_atoms, num_neighbors)),
            "neighbor_vectors": (neighbor_vectors, (num_frames, num_atoms, num_neighbors, 4)),
        }

        for name, (tensor, expected_shape) in neighbor_info.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, but got {type(tensor)}.")
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, but got {tensor.shape}.")

        self.element_types = element_types
        self.central_atoms = central_atoms
        self.neighbor_indices = neighbor_indices
        self.neighbor_types = neighbor_types
        self.neighbor_vectors = neighbor_vectors

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        sample = {
            "num_atoms": self.num_atoms,
            "element_types": self.element_types,
            "central_atoms": self.central_atoms[index],
            "neighbor_indices": self.neighbor_indices[index],
            "neighbor_types": self.neighbor_types[index],
            "neighbor_vectors": self.neighbor_vectors[index],
        }

        if self.energy is not None:
            sample["energy"] = self.energy[index]
        if self.atomic_energies is not None:
            sample["atomic_energies"] = self.atomic_energies[index]
        if self.forces is not None:
            sample["forces"] = self.forces[index]
        if self.virials is not None:
            sample["virials"] = self.virials[index]

        return sample
