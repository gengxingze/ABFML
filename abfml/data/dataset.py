import torch
from torch.utils.data import Dataset


class ABFMLDataset(Dataset):
    def __init__(self,
                 n_frames: int,
                 n_atoms: int,
                 element_type: torch.Tensor,
                 Zi: torch.Tensor,
                 Nij: torch.Tensor,
                 Zij: torch.Tensor,
                 Rij: torch.Tensor,
                 energy: torch.Tensor = None,
                 atomic_energy: torch.Tensor = None,
                 force: torch.Tensor = None,
                 virial: torch.Tensor = None,
                 ):
        super(ABFMLDataset, self).__init__()
        self.n_frames = n_frames
        self.n_atoms = n_atoms

        physics_name = ["energy", "atomic_energy", "force", "virial"]
        physics_var = [energy, atomic_energy, force, virial]
        physics_shape = [(n_frames, 1), (n_frames, n_atoms, 1), (n_frames, n_atoms, 3), (n_frames, 9)]
        for tensor_p, shape_p, name_p in zip(physics_var, physics_shape, physics_name):
            if tensor_p is not None:
                if not isinstance(tensor_p, torch.Tensor):
                    raise TypeError(f"If {name_p} is required, then the type of a should be torch.Tensor, "
                                    f"but its type is {type(tensor_p)}")
                if tensor_p.shape != shape_p:
                    raise ValueError(f"Expected shape of {name_p} is {shape_p},"
                                     f"but the shape you get is {tensor_p.shape}")
        self.atomic_energy = atomic_energy
        self.energy = energy
        self.force = force
        self.virial = virial

        neighbor_name = ["Zi", "Nij", "Zij", "Rij"]
        neighbor_var = [Zi, Nij, Zij, Rij]
        neighbor_shape = [(n_frames, n_atoms), (n_frames, n_atoms, Nij.shape[-1]),
                          (n_frames, n_atoms, Nij.shape[-1]), (n_frames, n_atoms, Nij.shape[-1], 4)]
        for tensor_n, shape_n, name_n in zip(neighbor_var, neighbor_shape, neighbor_name):
            if not isinstance(tensor_n, torch.Tensor):
                raise TypeError(f"The type of {name_n} should to be torch.Tensor, but its {type(tensor_n)}")
            if tensor_n.shape != shape_n:
                raise ValueError(f"Expected shape of {name_n} is {shape_n}, but the shape you get is {tensor_n.shape}")
        self.element_type = element_type
        self.Zi = Zi
        self.Nij = Nij
        self.Zij = Zij
        self.Rij = Rij

    def __len__(self):
        return self.n_frames

    def __getitem__(self, index):
        mlff_dict = {
            'n_atoms': self.n_atoms,
            'element_type': self.element_type,
            'Zi': self.Zi[index],
            'Nij': self.Nij[index],
            'Zij': self.Zij[index],
            'Rij': self.Rij[index],
        }
        optional_name = ["energy", "atomic_energy", "force", "virial"]
        optional_val = [self.energy, self.atomic_energy, self.force, self.virial]
        for tensor, name in zip(optional_val, optional_name):
            if tensor is not None:
                mlff_dict[name] = tensor[index]
        return mlff_dict
