import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from abfml.model.math_fun import compute_forces_and_virial, calculate_distance_grad
from abfml.param.param import Param
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union


class FieldModel(nn.Module, ABC):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: List[int]):
        super(FieldModel, self).__init__()
        self.type_map = type_map
        self.neighbor = neighbor
        self.cutoff = cutoff

    def forward(self,
                element_map: torch.Tensor,
                Zi: torch.Tensor,
                Nij: torch.Tensor,
                Zij: torch.Tensor,
                Rij: torch.Tensor,
                n_ghost: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_atoms, max_neighbor = Nij.shape
        device = Rij.device
        dtype = Rij.dtype
        Ri, dRi = calculate_distance_grad(Ri=Rij)
        Ri.requires_grad_(True)
        Etot, Ei = self.field(element_map=element_map,
                              Zi=Zi,
                              Nij=Nij,
                              Zij=Zij,
                              Rij=Ri,
                              n_ghost=n_ghost)
        if Etot.shape != (batch, 1):
            raise ValueError(" Etot must be torch.Tensor and have shape (batch, 1)")
        if Ei.shape != (batch, n_atoms, 1):
            raise ValueError(" Ei must be torch.Tensor and have shape (batch, n_atoms, 1)")
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Ei)]
        dE = torch.autograd.grad([Ei], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        assert dE is not None
        dE = torch.unsqueeze(dE, dim=-1)
        dE_Rid = -1.0 * torch.mul(dE, dRi).sum(dim=-2)
        Force, Virial, virial = compute_forces_and_virial(dE_Rid=dE_Rid,
                                                          Rij=Rij, Nij=Nij,
                                                          n_ghost=n_ghost, dtype=dtype, device=device)
        # print(Etot, Force)
        return Etot, Ei, Force, Virial, virial

    @abstractmethod
    def field(self,
              element_map: torch.Tensor,
              Zi: torch.Tensor,
              Nij: torch.Tensor,
              Zij: torch.Tensor,
              Rij: torch.Tensor,
              n_ghost: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


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
            Zi = image_batch["Zi"]
            element_type = image_batch["element_type"][0].to(torch.int64).tolist()
            for i_type, element in enumerate(element_type):
                mask = (Zi == element)
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


