import torch
import torch.nn as nn
from abfml.core.model.method import FieldModel, NormalModel
from typing import List, Tuple, Dict, Union, Any, Optional
from abfml.param.param import Param


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class LJ(FieldModel):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: List[int],
                 config: Dict[str, Any],
                 normal: Optional[List[torch.Tensor]]):
        super(LJ, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.config = config
        self.epsilon = nn.Parameter(torch.tensor([self.config['epsilon']]))
        self.sigma = nn.Parameter(torch.tensor([self.config['sigma']]))

    def field(self,
              element_map: torch.Tensor,
              Zi: torch.Tensor,
              Zij: torch.Tensor,
              Nij: torch.Tensor,
              Rij: torch.Tensor,
              n_ghost: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = Rij.device
        dtype = Rij.dtype
        # Rij[batch, n_atom, neighbor, 4] 4->[rij, xij, yij, zij]
        # 12/6 Lennard-Jones potential: E= 4*epsilon[(sigma/rij)^12 - (sigma/rij)^6]
        rij = Rij[:, :, :, 0]
        mask_rij = (rij > 1e-5)
        rr = torch.zeros(rij.shape, dtype=dtype, device=device)
        rr[mask_rij] = 1 / rij[mask_rij]
        Eij = 0.5 * 4 * self.epsilon * ((self.sigma * rr) ** 12 - (self.sigma * rr) ** 6)
        Ei = torch.sum(Eij, dim=-1, keepdim=True)
        Etot = torch.sum(Ei, dim=1)
        return Etot, Ei


class LJNormal(NormalModel):
    def __init__(self,
                 normal_data,
                 param_class: Param,
                 normal_rate: Union[float, str],
                 ):
        super().__init__(normal_data=normal_data,
                         param_class=param_class,
                         normal_rate=normal_rate,
                         is_get_energy_shift=True)
        pass

    def normal(self, normal_loader, param_class):
        type_map = param_class.GlobalSet.type_map
        ntype = len(type_map)
        type_num = torch.zeros(ntype)
        std_mean: List[torch.Tensor] = [torch.zeros(ntype, 4, requires_grad=False),
                                        torch.zeros(ntype, 4, requires_grad=False)]

        for i, image_batch in enumerate(normal_loader):
            Rij = image_batch["Rij"]
            pass

        return std_mean