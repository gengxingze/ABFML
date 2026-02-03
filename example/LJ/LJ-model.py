import torch
import torch.nn as nn
from abfml.core.model.method import FieldModel, NormalModel
from abfml.core.model.builder import register_model, BaseModelBuilder
from typing import List, Tuple, Dict, Union, Any, Optional
from abfml.param.param import Param


class LJ(FieldModel):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: List[int],
                 config: Dict[str, Any]):
        super(LJ, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.config = config
        self.epsilon = nn.Parameter(torch.tensor([self.config['epsilon']]))
        self.sigma = nn.Parameter(torch.tensor([self.config['sigma']]))

    def field(self,
              element_map: torch.Tensor,
              central_atoms: torch.Tensor,
              neighbor_indices: torch.Tensor,
              neighbor_types: torch.Tensor,
              neighbor_vectors: torch.Tensor,
              n_ghost: int) -> Dict[str, torch.Tensor]:
        device = neighbor_vectors.device
        dtype = neighbor_vectors.dtype
        # neighbor_vectors[batch, n_atom, neighbor, 4] 4->[rij, xij, yij, zij]
        # 12/6 Lennard-Jones potential: E= 4*epsilon[(sigma/rij)^12 - (sigma/rij)^6]
        rij = neighbor_vectors[:, :, :, 0]
        mask_rij = (rij > 1e-5)
        rr = torch.zeros(rij.shape, dtype=dtype, device=device)
        rr[mask_rij] = 1 / rij[mask_rij]
        Eij = 0.5 * 4 * self.epsilon * ((self.sigma * rr) ** 12 - (self.sigma * rr) ** 6)
        Ei = torch.sum(Eij, dim=-1, keepdim=True)
        Etot = torch.sum(Ei, dim=1)
        return {'Etot': Etot, 'Ei': Ei}


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

@register_model("LJ")
class UserModelBuilder(BaseModelBuilder):
    def build(self):
        # energy_shift, std_mean = LJNormal(normal_data=self.normal_data, param_class=self.param)
        return LJ(
            type_map=self.param.GlobalSet.type_map,
            cutoff=self.param.GlobalSet.cutoff,
            neighbor=self.param.GlobalSet.neighbor,
            config=self.param.user_defined.config
        )




