import time
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple, Any
from abfml.model.math_fun import smooth_fun
from abfml.model.network import FittingNet
from abfml.model.method import FieldModel


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BPMlp(FieldModel):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: List[int],
                 fit_config: Dict[str, Any],
                 bp_features_information: List[Tuple[str, int]],
                 bp_features_param: List[Tuple[Dict[str, torch.Tensor], Dict[str, str]]],
                 energy_shift: List[float],
                 std_mean: Optional[List[torch.Tensor]],
                 ):
        super(BPMlp, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.type_map = type_map
        self.neighbor = neighbor
        self.cutoff = cutoff
        self.std_mean = std_mean
        self.bp_features_information = bp_features_information
        self.bp_features_param = bp_features_param
        self.fitting_net = nn.ModuleList()
        self.fitting_net_index = []

        total_feature_num = BPMlp.calculate_feature_num(num_element=len(type_map),
                                                        bp_features_information=bp_features_information)
        for i, element in enumerate(self.type_map):
            self.fitting_net.append(FittingNet(network_size=[total_feature_num] + fit_config["network_size"],
                                               activate=fit_config["activate_function"],
                                               bias=fit_config["bias"],
                                               resnet_dt=fit_config["resnet_dt"],
                                               energy_shift=energy_shift[i]))
            self.fitting_net_index.append(str(element))

    def field(self,
              element_map: torch.Tensor,
              Zi: torch.Tensor,
              Nij: torch.Tensor,
              Zij: torch.Tensor,
              Rij: torch.Tensor,
              n_ghost: int):
        batch, n_atoms, max_neighbor = Nij.shape
        device = Rij.device
        dtype = Rij.dtype
        type_map_temp: torch.Tensor = element_map.to(torch.int64)
        type_map: List[int] = type_map_temp.tolist()
        feature = BPMlp.calculate_bp_feature(type_map=self.type_map,
                                             bp_features_information=self.bp_features_information,
                                             bp_features_param=self.bp_features_param,
                                             element_map=element_map,
                                             Nij=Nij,
                                             Zij=Zij,
                                             Rij=Rij)
        feature = BPMlp.scale(self.type_map, type_map, self.std_mean, feature, Zi)
        Ei = torch.zeros(batch, n_atoms, 1, dtype=dtype, device=device)
        for i, itype in enumerate(type_map):
            mask_itype = (Zi == itype)
            if not mask_itype.any():
                continue
            iifeat = feature[mask_itype].reshape(batch, -1, feature.shape[-1])
            ii = self.fitting_net_index.index(str(itype))
            fitting_net: ModuleInterface = self.fitting_net[ii]
            Ei_itype = fitting_net.forward(iifeat)
            Ei[mask_itype] = Ei_itype.reshape(-1, 1)
        Etot = torch.sum(Ei, dim=1)
        return Etot, Ei

    @staticmethod
    def scale(type_map_all: List[int],
              type_map_use: List[int],
              std_mean: List[torch.Tensor],
              feature: torch.Tensor,
              Zi: torch.Tensor) -> torch.Tensor:
        for i, element in enumerate(type_map_use):
            device = feature.device
            indices = type_map_all.index(element)
            mask = (Zi == element)
            std = std_mean[0][indices].detach().to(device)
            avg = std_mean[1][indices].detach().to(device)
            feature[mask] = (feature[mask] - avg) / std
        return feature

    @staticmethod
    def calculate_bp_feature(type_map: List[int],
                             bp_features_information: List[Tuple[str, int]],
                             bp_features_param: List[Tuple[Dict[str, torch.Tensor], Dict[str, str]]],
                             element_map: torch.Tensor,
                             Nij: torch.Tensor,
                             Zij: torch.Tensor,
                             Rij: torch.Tensor) -> torch.Tensor:
        feature_list: List[torch.Tensor] = []
        for i, feature_information in enumerate(bp_features_information):
            feature_name = feature_information[0]
            if feature_name == "SymFunRad":
                Rad_feature = BPMlp.sym_rad(type_map=type_map,
                                            R_min=(bp_features_param[i][0]['R_min']).item(),
                                            R_max=(bp_features_param[i][0]['R_max']).item(),
                                            eta=bp_features_param[i][0]['eta'],
                                            rs=bp_features_param[i][0]['rs'],
                                            smooth_type=bp_features_param[i][1]['smooth_fun'],
                                            element_map=element_map,
                                            Nij=Nij,
                                            Zij=Zij,
                                            Rij=Rij)
                feature_list.append(Rad_feature)

            if feature_name == "SymFunAngW":
                Ang_w_feature = BPMlp.sym_ang_w(type_map=type_map,
                                                R_min=(bp_features_param[i][0]['R_min'].item()),
                                                R_max=(bp_features_param[i][0]['R_max'].item()),
                                                lambd=bp_features_param[i][0]['lambd'],
                                                eta=bp_features_param[i][0]['eta'],
                                                zeta=bp_features_param[i][0]['zeta'],
                                                rs=bp_features_param[i][0]['rs'],
                                                smooth_type=bp_features_param[i][1]['smooth_fun'],
                                                element_map=element_map,
                                                Nij=Nij,
                                                Zij=Zij,
                                                Rij=Rij)
                feature_list.append(Ang_w_feature)

            if feature_name == "SymFunAngN":
                Ang_n_feature = BPMlp.sym_ang_n(type_map=type_map,
                                                R_min=(bp_features_param[i][0]['R_min'].item()),
                                                R_max=(bp_features_param[i][0]['R_max'].item()),
                                                lambd=bp_features_param[i][0]['lambd'],
                                                eta=bp_features_param[i][0]['eta'],
                                                zeta=bp_features_param[i][0]['zeta'],
                                                rs=bp_features_param[i][0]['rs'],
                                                smooth_type=bp_features_param[i][1]['smooth_fun'],
                                                element_map=element_map,
                                                Nij=Nij,
                                                Zij=Zij,
                                                Rij=Rij)
                feature_list.append(Ang_n_feature)

        feature = torch.cat([tensor for tensor in feature_list], dim=-1)
        return feature

    @staticmethod
    def calculate_feature_num(num_element: int, bp_features_information: List[Tuple[str, int]]) -> int:
        feature_num = 0
        for i, feature_information in enumerate(bp_features_information):
            feature_name = feature_information[0]
            num_basis = feature_information[1]
            if feature_name == "SymFunRad":
                feature_num += num_basis * num_element
            if "SymFunAng" in feature_name:
                feature_num += num_basis * num_element ** 2
        return feature_num

    @staticmethod
    def sym_rad(type_map: List[int],
                R_min: float,
                R_max: float,
                eta: torch.Tensor,
                rs: torch.Tensor,
                smooth_type: str,
                element_map: torch.Tensor,
                Nij: torch.Tensor,
                Zij: torch.Tensor,
                Rij: torch.Tensor) -> torch.Tensor:
        batch, n_atoms, max_neighbor = Nij.shape
        n_basis = eta.shape[0]
        device = Rij.device
        dtype = Rij.dtype
        include_element_list: List[int] = element_map.to(torch.int64).tolist()

        rij = Rij[:, :, :, 0].unsqueeze(-1)

        f_rij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        g_rij = torch.exp(-eta * (rij - rs) ** 2) * f_rij
        Gi = torch.zeros(batch, n_atoms, int(n_basis * len(type_map)), dtype=dtype, device=device)

        for i, itype in enumerate(include_element_list):
            indices = type_map.index(itype)
            mask_g = (Zij == itype).unsqueeze(-1)
            Gi[:, :,  n_basis * indices:n_basis * (indices + 1)] = torch.sum(g_rij * mask_g, dim=2)

        return Gi

    @staticmethod
    def sym_ang_n(type_map: List[int],
                  R_min: float,
                  R_max: float,
                  lambd: torch.Tensor,
                  eta: torch.Tensor,
                  zeta: torch.Tensor,
                  rs: torch.Tensor,
                  smooth_type: str,
                  element_map: torch.Tensor,
                  Nij: torch.Tensor,
                  Zij: torch.Tensor,
                  Rij: torch.Tensor) -> torch.Tensor:
        batch, n_atoms, max_neighbor = Nij.shape
        n_basis = eta.shape[0]
        device = Rij.device
        dtype = Rij.dtype
        include_element_list: List[int] = element_map.to(torch.int64).tolist()
        rij = Rij[:, :, :, 0:1]
        # Guaranteed not to divide by 0
        mask_rij = (rij > 1e-5)
        rr = torch.zeros(rij.shape, dtype=dtype, device=device)
        rr[mask_rij] = 1 / rij[mask_rij]

        frij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        grij = torch.exp(-eta * (rij - rs) ** 2) * frij
        cos_ijk = (torch.matmul(Rij[:, :, :, 1:], Rij[:, :, :, 1:].transpose(-1, -2))
                   * (rr * rr.transpose(-2, -1))).unsqueeze(2)
        Gi = torch.zeros(batch, n_atoms, int(n_basis * len(type_map) ** 2), dtype=dtype, device=device)

        for i, itype in enumerate(include_element_list):
            i_indices = type_map.index(itype)
            mask_i = (Zij == itype).unsqueeze(-1)
            gij = grij * mask_i
            # gij [batch, n_atoms, n_basis, max_neighbor, 1]
            gij = gij.transpose(2, 3).unsqueeze(-1)

            zeta = zeta.reshape(1, 1, n_basis, 1, 1)
            lambd = lambd.reshape(1, 1, n_basis, 1, 1)
            # t41 = time.time()
            for j, jtype in enumerate(include_element_list):
                j_indices = type_map.index(jtype)
                indices = i_indices * len(type_map) + j_indices
                mask_j = (Zij == itype).unsqueeze(-1)
                gik = grij * mask_j
                # gik [batch, n_atoms, n_basis, 1, max_neighbor]
                gik = gik.transpose(2, 3).unsqueeze(-2)
                # Gi_temp [batch, n_atoms, n_basis, max_neighbor, max_neighbor]
                Gi_temp = 2 ** (1 - zeta) * ((1 + lambd * cos_ijk) ** zeta) * (gij * gik)
                Gi[:, :,  n_basis * indices:n_basis * (indices + 1)] = Gi_temp.sum(dim=[-1, -2])
        return Gi

    @staticmethod
    def sym_ang_w(type_map: List[int],
                  R_min: float,
                  R_max: float,
                  lambd: torch.Tensor,
                  eta: torch.Tensor,
                  zeta: torch.Tensor,
                  rs: torch.Tensor,
                  smooth_type: str,
                  element_map: torch.Tensor,
                  Nij: torch.Tensor,
                  Zij: torch.Tensor,
                  Rij: torch.Tensor):
        batch, n_atoms, max_neighbor = Nij.shape
        n_basis = eta.shape[0]
        device = Rij.device
        dtype = Rij.dtype
        include_element_list: List[int] = element_map.to(torch.int64).tolist()
        rij = Rij[:, :, :, 0:1]

        # Guaranteed not to divide by 0
        mask_rij = (rij > 1e-5)
        rr = torch.zeros(rij.shape, dtype=dtype, device=device)
        rr[mask_rij] = 1 / rij[mask_rij]

        frij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        grij = torch.exp(-eta * (rij - rs) ** 2) * frij
        # cos_ijk [batch, n_atoms, 1,max_neighbor, max_neighbor]
        cos_ijk = (torch.matmul(Rij[:, :, :, 1:], Rij[:, :, :, 1:].transpose(-1, -2))
                   * (rr * rr.transpose(-2, -1))).unsqueeze(2)
        rjk2 = ((Rij[:, :, :, 1:].unsqueeze(3) - Rij[:, :, :, 1:].unsqueeze(3).permute(0, 1, 3, 2, 4)) ** 2).sum(-1)
        mask_rjk = (rjk2 > 1e-5)
        rjk = torch.zeros(rjk2.shape, dtype=dtype, device=device)
        rjk[mask_rjk] = rjk[mask_rjk] ** 0.5
        frjk = smooth_fun(smooth_type=smooth_type, rij=rjk, r_inner=R_min, r_outer=R_max)
        grjk = (torch.exp(-eta * (rjk.unsqueeze(-1) - rs) ** 2) * frjk.unsqueeze(-1)).transpose(2, 4).squeeze(-1)

        Gi = torch.zeros(batch, n_atoms, int(n_basis * len(type_map) ** 2), dtype=dtype, device=device)
        for i, itype in enumerate(include_element_list):
            i_indices = type_map.index(itype)
            mask_i = (Zij == itype).unsqueeze(-1)
            gij = grij * mask_i
            # gij [batch, n_atoms, n_basis, max_neighbor, 1]
            gij = gij.transpose(2, 3).unsqueeze(-1)

            zeta = zeta.reshape(1, 1, n_basis, 1, 1)
            lambd = lambd.reshape(1, 1, n_basis, 1, 1)

            # t41 = time.time()
            for j, jtype in enumerate(include_element_list):
                j_indices = type_map.index(jtype)
                indices = i_indices * len(type_map) + j_indices
                mask_j = (Zij == itype).unsqueeze(-1)
                gik = grij * mask_j
                # gik [batch, n_atoms, n_basis, 1, max_neighbor]
                gik = gik.transpose(2, 3).unsqueeze(-2)
                # Gi_temp [batch, n_atoms, n_basis, max_neighbor, max_neighbor]
                Gi_temp = 2 ** (1 - zeta) * ((1 + lambd * cos_ijk) ** zeta) * (gij * gik * grjk)
                Gi[:, :,  n_basis * indices:n_basis * (indices + 1)] = Gi_temp.sum(dim=[-1, -2])
        return Gi
