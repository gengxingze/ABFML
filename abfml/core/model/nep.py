import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from abfml.core.model.math_fun import smooth_fun, polynomial_fun
from abfml.core.model.network import FittingNet
from abfml.core.model.method import FieldModel


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class NEP(FieldModel):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: List[int],
                 fitting_config: Dict[str, Any],
                 feature_config: Dict[str, Any],
                 energy_shift: List[float],
                 std_mean: Optional[List[torch.Tensor]],
                 ):
        super(NEP, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.std_mean = std_mean
        self.fitting_net = nn.ModuleList()
        self.fitting_net_index = []
        self.R_max_R = feature_config["R_max_R"]
        self.R_max_A = feature_config["R_max_A"]
        self.n_max_R = feature_config["n_max_R"]
        self.n_max_A = feature_config["n_max_A"]
        self.k_basis_R = feature_config["k_basis_R"]
        self.k_basis_A = feature_config["k_basis_A"]
        self.l_3b_max = feature_config["l_3b_max"]
        self.l_4b_max = feature_config["l_4b_max"]
        self.l_5b_max = feature_config["l_5b_max"]
        c2ij_nk = torch.zeros(max(type_map) + 1, max(type_map) + 1, self.n_max_R, self.k_basis_R + 1)
        c3ij_nk = torch.zeros(max(type_map) + 1, max(type_map) + 1, self.n_max_A, self.k_basis_A + 1)
        feature_num = (self.n_max_R + self.n_max_A * self.l_3b_max) * len(type_map)
        for i, element in enumerate(self.type_map):
            self.fitting_net.append(FittingNet(network_size=[feature_num] + fitting_config["network_size"],
                                               activate=fitting_config["activate_function"],
                                               bias=fitting_config["bias"],
                                               resnet_dt=fitting_config["resnet_dt"],
                                               energy_shift=energy_shift[i]))
            self.fitting_net_index.append(str(element))
            c2ij_nk[element] = torch.rand(c2ij_nk[element].shape)
            c3ij_nk[element] = torch.rand(c3ij_nk[element].shape)
        self.c2ij_nk = nn.Parameter(c2ij_nk)
        self.c3ij_nk = nn.Parameter(c3ij_nk)

    def field(self,
              element_map: torch.Tensor,
              Zi: torch.Tensor,
              Nij: torch.Tensor,
              Zij: torch.Tensor,
              image_dR: torch.Tensor,
              n_ghost: int):

        batch, n_atoms, max_neighbor = Nij.shape
        device = image_dR.device
        dtype = image_dR.dtype
        type_map_temp: torch.Tensor = element_map.to(torch.int64)
        type_map: List[int] = type_map_temp.tolist()

        feature = NEP.calculate_feature(type_map=self.type_map,
                                        c2ij_nk=self.c2ij_nk,
                                        c3ij_nk=self.c3ij_nk,
                                        R_max_R=self.R_max_R,
                                        R_max_A=self.R_max_A,
                                        n_max_R=self.n_max_R,
                                        n_max_A=self.n_max_A,
                                        k_basis_R=self.k_basis_R,
                                        k_basis_A=self.k_basis_A,
                                        l_3b_max=self.l_3b_max,
                                        l_4b_max=self.l_4b_max,
                                        l_5b_max=self.l_5b_max,
                                        element_map=element_map,
                                        Zi=Zi, Zij=Zij, Ri=image_dR)
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
    def calculate_fk_rij(R_max: float,
                         k_basis: int,
                         rij: torch.Tensor) -> torch.Tensor:
        batch, n_atoms, _, _ = rij.shape
        R_min = 0.0
        fc_rij = smooth_fun(smooth_type='cos', rij=rij, r_inner=R_min, r_outer=R_max)
        Tk = polynomial_fun(fun_name='chebyshev', n=k_basis, rij=rij, r_inner=R_min, r_outer=R_max)
        fk_rij = 0.5 * (Tk + 1) * fc_rij

        return fk_rij

    @staticmethod
    def calculate_blm(xij: torch.Tensor,
                      yij: torch.Tensor,
                      zij: torch.Tensor,
                      rij: torch.Tensor,
                      R_max: float,
                      l_3b_max: int) -> torch.Tensor:
        batch, n_atoms, max_neighbor, _ = rij.shape
        device = rij.device
        dtype = rij.dtype

        xij = xij.squeeze(-1)
        yij = yij.squeeze(-1)
        zij = zij.squeeze(-1)
        xij_2 = xij ** 2
        yij_2 = yij ** 2
        zij_2 = zij ** 2
        rij_2 = xij_2 + yij_2 + zij_2

        blm = torch.zeros(batch, n_atoms, max_neighbor, 24, dtype=dtype, device=device)
        blm[..., 0] = zij
        blm[..., 1] = xij
        blm[..., 2] = yij
        blm[..., 3] = 3 * zij_2 - rij_2
        blm[..., 4] = xij * zij
        blm[..., 5] = yij * zij
        blm[..., 6] = xij_2 - yij_2
        blm[..., 7] = 2 * xij * yij
        blm[..., 8] = (5 * zij_2 - 3 * rij_2) * zij
        blm[..., 9] = (5 * zij_2 - rij_2) * xij
        blm[..., 10] = (5 * zij_2 - rij_2) * yij
        blm[..., 11] = (xij_2 - yij_2) * zij
        blm[..., 12] = 2 * xij * yij * zij
        blm[..., 13] = (xij_2 - 3 * yij_2) * xij
        blm[..., 14] = (3 * xij_2 - yij_2) * yij
        blm[..., 15] = (35 * zij_2 - 30 * rij_2) * zij_2 + 3 * rij_2 ** 2
        blm[..., 16] = (7 * zij_2 - 3 * rij_2) * xij * zij
        blm[..., 17] = (7 * zij_2 - 3 * rij_2) * yij * zij
        blm[..., 18] = (7 * zij_2 - rij_2) * (xij_2 - yij_2)
        blm[..., 19] = (7 * zij_2 - rij_2) * 2 * xij * yij
        blm[..., 20] = (xij_2 - 3 * yij_2) * xij * zij
        blm[..., 21] = (3 * xij_2) * yij * zij
        blm[..., 22] = (xij_2 - yij_2) ** 2 - 4 * xij_2 * yij_2
        blm[..., 23] = 4 * (xij_2 - yij_2) * xij * yij

        mask = rij.squeeze(-1) > 1e-5
        blm[mask] = blm[mask] / rij[mask]

        return blm

    @staticmethod
    def calculate_s_nlm(blm: torch.Tensor,
                        gn_rij: torch.Tensor,
                        rij: torch.Tensor):

        batch, n_atoms, max_neighbor, _ = rij.shape
        device = rij.device
        dtype = rij.dtype
        mask_rij = (rij[:, :, :, 0:1] > 1e-5)
        rr = torch.zeros(rij[:, :, :, 0:1].shape, dtype=dtype, device=device)
        rr[mask_rij] = 1 / rij[:, :, :, 0:1][mask_rij]
        S_nlm = (gn_rij.unsqueeze(-1) * (blm * rr).unsqueeze(-2))

        return S_nlm

    @staticmethod
    def calculate_feature(type_map: List[int],
                          c2ij_nk: torch.Tensor,
                          c3ij_nk: torch.Tensor,
                          R_max_R: float,
                          R_max_A: float,
                          n_max_R: int,
                          n_max_A: int,
                          k_basis_R: int,
                          k_basis_A: int,
                          l_3b_max: int,
                          l_4b_max: int,
                          l_5b_max: int,
                          element_map: torch.Tensor,
                          Zi: torch.Tensor,
                          Zij: torch.Tensor,
                          Ri: torch.Tensor):
        batch, n_atoms, max_neighbor, _ = Ri.shape
        dtype = Ri.dtype
        device = Ri.device
        include_element_list = element_map.to(torch.int64)

        C3lm = torch.tensor(data=[0.238732414637843, 0.238732414637843, 0.238732414637843, 0.099471839432435,
                                  1.193662073189215, 1.193662073189215, 0.298415518297303, 0.298415518297303,
                                  0.139260575205408, 0.208890862808112, 0.208890862808112, 2.088908628081126,
                                  2.088908628081126, 0.348151438013521, 0.348151438013521, 0.011190581936148,
                                  0.447623277445955, 0.447623277445955, 0.223811638722977, 0.223811638722977,
                                  3.133362942121689, 3.133362942121689, 0.391670367765211, 0.391670367765211],
                            dtype=dtype, device=device)
        C4lm = torch.tensor(data=[-0.00749948082666, -0.13499065487995, 0.067495327439977, 0.404971964639861,
                                  -0.80994392927972],
                            dtype=dtype, device=device)

        C5lm = torch.tensor(data=[0.026596810706114, 0.053193621412227, 0.026596810706114], dtype=dtype, device=device)
        fk_rij = NEP.calculate_fk_rij(R_max=max(R_max_R, R_max_A),
                                      k_basis=max(k_basis_R, k_basis_A),
                                      rij=Ri[..., 0:1])
        fk_rij_2 = torch.zeros(fk_rij.shape, dtype=dtype, device=device)
        mask = Ri[..., 0] < R_max_R
        fk_rij_2[mask] = fk_rij[mask]
        c2ij_n = c2ij_nk[Zi.unsqueeze(-1), Zij]
        gn2_rij = (torch.mul(fk_rij.unsqueeze(-2), c2ij_n).squeeze(-1)).sum(dim=-1)
        gn = torch.zeros(batch, n_atoms, len(type_map) * n_max_R, dtype=dtype, device=device)
        for i, itype in enumerate(include_element_list):
            indices = type_map.index(itype)
            mask_gn = (Zij == itype)
            gn[:, :, n_max_R * indices:n_max_R * (indices + 1)] = torch.sum(gn2_rij * mask_gn.unsqueeze(-1), dim=2)
        feature = gn
        if l_3b_max > 0:
            blm = NEP.calculate_blm(xij=Ri[..., 1:2], yij=Ri[..., 2:3], zij=Ri[..., 3:4], rij=Ri[..., 0:1], R_max=6.0, l_3b_max=4)

            fk_rij_3 = torch.zeros(fk_rij.shape, dtype=dtype, device=device)
            mask = Ri[..., 0] < R_max_A
            fk_rij_3[mask] = fk_rij[mask]
            c3ij_n = c3ij_nk[Zi.unsqueeze(-1), Zij]
            gn3_rij = (torch.mul(fk_rij.unsqueeze(-2), c3ij_n).squeeze(-1)).sum(dim=-1)
            S_nlm = NEP.calculate_s_nlm(blm=blm,
                                        gn_rij=gn3_rij,
                                        rij=Ri[..., 0:1])
            Cs = torch.zeros(batch, n_atoms, len(type_map), n_max_A, 24, dtype=dtype, device=device)
            for i, itype in enumerate(include_element_list):
                indices = type_map.index(itype)
                mask_s = (Zij == itype)[..., None, None]
                Cs[:, :, indices] = torch.sum(S_nlm * mask_s, dim=2)

            Cs = Cs ** 2 * C3lm
            qnl = torch.zeros(batch, n_atoms, len(type_map), n_max_A, l_3b_max, dtype=dtype, device=device)
            qnl[..., 0] = Cs[..., 0:3].sum(dim=-1)
            qnl[..., 1] = Cs[..., 3:8].sum(dim=-1)
            qnl[..., 2] = Cs[..., 8:15].sum(dim=-1)
            qnl[..., 3] = Cs[..., 15:24].sum(dim=-1)
            qnl = qnl.reshape(batch, n_atoms, n_max_A * l_3b_max * len(type_map))
            feature = torch.cat([gn, qnl], dim=-1)
        return feature



