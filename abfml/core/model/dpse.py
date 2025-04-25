import torch
import torch.nn as nn
from abfml.core.model.method import FieldModel
from typing import List, Optional, Dict, Any, Union
from abfml.core.model.math_fun import smooth_fun
from abfml.core.model.network import EmbeddingNet, FittingNet



@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DpSe2a(FieldModel):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: Union[Dict[int, int], int],
                 fitting_config: Dict[str, Any],
                 embedding_config: Dict[str, Any],
                 energy_shift: List[float],
                 std_mean: Optional[List[torch.Tensor]]):
        super(DpSe2a, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.M2 = embedding_config["M2"]
        self.R_min = embedding_config["R_min"]
        self.R_max = embedding_config["R_max"]
        self.smooth_fun = embedding_config['smooth_fun']

        self.embedding_net = nn.ModuleList()
        self.fitting_net = nn.ModuleList()
        self.embedding_net_index = []
        self.fitting_net_index = []
        self.embedding_size = embedding_config["network_size"]
        self.std_mean = std_mean

        for i, itype in enumerate(self.type_map):
            for j, jtype in enumerate(self.type_map):
                self.embedding_net.append(EmbeddingNet(network_size=embedding_config["network_size"],
                                                       activate=embedding_config["activate_function"],
                                                       bias=embedding_config["bias"],
                                                       resnet_dt=embedding_config["resnet_dt"]))
                self.embedding_net_index.append(str(itype) + "_" + str(jtype))
            fit_network_size = [embedding_config["network_size"][-1] * embedding_config["M2"]] + fitting_config[
                "network_size"]
            self.fitting_net.append(FittingNet(network_size=fit_network_size,
                                               activate=fitting_config["activate_function"],
                                               bias=fitting_config["bias"],
                                               resnet_dt=fitting_config["resnet_dt"],
                                               energy_shift=energy_shift[i]))
            self.fitting_net_index.append(str(itype))

    def field(self,
              element_map: torch.Tensor,
              central_atoms: torch.Tensor,
              neighbor_indices: torch.Tensor,
              neighbor_types: torch.Tensor,
              neighbor_vectors: torch.Tensor,
              n_ghost: int) -> Dict[str, torch.Tensor]:
        # t1 = time.time()
        batch, n_atoms, max_neighbor = neighbor_types.shape
        device, dtype = neighbor_vectors.device, neighbor_vectors.dtype
        type_map: List[int] = element_map.to(torch.int64).tolist()
        width_map: List[int] = [0]
        for element in type_map:
            width_map.append(self.neighbor[element] + width_map[-1])
        # t2 = time.time()
        # Ri[batch, n_atoms, max_neighbor, 4] 4-->(srij, srij * xij/rij, srij * zij/rij, srij * zij/rij)
        Ri = DpSe2a.calculate_coordinate_matrix(R_min=self.R_min, R_max=self.R_max,
                                                smooth_type=self.smooth_fun, Rij=neighbor_vectors)
        Ri = DpSe2a.scale(self.type_map, type_map, self.std_mean, Ri, central_atoms)

        # Ei[batch, n_atoms, 1]
        Ei = torch.zeros(batch, n_atoms, 1, dtype=dtype, device=device)
        for i, itype in enumerate(type_map):
            mask_itype = (central_atoms == itype)
            if not mask_itype.any():
                continue
            i_Ri = Ri[mask_itype].reshape(batch, -1, max_neighbor, 4)
            RiT_G = torch.zeros(batch, i_Ri.shape[1], 1, self.embedding_size[-1], dtype=dtype, device=device)

            for j, jtype in enumerate(type_map):
                # srij[batch, n_atoms of itype, max_neighbor, 1]
                srij = i_Ri[:, :, width_map[j]:width_map[j+1], 0].unsqueeze(-1)
                ij = self.embedding_net_index.index(str(itype) + "_" + str(jtype))
                embedding_net: ModuleInterface = self.embedding_net[ij]
                # G[batch, n_atoms of itype, max_neighbor, embedding_net_size[-1]]
                G = embedding_net.forward(srij)
                RiT = i_Ri[:, :, width_map[j]:width_map[j+1], :].transpose(-2, -1)
                temp_b = torch.matmul(RiT, G)
                RiT_G = RiT_G + temp_b

            RiT_G = RiT_G / max_neighbor
            RiT_GM = RiT_G[:, :, :, :self.M2]
            # Di[batch, n_atoms of itype, M2 * embedding_net_size[-1]]
            Di = torch.matmul(RiT_G.transpose(-2, -1), RiT_GM).reshape(batch, i_Ri.shape[1], -1)
            ii = self.fitting_net_index.index(str(itype))
            fitting_net: ModuleInterface = self.fitting_net[ii]
            # Ei_mask[batch, n_atoms of itype, 1]
            Ei_mask = fitting_net.forward(Di)
            Ei[mask_itype] = Ei_mask.reshape(-1, 1)

        Etot = torch.sum(Ei, dim=1)
        return {'Etot': Etot, 'Ei':Ei}

    @staticmethod
    def calculate_coordinate_matrix(R_min: float,
                                    R_max: float,
                                    smooth_type: str,
                                    Rij: torch.Tensor) -> torch.Tensor:
        batch, n_atoms, max_neighbor, _ = Rij.shape
        device, dtype = Rij.device, Rij.dtype
        rij = Rij[:, :, :, 0]
        xij = Rij[:, :, :, 1]
        yij = Rij[:, :, :, 2]
        zij = Rij[:, :, :, 3]

        # Guaranteed not to divide by 0
        mask_rij = (rij > 1e-5)
        rr = torch.zeros(rij.shape, dtype=dtype, device=device)
        rr[mask_rij] = 1 / rij[mask_rij]
        r2 = torch.zeros(rij.shape, dtype=dtype, device=device)
        r2[mask_rij] = 1 / rij[mask_rij] ** 2

        Srij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)

        # Ri[batch, n_atoms, max_neighbor, 4] 4-->[Srij, Srij * xij / rij, Srij * yij / rij, Srij * yij / rij]
        Ri = torch.zeros(batch, n_atoms, max_neighbor, 4, dtype=dtype, device=device)
        Ri[:, :, :, 0] = Srij
        Ri[:, :, :, 1] = Srij * xij * rr
        Ri[:, :, :, 2] = Srij * yij * rr
        Ri[:, :, :, 3] = Srij * zij * rr

        # t3 = time.time()
        # print(t3-t2, t2-t1, t3-t1)
        return Ri

    @staticmethod
    def scale(type_map_all: List[int],
              type_map_use: List[int],
              std_mean: List[torch.Tensor],
              Ri: torch.Tensor,
              central_atoms: torch.Tensor) -> torch.Tensor:
        for i, element in enumerate(type_map_use):
            indices = type_map_all.index(element)
            device = Ri.device
            mask = (central_atoms == element)
            std = std_mean[0][indices].detach().to(device)
            avg = std_mean[1][indices].detach().to(device)
            Ri[mask] = (Ri[mask] - avg) / std
        return Ri


class DpSe2r(FieldModel):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: Union[Dict[int, int], int],
                 fitting_config: Dict[str, Any],
                 embedding_config: Dict[str, Any],
                 energy_shift: List[float],
                 std_mean: Optional[List[torch.Tensor]]):
        super(DpSe2r, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.type_map = type_map
        self.neighbor = neighbor
        self.cutoff = cutoff
        self.M2 = embedding_config["M2"]
        self.R_min = embedding_config["R_min"]
        self.R_max = embedding_config["R_max"]
        self.smooth_fun = embedding_config['smooth_fun']

        self.embedding_net = nn.ModuleList()
        self.fitting_net = nn.ModuleList()
        self.embedding_net_index = []
        self.fitting_net_index = []
        self.embedding_size = embedding_config["network_size"]
        self.std_mean = std_mean

        for i, itype in enumerate(self.type_map):
            for j, jtype in enumerate(self.type_map):
                self.embedding_net.append(EmbeddingNet(network_size=embedding_config["network_size"],
                                                       activate=embedding_config["activate_function"],
                                                       bias=embedding_config["bias"],
                                                       resnet_dt=embedding_config["resnet_dt"]))
                self.embedding_net_index.append(str(itype) + "_" + str(jtype))
            fit_network_size = [embedding_config["network_size"][-1] * embedding_config["M2"]] + fitting_config[
                "network_size"]
            self.fitting_net.append(FittingNet(network_size=fit_network_size,
                                               activate=fitting_config["activate_function"],
                                               bias=fitting_config["bias"],
                                               resnet_dt=fitting_config["resnet_dt"],
                                               energy_shift=energy_shift[i]))
            self.fitting_net_index.append(str(itype))

    def field(self,
              element_map: torch.Tensor,
              central_atoms: torch.Tensor,
              neighbor_indices: torch.Tensor,
              neighbor_types: torch.Tensor,
              neighbor_vectors: torch.Tensor,
              n_ghost: int):
        # time1 = time.time()
        batch, n_atoms, max_neighbor = neighbor_types.shape
        device, dtype = neighbor_vectors.device, neighbor_vectors.dtype
        # Must be set to int64, or it will cause a type mismatch when run in c++.
        type_map = element_map.to(torch.int64).tolist()
        width_map: List[int] = [0]
        for ii in type_map:
            width_map.append(self.neighbor[self.type_map.index(ii)] + width_map[-1])
        # Ri[batch, n_atoms, max_neighbor,1] 1-->(srij) For se_r consider only Srij

        Ri = DpSe2r.calculate_coordinate_matrix(R_min=self.R_min, R_max=self.R_max,
                                                smooth_type=self.smooth_fun, Rij=neighbor_vectors)
        Ri = DpSe2r.scale(self.type_map, type_map, self.std_mean, Ri, central_atoms)
        # Ei[batch, n_atoms, 1]
        Ei = torch.zeros(batch, n_atoms, 1, dtype=dtype, device=device)
        for i, itype in enumerate(type_map):
            mask_itype = (central_atoms == itype)
            if not mask_itype:
                continue
            i_Ri = Ri[mask_itype].reshape(batch, -1, max_neighbor, 1)
            Gi = torch.zeros(batch, i_Ri.shape[1], max_neighbor, self.embedding_size[-1], dtype=dtype, device=device)
            for j, jtype in enumerate(type_map):
                # srij[batch, n_atoms of itype, max_neighbor, 1]
                srij = i_Ri[:, :, width_map[j]:width_map[j+1], 0].unsqueeze(-1)
                ij = self.embedding_net_index.index(str(itype) + "_" + str(jtype))
                embedding_net: ModuleInterface = self.embedding_net[ij]
                # G[batch, n_atoms of itype, max_neighbor, embedding_net_size[-1]]
                Gj = embedding_net.forward(srij)
                Gi[:, :, width_map[j]:width_map[j+1], :] = Gj
            G = torch.einsum('bijm,bikm->bim', Gi, Gi)
            G = G / max_neighbor ** 2
            Di = G.reshape(batch, i_Ri.shape[1], -1)
            # Di[batch, n_atoms of itype, embedding_net_size[-1]]
            ii = self.fitting_net_index.index(str(itype))
            fitting_net: ModuleInterface = self.fitting_net[ii]
            # Ei_mask[batch, n_atoms of itype, 1]
            Ei_mask = fitting_net.forward(Di)
            Ei[mask_itype] = Ei_mask.reshape(-1, 1)
        # Etot[batch, 1]
        Etot = torch.sum(Ei, dim=1)

        return {'Etot': Etot, 'Ei':Ei}

    @staticmethod
    def calculate_coordinate_matrix(R_min: float,
                                    R_max: float,
                                    smooth_type: str,
                                    Rij: torch.Tensor) -> torch.Tensor:
        batch, n_atoms, max_neighbor, _ = Rij.shape
        device = Rij.device
        dtype = Rij.dtype

        rij = Rij[:, :, :, 0]
        Srij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        # Ri [batch, n_atoms, max_neighbor * len(type_map), 1]
        Ri = torch.zeros(batch, n_atoms, max_neighbor, 1, dtype=dtype, device=device)
        Ri[:, :, :, 0] = Srij

        return Ri

    @staticmethod
    def scale(type_map_all: List[int],
              type_map_use: List[int],
              std_mean: List[torch.Tensor],
              Ri: torch.Tensor,
              central_atoms: torch.Tensor) -> torch.Tensor:
        for i, element in enumerate(type_map_use):
            indices = type_map_all.index(element)
            device = Ri.device
            mask = (central_atoms == element)
            std = std_mean[0][indices].detach().to(device)
            avg = std_mean[1][indices].detach().to(device)
            Ri[mask] = (Ri[mask] - avg[0]) / std[0]
        return Ri


class DpSe3(FieldModel):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: Union[Dict[int, int], int],
                 fitting_config: Dict[str, Any],
                 embedding_config: Dict[str, Any],
                 energy_shift: List[float],
                 std_mean: Optional[List[torch.Tensor]]):
        super(DpSe3, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.type_map = type_map
        self.neighbor = neighbor
        self.cutoff = cutoff
        self.M2 = embedding_config["M2"]
        self.R_min = embedding_config["R_min"]
        self.R_max = embedding_config["R_max"]
        self.smooth_fun = embedding_config['smooth_fun']

        self.embedding_net = nn.ModuleList()
        self.fitting_net = nn.ModuleList()
        self.embedding_net_index = []
        self.fitting_net_index = []
        self.embedding_size = embedding_config["network_size"]
        self.std_mean = std_mean

        for i, itype in enumerate(self.type_map):
            for j, jtype in enumerate(self.type_map):
                self.embedding_net.append(EmbeddingNet(network_size=embedding_config["network_size"],
                                                       activate=embedding_config["activate_function"],
                                                       bias=embedding_config["bias"],
                                                       resnet_dt=embedding_config["resnet_dt"]))
                self.embedding_net_index.append(str(itype) + "_" + str(jtype))
            fit_network_size = [self.embedding_size[-1]] + fitting_config[
                "network_size"]
            self.fitting_net.append(FittingNet(network_size=fit_network_size,
                                               activate=fitting_config["activate_function"],
                                               bias=fitting_config["bias"],
                                               resnet_dt=fitting_config["resnet_dt"],
                                               energy_shift=energy_shift[i]))
            self.fitting_net_index.append(str(itype))

    def field(self,
              element_map: torch.Tensor,
              central_atoms: torch.Tensor,
              neighbor_indices: torch.Tensor,
              neighbor_types: torch.Tensor,
              neighbor_vectors: torch.Tensor,
              n_ghost: int):
        batch, n_atoms, max_neighbor = neighbor_types.shape
        device, dtype = neighbor_vectors.device, neighbor_vectors.dtype
        # Must be set to int64, or it will cause a type mismatch when run in c++.
        type_map: List[int] = element_map.to(torch.int64).tolist()
        width_map: List[int] = [0]
        for ii in type_map:
            width_map.append(self.neighbor[self.type_map.index(ii)] + width_map[-1])

        # Ri[batch, n_atoms, max_neighbor, 4] 4-->(srij, srij * xij/rij, srij * zij/rij, srij * zij/rij)
        Ri = DpSe2a.calculate_coordinate_matrix(R_min=self.R_min, R_max=self.R_max,
                                                smooth_type=self.smooth_fun, Rij=neighbor_vectors)
        Ri = DpSe2a.scale(self.type_map, type_map, self.std_mean, Ri, central_atoms)

        rij = neighbor_vectors[:, :, :, 0].unsqueeze(-1)
        # Guaranteed not to divide by 0
        mask_rij = (rij > 1e-5)
        rr = torch.zeros(rij.shape, dtype=dtype, device=device)
        rr[mask_rij] = 1 / rij[mask_rij]
        # theta_ijk [batch, n_atoms, max_neighbor, max_neighbor]
        theta_ijk = (Ri[:, :, :, 0].unsqueeze(-1) * Ri[:, :, :, 0].unsqueeze(-2)
                     * (torch.matmul(neighbor_vectors[:, :, :, 1:], neighbor_vectors[:, :, :, 1:].transpose(-1, -2))
                     * (rr * rr.transpose(-2, -1))))

        # Ei[batch, n_atoms, 1]
        Ei = torch.zeros(batch, n_atoms, 1, dtype=dtype, device=device)
        for i, itype in enumerate(type_map):

            mask_itype = (central_atoms == itype)
            if not mask_itype.any():
                continue
            i_theta_ijk = theta_ijk[mask_itype].reshape(batch, -1, max_neighbor, max_neighbor, 1)
            Gi = torch.zeros(batch, i_theta_ijk.shape[1], max_neighbor, max_neighbor, self.embedding_size[-1],
                             dtype=dtype, device=device)

            for j, jtype in enumerate(type_map):
                theta = i_theta_ijk[:, :, width_map[j]:width_map[j+1], :, 0].unsqueeze(-1)
                ij = self.embedding_net_index.index(str(itype) + "_" + str(jtype))
                embedding_net: ModuleInterface = self.embedding_net[ij]
                # G[batch, n_atoms of itype, max_neighbor, max_neighbor, embedding_net_size[-1]]
                Gj = embedding_net.forward(theta)
                Gi[:, :, width_map[j]:width_map[j + 1], :, :] = Gj
            # Di[batch, n_atoms of itype, embedding_size[-1]]
            Di = (torch.mul(i_theta_ijk, Gi)).sum(dim=[2, 3]) / max_neighbor ** 2
            ii = self.fitting_net_index.index(str(itype))
            fitting_net: ModuleInterface = self.fitting_net[ii]
            # Ei_mask[batch, n_atoms of itype, 1]
            Ei_mask = fitting_net.forward(Di)
            Ei[mask_itype] = Ei_mask.reshape(-1, 1)
        Etot = torch.sum(Ei, dim=1)
        return {'Etot': Etot, 'Ei':Ei}
