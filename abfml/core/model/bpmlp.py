import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple, Any, Union
from abfml.core.model.math_fun import smooth_fun
from abfml.core.model.network import FittingNet, AtomFitNet
from abfml.core.model.method import FieldModel


@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BPMlp(FieldModel):
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: Union[Dict[int, int], int],
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
              central_atoms: torch.Tensor,
              neighbor_indices: torch.Tensor,
              neighbor_types: torch.Tensor,
              neighbor_vectors: torch.Tensor,
              n_ghost: int):
        batch, n_atoms, max_neighbor = neighbor_types.shape
        device, dtype = neighbor_vectors.device, neighbor_vectors.dtype
        type_map_temp: torch.Tensor = element_map.to(torch.int64)
        type_map: List[int] = type_map_temp.tolist()
        feature = BPMlp.calculate_bp_feature(type_map=self.type_map,
                                             bp_features_information=self.bp_features_information,
                                             bp_features_param=self.bp_features_param,
                                             element_map=element_map,
                                             neighbor_indices=neighbor_indices,
                                             neighbor_types=neighbor_types,
                                             neighbor_vectors=neighbor_vectors)
        feature = BPMlp.scale(self.type_map, type_map, self.std_mean, feature, central_atoms)
        Ei = torch.zeros(batch, n_atoms, 1, dtype=dtype, device=device)
        for i, itype in enumerate(type_map):
            mask_itype = (central_atoms == itype)
            if not mask_itype.any():
                continue
            iifeat = feature[mask_itype].reshape(batch, -1, feature.shape[-1])
            ii = self.fitting_net_index.index(str(itype))
            fitting_net: ModuleInterface = self.fitting_net[ii]
            Ei_itype = fitting_net.forward(iifeat)
            Ei[mask_itype] = Ei_itype.reshape(-1, 1)
        Etot = torch.sum(Ei, dim=1)
        return {'Etot': Etot, 'Ei':Ei}

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
                             neighbor_indices: torch.Tensor,
                             neighbor_types: torch.Tensor,
                             neighbor_vectors: torch.Tensor,) -> torch.Tensor:
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
                                            neighbor_indices=neighbor_indices,
                                            neighbor_types=neighbor_types,
                                            neighbor_vectors=neighbor_vectors)
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
                                                neighbor_indices=neighbor_indices,
                                                neighbor_types=neighbor_types,
                                                neighbor_vectors=neighbor_vectors)
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
                                                neighbor_indices=neighbor_indices,
                                                neighbor_types=neighbor_types,
                                                neighbor_vectors=neighbor_vectors)
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
                neighbor_indices: torch.Tensor,
                neighbor_types: torch.Tensor,
                neighbor_vectors: torch.Tensor,) -> torch.Tensor:
        batch, n_atoms, max_neighbor = neighbor_types.shape
        device, dtype = neighbor_vectors.device, neighbor_vectors.dtype
        n_basis = eta.shape[0]
        include_element_list: List[int] = element_map.to(torch.int64).tolist()

        rij = neighbor_vectors[:, :, :, 0].unsqueeze(-1)

        f_rij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        g_rij = torch.exp(-eta * (rij - rs) ** 2) * f_rij
        Gi = torch.zeros(batch, n_atoms, int(n_basis * len(type_map)), dtype=dtype, device=device)

        for i, itype in enumerate(include_element_list):
            indices = type_map.index(itype)
            mask_g = (neighbor_indices == itype).unsqueeze(-1)
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
                  neighbor_indices: torch.Tensor,
                  neighbor_types: torch.Tensor,
                  neighbor_vectors: torch.Tensor,) -> torch.Tensor:
        batch, n_atoms, max_neighbor = neighbor_types.shape
        device, dtype = neighbor_vectors.device, neighbor_vectors.dtype
        n_basis = eta.shape[0]
        include_element_list: List[int] = element_map.to(torch.int64).tolist()
        rij = neighbor_vectors[:, :, :, 0:1]
        # Guaranteed not to divide by 0
        mask_rij = (rij > 1e-5)
        rr = torch.zeros(rij.shape, dtype=dtype, device=device)
        rr[mask_rij] = 1 / rij[mask_rij]

        frij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        grij = torch.exp(-eta * (rij - rs) ** 2) * frij
        cos_ijk = (torch.matmul(neighbor_vectors[:, :, :, 1:], neighbor_vectors[:, :, :, 1:].transpose(-1, -2))
                   * (rr * rr.transpose(-2, -1))).unsqueeze(2)
        Gi = torch.zeros(batch, n_atoms, int(n_basis * len(type_map) ** 2), dtype=dtype, device=device)

        for i, itype in enumerate(include_element_list):
            i_indices = type_map.index(itype)
            mask_i = (neighbor_indices == itype).unsqueeze(-1)
            gij = grij * mask_i
            # gij [batch, n_atoms, n_basis, max_neighbor, 1]
            gij = gij.transpose(2, 3).unsqueeze(-1)

            zeta = zeta.reshape(1, 1, n_basis, 1, 1)
            lambd = lambd.reshape(1, 1, n_basis, 1, 1)
            # t41 = time.time()
            for j, jtype in enumerate(include_element_list):
                j_indices = type_map.index(jtype)
                indices = i_indices * len(type_map) + j_indices
                mask_j = (neighbor_indices == itype).unsqueeze(-1)
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
                  neighbor_indices: torch.Tensor,
                  neighbor_types: torch.Tensor,
                  neighbor_vectors: torch.Tensor,):
        n_basis = eta.shape[0]
        batch, n_atoms, max_neighbor = neighbor_types.shape
        device, dtype = neighbor_vectors.device, neighbor_vectors.dtype
        include_element_list: List[int] = element_map.to(torch.int64).tolist()
        rij = neighbor_vectors[:, :, :, 0:1]

        # Guaranteed not to divide by 0
        mask_rij = (rij > 1e-5)
        rr = torch.zeros(rij.shape, dtype=dtype, device=device)
        rr[mask_rij] = 1 / rij[mask_rij]

        frij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        grij = torch.exp(-eta * (rij - rs) ** 2) * frij
        # cos_ijk [batch, n_atoms, 1,max_neighbor, max_neighbor]
        cos_ijk = (torch.matmul(neighbor_vectors[:, :, :, 1:], neighbor_vectors[:, :, :, 1:].transpose(-1, -2))
                   * (rr * rr.transpose(-2, -1))).unsqueeze(2)
        rjk2 = ((neighbor_vectors[:, :, :, 1:].unsqueeze(3) - neighbor_vectors[:, :, :, 1:].unsqueeze(3).permute(0, 1, 3, 2, 4)) ** 2).sum(-1)
        mask_rjk = (rjk2 > 1e-5)
        rjk = torch.zeros(rjk2.shape, dtype=dtype, device=device)
        rjk[mask_rjk] = rjk[mask_rjk] ** 0.5
        frjk = smooth_fun(smooth_type=smooth_type, rij=rjk, r_inner=R_min, r_outer=R_max)
        grjk = (torch.exp(-eta * (rjk.unsqueeze(-1) - rs) ** 2) * frjk.unsqueeze(-1)).transpose(2, 4).squeeze(-1)

        Gi = torch.zeros(batch, n_atoms, int(n_basis * len(type_map) ** 2), dtype=dtype, device=device)
        for i, itype in enumerate(include_element_list):
            i_indices = type_map.index(itype)
            mask_i = (neighbor_indices == itype).unsqueeze(-1)
            gij = grij * mask_i
            # gij [batch, n_atoms, n_basis, max_neighbor, 1]
            gij = gij.transpose(2, 3).unsqueeze(-1)

            zeta = zeta.reshape(1, 1, n_basis, 1, 1)
            lambd = lambd.reshape(1, 1, n_basis, 1, 1)

            # t41 = time.time()
            for j, jtype in enumerate(include_element_list):
                j_indices = type_map.index(jtype)
                indices = i_indices * len(type_map) + j_indices
                mask_j = (neighbor_indices == itype).unsqueeze(-1)
                gik = grij * mask_j
                # gik [batch, n_atoms, n_basis, 1, max_neighbor]
                gik = gik.transpose(2, 3).unsqueeze(-2)
                # Gi_temp [batch, n_atoms, n_basis, max_neighbor, max_neighbor]
                Gi_temp = 2 ** (1 - zeta) * ((1 + lambd * cos_ijk) ** zeta) * (gij * gik * grjk)
                Gi[:, :,  n_basis * indices:n_basis * (indices + 1)] = Gi_temp.sum(dim=[-1, -2])
        return Gi


class BPNN(FieldModel):
    """
    This is a reference to the “https://github.com/cc-ats/mlp_tutorial” tutorial.
    """
    def __init__(self,
                 type_map: List[int],
                 cutoff: float,
                 neighbor: List[int],
                 config: Dict[str, Any],
                 normal: Optional[List[torch.Tensor]],
                 ):
        super(BPNN, self).__init__(type_map=type_map, cutoff=cutoff, neighbor=neighbor)
        self.type_map = type_map
        self.neighbor = neighbor
        self.cutoff = cutoff
        descriptor_name = [descriptor_name['type_name'] for descriptor_name in config['list']]
        g1_config, g2_config, g3_config = None, None, None
        if 'SymFunRad' in descriptor_name:
            g1_config = next((config_dict for config_dict in config['list'] if config_dict.get('type_name') == 'SymFunRad'), None)
            g1_config['rs'] =  torch.Tensor(g1_config['rs'])
            g1_config['eta'] = torch.Tensor(g1_config['eta'])
            if g1_config['eta'].shape[0] != len(type_map):
                g1_config['rs'] = g1_config['rs'].repeat(len(type_map), 1)
                g1_config['eta'] = g1_config['eta'].repeat(len(type_map), 1)
        if 'SymFunAng' in descriptor_name:
            g2_config = next((config_dict for config_dict in config['list'] if config_dict.get('type_name') == 'SymFunAng'), None)
            g2_config['rs'] = torch.Tensor(g2_config['rs'])
            g2_config['eta'] = torch.Tensor(g2_config['eta'])
            g2_config['lambd'] = torch.Tensor(g2_config['lambd'])
            g2_config['zeta'] = torch.Tensor(g2_config['zeta'])
            if g2_config['zeta'].shape[0] != len(type_map):
                g2_config['lambd'] = g2_config['lambd'].repeat(len(type_map), 1)
                g2_config['zeta'] = g2_config['zeta'].repeat(len(type_map), 1)
                g2_config['rs'] = g2_config['rs'].repeat(len(type_map), 1)
                g2_config['eta'] = g2_config['eta'].repeat(len(type_map), 1)
        if 'SymFunAngN' in descriptor_name:
            g3_config = next((config_dict for config_dict in config['list'] if config_dict.get('type_name') == 'SymFunAngN'), None)
            g3_config['rs'] = torch.Tensor(g3_config['rs'])
            g3_config['eta'] = torch.Tensor(g3_config['eta'])
            g3_config['lambd'] = torch.Tensor(g3_config['lambd'])
            g3_config['zeta'] = torch.Tensor(g3_config['zeta'])
            if g3_config['zeta'].shape[0] != len(type_map):
                g3_config['lambd'] = g3_config['lambd'].repeat(len(type_map), 1)
                g3_config['zeta'] = g3_config['zeta'].repeat(len(type_map), 1)
                g3_config['rs'] = g3_config['rs'].repeat(len(type_map), 1)
                g3_config['eta'] = g3_config['eta'].repeat(len(type_map), 1)

        if descriptor_name == ['SymFunRad','SymFunAng','SymFunAngN']:
            self.descriptor = Feature()
        elif descriptor_name == ['SymFunRad','SymFunAngN']:
            self.descriptor = Feature(g1_Rmax=g1_config['R_max'], g1_Rmin=g1_config['R_min'], g1_eta=g1_config['eta'],
                                             g1_smooth_type=g1_config['smooth_type'], g1_rs=g1_config['rs'],
                                             g2_Rmax=g2_config['R_max'], g2_Rmin=g2_config['R_min'], g2_eta=g2_config['eta'],
                                             g2_smooth_type=g2_config['smooth_fun'], g2_lambd=g2_config['lambd'],
                                             g2_theta=g2_config['theta'], g2_rs=g2_config['rs'])
        elif descriptor_name == ['SymFunRad']:
            self.descriptor= Feature(g1_Rmax=g1_config['R_max'],g1_Rmin=g1_config['R_min'],g1_eta=g1_config['eta'],
                                             g1_smooth_type=g1_config['smooth_fun'],g1_rs=g1_config['rs'])
        else:
            raise Exception(f"Symmetric function configuration error")
        feature_length = self.descriptor.feature_length
        self.fitting_net = AtomFitNet(num_channels=len(type_map), network_size= [feature_length,200,220,100,1],
                                      activate='tanh',bias=True, resnet_dt= False)


    def field(self,
              element_map: torch.Tensor,
              central_atoms: torch.Tensor,
              neighbor_indices: torch.Tensor,
              neighbor_types: torch.Tensor,
              neighbor_vectors: torch.Tensor,
              n_ghost: int):
        device, dtype = neighbor_vectors.device, neighbor_vectors.dtype
        type_map_tensor = torch.tensor(self.type_map)
        mapped_value = torch.arange(0, len(self.type_map))
        mapped_table = torch.full(size=(int(torch.max(type_map_tensor).item()) + 1, ), fill_value=-1).to(device=device)
        mapped_table[type_map_tensor] = mapped_value
        mapped_Zi = mapped_table[central_atoms]
        feature = self.descriptor(mapped_Zi=mapped_Zi, neighbor_vectors=neighbor_vectors)
        Ei = self.fitting_net(feature, mapped_Zi)
        Etot = torch.sum(Ei,dim=1)
        return Etot, Ei


class Feature(nn.Module):
    def __init__(self,
                 g1_Rmin: Optional[float] = None,
                 g1_Rmax: Optional[float] = None,
                 g1_eta: Optional[torch.Tensor] = None,
                 g1_smooth_type: Optional[str] = None,
                 g1_rs: Optional[torch.Tensor] = None,
                 g2_Rmax: Optional[float] = None,
                 g2_Rmin: Optional[float] = None,
                 g2_eta: Optional[torch.Tensor] = None,
                 g2_smooth_type: Optional[str] = None,
                 g2_lambd: Optional[torch.Tensor] = None,
                 g2_theta: Optional[torch.Tensor] = None,
                 g2_rs: Optional[torch.Tensor] = None,
                 g3_Rmin: Optional[float] = None,
                 g3_Rmax: Optional[float] = None,
                 g3_eta: Optional[torch.Tensor] = None,
                 g3_smooth_type: Optional[str] = None,
                 g3_lambd: Optional[torch.Tensor] = None,
                 g3_theta: Optional[torch.Tensor] = None,
                 g3_rs: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the Feature class with the specified parameters.

        Parameters for g1 (radial symmetry functions):
        - g1_Rmin, g1_Rmax: Radial distance range.
        - g1_eta: Scaling parameter for radial functions.
        - g1_smooth_type: Type of smoothing function ('cosine' or 'linear').
        - g1_rs: Shift parameters for radial symmetry functions.

        Parameters for g2 (angular symmetry functions with angular dependencies):
        - g2_Rmin, g2_Rmax: Distance range.
        - g2_eta: Scaling parameter for angular functions.
        - g2_smooth_type: Type of smoothing function ('cosine' or 'linear').
        - g2_lambd: Lambda parameter for angular symmetry.
        - g2_theta: Angular shift parameters.
        - g2_rs: Shift parameters for angular symmetry functions.

        Parameters for g3 (angular symmetry functions with triple dependencies):
        - g3_Rmin, g3_Rmax: Distance range.
        - g3_eta: Scaling parameter for triple angular functions.
        - g3_smooth_type: Type of smoothing function ('cosine' or 'linear').
        - g3_lambd: Lambda parameter for triple angular symmetry.
        - g3_theta: Angular shift parameters.
        - g3_rs: Shift parameters for triple angular symmetry functions.
        """
        super().__init__()
        # Radial symmetry function parameters
        self.g1_Rmin = g1_Rmin
        self.g1_Rmax = g1_Rmax
        self.g1_eta = g1_eta
        self.g1_smooth_type = g1_smooth_type
        self.g1_rs = g1_rs

        # Angular symmetry function parameters (pairwise angular dependencies)
        self.g2_Rmin = g2_Rmin
        self.g2_Rmax = g2_Rmax
        self.g2_eta = g2_eta
        self.g2_smooth_type = g2_smooth_type
        self.g2_lambd = g2_lambd
        self.g2_theta = g2_theta
        self.g2_rs = g2_rs

        # Angular symmetry function parameters (triple dependencies)
        self.g3_Rmin = g3_Rmin
        self.g3_Rmax = g3_Rmax
        self.g3_eta = g3_eta
        self.g3_smooth_type = g3_smooth_type
        self.g3_lambd = g3_lambd
        self.g3_theta = g3_theta
        self.g3_rs = g3_rs

    def forward(self, Rij: torch.Tensor, mapped_Zi: torch.Tensor) -> torch.Tensor:
        device, dtype = Rij.device, Rij.dtype
        feature_list = []
        # Compute g1 features if parameters are set
        if self.g1_eta is not None:
            g1_eta = self.g1_eta.to(device=device, dtype=dtype)[mapped_Zi]
            g1_rs = self.g1_rs.to(device=device, dtype=dtype)[mapped_Zi]
            feature_list.append(Feature.sym_rad(R_min=self.g1_Rmin, R_max=self.g1_Rmax, eta=g1_eta,
                                                smooth_type=self.g1_smooth_type, rs=g1_rs, Rij=Rij))

        # Prepare parameters for g2
        if self.g2_eta is not None:
            g2_eta = self.g2_eta.to(device=device, dtype=dtype)[mapped_Zi]
            g2_lambd = self.g2_lambd.to(device=device, dtype=dtype)[mapped_Zi]
            g2_theta = self.g2_theta.to(device=device, dtype=dtype)[mapped_Zi]
            g2_rs = self.g2_rs.to(device=device, dtype=dtype)[mapped_Zi]
            g2_zeta = self.g2_zeta.to(device=device, dtype=dtype)[mapped_Zi]

            feature_list.append(Feature.sym_ang_n(R_min=self.g2_Rmin, R_max=self.g2_Rmax, lambd=g2_lambd,eta=g2_eta,
                                                  zeta=g2_zeta, theta=g2_theta, rs=g2_rs,smooth_type=self.g2_smooth_type,
                                                  Rij=Rij))

        # Prepare parameters for g3
        if self.g3_eta is not None:
            g3_eta = self.g3_eta.to(device=device, dtype=dtype)[mapped_Zi]
            g3_lambd = self.g3_lambd.to(device=device, dtype=dtype)
            g3_theta = self.g3_theta.to(device=device, dtype=dtype)
            g3_rs = self.g3_rs.to(device=device, dtype=dtype)

            feature_list.append(Feature.sym_ang_w(R_min=self.g3_Rmin, R_max=self.g3_Rmax, lambd=g3_lambd, eta=g3_eta,
                                                  zeta=g3_theta,rs=g3_rs,smooth_type=self.g3_smooth_type,Rij=Rij))

        # Concatenate all computed features
        feature = torch.cat([tensor for tensor in feature_list], dim=-1)
        return feature

    @property
    def feature_length(self)->int:
        lengths = self.g1_eta.shape[-1]
        if self.g2_eta is not None:
            lengths += self.g2_eta[-1]
        if self.g3_eta is not None:
            lengths += self.g3_eta[-1]
        return lengths

    @staticmethod
    def sym_rad(R_min: float,
                R_max: float,
                eta: torch.Tensor,
                rs: torch.Tensor,
                smooth_type: str,
                Rij: torch.Tensor) -> torch.Tensor:
        rij = Rij[:, :, :, 0:1]
        f_rij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        g_rij = torch.exp(-eta.unsqueeze(dim=2) * (rij - rs.unsqueeze(dim=2)) ** 2) * f_rij
        Gi =  torch.sum(g_rij, dim=2)
        return Gi

    @staticmethod
    def sym_ang_n(R_min: float,
                  R_max: float,
                  lambd: torch.Tensor,
                  eta: torch.Tensor,
                  zeta: torch.Tensor,
                  theta: torch.Tensor,
                  rs: torch.Tensor,
                  smooth_type: str,
                  Rij: torch.Tensor) -> torch.Tensor:

        Rij = Rij[:, :, :, 1:4]
        c = torch.combinations(torch.arange(Rij.size(2)), r=2)
        Rij = Rij[:, :, c]
        R_ij = Rij[:, :, :, 0]
        R_ik = Rij[:, :, :, 1]
        rij = torch.norm(R_ij, dim=3)
        rik = torch.norm(R_ik, dim=3)
        f_rij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        f_rik = smooth_fun(smooth_type=smooth_type, rij=rik, r_inner=R_min, r_outer=R_max)
        cosine = torch.einsum('ijkl,ijkl->ijk', R_ij, R_ik) / (rij * rik)
        # cosine = torch.cos(torch.acos(cosine).unsqueeze(dim=-1) - theta.unsqueeze(dim=1))

        Gi = torch.sum(2 ** (1 - zeta.unsqueeze(dim=1)) * (1 + cosine) ** zeta.unsqueeze(dim=1) * torch.exp(
            -eta.unsqueeze(dim=1) * (0.5 * (rij + rik).unsqueeze(dim=-1) - rs.unsqueeze(dim=1)) ** 2) * (
                                   f_rij * f_rik).unsqueeze(dim=-1), dim=2)
        return Gi

    @staticmethod
    def sym_ang_w(R_min: float,
                  R_max: float,
                  lambd: torch.Tensor,
                  eta: torch.Tensor,
                  zeta: torch.Tensor,
                  rs: torch.Tensor,
                  smooth_type: str,
                  Rij: torch.Tensor):

        Rij = Rij[:, :, :, 1:4]
        c = torch.combinations(torch.arange(Rij.size(2)), r=2)
        Rij = Rij[:, :, c]
        R_ij = Rij[:, :, :, 0]
        R_ik = Rij[:, :, :, 1]
        R_jk = R_ij - R_ik
        rij = torch.norm(R_ij, dim=3)
        rik = torch.norm(R_ik, dim=3)
        rjk = torch.norm(R_jk, dim=3)
        f_rij = smooth_fun(smooth_type=smooth_type, rij=rij, r_inner=R_min, r_outer=R_max)
        f_rik = smooth_fun(smooth_type=smooth_type, rij=rik, r_inner=R_min, r_outer=R_max)
        f_rjk = smooth_fun(smooth_type=smooth_type, rij=rik, r_inner=R_min, r_outer=R_max)
        cosine = torch.einsum('ijkl,ijkl->ijk', R_ij, R_ik) / (rij * rik)

        Gi = torch.sum(
            2 ** (1 - zeta.unsqueeze(dim=1)) * (1 + lambd.unsqueeze(dim=1) * cosine.unsqueeze(dim=-1)) ** zeta.unsqueeze(
                dim=1) * torch.exp(-eta.unsqueeze(dim=1) * (rij** 2 + rik ** 2 + rjk ** 2).unsqueeze(dim=-1)) * (
                        f_rij * f_rik * f_rjk).unsqueeze(dim=-1), dim=2)
        return Gi
