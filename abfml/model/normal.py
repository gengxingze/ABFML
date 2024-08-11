import torch
import random
from typing import List, Tuple, Callable, Dict, Union
from abfml.param.param import Param
from abfml.model.method import NormalModel


class DPNormal(NormalModel):
    def __init__(self,
                 normal_data,
                 param_class: Param,
                 normal_rate: Union[float, str],
                 coordinate_matrix: str):
        super().__init__(normal_data=normal_data,
                         param_class=param_class,
                         normal_rate=normal_rate,
                         is_get_energy_shift=True)
        self.coordinate_matrix = coordinate_matrix
        self.R_min = None
        self.R_max = None
        self.smooth_type = None

        self.reset(param_class=param_class)
        self.initialize()

    def reset(self, param_class: Param):
        self.R_min = param_class.DeepSe.embedding_config['R_min']
        self.R_max = param_class.DeepSe.embedding_config['R_max']
        self.smooth_type = param_class.DeepSe.embedding_config['smooth_fun']

    def normal(self, normal_loader, param_class: Param):
        type_map = param_class.GlobalSet.type_map
        ntype = len(type_map)
        type_num = torch.zeros(ntype)
        std_mean: List[torch.Tensor] = [torch.zeros(ntype, 4, requires_grad=False),
                                        torch.zeros(ntype, 4, requires_grad=False)]
        if self.coordinate_matrix == 'a':
            from abfml.model.field_model_dp import DpSe2a
            calculate_coordinate_matrix = DpSe2a.calculate_coordinate_matrix
        elif self.coordinate_matrix == 'r':
            from abfml.model.field_model_dp import DpSe2r
            calculate_coordinate_matrix = DpSe2r.calculate_coordinate_matrix
        else:
            raise KeyError("Key word of coordinate_matrix have 'a' or 'r'")

        for i, image_batch in enumerate(normal_loader):
            Rij = image_batch["Rij"]
            Zi = image_batch["Zi"]
            element_type = image_batch["element_type"][0].to(torch.int64).tolist()

            Ri = calculate_coordinate_matrix(R_min=self.R_min,
                                             R_max=self.R_max,
                                             smooth_type=self.smooth_type,
                                             Rij=Rij)
            # I think the sample standard deviation should be used instead of the standard deviation
            for i_type, element in enumerate(element_type):
                mask = (Zi == element)
                s_rij = Ri[:, :, :, 0][mask]
                s_rij_std, s_rij_mean = torch.std_mean(s_rij)

                if Ri.shape[-1] == 4:
                    xyz = Ri[:, :, :, 1:][mask]
                    xyz_std, xyz_mean = torch.std_mean(xyz)
                else:
                    xyz_std, xyz_mean = 0.0, 0.0

                indices = type_map.index(element)
                type_num[indices] += 1

                std_mean[0][indices][0] = std_mean[0][indices][0] + s_rij_std
                std_mean[0][indices][1] = std_mean[0][indices][1] + xyz_std
                std_mean[0][indices][2] = std_mean[0][indices][2] + xyz_std
                std_mean[0][indices][3] = std_mean[0][indices][3] + xyz_std
                std_mean[1][indices][0] = std_mean[1][indices][0] + s_rij_mean

        type_num[type_num == 0] = 1
        for i in range(ntype):
            std_mean[0][i] = std_mean[0][i] / type_num[i]
            std_mean[1][i] = std_mean[1][i] / type_num[i]

        return std_mean


class FeatureNormal(NormalModel):
    def __init__(self,
                 normal_data,
                 param_class: Param,
                 normal_rate: Union[float, str],
                 feature_name: str):
        super().__init__(normal_data=normal_data,
                         param_class=param_class,
                         normal_rate=normal_rate,
                         is_get_energy_shift=True)
        self.feature_name = feature_name
        self.initialize()

    def normal(self, normal_loader, param_class: Param):
        type_map = param_class.GlobalSet.type_map
        ntype = len(type_map)
        type_num = torch.zeros(ntype)
        std_mean: List[torch.Tensor] = [torch.zeros(ntype,  requires_grad=False),
                                        torch.zeros(ntype, requires_grad=False)]
        from abfml.model.field_model_bp import BPMlp
        for i, image_batch in enumerate(normal_loader):
            neighbor_list = image_batch['neighbor_list']
            Zij = image_batch['Zij']
            Rij = image_batch["Rij"]
            Zi = image_batch["Zi"]

            element_type = image_batch["element_type"][0].to(torch.int64)
            Gi = BPMlp.calculate_bp_feature(type_map=type_map,
                                            bp_features_information=param_class.BPDescriptor.bp_features_information,
                                            bp_features_param=param_class.BPDescriptor.bp_features_param,
                                            element_map=element_type,
                                            neighbor_list=neighbor_list,
                                            Zij=Zij,
                                            Rij=Rij)

            # I think the sample standard deviation should be used instead of the standard deviation
            Gi = Gi.clone().detach()
            for i_type, element in enumerate(element_type):
                mask = (Zi == element)
                feature = Gi[mask]
                feature_std, feature_mean = torch.std_mean(feature)

                indices = type_map.index(element)
                type_num[indices] += 1

                std_mean[0][indices] = std_mean[0][indices] + feature_std
                std_mean[1][indices] = std_mean[1][indices] + feature_mean

        type_num[type_num == 0] = 1
        for i in range(ntype):
            std_mean[0][i] = std_mean[0][i] / type_num[i]
            std_mean[1][i] = std_mean[1][i] / type_num[i]

        return std_mean

