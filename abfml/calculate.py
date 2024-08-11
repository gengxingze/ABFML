import ase
import torch
import numpy as np
from typing import List
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from abfml.data.read_data import ReadData


class ABFML(Calculator):
    implemented_properties = ['energy', 'energies', 'forces', 'stress', 'stresses']

    def __init__(self, model: str, **kwargs) -> None:
        Calculator.__init__(self, **kwargs)
        self.model = torch.jit.load(model)
        self.model.eval()

    def calculate(self,
                  atoms: ase.Atoms = None,
                  properties: List[str] = None,
                  system_changes: List[str] = all_changes) -> None:

        type_map = self.model.type_map
        cutoff = self.model.cutoff
        neighbor = self.model.neighbor
        information_tulp = ReadData.calculate_neighbor(atom=atoms, cutoff=cutoff, neighbor=neighbor, type_map=type_map)
        element_type, image_type, neighbor_list, neighbor_type, image_dR = information_tulp
        predict_tulp = self.model(torch.tensor(element_type),
                                  torch.tensor(image_type),
                                  torch.tensor(neighbor_list),
                                  torch.tensor(neighbor_type),
                                  torch.tensor(image_dR), 0)
        self.results["energy"] = predict_tulp[0].detach().numpy().item()
        self.results["energies"] = predict_tulp[1].detach().numpy().reshape(-1)
        self.results["forces"] = predict_tulp[2].detach().numpy().reshape(-1, 3)
        if "stress" in properties:
            stress = predict_tulp[3].detach().numpy().reshape(3, 3)
            if any(atoms.get_pbc()):
                stress = -0.5 * (stress * stress.T) / atoms.get_volume()
                self.results["stress"] = stress

        if "stresses" in properties:
            stresses = predict_tulp[4].detach().numpy().reshape(-1, 3, 3)
            if any(atoms.get_pbc()):
                self.results["stresses"] = -1.0 * stresses / atoms.get_volume()
            else:
                raise PropertyNotImplementedError

