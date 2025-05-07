import ase
import torch
from typing import List
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from abfml.data.read_data import ReadData


class ABFML(Calculator):
    implemented_properties = ['energy', 'energies', 'forces', 'stress', 'stresses']

    def __init__(self, model: str = None, dtype: str = 'float64', model_class=None, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32 if dtype == "float32" else torch.float64

        self.model = self._load_model(model, model_class)
        self.model.to(self.dtype).to(self.device).eval()

    def _load_model(self, model, model_class):
        if model:
            try:
                return torch.jit.load(model, map_location=self.device)
            except Exception:
                try:
                    return torch.load(model, map_location=self.device)
                except Exception as e:
                    raise RuntimeError(f"Failed to load model '{model}' with jit and torch.load: {e}")
        elif model_class:
            return model_class
        else:
            raise RuntimeError("No model or model_class provided.")

    def calculate(self, atoms: ase.Atoms = None, properties: List[str] = None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        props = properties or self.implemented_properties
        model = self.model
        type_map = model.type_map
        cutoff = model.cutoff
        neighbor = model.neighbor

        # Get neighbor info
        element_type, central_atoms, neighbor_indices, neighbor_types, neighbor_vectors = \
            ReadData.calculate_neighbor(atoms=atoms, cutoff=cutoff, neighbor=neighbor)

        # Prepare inputs
        input_tensors = [
            torch.tensor(element_type),
            torch.tensor(central_atoms, device=self.device),
            torch.tensor(neighbor_indices, device=self.device),
            torch.tensor(neighbor_types, device=self.device),
            torch.tensor(neighbor_vectors, dtype=self.dtype, device=self.device),
            0  # In ASE, it is zero. Because neighbor of structure has not ghost atoms.
        ]

        # Model inference
        energy, energies, forces, stress, stresses = model(*input_tensors)
        self.results["energy"] = energy.cpu().detach().numpy().item()
        self.results["energies"] = energies.cpu().detach().numpy().reshape(-1)
        self.results["forces"] = forces.cpu().detach().numpy().reshape(-1, 3)
        if "stress" in properties:
            stress = stress.cpu().detach().numpy().reshape(3, 3)
            if any(atoms.get_pbc()):
                stress = -0.5 * (stress * stress.T) / atoms.get_volume()
                self.results["stress"] = stress

        if "stresses" in properties:
            stresses = stresses.cpu().detach().numpy().reshape(-1, 3, 3)
            if any(atoms.get_pbc()):
                self.results["stresses"] = -1.0 * stresses / atoms.get_volume()
            else:
                raise PropertyNotImplementedError

