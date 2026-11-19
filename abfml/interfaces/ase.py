"""ASE calculator mapping model inputs to energy, forces, and stress."""

import numpy as np
import torch
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from abfml.data.reader import from_ase
from abfml.data.collate import StructureCollator
from abfml.core.neighbor import resolve_builder
from abfml.serialization import load_model


class ABFMLCalculator(Calculator):
    """Load a standard model or checkpoint and expose its declared properties to ASE."""

    def __init__(self, model, *, device="cpu", dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.model = load_model(model, device=device, dtype=dtype)
        parameter = next(self.model.parameters(), None)
        self.dtype = parameter.dtype if parameter is not None else (dtype or torch.float64)
        self.device = torch.device(device)
        capabilities = self.model.capabilities
        mapping = {"energy": "energy", "atomic_energies": "energies", "forces": "forces", "virial": "stress"}
        self.implemented_properties = [target for source, target in mapping.items() if source in capabilities]
        self.collator = StructureCollator(resolve_builder(self.model))

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        requested = properties or ["energy"]
        unsupported = set(requested) - set(self.implemented_properties)
        if unsupported:
            raise PropertyNotImplementedError(f"Unsupported model properties: {sorted(unsupported)}")
        sample = from_ase(self.atoms, dtype=self.dtype)
        inputs, _ = self.collator([sample])
        inputs = inputs.to(device=self.device, dtype=self.dtype)
        with torch.enable_grad():
            output = self.model(inputs)
        results = {}
        if output.energy is not None:
            results["energy"] = float(output.energy.detach().cpu().reshape(-1)[0])
        if output.atomic_energies is not None:
            results["energies"] = output.atomic_energies.detach().cpu().numpy()
        if output.forces is not None:
            results["forces"] = output.forces.detach().cpu().numpy()
        if output.virial is not None and bool(self.atoms.pbc.any()) and self.atoms.cell.rank == 3:
            stress = -output.virial.detach().cpu().numpy().reshape(3, 3) / self.atoms.get_volume()
            results["stress"] = full_3x3_to_voigt_6_stress(stress)
        for name in requested:
            if name not in results:
                raise PropertyNotImplementedError(f"Model or geometry does not provide {name}")
        if any(not np.isfinite(value).all() for value in results.values()):
            raise ValueError("Model returned non-finite predictions")
        self.results = results
