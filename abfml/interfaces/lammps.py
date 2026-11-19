"""Python runtime bridge for local, external-neighbor ABFML potentials."""
import numpy as np
import torch
from abfml.serialization import load_model
from abfml.core.types import ModelInput, collect_outputs


class Bridge:
    def __init__(self, path, device='cpu', mode='eager', precision='float32'):
        if mode not in {'eager', 'compile'}:
            raise ValueError("mode must be eager or compile")
        if precision not in {'float32', 'float64'}:
            raise ValueError("precision must be float32 or float64")
        torch.set_num_threads(1)
        self.mode = mode
        self.dtype = getattr(torch, precision)
        # Convert model parameters, floating buffers, and inputs explicitly; default to float32.
        self.model = load_model(path, device=device, dtype=self.dtype)
        if self.model.geometry_mode != 'local' or self.model.neighbor_mode != 'external':
            raise ValueError('LAMMPS bridge requires local/external models')
        self.device = device
        self.cutoff = float(self.model.cutoff)
        self.type_map = list(self.model.type_map)
        # Create the compiled wrapper once; the first computation compiles reusable dynamic-shape graphs.
        # Report compilation failures explicitly; users can switch to eager rather than falling back silently.
        if mode == 'compile':
            self.model = torch.compile(self.model, dynamic=True)

    def compute(self, numbers, centers, ptr, neighbors, vectors, atomic_virial=False):
        def index(value):
            return torch.as_tensor(value, dtype=torch.long, device=self.device)
        inputs = ModelInput(atomic_numbers=index(numbers), center_indices=index(centers),
                            edge_ptr=index(ptr), neighbor_indices=index(neighbors),
                            edge_vectors=torch.as_tensor(vectors, dtype=self.dtype, device=self.device),
                            is_full_structure=False, atomic_virial=atomic_virial)
        with torch.enable_grad():
            result = self.model(inputs)
        # Return available properties by name; avoid transferring potentially large descriptors.
        outputs = {}
        supported = {"energy", "atomic_energies", "forces", "virial", "charges", "atomic_virial"}
        for name, value in collect_outputs(result).items():
            if name not in supported:
                continue
            array = value.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite model output: {name}")
            outputs[name] = array
        return outputs
