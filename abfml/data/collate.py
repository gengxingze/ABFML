"""Sample collation and optional neighbor construction, usable without a DataLoader."""

from collections.abc import Sequence
import torch

from abfml.core.types import ModelInput
from abfml.core.structure import AtomicSample, _STANDARD_LABELS
from abfml.core.neighbor import CSRNeighbors, NeighborBuilder
from abfml.core.schema import LABELS

class StructureCollator:
    """Single batch assembly entry point returning (ModelInput, label dictionary).
    
    External neighbors may come from a provider or supplied CSR graphs. No batch state is
    retained; each call produces independent tensors.
    """

    def __init__(self, provider: NeighborBuilder | None = None, *, label_scopes=None):
        self.provider = provider
        self.label_scopes = dict(label_scopes or {})

    def __call__(self, structures: Sequence[AtomicSample],
                 neighbors: Sequence[CSRNeighbors] | None = None
                 ) -> tuple[ModelInput, dict[str, torch.Tensor]]:
        label_scopes = self.label_scopes
        if neighbors is None and self.provider is not None:
            neighbors = [self.provider.build(structure) for structure in structures]
        if not structures:
            raise ValueError("structures must not be empty")
        if neighbors is not None and len(structures) != len(neighbors):
            raise ValueError("Each structure must have exactly one CSRNeighbors")

        first = structures[0]
        for structure in structures[1:]:
            if structure.device != first.device or structure.dtype != first.dtype:
                raise ValueError("All structures must share device and floating-point dtype")

        atom_counts = [structure.num_atoms for structure in structures]
        atom_ptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=first.device),
                torch.as_tensor(atom_counts, dtype=torch.long, device=first.device).cumsum(0),
            )
        )
        atom_batch = torch.repeat_interleave(
            torch.arange(len(structures), device=first.device),
            atom_ptr[1:] - atom_ptr[:-1],
        )

        graph: CSRNeighbors | None = None
        if neighbors is None:
            # Without center slots, center_batch=arange(N) would be invalid.
            center_batch = atom_batch
        else:
            center_parts: list[torch.Tensor] = []
            neighbor_parts: list[torch.Tensor] = []
            vector_parts: list[torch.Tensor] = []
            shift_parts: list[torch.Tensor] = []
            ptr_parts = [torch.zeros(1, dtype=torch.long, device=first.device)]
            edge_offset = 0
            has_shifts = neighbors[0].edge_shifts is not None

            for structure_index, (structure, neighbor) in enumerate(
                zip(structures, neighbors)
            ):
                if neighbor.center_indices.device != structure.device:
                    raise ValueError("Structure and neighbor data must be on the same device")
                if (
                    neighbor.num_centers
                    and int(neighbor.center_indices.max()) >= structure.num_atoms
                ):
                    raise ValueError("Neighbor center index exceeds its structure")
                if (
                    neighbor.num_edges
                    and int(neighbor.neighbor_indices.max()) >= structure.num_atoms
                ):
                    raise ValueError("Neighbor index exceeds its structure")
                if (neighbor.edge_shifts is not None) != has_shifts:
                    raise ValueError("edge_shifts must be present for all structures or none")

                offset = int(atom_ptr[structure_index])
                center_parts.append(neighbor.center_indices + offset)
                neighbor_parts.append(neighbor.neighbor_indices + offset)
                vector_parts.append(neighbor.edge_vectors)
                if neighbor.edge_shifts is not None:
                    shift_parts.append(neighbor.edge_shifts)
                ptr_parts.append(neighbor.edge_ptr[1:] + edge_offset)
                edge_offset += neighbor.num_edges

            center_parts_cat = torch.cat(center_parts)
            graph = CSRNeighbors(
                center_indices=center_parts_cat,
                edge_ptr=torch.cat(ptr_parts),
                neighbor_indices=torch.cat(neighbor_parts),
                edge_vectors=torch.cat(vector_parts),
                edge_shifts=torch.cat(shift_parts) if has_shifts else None,
            )
            center_batch = atom_batch[center_parts_cat]

        standard: dict[str, torch.Tensor | None] = {}
        for name in _STANDARD_LABELS:
            values = [getattr(structure, name) for structure in structures]
            present = [value is not None for value in values]
            if not any(present):
                standard[name] = None
                continue
            if not all(present):
                raise ValueError(f"Label {name} must be present for every structure or none")
            tensors = [value for value in values if value is not None]
            standard[name] = (
                torch.cat(tensors, dim=0) if LABELS[name].scope in {"atom", "center"} else torch.stack(tensors)
            )

        scopes = {name: spec.scope for name, spec in LABELS.items()
                  if not spec.sample_field and spec.source is None and spec.scope in {"atom", "structure"}}
        if label_scopes is not None:
            for name, scope in label_scopes.items():
                if name in _STANDARD_LABELS:
                    raise ValueError(f"{name} is a standard label with a fixed scope")
                if scope not in {"atom", "structure"}:
                    raise ValueError("Label scope must be 'atom' or 'structure'")
                spec = LABELS.get(name)
                if spec is not None and scope != spec.scope and not (spec.scope == "center" and scope == "atom"):
                    raise ValueError(f"{name} has a fixed scope: {spec.scope}")
                scopes[name] = scope

        extra_names: list[str] = []
        for structure in structures:
            for name in structure.extra_labels:
                if name not in extra_names:
                    extra_names.append(name)

        extra_labels: dict[str, torch.Tensor] = {}
        for name in extra_names:
            if name not in scopes:
                raise ValueError(f"Extended label {name} requires an explicit label scope")
            scope = scopes[name]
            values = [structure.extra_labels.get(name) for structure in structures]
            if any(value is None for value in values):
                raise ValueError(f"Label {name} must be present for every structure or none")
            tensors = [value for value in values if value is not None]
            if scope == "atom" and any(
                value.ndim == 0 or value.shape[0] != structure.num_atoms
                for structure, value in zip(structures, tensors)
            ):
                raise ValueError(f"Atom label {name} must have one entry per atom")
            extra_labels[name] = (
                torch.cat(tensors, dim=0) if scope == "atom" else torch.stack(tensors)
            )

        inputs = ModelInput(
            atomic_numbers=torch.cat([s.atomic_numbers for s in structures]),
            positions=torch.cat([s.positions for s in structures]),
            cell=torch.stack([s.cell for s in structures]),
            pbc=torch.stack([s.pbc for s in structures]),
            atom_ptr=atom_ptr,
            center_batch=center_batch,
            num_structures=len(structures),
            is_full_structure=True,
            center_indices=None if graph is None else graph.center_indices,
            edge_ptr=None if graph is None else graph.edge_ptr,
            neighbor_indices=None if graph is None else graph.neighbor_indices,
            edge_vectors=None if graph is None else graph.edge_vectors,
            edge_shifts=None if graph is None else graph.edge_shifts,
        )
        labels = {name: value for name, value in standard.items() if value is not None}
        labels.update(extra_labels)
        return inputs, labels
