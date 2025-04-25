import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from ase import Atoms, io
from ase.neighborlist import NeighborList
from typing import List, Optional, Union, Dict
from torch_geometric.data import Data
from dataclasses import dataclass
from abfml.data.dataset import ABFMLDataset


@dataclass
class DataClass:
    filename: str
    atoms_list: List[Atoms]
    include_element: set = None
    include_atoms_numbers: set = None
    num_frames: int = 0
    def __init__(self, filename: str, file_format: Optional[str]):
        self.filename: str = filename
        if file_format == "pwmat-config":
            atoms_list = []
        elif file_format == "pwmat-movement":
            atoms_list = []
        else:
            atoms_list = io.read(filename, format=file_format, index=':')
        self.atoms_list: List[Atoms] = atoms_list
        self.include_element: set = set()
        include_atoms_number = []
        for atom_information in atoms_list:
            self.include_element = self.include_element.union(set(atom_information.get_chemical_symbols()))
            include_atoms_number.append(len(atom_information))
        self.include_atoms_number: set = set(include_atoms_number)
        self.n_frames: int = len(self.atoms_list)


class ReadData:
    def __init__(
            self,
            filename: Union[str, List[str]],
            cutoff: float,
            neighbor: Union[Dict[int, int], int],
            file_format: Optional[str]):
        self.cutoff = cutoff
        self.neighbor = neighbor
        self.filename = filename
        self.atoms_list: list[Atoms] = []
        if isinstance(filename, str):
            filename = [filename]
        self.data_information: List[dict] = []
        for file in filename:
            data_information = DataClass(file, file_format)
            self.data_information.append({'file_name': data_information.filename,
                                          'n_frames': data_information.n_frames,
                                          'include_element': data_information.include_element,
                                          'include_atoms_number': data_information.include_atoms_number})
            self.atoms_list = self.atoms_list + data_information.atoms_list
        self.n_frames: int = len(self.atoms_list)
        self.unique_image = {}
        for i, image in enumerate(self.atoms_list):
            image_tuple = tuple(image.get_atomic_numbers())
            if image_tuple in self.unique_image:
                self.unique_image[image_tuple].append(i)
            else:
                self.unique_image[image_tuple] = [i]

    def create_dataset(self):
        dataset = []
        for same_image_index in self.unique_image.values():
            n_frames: int = len(same_image_index)
            n_atoms: int = len(self.atoms_list[same_image_index[0]])
            n_elements: int = len(set(self.atoms_list[same_image_index[0]].get_atomic_numbers()))
            if isinstance(self.neighbor, int):
                max_neighbor: int = self.neighbor
            elif isinstance(self.neighbor, dict):
                max_neighbor = sum(self.neighbor[atomic_numbers]
                                       for atomic_numbers in set(self.atoms_list[same_image_index[0]].get_atomic_numbers()))
            else:
                raise ValueError(f'Expected a member of Dict[int, int] and int but instead found type {type(self.neighbor)}')

            central_atoms = torch.empty(n_frames, n_atoms, dtype=torch.int)
            neighbor_indices = torch.empty(n_frames, n_atoms, max_neighbor, dtype=torch.int)
            neighbor_types = torch.empty(n_frames, n_atoms, max_neighbor, dtype=torch.int)
            neighbor_vectors = torch.zeros(n_frames, n_atoms, max_neighbor, 4, dtype=torch.float64)
            element_types = torch.zeros(n_frames, n_elements, dtype=torch.int)
            energy = torch.zeros(n_frames, 1, dtype=torch.float64)
            atomic_energies = torch.zeros(n_frames, n_atoms, 1, dtype=torch.float64)
            forces = torch.zeros(n_frames, n_atoms, 3, dtype=torch.float64)
            virials = torch.zeros(n_frames, 9, dtype=torch.float64)
            for i, image_index in enumerate(same_image_index):
                atoms_ase = self.atoms_list[image_index]
                neighbor_information = ReadData.calculate_neighbor(atoms=atoms_ase,
                                                                   cutoff=self.cutoff,
                                                                   neighbor=self.neighbor)
                element_types = torch.tensor(neighbor_information[0])
                central_atoms[i] = torch.tensor(neighbor_information[1])
                neighbor_indices[i] = torch.tensor(neighbor_information[2])
                neighbor_types[i] = torch.tensor(neighbor_information[3])
                neighbor_vectors[i] = torch.tensor(neighbor_information[4])
                try:
                    energy[i] = torch.tensor(atoms_ase.get_potential_energy())
                except (RuntimeError, AttributeError):
                    energy = None
                try:
                    atomic_energies[i] = torch.tensor(atoms_ase.atomic_energies)
                except (RuntimeError, AttributeError):
                    atomic_energies = None
                try:
                    forces[i] = torch.tensor(atoms_ase.get_forces(apply_constraint=False))
                except (RuntimeError, AttributeError):
                    forces = None
                try:
                    # ASE calculated in units of eV/A^3, virial: eV
                    virials[i] = torch.tensor(-1.0 * atoms_ase.get_stress(voigt=False).reshape(9) * atoms_ase.get_volume())
                except (RuntimeError, AttributeError):
                    virials = None
            dataset.append(ABFMLDataset(num_frames=n_frames,
                                        num_atoms=n_atoms,
                                        element_types=element_types,
                                        central_atoms=central_atoms,
                                        neighbor_indices=neighbor_indices,
                                        neighbor_types=neighbor_types,
                                        neighbor_vectors=neighbor_vectors,
                                        energy=energy,
                                        atomic_energies=atomic_energies,
                                        forces=forces,
                                        virials=virials))
        return dataset

    @staticmethod
    def calculate_neighbor(atoms: Atoms,
                           cutoff: float,
                           neighbor: Union[Dict[int, int], int]):
        nl = NeighborList([cutoff / 2] * len(atoms), skin=0, self_interaction=False, bothways=True, sorted=False)
        nl.update(atoms)
        atoms_num = len(atoms)
        central_atoms = atoms.numbers
        element_types = np.unique(central_atoms)
        if isinstance(neighbor, int):
            max_neighbor = neighbor
            neighbor_vectors = np.zeros(shape=(1, atoms_num, max_neighbor, 4), dtype=np.float64)
            neighbor_indices = np.full(shape=(1, atoms_num, max_neighbor), fill_value=-1, dtype=np.int32)
            neighbor_types = np.full(shape=(1, atoms_num, max_neighbor), fill_value=-1, dtype=np.int32)
            for i in range(atoms_num):
                indices_neighbor, offsets = nl.get_neighbors(i)
                neighbor_positions = atoms.positions[indices_neighbor] + np.dot(offsets, atoms.get_cell())
                delta = neighbor_positions - atoms.positions[i]
                rij = np.linalg.norm(delta, axis=1)
                indices = np.arange(indices_neighbor.shape[0])
                if indices_neighbor.shape[0] <= max_neighbor:
                    sorted_indices = indices
                else:
                    order = np.argsort(rij, axis=-1)
                    sorted_indices = indices[order[:max_neighbor]]
                valid_count = len(sorted_indices)

                neighbor_vectors[:, i, :valid_count, 1:] = delta[sorted_indices, :]
                neighbor_vectors[:, i, :valid_count, 0] = rij[sorted_indices]
                neighbor_indices[:, i, :valid_count] = indices_neighbor[sorted_indices]
            mask = (neighbor_indices != -1)
            neighbor_types[mask] = central_atoms[neighbor_indices[mask]]

        elif isinstance(neighbor, Dict):
            max_neighbor = sum(neighbor[element] for element in element_types)
            slot_map = {}
            current_slot = 0
            for atomic_number in neighbor.keys():
                slot_size = neighbor[atomic_number]
                slot_map[atomic_number] = (current_slot, current_slot + slot_size)
                current_slot += slot_size
            central_atoms = atoms.numbers
            neighbor_vectors = np.zeros(shape=(1, atoms_num, max_neighbor, 4), dtype=np.float64)
            neighbor_indices = np.full(shape=(1, atoms_num, max_neighbor), fill_value=-1, dtype=np.int32)
            neighbor_types = np.full(shape=(1, atoms_num, max_neighbor), fill_value=-1, dtype=np.int32)
            for i in range(atoms_num):
                indices_neighbor, offsets = nl.get_neighbors(i)
                neighbor_positions = atoms.positions[indices_neighbor] + np.dot(offsets, atoms.get_cell())
                delta = neighbor_positions - atoms.positions[i]
                rij = np.linalg.norm(delta, axis=1)

                for atomic_numbers in np.unique(element_types):
                    mask = (central_atoms[indices_neighbor])==atomic_numbers
                    if not np.any(mask):
                        continue
                    start_slot, end_slot = slot_map[atomic_numbers]
                    max_slots = end_slot - start_slot
                    valid_indices = np.where(mask)[0]
                    valid_rij = rij[valid_indices]
                    if valid_indices.shape[0] <= max_slots:
                        sorted_indices = valid_indices
                    else:
                        order = np.argsort(valid_rij, axis=-1)
                        sorted_indices = valid_indices[order[:max_slots]]
                    valid_count = len(sorted_indices)
                    slot_slice = slice(start_slot, start_slot + valid_count)
                    neighbor_vectors[:, i, slot_slice, 1:] = delta[sorted_indices, :]
                    neighbor_vectors[:, i, slot_slice, 0] = rij[sorted_indices]
                    neighbor_indices[:, i, slot_slice] = indices_neighbor[sorted_indices]
            mask = (neighbor_indices != -1)
            neighbor_types[mask] = central_atoms[neighbor_indices[mask]]
        else:
            raise Exception('neighbor[0] and neighbor[1] should have the same length '
                            'or neighbor[1] have only one element. Maybe you should read the manual!')
        central_atoms = central_atoms[np.newaxis, ...]
        return element_types, central_atoms, neighbor_indices, neighbor_types, neighbor_vectors


def to_graph(central_atoms: torch.Tensor,
             neighbor_indices: torch.Tensor,
             neighbor_types: torch.Tensor,
             neighbor_vectors: torch.Tensor) -> List[Data]:
    """
    Convert atomic neighbor information into a list of PyG graph objects (one per batch entry).

    Args:
        central_atoms (Tensor): Shape (B, N), atomic types of central atoms.
        neighbor_indices (Tensor): Shape (B, N, M), indices of neighbors (-1 for invalid).
        neighbor_types (Tensor): Shape (B, N, M), types of neighboring atoms.
        neighbor_vectors (Tensor): Shape (B, N, M, 4), relative vectors to neighbors (r, dx, dy, dz).

    Notes: For the safety of subsequent operations, this out returns a list instead of a Batch of torch_geometric.data.
    Returns:
        List[Data]: A list of torch_geometric.data.Data graphs, one for each batch item.
        Each Data object contains:
            - x (Tensor): Node features, shape [num_nodes, num_node_features]. Here, it's the atomic type.
            - edge_index (LongTensor): Graph connectivity in COO format, shape [2, num_edges].
            - relative_pos (Tensor): Edge features (e.g., distances and relative positions), shape [num_edges, 4].
    """
    batch, n_atoms, max_neighbor = neighbor_types.shape
    device, dtype = neighbor_vectors.device, neighbor_vectors.dtype
    graphs = []

    for b in range(batch):
        # Create node features from central atom types (shape: [N, 1])
        x = central_atoms[b].unsqueeze(-1).to(torch.long)

        # Identify valid neighbor connections (exclude -1 entries)
        valid_mask = neighbor_indices[b] != -1

        # Get valid (source_atom_index, neighbor_index_in_M) pairs
        src_idx, nbr_idx = valid_mask.nonzero(as_tuple=True)

        # Source atoms are the indices from 0 to N-1 (repeated)
        edge_src = src_idx

        # Destination atoms are the valid neighbor indices
        edge_dst = neighbor_indices[b][src_idx, nbr_idx]

        # Stack to form edge_index of shape [2, num_edges]
        edge_index = torch.stack([edge_src, edge_dst], dim=0).to(device)

        # Extract relative vectors (r, dx, dy, dz)
        relative_pos = neighbor_vectors[b][src_idx, nbr_idx].to(device=device, dtype=dtype)
        # Construct a PyG Data object for this graph
        graph = Data(x=x, edge_index=edge_index, relative_pos=relative_pos)
        graphs.append(graph)
    return graphs


