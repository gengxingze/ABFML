import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from ase import Atoms, io
from ase.neighborlist import NeighborList
from typing import List, Optional, Union
from dataclasses import dataclass
from abfml.data.dataset import ABFMLDataset


@dataclass
class DataClass:
    def __init__(self, filename: str, file_format: Optional[str]):
        self.filename: str = filename
        if file_format == "pwmat-config":
            image_list = []
        elif file_format == "pwmat-movement":
            image_list = []
        else:
            image_list = io.read(filename, format=file_format, index=':')
        self.image_list: List[Atoms] = image_list
        self.include_element: set = set()
        include_atoms_number = []
        for atom_information in image_list:
            self.include_element = self.include_element.union(set(atom_information.get_chemical_symbols()))
            include_atoms_number.append(len(atom_information))
        self.include_atoms_number: set = set(include_atoms_number)
        self.n_frames: int = len(self.image_list)


class ReadData:
    def __init__(self, filename: Union[str, List[str]],  cutoff: float, neighbor: List[int],
                 type_map: Optional[List[int]], file_format: Optional[str]):
        self.cutoff = cutoff
        self.neighbor = neighbor
        self.filename = filename
        self.type_map = type_map
        self.image_list: list[Atoms] = []
        if isinstance(filename, str):
            filename = [filename]
        self.data_information: List[dict] = []
        for file in filename:
            data_information = DataClass(file, file_format)
            self.data_information.append({'file_name': data_information.filename,
                                          'n_frames': data_information.n_frames,
                                          'include_element': data_information.include_element,
                                          'include_atoms_number': data_information.include_atoms_number})
            self.image_list = self.image_list + data_information.image_list
        self.n_frames: int = len(self.image_list)
        self.unique_image = {}
        for i, image in enumerate(self.image_list):
            image_tuple = tuple(image.get_atomic_numbers())
            if image_tuple in self.unique_image:
                self.unique_image[image_tuple].append(i)
            else:
                self.unique_image[image_tuple] = [i]

    def get_mlffdata(self):
        dataset = []
        for same_image_index in self.unique_image.values():
            n_frames: int = len(same_image_index)
            n_atoms: int = len(self.image_list[same_image_index[0]])
            n_elements: int = len(set(self.image_list[same_image_index[0]].get_atomic_numbers()))
            if isinstance(self.neighbor, int):
                max_neighbor: int = self.neighbor
            elif isinstance(self.neighbor, list):
                if len(self.neighbor) == len(self.type_map):
                    max_neighbor = sum(self.neighbor[self.type_map.index(atomic_numbers)]
                                       for atomic_numbers in set(self.image_list[same_image_index[0]].get_atomic_numbers()))

                else:
                    raise Exception('neighbor and type_map must have the same length')
            else:
                raise ValueError(f'Expected a member of List[int] and int but instead found type {type(self.neighbor)}')

            Zi = torch.empty(n_frames, n_atoms, dtype=torch.int)
            Nij = torch.empty(n_frames, n_atoms, max_neighbor, dtype=torch.int)
            Zij = torch.empty(n_frames, n_atoms, max_neighbor, dtype=torch.int)
            Rij = torch.zeros(n_frames, n_atoms, max_neighbor, 4, dtype=torch.float64)
            element_type = torch.zeros(n_frames, n_elements, dtype=torch.int)
            energy = torch.zeros(n_frames, 1, dtype=torch.float64)
            atomic_energy = torch.zeros(n_frames, n_atoms, 1, dtype=torch.float64)
            force = torch.zeros(n_frames, n_atoms, 3, dtype=torch.float64)
            virial = torch.zeros(n_frames, 9, dtype=torch.float64)
            for i, image_index in enumerate(same_image_index):
                atom_ase = self.image_list[image_index]
                neighbor_information = ReadData.calculate_neighbor(atom=atom_ase,
                                                                   cutoff=self.cutoff,
                                                                   neighbor=self.neighbor,
                                                                   type_map=self.type_map)
                element_type = torch.tensor(neighbor_information[0])
                Zi[i] = torch.tensor(neighbor_information[1])
                Nij[i] = torch.tensor(neighbor_information[2])
                Zij[i] = torch.tensor(neighbor_information[3])
                Rij[i] = torch.tensor(neighbor_information[4])
                try:
                    energy[i] = torch.tensor(atom_ase.get_potential_energy())
                except (RuntimeError, AttributeError):
                    energy = None
                try:
                    atomic_energy[i] = torch.tensor(atom_ase.atomic_energy)
                except (RuntimeError, AttributeError):
                    atomic_energy = None
                try:
                    force[i] = torch.tensor(atom_ase.get_forces(apply_constraint=False))
                except (RuntimeError, AttributeError):
                    force = None
                try:
                    # ASE calculated in units of eV/A^3, virial: eV
                    virial[i] = torch.tensor(-1.0 * atom_ase.get_stress(voigt=False).reshape(9) * atom_ase.get_volume())
                except (RuntimeError, AttributeError):
                    virial = None
            dataset.append(ABFMLDataset(n_frames=n_frames,
                                        n_atoms=n_atoms,
                                        element_type=element_type,
                                        Zi=Zi,
                                        Nij=Nij,
                                        Zij=Zij,
                                        Rij=Rij,
                                        energy=energy,
                                        atomic_energy=atomic_energy,
                                        force=force,
                                        virial=virial))
        return dataset

    @staticmethod
    def calculate_neighbor(atom: Atoms, cutoff: float, neighbor: Union[List[int], int], type_map: Optional[List[int]]):
        nl = NeighborList([cutoff / 2] * len(atom), skin=0, self_interaction=False, bothways=True, sorted=False)
        nl.update(atom)
        atoms_num = len(atom)
        Zi = atom.numbers
        element_type = np.sort(np.unique(Zi))
        if isinstance(neighbor, int) or type_map is None:
            max_neighbor: int = neighbor
            Rij = np.zeros(shape=(1, atoms_num, max_neighbor, 4), dtype=np.float64)
            Nij = np.full(shape=(1, atoms_num, max_neighbor), fill_value=-1, dtype=np.int32)
            Zij = np.full(shape=(1, atoms_num, max_neighbor), fill_value=-1, dtype=np.int32)
            for i in range(atoms_num):
                indices_neighbor, offsets = nl.get_neighbors(i)
                j_xyz = atom.positions[indices_neighbor] + np.dot(offsets, atom.get_cell())
                ij_xyz = j_xyz - atom.positions[i]
                rij = np.sqrt(np.sum(ij_xyz ** 2, axis=1))
                if indices_neighbor.shape[0] <= max_neighbor:
                    end_index = indices_neighbor.shape[0]
                else:
                    indices_rij = np.argsort(rij, axis=-1)
                    for j in range(3):
                        ij_xyz[..., j] = np.take_along_axis(ij_xyz[..., j], indices=indices_rij, axis=-1)
                    rij = np.take_along_axis(rij, indices=indices_rij, axis=-1)
                    indices_neighbor = np.take_along_axis(indices_neighbor, indices=indices_rij, axis=-1)
                    end_index = max_neighbor
                Rij[:, i, :end_index, 1:] = ij_xyz[:end_index, :]
                Rij[:, i, :end_index, 0] = rij[:end_index]
                Nij[:, i, :end_index] = indices_neighbor[:end_index]
            mask = (Nij != -1)
            Zij[mask] = Zi[Nij[mask]]

        elif len(type_map) == len(neighbor):
            atom_map = type_map
            neighbor_map = neighbor
            Zi = atom.numbers
            max_neighbor = sum(neighbor_map[atom_map.index(atomic_numbers)] for atomic_numbers in element_type)
            width_map = [0]
            Zi_unique = list(element_type)
            for ii in Zi_unique:
                width_map.append(neighbor_map[atom_map.index(ii)]+width_map[-1])
            Rij = np.zeros(shape=(1, atoms_num, max_neighbor, 4), dtype=np.float64)
            Nij = np.full(shape=(1, atoms_num, max_neighbor), fill_value=-1, dtype=np.int32)
            Zij = np.full(shape=(1, atoms_num, max_neighbor), fill_value=-1, dtype=np.int32)
            for i in range(atoms_num):
                indices_neighbor, offsets = nl.get_neighbors(i)
                j_xyz = atom.positions[indices_neighbor] + np.dot(offsets, atom.get_cell())
                ij_xyz = j_xyz - atom.positions[i]
                rij = np.sqrt(np.sum(ij_xyz ** 2, axis=1))
                Zij_i = Zi[indices_neighbor]

                for atomic_numbers in Zi_unique:
                    mask = (Zij_i == atomic_numbers)
                    width = Zij_i[mask].shape[0]
                    start_index = width_map[Zi_unique.index(atomic_numbers)]
                    width_i = neighbor_map[Zi_unique.index(atomic_numbers)]
                    if width <= width_i:
                        end_index = start_index + width
                    else:
                        indices_rij = np.argsort(rij[mask], axis=-1)
                        for j in range(3):
                            ij_xyz[..., j][mask] = np.take_along_axis(ij_xyz[..., j][mask],
                                                                      indices=indices_rij,
                                                                      axis=-1)
                        rij[mask] = np.take_along_axis(rij[mask], indices=indices_rij, axis=-1)
                        indices_neighbor[mask] = np.take_along_axis(indices_neighbor[mask],
                                                                    indices=indices_rij,
                                                                    axis=-1)
                        end_index = start_index + width_i
                    Zij[:, i, start_index:end_index] = Zij_i[mask][:width_i]
                    Nij[:, i, start_index:end_index] = indices_neighbor[mask][:width_i]
                    Rij[:, i, start_index:end_index, 0] = rij[mask][:width_i]
                    Rij[:, i, start_index:end_index, 1:] = ij_xyz[mask][:width_i, :]
        else:
            raise Exception('neighbor[0] and neighbor[1] should have the same length '
                            'or neighbor[1] have only one element. Maybe you should read the manual!')
        Zi = Zi[np.newaxis,...]
        return element_type, Zi, Nij, Zij, Rij

