import torch
import numpy as np
from ase import Atoms
from ase.build import bulk
from abfml.data import ReadData
from abfml.data.read_data import to_graph


def main(atoms: Atoms):
    # 设置 cutoff 和 max_neighbors（字典形式）
    cutoff = 5.0
    neighbor_dict = {14: 4, 8: 3}  # Si:4, O:3
    # 调用 calculate_neighbor
    element_types, central_atoms, neighbor_indices, neighbor_types, neighbor_vectors = ReadData.calculate_neighbor(
        atoms=atoms,
        cutoff=cutoff,
        neighbor=neighbor_dict
    )

    # 打印结果
    print("Element types:", element_types)
    print("Central atoms:", central_atoms)
    print("Neighbor indices:\n", neighbor_indices)
    print("Neighbor types:\n", neighbor_types)
    print("Neighbor vectors (r, dx, dy, dz):\n", neighbor_vectors)

    # 设置 cutoff 和 max_neighbors（整数形式）
    cutoff = 5.0
    neighbor_dict = 8  # Si:4, O:3
    # 调用 calculate_neighbor
    element_types_1, central_atoms_1, neighbor_indices_1, neighbor_types_1, neighbor_vectors_1 = ReadData.calculate_neighbor(
        atoms=atoms,
        cutoff=cutoff,
        neighbor=neighbor_dict
    )

    # 打印结果
    print("Element types:", element_types_1)
    print("Central atoms:", central_atoms_1)
    print("Neighbor indices:\n", neighbor_indices_1)
    print("Neighbor types:\n", neighbor_types_1)
    print("Neighbor vectors[0] (r, dx, dy, dz):\n", neighbor_vectors_1)


    device = torch.device("cpu")

    element_types = torch.tensor(element_types, dtype=torch.long, device=device)
    central_atoms = torch.tensor(central_atoms, dtype=torch.long, device=device)
    neighbor_indices = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
    neighbor_types = torch.tensor(neighbor_types, dtype=torch.long, device=device)
    neighbor_vectors = torch.tensor(neighbor_vectors, dtype=torch.float, device=device)
    graph = to_graph(central_atoms, neighbor_indices, neighbor_types, neighbor_vectors)
    print("Data.x:", graph[0].x)
    print("Data.edge_index:", graph[0].edge_index)
    print("Data.relative_pos:", graph[0].relative_pos)


if __name__ == "__main__":
    # === SiO2 测试结构 ===
    positions = [
        (0, 0, 0),  # Si
        (1.35, 1.35, 1.35),  # O
        (2.7, 2.7, 2.7),  # O
        (4.05, 4.05, 4.05),  # Si
        (5.4, 5.4, 5.4),  # O
        (6.75, 6.75, 6.75)  # O
    ]
    symbols = ['Si', 'O', 'O', 'Si', 'O', 'O']
    cell = np.eye(3) * 30.0
    atoms_sio2 = Atoms(symbols=symbols, positions=positions, pbc=True, cell=cell)
    atoms_sio2.write('atoms_sio2.xyz')
    main(atoms=atoms_sio2)
    print("SiO2 successful")

    # === FCC Cu 3x3x3 结构 ===
    atoms_cu = bulk('Cu', 'fcc', a=3.615) * (3, 3, 3)
    atoms_cu.write('atoms_cu_3x3x3.xyz')
    main(atoms=atoms_cu)
    print("FCC Cu successful")
