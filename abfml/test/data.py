import torch
import numpy as np
from ase import Atoms
from abfml.data import ReadData
from abfml.data.read_data import to_graph

# 构建一个 SiO2 样式的双元体系（人为设置）
positions = [
    (0, 0, 0),        # Si
    (1.35, 1.35, 1.35),  # O
    (2.7, 2.7, 2.7),     # O
    (4.05, 4.05, 4.05),  # Si
    (5.4, 5.4, 5.4),     # O
    (6.75, 6.75, 6.75)  # O
]
symbols = ['Si', 'O', 'O', 'Si', 'O', 'O']
cell = np.eye(3) * 30.0 # 设置一个较大的盒子避免周期混淆
atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

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

# 设置 cutoff 和 max_neighbors（字典形式）
cutoff = 5.0
neighbor_dict = 8 # Si:4, O:3
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
print("Neighbor vectors[0] (r, dx, dy, dz):\n", neighbor_vectors)
device = torch.device("cpu")

element_types = torch.tensor(element_types, dtype=torch.long, device=device)
central_atoms = torch.tensor(central_atoms, dtype=torch.long, device=device)
neighbor_indices = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
neighbor_types = torch.tensor(neighbor_types, dtype=torch.long, device=device)
neighbor_vectors = torch.tensor(neighbor_vectors, dtype=torch.float, device=device)
graph = to_graph(central_atoms, neighbor_indices, neighbor_types, neighbor_vectors)

print("successful")
