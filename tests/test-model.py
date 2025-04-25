from typing import Dict
import torch
import numpy as np
from ase import Atoms
from abfml import to_graph
from abfml import ReadData
from abfml import FieldModel
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv



class TestModel(FieldModel):
    def __init__(self):
        super().__init__(type_map=[0], cutoff=6.0, neighbor=100)

        # Edge encoder: MLP to process edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        # GINEConv layer with MLP edge encoder
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ), edge_dim=32
        )

    def field(self,
              element_map: torch.Tensor,
              central_atoms: torch.Tensor,
              neighbor_indices: torch.Tensor,
              neighbor_types: torch.Tensor,
              neighbor_vectors: torch.Tensor,
              n_ghost: int) -> Dict[str, torch.Tensor]:
        # Convert atom data to graph representation
        graph = to_graph(central_atoms, neighbor_indices, neighbor_types, neighbor_vectors)
        x = graph[0]['x']
        edge_index = graph[0]['edge_index']
        edge_attr = graph[0]['relative_pos']

        # Process edge attributes
        edge_attr = edge_attr[:, 0:1] + torch.norm(edge_attr[:, 0:], dim=-1, keepdim=True)
        edge_attr = self.edge_encoder(edge_attr)

        # Create Data object for PyTorch Geometric
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Apply GINEConv
        x = self.conv1(x, edge_index, edge_attr)

        # Energy computations
        Ei = x.unsqueeze(0)
        Etot = torch.sum(Ei, dim=1)

        return {'Etot': Etot, 'Ei': Ei}

def main():
    # 设置原子的位置
    positions = [
        (5.0, 5.0, 5.0),  # 第一个原子的位置
        (5.0, 6.0, 6.0)  # 第二个原子的位置
    ]

    # 设置原子类型
    symbols = ['Cu', 'Cu']

    # 设置模拟盒子，较大的盒子
    cell = np.array([[30.0, 0.0, 0.0],
                     [0.0, 30.0, 0.0],
                     [0.0, 0.0, 30.0]])

    # 创建 Atoms 对象
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    # 设置 cutoff 和 max_neighbors
    cutoff = 5.0
    neighbor_dict = 100 # Si:4, O:3
    # 调用 calculate_neighbor
    element_types, central_atoms, neighbor_indices, neighbor_types, neighbor_vectors = ReadData.calculate_neighbor(
        atoms=atoms,
        cutoff=cutoff,
        neighbor=neighbor_dict
    )
    device = torch.device("cpu")
    element_types = torch.tensor(element_types, dtype=torch.long, device=device)
    central_atoms = torch.tensor(central_atoms, dtype=torch.long, device=device)
    neighbor_indices = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
    neighbor_types = torch.tensor(neighbor_types, dtype=torch.long, device=device)
    neighbor_vectors = torch.tensor(neighbor_vectors, dtype=torch.float, device=device)

    model = TestModel()
    output = model(element_types, central_atoms, neighbor_indices, neighbor_types, neighbor_vectors, 0)
    print("Etot:", output[0])
    print("Ei:", output[1])
    print("force:", output[2])
    print("virial:", output[3])
    print("atomic_virial:", output[4])
    try:
        jit_model = torch.jit.script(model)
        jit_model.save('test-jit-script-model.pt')
        print("Model successfully scripted and saved.")
    except Exception as e:
        print(f"JIT model not compiled using script. Error: {e}")

if __name__ == "__main__":
    main()
    print("successful")
