"""DeepPot-SE center/batch consistency, derivative, and padding tests."""

import unittest

import torch

from abfml.core.types import ModelInput
from abfml.builtin.deepse import DeepPotSE2A, DeepPotSE2R, DeepPotSE3


class DeepPotSETest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.type_map = [1, 8]
        self.atomic_numbers = torch.tensor([1, 8, 1], dtype=torch.long)
        self.center_indices = torch.tensor([0, 1], dtype=torch.long)
        self.center_per_edge = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        self.edge_ptr = torch.tensor([0, 4, 8], dtype=torch.long)
        self.neighbor_indices = torch.tensor([2, 2, 1, 1, 0, 2, 1, 1], dtype=torch.long)
        self.edge_vectors = torch.tensor(
            [
                [0.8, 0.2, 0.1],
                [1.1, -0.3, 0.2],
                [0.4, 1.0, 0.3],
                [-0.5, 0.9, 0.2],
                [-0.8, -0.2, -0.1],
                [0.3, 0.5, 0.9],
                [0.2, 1.2, -0.4],
                [0.7, -0.6, 0.5],
            ],
            dtype=torch.float64,
        )

    def configurations(self, variant: str):
        coordinate_size = 1 if variant == "se_e2_r" else 4
        std = torch.linspace(0.8, 1.2, self.type_map.__len__() * coordinate_size).reshape(
            len(self.type_map), coordinate_size
        )
        avg = torch.linspace(-0.1, 0.1, self.type_map.__len__() * coordinate_size).reshape(
            len(self.type_map), coordinate_size
        )
        fitting = {
            "network_size": [5],
            "activate_function": "tanh",
            "bias": True,
            "resnet_dt": False,
        }
        embedding = {
            "M2": 1 if variant == "se_e2_r" else 2,
            "R_min": 0.0,
            "R_max": 4.0,
            "smooth_fun": "cos",
            "network_size": [4, 3],
            "activate_function": "tanh",
            "bias": True,
            "resnet_dt": False,
        }
        return fitting, embedding, [std, avg]

    def model_input(
        self,
        edge_vectors: torch.Tensor | None = None,
        center_indices: torch.Tensor | None = None,
        edge_ptr: torch.Tensor | None = None,
        neighbor_indices: torch.Tensor | None = None,
    ) -> ModelInput:
        return ModelInput(
            atomic_numbers=self.atomic_numbers,
            center_indices=self.center_indices if center_indices is None else center_indices,
            edge_ptr=self.edge_ptr if edge_ptr is None else edge_ptr,
            neighbor_indices=self.neighbor_indices if neighbor_indices is None else neighbor_indices,
            edge_vectors=self.edge_vectors if edge_vectors is None else edge_vectors,
        )

    def check_variant(self, variant: str, type_slots=None) -> None:
        fitting, embedding, std_mean = self.configurations(variant)
        if variant == "se_e2_a":
            model_class = DeepPotSE2A
        elif variant == "se_e2_r":
            model_class = DeepPotSE2R
        else:
            model_class = DeepPotSE3
        model = model_class(
            self.type_map,
            4.0,
            2,  # Default uniform slot count; type_slots may override it explicitly.
            fitting,
            embedding,
            [0.1, -0.2],
            std_mean,
            type_slots=type_slots,
        ).to(torch.float64).eval()
        outputs = model.compute(self.model_input())
        descriptors = outputs.properties["descriptors"]
        self.assertEqual(descriptors.shape, (2, model.descriptor_dim))
        # Centers have different species; each exported row must match its fitting network's actual input.
        reconstructed = torch.stack([
            model.fitting_net[index](descriptors[index:index + 1]).reshape(())
            for index in range(2)
        ])
        torch.testing.assert_close(reconstructed, outputs.atomic_energies)
        differentiable = model.compute(self.model_input(
            self.edge_vectors.clone().requires_grad_(True)))
        self.assertTrue(differentiable.properties["descriptors"].requires_grad)
        expected = []
        for center in range(2):
            expected.append(model.compute(self.model_input(
                edge_vectors=self.edge_vectors[center*4:(center+1)*4],
                center_indices=self.center_indices[center:center+1],
                edge_ptr=torch.tensor([0,4]),
                neighbor_indices=self.neighbor_indices[center*4:(center+1)*4],
            )).energy)
        torch.testing.assert_close(outputs.atomic_energies, torch.stack(expected))

        reference_vectors = self.edge_vectors.detach().clone().requires_grad_(True)
        reference_energy = model.compute(self.model_input(reference_vectors)).energy
        gradient = torch.autograd.grad(reference_energy, reference_vectors)[0]
        expected_forces = torch.zeros(3, 3, dtype=torch.float64)
        expected_forces = expected_forces.index_add(0, self.center_per_edge, gradient)
        expected_forces = expected_forces.index_add(0, self.neighbor_indices, -gradient)
        result = model(self.model_input())
        torch.testing.assert_close(result.properties["descriptors"], descriptors)
        torch.testing.assert_close(result.forces, expected_forces, rtol=1.0e-8, atol=1.0e-9)
        self.assertTrue(torch.isfinite(result.virial).all())


        padded_result = model(self.model_input(
            edge_vectors=self.edge_vectors[:1],
            center_indices=torch.tensor([0], dtype=torch.long),
            edge_ptr=torch.tensor([0, 1], dtype=torch.long),
            neighbor_indices=self.neighbor_indices[:1],
        ))
        self.assertTrue(torch.isfinite(padded_result.forces).all())
        self.assertTrue(torch.isfinite(padded_result.virial).all())

    def test_se_e2_a(self) -> None:
        self.check_variant("se_e2_a", type_slots=[2, 2])

    def test_se_e2_r(self) -> None:
        self.check_variant("se_e2_r")

    def test_se_e3(self) -> None:
        self.check_variant("se_e3")


if __name__ == "__main__":
    unittest.main()
