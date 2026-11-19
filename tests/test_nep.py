"""Independent numerical and export tests for the NEP implementation."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from abfml.core.types import ModelInput
from abfml.builtin.nep import NepDescriptorRange
from abfml.builtin.nep import NepModel, radial_basis, angular_basis


class NepTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.atomic_numbers = torch.tensor([1, 8, 1], dtype=torch.long)
        self.center_indices = torch.tensor([0, 1, 2], dtype=torch.long)
        self.edge_ptr = torch.tensor([0, 2, 3, 6], dtype=torch.long)
        self.neighbor_indices = torch.tensor([1, 2, 0, 0, 1, 0], dtype=torch.long)
        self.edge_vectors = torch.tensor(
            [
                [1.1, 0.2, 0.1],
                [0.3, 1.4, -0.2],
                [-1.1, -0.2, -0.1],
                [-0.3, -1.4, 0.2],
                [0.8, -0.5, 0.4],
                [0.2, 0.7, 1.0],
            ],
            dtype=torch.float64,
        )

    def make_model(self, l_max=None, network_size=None) -> NepModel:
        if l_max is None:
            l_max = [2]
        network_config = {"num_neurons": 5}
        if network_size is not None:
            network_config = {"network_size": network_size}
        model = NepModel(
            type_map=[1, 8],
            cutoff_radial=4.0,
            cutoff_angular=3.0,
            n_max_radial=2,
            n_max_angular=2,
            basis_size_radial=3,
            basis_size_angular=3,
            l_max=l_max,
            output_bias=0.17,
            **network_config,
        ).to(torch.float64).eval()
        with torch.no_grad():
            model.radial_coefficients.normal_(0.0, 0.2)
            model.angular_coefficients.normal_(0.0, 0.2)
            model.scaler.q_scaler.uniform_(0.7, 1.3)
        return model

    def arguments(self, edge_vectors: torch.Tensor | None = None) -> ModelInput:
        return ModelInput(
            atomic_numbers=self.atomic_numbers,
            center_indices=self.center_indices,
            edge_ptr=self.edge_ptr,
            neighbor_indices=self.neighbor_indices,
            edge_vectors=self.edge_vectors if edge_vectors is None else edge_vectors,
        )

    def test_chebyshev_basis_matches_recurrence(self) -> None:
        distances = torch.tensor([0.4, 1.0, 2.2], dtype=torch.float64)
        actual = radial_basis(distances, 3.0, 4)
        ratio = distances / 3.0
        cutoff = 0.5 * (torch.cos(math.pi * ratio) + 1.0)
        x = 2.0 * (ratio - 1.0).square() - 1.0
        polynomials = [torch.ones_like(x), x]
        for _ in range(2, 5):
            polynomials.append(2.0 * x * polynomials[-1] - polynomials[-2])
        expected = torch.stack(polynomials, dim=-1)
        expected = 0.5 * (expected + 1.0) * cutoff.unsqueeze(-1)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (3, 5))

    def test_angular_basis_low_orders(self) -> None:
        vectors = torch.tensor(
            [[0.3, -0.4, 0.5], [-0.2, 0.7, 0.1]],
            dtype=torch.float64,
        )
        actual = angular_basis(vectors, 2)
        x, y, z = vectors.unbind(dim=-1)
        expected = torch.stack(
            [z, x, y, 3.0 * z.square() - 1.0, x * z, y * z, x.square() - y.square(), 2.0 * x * y],
            dim=-1,
        )
        torch.testing.assert_close(actual, expected)

    def test_radial_and_three_body_match_explicit_pair_sum(self) -> None:
        model = self.make_model([2])
        atom_types = model.type_lut[self.atomic_numbers]
        actual = model.descriptor(
            atom_types,
            self.center_indices,
            self.edge_ptr,
            self.neighbor_indices,
            self.edge_vectors,
        )
        descriptor = model
        center_slots = torch.repeat_interleave(
            torch.arange(3),
            self.edge_ptr[1:] - self.edge_ptr[:-1],
        )
        expected = torch.zeros(3, descriptor.descriptor_dim, dtype=torch.float64)
        for center in range(3):
            center_type = int(atom_types[center])
            radial_neighbors = []
            angular_neighbors = []
            for edge in range(self.neighbor_indices.numel()):
                if int(center_slots[edge]) != center:
                    continue
                vector = self.edge_vectors[edge]
                distance = vector.square().sum().sqrt()
                neighbor = int(self.neighbor_indices[edge])
                neighbor_type = int(atom_types[neighbor])
                if float(distance) < descriptor.cutoff_radial:
                    basis = radial_basis(
                        distance.reshape(1),
                        descriptor.cutoff_radial,
                        descriptor.basis_size_radial,
                    )[0]
                    values = (
                        descriptor.radial_coefficients[center_type, neighbor_type]
                        @ basis
                    )
                    expected[center, :descriptor.radial_dim] += values
                    radial_neighbors.append((vector, values))
                if float(distance) < descriptor.cutoff_angular:
                    basis = radial_basis(
                        distance.reshape(1),
                        descriptor.cutoff_angular,
                        descriptor.basis_size_angular,
                    )[0]
                    values = (
                        descriptor.angular_coefficients[center_type, neighbor_type]
                        @ basis
                    )
                    angular_neighbors.append((vector, values))

            for order in (1, 2):
                for channel in range(descriptor.angular_channels):
                    value = torch.tensor(0.0, dtype=torch.float64)
                    for vector_j, values_j in angular_neighbors:
                        unit_j = vector_j / vector_j.square().sum().sqrt()
                        for vector_k, values_k in angular_neighbors:
                            unit_k = vector_k / vector_k.square().sum().sqrt()
                            cosine = (unit_j * unit_k).sum().clamp(-1.0, 1.0)
                            if order == 1:
                                legendre = cosine
                            else:
                                legendre = 0.5 * (3.0 * cosine.square() - 1.0)
                            value = value + (
                                (2.0 * order + 1.0)
                                / (4.0 * math.pi)
                                * values_j[channel]
                                * values_k[channel]
                                * legendre
                            )
                    column = descriptor.radial_dim + (order - 1) * descriptor.angular_channels + channel
                    expected[center, column] = value
        torch.testing.assert_close(actual, expected, rtol=1.0e-10, atol=1.0e-11)

    def test_standard_higher_body_descriptors_are_rotation_invariant(self) -> None:
        model = self.make_model([4, 1, 1, 1])
        atom_types = model.type_lut[self.atomic_numbers]
        original = model.descriptor(
            atom_types,
            self.center_indices,
            self.edge_ptr,
            self.neighbor_indices,
            self.edge_vectors,
        )
        angle = 0.731
        rotation = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        rotated = model.descriptor(
            atom_types,
            self.center_indices,
            self.edge_ptr,
            self.neighbor_indices,
            self.edge_vectors @ rotation.transpose(0, 1),
        )
        self.assertEqual(original.shape, (3, 24))
        torch.testing.assert_close(original, rotated, rtol=1.0e-9, atol=1.0e-10)

    def test_q_scaler_statistics_are_incremental_and_multiplicative(self) -> None:
        statistics = NepDescriptorRange(3)
        first = torch.tensor([[1.0, 4.0, -2.0], [3.0, 2.0, 0.0]])
        second = torch.tensor([[2.0, 8.0, 1.0]])
        statistics.update(first)
        statistics.update(second)
        scaler = statistics.finalize()
        torch.testing.assert_close(scaler, torch.tensor([0.5, 1.0 / 6.0, 1.0 / 3.0]))
        model = self.make_model()
        model.scaler.q_scaler.copy_(scaler.repeat(3))
        sample = torch.tensor([[2.0, 3.0, 0.5]], dtype=torch.float64).repeat(1, 3)
        torch.testing.assert_close(model.scaler(sample), sample * scaler.repeat(3))

    def test_single_hidden_layer_matches_gpumd_formula(self) -> None:
        model = self.make_model(network_size=[5])
        atom_types = model.type_lut[self.atomic_numbers]
        center_types = atom_types[self.center_indices]
        descriptors = model.scaler(model.descriptor(
            atom_types,
            self.center_indices,
            self.edge_ptr,
            self.neighbor_indices,
            self.edge_vectors,
        )).detach().requires_grad_(True)
        reference_descriptors = descriptors.detach().clone().requires_grad_(True)

        input_layer = model.fitting_network.layers[0]
        output_layer = model.fitting_network.output_layer
        hidden = torch.tanh(
            torch.bmm(
                reference_descriptors.unsqueeze(1),
                input_layer.weight[center_types].transpose(1, 2),
            ).squeeze(1) - input_layer.bias[center_types]
        )
        expected = torch.bmm(
            hidden.unsqueeze(1),
            output_layer.weight[center_types].transpose(1, 2),
        ).reshape(-1)
        actual = model.fitting_network(descriptors, center_types)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(descriptors.grad, reference_descriptors.grad)

    def test_multilayer_network_gradient(self) -> None:
        model = self.make_model(network_size=[5, 4])
        self.assertEqual(len(model.fitting_network.layers), 2)
        atom_types = model.type_lut[self.atomic_numbers]
        center_types = atom_types[self.center_indices]
        descriptors = model.scaler(model.descriptor(
            atom_types,
            self.center_indices,
            self.edge_ptr,
            self.neighbor_indices,
            self.edge_vectors,
        )).detach().requires_grad_(True)
        reference_descriptors = descriptors.detach().clone().requires_grad_(True)
        expected = reference_descriptors
        for layer in model.fitting_network.layers:
            expected = torch.tanh(layer(expected, center_types))
        expected = model.fitting_network.output_layer(
            expected, center_types
        ).reshape(-1)
        actual = model.fitting_network(descriptors, center_types)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(descriptors.grad, reference_descriptors.grad)

    def test_coordinate_forces(self) -> None:
        model = self.make_model([4, 1, 1, 1])
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [0.3, 1.4, -0.2]],
            dtype=torch.float64,
            requires_grad=True,
        )
        center_per_edge = self.center_indices.repeat_interleave(
            self.edge_ptr[1:] - self.edge_ptr[:-1]
        )
        edge_vectors = positions[self.neighbor_indices] - positions[center_per_edge]
        result = model(self.arguments(edge_vectors))

        reference_positions = positions.detach().clone().requires_grad_(True)
        reference_vectors = (
            reference_positions[self.neighbor_indices]
            - reference_positions[center_per_edge]
        )
        reference_energy = model.compute(self.arguments(reference_vectors)).energy
        expected_forces = -torch.autograd.grad(reference_energy, reference_positions)[0]
        torch.testing.assert_close(result.forces, expected_forces, rtol=1.0e-9, atol=1.0e-9)
        self.assertTrue(torch.isfinite(result.virial).all())

    def test_empty_neighbors_have_zero_mechanical_derivatives(self) -> None:
        model = self.make_model([4, 1, 1, 1])
        result = model(ModelInput(
            atomic_numbers=torch.tensor([1]),
            center_indices=torch.tensor([0]),
            edge_ptr=torch.tensor([0, 0]),
            neighbor_indices=torch.empty(0, dtype=torch.long),
            edge_vectors=torch.empty(0, 3, dtype=torch.float64),
        ))
        torch.testing.assert_close(result.energy, torch.tensor(-0.17, dtype=torch.float64))
        torch.testing.assert_close(result.forces, torch.zeros(1, 3, dtype=torch.float64))
        torch.testing.assert_close(result.virial, torch.zeros(3, 3, dtype=torch.float64))

    def test_zero_length_edge_is_ignored_without_nan(self) -> None:
        model = self.make_model([2])
        result = model(ModelInput(
            atomic_numbers=torch.tensor([1]),
            center_indices=torch.tensor([0]),
            edge_ptr=torch.tensor([0, 1]),
            neighbor_indices=torch.tensor([0]),
            edge_vectors=torch.zeros(1, 3, dtype=torch.float64),
        ))
        torch.testing.assert_close(result.energy, torch.tensor(-0.17, dtype=torch.float64))
        self.assertTrue(torch.isfinite(result.forces).all())
        torch.testing.assert_close(result.forces, torch.zeros(1, 3, dtype=torch.float64))

    def test_radial_only_and_unsupported_channels(self) -> None:
        model = self.make_model([0])
        atom_types = model.type_lut[self.atomic_numbers]
        descriptors = model.descriptor(
            atom_types,
            self.center_indices,
            self.edge_ptr,
            self.neighbor_indices,
            self.edge_vectors,
        )
        self.assertEqual(descriptors.shape, (3, 3))
        with self.assertRaisesRegex(ValueError, "not implemented"):
            self.make_model([4, 1, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "require l_max_3b"):
            self.make_model([1, 1])


if __name__ == "__main__":
    unittest.main()
