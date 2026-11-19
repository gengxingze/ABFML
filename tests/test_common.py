"""Numerical parity and derivative tests for shared model components."""
import unittest
import torch
from abfml.builtin.networks import SpeciesFittingNetwork, ResidualMLP
from abfml.builtin.functions import cutoff_function

class CutoffFunctionTest(unittest.TestCase):

    def test_cutoff_values_and_finite_difference(self):
        for name in ['cos', 'cos_r', 'tanh_u', 'exp', 'poly1', 'poly2', 'poly3', 'poly4', 'poly1_r', 'poly2_r', 'poly3_r']:
            distances = torch.tensor([1.2, 2.8, 4.1], dtype=torch.float64, requires_grad=True)
            values = cutoff_function(name, distances, 0.8, 4.0)
            self.assertEqual(float(values[-1]), 0.0)
            self.assertTrue(torch.isfinite(values).all())
            torch.autograd.gradcheck(lambda x: cutoff_function(name, x, 0.8, 4.0), (distances,))
        radii = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(cutoff_function('cos', radii, 0.0, 4.0), 0.5 * (1 + torch.cos(torch.pi * radii / 4.0)))

    def test_unknown_cutoff_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cutoff_function('unknown', torch.ones(2), 0.0, 4.0)

class ResidualNetworkTest(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(41)

    def test_network_first_and_second_derivatives(self):
        for network in (ResidualMLP(network_size=[1] + [4, 4, 8], activate='tanh', bias=True, resnet_dt=True, activate_output=True), ResidualMLP(network_size=[3, 4, 4] + [1], activate='tanh', bias=True, resnet_dt=True, energy_shift=-0.3, activate_output=False, shifted_output=True)):
            network.double()
            inputs = torch.randn(2, network.network_size[0], dtype=torch.float64, requires_grad=True)
            torch.autograd.gradcheck(network, (inputs,))
            torch.autograd.gradgradcheck(network, (inputs,))

    def check_species_fitting(self, resnet_dt: bool, bias: bool) -> None:
        energy_shift = [0.1, -0.2, 0.3]
        batched = SpeciesFittingNetwork(num_types=3, network_size=[3, 4, 4], activate='tanh', bias=bias, resnet_dt=resnet_dt, energy_shift=energy_shift).to(torch.float64).eval()
        references = [ResidualMLP(network_size=[3, 4, 4] + [1], activate='tanh', bias=bias, resnet_dt=resnet_dt, energy_shift=energy_shift[type_id], activate_output=False, shifted_output=True).to(torch.float64).eval() for type_id in range(3)]
        with torch.no_grad():
            for type_id, reference in enumerate(references):
                for packed_layer, single_layer in zip(batched.layers, reference.linear):
                    single_layer.weight.copy_(packed_layer.weight[type_id])
                    if single_layer.bias is not None:
                        single_layer.bias.copy_(packed_layer.bias[type_id])
                if resnet_dt:
                    reference.resnet[1, :4].copy_(batched.resnet[type_id, 1, :4])
        species = torch.tensor([0, 1, 2, 1, 0, 2, 2, 0, 1], dtype=torch.long)
        inputs = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        expected = torch.zeros(9, 1, dtype=torch.float64)
        for type_id, reference in enumerate(references):
            mask = species == type_id
            expected[mask] = reference(reference_inputs[mask])
        output = batched(inputs, species)
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)
        for type_id, reference in enumerate(references):
            for packed_layer, single_layer in zip(batched.layers, reference.linear):
                torch.testing.assert_close(packed_layer.weight.grad[type_id], single_layer.weight.grad)
                if single_layer.bias is not None:
                    torch.testing.assert_close(packed_layer.bias.grad[type_id], single_layer.bias.grad)
        empty = batched(inputs[:0].detach(), species[:0])
        self.assertEqual(empty.shape, (0, 1))

    def test_species_fitting_matches_independent_networks(self) -> None:
        self.check_species_fitting(resnet_dt=False, bias=False)
        self.check_species_fitting(resnet_dt=True, bias=True)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ResidualMLP(network_size=[1] + [], activate='tanh', bias=True, resnet_dt=False, activate_output=True)
        with self.assertRaises(ValueError):
            ResidualMLP(network_size=[] + [1], activate='tanh', bias=True, resnet_dt=False, energy_shift=0.0, activate_output=False, shifted_output=True)
        with self.assertRaises(ValueError):
            ResidualMLP(network_size=[3, 4] + [1], activate='not_an_activation', bias=True, resnet_dt=False, energy_shift=0.0, activate_output=False, shifted_output=True)
if __name__ == '__main__':
    unittest.main()
