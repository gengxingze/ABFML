"""Shared network layers and fitting networks for models."""

from typing import List
import torch
from torch import nn


# Maintain names and implementations here; parsing and construction share this mapping.
ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "celu": nn.CELU,
    "glu": nn.GLU,
    "logsigmoid": nn.LogSigmoid,
    "rrelu": nn.RReLU,
    "hardshrink": nn.Hardshrink,
    "hardtanh": nn.Hardtanh,
    "softshrink": nn.Softshrink,
    "tanhshrink": nn.Tanhshrink,
}


class ActivationModule(nn.Module):
    """Construct one named activation and fail early on invalid names."""

    def __init__(self, activation_name: str):
        super().__init__()
        if activation_name not in ACTIVATIONS:
            raise ValueError("Unsupported activation: " + activation_name)
        self.activation = ACTIVATIONS[activation_name]()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs)


class ResidualMLP(nn.Module):
    """MLP supporting equal-width and doubled-width residual connections.
    
    Learn residual scaling only when resnet_dt is enabled; select scaling parameters by layer
    index.
    """

    def __init__(
        self,
        network_size: List[int],
        activate: str,
        bias: bool,
        resnet_dt: bool,
        activate_output: bool,
        energy_shift: float = 0.0,
        shifted_output: bool = False,
    ):
        super().__init__()
        if len(network_size) < 2 or any(width < 1 for width in network_size):
            raise ValueError("network_size must contain at least two positive widths")
        self.network_size = network_size
        self.bias = bool(bias)
        self.resnet_dt = bool(resnet_dt)
        self.activate_output = bool(activate_output)
        self.shifted_output = bool(shifted_output)
        self.activate = ActivationModule(activate)
        self.linear = nn.ModuleList()
        num_layers = len(network_size) - 1
        if self.resnet_dt:
            self.resnet = nn.Parameter(torch.zeros(num_layers, max(network_size)))
        else:
            self.register_parameter("resnet", None)

        final_index = len(network_size) - 2
        for index in range(num_layers):
            in_width = network_size[index]
            out_width = network_size[index + 1]
            is_shifted_output = shifted_output and index == final_index
            layer_bias = True if is_shifted_output else bias
            linear = nn.Linear(in_width, out_width, bias=layer_bias)
            nn.init.normal_(linear.weight, std=1.0 / (in_width + out_width) ** 0.5)
            if linear.bias is not None:
                mean = energy_shift if is_shifted_output else 0.0
                nn.init.normal_(linear.bias, mean=mean, std=1.0)
            self.linear.append(linear)

            residual_shape = in_width == out_width or in_width * 2 == out_width
            if self.resnet is not None and residual_shape and not is_shifted_output:
                nn.init.normal_(
                    self.resnet.data[index, :out_width],
                    mean=0.1,
                    std=0.001,
                )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        residual_scales = self.resnet
        final_index = len(self.linear) - 1
        for index, linear in enumerate(self.linear):
            hidden = linear(outputs)
            if not self.activate_output and index == final_index:
                outputs = hidden
            else:
                hidden = self.activate(hidden)
                in_width = self.network_size[index]
                out_width = self.network_size[index + 1]
                if residual_scales is not None and (
                    in_width == out_width or in_width * 2 == out_width
                ):
                    hidden = hidden * residual_scales[index, :out_width]
                if in_width == out_width:
                    outputs = hidden + outputs
                elif in_width * 2 == out_width:
                    outputs = hidden + torch.cat((outputs, outputs), dim=-1)
                else:
                    outputs = hidden
        return outputs


class SpeciesLinear(nn.Module):
    """Apply a different packed linear transform to each element species."""

    def __init__(
        self,
        num_types: int,
        in_features: int,
        out_features: int,
        bias: bool,
        bias_mean: float = 0.0,
        bias_sign: float = 1.0,
    ):
        super().__init__()
        self.bias_sign = float(bias_sign)
        self.weight = nn.Parameter(torch.empty(num_types, out_features, in_features))
        nn.init.normal_(self.weight, std=1.0 / (in_features + out_features) ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.empty(num_types, out_features))
            nn.init.normal_(self.bias, mean=bias_mean, std=1.0)
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        selected_weights = self.weight[species]
        outputs = torch.bmm(
            inputs.unsqueeze(1),
            selected_weights.transpose(1, 2),
        ).squeeze(1)
        bias = self.bias
        if bias is not None:
            outputs = outputs + self.bias_sign * bias[species]
        return outputs


class SpeciesFittingNetwork(nn.Module):
    """Evaluate element-specific fitting MLPs with batched matrix products.

    Parameters are packed by species and layer. All centers share each network
    launch, while gathered weights preserve independent element parameters.
    The return shape is ``[N, 1]``, matching a scalar-output :class:`ResidualMLP`.
    """

    def __init__(
        self,
        num_types: int,
        network_size: List[int],
        activate: str,
        bias: bool,
        resnet_dt: bool,
        energy_shift: List[float],
    ):
        super().__init__()
        if num_types < 1:
            raise ValueError("num_types must be positive")
        if len(energy_shift) != num_types:
            raise ValueError("energy_shift must contain one value per species")
        sizes = network_size + [1]
        if len(sizes) < 2 or any(width < 1 for width in sizes):
            raise ValueError("network_size must contain positive input and hidden widths")
        self.num_types = int(num_types)
        self.network_size = sizes
        self.resnet_dt = bool(resnet_dt)
        self.activate = ActivationModule(activate)
        self.layers = nn.ModuleList()
        num_layers = len(sizes) - 1
        for index in range(num_layers):
            is_output = index == num_layers - 1
            self.layers.append(SpeciesLinear(
                num_types,
                sizes[index],
                sizes[index + 1],
                bias=True if is_output else bias,
                bias_mean=0.0,
            ))
            if is_output:
                for species, shift in enumerate(energy_shift):
                    nn.init.normal_(
                        self.layers[index].bias.data[species],
                        mean=shift,
                        std=1.0,
                    )
        if self.resnet_dt:
            self.resnet = nn.Parameter(torch.zeros(
                num_types,
                num_layers,
                max(sizes),
            ))
            for index in range(num_layers - 1):
                in_width = sizes[index]
                out_width = sizes[index + 1]
                if in_width == out_width or in_width * 2 == out_width:
                    nn.init.normal_(
                        self.resnet.data[:, index, :out_width],
                        mean=0.1,
                        std=0.001,
                    )
        else:
            self.register_parameter("resnet", None)

    def forward(self, inputs: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.network_size[0]:
            raise ValueError("inputs must have shape [N, input_width]")
        if species.shape != (inputs.shape[0],):
            raise ValueError("species must have shape [N]")
        if bool(((species < 0) | (species >= self.num_types)).any()):
            raise ValueError("species contains an invalid element index")

        outputs = inputs
        residual_scales = self.resnet
        final_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            hidden = layer(outputs, species)
            if index == final_index:
                outputs = hidden
            else:
                hidden = self.activate(hidden)
                in_width = self.network_size[index]
                out_width = self.network_size[index + 1]
                if residual_scales is not None and (
                    in_width == out_width or in_width * 2 == out_width
                ):
                    hidden = hidden * residual_scales[species, index, :out_width]
                if in_width == out_width:
                    outputs = hidden + outputs
                elif in_width * 2 == out_width:
                    outputs = hidden + torch.cat((outputs, outputs), dim=-1)
                else:
                    outputs = hidden
        return outputs
