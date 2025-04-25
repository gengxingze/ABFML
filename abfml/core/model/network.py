import torch
import torch.nn as nn
from typing import List, Tuple
from abfml.core.model.math_fun import ActivationModule


class EmbeddingNet(nn.Module):
    def __init__(self,
                 network_size: List[int],
                 activate: str = 'tanh',
                 bias: bool = True,
                 resnet_dt: bool = False):
        super(EmbeddingNet, self).__init__()
        self.network_size = [1] + network_size  # [1, 25, 50, 100]
        self.bias = bias
        self.resnet_dt = resnet_dt
        self.activate = ActivationModule(activation_name=activate)
        self.linear = nn.ModuleList()
        self.resnet = nn.ParameterList()
        for i in range(len(self.network_size) - 1):
            self.linear.append(nn.Linear(in_features=self.network_size[i],
                                         out_features=self.network_size[i + 1], bias=self.bias))
            if self.bias:
                nn.init.normal_(self.linear[i].bias, mean=0.0, std=1.0)
            if self.network_size[i] == self.network_size[i+1] or self.network_size[i] * 2 == self.network_size[i+1]:
                resnet_tensor = torch.Tensor(1, self.network_size[i + 1])
                nn.init.normal_(resnet_tensor, mean=0.1, std=0.001)
                self.resnet.append(nn.Parameter(resnet_tensor, requires_grad=True))
            nn.init.normal_(self.linear[i].weight, mean=0.0,
                            std=(1.0 / (self.network_size[i] + self.network_size[i + 1]) ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = 0
        for i, linear in enumerate(self.linear):
            hidden = linear(x)
            hidden = self.activate(hidden)
            if self.network_size[i] == self.network_size[i+1] and self.resnet_dt:
                for ii, resnet in enumerate(self.resnet):
                    if ii == m:
                        x = hidden * resnet + x
                m = m + 1
            elif self.network_size[i] == self.network_size[i+1] and (not self.resnet_dt):
                x = hidden + x
            elif self.network_size[i] * 2 == self.network_size[i+1] and self.resnet_dt:
                for ii, resnet in enumerate(self.resnet):
                    if ii == m:
                        x = hidden * resnet + torch.cat((x, x), dim=-1)
                m = m + 1
            elif self.network_size[i] * 2 == self.network_size[i+1] and (not self.resnet_dt):
                x = hidden + torch.cat((x, x), dim=-1)
            else:
                x = hidden
        return x


class FittingNet(nn.Module):
    def __init__(self,
                 network_size: List[int],
                 activate: str,
                 bias: bool,
                 resnet_dt: bool,
                 energy_shift: float):
        super(FittingNet, self).__init__()
        self.network_size = network_size + [1]  # [input, 25, 50, 100, 1]
        self.bias = bias
        self.resnet_dt = resnet_dt
        self.activate = ActivationModule(activation_name=activate)
        self.linear = nn.ModuleList()
        self.resnet = nn.ParameterList()
        for i in range(len(self.network_size) - 1):
            if i == (len(self.network_size) - 2):
                self.linear.append(nn.Linear(in_features=self.network_size[i],
                                             out_features=self.network_size[i + 1], bias=True))
                nn.init.normal_(self.linear[i].bias, mean=energy_shift, std=1.0)
            else:
                self.linear.append(nn.Linear(in_features=self.network_size[i],
                                             out_features=self.network_size[i + 1], bias=self.bias))
                if self.bias:
                    nn.init.normal_(self.linear[i].bias, mean=0.0, std=1.0)
            if self.network_size[i] == self.network_size[i+1] and self.resnet_dt:
                resnet_tensor = torch.Tensor(1, self.network_size[i + 1])
                nn.init.normal_(resnet_tensor, mean=0.1, std=0.001)
                self.resnet.append(nn.Parameter(resnet_tensor, requires_grad=True))
            nn.init.normal_(self.linear[i].weight, mean=0.0,
                            std=(1.0 / (self.network_size[i] + self.network_size[i + 1]) ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = 0
        for i, linear in enumerate(self.linear):
            if i == (len(self.network_size) - 2):
                hidden = linear(x)
                x = hidden
            else:
                hidden = linear(x)
                hidden = self.activate(hidden)
                if self.network_size[i] == self.network_size[i+1] and self.resnet_dt:
                    for ii, resnet in enumerate(self.resnet):
                        if ii == m:
                            x = hidden * resnet + x
                    m = m + 1
                elif self.network_size[i] == self.network_size[i+1] and (not self.resnet_dt):
                    x = hidden + x
                elif self.network_size[i] * 2 == self.network_size[i+1] and self.resnet_dt:
                    for ii, resnet in enumerate(self.resnet):
                        if ii == m:
                            x = hidden * resnet + torch.cat((x, x), dim=-1)
                    m = m + 1
                elif self.network_size[i] * 2 == self.network_size[i+1] and (not self.resnet_dt):
                    x = hidden + torch.cat((x, x), dim=-1)
                else:
                    x = hidden
        return x


class Dense(nn.Module):
    def __init__(self, num_channels: int, in_features: int, out_features: int, bias: bool = True, activate: str = 'tanh', residual: bool = False) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_channels, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_channels, out_features))
        else:
            self.register_parameter('bias', None)
        self.activation = ActivationModule(activation_name=activate)
        self.residual = residual
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w in self.weight:
            nn.init.kaiming_uniform_(w, a=5**0.5)
        if self.bias is not None:
            for b, w in zip(self.bias, self.weight):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / (fan_in ** 0.5)
                nn.init.uniform_(b, -bound, bound)

    def forward(self, xx: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, channels = xx
        weight: torch.Tensor = self.weight[channels]
        output: torch.Tensor = torch.bmm(x.transpose(0, 1), weight.transpose(1, 2)).transpose(0, 1)

        if self.bias is not None:
            bias = self.bias[channels]
            output = output + bias

        output = self.activation(output)

        if self.residual:
            if output.shape[2] == x.shape[2]:
                output = output + x
            elif output.shape[2] == x.shape[2] * 2:
                output = output + torch.cat([x, x], dim=2)
            else:
                raise NotImplementedError("Not implemented")

        return output, channels


class Sequential(nn.Sequential):
    def forward(self, xx: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        for module in self:
            xx = module(xx)
        return xx


class AtomFitNet(nn.Module):
    def __init__(self,
                 num_channels: int,
                 network_size: List[int],
                 activate: str,
                 bias: bool,
                 resnet_dt: bool):
        super(AtomFitNet, self).__init__()

        self.bias = bias
        self.resnet_dt = resnet_dt

        layers = []
        for i in range(len(network_size) - 1):
            layers.append(Dense(num_channels=num_channels, in_features=network_size[i], out_features=network_size[i+1],
                                bias=bias, activate=activate, residual=resnet_dt)) # iterating through the neurons

        self.fitting_net = Sequential(*layers)

    def forward(self, x: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:

        xx: Tuple[torch.Tensor, torch.Tensor] = (x, channels[0])
        output, _ = self.fitting_net(xx)
        return output

