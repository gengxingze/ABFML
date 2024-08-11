import torch
import torch.nn as nn
from typing import List
from abfml.model.math_fun import ActivationModule


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
