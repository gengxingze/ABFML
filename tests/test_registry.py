import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_dim=10, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

MODEL_REGISTRY = {}

def register_model(name):
    def wrap(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrap


class BaseBuilder:
    def __init__(self, config):
        self.config = config

    def build(self):
        raise NotImplementedError


@register_model("simple")
class SimpleBuilder(BaseBuilder):
    def build(self):
        return SimpleNet(
            in_dim=self.config["in_dim"],
            hidden=self.config["hidden"]
        )

config = {"in_dim": 10, "hidden": 64}

builder = MODEL_REGISTRY["simple"](config)
model = builder.build()

x = torch.randn(4, 10)

# 普通前向
y = model(x)
print("raw:", y.shape)

# ============ JIT SCRIPT ============
scripted = torch.jit.script(model)
scripted.save("simple_script.pt")

print("saved:")




