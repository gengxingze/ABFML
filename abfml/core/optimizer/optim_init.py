import torch
from abfml.param.param import TrainSet


def optim_init(model, param: TrainSet):
    """
    Initialize optimizer based on param.optimizer and param.lr.
    Optionally uses param.weight_decay and param.momentum if provided.
    """
    optim_name = param.optimizer
    lr = 0.001
    weight_decay = getattr(param, "weight_decay", 0.0)
    momentum = getattr(param, "momentum", 0.9)

    if optim_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-5)
    elif optim_name == "Adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    elif optim_name == "user_defined":
        optimizer = None
    else:
        raise ValueError(f"Unsupported optimizer type: {param.optimizer}")

    return optimizer



