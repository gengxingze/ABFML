import torch
from abfml.param.param import TrainSet


def optim_init(model, param: TrainSet):
    if param.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), eps=1e-5)
    elif param.optimizer == "user_defined":
        optimizer = None
    else:
        raise Exception("")
    return optimizer




