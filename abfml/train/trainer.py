import time
import torch
from collections import OrderedDict
from ase import Atoms
from abfml.param.param import Param
from abfml.logger.loggers import AverageMeter, Logger
from abfml.loss.losser import calculate_weight
from abfml.optimizer.learn_rate import adjust_lr
from abfml.data.read_data import ReadData
import sys


def train_loop(data_load, model, optimizer, iters_step: int, config: Param):
    logger = Logger(config.GlobalSet.logger_file).logger
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    batch_size = config.TrainSet.batch_size
    criterion = torch.nn.MSELoss(reduction='sum')
    predict_name = ["energy",  "atomic_energy", "force", "virial"]
    # register losses start
    loss_dict = OrderedDict()
    loss_dict["loss"] = AverageMeter(name="Loss", fmt=".2e", summary_type="AVERAGE")
    loss_dict["time"] = AverageMeter(name="Time", fmt="2.3f", summary_type="SUM")

    loss_set = config.LossSet
    if loss_set.start_energy_weight != 0.0 or loss_set.limit_energy_weight != 0.0:
        loss_dict["energy"] = AverageMeter(name="E_tot", fmt=".2e", summary_type="AVERAGE")
    if loss_set.start_ei_weight != 0.0 or loss_set.limit_ei_weight != 0.0:
        loss_dict["atomic_energy"] = AverageMeter(name="Ei", fmt=".2e", summary_type="AVERAGE")
    if loss_set.start_force_weight != 0.0 or loss_set.limit_force_weight != 0.0:
        loss_dict["force"] = AverageMeter(name="Force", fmt=".2e", summary_type="AVERAGE")
    if loss_set.start_virial_weight != 0.0 or loss_set.limit_virial_weight != 0.0:
        loss_dict["virial"] = AverageMeter(name="Virial", fmt=".2e", summary_type="AVERAGE")
    # register losses end
    weight_dict = {}
    model.train()
    for iters, image_batch in enumerate(data_load):
        time_start = time.time()

        lr_real = adjust_lr(Lr_param=config.LrSet, iters_step=iters_step + iters + 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_real

        n_atoms = image_batch["n_atoms"][0].int().to(device)
        Rij = image_batch["Rij"].to(dtype=dtype, device=device)
        element_type = image_batch["element_type"][0].int().to(device)
        Zi = image_batch["Zi"].int().to(device)
        Nij = image_batch["Nij"].int().to(device)
        Zij = image_batch["Zij"].int().to(device)

        predict_tulp = model(element_type, Zi, Nij, Zij, Rij, 0)
        Lr_ratio = lr_real / config.LrSet.start_lr
        weight_tulp = calculate_weight(param=config.LossSet, learn_rate=Lr_ratio)
        weight_dict["energy"] = batch_size * n_atoms ** 2
        weight_dict["atomic_energy"] = batch_size * n_atoms
        weight_dict["force"] = batch_size * 3 * n_atoms
        weight_dict["virial"] = batch_size * 9 * n_atoms ** 2

        loss = torch.tensor([0.0], dtype=dtype, device=device)
        loss_val = torch.tensor([0.0], dtype=dtype, device=device)
        for name in loss_dict.keys():
            if name in predict_name:
                try:
                    label_val = image_batch[name].to(dtype=dtype, device=device)
                    indices = predict_name.index(name)
                    predict_val = predict_tulp[indices]
                    MSE = criterion(label_val, predict_val) / weight_dict[name]
                    loss_label_RMSE = torch.sqrt(MSE)
                    loss_val = loss_val + loss_label_RMSE
                    loss = loss + MSE * weight_tulp[indices]
                    loss_dict[name].update(loss_label_RMSE.item(), batch_size)
                except KeyError:
                    raise Exception(f"You are trying to train {name}, "
                                    f"but the dataset doesn't include data for {name}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        time_end = time.time()
        loss_dict["loss"].update(loss_val.item(), batch_size)
        loss_dict["time"].update(time_end - time_start)
        if iters % config.TrainSet.print_freq == 0:
            logger.info(f"| Train Iters:{iters + 1:>6d} ," + ' ,'.join([str(dit) for dit in loss_dict.values()]))
    logger.info(f"| Train Information: " + ' ,'.join([dit.summary() for dit in loss_dict.values()]))
    return loss_dict


def valid_loop(data_load, model, logger_name: str = 'valid.log', print_freq: int = 1, save_predict: bool = False):
    logger = Logger(logger_name).logger
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    batch_size = data_load.batch_size
    criterion = torch.nn.MSELoss(reduction='sum')
    predict_name = ["energy", "atomic_energy", "force", "virial"]
    predict_data = {"energy": [], "atomic_energy": [], "force": [], "virial": []}
    # register losses start
    loss_dict = OrderedDict()
    loss_dict["loss"] = AverageMeter(name="Loss", fmt=".2e", summary_type="AVERAGE")
    loss_dict["time"] = AverageMeter(name="Time", fmt="2.3f", summary_type="SUM")
    loss_dict["energy"] = AverageMeter(name="E_tot", fmt=".2e", summary_type="AVERAGE")
    loss_dict["atomic_energy"] = AverageMeter(name="Ei", fmt=".2e", summary_type="AVERAGE")
    loss_dict["force"] = AverageMeter(name="Force", fmt=".2e", summary_type="AVERAGE")
    loss_dict["virial"] = AverageMeter(name="Virial", fmt=".2e", summary_type="AVERAGE")
    # register losses end
    weight_dict = {}
    model.eval()
    common_label = set([])
    for iters, image_batch in enumerate(data_load):
        time_start = time.time()
        n_atoms = image_batch["n_atoms"][0].int().to(device)
        Rij = image_batch["Rij"].to(dtype=dtype, device=device)
        element_type = image_batch["element_type"][0].int().to(device)
        Zi = image_batch["Zi"].int().to(device)
        Nij = image_batch["Nij"].int().to(device)
        Zij = image_batch["Zij"].int().to(device)

        predict_tulp = model(element_type, Zi, Nij, Zij, Rij, 0)

        weight_dict["energy"] = batch_size * n_atoms ** 2
        weight_dict["atomic_energy"] = batch_size * n_atoms
        weight_dict["force"] = batch_size * 3 * n_atoms
        weight_dict["virial"] = batch_size * 9 * n_atoms ** 2

        loss_val = torch.tensor([0], dtype=dtype, device=device)

        common_label = set(loss_dict.keys()).intersection(set(image_batch.keys()))
        if len(common_label) > 0:
            for name in common_label:
                label_val = image_batch[name].to(dtype=dtype, device=device)
                indices = predict_name.index(name)
                predict_val = predict_tulp[indices]
                MSE = criterion(label_val, predict_val) / weight_dict[name]
                loss_label_RMSE = torch.sqrt(MSE)
                loss_val = loss_val + loss_label_RMSE
                loss_dict[name].update(loss_label_RMSE.item(), batch_size)
                if save_predict:
                    predict_data[name].append(predict_val.detach())
        else:
            raise Exception("If you want to verify the accuracy of a dataset, then at least one of the labels "
                            "(energy, atomic_energy, force, stress) should be present in your dataset")
        time_end = time.time()
        loss_dict["loss"].update(loss_val.item(), batch_size)
        loss_dict["time"].update(time_end - time_start)
        if iters % print_freq == 0:
            logger.info(f"| Valid Iters:{iters + 1:>6d} ,"
                        + ' ,'.join([str(loss_dict[label]) for label in ["loss", "time"] + list(common_label)]))
    logger.info(f"| Valid Information: " +
                ' ,'.join([loss_dict[label].summary() for label in ["loss", "time"] + list(common_label)]))
    return loss_dict, predict_data


def predict(atom: Atoms, model):
    type_map = model.type_map
    cutoff = model.cutoff
    neighbor = model.neighbor
    information_tulp = ReadData.calculate_neighbor(atom=atom, cutoff=cutoff, neighbor=neighbor, type_map=type_map)
    element_type, Zi, Nij, Zij, Rij = information_tulp
    predict_tulp = model(element_type, Zi, Nij, Zij, Rij, 0)
    Etot, Ei, Force, Virial, virial = predict_tulp
    return Etot, Ei, Force, Virial, virial
