import time
import torch
from abfml.param.param import Param
from abfml.logger.loggers import Logger
from abfml.core.loss.losser import calculate_weight, register_loss_meters
from abfml.core.optimizer.learn_rate import adjust_lr


def train_loop(data_load, model, optimizer, iters_step: int, config: Param):
    logger = Logger(config.GlobalSet.logger_file).logger
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    batch_size = config.TrainSet.batch_size
    criterion = torch.nn.MSELoss(reduction='sum')
    predict_name = ["energy",  "atomic_energies", "forces", "virials"]
    # register losses start
    loss_dict = register_loss_meters(loss_set=config.LossSet)
    # register losses end
    weight_dict = {}
    model.train()
    for iters, image_batch in enumerate(data_load):
        time_start = time.time()

        lr_real = adjust_lr(lr_param=config.LrSet, iters_step=iters_step + iters + 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_real

        n_atoms = image_batch["num_atoms"][0].int().to(device)
        neighbor_vectors = image_batch["neighbor_vectors"].to(dtype=dtype, device=device)
        element_types = image_batch["element_types"][0].int().to(device)
        central_atoms = image_batch["central_atoms"].int().to(device)
        neighbor_indices = image_batch["neighbor_indices"].int().to(device)
        neighbor_types = image_batch["neighbor_types"].int().to(device)

        predict_tulp = model(element_types, central_atoms, neighbor_indices, neighbor_types, neighbor_vectors, 0)
        lr_ratio = lr_real / config.LrSet.start_lr
        weight_tulp = calculate_weight(param=config.LossSet, learn_rate=lr_ratio)
        weight_dict["energy"] = batch_size * n_atoms ** 2
        weight_dict["atomic_energies"] = batch_size * n_atoms
        weight_dict["forces"] = batch_size * 3 * n_atoms
        weight_dict["virials"] = batch_size * 9 * n_atoms ** 2

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
    predict_name = ["energy", "atomic_energies", "forces", "virials"]
    predict_data = {"energy": [], "atomic_energies": [], "forces": [], "virials": []}
    # register losses start
    loss_dict = register_loss_meters()
    # register losses end
    weight_dict = {}
    model.eval()
    common_label = set([])
    for iters, image_batch in enumerate(data_load):
        time_start = time.time()
        n_atoms = image_batch["num_atoms"][0].int().to(device)
        neighbor_vectors = image_batch["neighbor_vectors"].to(dtype=dtype, device=device)
        element_types = image_batch["element_types"][0].int().to(device)
        central_atoms = image_batch["central_atoms"].int().to(device)
        neighbor_indices = image_batch["neighbor_indices"].int().to(device)
        neighbor_types = image_batch["neighbor_types"].int().to(device)

        predict_tulp = model(element_types, central_atoms, neighbor_indices, neighbor_types, neighbor_vectors, 0)

        weight_dict["energy"] = batch_size * n_atoms ** 2
        weight_dict["atomic_energies"] = batch_size * n_atoms
        weight_dict["forces"] = batch_size * 3 * n_atoms
        weight_dict["virials"] = batch_size * 9 * n_atoms ** 2

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
