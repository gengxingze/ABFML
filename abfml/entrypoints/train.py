import json
import os
import sys
import importlib.util
import time
import torch
import numpy as np
from typing import Optional
from torch.utils.data import ConcatDataset, DataLoader, Subset
from abfml.param.param import Param
from abfml.model.normal import DPNormal, FeatureNormal
from abfml.logger.loggers import Logger, log_data_info
from abfml.data.read_data import ReadData
from abfml.train.trainer import train_loop, valid_loop
from abfml.optimizer.optim_init import optim_init
from abfml.optimizer.learn_rate import calc_lr_param
import torch.multiprocessing


def train_mlff(
        *,
        INPUT: str,
        init_model: Optional[str],
        restart: Optional[str],
        **kwargs):
    with open(INPUT, 'r') as file:
        input_json = json.load(file)
    # with open('std_input.json', 'w') as file:
    #     json.dump(input_set_dict, file, indent=4)
    param_class = Param(input_dict=input_json)

    logger = Logger(param_class.GlobalSet.logger_file).logger
    logger.info(r"################################################################################################")
    logger.info(r"#                         _       ______   ________  ____    ____  _____                       #")
    logger.info(r"#                        / \     |_   _ \ |_   __  ||_   \  /   _||_   _|                      #")
    logger.info(r"#                       / _ \      | |_) |  | |_ \_|  |   \/   |    | |                        #")
    logger.info(r"#                      / ___ \     |  __'.  |  _|     | |\  /| |    | |   _                    #")
    logger.info(r"#                    _/ /   \ \_  _| |__) |_| |_     _| |_\/_| |_  _| |__/ |                   #")
    logger.info(r"#                   |____| |____||_______/|_____|   |_____||_____||________|                   #")
    logger.info(r"#                                                                                              #")
    logger.info(r"################################################################################################")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_threads = torch.get_num_threads()
    num_worker = int(num_threads / 2)
    # torch.set_num_threads(1)
    logger.info("num_threads : %d" % num_threads)
    torch.set_printoptions(precision=10)
    if param_class.DataSet.ratio is None:
        logger.info("+-------------------------------------- train data file ---------------------------------------+")
        train_dataclass = ReadData(filename=param_class.DataSet.train_file,
                                   cutoff=param_class.GlobalSet.cutoff,
                                   neighbor=param_class.GlobalSet.neighbor,
                                   type_map=param_class.GlobalSet.type_map,
                                   file_format=param_class.DataSet.file_format)
        log_data_info(logger, train_dataclass)
        train_data = ConcatDataset(train_dataclass.get_mlffdata())

        logger.info("+-------------------------------------- valid data file ---------------------------------------+")
        valid_dataclass = ReadData(filename=param_class.DataSet.valid_file,
                                   cutoff=param_class.GlobalSet.cutoff,
                                   neighbor=param_class.GlobalSet.neighbor,
                                   type_map=param_class.GlobalSet.type_map,
                                   file_format=param_class.DataSet.file_format)
        log_data_info(logger, valid_dataclass)
        valid_data = ConcatDataset(valid_dataclass.get_mlffdata())
    else:
        logger.info("+----------------------------------------- data file -------------------------------------------+")
        total_dataclass = ReadData(filename=param_class.DataSet.train_file,
                                   cutoff=param_class.GlobalSet.cutoff,
                                   neighbor=param_class.GlobalSet.neighbor,
                                   type_map=param_class.GlobalSet.type_map,
                                   file_format=param_class.DataSet.file_format)
        log_data_info(logger, total_dataclass)
        total_data = ConcatDataset(total_dataclass.get_mlffdata())
        total_indices = np.arange(len(total_data))
        if param_class.DataSet.shuffle:
            logger.info(f"| Because of shuffle is set to true, ")
            logger.info(f"| The training and validation sets will be randomly selected from all the data.")
            np.random.shuffle(total_indices)
        else:
            logger.info(f"| Because of shuffle is set to false, preserve the order of the dataset, scaled")
        split_point = int(len(total_data) * param_class.DataSet.ratio)
        train_indices, valid_indices = total_indices[:split_point], total_indices[split_point:]
        train_data = Subset(total_data, train_indices)
        valid_data = Subset(total_data, valid_indices)
    train_data_load = DataLoader(train_data, batch_size=param_class.TrainSet.batch_size,
                                 shuffle=True, num_workers=num_worker)
    valid_data_load = DataLoader(valid_data, batch_size=param_class.TrainSet.batch_size,
                                 shuffle=False, num_workers=num_worker)

    if init_model is not None:
        # old_model = torch.jit.script(init_model)
        # if old_model.Rcut != param_class.GlobalSet.cutoff:
        #     raise Exception("")
        # if old_model.neighbor != param_class.GlobalSet.neighbor:
        #     raise Exception("")
        # if param_class.GlobalSet.type_map in old_model.type():
        #     raise Exception("")
        # if param_class.model_name == "deepmd":
        #     if param_class.deep.name == "se_e2_a":
        #         from src.model.dp_se_e2_a import DpSe2a
        #         model = DpSe2a
        # elif param_class.model_name == "user_defined":
        #     model_path = os.path.dirname(param_class.UseDefined.model_path)
        #     sys.path.append(model_path)
        #     try:
        #         import user_model
        #         model = user_model.user_model(param_class.UseDefined.config)
        #     except:
        #         raise Exception(f"Undefined deepmd network: {param_class.deep.name}")
        # else:
        #     raise Exception(f"If you define your own machine learning force field, use the name: user_defined")
        pass

    elif restart is not None:
        check_point_class = torch.load(restart)
        model = check_point_class["model"]
    else:
        if param_class.model_name in ["dp_se_e2_a", "dp_se_e3", "dp_se_e2_r"]:
            if param_class.model_name == "dp_se_e2_r":
                from abfml.model.field_model_dp import DpSe2r
                energy_shift, std_mean = DPNormal(normal_data=train_data,
                                                  param_class=param_class,
                                                  normal_rate='auto',
                                                  coordinate_matrix='r').normal_data
                model_class = DpSe2r
            elif param_class.model_name == "dp_se_e2_a":
                from abfml.model.field_model_dp import DpSe2a
                energy_shift, std_mean = DPNormal(normal_data=train_data,
                                                  param_class=param_class,
                                                  normal_rate='auto',
                                                  coordinate_matrix='a').normal_data
                model_class = DpSe2a
            else:
                from abfml.model.field_model_dp import DpSe3
                energy_shift, std_mean = DPNormal(normal_data=train_data,
                                                  param_class=param_class,
                                                  normal_rate='auto',
                                                  coordinate_matrix='a').normal_data
                model_class = DpSe3

            model = model_class(type_map=param_class.GlobalSet.type_map,
                                cutoff=param_class.GlobalSet.cutoff,
                                neighbor=param_class.GlobalSet.neighbor,
                                fitting_config=param_class.DeepSe.fitting_config,
                                embedding_config=param_class.DeepSe.embedding_config,
                                energy_shift=energy_shift,
                                std_mean=std_mean)

        elif param_class.model_name == "BPMlp":
            from abfml.model.field_model_bp import BPMlp
            energy_shift, std_mean = FeatureNormal(normal_data=train_data,
                                                   param_class=param_class,
                                                   normal_rate='auto',
                                                   feature_name='BP').normal_data
            model = BPMlp(type_map=param_class.GlobalSet.type_map,
                          cutoff=param_class.GlobalSet.cutoff,
                          neighbor=param_class.GlobalSet.neighbor,
                          fit_config=param_class.BPDescriptor.fitting_config,
                          bp_features_information=param_class.BPDescriptor.bp_features_information,
                          bp_features_param=param_class.BPDescriptor.bp_features_param,
                          energy_shift=energy_shift,
                          std_mean=std_mean)

        elif param_class.model_name == "NEP":
            from abfml.model.field_model_nep import NEP
            model = NEP(type_map=param_class.GlobalSet.type_map,
                        cutoff=param_class.GlobalSet.cutoff,
                        neighbor=param_class.GlobalSet.neighbor,
                        fitting_config=param_class.NEPParam.fitting_config,
                        feature_config=param_class.NEPParam.feature_config,
                        energy_shift=[0,0],
                        std_mean=[0])
        elif param_class.model_name == "user_defined":
            field_model_name = param_class.user_defined.field_name
            model_path = param_class.user_defined.model_path
            spec = importlib.util.spec_from_file_location(field_model_name, model_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[field_model_name] = module
            spec.loader.exec_module(module)

            field_model = getattr(module, field_model_name)
            if param_class.user_defined.normal_name is not None:
                normal_class = getattr(module, param_class.user_defined.normal_name)
                normal_model = normal_class(normal_data=train_data, param_class=param_class, normal_rate="auto")
                normal_data = normal_model.normal_data
            else:
                normal_data = None
            model = field_model(type_map=param_class.GlobalSet.type_map,
                                cutoff=param_class.GlobalSet.cutoff,
                                neighbor=param_class.GlobalSet.neighbor,
                                config=param_class.user_defined.config,
                                normal=normal_data)
        else:
            raise Exception(f"If you define your own machine learning force field, use the name: user_defined")

    if param_class.GlobalSet.dtype == "float32":
        model.to(torch.float32)
    else:
        model.to(torch.float64)
    model.to(device=device)

    if len(param_class.TrainSet.freeze_param) != 0:
        logger.info(f"+----------------------------------- freeze or train param ------------------------------------+")
        namelist = [name for name, _ in model.named_parameters()]
        if not all(name in namelist for name in param_class.TrainSet.freeze_param[1:]):
            raise Exception("Error: Some parameters were not found in the model.")
        if param_class.TrainSet.freeze_param[0] == "freeze":
            logger.info(f"| The following parameters will be frozen.")
            for name, param in model.named_parameters():
                if name in param_class.TrainSet.freeze_param[1:]:
                    param.requires_grad = False
        elif param_class.TrainSet.freeze_param[0] == "train":
            logger.info(f"| The following parameters will be trained.")
            for name, param in model.named_parameters():
                if name not in param_class.TrainSet.freeze_param:
                    param.requires_grad = False
        else:
            raise Exception(f"Keyword {param_class.TrainSet.freeze_param[0]} error, flags only allowed freeze and train!")
        for name in param_class.TrainSet.freeze_param[1:]:
            logger.info(f"| {name:<60s}")
        logger.info(f"+----------------------------------------------------------------------------------------------+")

    iters_step: int = 0
    start_epoch: int = 1
    if restart is not None:
        check_point_class = torch.load(restart)
        optimizer = check_point_class["optimizer"]
        start_epoch = check_point_class["epoch"]
        iters_step = check_point_class["iters_step"]
    else:
        optimizer = optim_init(model=model, param=param_class.TrainSet)
        calc_lr_param(Lr_param=param_class.LrSet,
                      epoch=param_class.TrainSet.epoch,
                      train_iters=int(len(train_data)/param_class.TrainSet.batch_size),
                      logger_name=param_class.GlobalSet.logger_file)

    logger.info(r"########################################--Start  Train--########################################")
    logger.info(f"+------------------------------------------Start Loop------------------------------------------+")
    time_loop_start = time.time()
    for epoch in range(start_epoch, param_class.TrainSet.epoch + 1):
        logger.info(f"|-- Train Epoch:{epoch:>3d}, batch size:{train_data_load.batch_size:>3d}, "
                    f"total iters: {len(train_data_load):>6d} --------------------------------------|")
        train_loss_dict = train_loop(data_load=train_data_load,
                                     model=model,
                                     optimizer=optimizer,
                                     iters_step=iters_step,
                                     config=param_class)

        iters_step = iters_step + len(train_data_load)

        logger.info(f"|-- Valid Epoch:{epoch:>3d}, batch size:{valid_data_load.batch_size:>3d}, "
                    f"total iters: {len(valid_data_load):>6d}                                       |")
        valid_loss_dict, _ = valid_loop(data_load=valid_data_load,
                                        model=model,
                                        logger_name=param_class.GlobalSet.logger_file,
                                        print_freq=param_class.TrainSet.print_freq,
                                        save_predict=False)

        # ************************************** Start writing loss file ************************************** #
        if not os.path.exists(param_class.LossSet.loss_file) or epoch == 1:
            header = '{:>8}'.format("Epoch")
            header_time = ""
            for train_loss_key in train_loss_dict.keys():
                if train_loss_key == "time":
                    header_time = '{:>16s}(s)'.format("Total_Time")
                else:
                    header += '{:>18s}{:>18s}'.format("T_RMSE_" + train_loss_dict[train_loss_key].name,
                                                      "V_RMSE_" + train_loss_dict[train_loss_key].name)
            header += header_time
            with open(param_class.LossSet.loss_file, 'w') as fp:
                fp.write(header)

        summaries = f"\n{epoch:>8d}"
        summaries_time = ""
        for train_loss_key in train_loss_dict.keys():
            if train_loss_key == "time":
                total_time = '{:6.2f}'.format(train_loss_dict["time"].sum + valid_loss_dict["time"].sum)
                summaries_time = '{:>16s}'.format(total_time)
            else:
                summaries += '{:>18s}'.format(format(train_loss_dict[train_loss_key].avg, '.5e'))
                summaries += '{:>18s}'.format(format(valid_loss_dict[train_loss_key].avg, '.5e'))
        summaries += summaries_time
        with open(param_class.LossSet.loss_file, 'a') as fp:
            fp.write(summaries)
        # **************************************  Stop writing loss file ************************************** #

        if epoch % param_class.TrainSet.save_check == 0:
            check_point_path = "checkpoint_" + str(epoch) + ".ckpt"
        elif epoch == param_class.TrainSet.epoch:
            check_point_path = "checkpoint_final.ckpt"
        else:
            check_point_path = "checkpoint_latest.ckpt"

        torch.save({
            'epoch': epoch,
            'iters_step': iters_step,
            'model': model,
            'optimizer': optimizer
        }, check_point_path)
    try:
        jit_model = torch.jit.script(model)
        jit_model.save(param_class.TrainSet.model_file)
    except Exception as e:
        torch.save(model, param_class.TrainSet.model_file+'.pth')
        logger.info(f"| Information: Failed to save the model via torchscript, will save the model by torch.save! \n"
                    f"               The model can't be loaded in lammps, but it can be used in ASE")
    time_loop_end = time.time()
    logger.info(f"+------------------------------------------ End Loop ------------------------------------------+")
    logger.info(f"| Total training time:{time_loop_end - time_loop_start:8.2f}s, total training iters:{iters_step:>8d}")
    logger.info(r"######################################## --End  Train-- ########################################")

    return
