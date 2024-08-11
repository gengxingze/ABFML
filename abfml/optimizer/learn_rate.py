import math
from abfml.param.param import LrSet
from abfml.logger.loggers import Logger


def adjust_lr(Lr_param: LrSet, iters_step: int):
    if Lr_param.type_name == 'exp':
        lr_t = Lr_param.start_lr * Lr_param.decay_rate ** (iters_step / Lr_param.decay_step)
    elif Lr_param.type_name == 'exp_decay':
        lr_t = Lr_param.start_lr * Lr_param.decay_rate ** int(iters_step / Lr_param.decay_step)
    else:
        raise TypeError(f'Unsupported type_name of learning rate: {Lr_param.type_name}')
    if lr_t < Lr_param.limit_lr:
        lr_t = Lr_param.limit_lr
    return lr_t


def calc_lr_param(Lr_param: LrSet, epoch: int, train_iters: int, logger_name: str = None):
    total_batches = epoch * train_iters
    if Lr_param.decay_rate is None:
        Lr_param.decay_rate = (Lr_param.limit_lr / Lr_param.start_lr) ** int(Lr_param.decay_step / total_batches)
        if logger_name is not None:
            logger = Logger(logger_name).logger
            logger.info(f'start learn rate: {Lr_param.start_lr}, limit learn rate:{Lr_param.limit_lr}, '
                        f'decay step:{Lr_param.decay_step}, decay rate will be: {Lr_param.decay_rate}')
    else:
        need_batches = math.log(Lr_param.limit_lr / Lr_param.start_lr) / math.log(Lr_param.decay_rate) * Lr_param.decay_step
        if logger_name is not None:
            logger = Logger(logger_name).logger
            if need_batches < total_batches:
                logger.info(f'Will reach limit_lr in about {math.ceil(need_batches/train_iters)} epochs.')
            else:
                logger.warning(f'Need {math.ceil(need_batches/train_iters)} epochs to reach limit_lr, '
                               f'but only train for {epoch} epochs!')

