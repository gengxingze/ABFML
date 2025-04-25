import math
from abfml.param.param import LrSet
from abfml.logger.loggers import Logger


def adjust_lr(lr_param: LrSet, iters_step: int) -> float:
    """Calculate the current learning rate based on schedule type."""
    if lr_param.type_name == 'exp':
        lr = lr_param.start_lr * lr_param.decay_rate ** (iters_step / lr_param.decay_step)
    elif lr_param.type_name == 'exp_decay':
        lr = lr_param.start_lr * lr_param.decay_rate ** (iters_step // lr_param.decay_step)
    else:
        raise ValueError(f"Unsupported learning rate type: {lr_param.type_name}")

    return max(lr, lr_param.limit_lr)


def calc_lr_param(lr_param: LrSet, epochs: int, train_iters: int, logger_name: str = None):
    """
    Calculate the decay_rate if not set. Otherwise estimate when limit_lr will be reached.
    """
    total_iters = epochs * train_iters
    logger = Logger(logger_name).logger if logger_name else None

    if lr_param.decay_rate is None:
        # Compute decay_rate to reach limit_lr in total_iters
        if total_iters > 0:
            lr_param.decay_rate = (lr_param.limit_lr / lr_param.start_lr) ** (lr_param.decay_step / total_iters)
            if logger:
                logger.info(f"[Auto LR] start_lr: {lr_param.start_lr:.2e}, limit_lr: {lr_param.limit_lr:.2e}, "
                            f"decay_step: {lr_param.decay_step}, computed decay_rate: {lr_param.decay_rate:.4f}")
        else:
            raise ValueError("Total training iterations (epochs * train_iters) must be greater than zero.")
    else:
        # Estimate how many iterations are needed to reach limit_lr
        try:
            needed_iters = math.log(lr_param.limit_lr / lr_param.start_lr) / math.log(lr_param.decay_rate)
            needed_iters *= lr_param.decay_step
            estimated_epochs = math.ceil(needed_iters / train_iters) if train_iters > 0 else float('inf')

            if logger:
                if needed_iters < total_iters:
                    logger.info(f"[LR Info] limit_lr will be reached in ~{estimated_epochs} epochs.")
                else:
                    logger.warning(f"[LR Warning] Need ~{estimated_epochs} epochs to reach limit_lr, "
                                   f"but training is set to {epochs} epochs.")
        except (ValueError, ZeroDivisionError) as e:
            if logger:
                logger.error(f"[LR Error] Failed to compute decay trajectory: {e}")

