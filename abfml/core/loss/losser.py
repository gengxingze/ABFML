from collections import OrderedDict
from abfml.param.param import LossSet
from abfml.logger.loggers import AverageMeter


def calculate_weight(param: LossSet, learn_rate: float):
    attributes = ['energy',  'ei', 'force', 'virial']
    weights = []

    for attr in attributes:
        start_weight = getattr(param, f'start_{attr}_weight')
        limit_weight = getattr(param, f'limit_{attr}_weight')
        weight = limit_weight * (1 - learn_rate) + start_weight * learn_rate
        weights.append(weight)

    return tuple(weights)


def register_loss_meters(loss_set: LossSet = None, base_meters=True):
    """
       General-purpose meter registration function.

       Parameters:
           loss_set (object): A config object that contains start and limit weights for each loss term.
           base_meters (bool): Whether to include basic meters like 'loss' and 'time'.

       Returns:
           OrderedDict: A dictionary of AverageMeter instances keyed by loss type.
       """
    meters = OrderedDict()

    if base_meters:
        meters["loss"] = AverageMeter("Loss", fmt=".2e", summary_type="AVERAGE")
        meters["time"] = AverageMeter("Time", fmt="2.3f", summary_type="SUM")

    if loss_set is not None:
        # Register optional loss terms based on non-zero weights
        optional_losses = {
            "energy": ("E_tot", loss_set.start_energy_weight, loss_set.limit_energy_weight),
            "atomic_energies": ("Ei", loss_set.start_ei_weight, loss_set.limit_ei_weight),
            "forces": ("Force", loss_set.start_force_weight, loss_set.limit_force_weight),
            "virials": ("Virial", loss_set.start_virial_weight, loss_set.limit_virial_weight)
        }

        for key, (display_name, start, limit) in optional_losses.items():
            if start != 0.0 or limit != 0.0:
                meters[key] = AverageMeter(display_name, fmt=".2e", summary_type="AVERAGE")

    else:
        display_names = {
            "energy": "E_tot",
            "atomic_energies": "Ei",
            "forces": "Force",
            "virials": "Virial"
        }
        for key in display_names:
            meters[key] = AverageMeter(display_names[key], fmt=".2e", summary_type="AVERAGE")
    return meters
