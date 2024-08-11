from abfml.param.param import LossSet


def calculate_weight(param: LossSet, learn_rate: float):
    attributes = ['energy',  'ei', 'force', 'virial']
    weights = []

    for attr in attributes:
        start_weight = getattr(param, f'start_{attr}_weight')
        limit_weight = getattr(param, f'limit_{attr}_weight')
        weight = limit_weight * (1 - learn_rate) + start_weight * learn_rate
        weights.append(weight)

    return tuple(weights)



