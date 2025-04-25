import json
import sys
import importlib.util
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.build import bulk
from typing import Optional
from torch.utils.data import ConcatDataset
from abfml.param.param import Param
from abfml.logger.loggers import Logger, log_logo
from abfml.data.read_data import ReadData
from abfml.calculate import ABFML


def check_mlff(
        *,
        model_name: Optional[str] = None,
        input: Optional[str] = None,
        dtype: str = "float32",
        **kwargs):
    logger = Logger("check.log").logger
    log_logo(logger=logger)

    if input is not None:
        with open(input, 'r') as file:
            input_json = json.load(file)
        param_class = Param(input_dict=input_json)
        field_model_name = param_class.user_defined.field_name
        model_path = param_class.user_defined.model_path
        spec = importlib.util.spec_from_file_location(field_model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[field_model_name] = module
        spec.loader.exec_module(module)

        field_model = getattr(module, field_model_name)
        if param_class.user_defined.normal_name is not None:
            logger.info(f"| Information: Since you defined Normal, we'll read the train_file as the dataset for Normal")
            train_dataclass = ReadData(filename=param_class.DataSet.train_file,
                                       cutoff=param_class.GlobalSet.cutoff,
                                       neighbor=param_class.GlobalSet.neighbor,
                                       file_format=param_class.DataSet.file_format)
            train_data = ConcatDataset(train_dataclass.create_dataset())
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
        calculator = ABFML(model_class=model,dtype=dtype)

    elif model_name is not None:
        calculator = ABFML(model=model_name,dtype=dtype)

    else:
        raise RuntimeError("model_name and input are None?")

    atomic_number = calculator.model.type_map[0]
    element_symbol = chemical_symbols[atomic_number]

    logger.info(f"+----------------------------------- rotate symmetry check ------------------------------------+")
    position = np.array([[15.0, 15.0, 15.0],[13.0, 15.0, 15.0],[15.0, 15.0, 13.0]])
    cluster = Atoms(element_symbol+'3', position)
    cluster.calc = calculator
    energy = [cluster.get_potential_energy()]
    force = [cluster.get_forces()]
    cluster.rotate(45, 'z', center='COP')
    energy.append(cluster.get_potential_energy())
    force.append(cluster.get_forces())
    dF = (np.linalg.norm(force[1], axis=1) - np.linalg.norm(force[0], axis=1)).sum().item()
    logger.info(f"       cluster-> energy: {energy[0]:>.6f}, force-sum: {force[0].sum().item():>.6f}")
    logger.info(f"rotate cluster-> energy: {energy[1]:>.6f}, force-sum: {force[1].sum().item():>.6f}")
    logger.info(f"    difference-> energy: {energy[1]-energy[0]:>.6f}, force: {dF:>.6f}")
    logger.info(f"Information: The magnitude of the energy/and the combined force remains the same after the")
    logger.info(f"             structure is rotated. Isolated system combined force should be 0.")
    logger.warning(f"Part “difference” should be 0. But if it is less than 1e-5, ")
    logger.warning(f"it may be a precision issue, otherwise check the model rotation conservation")
    logger.info(f"+----------------------------------------------------------------------------------------------+")

    logger.info(f"+-------------------------------------- periodicity check -------------------------------------+")
    cluster = bulk('Cu', 'fcc', a=3.62, cubic=True)
    cluster.calc = calculator
    energy = [cluster.get_potential_energy()]
    force = [cluster.get_forces()]
    cluster = cluster * (3, 3, 3)
    cluster.calc = calculator
    energy.append(cluster.get_potential_energy())
    force.append(cluster.get_forces())
    dE = energy[1] - energy[0] * 27
    dF = np.abs(force[1][:4] - force[0]).sum().item()
    logger.info(f"       bulk-> energy: {energy[0]:>.6f}, force: {force[0].sum().item():>.6f}")
    logger.info(f" 3*3*3 bulk-> energy: {energy[1]:>.6f}, force: {force[1].sum().item():>.6f}")
    logger.info(f" difference-> E[1]-E[0]*27: {dE:>.6f},  dF: {dF:>.6f}")
    logger.info(f"Information: The energy of the expanded cell of a periodic structure is proportional to the number")
    logger.info(f"             of times it is expanded and the force on the corresponding atoms is unchanged.")
    logger.warning(f"If the difference is less than 1e-5 it may be an accuracy problem,")
    logger.warning(f"otherwise the energy and force relationship should be checked.")
    logger.info(f"+----------------------------------------------------------------------------------------------+")

    logger.info(f"+------------------------------------- energy-force check -------------------------------------+")
    cluster = Atoms(element_symbol + '3', position)
    cluster.calc = calculator
    energy = [cluster.get_potential_energy()]
    force = [cluster.get_forces()[0]]
    cluster.positions[0] = [15.0, 15.0, 15.002]
    energy.append(cluster.get_potential_energy())
    force.append(cluster.get_forces()[0])
    Fdx = -1 * (force[0][2] + force[1][2]) / 2 * 0.002
    dE = energy[1] - energy[0]
    logger.info(f"       cluster-> energy: {energy[0]:>.6f}, force: {force[0][2].item():>.6f}")
    logger.info(f"  perturbation-> energy: {energy[1]:>.6f}, force: {force[1][2].item():>.6f}")
    logger.info(f"              -> dE: {dE:>.6f}, F * dx: {Fdx:>.6f}, difference: {dE - Fdx:>.6f}")
    logger.info(f"Information: Force and energy should satisfy F=-dE/dx, dE=F*dx for perturbation.")
    logger.warning(f"if the difference is less than 1e-5 it may be an accuracy problem,")
    logger.warning(f"otherwise the energy and force relationship should be checked.")
    logger.info(f"+----------------------------------------------------------------------------------------------+")

