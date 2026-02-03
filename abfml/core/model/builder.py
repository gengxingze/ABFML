import torch
import importlib.util
import sys
from pathlib import Path
from typing import Optional
from abfml.core.model.normal import DPNormal, FeatureNormal


MODEL_REGISTRY = {}

def register_model(name: str):
    def _register(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return _register


class BaseModelBuilder:
    def __init__(self, param_class, normal_data):
        self.param = param_class
        self.normal_data = normal_data

    def build(self):
        raise NotImplementedError



class BaseDpBuilder(BaseModelBuilder):
    model_cls = None
    coord = "a"

    def build(self):
        energy_shift, std_mean = DPNormal(
            normal_data=self.normal_data,
            param_class=self.param,
            normal_rate="auto",
            coordinate_matrix=self.coord
        ).normal_data

        return self.model_cls(
            type_map=self.param.GlobalSet.type_map,
            cutoff=self.param.GlobalSet.cutoff,
            neighbor=self.param.GlobalSet.neighbor,
            fitting_config=self.param.DeepSe.fitting_config,
            embedding_config=self.param.DeepSe.embedding_config,
            energy_shift=energy_shift,
            std_mean=std_mean
        )


@register_model("dp_se_e2_r")
class DpSe2rBuilder(BaseDpBuilder):
    from abfml.core.model.dpse import DpSe2r
    model_cls = DpSe2r
    coord = "r"


@register_model("dp_se_e2_a")
class DpSe2aBuilder(BaseDpBuilder):
    from abfml.core.model.dpse import DpSe2a
    model_cls = DpSe2a
    coord = "a"


@register_model("dp_se_e3")
class DpSe3Builder(BaseDpBuilder):
    from abfml.core.model.dpse import DpSe3
    model_cls = DpSe3
    coord = "a"


@register_model("BPMlp")
class BPMlpBuilder(BaseModelBuilder):
    def build(self):
        from abfml.core.model.bpmlp import BPMlp
        energy_shift, std_mean = FeatureNormal(
            normal_data=self.normal_data,
            param_class=self.param,
            normal_rate="auto",
            feature_name="BP"
        ).normal_data

        return BPMlp(
            type_map=self.param.GlobalSet.type_map,
            cutoff=self.param.GlobalSet.cutoff,
            neighbor=self.param.GlobalSet.neighbor,
            fit_config=self.param.BPDescriptor.fitting_config,
            bp_features_information=self.param.BPDescriptor.bp_features_information,
            bp_features_param=self.param.BPDescriptor.bp_features_param,
            energy_shift=energy_shift,
            std_mean=std_mean
        )

@register_model("NEP")
class NEPBuilder(BaseModelBuilder):
    def build(self):
        from abfml.core.model.nep import NEP
        return NEP(
            type_map=self.param.GlobalSet.type_map,
            cutoff=self.param.GlobalSet.cutoff,
            neighbor=self.param.GlobalSet.neighbor,
            fitting_config=self.param.NEPParam.fitting_config,
            feature_config=self.param.NEPParam.feature_config,
            energy_shift=[0, 0],
            std_mean=[torch.tensor([0.0])]
        )


def initialize_model(param_class, init_model: Optional[str], logger):
    # 尝试加载已有模型
    try:
        model = torch.jit.load(init_model)
        logger.info(f"Successfully loaded JIT model from {init_model}")
    except RuntimeError:
        try:
            model = torch.load(init_model, map_location="cpu")
            logger.info(f"Successfully loaded PyTorch model from {init_model}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model using both torch.jit.load and torch.load: {e}")

    # 检查模型基本参数是否匹配
    if hasattr(model, 'cutoff') and model.cutoff != param_class.GlobalSet.cutoff:
        raise ValueError(f"Cutoff mismatch: model={model.cutoff}, config={param_class.GlobalSet.cutoff}")
    if hasattr(model, 'neighbor') and model.neighbor != param_class.GlobalSet.neighbor:
        raise ValueError(f"Neighbor mismatch: model={model.neighbor}, config={param_class.GlobalSet.neighbor}")
    if hasattr(model, 'type_map') and param_class.GlobalSet.type_map not in model.type_map:
        raise ValueError(f"type_map mismatch: config type_map {param_class.GlobalSet.type_map} not in model")

    return model


def load_user_module(pyfile: str):
    pyfile = Path(pyfile).resolve()
    module_name = pyfile.stem   # user_model

    spec = importlib.util.spec_from_file_location(module_name, pyfile)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def build_model(param_class, normal_data, init_model=None, restart=None, logger=None):

    if init_model is not None:
        return initialize_model(param_class, init_model, logger)

    if restart is not None:
        ckpt = torch.load(restart, map_location="cpu")
        return ckpt["model"]

    name = param_class.user_defined.field_name

    if name not in MODEL_REGISTRY:
        if param_class.user_defined.model_path:
            load_user_module(param_class.user_defined.model_path)
        else:
            raise ValueError(f"Unknown model: {name}")

    builder = MODEL_REGISTRY[name](param_class, normal_data)
    return builder.build()





