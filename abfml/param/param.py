import torch
from jsonschema import Draft7Validator, validators
from dataclasses import make_dataclass
from typing import Union

global_set_schema = {
    "type": "object",
    "properties": {
                    "type_map": {
                                "type": "array",
                                "items": {"type": "integer"}
                    },
                    "neighbor": {
                                "type": ["integer", "array"],
                                "items": {"type": "integer"}
                    },
                    "cutoff": {
                                "type": "number"
                    },
                    "dtype": {
                                "type": "string",
                                "enum": ["float32", "float64"],
                                "default": "float32"
                    },
                    "logger_file": {
                                "type": "string",
                                "default": "mlff.log"
                    }
    },
    "required": ["type_map", "neighbor", "cutoff"]
}

data_set_schema = {
    "type": "object",
    "properties": {
                    "file_format": {
                                "type": "string",
                                "default": None
                    },
                    "ratio": {
                                "type": "number", "minimum": 0, "maximum": 1,
                                "default": None},
                    "train_file":  {
                                "type": "array",
                                "items": {"type": "string"}
                    },
                    "valid_file": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": None
                    },
                    "shuffle": {
                                "type": "boolean",
                                "default": True
                    }
    },
    "required": ["train_file"]
}

train_set_schema = {
    "type": "object",
    "properties": {
                    "epoch": {
                                "type": "integer"
                    },
                    "optimizer": {
                                "type": "string",
                                "enum": ["Adam", "AdamW", "SGD"],
                                "default": "Adam"
                            },
                    "optim_set": {
                                "type": "object",
                                "additionalProperties": True
                    },

                    "freeze_param": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": None
                    },
                    "save_check": {
                                "type": "integer",
                                "default": 5
                    },
                    "batch_size": {
                                "type": "integer",
                                "default": 1
                    },
                    "print_freq": {
                                "type": "integer",
                                "default": 1
                    },
                    "model_file": {
                                "type": "string",
                                "default": "model.pt"
                    }
    },
    "if": {
        "properties": {"optimizer": {"const": "Adam"}}
    },
    "then": {
        "properties": {"optim_set": {
                            "properties": {"AdamW": {"type": "object"}}}}
    },
    "else": {
        "if": {
            "properties": {"optimizer": {"const": "SGD"}}
        },
        "then": {
            "properties": {"optim_set": {
                            "properties": {"SGD": {"type": "object"}}}}
        },
        "else": {
            "properties": {"optim_set": {
                            "properties": {"RMSProp": {"type": "object"}}}}
        }
    },
    "required": ["epoch", "optimizer"]
}

loss_set_schema = {
    "type": "object",
    "properties": {
                    "start_energy_weight": {
                                "type": "number",
                                "default": 0
                    },
                    "limit_energy_weight": {
                                "type": "number",
                                "default": 0
                    },
                    "start_force_weight": {
                                "type": "number",
                                "default": 0
                    },
                    "limit_force_weight": {
                                "type": "number",
                                "default": 0
                    },
                    "start_virial_weight": {
                                "type": "number",
                                "default": 0
                    },
                    "limit_virial_weight": {
                                "type": "number",
                                "default": 0
                    },
                    "start_ei_weight": {
                                "type": "number",
                                "default": 0
                    },
                    "limit_ei_weight": {
                                "type": "number",
                                "default": 0
                    },
                    "loss_file": {
                                "type": "string",
                                "default": "loss.curve"
                    }
    }
}

learning_rate_schema = {
                "type": "object",
                "properties": {
                    "type_name": {
                        "type": "string",
                        "enum": ["exp"],
                        "default": "exp"
                    },
                    "decay_step": {
                        "type": "integer",
                        "default": 1000
                    },
                    "decay_rate": {
                        "type": "number",
                        "default": None
                    },
                    "start_lr": {
                        "type": "number",
                        "default": 1e-3
                    },
                    "limit_lr": {
                        "type": "number",
                        "default": 1e-8
                    },
                }
}

fitting_net_schema= {
            "type": "object",
            "properties": {
                        "network_size": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "bias": {
                            "type": "boolean",
                            "default": True
                        },
                        "resnet_dt": {
                            "type": "boolean",
                            "default": False
                        },
                        "activate_function": {
                            "type": "string",
                            "default": "tanh",
                            "enum": ["tanh", "relu", "sigmoid", "softplus", "softsign"]
                        }
            },
            "required": ["network_size"]
}

deep_se_e = {
          "type": "object",
          "properties": {
                        "R_max": {
                            "type": "number"
                        },
                        "R_min": {
                            "type": "number"
                        },
                        "smooth_fun": {
                            "type": "string",
                            "default": "poly2_r",
                            "enum": ["cos", "cos_r", "tanh_u", "exp", "poly1_r", "poly2_r", "poly3_r"]
                        },
                        "network_size": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "bias": {
                            "type": "boolean",
                            "default": True
                        },
                        "resnet_dt": {
                            "type": "boolean",
                            "default": False
                        },
                        "activate_function": {
                            "type": "string",
                            "default": "tanh",
                            "enum": ["tanh", "relu", "sigmoid", "softplus", "softsign"]
                        },
                        "M2": {
                            "type": "integer",
                            "default": 16
                        }
          },
          "required": ["R_max", "R_min", "smooth_fun", "network_size"]
}


deep_se_e2_e = {
          "type": "object",
          "properties": {
                        "type_name": {
                            "type": "string",
                            "const": "se_e2_e"
                        },
                        "R_max": {
                            "type": "number"
                        },
                        "R_min": {
                            "type": "number"
                        },
                        "smooth_fun": {
                            "type": "string",
                            "default": "poly2_r",
                            "enum": ["cos", "cos_r", "tanh_u", "exp", "poly1_r",  "poly2_r","poly3_r"]
                        },
                        "network_size": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "bias": {
                            "type": "boolean",
                            "default": True
                        },
                        "resnet_dt": {
                            "type": "boolean",
                            "default": False
                        },
                        "activate_function": {
                            "type": "string",
                            "default": "tanh",
                            "enum": ["tanh", "relu", "sigmoid", "softplus", "softsign"]
                        },
                        "M2": {
                            "type": "integer",
                            "default": 16
                        },
                        "A": {
                            "type": ["null", "string"]
                        }
          },
          "required": ["type_name", "R_max", "R_min", "smooth_fun", "network_size"]
}

BP_param_schema = {
                   "SymFunRad": {
                                   "type": "object",
                                   "properties": {
                                       "type_name": {
                                                   "type": "string",
                                                   "const": "SymFunRad"
                                       },
                                       "R_max": {
                                                   "type": "number"
                                       },
                                       "R_min": {
                                                   "type": "number"
                                       },
                                       "num_basis": {
                                                   "type": "integer"
                                       },
                                       "smooth_fun": {
                                                   "type": "string",
                                                   "default": "poly2_r",
                                                   "enum": ["cos", "cos_r", "tanh_u", "exp",
                                                            "poly1_r", "poly2_r", "poly3_r"]
                                       },
                                       "eta": {
                                           "oneOf": [
                                               {
                                                   "type": "number"
                                               },
                                               {
                                                   "type": "array",
                                                   "items": {
                                                       "type": "number"
                                                   }
                                               }
                                           ]
                                       },
                                       "rs": {
                                           "oneOf": [
                                               {
                                                   "type": "number"
                                               },
                                               {
                                                   "type": "array",
                                                   "items": {
                                                       "type": "number"
                                                   }
                                               }
                                           ]
                                       },
                                       "rs_fun": {
                                           "type": "string",
                                           "default": "linear",
                                           "enum": ["linear"]
                                       }
                                   },
                                   "required": ["R_max", "R_min", "smooth_fun"],
                                   "oneOf": [
                                       {
                                           "properties": {
                                               "num_basis": {
                                                   "type": "integer"
                                               },
                                               "rs_fun": {
                                                   "type": "string",
                                                   "enum": ["linear"]
                                               }
                                           },
                                           "required": ["num_basis", "rs_fun"]
                                       },
                                       {
                                           "properties": {
                                               "eta": {
                                                   "type": "array",
                                                   "items": {
                                                       "type": "number"
                                                   }
                                               },
                                               "rs": {
                                                   "type": "array",
                                                   "items": {
                                                       "type": "number"
                                                   }
                                               }
                                           },
                                           "required": ["eta", "rs"]
                                       }
                                   ]

                   },
                   "SymFunAng": {
                           "type": "object",
                           "properties": {
                                       "type_name": {
                                            "type": "string"
                                       },
                                       "R_max": {
                                           "type": "number"
                                       },
                                       "R_min": {
                                           "type": "number"
                                       },
                                       "num_basis": {
                                           "type": "integer"
                                       },
                                       "smooth_fun": {
                                           "type": "string",
                                           "default": "poly2_r",
                                           "enum": ["cos", "cos_r", "tanh_u", "exp", "poly1_r", "poly2_r", "poly3_r"]
                                       },
                                       "eta": {
                                           "type": ["number", "array"],
                                           "items": {
                                                    "type": "number"
                                           }
                                       },
                                       "lambd": {
                                           "type": ["number", "array"],
                                           "items": {
                                                    "type": "number"
                                           }
                                       },
                                       "zeta": {
                                           "type": ["number", "array"],
                                           "items": {
                                                    "type": "number"
                                           }
                                       },
                                       "rs_fun": {
                                           "type": "string"
                                       }
                           },
                           "required": ["R_max", "R_min", "num_basis", "eta", "lambd", "zeta", "rs_fun"]
                   }
}


NEP_schema = {
          "type": "object",
          "properties": {
                        "R_max_R": {
                            "type": "number",
                            "default": 6.0
                        },
                        "R_max_A": {
                            "type": "number",
                            "default": 6.0
                        },
                        "n_max_R": {
                            "type": "integer",
                            "default": 4
                        },
                        "n_max_A": {
                            "type": "integer",
                            "default": 4
                        },
                        "k_basis_R": {
                            "type": "integer",
                            "default": 4
                        },
                        "k_basis_A": {
                            "type": "integer",
                            "default": 4
                        },
                        "l_3b_max": {
                            "type": "integer",
                            "default": 4,
                            "enum": [4]
                        },
                        "l_4b_max": {
                            "type": "integer",
                            "default": 2,
                            "enum": [0, 2]
                        },
                        "l_5b_max": {
                            "type": "integer",
                            "default": 0,
                            "enum": [0, 1]
                        },
                        "smooth_fun": {
                            "type": "string",
                            "default": "cos",
                            "enum": ["cos", "cos_r", "tanh_u", "exp", "poly1_r",  "poly2_r","poly3_r"]
                        },
          }
}


# Define the validator, including default values
def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for error in validate_properties(validator, properties, instance, schema):
            yield error

        for property_value, sub_schema in properties.items():
            if "default" in sub_schema:
                instance.setdefault(property_value, sub_schema["default"])

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )


# Creating Validators with Default Values
DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)


# Dynamically generated parameter classes
def create_dataclass(schema: dict):
    properties = schema.get('properties', {})
    fields = []
    type_dict = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list
    }
    for key, value in properties.items():
        type_class = type_dict.get("type", Union)
        default_value = value.get('default', None)
        fields.append((key, type_class, default_value))
    return make_dataclass(cls_name='SchemaClass', fields=fields)


# Generate parameter classes
GlobalSet = create_dataclass(global_set_schema)
DataSet = create_dataclass(data_set_schema)
TrainSet = create_dataclass(train_set_schema)
LossSet = create_dataclass(loss_set_schema)
LrSet = create_dataclass(learning_rate_schema)


class DeepSe:
    def __init__(self, fitting_config: dict, embedding_config: dict):
        validator = DefaultValidatingDraft7Validator(fitting_net_schema)
        validator.validate(fitting_config)

        validator = DefaultValidatingDraft7Validator(deep_se_e)
        validator.validate(embedding_config)

        self.fitting_config = fitting_config
        self.embedding_config = embedding_config


class BPDescriptor:
    def __init__(self, fitting_config: dict, feature_config: dict):
        self.fitting_config = fitting_config
        self.bp_features_information = []
        self.bp_features_param = []
        for i, feature_information in enumerate(feature_config['list']):
            feature_tensor = {}
            feature_str = {}
            if feature_information.get('type_name') == 'SymFunRad':
                validator = DefaultValidatingDraft7Validator(BP_param_schema['SymFunRad'])
                validator.validate(feature_information)
                feature_name = 'SymFunRad'
                feature_tensor['R_min'] = torch.Tensor([feature_information['R_min']])
                feature_tensor['R_max'] = torch.Tensor([feature_information['R_max']])
                if feature_information.get('rs_fun', None) is not None:
                    feature_tensor['rs'] = BPDescriptor.rs_fun(feature_information['R_min'],
                                                               feature_information['R_max'],
                                                               feature_information['num_basis'],
                                                               feature_information['rs_fun'])
                else:
                    feature_tensor['rs'] = torch.Tensor([feature_information['rs']])
                num_basis = feature_tensor['rs'].shape[0]
                if isinstance(feature_information['eta'], float):
                    feature_tensor['eta'] = (torch.Tensor([feature_information['eta']])).repeat(num_basis)
                else:
                    feature_tensor['eta'] = torch.Tensor([feature_information['eta']])
                feature_str['smooth_fun'] = feature_information['smooth_fun']
            elif 'SymFunAng' in feature_information.get('type_name'):
                validator = DefaultValidatingDraft7Validator(BP_param_schema['SymFunAng'])
                validator.validate(feature_information)
                feature_name = feature_information.get('type_name')
                feature_tensor['R_min'] = torch.Tensor([feature_information['R_min']])
                feature_tensor['R_max'] = torch.Tensor([feature_information['R_max']])
                if feature_information.get('rs_fun', None) is not None:
                    feature_tensor['rs'] = BPDescriptor.rs_fun(feature_information['R_min'],
                                                               feature_information['R_max'],
                                                               feature_information['num_basis'],
                                                               feature_information['rs_fun'])
                else:
                    feature_tensor['rs'] = torch.Tensor([feature_information['rs']])
                num_basis = feature_tensor['rs'].shape[0]
                if isinstance(feature_information['eta'], float):
                    feature_tensor['eta'] = (torch.Tensor([feature_information['eta']])).repeat(num_basis)
                    feature_tensor['lambd'] = (torch.Tensor([feature_information['lambd']])).repeat(num_basis)
                    feature_tensor['zeta'] = (torch.Tensor([feature_information['zeta']])).repeat(num_basis)
                else:
                    feature_tensor['eta'] = torch.Tensor([feature_information['eta']])
                    feature_tensor['lambd'] = torch.Tensor([feature_information['lambd']])
                    feature_tensor['zeta'] = torch.Tensor([feature_information['zeta']])
                feature_str['smooth_fun'] = feature_information['smooth_fun']
            else:
                raise ValueError('Wrong type of BP descriptor')
            self.bp_features_information.append(tuple([feature_name, num_basis]))
            self.bp_features_param.append(tuple([feature_tensor, feature_str]))

    @staticmethod
    def rs_fun(R_min: float, R_max: float, num_basis: int, type_name: str) -> torch.Tensor:
        if type_name == 'linear':
            rs = torch.linspace(R_min, R_max, num_basis)
        else:
            raise ValueError('Wrong type of rs_fun')
        return rs


class NEPParam:
    def __init__(self, fitting_config: dict, feature_config: dict):
        validator = DefaultValidatingDraft7Validator(fitting_net_schema)
        validator.validate(fitting_config)

        validator = DefaultValidatingDraft7Validator(NEP_schema)
        validator.validate(feature_config)

        self.fitting_config = fitting_config
        self.feature_config = feature_config


class Param:
    def __init__(self, input_dict: dict):
        self.input_dict_new = {}
        global_dict = input_dict['global_setting']
        validator = DefaultValidatingDraft7Validator(global_set_schema)
        validator.validate(global_dict)
        self.GlobalSet = GlobalSet(**global_dict)
        if isinstance(self.GlobalSet.neighbor, list):
            self.GlobalSet.neighbor = {t: n for t, n in zip(self.GlobalSet.type_map, self.GlobalSet.neighbor)}

        data_dict = input_dict['data_setting']
        validator = DefaultValidatingDraft7Validator(data_set_schema)
        validator.validate(data_dict)
        self.DataSet = DataSet(**data_dict)

        train_dict = input_dict['train_setting']
        validator = DefaultValidatingDraft7Validator(train_set_schema)
        validator.validate(train_dict)
        self.TrainSet = TrainSet(**train_dict)

        loss_dict = input_dict['loss_setting']
        validator = DefaultValidatingDraft7Validator(loss_set_schema)
        validator.validate(loss_dict)
        self.LossSet = LossSet(**loss_dict)

        learning_rate_dict = input_dict['learning_rate']
        validator = DefaultValidatingDraft7Validator(learning_rate_schema)
        validator.validate(learning_rate_dict)
        self.LrSet = LrSet(**learning_rate_dict)

        self.model_name = input_dict['model_setting']['name']
        if input_dict['model_setting']['name'] in ['dp_se_e2_a', 'dp_se_e2_r', 'dp_se_e3']:
            fitting_config = input_dict['model_setting']['fitting_net']
            embedding_config = input_dict['model_setting']['descriptor']
            self.DeepSe = DeepSe(fitting_config=fitting_config, embedding_config=embedding_config)
        elif input_dict['model_setting']['name'] == 'BPMlp':
            fitting_config = input_dict['model_setting']['fitting_net']
            feature_config = input_dict['model_setting']['descriptor']
            self.BPDescriptor = BPDescriptor(fitting_config=fitting_config, feature_config=feature_config)
        elif input_dict['model_setting']['name'] == "NEP":
            fitting_config = input_dict['model_setting']['fitting_net']
            feature_config = input_dict['model_setting']['descriptor']
            self.NEPParam = NEPParam(fitting_config=fitting_config, feature_config=feature_config)
        elif input_dict['model_setting']['name'] == "test":
            self.test = self.Test(input_dict["model_setting"])
        elif input_dict['model_setting']['name'] == "user_defined":
            self.user_defined = self.UserDefined(input_dict["model_setting"])

        else:
            raise Exception(f"Undefined keyword: , Please read the manual!")

    class Test:
        def __init__(self, input_dict):
            self.input_dict = input_dict

    class UserDefined:
        config: dict = {}
        model_path: str = ""
        field_name: str = ""
        normal_name: str = None

        def __init__(self, config: dict):
            if 'descriptor' in config.keys():
                self.config = config["descriptor"]
            else:
                raise Exception("Please write the custom parameters in the descriptor")

            if "model_path" in config.keys():
                self.model_path = config["model_path"]
            else:
                raise Exception("Undefined model path:{model_path} ?")

            self.field_name = config["field_name"]

            if "normal_name" in config.keys():
                self.normal_name = config["normal_name"]
            else:
                self.normal_name = None


