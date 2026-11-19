"""Configuration parsing, typed conversion, model argument expansion, and error-path tests."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from abfml.config import CommonConfig, RootConfig, load_config
from abfml.config import ConfigError
from abfml.builtin.deepse import DeepPotSE2AConfig, DeepPotSE2RConfig, DeepPotSE3Config
from abfml.builtin.lennard_jones import LennardJonesConfig
from abfml.core.registry import build_model, parse_options

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def deep_pot_se_model(variant: str, **descriptor_overrides) -> dict:
    descriptor = {
        "r_min": 0.5,
        "smooth": "poly2_r",
        "embedding": {
            "network_size": [25, 50, 100],
            "activation": "tanh",
            "bias": True,
            "residual": False,
        },
    }
    descriptor.update(descriptor_overrides)
    return {
        "name": variant,
        "cutoff": 6.0,
        "type_slots": [100, 100],
        "descriptor": descriptor,
        "fitting": {
            "hidden_layers": [200, 200, 200],
            "activation": "tanh",
            "bias": True,
            "residual": True,
        },
    }


def load_overrides(**overrides) -> RootConfig:
    raw = json.loads((EXAMPLES / "deepse2a" / "input.json").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        raw[key] = value
    return RootConfig.from_dict(raw, base_dir=EXAMPLES / "deepse2a")


class ConfigLoadTest(unittest.TestCase):
    def test_load_split_example(self) -> None:
        config = load_config(EXAMPLES / "deepse2a" / "input.json")
        raw = json.loads((EXAMPLES / "deepse2a" / "input.json").read_text(encoding="utf-8"))
        self.assertEqual(config.config_version, 1)
        self.assertEqual(config.common.type_map, (29,))
        self.assertEqual(config.common.num_types, 1)
        self.assertEqual(config.model_name, "deep_pot_se_e2_a")
        self.assertIsInstance(config.model, DeepPotSE2AConfig)
        self.assertEqual(config.model.cutoff, raw["model"]["cutoff"])
        self.assertEqual(config.model.type_slots, tuple(raw["model"]["type_slots"]))
        self.assertEqual(config.model.max_neighbors, sum(raw["model"]["type_slots"]))
        self.assertEqual(config.model.m2, raw["model"]["descriptor"]["m2"])
        self.assertEqual(config.model.embedding_network_size,
                         tuple(raw["model"]["descriptor"]["embedding"]["network_size"]))
        self.assertEqual(config.data.split.train, 0.8)
        self.assertEqual(config.data.split.valid, 0.2)
        self.assertEqual(config.data.loader.batch_size, 2)
        self.assertEqual(config.training.optimizer.name, "adam")
        self.assertEqual(config.loss.metric, "mse")
        self.assertEqual(config.output.checkpoint.monitor, "valid_loss")

    def test_load_explicit_example(self) -> None:
        config = load_overrides(data={
            "train_files": ["../data/train.extxyz"], "valid_files": ["../data/valid.extxyz"], "loader": {"batch_size": 2}})
        self.assertIsNone(config.data.split)
        self.assertEqual(config.data.files, ())
        self.assertEqual(len(config.data.train_files), 1)
        self.assertEqual(len(config.data.valid_files), 1)
        self.assertTrue(config.data.train_files[0].endswith("train.extxyz"))


class DeepPotConfigTest(unittest.TestCase):
    def test_to_model_kwargs_and_round_trip(self) -> None:
        config = parse_options(deep_pot_se_model("deep_pot_se_e2_a", m2=16), type_map=(29, 8))
        kwargs = config.model_kwargs()
        self.assertEqual(kwargs["max_neighbors"], 200)
        self.assertEqual(kwargs["embedding_config"]["R_min"], 0.5)
        self.assertEqual(kwargs["embedding_config"]["R_max"], 6.0)
        self.assertEqual(kwargs["embedding_config"]["M2"], 16)
        self.assertEqual(kwargs["embedding_config"]["network_size"], [25, 50, 100])
        self.assertEqual(kwargs["fitting_config"]["network_size"], [200, 200, 200])

        model = build_model(
            "deep_pot_se_e2_a", **kwargs, energy_shift=[0.0, 0.0], std_mean=None
        )
        self.assertEqual(model.embedding_size, 100)
        self.assertEqual(model.m2, 16)
        self.assertEqual(model.total_neighbors, 200)

    def test_complete_embedding_network_for_all_variants(self) -> None:
        for variant in ("deep_pot_se_e2_a", "deep_pot_se_e2_r", "deep_pot_se_e3"):
            for widths in ([8, 16, 16], [16]):
                with self.subTest(variant=variant, widths=widths):
                    descriptor = {"embedding": {"network_size": widths}}
                    if variant == "deep_pot_se_e2_a":
                        descriptor["m2"] = widths[-1]
                    config = parse_options(
                        deep_pot_se_model(variant, **descriptor), type_map=(29, 8)
                    )
                    self.assertEqual(config.embedding_network_size, tuple(widths))
                    model = build_model(
                        variant, **config.model_kwargs(), energy_shift=[0.0, 0.0], std_mean=None
                    )
                    self.assertEqual(model.embedding_size, widths[-1])
                    for network in model.embedding_net:
                        self.assertEqual(network.network_size, [1, *widths])
                        self.assertEqual([layer.out_features for layer in network.linear], widths)

    def test_embedding_width_validation(self) -> None:
        for widths in ([], [8, 0], [8, -1], [8, 1.5], [True], None):
            with self.subTest(widths=widths):
                raw = deep_pot_se_model("deep_pot_se_e2_a", m2=1,
                                        embedding={"network_size": widths})
                with self.assertRaisesRegex(ConfigError, r"descriptor\.embedding\.network_size"):
                    parse_options(raw, type_map=(29, 8))
        raw = deep_pot_se_model("deep_pot_se_e2_a", m2=9,
                                embedding={"network_size": [32, 8]})
        with self.assertRaisesRegex(ConfigError, "must not exceed"):
            parse_options(raw, type_map=(29, 8))

    def test_split_embedding_fields_rejected(self) -> None:
        for field, value in (("hidden_layers", [8, 16]), ("output_dim", 16)):
            with self.subTest(field=field):
                raw = deep_pot_se_model("deep_pot_se_e2_a", m2=8,
                                        embedding={"network_size": [8, 16, 16], field: value})
                with self.assertRaisesRegex(ConfigError, "unknown field"):
                    parse_options(raw, type_map=(29, 8))

    def test_scalar_type_slots_expands(self) -> None:
        raw = deep_pot_se_model("deep_pot_se_e2_a", m2=16)
        raw["type_slots"] = 100
        config = parse_options(raw, type_map=(29, 8))
        self.assertEqual(config.type_slots, (100, 100))

    def test_variants_without_m2(self) -> None:
        for variant in ("deep_pot_se_e2_r", "deep_pot_se_e3"):
            config = parse_options(deep_pot_se_model(variant), type_map=(29, 8))
            self.assertNotIn("M2", config.embedding_options())
            expected = DeepPotSE2RConfig if variant == "deep_pot_se_e2_r" else DeepPotSE3Config
            self.assertIsInstance(config, expected)

    def test_m2_forbidden_on_radial_variant(self) -> None:
        with self.assertRaisesRegex(ConfigError, "not used by se_e2_r"):
            parse_options(deep_pot_se_model("deep_pot_se_e2_r", m2=16), type_map=(29, 8))


class OtherModelConfigTest(unittest.TestCase):
    def test_lennard_jones(self) -> None:
        config = parse_options(
            {"name": "lennard_jones", "cutoff": 3.0, "epsilon": 0.1, "sigma": 2.0},
            type_map=(29,),
        )
        self.assertIsInstance(config, LennardJonesConfig)
        self.assertEqual(config.cutoff, 3.0)
        model = build_model("lennard_jones", **config.model_kwargs())
        self.assertEqual(float(model.sigma.detach()), 2.0)

    def _nep_base(self) -> dict:
        return {
            "name": "nep",
            "cutoff_radial": 4.0,
            "cutoff_angular": 3.0,
            "n_max_radial": 2,
            "n_max_angular": 2,
            "basis_size_radial": 3,
            "basis_size_angular": 3,
            "l_max": [4, 1, 1, 1],
        }

    def test_nep_num_neurons_and_hidden_layers(self) -> None:
        single = parse_options({**self._nep_base(), "num_neurons": 5}, type_map=(29, 8))
        self.assertEqual(single.network_size, (5,))
        self.assertEqual(single.cutoff, 4.0)
        build_model("nep", **single.model_kwargs())

        multi = parse_options(
            {**self._nep_base(), "hidden_layers": [5, 4]}, type_map=(29, 8)
        )
        self.assertEqual(multi.network_size, (5, 4))
        model = build_model("nep", **multi.model_kwargs())
        self.assertEqual(len(model.fitting_network.layers), 2)

    def test_nep_requires_one_network_spec(self) -> None:
        base = {
            "name": "nep",
            "cutoff_radial": 4.0,
            "cutoff_angular": 3.0,
            "n_max_radial": 2,
            "n_max_angular": 2,
            "basis_size_radial": 3,
            "basis_size_angular": 3,
            "l_max": [2],
        }
        with self.assertRaisesRegex(ConfigError, "requires num_neurons"):
            parse_options(base, type_map=(29, 8))
        with self.assertRaisesRegex(ConfigError, "not both"):
            parse_options(
                {**base, "num_neurons": 5, "hidden_layers": [5]}, type_map=(29, 8)
            )


class ConfigErrorTest(unittest.TestCase):
    def test_activation_definition_is_shared(self) -> None:
        from abfml.config import parse_network
        from abfml.builtin.networks import ACTIVATIONS, ActivationModule

        # Extend only the implementation mapping; validation and construction should accept the new name.
        with patch.dict(ACTIVATIONS, {'test_identity': torch.nn.Identity}):
            _, name, _, _ = parse_network({'hidden_layers': [4], 'activation': 'test_identity'}, path='network')
            self.assertIsInstance(ActivationModule(name).activation, torch.nn.Identity)
        with self.assertRaisesRegex(ConfigError, 'network.activation'):
            parse_network({'hidden_layers': [4], 'activation': 'test_identity'}, path='network')

    def test_cutoff_definition_is_shared(self) -> None:
        from abfml.builtin.functions import SMOOTH_FUNCTIONS, cutoff_function

        with patch.dict(SMOOTH_FUNCTIONS, {'test_linear': (lambda x: 1.-x, False, False, False)}):
            raw = deep_pot_se_model('deep_pot_se_e2_a', m2=16, smooth='test_linear')
            options = parse_options(raw, type_map=(29, 8))
            self.assertEqual(options.smooth, 'test_linear')
            torch.testing.assert_close(cutoff_function('test_linear', torch.tensor([1., 2., 4.]), .5, 3.5),
                                       torch.tensor([5./6., .5, 0.]))

    def test_log_every_alias_is_rejected(self) -> None:
        raw = json.loads((EXAMPLES / "deepse2a" / "input.json").read_text(encoding="utf-8"))
        raw["training"]["log_every"] = 1
        with self.assertRaisesRegex(ConfigError, "unknown field"):
            RootConfig.from_dict(raw, base_dir=EXAMPLES / "deepse2a")

    def test_unknown_top_level_field(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown top-level field"):
            load_overrides(extra=1)

    def test_unknown_model_field(self) -> None:
        raw = deep_pot_se_model("deep_pot_se_e2_a", m2=16)
        raw["cutoffx"] = 6.0
        with self.assertRaisesRegex(ConfigError, "model: unknown field"):
            parse_options(raw, type_map=(29, 8))

    def test_unknown_descriptor_field(self) -> None:
        raw = deep_pot_se_model("deep_pot_se_e2_a", m2=16)
        raw["descriptor"]["smoothx"] = "cos"
        with self.assertRaisesRegex(ConfigError, "descriptor: unknown field"):
            parse_options(raw, type_map=(29, 8))

    def test_invalid_smooth_function(self) -> None:
        raw = deep_pot_se_model("deep_pot_se_e2_a", m2=16, smooth="gaussian")
        with self.assertRaisesRegex(ConfigError, "unsupported smooth"):
            parse_options(raw, type_map=(29, 8))

    def test_m2_exceeds_final_embedding_width(self) -> None:
        raw = deep_pot_se_model("deep_pot_se_e2_a", m2=200)
        with self.assertRaisesRegex(ConfigError, "must not exceed"):
            parse_options(raw, type_map=(29, 8))

    def test_r_min_not_below_cutoff(self) -> None:
        raw = deep_pot_se_model("deep_pot_se_e2_a", m2=16, r_min=7.0)
        with self.assertRaisesRegex(ConfigError, "0 <= r_min < cutoff"):
            parse_options(raw, type_map=(29, 8))

    def test_unknown_model_name(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown model"):
            parse_options({"name": "mace", "cutoff": 1.0}, type_map=(29,))

    def test_duplicate_elements(self) -> None:
        with self.assertRaisesRegex(ConfigError, "duplicate"):
            CommonConfig.from_dict({"elements": ["Cu", "Cu"]})

    def test_unknown_element_symbol(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown element"):
            CommonConfig.from_dict({"elements": ["Xx"]})

    def test_split_must_sum_to_one(self) -> None:
        from abfml.config import SplitConfig

        with self.assertRaisesRegex(ConfigError, "sum to 1"):
            SplitConfig.from_dict({"train": 0.8, "valid": 0.3}, path="data.split")


if __name__ == "__main__":
    unittest.main()
