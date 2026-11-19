"""Model registration, shared construction, and error-handling tests."""

import unittest

import torch

from abfml.core import AtomicModel
from abfml.core.registry import MODEL_REGISTRY, ModelRegistry, ModelSpec, build_model, model_spec
from abfml.builtin.acsf import ACSFModel, ACSFConfig
from abfml.builtin.deepse import DeepPotSE2AConfig, DeepPotSE2RConfig, DeepPotSE3Config, DeepPotSE2A, DeepPotSE2R, DeepPotSE3
from abfml.builtin.lennard_jones import LennardJones, LennardJonesConfig
from abfml.builtin.nep import NepConfig, NepModel


class ModelRegistryTest(unittest.TestCase):
    def test_builtin_registrations(self) -> None:
        expected = {
            "acsf": (ACSFModel, ACSFConfig),
            "deep_pot_se_e2_a": (DeepPotSE2A, DeepPotSE2AConfig),
            "deep_pot_se_e2_r": (DeepPotSE2R, DeepPotSE2RConfig),
            "deep_pot_se_e3": (DeepPotSE3, DeepPotSE3Config),
            "nep": (NepModel, NepConfig),
            "lennard_jones": (LennardJones, LennardJonesConfig),
        }
        print("\n[ModelRegistry] Builtin models:", MODEL_REGISTRY.names)
        self.assertEqual(set(MODEL_REGISTRY.names), set(expected))
        self.assertEqual(len(MODEL_REGISTRY), len(expected))
        for name, (builder, config) in expected.items():
            self.assertIn(name, MODEL_REGISTRY)
            self.assertIs(model_spec(name).builder, builder)
            self.assertIs(model_spec(name).config, config)

    def test_models_declare_required_statistics(self) -> None:
        expected = {
            "acsf": ("acsf_calibration",),
            "deep_pot_se_e2_a": ("energy_shift", "std_mean"),
            "deep_pot_se_e2_r": ("energy_shift", "std_mean"),
            "deep_pot_se_e3": ("energy_shift", "std_mean"),
            "nep": ("q_scaler", "energy_zero"),
            "lennard_jones": (),
        }
        self.assertEqual(AtomicModel.required_statistics, ())
        for name, statistics in expected.items():
            self.assertEqual(
                model_spec(name).builder.required_statistics,
                statistics,
            )

    def test_build_model_uses_explicit_constructor_arguments(self) -> None:
        model = build_model(
            "lennard_jones",
            type_map=[1],
            cutoff=3.0,
            epsilon=0.1,
            sigma=2.0,
        )
        self.assertIsInstance(model, AtomicModel)
        self.assertIsInstance(model, LennardJones)
        torch.testing.assert_close(model.epsilon, torch.tensor(0.1))
        torch.testing.assert_close(model.sigma, torch.tensor(2.0))

    def test_independent_registry_accepts_custom_builder(self) -> None:
        registry = ModelRegistry()
        registry.register("custom_lj", ModelSpec(LennardJones))
        model = registry.create("custom_lj", [1], 3.0, 0.2, 1.5)
        self.assertIsInstance(model, LennardJones)
        self.assertNotIn("custom_lj", MODEL_REGISTRY)

    def test_duplicate_and_unknown_names_are_rejected(self) -> None:
        registry = ModelRegistry()
        registry.register("model", ModelSpec(LennardJones))
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("model", ModelSpec(LennardJones))
        with self.assertRaisesRegex(KeyError, "Available models: model"):
            registry.get("missing")

    def test_invalid_name_and_builder_result_are_rejected(self) -> None:
        registry = ModelRegistry()
        with self.assertRaisesRegex(ValueError, "Model name"):
            registry.register(" bad name ", ModelSpec(LennardJones))
        registry.register("invalid", ModelSpec(lambda: object()))
        with self.assertRaisesRegex(TypeError, "expected AtomicModel"):
            registry.create("invalid")


if __name__ == "__main__":
    unittest.main()
