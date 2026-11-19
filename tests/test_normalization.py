"""Model normalization, statistics accumulation, and shape validation."""

import unittest

import torch

from abfml.builtin.acsf import ACSFFeatureStatistics
from abfml.builtin.deepse import DeepPotSECoordinateStatistics
from abfml.builtin.acsf import ACSFFeatureNormalizer


class ACSFFeatureNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float64,
        )
        self.center_types = torch.tensor([0, 1, 0], dtype=torch.long)

    def test_initialization_is_identity(self) -> None:
        normalizer = ACSFFeatureNormalizer(2, 3).to(torch.float64)
        actual = normalizer(self.features, self.center_types)
        torch.testing.assert_close(actual, self.features)

    def test_scalar_statistics_are_rejected(self) -> None:
        std = torch.tensor([2.0, 4.0], dtype=torch.float64)
        avg = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "shape"):
            ACSFFeatureNormalizer(2, 3).set_statistics(std, avg)

    def test_per_feature_and_zero_std_protection(self) -> None:
        std = torch.tensor(
            [[2.0, 0.0, 4.0], [1.0, 2.0, 5.0]],
            dtype=torch.float64,
        )
        avg = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        normalizer = ACSFFeatureNormalizer(2, 3).double()
        normalizer.set_statistics(std, avg)
        actual = normalizer(self.features, self.center_types)
        scale = torch.where(std > 1.0e-8, std, torch.ones_like(std))
        expected = (self.features - avg[self.center_types]) / scale[self.center_types]
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())

    def test_invalid_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            ACSFFeatureNormalizer(2, 3).set_statistics(torch.ones(2, 2), torch.zeros(2, 2))


class ACSFFeatureStatisticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 10.0],
                [2.0, 4.0, 8.0],
                [3.0, 6.0, 9.0],
            ],
            dtype=torch.float64,
        )
        self.center_types = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)

    def test_multiple_updates_match_column_statistics(self) -> None:
        statistics = ACSFFeatureStatistics(2, 3)
        statistics.update(self.features[:2], self.center_types[:2])
        statistics.update(self.features[2:], self.center_types[2:])
        std, avg = statistics.finalize()
        for species in range(2):
            selected = self.features[self.center_types == species]
            expected_std, expected_avg = torch.std_mean(selected, dim=0)
            torch.testing.assert_close(std[species], expected_std)
            torch.testing.assert_close(avg[species], expected_avg)

    def test_per_feature_matches_column_std_mean(self) -> None:
        statistics = ACSFFeatureStatistics(2, 3)
        statistics.update(self.features, self.center_types)
        std, avg = statistics.finalize()
        for species in range(2):
            selected = self.features[self.center_types == species]
            expected_std, expected_avg = torch.std_mean(selected, dim=0)
            torch.testing.assert_close(std[species], expected_std)
            torch.testing.assert_close(avg[species], expected_avg)

    def test_empty_update_and_insufficient_species(self) -> None:
        statistics = ACSFFeatureStatistics(2, 3)
        statistics.update(self.features[:0], self.center_types[:0])
        statistics.update(self.features, self.center_types)
        with self.assertRaisesRegex(ValueError, "two samples"):
            ACSFFeatureStatistics(2, 3).finalize()


class DeepPotSECoordinateStatisticsTest(unittest.TestCase):
    def test_per_structure_statistics_include_zero_padding(self) -> None:
        first = torch.tensor(
            [
                [[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                [[3.0, 1.0, -1.0, 0.0], [1.0, 0.0, 2.0, 0.0]],
            ],
            dtype=torch.float64,
        )
        second = torch.tensor(
            [
                [[2.0, 1.0, 1.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=torch.float64,
        )
        center_types = torch.zeros(2, dtype=torch.long)
        statistics = DeepPotSECoordinateStatistics(1, 4)
        statistics.update(first, center_types)
        statistics.update(second, center_types)
        std, avg = statistics.finalize()

        radial = [torch.std_mean(value[..., 0]) for value in (first, second)]
        xyz_std = [torch.std_mean(value[..., 1:])[0] for value in (first, second)]
        expected_std = torch.tensor(
            [[(radial[0][0] + radial[1][0]) / 2.0] + [sum(xyz_std) / 2.0] * 3]
        )
        expected_avg = torch.tensor(
            [[(radial[0][1] + radial[1][1]) / 2.0, 0.0, 0.0, 0.0]]
        )
        torch.testing.assert_close(std, expected_std)
        torch.testing.assert_close(avg, expected_avg)


if __name__ == "__main__":
    unittest.main()
