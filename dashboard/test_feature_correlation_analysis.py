import unittest

import numpy as np
import pandas as pd

from dashboard.logic.feature_correlation_analysis import (
    _cluster_feature_families,
    _feature_family,
)


class FeatureFamilyTest(unittest.TestCase):
    def test_project_feature_taxonomy(self):
        cases = {
            ("actigraphy", "Sleep onset latency"): "Sleep timing: onset latency",
            ("diary_norm", "Wake after sleep onset"): "Nocturnal wakefulness",
            ("actigraphy", "Awakening > 5 minutes"): "Awakenings",
            ("activity", "Median Absolute Deviation"): (
                "Activity variability and dispersion"
            ),
            ("activity", "Mean Excluding Outliers (30)"): (
                "Activity level: central tendency"
            ),
            ("activity", "95th Percentile"): (
                "Activity level: upper distribution"
            ),
            ("activity", "Teager Kaiser Energy Operator Max"): (
                "Activity complexity and energy"
            ),
        }
        for (source, measurement), expected in cases.items():
            with self.subTest(source=source, measurement=measurement):
                self.assertEqual(
                    _feature_family(source, measurement),
                    expected,
                )

    def test_redundant_variants_cluster_within_family(self):
        features = [
            "Mean.actigraphy.Sleep onset latency",
            "Median.actigraphy.Sleep onset latency",
            "Mean.diary.Sleep onset latency",
            "Median.activity.Median Absolute Deviation",
        ]
        base = np.arange(12, dtype=float)
        frame = pd.DataFrame(
            {
                features[0]: base,
                features[1]: base * 2,
                features[2]: -base,
                features[3]: [0, 3, 1, 8, 2, 5, 11, 4, 9, 6, 10, 7],
            }
        )

        result = _cluster_feature_families(frame, features)
        clusters = result.set_index("Features")["Redundancy cluster"]

        self.assertEqual(clusters[features[0]], clusters[features[1]])
        self.assertEqual(clusters[features[0]], clusters[features[2]])
        self.assertNotEqual(clusters[features[0]], clusters[features[3]])


if __name__ == "__main__":
    unittest.main()
