import unittest

import numpy as np
import pandas as pd

from dashboard.logic.feature_correlation_analysis import (
    _cluster_feature_families,
    _feature_family,
)
from dashboard.logic.feature_families import (
    feature_family_label_for_feature,
    parse_feature_name,
)


class FeatureFamilyTest(unittest.TestCase):
    def test_project_feature_taxonomy(self):
        cases = {
            ("actigraphy", "Sleep onset latency"): "Sleep onset latency",
            ("diary_norm", "Wake after sleep onset"): "Wake after sleep onset",
            ("actigraphy", "Awakening > 5 minutes"): "Long awakenings",
            ("diary", "Wake bouts"): "Wake-bout frequency",
            ("activity", "Median Absolute Deviation"): (
                "Activity variability/dispersion"
            ),
            ("activity", "Mean Excluding Outliers (30)"): (
                "Activity level/intensity"
            ),
            ("activity", "95th Percentile"): (
                "Activity level/intensity"
            ),
            ("activity", "Teager Kaiser Energy Operator Max"): (
                "Activity shape/complexity"
            ),
        }
        for (source, measurement), expected in cases.items():
            with self.subTest(source=source, measurement=measurement):
                self.assertEqual(
                    _feature_family(source, measurement),
                    expected,
                )

    def test_mapper_supports_classifier_and_lifestyle_names(self):
        cases = {
            "actigraphy.Awakening > 5 minutes (Median)": "Long awakenings",
            "diary_norm.Wake after sleep onset (MAD)": "Wake after sleep onset",
            "activity.Relative Interquartile Range (Mean)": (
                "Activity variability/dispersion"
            ),
            "activity.Teager Kaiser Energy Operator Max (Mean)": (
                "Activity shape/complexity"
            ),
            "rest_quality_mean": "Subjective sleep/rest quality",
            "alcohol_time_mean": "Alcohol exposure/timing",
            "caffeine_time_std": "Caffeine exposure/timing",
            "sleeping_pill_rate": "Sleeping-pill use",
            "day_sleep_count_mean": "Day sleep / naps",
        }
        for feature, expected in cases.items():
            with self.subTest(feature=feature):
                self.assertEqual(feature_family_label_for_feature(feature), expected)

    def test_parse_feature_name_supports_both_export_styles(self):
        correlation_feature = parse_feature_name(
            "Mean.actigraphy.Wake after sleep onset"
        )
        self.assertEqual(correlation_feature.aggregation, "Mean")
        self.assertEqual(correlation_feature.source, "actigraphy")
        self.assertEqual(
            correlation_feature.measurement,
            "Wake after sleep onset",
        )

        classifier_feature = parse_feature_name(
            "actigraphy.Wake after sleep onset (IQR)"
        )
        self.assertEqual(classifier_feature.aggregation, "IQR")
        self.assertEqual(classifier_feature.source, "actigraphy")
        self.assertEqual(
            classifier_feature.measurement,
            "Wake after sleep onset",
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
