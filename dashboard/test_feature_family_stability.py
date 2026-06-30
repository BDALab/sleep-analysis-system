import unittest

from dashboard.logic.feature_family_stability import classify_stability


class FeatureFamilyStabilityTest(unittest.TestCase):
    def test_primary_sleep_family_requires_classifier_and_association_support(self):
        result = classify_stability(
            {
                "Feature family ID": "long_awakenings",
                "Feature family domain": "sleep",
                "Feature family role": "primary",
                "Classification run support": 2,
                "Association method support": 2,
                "Classification source count": 4,
                "Association source count": 4,
            }
        )
        self.assertEqual(result, "primary_cross_method_stable")

    def test_activity_variability_can_be_stable_in_activity_enhanced_data(self):
        result = classify_stability(
            {
                "Feature family ID": "activity_variability",
                "Feature family domain": "activity",
                "Feature family role": "primary_activity_enhanced",
                "Classification run support": 1,
                "Association method support": 2,
                "Classification source count": 2,
                "Association source count": 3,
            }
        )
        self.assertEqual(result, "activity_enhanced_cross_method_stable")

    def test_waso_gets_explicit_corrected_cross_method_class(self):
        result = classify_stability(
            {
                "Feature family ID": "waso",
                "Feature family domain": "sleep",
                "Feature family role": "primary_waso_corrected",
                "Classification run support": 1,
                "Association method support": 1,
                "Classification source count": 1,
                "Association source count": 1,
            }
        )
        self.assertEqual(result, "waso_corrected_cross_method")

    def test_lifestyle_families_remain_secondary(self):
        result = classify_stability(
            {
                "Feature family ID": "alcohol",
                "Feature family domain": "diary_lifestyle",
                "Feature family role": "secondary_confounding_sensitive",
                "Classification run support": 3,
                "Association method support": 2,
                "Classification source count": 5,
                "Association source count": 3,
            }
        )
        self.assertEqual(result, "secondary_lifestyle_signal")


if __name__ == "__main__":
    unittest.main()
