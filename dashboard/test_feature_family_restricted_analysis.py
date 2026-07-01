import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard.logic.feature_family_restricted_analysis import (
    filter_grouped_stats_by_feature_families,
)


class FeatureFamilyRestrictedAnalysisTest(unittest.TestCase):
    def test_filter_grouped_stats_keeps_only_requested_families(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "grouped.xlsx"
            output_path = Path(tmp_dir) / "filtered.xlsx"
            source_df = pd.DataFrame(
                {
                    "#Subject": ["HC-001", "preDLB_001"],
                    "#Age": [70, 72],
                    "Mean.actigraphy.Awakening > 5 minutes": [1.0, 3.0],
                    "Median.actigraphy.Total sleep time": [420.0, 390.0],
                    "IQR.activity.Relative Interquartile Range": [0.12, 0.31],
                }
            )
            source_df.to_excel(source_path, index=False)

            result = filter_grouped_stats_by_feature_families(
                source_path=source_path,
                output_path=output_path,
                allowed_family_ids={
                    "long_awakenings",
                    "activity_variability",
                },
            )

            filtered_df = pd.read_excel(output_path)
            self.assertEqual(
                list(filtered_df.columns),
                [
                    "#Subject",
                    "#Age",
                    "Mean.actigraphy.Awakening > 5 minutes",
                    "IQR.activity.Relative Interquartile Range",
                ],
            )
            self.assertEqual(result["feature_count"], 2)
            self.assertTrue(Path(result["feature_manifest_path"]).exists())


if __name__ == "__main__":
    unittest.main()
