import math
import unittest

import numpy as np
import pandas as pd

from dashboard.logic.classification_grouped_statistics import TARGET_COLUMN
from dashboard.logic.clinical_scale_regression import (
    HC_VS_PRE_DLB_CODES,
    _build_group_cv,
    _compute_regression_metrics,
    _dataset_overview,
    _filter_diagnosis_codes,
    _person_group_id,
)


class ClinicalScaleRegressionTest(unittest.TestCase):
    def test_metrics_include_error_rate_against_observed_scale_size(self):
        metrics = _compute_regression_metrics(
            y_true=np.array([0.0, 10.0]),
            y_pred=np.array([2.0, 7.0]),
            scale_size=10.0,
            baseline_pred=np.array([5.0, 5.0]),
        )

        self.assertAlmostEqual(metrics["mae"], 2.5)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(6.5))
        self.assertAlmostEqual(metrics["estimation_error_rate"], 0.25)
        self.assertAlmostEqual(metrics["estimation_error_rate_percent"], 25.0)
        self.assertAlmostEqual(metrics["baseline_mae"], 5.0)
        self.assertAlmostEqual(metrics["mae_improvement_vs_baseline"], 2.5)

    def test_person_group_id_collapses_repeated_visits(self):
        self.assertEqual(_person_group_id("pre-LBD-102"), "pre-LBD-102")
        self.assertEqual(_person_group_id("pre-LBD2-102"), "pre-LBD-102")
        self.assertEqual(_person_group_id("preDLB2_102"), "preDLB_102")
        self.assertEqual(_person_group_id("HC2-17"), "HC-17")
        self.assertEqual(_person_group_id("COBEN-123"), "COBEN-123")

    def test_group_cv_caps_splits_by_available_groups(self):
        cv = _build_group_cv(groups=np.array(["a", "a", "b", "c"]), max_splits=5)

        self.assertEqual(cv.n_splits, 3)

    def test_filter_diagnosis_codes_keeps_only_hc_and_predlb(self):
        df = pd.DataFrame(
            {
                "#Subject": ["HC-1", "pre-LBD-1", "MCI-1", "NonHC-1"],
                TARGET_COLUMN: [0, 3, 2, 1],
            }
        )

        filtered = _filter_diagnosis_codes(df, HC_VS_PRE_DLB_CODES)

        self.assertEqual(filtered["#Subject"].tolist(), ["HC-1", "pre-LBD-1"])
        self.assertEqual(filtered[TARGET_COLUMN].tolist(), [0, 3])

    def test_dataset_overview_reflects_focused_cohort_counts(self):
        df = pd.DataFrame(
            {
                "#Subject": ["HC-1", "HC-2", "pre-LBD-1"],
                TARGET_COLUMN: [0, 0, 3],
            }
        )

        overview = _dataset_overview(df)
        counts = dict(zip(overview["diagnosis_label"], overview["subject_count"]))

        self.assertEqual(counts["HC"], 2)
        self.assertEqual(counts["preDLB"], 1)
        self.assertEqual(counts["MCI-AD"], 0)
        self.assertEqual(counts["NonHC"], 0)


if __name__ == "__main__":
    unittest.main()
