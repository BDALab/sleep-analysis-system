import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

from dashboard.logic.state_activity_features import (
    METRICS, STATES, VISIT_FEATURES, StateActivityConfig, aggregate_visits,
)
from dashboard.logic.state_activity_rbdq import (
    MODEL_VARIANTS, fit_models, paired_night_aggregates, planned_fdr,
    prepare_analysis, rank_correlations, validate_exported_aggregates,
)


def example_nights():
    rows = []
    for code, diagnosis in (("HC-1", 0), ("HC2-1", 0), ("pre-LBD-1", 3)):
        for day in range(6):
            row = {
                "#Subject": code, "#Date": f"2026-01-{day + 1:02}",
                "#DiagnosisCode": diagnosis, "#Age": 70, "#Gender": 1,
                "#Education": 12., "RBDq": 3., "status": "ok",
                "sleep_status": "ok", "wake_status": "ok" if day < 5 else "insufficient_minutes",
            }
            for state in STATES:
                for metric in METRICS:
                    row[f"{state}_{metric}"] = float(day if day < 5 else 100) if state == "sleep" or day < 5 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


class StateActivityRbdqPreparationTest(unittest.TestCase):
    def setUp(self):
        self.config = StateActivityConfig()
        self.nightly = example_nights()
        self.visits = aggregate_visits(self.nightly)
        self.core = self.visits[["#Subject", "#Age", "#Gender", "#Education"]].copy()
        # This legacy field is not an authoritative diagnosis and must be ignored.
        self.core["#Disease"] = "NonHC"
        self.diagnoses = dict(zip(self.visits["#Subject"], self.visits["#DiagnosisCode"]))

    def test_paired_nights_change_both_summaries_on_the_same_days(self):
        result = paired_night_aggregates(self.nightly, self.config)
        self.assertEqual(result.paired_valid_nights.tolist(), [5, 5, 5])
        self.assertEqual(result["Median.activity_sleep.mean"].tolist(), [2, 2, 2])
        self.assertEqual(result["IQR.activity_sleep.mean"].tolist(), [2, 2, 2])
        self.assertEqual(self.visits["Median.activity_sleep.mean"].tolist(), [2.5] * 3)

    def test_preparation_uses_numeric_diagnoses_and_clusters_repeat_visits(self):
        primary, sensitivity, audit = prepare_analysis(self.nightly, self.visits, self.core, self.visits["#Subject"],
                                                       self.diagnoses, self.config)
        self.assertEqual(len(primary), 3)
        self.assertEqual(primary["Person ID"].nunique(), 2)
        self.assertEqual(set(primary.Diagnosis), {"HC", "MCI-LB"})
        self.assertEqual(primary["Median.activity_sleep.mean"].tolist(), [2] * 3)
        self.assertEqual(sensitivity["Median.activity_sleep.mean"].tolist(), [2.5] * 3)
        self.assertTrue(audit.common_model_eligible.all())
        self.assertEqual(audit.original_sleep_valid_nights.tolist(), [6, 6, 6])
        self.assertEqual(audit.original_wake_valid_nights.tolist(), [5, 5, 5])

    def test_missing_covariate_removes_same_visit_from_every_model(self):
        nightly = self.nightly.copy()
        nightly.loc[nightly["#Subject"] == "HC2-1", "#Education"] = np.nan
        visits = aggregate_visits(nightly)
        core = visits[["#Subject", "#Age", "#Gender", "#Education"]]
        primary, sensitivity, audit = prepare_analysis(nightly, visits, core, visits["#Subject"], self.diagnoses,
                                                       self.config)
        self.assertEqual(set(primary["#Subject"]), {"HC-1", "pre-LBD-1"})
        self.assertEqual(primary["#Subject"].tolist(), sensitivity["#Subject"].tolist())
        self.assertIn("missing #Education", audit.loc[audit["#Subject"] == "HC2-1", "exclusion_reason"].iloc[0])

    def test_changed_csv_aggregates_cannot_silently_enter_models(self):
        visits = self.visits.copy()
        visits.loc[0, VISIT_FEATURES[0]] += 1
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            validate_exported_aggregates(self.nightly, visits, self.config)

    def test_duplicate_visit_and_diagnosis_drift_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_exported_aggregates(self.nightly, pd.concat([self.visits, self.visits.iloc[:1]]), self.config)
        diagnoses = {**self.diagnoses, "HC-1": 3}
        with self.assertRaisesRegex(ValueError, "Diagnosis codes changed"):
            prepare_analysis(self.nightly, self.visits, self.core, self.visits["#Subject"], diagnoses, self.config)

    def test_frozen_demographics_and_cohort_are_checked(self):
        core = self.core.copy()
        core.loc[0, "#Age"] += 1
        with self.assertRaisesRegex(ValueError, "Demographics differ"):
            prepare_analysis(self.nightly, self.visits, core, self.visits["#Subject"], self.diagnoses, self.config)
        with self.assertRaisesRegex(ValueError, "eligibility differs"):
            prepare_analysis(self.nightly, self.visits, self.core, ["HC-1"], self.diagnoses, self.config)

    def test_five_independent_state_nights_need_not_make_five_paired_nights(self):
        nightly = self.nightly.copy()
        nightly.loc[nightly["#Date"] == "2026-01-01", "sleep_status"] = "insufficient_minutes"
        result = paired_night_aggregates(nightly, self.config)
        self.assertEqual(result.paired_valid_nights.tolist(), [4] * 3)
        self.assertTrue(result[list(VISIT_FEATURES)].isna().all().all())


class StateActivityRbdqStatisticsTest(unittest.TestCase):
    def synthetic_data(self):
        rng = np.random.default_rng(20261026)
        n = 80
        data = pd.DataFrame({feature: rng.normal(size=n) for feature in VISIT_FEATURES})
        data["Diagnosis"] = np.where(np.arange(n) % 4 < 2, "HC", "MCI-LB")
        data["diagnosis_code"] = np.where(data.Diagnosis == "HC", 0, 3)
        data["Person ID"] = [f"P{i // 2}" for i in range(n)]
        data["#Gender"] = rng.integers(1, 3, n)
        data["#Education"] = rng.normal(14, 2, n)
        data["#Age"] = rng.normal(70, 5, n)
        data["Clinical collection"] = np.where(np.arange(n) % 3 == 0, "NINR", "NU20")
        data["rbdq"] = 2 + 1.2 * data[VISIT_FEATURES[0]] + .4 * (data.Diagnosis == "MCI-LB") + .2 * data[
            "#Education"] + rng.normal(0, .3, n)
        return data

    def test_fdr_keeps_all_sixteen_planned_tests_including_failures(self):
        p = pd.Series([.01, *([np.nan] * 15)])
        q = planned_fdr(p)
        self.assertAlmostEqual(q.iloc[0], .16)
        self.assertTrue(q.iloc[1:].isna().all())
        with self.assertRaisesRegex(ValueError, "all 16"):
            planned_fdr(pd.Series([.01]))

    def test_models_match_cluster_robust_ols_and_keep_person_ids(self):
        data = self.synthetic_data()
        with patch("dashboard.logic.state_activity_rbdq.MODEL_VARIANTS", MODEL_VARIANTS[:1]), patch(
                "dashboard.logic.state_activity_rbdq.WITHIN_COLLECTION_VARIANTS", ()):
            models, slopes = fit_models(data, data)
        primary = models.loc[models.Dataset == "paired-nights"]
        self.assertEqual(len(primary), 16)
        self.assertTrue((primary["n"] == 80).all())
        self.assertTrue((primary["Person clusters"] == 40).all())
        self.assertTrue((primary["Main model status"] == "ok").all())
        self.assertEqual(len(slopes), 64)
        x = data[VISIT_FEATURES[0]].to_numpy()
        x = (x - x.mean()) / x.std(ddof=0)
        design = pd.DataFrame({"feature": x, "diagnosis": (data.Diagnosis == "MCI-LB").astype(float),
                               "sex": data["#Gender"] - data["#Gender"].min(),
                               "education": (data["#Education"] - data["#Education"].mean()) / data["#Education"].std(
                                   ddof=0)})
        ols = OLS(data.rbdq, add_constant(design)).fit(cov_type="cluster",
                                                       cov_kwds={"groups": data["Person ID"], "use_correction": False})
        result = primary.loc[primary["Representative feature"] == VISIT_FEATURES[0]].iloc[0]
        self.assertAlmostEqual(result["Feature beta per SD"], ols.params["feature"], places=10)
        self.assertAlmostEqual(result["Feature robust SE"], ols.bse["feature"], places=10)

    def test_rank_associations_use_same_cohort_and_sixteen_per_scope(self):
        data = self.synthetic_data()
        correlations = rank_correlations(data)
        self.assertEqual(len(correlations), 48)
        self.assertEqual(correlations.groupby("Diagnosis").size().to_dict(), {"HC": 16, "MCI-LB": 16, "Pooled": 16})
        row = correlations.loc[(correlations.Diagnosis == "Pooled") & (correlations.Feature == VISIT_FEATURES[0])].iloc[
            0]
        expected = data[[VISIT_FEATURES[0], "rbdq"]].corr(method="spearman").iloc[0, 1]
        self.assertAlmostEqual(row.rho, expected, places=12)
        self.assertEqual(row.people, 40)


if __name__ == "__main__":
    unittest.main()
