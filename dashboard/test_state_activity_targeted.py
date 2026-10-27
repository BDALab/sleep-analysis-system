import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

from dashboard.logic.state_activity_features import StateActivityConfig, align_activity_states, summarize_night
from dashboard.logic.state_activity_targeted_analysis import (
    MODEL_VARIANTS, clinical_snapshot, fit_target_models, planned_fdr,
    prepare_target_samples, rank_correlations,
)
from dashboard.logic.state_activity_targeted_features import (
    CONTEXTS, TARGETS, aggregate_target_nights, audit_legacy_nightly,
    extract_targeted_nights, target_distribution, target_feature, validate_source_fingerprints,
)


def nightly_fixture():
    rows = []
    for subject in ("HC-1", "HC2-1", "pre-LBD-1"):
        for day in range(1, 7):
            row = {"#Subject": subject, "#Date": f"2026-01-{day:02}", "status": "ok",
                   "sleep_status": "ok", "wake_status": "ok" if day < 6 else "insufficient_minutes"}
            for context in (*CONTEXTS, "legacy_whole"):
                row[f"{context}_mad"] = float(day if day < 6 else 100)
                row[f"{context}_relative_interdecile"] = day / 10
            rows.append(row)
    return pd.DataFrame(rows)


class TargetedActivityFeatureTest(unittest.TestCase):
    def test_exact_unscaled_mad_and_original_ratio(self):
        result = target_distribution([0, 1, 2, 100])
        self.assertEqual(result["mad"], 1)
        self.assertAlmostEqual(result["relative_interdecile"], .703)

    def test_ratio_is_scale_invariant_and_mad_is_not(self):
        a = target_distribution([1, 2, 3, 5])
        b = target_distribution([4, 8, 12, 20])
        self.assertAlmostEqual(b["mad"], 4 * a["mad"])
        self.assertAlmostEqual(a["relative_interdecile"], b["relative_interdecile"])

    def test_zero_denominator_is_missing_but_zero_mad_is_valid(self):
        result = target_distribution([0, 0, 0])
        self.assertEqual(result["mad"], 0)
        self.assertTrue(np.isnan(result["relative_interdecile"]))
        self.assertTrue(np.isnan(target_distribution([])["mad"]))

    def test_invalid_activity_cannot_silently_enter_a_metric(self):
        for values in ([1, -1], [1, np.nan], [1, np.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                target_distribution(values)

    def test_same_nights_used_for_all_contexts(self):
        result = aggregate_target_nights(nightly_fixture(), TARGETS[0], StateActivityConfig())
        self.assertEqual(result.target_valid_nights.tolist(), [5] * 3)
        for context in (*CONTEXTS, "legacy_whole"):
            self.assertEqual(result[target_feature(TARGETS[0], context)].tolist(), [2] * 3)

    def test_attention_uses_population_not_sample_sd(self):
        result = aggregate_target_nights(nightly_fixture(), TARGETS[1], StateActivityConfig())
        self.assertAlmostEqual(result[target_feature(TARGETS[1], "sleep")].iloc[0], np.sqrt(2) / 10)

    def test_target_night_eligibility_is_metric_specific(self):
        data = nightly_fixture()
        data["wake_status"] = "ok"
        data.loc[data["#Date"] == "2026-01-06", "wake_relative_interdecile"] = np.nan
        mad = aggregate_target_nights(data, TARGETS[0], StateActivityConfig())
        ratio = aggregate_target_nights(data, TARGETS[1], StateActivityConfig())
        self.assertEqual(mad.target_valid_nights.tolist(), [6] * 3)
        self.assertEqual(ratio.target_valid_nights.tolist(), [5] * 3)

    def test_four_paired_nights_are_not_enough_and_duplicates_fail(self):
        data = nightly_fixture()
        data.loc[data["#Date"] == "2026-01-01", "sleep_status"] = "insufficient_minutes"
        result = aggregate_target_nights(data, TARGETS[0], StateActivityConfig())
        self.assertTrue(result[[target_feature(TARGETS[0], c) for c in CONTEXTS]].isna().all().all())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_target_nights(pd.concat([data, data.iloc[:1]]), TARGETS[0], StateActivityConfig())

    def test_cache_hash_and_size_are_bound_to_original_extraction(self):
        a = [{"path": "/example", "sha256": "abc", "size_bytes": 12}]
        validate_source_fingerprints(a, a)
        for changed in ([{"path": "/example", "sha256": "def", "size_bytes": 12}],
                        [{"path": "/example", "sha256": "abc", "size_bytes": 13}]):
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_source_fingerprints(a, changed)

    def test_legacy_nightly_audit_finds_mismatches(self):
        actual = pd.DataFrame({"#Subject": ["HC-1"], "#Date": ["2026-01-01"], "legacy_whole_mad": [2.],
                               "legacy_whole_relative_interdecile": [.3]})
        frozen = pd.DataFrame(
            {"#Subject": ["HC-1"], "#Date": [pd.Timestamp("2026-01-01")], TARGETS[0]["canonical_nightly"]: [2.],
             TARGETS[1]["canonical_nightly"]: [.4]})
        self.assertEqual(audit_legacy_nightly(actual, frozen).status.tolist(), ["pass", "mismatch"])

    def test_extraction_distinguishes_legacy_endpoints_and_reproduces_support(self):
        activity = pd.Series([1., 2., 3.], index=pd.date_range("2026-01-01", periods=3, freq="60s", tz="UTC"))
        prediction = pd.Series([1., 1., 1., 1., 0., 0.],
                               index=pd.date_range("2026-01-01 00:00:15", periods=6, freq="30s", tz="UTC"))
        config = StateActivityConfig(min_state_minutes=1)
        aligned = align_activity_states(activity, prediction, "2026-01-01", "2026-01-01 00:03:00", config)
        row = {"#Subject": "HC-1", "#Date": "2026-01-01", "#DiagnosisCode": 0, "night_id": 7, "recording_id": 11,
               "interval_start_utc": "2026-01-01T00:00:00+00:00", "interval_end_utc": "2026-01-01T00:03:00+00:00",
               "status": "ok", **summarize_night(aligned, config)}
        night = SimpleNamespace(pk=7, data_id=11, subject_id=1, subject=SimpleNamespace(code="HC-1"),
                                data=SimpleNamespace(subject_id=1),
                                diary_day=SimpleNamespace(subject_id=1, date=date(2026, 1, 1)),
                                sleep_onset=pd.Timestamp("2026-01-01 00:00:15", tz="UTC"),
                                sleep_end=pd.Timestamp("2026-01-01 00:02:45", tz="UTC"))
        query = MagicMock()
        query.select_related.return_value.order_by.return_value = [night]
        manifest = {"sources": [{"recording_id": 11, "files": []}]}
        with patch("dashboard.models.SleepNight.objects.filter", return_value=query), patch(
                "dashboard.logic.state_activity_targeted_features._load_signals",
                return_value=(activity, prediction, [])):
            result, _ = extract_targeted_nights(pd.DataFrame([row]), manifest, config)
        self.assertEqual(result.whole_mad.iloc[0], 1)
        self.assertEqual(result.legacy_whole_mad.iloc[0], .5)
        self.assertEqual(result.sleep_mad.iloc[0], .5)
        self.assertEqual(result.wake_mad.iloc[0], 0)


class TargetedActivityAnalysisTest(unittest.TestCase):
    def snapshot(self):
        return clinical_snapshot([
            {"code": code, "diagnosis_code": diagnosis, "age": 70, "sex": "F", "education_years": 12., "updrs": updrs,
             "attention": 1.}
            for code, diagnosis, updrs in (("HC-1", 0, 3.), ("HC2-1", 0, np.nan), ("pre-LBD-1", 3, 5.))
        ])

    def test_each_outcome_has_its_own_sample_without_rbdq_requirement(self):
        snapshot = self.snapshot()
        self.assertNotIn("rbdq", snapshot)
        samples, audit = prepare_target_samples(nightly_fixture(), snapshot, set(snapshot["#Subject"]),
                                                StateActivityConfig())
        self.assertEqual(len(samples["UPDRS"]), 2)
        self.assertEqual(len(samples["Attention"]), 3)
        self.assertEqual(samples["Attention"]["Person ID"].nunique(), 2)
        self.assertIn("missing updrs", audit.loc[
            (audit["Clinical outcome"] == "UPDRS") & (audit["#Subject"] == "HC2-1"), "exclusion_reason"].iloc[0])

    def test_diagnosis_is_not_inferred_from_subject_prefix(self):
        data = clinical_snapshot(
            [{"code": "HC-1", "diagnosis_code": 3, "age": 70, "sex": "M", "education_years": 12, "updrs": 3.,
              "attention": 1.}])
        self.assertEqual(data.Diagnosis.iloc[0], "MCI-LB")

    def test_planned_fdr_includes_unestimable_tests(self):
        result = planned_fdr(pd.Series([.01, np.nan, np.nan]))
        self.assertAlmostEqual(result.iloc[0], .03)
        self.assertTrue(result.iloc[1:].isna().all())
        with self.assertRaises(ValueError):
            planned_fdr(pd.Series([.01]))

    def synthetic_samples(self):
        rng = np.random.default_rng(20261027)
        samples = {}
        for n, target in zip((80, 76), TARGETS):
            data = pd.DataFrame({target_feature(target, c): rng.normal(size=n) for c in (*CONTEXTS, "legacy_whole")})
            data["Diagnosis"] = np.where(np.arange(n) % 4 < 2, "HC", "MCI-LB")
            data["Person ID"] = [f"P{i // 2}" for i in range(n)]
            data["#Gender"] = rng.integers(1, 3, n)
            data["#Education"] = rng.normal(14, 2, n)
            data["#Age"] = rng.normal(70, 5, n)
            data["Clinical collection"] = "NU20"
            data[target["field"]] = 2 + 1.2 * data[target_feature(target, "sleep")] + .3 * data["#Gender"] + rng.normal(
                0, .3, n)
            samples[target["outcome"]] = data
        return samples

    def test_common_samples_and_clustered_ols_reproduction(self):
        samples = self.synthetic_samples()
        with patch("dashboard.logic.state_activity_targeted_analysis.MODEL_VARIANTS", MODEL_VARIANTS[:1]), patch(
                "dashboard.logic.state_activity_targeted_analysis.WITHIN_COLLECTION_VARIANTS", ()):
            models, slopes = fit_target_models(samples)
        self.assertEqual(len(models), 12)
        self.assertEqual(len(slopes), 24)
        self.assertTrue((models["Main model status"] == "ok").all())
        for target in TARGETS:
            data = samples[target["outcome"]]
            selected = models.loc[
                (models["Clinical outcome"] == target["outcome"]) & (models["Window definition"] == "aligned")]
            self.assertEqual(selected.n.tolist(), [len(data)] * 3)
            self.assertEqual(selected["Person clusters"].tolist(), [len(data) // 2] * 3)
            z = lambda v: (v - v.mean()) / v.std(ddof=0)
            design = pd.DataFrame({"feature": z(data[target_feature(target, "sleep")]),
                                   "diagnosis": (data.Diagnosis == "MCI-LB").astype(float),
                                   "sex": data["#Gender"] - data["#Gender"].min(), "education": z(data["#Education"])})
            ols = OLS(data[target["field"]], add_constant(design)).fit(cov_type="cluster",
                                                                       cov_kwds={"groups": data["Person ID"],
                                                                                 "use_correction": False})
            actual = selected.loc[selected.Context == "sleep"].iloc[0]
            self.assertAlmostEqual(actual["Feature beta per SD"], ols.params["feature"], places=10)
            self.assertAlmostEqual(actual["Feature robust SE"], ols.bse["feature"], places=10)

    def test_rank_coefficients_match_spearman_on_each_common_sample(self):
        samples = self.synthetic_samples()
        ranks = rank_correlations(samples)
        self.assertEqual(len(ranks), 18)
        for target in TARGETS:
            data = samples[target["outcome"]]
            feature = target_feature(target, "sleep")
            expected = data[[feature, target["field"]]].corr(method="spearman").iloc[0, 1]
            actual = ranks.loc[(ranks["Clinical outcome"] == target["outcome"]) & (ranks.Diagnosis == "Pooled") & (
                        ranks.Context == "sleep")].iloc[0]
            self.assertAlmostEqual(actual.rho, expected, places=12)


if __name__ == "__main__":
    unittest.main()
