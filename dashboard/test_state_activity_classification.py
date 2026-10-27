import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dashboard.logic import classification_person_grouped_validation as engine
from dashboard.logic import state_activity_classification as experiment
from dashboard.logic.state_activity_classification_data import (
    ADDED_FEATURES, ADJUSTMENT_COLUMNS, AUGMENTED, BASELINE,
)
from dashboard.logic.state_activity_classification_data import prepare_matched_features
from dashboard.logic.state_activity_features import METRICS, STATES, StateActivityConfig, aggregate_visits
from dashboard.logic.state_activity_targeted_features import QC_COLUMNS, TARGETS


def feature_fixture():
    rows, targets, raw = [], [], []
    for visit, (code, diagnosis) in enumerate((("HC-1", 0), ("HC2-1", 0), ("pre-LBD-1", 3))):
        for day in range(6):
            value = float(day if day < 5 else 1000)
            row = {"#Subject": code, "#Date": f"2026-01-{day + 1:02}", "#DiagnosisCode": diagnosis,
                   "#Age": 70., "#Gender": 1., "#Education": np.nan if visit == 0 else 12.,
                   "RBDq": np.nan, "status": "ok", "sleep_status": "ok",
                   "wake_status": "ok" if day < 5 else "insufficient_minutes",
                   "night_id": visit * 6 + day, "recording_id": visit}
            row.update({column: 10. for column in QC_COLUMNS if column not in row})
            row.update({f"{state}_{metric}": value for state in STATES for metric in METRICS})
            rows.append(row)
            target = {key: row[key] for key in
                      ("#Subject", "#Date", "#DiagnosisCode", "night_id", "recording_id", "status", *QC_COLUMNS)}
            target.update({f"{context}_{item['metric']}": value for item in TARGETS
                           for context in ("whole", "sleep", "wake", "legacy_whole")})
            targets.append(target)
            # Deliberately misleading legacy labels and clinical-score columns.
            raw.append({key: row[key] for key in ("#Subject", "#Date", "#Age", "#Gender", "#Education")}
                       | {"#Disease": "NonHC", "activity.Median Absolute Deviation": value,
                          "activity.Relative Interdencile Range": value, "actigraphy.WASO": value,
                          "updrs": 999., "RBDq": 999., "attention": 999.})
    nightly = pd.DataFrame(rows)
    visits = aggregate_visits(nightly)
    cohort = visits[["#Subject", "#Age", "#Gender", "#Education"]].copy()
    diagnoses = dict(zip(visits["#Subject"], visits["#DiagnosisCode"]))
    return nightly, visits, pd.DataFrame(targets), pd.DataFrame(raw), cohort, diagnoses, StateActivityConfig()


def synthetic_input():
    rng = np.random.default_rng(71)
    people = np.array([f"P-{index}" for index in range(16)])
    # Repeated visits of every participant, plus both diagnoses in each source.
    groups = np.repeat(people, 2)
    y = np.repeat(np.array([0, 1] * 8), 2)
    subjects = np.array([f"{person}-V{visit}" for person in people for visit in (1, 2)])
    return {"X": np.column_stack((rng.normal(size=(len(y), 4)), rng.integers(1, 3, len(y)),
                                  rng.normal(13, 2, len(y)))), "y": y, "groups": groups, "subjects": subjects,
            "source_cohorts": np.repeat(np.array(["HC/HC2"] * 8 + ["COBEN"] * 8), 2),
            "visit_indices": np.tile([1, 2], len(people)), "feature_columns": [f"feature_{i}" for i in range(4)],
            "adjustment_columns": list(ADJUSTMENT_COLUMNS)}


class StateActivityClassifierPreparationTests(unittest.TestCase):
    def test_same_nights_numeric_labels_no_clinical_eligibility(self):
        result = prepare_matched_features(*feature_fixture())
        table = result["table"]
        self.assertEqual(len(table), 3)
        self.assertEqual(table.person_group.nunique(), 2)
        self.assertEqual(len(result["matched_nights"]), 15)
        self.assertTrue(result["cohort_audit"].eligible.all())
        self.assertEqual(table["Median.activity.Median Absolute Deviation"].tolist(), [2.] * 3)
        self.assertEqual(table["Median.activity_sleep.mean"].tolist(), [2.] * 3)
        self.assertEqual(table["IQR.activity_target_sleep.mad"].tolist(), [2.] * 3)
        self.assertEqual(table.y_true.tolist(), [0, 0, 1])
        self.assertTrue(table.loc[table["#Subject"].eq("HC-1"), "#Education"].isna().all())
        for score in ("updrs", "RBDq", "attention", "#Disease"):
            self.assertNotIn(score, table)
        base, added = experiment.model_input(result, BASELINE), experiment.model_input(result, AUGMENTED)
        self.assertEqual(added["X"].shape[1] - base["X"].shape[1], 20)
        self.assertEqual(added["feature_columns"], base["feature_columns"] + list(ADDED_FEATURES))
        np.testing.assert_array_equal(base["subjects"], added["subjects"])
        np.testing.assert_allclose(base["X"][:, -2:], added["X"][:, -2:], equal_nan=True)

    def test_target_support_is_joint_for_every_feature_not_outcome_specific(self):
        fixture = list(feature_fixture())
        fixture[2].loc[(fixture[2]["#Subject"] == "HC-1") & (fixture[2]["#Date"] == "2026-01-01"),
        "sleep_relative_interdecile"] = np.nan
        result = prepare_matched_features(*fixture)
        self.assertNotIn("HC-1", set(result["table"]["#Subject"]))
        self.assertEqual(set(result["table"]["#Subject"]), {"HC2-1", "pre-LBD-1"})

    def test_shuffled_target_rows_are_joined_by_identity(self):
        fixture = list(feature_fixture())
        fixture[2] = fixture[2].sample(frac=1, random_state=10)
        pd.testing.assert_frame_equal(prepare_matched_features(*feature_fixture())["table"],
                                      prepare_matched_features(*fixture)["table"])

    def test_duplicate_misaligned_and_changed_inputs_fail(self):
        fixture = list(feature_fixture())
        fixture[2] = pd.concat([fixture[2], fixture[2].iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_matched_features(*fixture)
        fixture = list(feature_fixture())
        fixture[2].loc[0, "recording_id"] = 500
        with self.assertRaisesRegex(ValueError, "different night identities"):
            prepare_matched_features(*fixture)
        fixture = list(feature_fixture())
        fixture[2].loc[0, "sleep_minutes"] += 1
        with self.assertRaisesRegex(ValueError, "support differs"):
            prepare_matched_features(*fixture)
        fixture = list(feature_fixture())
        fixture[2].loc[0, "legacy_whole_mad"] += 1
        with self.assertRaisesRegex(ValueError, "legacy metrics"):
            prepare_matched_features(*fixture)

    def test_diagnosis_and_covariate_drift_fail(self):
        fixture = list(feature_fixture())
        fixture[5]["HC-1"] = 3
        with self.assertRaisesRegex(ValueError, "diagnoses changed"):
            prepare_matched_features(*fixture)
        fixture = list(feature_fixture())
        fixture[4].loc[0, "#Gender"] = 2
        with self.assertRaisesRegex(ValueError, "Demographic drift"):
            prepare_matched_features(*fixture)
        fixture = list(feature_fixture())
        fixture[3].loc[0, "#Gender"] = 2
        with self.assertRaisesRegex(ValueError, "nightly demographics"):
            prepare_matched_features(*fixture)


class StateActivityClassifierValidationTests(unittest.TestCase):
    def test_split_plan_groups_all_visits_in_outer_and_inner_folds(self):
        prepared = synthetic_input()
        plan = experiment.split_plan(prepared)
        for _, fold in plan.groupby(["outer_fold", "inner_fold"]):
            self.assertFalse(set(fold.loc[fold.partition == "train", "person_group"]) &
                             set(fold.loc[fold.partition == "test", "person_group"]))
        outer = plan.loc[(plan.inner_fold == 0) & plan.partition.eq("test")]
        self.assertEqual(len(outer), len(prepared["y"]))
        with self.assertRaisesRegex(ValueError, "one diagnosis"):
            experiment.checked_splits(np.array([0, 1]), np.array(["same", "same"]), 2)

    def test_bootstrap_matches_folds_and_includes_all_repeated_visits(self):
        prepared = synthetic_input()
        base = pd.DataFrame({"#Subject": prepared["subjects"], "person_group": prepared["groups"],
                             "source_cohort": prepared["source_cohorts"], "visit_index": prepared["visit_indices"],
                             "fold": 1, "y_true": prepared["y"], "y_pred_default": prepared["y"],
                             "pred_probability_positive": np.where(prepared["y"] == 1, .8, .2)})
        result = experiment.paired_feature_comparison(base, base.sample(frac=1, random_state=2), 100)
        self.assertEqual(result.people.unique().tolist(), [16])
        self.assertEqual(result.visits.unique().tolist(), [32])
        self.assertTrue(result.difference_augmented_minus_baseline.eq(0).all())
        self.assertTrue(result.difference_ci_low.eq(0).all())
        changed = base.copy()
        changed.loc[0, "fold"] = 2
        with self.assertRaisesRegex(ValueError, "identical visits"):
            experiment.paired_feature_comparison(base, changed, 100)
        # A truly better probability model must have the advertised sign.
        weak = base.assign(pred_probability_positive=np.where(base.y_true == 1, .2, .8),
                           y_pred_default=1 - base.y_true)
        result = experiment.paired_feature_comparison(weak, base, 100).set_index("metric")
        self.assertGreater(result.loc["AUC", "difference_ci_low"], 0)
        self.assertLess(result.loc["BRIER", "difference_ci_high"], 0)

    def test_small_real_nested_fits_for_both_model_pipelines(self):
        prepared = synthetic_input()
        with patch.object(engine, "PERSON_GROUPED_OUTER_CV_SPLITS", 2), patch.object(
                engine, "PERSON_GROUPED_INNER_CV_SPLITS", 2):
            plan = experiment.split_plan(prepared)
            for model in engine.MODEL_FAMILIES:
                settings = engine._search_settings_for_model(model, experiment._spec(model)["feature_selector_mode"])
                settings.update(n_iter=1, verbose=0)
                settings["param_distributions"] = ({"clf__n_estimators": [5], "clf__max_depth": [2],
                                                    "feature_selector__k": ["all"]}
                                                   if model == "xgboost" else
                                                   {"clf__C": [.1], "clf__l1_ratio": [.5]})
                with patch.object(engine, "_search_settings_for_model", return_value=settings):
                    predictions, folds, _ = experiment._fit_nested(prepared, experiment._spec(model), plan)
                self.assertEqual(len(predictions), 32)
                self.assertEqual(len(folds), 2)
                self.assertTrue(predictions.pred_probability_positive.between(0, 1).all())
                self.assertEqual(set(predictions.diagnosis_label), {"HC", "MCI-LB"})

    def test_prepare_only_never_trains_and_refuses_existing_directory(self):
        fixture = prepare_matched_features(*feature_fixture())
        # A larger synthetic cohort is needed for the precomputed nested split plan.
        prepared = synthetic_input()
        columns = prepared["feature_columns"]
        table = pd.DataFrame(prepared["X"][:, :4], columns=columns)
        for key, source in (("#Subject", "subjects"), ("person_group", "groups"), ("source_cohort", "source_cohorts"),
                            ("visit_index", "visit_indices"), ("y_true", "y")):
            table[key] = prepared[source]
        table["diagnosis_code"] = np.where(prepared["y"] == 1, 3, 0)
        table[list(ADJUSTMENT_COLUMNS)] = prepared["X"][:, -2:]
        fixture.update(table=table, baseline_features=columns)
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                experiment, "load_experiment_inputs",
                return_value=(fixture, [], StateActivityConfig().to_dict())), patch.object(
            engine, "_run_grouped_nested_cv", side_effect=AssertionError("Must not train")):
            directory = Path(temporary) / "new"
            result = experiment.run_state_activity_classification("unused", "unused", directory, prepare_only=True)
            self.assertEqual(result["status"], "prepared_only")
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["adjustment_columns"], list(ADJUSTMENT_COLUMNS))
            self.assertTrue((directory / "split_plan.csv").exists())
            with self.assertRaises(FileExistsError):
                experiment.run_state_activity_classification("unused", "unused", directory, prepare_only=True)

    def test_source_labels_are_safe_paths_and_infeasible_checks_are_explicit(self):
        prepared = synthetic_input()
        # Both source cohorts become diagnosis-pure: every sensitivity must skip.
        prepared["source_cohorts"] = np.where(prepared["y"] == 0, "HC/HC2", "pre-LBD/pre-LBD2")
        with tempfile.TemporaryDirectory() as directory, patch.object(
                engine, "_run_grouped_search", side_effect=AssertionError("No feasible model")):
            experiment._source_checks(prepared, experiment._spec("xgboost"), Path(directory))
            rows = pd.read_csv(Path(directory) / "source_validation_metrics.csv")
        self.assertEqual(len(rows), 4)
        self.assertTrue(rows.status.eq("skipped").all())

    def test_full_runner_on_synthetic_data_saves_every_comparison_and_source_check(self):
        prepared = synthetic_input()
        data = prepare_matched_features(*feature_fixture())
        columns = prepared["feature_columns"]
        table = pd.DataFrame(prepared["X"][:, :4], columns=columns)
        table = pd.concat([table, pd.DataFrame(np.random.default_rng(31).normal(size=(32, 20)),
                                               columns=ADDED_FEATURES)], axis=1)
        for key, source in (("#Subject", "subjects"), ("person_group", "groups"), ("source_cohort", "source_cohorts"),
                            ("visit_index", "visit_indices"), ("y_true", "y")):
            table[key] = prepared[source]
        table["diagnosis_code"] = np.where(prepared["y"] == 1, 3, 0)
        table[list(ADJUSTMENT_COLUMNS)] = prepared["X"][:, -2:]
        data.update(table=table, baseline_features=columns)
        original_settings = engine._search_settings_for_model

        def tiny_settings(model, selector):
            settings = original_settings(model, selector)
            settings.update(n_iter=1, verbose=0)
            settings["param_distributions"] = ({"clf__n_estimators": [3], "clf__max_depth": [2],
                                                "feature_selector__k": ["all"]} if model == "xgboost" else
                                               {"clf__C": [.1], "clf__l1_ratio": [.5]})
            return settings

        with tempfile.TemporaryDirectory() as temporary, patch.object(
                experiment, "load_experiment_inputs",
                return_value=(data, [], StateActivityConfig().to_dict())), patch.object(
            engine, "PERSON_GROUPED_OUTER_CV_SPLITS", 2), patch.object(
            engine, "PERSON_GROUPED_INNER_CV_SPLITS", 2), patch.object(
            engine, "_search_settings_for_model", side_effect=tiny_settings):
            directory = Path(temporary) / "synthetic_run"
            result = experiment.run_state_activity_classification("unused", "unused", directory, n_bootstrap=100)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(pd.read_csv(directory / "primary_metrics.csv")), 4)
            self.assertEqual(len(pd.read_csv(directory / "paired_feature_differences.csv")), 12)
            for model in engine.MODEL_FAMILIES:
                for feature_set in (BASELINE, AUGMENTED):
                    branch = directory / model / feature_set
                    self.assertTrue((branch / "within_HC_HC2/predictions.csv").exists())
                    self.assertTrue((branch / "leave_out_HC_HC2_predictions.csv").exists())
                    source_results = pd.read_csv(branch / "source_validation_metrics.csv")
                    self.assertTrue(source_results.status.eq("complete").all())
                    self.assertEqual(len(source_results), 4)
            self.assertTrue((directory / "source_only_predictions.csv").exists())

    def test_training_failure_is_recorded_not_reported_as_complete(self):
        prepared = synthetic_input()
        data = prepare_matched_features(*feature_fixture())
        with tempfile.TemporaryDirectory() as temporary, patch.object(
                experiment, "load_experiment_inputs",
                return_value=(data, [], StateActivityConfig().to_dict())), patch.object(
            experiment, "model_input", return_value=prepared), patch.object(
            experiment, "_fit_nested", side_effect=RuntimeError("synthetic failure")):
            directory = Path(temporary) / "failed_run"
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                experiment.run_state_activity_classification("unused", "unused", directory)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("synthetic failure", manifest["error"])


if __name__ == "__main__":
    unittest.main()
