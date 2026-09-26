import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dashboard.logic.classification_person_grouped_validation import (
    FEATURE_SELECTOR_MODE_EMBEDDED,
    MODEL_FAMILY_ELASTIC_NET,
    MODEL_FAMILY_XGBOOST,
    PERSON_GROUPED_RUN_SPECS,
    _build_stratified_group_cv,
    _build_model_pipeline,
    _grouped_cv_skip_reason,
    _holdout_skip_reason,
    _paired_model_comparisons,
    _run_grouped_nested_cv,
    _search_settings_for_model,
)


class ClassificationPersonGroupedValidationTest(unittest.TestCase):
    def test_stratified_group_cv_keeps_person_groups_intact(self):
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        groups = np.array(["HC-1", "HC-1", "HC-2", "P-1", "P-1", "P-2", "HC-3", "P-3"])

        cv = _build_stratified_group_cv(y, groups, max_splits=5)

        self.assertEqual(cv.n_splits, 3)
        for train_index, test_index in cv.split(np.zeros((len(y), 1)), y, groups=groups):
            train_groups = set(groups[train_index])
            test_groups = set(groups[test_index])
            self.assertFalse(train_groups & test_groups)

    def test_grouped_cv_skip_reason_detects_too_few_positive_groups(self):
        y = np.array([0, 0, 0, 1, 1])
        groups = np.array(["HC-1", "HC-2", "HC-3", "P-1", "P-1"])

        reason = _grouped_cv_skip_reason(y, groups)

        self.assertIn("at least two person groups per class", reason)

    def test_holdout_skip_reason_uses_person_groups(self):
        y_train = np.array([0, 0, 1, 1])
        y_test = np.array([0, 1])
        groups_train = np.array(["HC-1", "HC-2", "P-1", "P-1"])

        reason = _holdout_skip_reason(y_train, y_test, groups_train)

        self.assertIn("inner grouped CV not possible", reason)

    def test_default_specs_pair_both_models_for_each_feature_set(self):
        by_comparison = {}
        for spec in PERSON_GROUPED_RUN_SPECS:
            by_comparison.setdefault(spec["comparison_key"], set()).add(spec["model_family"])

        self.assertEqual(len(PERSON_GROUPED_RUN_SPECS), 6)
        self.assertEqual(
            set(by_comparison),
            {"broad", "stable_sleep", "stable_sleep_activity"},
        )
        for model_families in by_comparison.values():
            self.assertEqual(
                model_families,
                {MODEL_FAMILY_XGBOOST, MODEL_FAMILY_ELASTIC_NET},
            )

    def test_elastic_net_pipeline_uses_saga_and_embedded_selection(self):
        pipeline = _build_model_pipeline(
            model_family=MODEL_FAMILY_ELASTIC_NET,
            feature_selector_mode=FEATURE_SELECTOR_MODE_EMBEDDED,
            n_covariates=2,
        )

        self.assertNotIn("feature_selector", pipeline.named_steps)
        classifier = pipeline.named_steps["clf"]
        self.assertIsInstance(classifier, LogisticRegression)
        self.assertEqual(classifier.solver, "saga")
        self.assertEqual(classifier.class_weight, "balanced")
        self.assertEqual(pipeline.named_steps["covariate_residualizer"].n_covariates, 2)

    def test_elastic_net_search_tunes_regularization_inside_inner_cv(self):
        settings = _search_settings_for_model(
            MODEL_FAMILY_ELASTIC_NET,
            FEATURE_SELECTOR_MODE_EMBEDDED,
        )

        self.assertEqual(settings["scoring"], "balanced_accuracy")
        self.assertIn("clf__C", settings["param_distributions"])
        self.assertIn("clf__l1_ratio", settings["param_distributions"])
        self.assertNotIn("feature_selector__k", settings["param_distributions"])

    def test_elastic_net_completes_grouped_nested_cv_and_exports_coefficients(self):
        rng = np.random.default_rng(42)
        y = np.repeat([0, 1], 12)
        X = rng.normal(size=(len(y), 6))
        X[:, 0] += y * 1.5
        groups = np.array([f"person-{index:02d}" for index in range(len(y))])
        subjects = np.array([f"subject-{index:02d}" for index in range(len(y))])
        source_cohorts = np.where(np.arange(len(y)) % 2, "source-a", "source-b")
        visit_indices = np.ones(len(y), dtype=int)

        with patch(
            "dashboard.logic.classification_person_grouped_validation."
            "ELASTIC_NET_SEARCH_ITER",
            2,
        ):
            predictions, folds, coefficients = _run_grouped_nested_cv(
                X=X,
                y=y,
                groups=groups,
                subjects=subjects,
                source_cohorts=source_cohorts,
                visit_indices=visit_indices,
                model_family=MODEL_FAMILY_ELASTIC_NET,
                feature_selector_mode=FEATURE_SELECTOR_MODE_EMBEDDED,
                n_covariates=0,
                feature_names=[f"feature-{index}" for index in range(X.shape[1])],
                outer_splits=3,
                inner_splits=2,
            )

        self.assertEqual(len(predictions), len(y))
        self.assertEqual(predictions["person_group"].nunique(), len(groups))
        self.assertEqual(len(folds), 3)
        self.assertEqual(len(coefficients), X.shape[1] * len(folds))
        self.assertEqual(set(coefficients["feature"]), {f"feature-{i}" for i in range(6)})
        self.assertTrue(predictions["pred_probability_positive"].between(0, 1).all())

    def test_paired_bootstrap_compares_identical_people(self):
        y_true = np.array([0] * 6 + [1] * 6)
        subjects = [f"S-{index}" for index in range(len(y_true))]
        base = pd.DataFrame(
            {
                "#Subject": subjects,
                "person_group": subjects,
                "source_cohort": ["source-a"] * 6 + ["source-b"] * 6,
                "visit_index": [1] * len(y_true),
                "y_true": y_true,
            }
        )
        xgboost_predictions = base.assign(
            y_pred_default=[0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            pred_probability_positive=[
                0.10,
                0.20,
                0.65,
                0.70,
                0.30,
                0.55,
                0.40,
                0.60,
                0.45,
                0.70,
                0.80,
                0.90,
            ],
        )
        elastic_predictions = base.assign(
            y_pred_default=y_true,
            pred_probability_positive=[
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
            ],
        )
        completed_runs = [
            {
                "spec": {
                    "comparison_key": "test",
                    "feature_set_label": "Test features",
                    "model_family": MODEL_FAMILY_XGBOOST,
                },
                "predictions": xgboost_predictions,
            },
            {
                "spec": {
                    "comparison_key": "test",
                    "feature_set_label": "Test features",
                    "model_family": MODEL_FAMILY_ELASTIC_NET,
                },
                "predictions": elastic_predictions,
            },
        ]

        comparison = _paired_model_comparisons(completed_runs, n_bootstrap=200)
        auc_row = comparison[comparison["metric"].eq("AUC")].iloc[0]

        self.assertEqual(len(comparison), 6)
        self.assertEqual(auc_row["person_group_count"], 12)
        self.assertGreater(auc_row["difference_elastic_minus_xgboost"], 0)
        self.assertGreater(auc_row["probability_elastic_net_better"], 0.9)


if __name__ == "__main__":
    unittest.main()
