#!/usr/bin/env python3
"""Create reproducible TRIPOD+AI reporting outputs from saved OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

MODEL_SPECS = {
    "Broad sleep/diary + RFE": "broad-strict-rfe",
    "Stable sleep families": "stable-primary-sleep",
    "Stable sleep + activity variability": "stable-sleep-activity",
}
METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    "balanced_accuracy": "Balanced accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "mcc": "Matthews correlation coefficient",
    "brier": "Brier score",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 1701
SEARCH_SPACE = {
    "analysis_seed": 17,
    "outer_cv": "StratifiedGroupKFold(5)",
    "inner_cv": "StratifiedGroupKFold(5)",
    "search_scoring": "balanced_accuracy",
    "rfe_randomized_search_iterations": 12,
    "kbest_randomized_search_iterations": 20,
    "threshold_objective": "Matthews correlation coefficient",
    "threshold_grid": {"start": 0.0, "stop_exclusive": 1.0, "step": 0.0001},
    "feature_selection": {
        "rfe_features": [20, 40, 80, 120],
        "rfe_step": [0.05, 0.1, 0.2],
        "kbest_features": [20, 40, 80, 120, "all"],
    },
    "xgboost": {
        "n_estimators": [150, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "gamma": [0, 0.05, 0.1, 0.25, 0.5],
        "max_depth": [2, 3, 4, 5, 6],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bylevel": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
        "min_child_weight": [1.0, 3.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "scale_pos_weight": [1, 2, 3, 4, 5],
    },
}


def _metrics(y_true: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    positive = y_true == 1
    negative = ~positive
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())

    ranks = rankdata(probability)
    roc_auc = (
                      ranks[positive].sum() - n_positive * (n_positive + 1) / 2
              ) / (n_positive * n_negative)
    order = np.argsort(-probability, kind="mergesort")
    ordered_y = y_true[order]
    ordered_probability = probability[order]
    group_ends = np.r_[
        np.flatnonzero(ordered_probability[1:] != ordered_probability[:-1]),
        len(ordered_probability) - 1,
    ]
    cumulative_positive = np.cumsum(ordered_y)[group_ends]
    group_positive = np.diff(np.r_[0, cumulative_positive])
    precision_at_group_end = cumulative_positive / (group_ends + 1)
    pr_auc = np.sum(precision_at_group_end * group_positive) / n_positive

    true_positive = int(np.sum(positive & (prediction == 1)))
    false_negative = n_positive - true_positive
    true_negative = int(np.sum(negative & (prediction == 0)))
    false_positive = n_negative - true_negative
    sensitivity = true_positive / n_positive
    specificity = true_negative / n_negative
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (true_positive * true_negative - false_positive * false_negative) / denominator
        if denominator
        else 0.0
    )
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "mcc": float(mcc),
        "brier": float(np.mean((probability - y_true) ** 2)),
    }


def _stratified_person_bootstrap(
        predictions: pd.DataFrame,
        probability_column: str,
        prediction_columns: tuple[str, ...],
        seed: int,
) -> dict[str, np.ndarray]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    probability = predictions[probability_column].to_numpy(dtype=float)
    predicted = {
        column: predictions[column].to_numpy(dtype=int)
        for column in prediction_columns
    }
    group_indices = {
        str(group): np.asarray(indices, dtype=int)
        for group, indices in predictions.groupby("person_group", sort=False).indices.items()
    }
    group_labels = {
        group: int(y_true[indices[0]])
        for group, indices in group_indices.items()
    }
    if any(np.unique(y_true[indices]).size != 1 for indices in group_indices.values()):
        raise ValueError("A person group contains more than one outcome label")
    strata = {
        label: np.asarray(
            [group for group, value in group_labels.items() if value == label],
            dtype=object,
        )
        for label in (0, 1)
    }

    metric_names = tuple(METRIC_LABELS)
    samples = {
        column: np.empty((BOOTSTRAP_REPLICATES, len(metric_names)), dtype=float)
        for column in prediction_columns
    }
    rng = np.random.default_rng(seed)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_groups = np.concatenate(
            [rng.choice(groups, size=len(groups), replace=True) for groups in strata.values()]
        )
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        for column in prediction_columns:
            values = _metrics(
                y_true[sampled_indices],
                probability[sampled_indices],
                predicted[column][sampled_indices],
            )
            samples[column][replicate] = [values[name] for name in metric_names]
    return samples


def _uncertainty_rows(
        model: str,
        predictions: pd.DataFrame,
        probability_column: str,
        prediction_columns: dict[str, str],
        seed: int,
        validation: str,
) -> list[dict[str, object]]:
    bootstrap = _stratified_person_bootstrap(
        predictions,
        probability_column,
        tuple(prediction_columns.values()),
        seed,
    )
    rows: list[dict[str, object]] = []
    y_true = predictions["y_true"].to_numpy(dtype=int)
    probability = predictions[probability_column].to_numpy(dtype=float)
    metric_names = tuple(METRIC_LABELS)
    for threshold_label, prediction_column in prediction_columns.items():
        point = _metrics(
            y_true,
            probability,
            predictions[prediction_column].to_numpy(dtype=int),
        )
        intervals = np.quantile(bootstrap[prediction_column], [0.025, 0.975], axis=0)
        for index, metric in enumerate(metric_names):
            rows.append(
                {
                    "validation": validation,
                    "model": model,
                    "threshold": threshold_label,
                    "visits": int(len(predictions)),
                    "people": int(predictions["person_group"].nunique()),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "estimate": point[metric],
                    "ci_lower": float(intervals[0, index]),
                    "ci_upper": float(intervals[1, index]),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": seed,
                }
            )
    return rows


def _calibration_summary(model: str, predictions: pd.DataFrame) -> dict[str, object]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    probability = np.clip(
        predictions["pred_probability_positive"].to_numpy(dtype=float),
        1e-6,
        1 - 1e-6,
    )
    logit_probability = np.log(probability / (1 - probability)).reshape(-1, 1)
    calibration_model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=10_000)
    calibration_model.fit(logit_probability, y_true)
    return {
        "model": model,
        "visits": int(len(predictions)),
        "people": int(predictions["person_group"].nunique()),
        "outcome_prevalence": float(y_true.mean()),
        "mean_predicted_probability": float(probability.mean()),
        "brier_score": float(np.mean((probability - y_true) ** 2)),
        "calibration_intercept": float(calibration_model.intercept_[0]),
        "calibration_slope": float(calibration_model.coef_[0, 0]),
        "interpretation": "Descriptive OOF calibration only; not a clinical risk model.",
    }


def _plot_calibration(predictions_by_model: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Ideal")
    for model, predictions in predictions_by_model.items():
        observed, predicted = calibration_curve(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["pred_probability_positive"].to_numpy(dtype=float),
            n_bins=5,
            strategy="quantile",
        )
        ax.plot(predicted, observed, marker="o", linewidth=1.5, label=model)
    ax.set(
        xlabel="Mean out-of-fold predicted probability",
        ylabel="Observed preDLB proportion",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Descriptive calibration of secondary classifiers",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _fold_hyperparameters(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, directory in MODEL_SPECS.items():
        folds = pd.read_excel(run_root / directory / "outer_fold_details.xlsx")
        for _, fold in folds.iterrows():
            params = json.loads(fold["best_params_json"])
            rows.append(
                {
                    "model": model,
                    "outer_fold": int(fold["fold"]),
                    "inner_cv_splits": int(fold["inner_cv_splits"]),
                    "best_inner_balanced_accuracy": float(fold["best_inner_score"]),
                    "inner_cv_mcc_threshold": float(fold["tuned_threshold"]),
                    **params,
                }
            )
    return pd.DataFrame(rows)


def _missingness_and_flow(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, directory in MODEL_SPECS.items():
        model_root = run_root / directory
        subjects = pd.read_excel(model_root / "scenario_subjects.xlsx")
        coverage = pd.read_excel(model_root / "feature_coverage.xlsx")
        rows.append(
            {
                "model": model,
                "analysed_visits": int(len(subjects)),
                "analysed_people": int(subjects["person_group"].nunique()),
                "hc_visits": int((subjects["binary_target"] == 0).sum()),
                "predlb_visits": int((subjects["binary_target"] == 1).sum()),
                "missing_age": int(subjects["#Age"].isna().sum()),
                "missing_sex": int(subjects["#Gender"].isna().sum()),
                "missing_education": int(subjects["#Education"].isna().sum()),
                "candidate_columns": int(len(coverage)),
                "columns_removed_below_coverage_threshold": int((~coverage["kept"]).sum()),
                "columns_with_any_missingness_retained": int(
                    ((coverage["non_missing_ratio"] < 1) & coverage["kept"]).sum()
                ),
                "minimum_retained_non_missing_ratio": float(
                    coverage.loc[coverage["kept"], "non_missing_ratio"].min()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    run_root = (
            repo_root
            / "media/classification/person-grouped-thesis/hc-vs-predlb/20260702_111118"
    )
    output_root = repo_root / "manuscript/cognitive-computation/supplement"
    output_root.mkdir(parents=True, exist_ok=True)

    predictions_by_model: dict[str, pd.DataFrame] = {}
    uncertainty_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for model_index, (model, directory) in enumerate(MODEL_SPECS.items()):
        predictions = pd.read_excel(run_root / directory / "subject_predictions.xlsx")
        predictions_by_model[model] = predictions
        uncertainty_rows.extend(
            _uncertainty_rows(
                model,
                predictions,
                "pred_probability_positive",
                {"default 0.5": "y_pred_default", "inner-CV tuned": "y_pred_tuned"},
                BOOTSTRAP_SEED + model_index,
                "nested person-grouped OOF",
            )
        )
        calibration_rows.append(_calibration_summary(model, predictions))

    broad_root = run_root / "broad-strict-rfe"
    source_predictions = pd.read_excel(
        broad_root / "source_only_negative_control_predictions.xlsx"
    ).rename(
        columns={
            "source_only_probability_positive": "pred_probability_positive",
            "source_only_predicted": "y_pred_default",
        }
    )
    uncertainty_rows.extend(
        _uncertainty_rows(
            "Ascertainment-stratum-only benchmark",
            source_predictions,
            "pred_probability_positive",
            {"default 0.5": "y_pred_default"},
            BOOTSTRAP_SEED + len(MODEL_SPECS),
            "person-grouped OOF benchmark",
        )
    )

    for model_index, (model, directory) in enumerate(list(MODEL_SPECS.items())[:2]):
        within = pd.read_excel(run_root / directory / "within_cohort_nested_cv_predictions.xlsx")
        for cohort, cohort_predictions in within.groupby("source_cohort", sort=False):
            uncertainty_rows.extend(
                _uncertainty_rows(
                    f"{model} - {cohort}",
                    cohort_predictions.reset_index(drop=True),
                    "pred_probability_positive",
                    {"default 0.5": "y_pred_default"},
                    BOOTSTRAP_SEED + 10 + model_index,
                    "within-stratum nested person-grouped OOF",
                )
            )

    pd.DataFrame(uncertainty_rows).to_csv(
        output_root / "classification_uncertainty.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        output_root / "classification_calibration.csv", index=False
    )
    _fold_hyperparameters(run_root).to_csv(
        output_root / "classification_fold_hyperparameters.csv", index=False
    )
    _missingness_and_flow(run_root).to_csv(
        output_root / "classification_missingness_and_flow.csv", index=False
    )
    (output_root / "classification_search_space.json").write_text(
        json.dumps(SEARCH_SPACE, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_calibration(
        predictions_by_model,
        output_root / "classification_calibration.pdf",
    )


if __name__ == "__main__":
    main()
