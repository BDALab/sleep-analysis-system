import json
import logging
import os
import pickle
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from dashboard.models import Subject
from mysite.settings import MEDIA_ROOT

matplotlib.use("Agg")

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

GROUPED_STATS_DATASET_CLINICAL_PATH = (
        Path(MEDIA_ROOT)
        / "covariates"
        / "dataset-clinical"
        / "data"
        / "grouped_clinical_matrix_with_stats.xlsx"
)
GROUPED_STATS_DATASET_CLINICAL_ACC_PATH = (
        Path(MEDIA_ROOT)
        / "covariates"
        / "dataset-clinical-acc"
        / "data"
        / "grouped_clinical_matrix_with_stats.xlsx"
)
CLASSIFICATION_RESULTS_ROOT = Path(MEDIA_ROOT) / "classification" / "grouped-statistics"
IDENTITY_COLUMNS = ("#Subject", "#Gender", "#Age", "#Disease")
TARGET_COLUMN = "#DiseaseNew"
TARGET_LABEL_COLUMN = "#DiseaseNewLabel"
STATS_PREFIXES = ("SD.", "MAD.", "Range.", "IQR.", "CV.")
FEATURE_COVERAGE_THRESHOLD = 0.90
SEED = 17
LABEL_MAPPING = dict(Subject.DIAGNOSIS_CODE)
SCENARIOS = (
    ((3,), (0,)),
    ((3, 2), (0,)),
    ((2,), (0,)),
    ((3, 2, 1), (0,)),
    ((0,), (1,)),
)
MODEL_PARAMS = {
    "booster": "dart",
    "verbosity": 0,
    "n_jobs": -1,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "seed": SEED,
    "n_estimators": 100,
    "learning_rate": 0.20,
    "gamma": 1.0,
    "max_depth": 10,
    "subsample": 1.0,
    "colsample_bylevel": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 5.0,
    "tree_method": "hist",
    "device": None,
}
PARAM_GRID = {
    "clf__learning_rate": [0.001, 0.01, 0.1, 0.15, 0.2, 0.3],
    "clf__gamma": [0, 0.025, 0.05, 0.10, 0.20, 0.25, 0.30],
    "clf__max_depth": [10, 11, 12, 13],
    "clf__subsample": [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0],
    "clf__colsample_bylevel": [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0],
    "clf__min_child_weight": [0.125, 0.25, 0.5, 1.0, 3.0, 5.0, 7.0],
    "clf__scale_pos_weight": [1, 2, 3, 4, 5],
}
SEARCH_SETTINGS = {
    "param_distributions": PARAM_GRID,
    "scoring": "balanced_accuracy",
    "n_jobs": 1,
    "n_iter": 100,
    "verbose": 1,
    "random_state": SEED,
    "return_train_score": False,
}


def classification_grouped_statistics_dataset_clinical():
    return run_classification_grouped_statistics(GROUPED_STATS_DATASET_CLINICAL_PATH)


def classification_grouped_statistics_dataset_clinical_acc():
    return run_classification_grouped_statistics(GROUPED_STATS_DATASET_CLINICAL_ACC_PATH)


def run_classification_grouped_statistics(grouped_stats_path):
    grouped_stats_path = Path(grouped_stats_path)
    if not grouped_stats_path.exists():
        raise FileNotFoundError(
            f"Grouped statistics dataset not found: {grouped_stats_path}. "
            f"Run grouped clinical data first."
        )

    dataset_name = grouped_stats_path.parents[1].name
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = CLASSIFICATION_RESULTS_ROOT / dataset_name / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting grouped-statistics classification for {dataset_name} "
        f"from {grouped_stats_path}"
    )

    base_df = pd.read_excel(grouped_stats_path)
    prepared_df, excluded_labels_df, dataset_overview_df = _prepare_dataset(base_df)

    prepared_df.to_excel(run_dir / "prepared_dataset.xlsx", index=False)
    dataset_overview_df.to_excel(run_dir / "dataset_overview.xlsx", index=False)
    if not excluded_labels_df.empty:
        excluded_labels_df.to_excel(run_dir / "excluded_subjects_missing_labels.xlsx", index=False)

    _save_json(
        {
            "dataset_name": dataset_name,
            "source_path": str(grouped_stats_path),
            "run_dir": str(run_dir),
            "seed": SEED,
            "stats_prefixes": list(STATS_PREFIXES),
            "feature_coverage_threshold": FEATURE_COVERAGE_THRESHOLD,
            "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
            "scenarios": [
                {
                    "positive_codes": list(positive_codes),
                    "positive_labels": [LABEL_MAPPING[code] for code in positive_codes],
                    "negative_codes": list(negative_codes),
                    "negative_labels": [LABEL_MAPPING[code] for code in negative_codes],
                }
                for positive_codes, negative_codes in SCENARIOS
            ],
            "model_params": _json_ready_dict(_resolved_model_params()),
            "search_settings": _json_ready_dict(SEARCH_SETTINGS),
        },
        run_dir / "analysis_metadata.json",
    )

    default_summary_rows = []
    tuned_summary_rows = []

    for positive_codes, negative_codes in SCENARIOS:
        scenario_result = _run_scenario_analysis(
            prepared_df=prepared_df,
            positive_codes=positive_codes,
            negative_codes=negative_codes,
            run_dir=run_dir,
        )
        default_summary_rows.append(scenario_result["default_summary"])
        tuned_summary_rows.append(scenario_result["tuned_summary"])

    summary_path = run_dir / "classification_summary.xlsx"
    with pd.ExcelWriter(summary_path) as writer:
        dataset_overview_df.to_excel(writer, sheet_name="dataset_overview", index=False)
        if not excluded_labels_df.empty:
            excluded_labels_df.to_excel(writer, sheet_name="missing_labels", index=False)
        pd.DataFrame(default_summary_rows).to_excel(writer, sheet_name="default_metrics", index=False)
        pd.DataFrame(tuned_summary_rows).to_excel(writer, sheet_name="tuned_metrics", index=False)

    logger.info(
        f"Grouped-statistics classification finished for {dataset_name}. "
        f"Results saved to {run_dir}"
    )
    return {
        "dataset_name": dataset_name,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "prepared_dataset_path": str(run_dir / "prepared_dataset.xlsx"),
    }


def _prepare_dataset(df):
    if "#Subject" not in df.columns:
        raise KeyError("Grouped dataset must contain #Subject")

    prepared = df.copy()
    prepared["#Subject"] = prepared["#Subject"].astype(str)
    diagnosis_mapping = {
        subject.code: subject.diagnosis_code
        for subject in Subject.objects.filter(code__in=prepared["#Subject"].tolist())
    }
    prepared[TARGET_COLUMN] = prepared["#Subject"].map(diagnosis_mapping)
    prepared[TARGET_LABEL_COLUMN] = prepared[TARGET_COLUMN].map(LABEL_MAPPING)

    excluded_labels_df = prepared[prepared[TARGET_COLUMN].isna()][
        [column for column in prepared.columns if column in IDENTITY_COLUMNS or column == "#Subject"]
    ].drop_duplicates().copy()
    excluded_labels_df["exclusion_reason"] = "Missing diagnosis_code in Subject model"

    prepared = prepared[prepared[TARGET_COLUMN].notna()].copy()
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype(int)
    prepared[TARGET_LABEL_COLUMN] = prepared[TARGET_COLUMN].map(LABEL_MAPPING)

    stats_columns = [column for column in prepared.columns if str(column).startswith(STATS_PREFIXES)]
    if not stats_columns:
        raise ValueError("Grouped dataset does not contain statistics columns")

    prepared[stats_columns] = prepared[stats_columns].replace([np.inf, -np.inf], np.nan)
    ordered_columns = [
                          column
                          for column in (*IDENTITY_COLUMNS, TARGET_COLUMN, TARGET_LABEL_COLUMN)
                          if column in prepared.columns
                      ] + stats_columns
    prepared = prepared[ordered_columns]

    dataset_overview_rows = []
    counts = Counter(prepared[TARGET_COLUMN].tolist())
    for code in sorted(LABEL_MAPPING.keys()):
        dataset_overview_rows.append(
            {
                "diagnosis_code": code,
                "diagnosis_label": LABEL_MAPPING[code],
                "subject_count": counts.get(code, 0),
            }
        )
    dataset_overview_rows.append(
        {
            "diagnosis_code": "missing",
            "diagnosis_label": "Missing diagnosis_code",
            "subject_count": int(len(excluded_labels_df)),
        }
    )

    return prepared, excluded_labels_df, pd.DataFrame(dataset_overview_rows)


def _run_scenario_analysis(prepared_df, positive_codes, negative_codes, run_dir):
    scenario_label = _scenario_label(positive_codes, negative_codes)
    scenario_dir = run_dir / scenario_label
    scenario_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running classification scenario {scenario_label}")

    scenario_codes = set(positive_codes) | set(negative_codes)
    scenario_df = prepared_df[prepared_df[TARGET_COLUMN].isin(scenario_codes)].copy()
    scenario_df["binary_target"] = scenario_df[TARGET_COLUMN].apply(
        lambda code: 1 if code in positive_codes else 0
    )
    scenario_df["binary_target_label"] = scenario_df["binary_target"].map(
        {0: _codes_to_label(negative_codes), 1: _codes_to_label(positive_codes)}
    )
    scenario_df.to_excel(scenario_dir / "df_original.xlsx", index=False)

    scenario_subjects_df = scenario_df[
        [
            column
            for column in (
            "#Subject",
            "#Age",
            "#Gender",
            "#Disease",
            TARGET_COLUMN,
            TARGET_LABEL_COLUMN,
            "binary_target",
            "binary_target_label",
        )
            if column in scenario_df.columns
        ]
    ].copy()
    scenario_subjects_df.to_excel(scenario_dir / "scenario_subjects.xlsx", index=False)

    class_counts = Counter(scenario_df["binary_target"].tolist())
    scenario_overview = pd.DataFrame(
        [
            {
                "scenario": scenario_label,
                "class_label": _codes_to_label(negative_codes),
                "binary_label": 0,
                "subject_count": class_counts.get(0, 0),
            },
            {
                "scenario": scenario_label,
                "class_label": _codes_to_label(positive_codes),
                "binary_label": 1,
                "subject_count": class_counts.get(1, 0),
            },
        ]
    )
    scenario_overview.to_excel(scenario_dir / "scenario_overview.xlsx", index=False)

    default_summary = _base_summary_row(
        scenario_label=scenario_label,
        positive_codes=positive_codes,
        negative_codes=negative_codes,
        subject_count=len(scenario_df),
        positive_count=class_counts.get(1, 0),
        negative_count=class_counts.get(0, 0),
    )
    tuned_summary = default_summary.copy()

    if len(class_counts) < 2 or min(class_counts.values()) == 0:
        reason = "Scenario does not contain both binary classes"
        logger.warning(f"Skipping {scenario_label}: {reason}")
        default_summary.update({"status": "skipped", "skip_reason": reason})
        tuned_summary.update({"status": "skipped", "skip_reason": reason})
        return {
            "default_summary": default_summary,
            "tuned_summary": tuned_summary,
        }

    stats_columns = [column for column in scenario_df.columns if str(column).startswith(STATS_PREFIXES)]
    filtered_df, feature_mapping_df, feature_coverage_df = _prepare_scenario_features(
        scenario_df,
        stats_columns=stats_columns,
    )

    feature_coverage_df.to_excel(scenario_dir / "feature_coverage.xlsx", index=False)
    feature_mapping_df.to_excel(scenario_dir / "feature_name_mapping.xlsx", index=False)
    filtered_df.to_excel(scenario_dir / "df_preprocessed.xlsx", index=False)

    feature_columns = [
        column
        for column in filtered_df.columns
        if column not in {
            "#Subject",
            "#Age",
            "#Gender",
            "#Disease",
            TARGET_COLUMN,
            TARGET_LABEL_COLUMN,
            "binary_target",
            "binary_target_label",
        }
    ]
    if not feature_columns:
        reason = "No statistics features left after filtering"
        logger.warning(f"Skipping {scenario_label}: {reason}")
        default_summary.update({"status": "skipped", "skip_reason": reason})
        tuned_summary.update({"status": "skipped", "skip_reason": reason})
        return {
            "default_summary": default_summary,
            "tuned_summary": tuned_summary,
        }

    X = filtered_df[feature_columns].apply(pd.to_numeric, errors="coerce").values
    y = filtered_df["binary_target"].astype(int).values
    subjects = filtered_df["#Subject"].astype(str).tolist()

    _save_json({"feature_labels": feature_columns}, scenario_dir / "feature_labels.json")
    np.save(scenario_dir / "X_original.npy", X)
    np.save(scenario_dir / "y_original.npy", y)

    best_estimator, random_search = _run_hyperparameter_search(X, y)
    _save_pickle(best_estimator, scenario_dir / "trained_model.pkl")
    _save_json(
        _json_ready_dict(best_estimator.get_params()),
        scenario_dir / "trained_model_hyper_parameters.json",
    )
    pd.DataFrame(random_search.cv_results_).sort_values(
        by="rank_test_score"
    ).to_excel(scenario_dir / "hyperparameter_search_results.xlsx", index=False)

    feature_importance_df = _feature_importances_dataframe(
        best_estimator.named_steps["clf"],
        feature_columns,
    )
    feature_importance_df.to_excel(scenario_dir / "feature_importances.xlsx", index=False)
    _save_feature_importance_plot(
        feature_importance_df,
        title=scenario_label,
        output_path=scenario_dir / "feature_importances.pdf",
    )

    evaluation = _evaluate_leave_one_out(
        estimator=best_estimator,
        X=X,
        y=y,
        subjects=subjects,
        diagnosis_codes=filtered_df[TARGET_COLUMN].tolist(),
        diagnosis_labels=filtered_df[TARGET_LABEL_COLUMN].tolist(),
    )

    default_metrics = _compute_binary_metrics(
        evaluation["y_true"],
        evaluation["y_pred_default"],
    )
    tuned_threshold, y_pred_tuned = _tune_threshold(
        evaluation["y_true"],
        evaluation["y_prob"],
    )
    tuned_metrics = _compute_binary_metrics(
        evaluation["y_true"],
        y_pred_tuned,
    )

    fpr, tpr, roc_thresholds = roc_curve(evaluation["y_true"], evaluation["y_prob"])
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        evaluation["y_true"],
        evaluation["y_prob"],
    )
    pr_auc = average_precision_score(evaluation["y_true"], evaluation["y_prob"])
    _save_curve_points(
        roc_path=scenario_dir / "roc_curve_points.xlsx",
        pr_path=scenario_dir / "pr_curve_points.xlsx",
        fpr=fpr,
        tpr=tpr,
        roc_thresholds=roc_thresholds,
        precision_curve=precision_curve,
        recall_curve=recall_curve,
        pr_thresholds=pr_thresholds,
    )

    target_names = [_codes_to_label(negative_codes), _codes_to_label(positive_codes)]
    default_report_df = _classification_report_dataframe(
        evaluation["y_true"],
        evaluation["y_pred_default"],
        target_names=target_names,
    )
    tuned_report_df = _classification_report_dataframe(
        evaluation["y_true"],
        y_pred_tuned,
        target_names=target_names,
    )
    default_report_df.to_excel(scenario_dir / "classification_report_default.xlsx")
    tuned_report_df.to_excel(scenario_dir / "classification_report_tuned.xlsx")

    default_confusion_df = _confusion_matrix_dataframe(
        evaluation["y_true"],
        evaluation["y_pred_default"],
        target_names=target_names,
    )
    tuned_confusion_df = _confusion_matrix_dataframe(
        evaluation["y_true"],
        y_pred_tuned,
        target_names=target_names,
    )
    default_confusion_df.to_excel(scenario_dir / "confusion_matrix_default.xlsx")
    tuned_confusion_df.to_excel(scenario_dir / "confusion_matrix_tuned.xlsx")

    predictions_df = pd.DataFrame(
        {
            "#Subject": subjects,
            TARGET_COLUMN: filtered_df[TARGET_COLUMN].tolist(),
            TARGET_LABEL_COLUMN: filtered_df[TARGET_LABEL_COLUMN].tolist(),
            "binary_target": evaluation["y_true"],
            "binary_target_label": [
                _codes_to_label(positive_codes) if value == 1 else _codes_to_label(negative_codes)
                for value in evaluation["y_true"]
            ],
            "pred_default": evaluation["y_pred_default"],
            "pred_default_label": [
                _codes_to_label(positive_codes) if value == 1 else _codes_to_label(negative_codes)
                for value in evaluation["y_pred_default"]
            ],
            "pred_tuned": y_pred_tuned,
            "pred_tuned_label": [
                _codes_to_label(positive_codes) if value == 1 else _codes_to_label(negative_codes)
                for value in y_pred_tuned
            ],
            "pred_probability_positive": evaluation["y_prob"],
        }
    )
    predictions_df.to_excel(scenario_dir / "subject_predictions.xlsx", index=False)

    _save_metrics(default_metrics, scenario_dir / "cls_results_original.xlsx")
    _save_metrics(
        {**tuned_metrics, "threshold": float(tuned_threshold)},
        scenario_dir / f"cls_results_tuned_({tuned_threshold:.6f}).xlsx",
    )
    _save_roc_pr_figure(
        y_true=evaluation["y_true"],
        y_prob=evaluation["y_prob"],
        y_pred_tuned=y_pred_tuned,
        target_names=target_names,
        tuned_threshold=tuned_threshold,
        output_path=scenario_dir / "cls_roc.pdf",
    )

    shap_importance_df = _save_shap_outputs(
        estimator=best_estimator,
        X=X,
        feature_columns=feature_columns,
        subjects=subjects,
        scenario_dir=scenario_dir,
    )

    top_feature_string = ", ".join(
        f"{row['feature']} ({row['importance']:.4f})"
        for _, row in feature_importance_df.head(10).iterrows()
    )
    top_shap_string = ", ".join(
        f"{row['feature']} ({row['mean_abs_shap']:.4f})"
        for _, row in shap_importance_df.head(10).iterrows()
    )

    default_summary.update(
        {
            "status": "completed",
            "important_features": top_feature_string,
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            **default_metrics,
        }
    )
    tuned_summary.update(
        {
            "status": "completed",
            "important_features": top_feature_string,
            "important_shap_features": top_shap_string,
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "threshold": round(float(tuned_threshold), 6),
            **tuned_metrics,
        }
    )

    logger.info(
        f"Scenario {scenario_label} completed: "
        f"default BACC={default_metrics['BACC']:.4f}, "
        f"tuned BACC={tuned_metrics['BACC']:.4f}"
    )
    return {
        "default_summary": default_summary,
        "tuned_summary": tuned_summary,
    }


def _prepare_scenario_features(scenario_df, stats_columns):
    filtered = scenario_df.copy()
    numeric_features = filtered[stats_columns].apply(pd.to_numeric, errors="coerce")
    coverage = numeric_features.notna().mean()
    keep_by_coverage = coverage >= FEATURE_COVERAGE_THRESHOLD
    kept_columns = coverage[keep_by_coverage].index.tolist()

    numeric_after_coverage = numeric_features[kept_columns]
    keep_by_nonzero = numeric_after_coverage.fillna(0).abs().sum(axis=0) > 0
    kept_columns = keep_by_nonzero[keep_by_nonzero].index.tolist()

    feature_mapping_rows = []
    coverage_rows = []
    for column in stats_columns:
        display_name = _feature_display_name(column)
        kept = column in kept_columns
        coverage_rows.append(
            {
                "original_feature": column,
                "display_feature": display_name,
                "non_missing_ratio": float(coverage.get(column, 0)),
                "kept": kept,
                "drop_reason": (
                    ""
                    if kept
                    else "coverage"
                    if coverage.get(column, 0) < FEATURE_COVERAGE_THRESHOLD
                    else "all_zero"
                ),
            }
        )
        if kept:
            feature_mapping_rows.append(
                {
                    "original_feature": column,
                    "display_feature": display_name,
                }
            )

    metadata_columns = [column for column in filtered.columns if column not in stats_columns]
    filtered = filtered[metadata_columns + kept_columns].copy()
    filtered = filtered.rename(columns={column: _feature_display_name(column) for column in kept_columns})

    return (
        filtered,
        pd.DataFrame(feature_mapping_rows),
        pd.DataFrame(coverage_rows),
    )


def _run_hyperparameter_search(X, y):
    cv = LeaveOneOut()
    search = RandomizedSearchCV(
        estimator=_build_pipeline(),
        cv=cv,
        **SEARCH_SETTINGS,
    )
    try:
        search.fit(X, y)
    except Exception as exc:
        if _is_gpu_error(exc) and search.estimator.get_params().get("clf__device") == "cuda":
            logger.warning(
                "CUDA hyperparameter search failed, retrying on CPU.",
                exc_info=True,
            )
            search.estimator.set_params(clf__device="cpu")
            search.fit(X, y)
        else:
            raise
    return search.best_estimator_, search


def _evaluate_leave_one_out(estimator, X, y, subjects, diagnosis_codes, diagnosis_labels):
    cv = LeaveOneOut()
    y_true_buffer = []
    y_pred_buffer = []
    y_prob_buffer = []

    for fold_index, (train_index, test_index) in enumerate(cv.split(X), start=1):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        fold_estimator = clone(estimator)
        try:
            fold_estimator.fit(X_train, y_train)
        except Exception as exc:
            if _is_gpu_error(exc) and fold_estimator.get_params().get("clf__device") == "cuda":
                logger.warning(
                    f"CUDA fit failed on fold {fold_index}, retrying on CPU.",
                    exc_info=True,
                )
                fold_estimator.set_params(clf__device="cpu")
                fold_estimator.fit(X_train, y_train)
            else:
                raise

        y_true_buffer.extend(y_test.tolist())
        y_pred_buffer.extend(fold_estimator.predict(X_test).tolist())
        y_prob_buffer.extend(fold_estimator.predict_proba(X_test)[:, 1].tolist())

    return {
        "subjects": subjects,
        "diagnosis_codes": diagnosis_codes,
        "diagnosis_labels": diagnosis_labels,
        "y_true": np.array(y_true_buffer, dtype=int),
        "y_pred_default": np.array(y_pred_buffer, dtype=int),
        "y_prob": np.array(y_prob_buffer, dtype=float),
    }


def _build_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            ("clf", xgb.XGBClassifier(**_resolved_model_params())),
        ]
    )


def _resolved_model_params():
    params = MODEL_PARAMS.copy()
    params["device"] = _default_xgb_device()
    return params


def _default_xgb_device():
    requested = os.environ.get("GENEACTIV_XGB_DEVICE")
    if requested:
        return requested
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def _scenario_label(positive_codes, negative_codes):
    return f"scenario-{_codes_to_label(positive_codes)}_vs_{_codes_to_label(negative_codes)}"


def _codes_to_label(codes):
    return "+".join(LABEL_MAPPING[code] for code in codes)


def _base_summary_row(scenario_label, positive_codes, negative_codes, subject_count, positive_count, negative_count):
    return {
        "scenario": scenario_label,
        "positive_codes": ",".join(str(code) for code in positive_codes),
        "positive_labels": _codes_to_label(positive_codes),
        "negative_codes": ",".join(str(code) for code in negative_codes),
        "negative_labels": _codes_to_label(negative_codes),
        "subject_count": int(subject_count),
        "positive_subject_count": int(positive_count),
        "negative_subject_count": int(negative_count),
    }


def _feature_display_name(feature_name):
    for prefix in STATS_PREFIXES:
        if str(feature_name).startswith(prefix):
            return f"{feature_name[len(prefix):]} ({prefix.rstrip('.')})"
    return str(feature_name)


def _compute_binary_metrics(y_true, y_pred):
    return {
        "BACC": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "SEN": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "SPE": round(float(_specificity_score(y_true, y_pred)), 4),
        "PRE": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return 0.0 if denominator == 0 else tn / denominator


def _tune_threshold(y_true, y_prob):
    thresholds = np.arange(0, 1, 0.0001)
    mcc_values = [matthews_corrcoef(y_true, _binarize_proba(y_prob, threshold)) for threshold in thresholds]
    best_index = int(np.argmax(mcc_values))
    threshold = float(thresholds[best_index])
    return threshold, _binarize_proba(y_prob, threshold)


def _binarize_proba(y_prob, threshold):
    return (y_prob >= threshold).astype(int)


def _classification_report_dataframe(y_true, y_pred, target_names):
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    return pd.DataFrame(report).transpose()


def _confusion_matrix_dataframe(y_true, y_pred, target_names):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in target_names],
        columns=[f"pred_{name}" for name in target_names],
    )


def _feature_importances_dataframe(model, feature_columns):
    importances = getattr(model, "feature_importances_", np.zeros(len(feature_columns)))
    df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    )
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


def _save_metrics(metrics, output_path):
    pd.DataFrame([metrics]).to_excel(output_path, index=False)


def _save_curve_points(
        roc_path,
        pr_path,
        fpr,
        tpr,
        roc_thresholds,
        precision_curve,
        recall_curve,
        pr_thresholds,
):
    pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": np.append(roc_thresholds, np.nan)[: len(fpr)],
        }
    ).to_excel(roc_path, index=False)
    pd.DataFrame(
        {
            "precision": precision_curve,
            "recall": recall_curve,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    ).to_excel(pr_path, index=False)


def _save_feature_importance_plot(feature_importance_df, title, output_path):
    _set_visual_styles()
    plot_df = feature_importance_df.head(10).copy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.barplot(data=plot_df, x="importance", y="feature", ax=ax, color="#3b5b92", edgecolor="0.2")
    ax.set_title(f"Feature Importance: {title}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_roc_pr_figure(y_true, y_prob, y_pred_tuned, target_names, tuned_threshold, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred_tuned, labels=[0, 1])

    _set_visual_styles()
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    ax = axes[0, 0]
    ax.plot(fpr, tpr, color="#0165fc", linewidth=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.plot(recall_curve, precision_curve, color="#f97306", linewidth=2, label=f"AP = {pr_auc:.2f}")
    ax.set_title("Precision-recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.hist(y_prob[y_true == 0], bins=20, alpha=0.65, color="#6a5acd", label=target_names[0])
    ax.hist(y_prob[y_true == 1], bins=20, alpha=0.65, color="#2ca02c", label=target_names[1])
    ax.axvline(tuned_threshold, color="black", linestyle="--", linewidth=1, label=f"threshold = {tuned_threshold:.2f}")
    ax.set_title("Predicted probability distribution")
    ax.set_xlabel("Predicted probability of class 1")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1, 1]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        cmap="Blues",
        ax=ax,
        xticklabels=target_names,
        yticklabels=target_names,
    )
    ax.set_title("Confusion matrix (tuned threshold)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_shap_outputs(estimator, X, feature_columns, subjects, scenario_dir):
    transformed = estimator.named_steps["scaler"].transform(
        estimator.named_steps["imputer"].transform(X)
    )
    transformed_df = pd.DataFrame(transformed, columns=feature_columns, index=subjects)

    try:
        explainer = shap.TreeExplainer(estimator.named_steps["clf"])
        shap_values = explainer.shap_values(transformed_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_explanation = shap.Explanation(
            values=shap_values,
            data=transformed_df.values,
            feature_names=feature_columns,
        )
    except Exception:
        logger.warning("TreeExplainer failed, falling back to generic SHAP explainer.", exc_info=True)

        def predict_positive(data):
            return estimator.named_steps["clf"].predict_proba(data)[:, 1]

        explainer = shap.Explainer(predict_positive, transformed_df)
        shap_explanation = explainer(transformed_df)

    shap_values_df = pd.DataFrame(
        shap_explanation.values,
        columns=feature_columns,
        index=subjects,
    ).reset_index().rename(columns={"index": "#Subject"})
    shap_values_df.to_excel(scenario_dir / "shap_values.xlsx", index=False)

    shap_importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": np.abs(shap_explanation.values).mean(axis=0),
        }
    ).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_importance_df.to_excel(scenario_dir / "shap_feature_importances.xlsx", index=False)

    _set_visual_styles()
    fig = plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    fig.tight_layout()
    fig.savefig(scenario_dir / "shap_beeswarm.pdf", bbox_inches="tight")
    plt.close(fig)

    _set_visual_styles()
    fig = plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_explanation, max_display=15, show=False)
    fig.tight_layout()
    fig.savefig(scenario_dir / "shap_summary_bar.pdf", bbox_inches="tight")
    plt.close(fig)

    return shap_importance_df


def _set_visual_styles():
    plt.style.use("classic")
    sns.set()
    sns.set(font_scale=1.0)
    sns.set_style({"font.family": "serif", "font.serif": ["Times New Roman"]})


def _save_pickle(data, output_path):
    with open(output_path, "wb") as file_handle:
        pickle.dump(data, file_handle)


def _save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, ensure_ascii=True)


def _json_ready_dict(data):
    if isinstance(data, dict):
        return {str(key): _json_ready_dict(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_ready_dict(value) for value in data]
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, Path):
        return str(data)
    return data


def _is_gpu_error(exc):
    message = str(exc).lower()
    return any(keyword in message for keyword in ("cuda", "gpu", "device"))
