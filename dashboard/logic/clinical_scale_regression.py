import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from dashboard.logic.analysis_preparation import prepare_analysis_dataset
from dashboard.logic.classification_covariates import resolve_adjustment_columns
from dashboard.logic.classification_grouped_statistics import (
    ADJUSTMENT_COVARIATE_COLUMNS,
    DIARY_COVARIATE_COLUMNS,
    FEATURE_SELECTION_K_OPTIONS,
    FoldwiseCovariateResidualizer,
    SEED,
    STATS_PREFIXES,
    TARGET_COLUMN,
    TARGET_LABEL_COLUMN,
    _feature_importances_dataframe,
    _is_gpu_error,
    _json_ready_dict,
    _pipeline_params_for_json,
    _prepare_dataset,
    _prepare_scenario_features,
    _save_feature_importance_plot,
    _save_json,
    _save_pickle,
    _selected_feature_columns,
    _set_visual_styles,
)
from dashboard.logic.feature_correlation_analysis import CLINICAL_OUTCOMES
from dashboard.logic.xgboost_runtime import (
    configure_xgboost_params,
    xgboost_runtime_metadata,
)
from dashboard.models import Subject
from mysite.settings import MEDIA_ROOT

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

RESULTS_ROOT = Path(MEDIA_ROOT) / "regression" / "clinical-scales"
HC_CODE = 0
PRE_DLB_CODE = 3
HC_VS_PRE_DLB_CODES = (HC_CODE, PRE_DLB_CODE)
DIAGNOSIS_LABELS = dict(Subject.DIAGNOSIS_CODE)
REGRESSION_ADJUSTMENT_COVARIATES = ("age", "gender", "education")
REGRESSION_OUTER_CV_SPLITS = max(
    2,
    int(os.environ.get("GENEACTIV_REGRESSION_OUTER_CV_SPLITS", "5")),
)
REGRESSION_INNER_CV_SPLITS = max(
    2,
    int(os.environ.get("GENEACTIV_REGRESSION_INNER_CV_SPLITS", "5")),
)
REGRESSION_SEARCH_ITER = max(
    1,
    int(os.environ.get("GENEACTIV_REGRESSION_SEARCH_ITER", "20")),
)
MIN_REGRESSION_SUBJECTS = max(
    4,
    int(os.environ.get("GENEACTIV_REGRESSION_MIN_SUBJECTS", "12")),
)

MODEL_PARAMS = {
    "booster": "gbtree",
    "verbosity": 0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": SEED,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "gamma": 0.0,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bylevel": 1.0,
    "colsample_bytree": 0.8,
    "min_child_weight": 3.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

PARAM_GRID = {
    "feature_selector__k": list(FEATURE_SELECTION_K_OPTIONS),
    "regressor__n_estimators": [150, 300, 500],
    "regressor__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "regressor__gamma": [0, 0.05, 0.1, 0.25, 0.5],
    "regressor__max_depth": [2, 3, 4, 5],
    "regressor__subsample": [0.6, 0.8, 1.0],
    "regressor__colsample_bylevel": [0.6, 0.8, 1.0],
    "regressor__colsample_bytree": [0.5, 0.7, 0.9, 1.0],
    "regressor__min_child_weight": [1.0, 3.0, 5.0, 10.0],
    "regressor__reg_alpha": [0.0, 0.01, 0.1, 1.0],
    "regressor__reg_lambda": [0.5, 1.0, 2.0, 5.0],
}


class AdaptiveSelectKBestRegression(BaseEstimator, TransformerMixin):
    def __init__(self, k="all"):
        self.k = k
        self.selector_ = None
        self.effective_k_ = k
        self.scores_ = None
        self.pvalues_ = None

    def fit(self, X, y=None):
        if self.k == "all":
            self.effective_k_ = "all"
        else:
            self.effective_k_ = max(1, min(int(self.k), X.shape[1]))

        self.selector_ = SelectKBest(
            score_func=_safe_f_regression,
            k=self.effective_k_,
        )
        self.selector_.fit(X, y)
        self.scores_ = self.selector_.scores_
        self.pvalues_ = self.selector_.pvalues_
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self, indices=False):
        return self.selector_.get_support(indices=indices)


def clinical_scale_regression_dataset_clinical():
    return _run_prepared_clinical_scale_regression("dataset-clinical")


def clinical_scale_regression_dataset_clinical_acc():
    return _run_prepared_clinical_scale_regression("dataset-clinical-acc")


def clinical_scale_regression_hc_predlb_core_dataset_clinical():
    return _run_prepared_clinical_scale_regression(
        "dataset-clinical",
        include_diary_predictors=False,
        diagnosis_codes=HC_VS_PRE_DLB_CODES,
        diagnosis_label="HC vs preDLB",
        run_slug="hc-vs-predlb-core",
        feature_set_label="core sleep/diary grouped features",
    )


def clinical_scale_regression_hc_predlb_extended_dataset_clinical():
    return _run_prepared_clinical_scale_regression(
        "dataset-clinical",
        include_diary_predictors=True,
        diagnosis_codes=HC_VS_PRE_DLB_CODES,
        diagnosis_label="HC vs preDLB",
        run_slug="hc-vs-predlb-extended",
        feature_set_label="core sleep/diary grouped features plus lifestyle diary predictors",
    )


def clinical_scale_regression_hc_predlb_core_dataset_clinical_acc():
    return _run_prepared_clinical_scale_regression(
        "dataset-clinical-acc",
        include_diary_predictors=False,
        diagnosis_codes=HC_VS_PRE_DLB_CODES,
        diagnosis_label="HC vs preDLB",
        run_slug="hc-vs-predlb-core",
        feature_set_label="core sleep/diary/activity grouped features",
    )


def clinical_scale_regression_hc_predlb_extended_dataset_clinical_acc():
    return _run_prepared_clinical_scale_regression(
        "dataset-clinical-acc",
        include_diary_predictors=True,
        diagnosis_codes=HC_VS_PRE_DLB_CODES,
        diagnosis_label="HC vs preDLB",
        run_slug="hc-vs-predlb-extended",
        feature_set_label="core sleep/diary/activity grouped features plus lifestyle diary predictors",
    )


def _run_prepared_clinical_scale_regression(dataset_name, **kwargs):
    preparation = prepare_analysis_dataset(dataset_name)
    return run_clinical_scale_regression(
        preparation["raw_grouped_stats_path"],
        dataset_name=dataset_name,
        preparation_manifest=preparation,
        **kwargs,
    )


def run_clinical_scale_regression(
        grouped_stats_path,
        dataset_name=None,
        include_diary_predictors=True,
        selected_covariates=REGRESSION_ADJUSTMENT_COVARIATES,
        results_root=RESULTS_ROOT,
        preparation_manifest=None,
        diagnosis_codes=None,
        diagnosis_label=None,
        run_slug=None,
        feature_set_label=None,
):
    grouped_stats_path = Path(grouped_stats_path)
    if not grouped_stats_path.exists():
        raise FileNotFoundError(
            f"Grouped statistics dataset not found: {grouped_stats_path}. "
            f"Run grouped clinical data first."
        )

    dataset_name = dataset_name or grouped_stats_path.parents[1].name
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(results_root) / dataset_name / run_slug / run_label
        if run_slug
        else Path(results_root) / dataset_name / run_label
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_codes = tuple(diagnosis_codes or ())
    diagnosis_label = diagnosis_label or (
        "all diagnoses"
        if not diagnosis_codes
        else " + ".join(_diagnosis_label(code) for code in diagnosis_codes)
    )
    feature_set_label = feature_set_label or (
        "core grouped features plus lifestyle diary predictors"
        if include_diary_predictors
        else "core grouped features"
    )
    logger.info(
        f"Starting clinical-scale regression for {dataset_name} from {grouped_stats_path} "
        f"[diagnosis={diagnosis_label}, feature_set={feature_set_label}]"
    )

    base_df = pd.read_excel(grouped_stats_path)
    prepared_df, excluded_labels_df, full_dataset_overview_df, predictor_info = _prepare_dataset(
        base_df,
        include_diary_covariates=include_diary_predictors,
    )
    prepared_df = _attach_clinical_outcomes(prepared_df)
    prepared_df["#PersonGroup"] = prepared_df["#Subject"].map(_person_group_id)
    if diagnosis_codes:
        prepared_df = _filter_diagnosis_codes(prepared_df, diagnosis_codes)
    dataset_overview_df = _dataset_overview(prepared_df, excluded_labels_df)

    prepared_df.to_excel(run_dir / "prepared_regression_dataset.xlsx", index=False)
    dataset_overview_df.to_excel(run_dir / "dataset_overview.xlsx", index=False)
    full_dataset_overview_df.to_excel(run_dir / "full_dataset_overview.xlsx", index=False)
    if not excluded_labels_df.empty:
        excluded_labels_df.to_excel(run_dir / "excluded_subjects_missing_labels.xlsx", index=False)

    stats_columns = [
        column for column in prepared_df.columns
        if str(column).startswith(STATS_PREFIXES)
    ]
    additional_feature_columns = (
        [column for column in DIARY_COVARIATE_COLUMNS if column in prepared_df.columns]
        if include_diary_predictors
        else []
    )

    metadata = {
        "dataset_name": dataset_name,
        "analysis_slug": run_slug or "all-diagnoses",
        "diagnosis_filter": {
            "label": diagnosis_label,
            "codes": list(diagnosis_codes),
            "code_labels": [_diagnosis_label(code) for code in diagnosis_codes],
        },
        "feature_set_label": feature_set_label,
        "source_path": str(grouped_stats_path),
        "run_dir": str(run_dir),
        "seed": SEED,
        "targets": CLINICAL_OUTCOMES,
        "include_diary_predictors": bool(include_diary_predictors),
        "diary_predictors": _json_ready_dict(predictor_info),
        "demographic_adjustment": {
            "selected_covariates": list(selected_covariates),
            "strategy": "foldwise linear residualization fitted on training folds only",
            "column_mapping": ADJUSTMENT_COVARIATE_COLUMNS,
        },
        "validation": {
            "outer_cv": "GroupKFold by person/visit group",
            "outer_cv_splits_cap": REGRESSION_OUTER_CV_SPLITS,
            "inner_cv": "GroupKFold by person/visit group",
            "inner_cv_splits_cap": REGRESSION_INNER_CV_SPLITS,
            "search_iterations": REGRESSION_SEARCH_ITER,
            "scoring": "negative mean absolute error",
        },
        "error_rate_definition": (
            "estimation_error_rate = MAE / observed clinical-scale range "
            "among evaluated subjects"
        ),
        "minimum_subjects_per_outcome": MIN_REGRESSION_SUBJECTS,
        "model_params": _json_ready_dict(_resolved_model_params()),
        "param_grid": _json_ready_dict(PARAM_GRID),
        "xgboost_runtime": xgboost_runtime_metadata(),
        "preparation_manifest": _json_ready_dict(preparation_manifest or {}),
    }
    _save_json(metadata, run_dir / "analysis_metadata.json")

    outcome_results = []
    for outcome_label, outcome_field in CLINICAL_OUTCOMES.items():
        outcome_results.append(
            _run_outcome_regression(
                prepared_df=prepared_df,
                outcome_label=outcome_label,
                outcome_field=outcome_field,
                stats_columns=stats_columns,
                additional_feature_columns=additional_feature_columns,
                selected_covariates=selected_covariates,
                run_dir=run_dir,
            )
        )

    summary_path = run_dir / "clinical_scale_regression_summary.xlsx"
    with pd.ExcelWriter(summary_path) as writer:
        dataset_overview_df.to_excel(writer, sheet_name="dataset_overview", index=False)
        full_dataset_overview_df.to_excel(writer, sheet_name="full_dataset_overview", index=False)
        if not excluded_labels_df.empty:
            excluded_labels_df.to_excel(writer, sheet_name="missing_diagnosis", index=False)
        pd.DataFrame(outcome_results).to_excel(writer, sheet_name="metrics", index=False)
        _outcome_coverage(prepared_df).to_excel(writer, sheet_name="outcome_coverage", index=False)

    logger.info(
        f"Clinical-scale regression finished for {dataset_name}. Results saved to {run_dir}"
    )
    return {
        "dataset_name": dataset_name,
        "analysis_slug": run_slug or "all-diagnoses",
        "diagnosis_label": diagnosis_label,
        "feature_set_label": feature_set_label,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "prepared_dataset_path": str(run_dir / "prepared_regression_dataset.xlsx"),
        "outcomes": outcome_results,
    }


def _run_outcome_regression(
        prepared_df,
        outcome_label,
        outcome_field,
        stats_columns,
        additional_feature_columns,
        selected_covariates,
        run_dir,
):
    outcome_dir = run_dir / f"outcome-{_slugify(outcome_label)}"
    outcome_dir.mkdir(parents=True, exist_ok=True)

    outcome_df = prepared_df[prepared_df[outcome_field].notna()].copy()
    outcome_df[outcome_field] = pd.to_numeric(outcome_df[outcome_field], errors="coerce")
    outcome_df = outcome_df[outcome_df[outcome_field].notna()].copy()
    base_summary = {
        "outcome": outcome_label,
        "outcome_field": outcome_field,
        "status": "skipped",
        "n_subjects": int(len(outcome_df)),
        "n_person_groups": int(outcome_df["#PersonGroup"].nunique()) if "#PersonGroup" in outcome_df else 0,
    }

    if len(outcome_df) < MIN_REGRESSION_SUBJECTS:
        reason = f"need at least {MIN_REGRESSION_SUBJECTS} labelled subjects"
        _write_skip_reason(outcome_dir, reason, outcome_df)
        return {**base_summary, "reason": reason}
    if outcome_df["#PersonGroup"].nunique() < 3:
        reason = "need at least 3 person groups for nested grouped cross-validation"
        _write_skip_reason(outcome_dir, reason, outcome_df)
        return {**base_summary, "reason": reason}

    scale_min = float(outcome_df[outcome_field].min())
    scale_max = float(outcome_df[outcome_field].max())
    scale_size = scale_max - scale_min
    if not np.isfinite(scale_size) or scale_size <= 0:
        reason = "observed outcome range is zero"
        _write_skip_reason(outcome_dir, reason, outcome_df)
        return {
            **base_summary,
            "reason": reason,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "scale_size": scale_size,
        }

    preprocessed_df, feature_mapping_df, feature_coverage_df = _prepare_scenario_features(
        outcome_df,
        stats_columns=stats_columns,
        additional_feature_columns=additional_feature_columns,
    )
    feature_columns = feature_mapping_df["display_feature"].tolist()
    adjustment_columns = resolve_adjustment_columns(
        selected_covariates,
        preprocessed_df.columns,
        ADJUSTMENT_COVARIATE_COLUMNS,
    )

    if not feature_columns:
        reason = "no candidate feature passed coverage and non-zero filtering"
        _write_skip_reason(outcome_dir, reason, outcome_df)
        return {
            **base_summary,
            "reason": reason,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "scale_size": scale_size,
        }

    preprocessed_df.to_excel(outcome_dir / "df_preprocessed.xlsx", index=False)
    outcome_df.to_excel(outcome_dir / "df_original.xlsx", index=False)
    feature_mapping_df.to_excel(outcome_dir / "feature_name_mapping.xlsx", index=False)
    feature_coverage_df.to_excel(outcome_dir / "feature_coverage.xlsx", index=False)

    model_columns = feature_columns + adjustment_columns
    X = (
        preprocessed_df[model_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=float)
    )
    y = preprocessed_df[outcome_field].to_numpy(dtype=float)
    groups = preprocessed_df["#PersonGroup"].astype(str).to_numpy()

    predictions_df, fold_details_df = _evaluate_grouped_nested_cv(
        X=X,
        y=y,
        groups=groups,
        preprocessed_df=preprocessed_df,
        outcome_label=outcome_label,
        outcome_field=outcome_field,
        n_covariates=len(adjustment_columns),
    )
    metrics = _compute_regression_metrics(
        predictions_df["y_true"].to_numpy(dtype=float),
        predictions_df["y_pred"].to_numpy(dtype=float),
        scale_size=scale_size,
        baseline_pred=predictions_df["baseline_pred"].to_numpy(dtype=float),
    )
    metrics_row = {
        **base_summary,
        "status": "completed",
        "reason": "",
        "n_features": int(len(feature_columns)),
        "adjustment_covariates": ", ".join(selected_covariates),
        "adjustment_columns": ", ".join(adjustment_columns),
        "scale_min": scale_min,
        "scale_max": scale_max,
        "scale_size": scale_size,
        **metrics,
    }

    predictions_df.to_excel(outcome_dir / "subject_predictions.xlsx", index=False)
    fold_details_df.to_excel(outcome_dir / "outer_fold_details.xlsx", index=False)
    pd.DataFrame([metrics_row]).to_excel(outcome_dir / "metrics.xlsx", index=False)
    _save_prediction_scatter(
        predictions_df=predictions_df,
        outcome_label=outcome_label,
        metrics=metrics_row,
        output_path=outcome_dir / "prediction_scatter.png",
    )
    _save_prediction_scatter(
        predictions_df=predictions_df,
        outcome_label=outcome_label,
        metrics=metrics_row,
        output_path=outcome_dir / "prediction_scatter.pdf",
    )

    final_model_dir = outcome_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    final_estimator, final_search = _run_regression_search(
        X=X,
        y=y,
        groups=groups,
        n_covariates=len(adjustment_columns),
    )
    selected_feature_columns = _selected_feature_columns(final_estimator, feature_columns)
    feature_importance_df = _feature_importances_dataframe(
        final_estimator.named_steps["regressor"],
        selected_feature_columns,
    )
    feature_importance_df.to_excel(final_model_dir / "feature_importances.xlsx", index=False)
    _save_feature_importance_plot(
        feature_importance_df,
        f"{outcome_label} regression",
        final_model_dir / "feature_importances.png",
    )
    _save_pickle(final_estimator, final_model_dir / "regression_model.pkl")
    _save_json(
        {
            "best_params": _json_ready_dict(final_search.best_params_),
            "best_score_neg_mae": float(final_search.best_score_),
            "pipeline_params": _pipeline_params_for_json(final_estimator),
            "selected_features": selected_feature_columns,
        },
        final_model_dir / "model_metadata.json",
    )
    pd.DataFrame(final_search.cv_results_).to_excel(
        final_model_dir / "hyperparameter_search_results.xlsx",
        index=False,
    )

    logger.info(
        f"Clinical-scale regression completed for {outcome_label}: "
        f"n={len(outcome_df)}, MAE={metrics_row['mae']:.4f}, "
        f"RMSE={metrics_row['rmse']:.4f}, "
        f"error_rate={metrics_row['estimation_error_rate']:.4f}"
    )
    return metrics_row


def _evaluate_grouped_nested_cv(
        X,
        y,
        groups,
        preprocessed_df,
        outcome_label,
        outcome_field,
        n_covariates,
):
    outer_cv = _build_group_cv(groups, REGRESSION_OUTER_CV_SPLITS)
    prediction_rows = []
    fold_rows = []

    for fold_index, (train_index, test_index) in enumerate(
            outer_cv.split(X, y, groups=groups),
            start=1,
    ):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train = groups[train_index]

        estimator, search = _run_regression_search(
            X=X_train,
            y=y_train,
            groups=groups_train,
            n_covariates=n_covariates,
        )
        baseline_value = float(np.nanmedian(y_train))
        y_pred = _predict_with_cpu_fallback(estimator, X_test)

        for row_index, true_value, pred_value in zip(test_index, y_test, y_pred):
            row = preprocessed_df.iloc[int(row_index)]
            prediction_rows.append(
                {
                    "fold": int(fold_index),
                    "#Subject": row.get("#Subject"),
                    "#PersonGroup": row.get("#PersonGroup"),
                    "diagnosis_code": row.get(TARGET_COLUMN),
                    "diagnosis_label": row.get(TARGET_LABEL_COLUMN),
                    "outcome": outcome_label,
                    "outcome_field": outcome_field,
                    "y_true": float(true_value),
                    "y_pred": float(pred_value),
                    "baseline_pred": baseline_value,
                    "absolute_error": float(abs(true_value - pred_value)),
                    "baseline_absolute_error": float(abs(true_value - baseline_value)),
                }
            )

        fold_rows.append(
            {
                "fold": int(fold_index),
                "train_subjects": int(len(train_index)),
                "test_subjects": int(len(test_index)),
                "train_person_groups": int(len(np.unique(groups_train))),
                "test_person_groups": int(len(np.unique(groups[test_index]))),
                "baseline_median": baseline_value,
                "best_score_neg_mae": float(search.best_score_),
                "best_params": json.dumps(
                    _json_ready_dict(search.best_params_),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            }
        )

    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows)


def _run_regression_search(X, y, groups, n_covariates):
    cv = _build_group_cv(groups, REGRESSION_INNER_CV_SPLITS)
    search = RandomizedSearchCV(
        estimator=_build_pipeline(n_covariates=n_covariates),
        param_distributions=PARAM_GRID,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        n_iter=REGRESSION_SEARCH_ITER,
        verbose=1,
        random_state=SEED,
        return_train_score=False,
        cv=cv,
    )
    try:
        search.fit(X, y, groups=groups)
    except Exception as exc:
        if _is_gpu_error(exc) and search.estimator.get_params().get("regressor__device") == "cuda":
            logger.warning(
                "CUDA regression search failed, retrying on CPU.",
                exc_info=True,
            )
            search.estimator.set_params(regressor__device="cpu")
            search.fit(X, y, groups=groups)
        else:
            raise
    return search.best_estimator_, search


def _build_pipeline(n_covariates=0):
    return Pipeline(
        [
            (
                "covariate_residualizer",
                FoldwiseCovariateResidualizer(n_covariates=n_covariates),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_filter", VarianceThreshold(threshold=0.0)),
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            ("feature_selector", AdaptiveSelectKBestRegression(k="all")),
            ("regressor", xgb.XGBRegressor(**_resolved_model_params())),
        ]
    )


def _build_group_cv(groups, max_splits):
    unique_groups = np.unique(np.asarray(groups, dtype=str))
    n_splits = min(int(max_splits), len(unique_groups))
    if n_splits < 2:
        raise ValueError(
            "Grouped CV requires at least 2 person groups, "
            f"got {len(unique_groups)}"
        )
    return GroupKFold(n_splits=n_splits)


def _resolved_model_params():
    return configure_xgboost_params(MODEL_PARAMS)


def _predict_with_cpu_fallback(estimator, X):
    try:
        return estimator.predict(X)
    except Exception as exc:
        if _is_gpu_error(exc) and estimator.get_params().get("regressor__device") == "cuda":
            logger.warning("CUDA regression predict failed, retrying on CPU.", exc_info=True)
            estimator.set_params(regressor__device="cpu")
            return estimator.predict(X)
        raise


def _compute_regression_metrics(y_true, y_pred, scale_size, baseline_pred=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scale_size = float(scale_size)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "estimation_error_rate": _safe_divide(mae, scale_size),
        "estimation_error_rate_percent": _safe_divide(mae, scale_size) * 100.0,
        "r2": _safe_r2(y_true, y_pred),
        "pearson_r": _safe_correlation(pearsonr, y_true, y_pred),
        "spearman_r": _safe_correlation(spearmanr, y_true, y_pred),
    }
    if baseline_pred is not None:
        baseline_pred = np.asarray(baseline_pred, dtype=float)
        baseline_mae = float(mean_absolute_error(y_true, baseline_pred))
        baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))
        metrics.update(
            {
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "baseline_estimation_error_rate": _safe_divide(
                    baseline_mae,
                    scale_size,
                ),
                "mae_improvement_vs_baseline": baseline_mae - mae,
                "mae_improvement_vs_baseline_percent": _safe_divide(
                    baseline_mae - mae,
                    baseline_mae,
                ) * 100.0,
            }
        )
    return metrics


def _safe_f_regression(X, y):
    scores, pvalues = f_regression(X, y)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(
        pvalues,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )


def _safe_divide(numerator, denominator):
    if denominator is None or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _safe_r2(y_true, y_pred):
    if len(y_true) < 2 or np.isclose(np.nanvar(y_true), 0.0):
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _safe_correlation(correlation_func, y_true, y_pred):
    if len(y_true) < 3:
        return float("nan")
    if np.isclose(np.nanvar(y_true), 0.0) or np.isclose(np.nanvar(y_pred), 0.0):
        return float("nan")
    result = correlation_func(y_true, y_pred)
    if hasattr(result, "statistic"):
        return float(result.statistic)
    return float(result[0])


def _attach_clinical_outcomes(prepared_df):
    prepared = prepared_df.copy()
    subject_codes = prepared["#Subject"].astype(str).dropna().unique().tolist()
    outcome_rows = {
        subject.code: {
            outcome_field: getattr(subject, outcome_field)
            for outcome_field in CLINICAL_OUTCOMES.values()
        }
        for subject in Subject.objects.filter(code__in=subject_codes)
    }
    for outcome_field in CLINICAL_OUTCOMES.values():
        prepared[outcome_field] = prepared["#Subject"].map(
            lambda subject_code: outcome_rows.get(str(subject_code), {}).get(outcome_field)
        )
    return prepared


def _filter_diagnosis_codes(prepared_df, diagnosis_codes):
    requested_codes = tuple(dict.fromkeys(int(code) for code in diagnosis_codes))
    available_codes = set(prepared_df[TARGET_COLUMN].dropna().astype(int).tolist())
    missing_codes = [code for code in requested_codes if code not in available_codes]
    if missing_codes:
        raise ValueError(
            "Clinical-scale regression diagnosis filter is missing requested "
            f"diagnosis codes: {missing_codes}"
        )

    filtered = prepared_df[
        prepared_df[TARGET_COLUMN].astype(int).isin(requested_codes)
    ].copy()
    if filtered.empty:
        raise ValueError(
            "Clinical-scale regression diagnosis filter produced no rows: "
            f"{requested_codes}"
        )
    return filtered


def _dataset_overview(prepared_df, excluded_labels_df=None):
    counts = prepared_df[TARGET_COLUMN].value_counts().to_dict()
    rows = []
    for code in sorted(DIAGNOSIS_LABELS.keys()):
        rows.append(
            {
                "diagnosis_code": code,
                "diagnosis_label": _diagnosis_label(code),
                "subject_count": int(counts.get(code, 0)),
            }
        )
    rows.append(
        {
            "diagnosis_code": "missing",
            "diagnosis_label": "Missing diagnosis_code",
            "subject_count": int(len(excluded_labels_df)) if excluded_labels_df is not None else 0,
        }
    )
    return pd.DataFrame(rows)


def _diagnosis_label(code):
    return DIAGNOSIS_LABELS.get(int(code), str(code))


def _outcome_coverage(prepared_df):
    rows = []
    for outcome_label, outcome_field in CLINICAL_OUTCOMES.items():
        values = pd.to_numeric(prepared_df[outcome_field], errors="coerce")
        rows.append(
            {
                "outcome": outcome_label,
                "outcome_field": outcome_field,
                "n_subjects_with_value": int(values.notna().sum()),
                "scale_min": float(values.min()) if values.notna().any() else np.nan,
                "scale_max": float(values.max()) if values.notna().any() else np.nan,
                "scale_size": (
                    float(values.max() - values.min())
                    if values.notna().any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _person_group_id(subject_code):
    code = str(subject_code)
    replacements = (
        ("pre-LBD2-", "pre-LBD-"),
        ("pre-LBD3-", "pre-LBD-"),
        ("pre-LBD2_", "pre-LBD_"),
        ("pre-LBD3_", "pre-LBD_"),
        ("preLBD2_", "preLBD_"),
        ("preLBD3_", "preLBD_"),
        ("preLBD2-", "preLBD-"),
        ("preLBD3-", "preLBD-"),
        ("preDLB2_", "preDLB_"),
        ("preDLB3_", "preDLB_"),
        ("preDLB2-", "preDLB-"),
        ("preDLB3-", "preDLB-"),
        ("HC2-", "HC-"),
        ("HC3-", "HC-"),
        ("HC2_", "HC_"),
        ("HC3_", "HC_"),
    )
    for prefix, replacement in replacements:
        if code.startswith(prefix):
            return f"{replacement}{code[len(prefix):]}"
    return code


def _save_prediction_scatter(predictions_df, outcome_label, metrics, output_path):
    if predictions_df.empty:
        return

    _set_visual_styles()
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.scatterplot(
        data=predictions_df,
        x="y_true",
        y="y_pred",
        hue="diagnosis_label",
        ax=ax,
        s=55,
        edgecolor="0.2",
        alpha=0.85,
    )
    value_min = float(np.nanmin([predictions_df["y_true"].min(), predictions_df["y_pred"].min()]))
    value_max = float(np.nanmax([predictions_df["y_true"].max(), predictions_df["y_pred"].max()]))
    margin = (value_max - value_min) * 0.05 if value_max > value_min else 1.0
    axis_min = value_min - margin
    axis_max = value_max + margin
    ax.plot([axis_min, axis_max], [axis_min, axis_max], "--", color="black", linewidth=1)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_title(
        f"{outcome_label}: predicted vs observed\n"
        f"MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, "
        f"error rate={metrics['estimation_error_rate']:.2%}"
    )
    ax.set_xlabel("Observed score")
    ax.set_ylabel("Predicted score")
    ax.grid(alpha=0.25)
    ax.legend(title="Diagnosis", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_skip_reason(outcome_dir, reason, outcome_df):
    logger.warning(f"Skipping clinical-scale regression for {outcome_dir.name}: {reason}")
    pd.DataFrame([{"reason": reason}]).to_excel(outcome_dir / "skip_reason.xlsx", index=False)
    outcome_df.to_excel(outcome_dir / "df_original.xlsx", index=False)


def _slugify(value):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
