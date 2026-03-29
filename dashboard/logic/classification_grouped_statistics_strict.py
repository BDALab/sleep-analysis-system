import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, StratifiedKFold, cross_val_predict

from dashboard.logic.classification_grouped_statistics import (
    FEATURE_COVERAGE_THRESHOLD,
    GROUPED_STATS_DATASET_CLINICAL_ACC_PATH,
    GROUPED_STATS_DATASET_CLINICAL_PATH,
    LABEL_MAPPING,
    SCENARIOS,
    SEARCH_SETTINGS,
    SEED,
    STATS_PREFIXES,
    TARGET_COLUMN,
    TARGET_LABEL_COLUMN,
    _base_summary_row,
    _binarize_proba,
    _build_pipeline,
    _classification_report_dataframe,
    _codes_to_label,
    _compute_binary_metrics,
    _confusion_matrix_dataframe,
    _feature_importances_dataframe,
    _is_gpu_error,
    _json_ready_dict,
    _prepare_dataset,
    _prepare_scenario_features,
    _save_curve_points,
    _save_feature_importance_plot,
    _save_json,
    _save_metrics,
    _save_pickle,
    _save_roc_pr_figure,
    _save_shap_outputs,
    _scenario_label,
    _tune_threshold,
)
from mysite.settings import MEDIA_ROOT

logger = logging.getLogger(__name__)

STRICT_RESULTS_ROOT = Path(MEDIA_ROOT) / "classification" / "grouped-statistics-strict"
STRICT_DEFAULT_SEARCH_ITER = max(1, int(os.environ.get("GENEACTIV_STRICT_SEARCH_ITER", "20")))
STRICT_MAX_INNER_CV_SPLITS = max(2, int(os.environ.get("GENEACTIV_STRICT_INNER_CV_SPLITS", "5")))


def classification_grouped_statistics_strict_dataset_clinical():
    return run_classification_grouped_statistics_strict(GROUPED_STATS_DATASET_CLINICAL_PATH)


def classification_grouped_statistics_strict_dataset_clinical_acc():
    return run_classification_grouped_statistics_strict(GROUPED_STATS_DATASET_CLINICAL_ACC_PATH)


def run_classification_grouped_statistics_strict(grouped_stats_path):
    grouped_stats_path = Path(grouped_stats_path)
    if not grouped_stats_path.exists():
        raise FileNotFoundError(
            f"Grouped statistics dataset not found: {grouped_stats_path}. "
            f"Run grouped clinical data first."
        )

    dataset_name = grouped_stats_path.parents[1].name
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = STRICT_RESULTS_ROOT / dataset_name / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting strict grouped-statistics classification for {dataset_name} "
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
            "mode": "strict_nested_cv_publication",
            "seed": SEED,
            "stats_prefixes": list(STATS_PREFIXES),
            "feature_coverage_threshold": FEATURE_COVERAGE_THRESHOLD,
            "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
            "strict_search_iterations": STRICT_DEFAULT_SEARCH_ITER,
            "strict_max_inner_cv_splits": STRICT_MAX_INNER_CV_SPLITS,
            "notes": [
                "Outer evaluation uses Leave-One-Out cross-validation.",
                "Hyperparameters are tuned inside each outer training fold only.",
                "Tuned-threshold predictions use a threshold selected from inner CV predictions on the training fold only.",
                "SHAP and final feature importances are computed on a final model fit on the full scenario dataset after evaluation."
            ],
            "scenarios": [
                {
                    "positive_codes": list(positive_codes),
                    "positive_labels": [LABEL_MAPPING[code] for code in positive_codes],
                    "negative_codes": list(negative_codes),
                    "negative_labels": [LABEL_MAPPING[code] for code in negative_codes],
                }
                for positive_codes, negative_codes in SCENARIOS
            ],
        },
        run_dir / "analysis_metadata.json",
    )

    default_summary_rows = []
    tuned_summary_rows = []

    for positive_codes, negative_codes in SCENARIOS:
        scenario_result = _run_strict_scenario_analysis(
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
        pd.DataFrame(default_summary_rows).to_excel(writer, sheet_name="nested_default_metrics", index=False)
        pd.DataFrame(tuned_summary_rows).to_excel(writer, sheet_name="nested_tuned_metrics", index=False)

    logger.info(
        f"Strict grouped-statistics classification finished for {dataset_name}. "
        f"Results saved to {run_dir}"
    )
    return {
        "dataset_name": dataset_name,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "prepared_dataset_path": str(run_dir / "prepared_dataset.xlsx"),
    }


def _run_strict_scenario_analysis(prepared_df, positive_codes, negative_codes, run_dir):
    scenario_label = _scenario_label(positive_codes, negative_codes)
    scenario_dir = run_dir / scenario_label
    scenario_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running strict classification scenario {scenario_label}")

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

    class_counts = scenario_df["binary_target"].value_counts().to_dict()
    scenario_overview = pd.DataFrame(
        [
            {
                "scenario": scenario_label,
                "class_label": _codes_to_label(negative_codes),
                "binary_label": 0,
                "subject_count": int(class_counts.get(0, 0)),
            },
            {
                "scenario": scenario_label,
                "class_label": _codes_to_label(positive_codes),
                "binary_label": 1,
                "subject_count": int(class_counts.get(1, 0)),
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
        logger.warning(f"Skipping strict scenario {scenario_label}: {reason}")
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
        logger.warning(f"Skipping strict scenario {scenario_label}: {reason}")
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

    nested_results = _run_nested_leave_one_out(
        X=X,
        y=y,
        subjects=subjects,
        scenario_dir=scenario_dir,
    )
    nested_results["outer_fold_details"].to_excel(scenario_dir / "outer_fold_details.xlsx", index=False)
    nested_results["subject_predictions"].to_excel(scenario_dir / "subject_predictions.xlsx", index=False)

    y_true = nested_results["y_true"]
    y_pred_default = nested_results["y_pred_default"]
    y_pred_tuned = nested_results["y_pred_tuned"]
    y_prob = nested_results["y_prob"]

    default_metrics = _compute_binary_metrics(y_true, y_pred_default)
    tuned_metrics = _compute_binary_metrics(y_true, y_pred_tuned)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
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
    _classification_report_dataframe(y_true, y_pred_default, target_names).to_excel(
        scenario_dir / "classification_report_default.xlsx"
    )
    _classification_report_dataframe(y_true, y_pred_tuned, target_names).to_excel(
        scenario_dir / "classification_report_tuned.xlsx"
    )
    _confusion_matrix_dataframe(y_true, y_pred_default, target_names).to_excel(
        scenario_dir / "confusion_matrix_default.xlsx"
    )
    _confusion_matrix_dataframe(y_true, y_pred_tuned, target_names).to_excel(
        scenario_dir / "confusion_matrix_tuned.xlsx"
    )

    _save_metrics(default_metrics, scenario_dir / "cls_results_original.xlsx")
    tuned_metrics_with_threshold = {
        **tuned_metrics,
        "threshold_strategy": "inner_cv_per_outer_fold",
        "mean_threshold": float(nested_results["outer_fold_details"]["tuned_threshold"].mean()),
        "median_threshold": float(nested_results["outer_fold_details"]["tuned_threshold"].median()),
    }
    _save_metrics(tuned_metrics_with_threshold, scenario_dir / "cls_results_tuned_nested.xlsx")
    _save_roc_pr_figure(
        y_true=y_true,
        y_prob=y_prob,
        y_pred_tuned=y_pred_tuned,
        target_names=target_names,
        tuned_threshold=float(nested_results["outer_fold_details"]["tuned_threshold"].median()),
        output_path=scenario_dir / "cls_roc.pdf",
    )

    final_model_dir = scenario_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_result = _fit_final_interpretation_model(
        X=X,
        y=y,
        feature_columns=feature_columns,
        subjects=subjects,
        output_dir=final_model_dir,
        title=scenario_label,
    )

    top_feature_string = ", ".join(
        f"{row['feature']} ({row['importance']:.4f})"
        for _, row in final_model_result["feature_importances"].head(10).iterrows()
    )
    top_shap_string = ", ".join(
        f"{row['feature']} ({row['mean_abs_shap']:.4f})"
        for _, row in final_model_result["shap_importances"].head(10).iterrows()
    )

    default_summary.update(
        {
            "status": "completed",
            "mode": "nested_loo",
            "important_features_final_model": top_feature_string,
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            **default_metrics,
        }
    )
    tuned_summary.update(
        {
            "status": "completed",
            "mode": "nested_loo",
            "important_features_final_model": top_feature_string,
            "important_shap_features_final_model": top_shap_string,
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "threshold_strategy": "inner_cv_per_outer_fold",
            "mean_threshold": round(float(nested_results["outer_fold_details"]["tuned_threshold"].mean()), 6),
            "median_threshold": round(float(nested_results["outer_fold_details"]["tuned_threshold"].median()), 6),
            **tuned_metrics,
        }
    )

    return {
        "default_summary": default_summary,
        "tuned_summary": tuned_summary,
    }


def _run_nested_leave_one_out(X, y, subjects, scenario_dir):
    outer_cv = LeaveOneOut()
    y_true_buffer = []
    y_pred_default_buffer = []
    y_pred_tuned_buffer = []
    y_prob_buffer = []
    prediction_rows = []
    fold_rows = []

    for fold_index, (train_index, test_index) in enumerate(outer_cv.split(X), start=1):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        subject_test = subjects[test_index[0]]
        inner_cv = _build_inner_cv(y_train)

        best_estimator, random_search = _run_search_with_cv(
            X=X_train,
            y=y_train,
            cv=inner_cv,
        )
        tuned_threshold = _estimate_threshold_from_training(
            estimator=best_estimator,
            X_train=X_train,
            y_train=y_train,
            cv=inner_cv,
        )

        fold_model = clone(best_estimator)
        fold_model = _fit_with_device_fallback(fold_model, X_train, y_train)
        y_prob = float(fold_model.predict_proba(X_test)[:, 1][0])
        y_pred_default = int(fold_model.predict(X_test)[0])
        y_pred_tuned = int(_binarize_proba(np.array([y_prob]), tuned_threshold)[0])
        y_true = int(y_test[0])

        best_clf_params = best_estimator.named_steps["clf"].get_params()
        fold_rows.append(
            {
                "fold": fold_index,
                "#Subject": subject_test,
                "y_true": y_true,
                "y_pred_default": y_pred_default,
                "y_pred_tuned": y_pred_tuned,
                "y_prob": y_prob,
                "tuned_threshold": float(tuned_threshold),
                "inner_cv_splits": inner_cv.get_n_splits(),
                "best_inner_score": float(random_search.best_score_),
                "best_params_json": json.dumps(_json_ready_dict(best_clf_params), ensure_ascii=True),
            }
        )
        prediction_rows.append(
            {
                "#Subject": subject_test,
                "y_true": y_true,
                "y_pred_default": y_pred_default,
                "y_pred_tuned": y_pred_tuned,
                "pred_probability_positive": y_prob,
                "tuned_threshold": float(tuned_threshold),
            }
        )
        y_true_buffer.append(y_true)
        y_pred_default_buffer.append(y_pred_default)
        y_pred_tuned_buffer.append(y_pred_tuned)
        y_prob_buffer.append(y_prob)

    return {
        "y_true": np.array(y_true_buffer, dtype=int),
        "y_pred_default": np.array(y_pred_default_buffer, dtype=int),
        "y_pred_tuned": np.array(y_pred_tuned_buffer, dtype=int),
        "y_prob": np.array(y_prob_buffer, dtype=float),
        "outer_fold_details": pd.DataFrame(fold_rows),
        "subject_predictions": pd.DataFrame(prediction_rows),
    }


def _fit_final_interpretation_model(X, y, feature_columns, subjects, output_dir, title):
    inner_cv = _build_inner_cv(y)
    final_estimator, random_search = _run_search_with_cv(X=X, y=y, cv=inner_cv)
    final_estimator = _fit_with_device_fallback(final_estimator, X, y)

    _save_pickle(final_estimator, output_dir / "trained_model.pkl")
    _save_json(
        _json_ready_dict(final_estimator.get_params()),
        output_dir / "trained_model_hyper_parameters.json",
    )
    pd.DataFrame(random_search.cv_results_).sort_values(by="rank_test_score").to_excel(
        output_dir / "hyperparameter_search_results.xlsx",
        index=False,
    )

    feature_importance_df = _feature_importances_dataframe(
        final_estimator.named_steps["clf"],
        feature_columns,
    )
    feature_importance_df.to_excel(output_dir / "feature_importances.xlsx", index=False)
    _save_feature_importance_plot(
        feature_importance_df=feature_importance_df,
        title=f"{title} (final model)",
        output_path=output_dir / "feature_importances.pdf",
    )
    shap_importance_df = _save_shap_outputs(
        estimator=final_estimator,
        X=X,
        feature_columns=feature_columns,
        subjects=subjects,
        scenario_dir=output_dir,
    )
    return {
        "estimator": final_estimator,
        "feature_importances": feature_importance_df,
        "shap_importances": shap_importance_df,
    }


def _run_search_with_cv(X, y, cv):
    search_settings = SEARCH_SETTINGS.copy()
    search_settings["n_iter"] = STRICT_DEFAULT_SEARCH_ITER
    search = RandomizedSearchCV(
        estimator=_build_pipeline(),
        cv=cv,
        **search_settings,
    )
    try:
        search.fit(X, y)
    except Exception as exc:
        if _is_gpu_error(exc) and search.estimator.get_params().get("clf__device") == "cuda":
            logger.warning(
                "CUDA hyperparameter search failed in strict pipeline, retrying on CPU.",
                exc_info=True,
            )
            search.estimator.set_params(clf__device="cpu")
            search.fit(X, y)
        else:
            raise
    return search.best_estimator_, search


def _estimate_threshold_from_training(estimator, X_train, y_train, cv):
    try:
        train_prob = _cross_val_predict_probabilities(estimator, X_train, y_train, cv)
        threshold, _ = _tune_threshold(y_train, train_prob)
        return threshold
    except Exception:
        logger.warning(
            "Inner CV threshold tuning failed in strict pipeline, using default threshold 0.5.",
            exc_info=True,
        )
        return 0.5


def _cross_val_predict_probabilities(estimator, X, y, cv):
    estimator_for_cv = clone(estimator)
    try:
        probabilities = cross_val_predict(
            estimator_for_cv,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
    except Exception as exc:
        if _is_gpu_error(exc) and estimator_for_cv.get_params().get("clf__device") == "cuda":
            logger.warning(
                "CUDA cross_val_predict failed in strict pipeline, retrying on CPU.",
                exc_info=True,
            )
            estimator_for_cv.set_params(clf__device="cpu")
            probabilities = cross_val_predict(
                estimator_for_cv,
                X,
                y,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
        else:
            raise
    return probabilities


def _fit_with_device_fallback(estimator, X, y):
    try:
        estimator.fit(X, y)
    except Exception as exc:
        if _is_gpu_error(exc) and estimator.get_params().get("clf__device") == "cuda":
            logger.warning(
                "CUDA fit failed in strict pipeline, retrying on CPU.",
                exc_info=True,
            )
            estimator.set_params(clf__device="cpu")
            estimator.fit(X, y)
        else:
            raise
    return estimator


def _build_inner_cv(y):
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min())
    n_splits = min(STRICT_MAX_INNER_CV_SPLITS, min_class_count)
    if n_splits < 2:
        raise ValueError(
            "Strict inner CV requires at least 2 samples in each class "
            f"but got class counts {class_counts.to_dict()}"
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
