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

from dashboard.logic.analysis_preparation import prepare_analysis_dataset
from dashboard.logic.classification_covariates import (
    resolve_adjustment_columns,
    validate_scenario_covariate_mapping,
)
from dashboard.logic.classification_grouped_statistics import (
    ADJUSTMENT_COVARIATE_COLUMNS,
    DIARY_COVARIATE_COLUMNS,
    FEATURE_COVERAGE_THRESHOLD,
    FEATURE_BLOCK_ALL,
    FEATURE_SELECTION_K_OPTIONS,
    FEATURE_SELECTION_RFE_OPTIONS,
    FEATURE_SELECTOR_MODE_KBEST,
    FEATURE_SELECTOR_MODE_RFE,
    FEATURE_SELECTOR_MODES,
    LABEL_MAPPING,
    SCENARIOS,
    SEARCH_SETTINGS,
    SEED,
    STATS_PREFIXES,
    TARGET_COLUMN,
    TARGET_LABEL_COLUMN,
    _base_summary_row,
    _binarize_proba,
    _build_pipeline_with_selector,
    _classification_report_dataframe,
    _codes_to_label,
    _compute_binary_metrics,
    _confusion_matrix_dataframe,
    _feature_importances_dataframe,
    _is_gpu_error,
    _json_ready_dict,
    _pipeline_params_for_json,
    _prepare_dataset,
    _prepare_scenario_features,
    _save_curve_points,
    _save_dataset_pca_projection,
    _save_feature_importance_plot,
    _save_json,
    _save_metrics,
    _save_pickle,
    _save_roc_pr_figure,
    _save_selected_feature_scores,
    _save_shap_outputs,
    _scenario_label,
    _scenario_covariate_mapping,
    _selected_feature_columns,
    _tune_threshold,
)
from dashboard.logic.feature_families import (
    ACTIVITY_EXTENSION_STABLE_FAMILY_IDS,
    feature_family_metadata,
    PRIMARY_SLEEP_STABLE_FAMILY_IDS,
)
from dashboard.logic.xgboost_runtime import xgboost_runtime_metadata
from mysite.settings import MEDIA_ROOT

logger = logging.getLogger(__name__)

STRICT_RESULTS_ROOT = Path(MEDIA_ROOT) / "classification" / "grouped-statistics-strict"
STRICT_RESULTS_WITH_COVARIATES_ROOT = (
        Path(MEDIA_ROOT) / "classification" / "grouped-statistics-strict-with-covariates"
)
STRICT_DEFAULT_SEARCH_ITER = max(1, int(os.environ.get("GENEACTIV_STRICT_SEARCH_ITER", "20")))
STRICT_MAX_INNER_CV_SPLITS = max(2, int(os.environ.get("GENEACTIV_STRICT_INNER_CV_SPLITS", "5")))
STRICT_RFE_DEFAULT_SEARCH_ITER = max(1, int(os.environ.get("GENEACTIV_STRICT_RFE_SEARCH_ITER", "12")))
HC_VS_PREDLB_SCENARIO_FILTER = (((3,), (0,)),)


def classification_grouped_statistics_strict_dataset_clinical():
    return _run_prepared_strict_classification("dataset-clinical")


def classification_grouped_statistics_strict_dataset_clinical_acc():
    return _run_prepared_strict_classification("dataset-clinical-acc")


def classification_grouped_statistics_strict_with_covariates_dataset_clinical():
    return _run_prepared_strict_classification(
        "dataset-clinical",
        include_diary_covariates=True,
        results_root=STRICT_RESULTS_WITH_COVARIATES_ROOT,
    )


def classification_grouped_statistics_strict_with_covariates_dataset_clinical_acc():
    return _run_prepared_strict_classification(
        "dataset-clinical-acc",
        include_diary_covariates=True,
        results_root=STRICT_RESULTS_WITH_COVARIATES_ROOT,
    )


def classification_grouped_statistics_strict_with_covariates_rfe_dataset_clinical():
    return _run_prepared_strict_classification(
        "dataset-clinical",
        include_diary_covariates=True,
        results_root=STRICT_RESULTS_WITH_COVARIATES_ROOT,
        feature_selector_mode=FEATURE_SELECTOR_MODE_RFE,
    )


def classification_grouped_statistics_strict_with_covariates_rfe_dataset_clinical_acc():
    return _run_prepared_strict_classification(
        "dataset-clinical-acc",
        include_diary_covariates=True,
        results_root=STRICT_RESULTS_WITH_COVARIATES_ROOT,
        feature_selector_mode=FEATURE_SELECTOR_MODE_RFE,
    )


def classification_grouped_statistics_strict_stable_families_hc_predlb_dataset_clinical():
    return _run_prepared_strict_classification(
        "dataset-clinical",
        output_dataset_name="dataset-clinical-stable-primary-sleep-hc-predlb",
        include_diary_covariates=False,
        results_root=STRICT_RESULTS_WITH_COVARIATES_ROOT,
        feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST,
        allowed_feature_family_ids=PRIMARY_SLEEP_STABLE_FAMILY_IDS,
        scenario_filter=HC_VS_PREDLB_SCENARIO_FILTER,
        analysis_notes=(
            "Stable-family strict confirmation: HC vs preDLB only, primary sleep families only.",
        ),
    )


def classification_grouped_statistics_strict_stable_families_hc_predlb_dataset_clinical_acc():
    return _run_prepared_strict_classification(
        "dataset-clinical-acc",
        output_dataset_name="dataset-clinical-acc-stable-primary-sleep-activity-hc-predlb",
        include_diary_covariates=False,
        results_root=STRICT_RESULTS_WITH_COVARIATES_ROOT,
        feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST,
        allowed_feature_family_ids=(
                PRIMARY_SLEEP_STABLE_FAMILY_IDS
                | ACTIVITY_EXTENSION_STABLE_FAMILY_IDS
        ),
        scenario_filter=HC_VS_PREDLB_SCENARIO_FILTER,
        analysis_notes=(
            "Stable-family strict confirmation: HC vs preDLB only, primary sleep families plus activity variability.",
        ),
    )


def _run_prepared_strict_classification(dataset_name, output_dataset_name=None, **kwargs):
    preparation = prepare_analysis_dataset(dataset_name)
    scenario_covariates = _scenario_covariate_mapping(preparation)
    return run_classification_grouped_statistics_strict(
        preparation["raw_grouped_stats_path"],
        dataset_name=output_dataset_name or dataset_name,
        scenario_covariates=scenario_covariates,
        preparation_manifest=preparation,
        **kwargs,
    )


def run_classification_grouped_statistics_strict(
        grouped_stats_path,
        include_diary_covariates=False,
        results_root=STRICT_RESULTS_ROOT,
        feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST,
        dataset_name=None,
        scenario_covariates=None,
        preparation_manifest=None,
        allowed_feature_family_ids=None,
        scenario_filter=None,
        analysis_notes=(),
):
    grouped_stats_path = Path(grouped_stats_path)
    if not grouped_stats_path.exists():
        raise FileNotFoundError(
            f"Grouped statistics dataset not found: {grouped_stats_path}. "
            f"Run grouped clinical data first."
        )
    if feature_selector_mode not in FEATURE_SELECTOR_MODES:
        raise ValueError(
            f"Unknown feature selector mode {feature_selector_mode}. "
            f"Available: {', '.join(FEATURE_SELECTOR_MODES)}"
        )

    dataset_name = dataset_name or grouped_stats_path.parents[1].name
    scenario_covariates = validate_scenario_covariate_mapping(
        scenario_covariates,
        SCENARIOS,
    )
    scenario_list = _filtered_scenarios(scenario_filter)
    allowed_feature_family_ids = (
        frozenset(allowed_feature_family_ids)
        if allowed_feature_family_ids
        else frozenset()
    )
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_dir = dataset_name if feature_selector_mode == FEATURE_SELECTOR_MODE_KBEST else f"{dataset_name}-{feature_selector_mode}"
    run_dir = results_root / mode_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting strict grouped-statistics classification for {dataset_name} "
        f"from {grouped_stats_path} "
        f"(diary covariates enabled={include_diary_covariates}, selector={feature_selector_mode})"
    )

    base_df = pd.read_excel(grouped_stats_path)
    prepared_df, excluded_labels_df, dataset_overview_df, covariate_info = _prepare_dataset(
        base_df,
        include_diary_covariates=include_diary_covariates,
    )

    prepared_df.to_excel(run_dir / "prepared_dataset.xlsx", index=False)
    dataset_overview_df.to_excel(run_dir / "dataset_overview.xlsx", index=False)
    if not excluded_labels_df.empty:
        excluded_labels_df.to_excel(run_dir / "excluded_subjects_missing_labels.xlsx", index=False)
    pca_output = _save_dataset_pca_projection(prepared_df, run_dir)

    _save_json(
        {
            "dataset_name": dataset_name,
            "source_path": str(grouped_stats_path),
            "run_dir": str(run_dir),
            "mode": "strict_nested_cv_publication",
            "seed": SEED,
            "include_diary_covariates": bool(include_diary_covariates),
            "diary_covariates": _json_ready_dict(covariate_info),
            "feature_family_filter": {
                "enabled": bool(allowed_feature_family_ids),
                "allowed_family_ids": sorted(allowed_feature_family_ids),
                "scope": "statistics features and diary-derived predictor columns before coverage filtering",
            },
            "stats_prefixes": list(STATS_PREFIXES),
            "feature_coverage_threshold": FEATURE_COVERAGE_THRESHOLD,
            "feature_selector_mode": feature_selector_mode,
            "feature_selection": _feature_selection_metadata(feature_selector_mode),
            "covariate_adjustment": {
                "strategy": "foldwise linear residualization fitted on training folds only",
                "scenario_covariates": {
                    _scenario_label(positive_codes, negative_codes): list(
                        scenario_covariates.get(
                            (tuple(positive_codes), tuple(negative_codes)),
                            (),
                        )
                    )
                    for positive_codes, negative_codes in scenario_list
                },
            },
            "preparation_manifest": _json_ready_dict(preparation_manifest or {}),
            "xgboost_runtime": xgboost_runtime_metadata(),
            "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
            "strict_search_iterations": (
                STRICT_DEFAULT_SEARCH_ITER
                if feature_selector_mode == FEATURE_SELECTOR_MODE_KBEST
                else STRICT_RFE_DEFAULT_SEARCH_ITER
            ),
            "strict_max_inner_cv_splits": STRICT_MAX_INNER_CV_SPLITS,
            "notes": [
                "Outer evaluation uses Leave-One-Out cross-validation.",
                "Hyperparameters are tuned inside each outer training fold only.",
                "Tuned-threshold predictions use a threshold selected from inner CV predictions on the training fold only.",
                "SHAP and final feature importances are computed on a final model fit on the full scenario dataset after evaluation.",
                *analysis_notes,
            ],
            "scenarios": [
                {
                    "positive_codes": list(positive_codes),
                    "positive_labels": [LABEL_MAPPING[code] for code in positive_codes],
                    "negative_codes": list(negative_codes),
                    "negative_labels": [LABEL_MAPPING[code] for code in negative_codes],
                }
                for positive_codes, negative_codes in scenario_list
            ],
        },
        run_dir / "analysis_metadata.json",
    )

    default_summary_rows = []
    tuned_summary_rows = []

    for positive_codes, negative_codes in scenario_list:
        scenario_result = _run_strict_scenario_analysis(
            prepared_df=prepared_df,
            positive_codes=positive_codes,
            negative_codes=negative_codes,
            run_dir=run_dir,
            feature_selector_mode=feature_selector_mode,
            allowed_feature_family_ids=allowed_feature_family_ids,
            selected_covariates=scenario_covariates.get(
                (tuple(positive_codes), tuple(negative_codes)),
                (),
            ),
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
        "pca_dir": str(pca_output["pca_dir"]),
    }


def _filtered_scenarios(scenario_filter):
    if not scenario_filter:
        return list(SCENARIOS)

    requested = {
        (tuple(positive_codes), tuple(negative_codes))
        for positive_codes, negative_codes in scenario_filter
    }
    return [
        (positive_codes, negative_codes)
        for positive_codes, negative_codes in SCENARIOS
        if (tuple(positive_codes), tuple(negative_codes)) in requested
    ]


def _filter_columns_by_feature_family(
        stats_columns,
        additional_feature_columns,
        allowed_feature_family_ids,
):
    allowed_feature_family_ids = frozenset(allowed_feature_family_ids)
    filter_rows = []
    kept_stats_columns = []
    kept_additional_columns = []

    for column, source in (
            [(column, "statistics") for column in stats_columns]
            + [(column, "diary_predictor") for column in additional_feature_columns]
    ):
        metadata = feature_family_metadata(column)
        family_id = metadata["Feature family ID"]
        kept = family_id in allowed_feature_family_ids
        filter_rows.append(
            {
                **metadata,
                "Classifier feature source": source,
                "Allowed family IDs": ", ".join(sorted(allowed_feature_family_ids)),
                "Kept by stable-family filter": kept,
            }
        )
        if kept and source == "statistics":
            kept_stats_columns.append(column)
        elif kept and source == "diary_predictor":
            kept_additional_columns.append(column)

    return (
        kept_stats_columns,
        kept_additional_columns,
        pd.DataFrame(filter_rows),
    )


def _run_strict_scenario_analysis(
        prepared_df,
        positive_codes,
        negative_codes,
        run_dir,
        feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST,
        selected_covariates=(),
        allowed_feature_family_ids=frozenset(),
):
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
            "#Education",
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
        feature_block_key=FEATURE_BLOCK_ALL,
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
    covariate_columns = [column for column in DIARY_COVARIATE_COLUMNS if column in scenario_df.columns]
    if allowed_feature_family_ids:
        stats_columns, covariate_columns, family_filter_df = _filter_columns_by_feature_family(
            stats_columns=stats_columns,
            additional_feature_columns=covariate_columns,
            allowed_feature_family_ids=allowed_feature_family_ids,
        )
        family_filter_df.to_excel(scenario_dir / "feature_family_filter.xlsx", index=False)

    filtered_df, feature_mapping_df, feature_coverage_df = _prepare_scenario_features(
        scenario_df,
        stats_columns=stats_columns,
        additional_feature_columns=covariate_columns,
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
            "#Education",
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

    adjustment_columns = resolve_adjustment_columns(
        selected_covariates=selected_covariates,
        available_columns=filtered_df.columns,
        covariate_columns=ADJUSTMENT_COVARIATE_COLUMNS,
    )
    X = filtered_df[feature_columns + adjustment_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).values
    y = filtered_df["binary_target"].astype(int).values
    subjects = filtered_df["#Subject"].astype(str).tolist()

    _save_json(
        {
            "feature_labels": feature_columns,
            "allowed_feature_family_ids": sorted(allowed_feature_family_ids),
            "foldwise_adjustment_covariates": list(selected_covariates),
            "foldwise_adjustment_columns": adjustment_columns,
            "adjustment_fit_scope": "inner/outer training fold only",
            "adjusted_data_source": "raw grouped statistics",
        },
        scenario_dir / "feature_labels.json",
    )
    adjustment_label = ", ".join(selected_covariates) if selected_covariates else "none"
    for summary in (default_summary, tuned_summary):
        summary.update(
            {
                "selected_adjustment_covariates": adjustment_label,
                "adjustment_fit_scope": "inner/outer training fold only",
            }
        )
    np.save(scenario_dir / "X_original.npy", X)
    np.save(scenario_dir / "y_original.npy", y)

    nested_results = _run_nested_leave_one_out(
        X=X,
        y=y,
        subjects=subjects,
        scenario_dir=scenario_dir,
        feature_selector_mode=feature_selector_mode,
        n_covariates=len(adjustment_columns),
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
        feature_selector_mode=feature_selector_mode,
        n_covariates=len(adjustment_columns),
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


def _run_nested_leave_one_out(
        X,
        y,
        subjects,
        scenario_dir,
        feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST,
        n_covariates=0,
):
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
            feature_selector_mode=feature_selector_mode,
            n_covariates=n_covariates,
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


def _fit_final_interpretation_model(
        X,
        y,
        feature_columns,
        subjects,
        output_dir,
        title,
        feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST,
        n_covariates=0,
):
    inner_cv = _build_inner_cv(y)
    final_estimator, random_search = _run_search_with_cv(
        X=X,
        y=y,
        cv=inner_cv,
        feature_selector_mode=feature_selector_mode,
        n_covariates=n_covariates,
    )
    final_estimator = _fit_with_device_fallback(final_estimator, X, y)

    _save_pickle(final_estimator, output_dir / "trained_model.pkl")
    _save_json(
        _pipeline_params_for_json(final_estimator),
        output_dir / "trained_model_hyper_parameters.json",
    )
    selected_feature_columns = _selected_feature_columns(final_estimator, feature_columns)
    _save_json({"feature_labels": selected_feature_columns}, output_dir / "feature_labels.json")
    _save_selected_feature_scores(
        estimator=final_estimator,
        selected_feature_columns=selected_feature_columns,
        output_path=output_dir / "selected_feature_scores.xlsx",
        candidate_feature_columns=feature_columns,
    )
    pd.DataFrame(random_search.cv_results_).sort_values(by="rank_test_score").to_excel(
        output_dir / "hyperparameter_search_results.xlsx",
        index=False,
    )

    feature_importance_df = _feature_importances_dataframe(
        final_estimator.named_steps["clf"],
        selected_feature_columns,
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
        feature_columns=selected_feature_columns,
        subjects=subjects,
        scenario_dir=output_dir,
    )
    return {
        "estimator": final_estimator,
        "feature_importances": feature_importance_df,
        "shap_importances": shap_importance_df,
    }


def _run_search_with_cv(
        X,
        y,
        cv,
        feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST,
        n_covariates=0,
):
    search_settings = _search_settings_for_selector_mode(feature_selector_mode)
    search = RandomizedSearchCV(
        estimator=_build_pipeline_with_selector(
            feature_selector_mode,
            n_covariates=n_covariates,
        ),
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


def _search_settings_for_selector_mode(feature_selector_mode):
    search_settings = SEARCH_SETTINGS.copy()
    param_distributions = {
        key: value
        for key, value in SEARCH_SETTINGS.get("param_distributions", {}).items()
    }

    if feature_selector_mode == FEATURE_SELECTOR_MODE_RFE:
        param_distributions.pop("feature_selector__k", None)
        param_distributions["feature_selector__n_features_to_select"] = list(FEATURE_SELECTION_RFE_OPTIONS)
        param_distributions["feature_selector__step"] = [0.05, 0.1, 0.2]
        search_settings["n_iter"] = STRICT_RFE_DEFAULT_SEARCH_ITER
    else:
        param_distributions["feature_selector__k"] = list(FEATURE_SELECTION_K_OPTIONS)
        search_settings["n_iter"] = STRICT_DEFAULT_SEARCH_ITER

    search_settings["param_distributions"] = param_distributions
    return search_settings


def _feature_selection_metadata(feature_selector_mode):
    if feature_selector_mode == FEATURE_SELECTOR_MODE_RFE:
        return {
            "pipeline_steps": [
                "median imputation",
                "constant-feature removal",
                "min-max scaling",
                "XGBoost RFE",
            ],
            "n_features_options": list(FEATURE_SELECTION_RFE_OPTIONS),
            "step_options": [0.05, 0.1, 0.2],
            "note": (
                "RFE runs inside each inner CV split; "
                "n_features_to_select is clipped if larger than current feature count."
            ),
        }
    return {
        "pipeline_steps": [
            "median imputation",
            "constant-feature removal",
            "min-max scaling",
            "ANOVA SelectKBest",
        ],
        "k_options": _json_ready_dict(
            SEARCH_SETTINGS.get("param_distributions", {}).get("feature_selector__k", [])
        ),
        "k_note": "k is tuned inside each inner CV and clipped if larger than the current feature count.",
    }


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
