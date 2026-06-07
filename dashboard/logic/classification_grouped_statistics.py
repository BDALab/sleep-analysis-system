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
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_classif
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
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dashboard.models import SleepDiaryDay, Subject
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
CLASSIFICATION_RESULTS_WITH_COVARIATES_ROOT = (
        Path(MEDIA_ROOT) / "classification" / "grouped-statistics-with-covariates"
)
ABLATION_RESULTS_ROOT = Path(MEDIA_ROOT) / "classification" / "grouped-statistics-ablation"
IDENTITY_COLUMNS = ("#Subject", "#Gender", "#Age", "#Education", "#Disease")
TARGET_COLUMN = "#DiseaseNew"
TARGET_LABEL_COLUMN = "#DiseaseNewLabel"
STATS_PREFIXES = (
    "Mean.",
    "Median.",
    "Min.",
    "Max.",
    "Slope.",
    "SD.",
    "MAD.",
    "Range.",
    "IQR.",
    "CV.",
)
FEATURE_COVERAGE_THRESHOLD = 0.90
TUNING_MAX_CV_SPLITS = max(2, int(os.environ.get("GENEACTIV_TUNING_CV_SPLITS", "5")))
RFE_TUNING_N_ITER = max(1, int(os.environ.get("GENEACTIV_RFE_SEARCH_ITER", "40")))
FEATURE_SELECTION_K_OPTIONS = (20, 40, 80, 120, "all")
FEATURE_SELECTION_RFE_OPTIONS = (20, 40, 80, 120)
FEATURE_SELECTOR_MODE_KBEST = "kbest"
FEATURE_SELECTOR_MODE_RFE = "rfe"
FEATURE_SELECTOR_MODES = (FEATURE_SELECTOR_MODE_KBEST, FEATURE_SELECTOR_MODE_RFE)
DIARY_COVARIATE_COLUMNS = (
    "sleeping_pill_rate",
    "rest_quality_mean",
    "sleep_quality_mean",
    "day_sleep_time_mean",
    "day_sleep_count_mean",
    "caffeine_count_mean",
    "alcohol_count_mean",
    "caffeine_time_mean",
    "alcohol_time_mean",
    "caffeine_time_std",
    "alcohol_time_std",
)
FEATURE_BLOCK_ALL = "all"
FEATURE_BLOCKS = {
    FEATURE_BLOCK_ALL: {
        "label": "all feature blocks",
        "description": "Use all grouped statistics features.",
    },
    "diary-only": {
        "label": "diary only",
        "description": "Use diary and diary_norm grouped statistics features only.",
    },
    "actigraphy-only": {
        "label": "actigraphy only",
        "description": "Use actigraphy and actigraphy_norm grouped statistics features only.",
    },
    "activity-only": {
        "label": "activity only",
        "description": "Use activity grouped statistics features only.",
    },
    "norm-only": {
        "label": "norm features only",
        "description": "Use *_norm grouped statistics features only.",
    },
    "non-norm-only": {
        "label": "non-norm features only",
        "description": "Exclude *_norm grouped statistics features.",
    },
    "level-only": {
        "label": "level features only",
        "description": "Use Mean, Median, Min, and Max statistics only.",
    },
    "trend-only": {
        "label": "trend features only",
        "description": "Use Slope statistics only.",
    },
    "variability-only": {
        "label": "variability features only",
        "description": "Use SD, MAD, Range, IQR, and CV statistics only.",
    },
}
SEED = 17
LABEL_MAPPING = dict(Subject.DIAGNOSIS_CODE)
SCENARIOS = (
    ((3,), (0,)),
    ((3, 2), (0,)),
    ((2,), (0,)),
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
    "feature_selector__k": list(FEATURE_SELECTION_K_OPTIONS),
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


class AdaptiveSelectKBest(BaseEstimator, TransformerMixin):
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
            score_func=_safe_f_classif,
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


class AdaptiveRFESelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=40, step=0.1):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.selector_ = None
        self.effective_n_features_ = None
        self.support_ = None
        self.ranking_ = None
        self.scores_ = None
        self.pvalues_ = None

    def fit(self, X, y=None):
        n_features = int(X.shape[1])
        if n_features <= 0:
            raise ValueError("AdaptiveRFESelector requires at least one feature")

        if self.n_features_to_select in (None, "all"):
            self.effective_n_features_ = n_features
            self.selector_ = None
            self.support_ = np.ones(n_features, dtype=bool)
            self.ranking_ = np.ones(n_features, dtype=int)
            return self

        requested = int(self.n_features_to_select)
        self.effective_n_features_ = max(1, min(requested, n_features))
        step = self.step
        if isinstance(step, float):
            step = min(max(step, 0.01), 0.99)
        else:
            step = max(1, int(step))

        self.selector_ = RFE(
            estimator=_build_rfe_estimator(),
            n_features_to_select=self.effective_n_features_,
            step=step,
            importance_getter="auto",
        )
        self.selector_.fit(X, y)
        self.support_ = self.selector_.support_
        self.ranking_ = self.selector_.ranking_
        return self

    def transform(self, X):
        if self.selector_ is None:
            return X
        return self.selector_.transform(X)

    def get_support(self, indices=False):
        if self.support_ is None:
            raise ValueError("Selector has to be fitted before calling get_support")
        if indices:
            return np.where(self.support_)[0]
        return self.support_


def classification_grouped_statistics_dataset_clinical():
    return run_classification_grouped_statistics(GROUPED_STATS_DATASET_CLINICAL_PATH)


def classification_grouped_statistics_dataset_clinical_acc():
    return run_classification_grouped_statistics(GROUPED_STATS_DATASET_CLINICAL_ACC_PATH)


def classification_grouped_statistics_with_covariates_dataset_clinical():
    return run_classification_grouped_statistics(
        GROUPED_STATS_DATASET_CLINICAL_PATH,
        include_diary_covariates=True,
        results_root=CLASSIFICATION_RESULTS_WITH_COVARIATES_ROOT,
    )


def classification_grouped_statistics_with_covariates_dataset_clinical_acc():
    return run_classification_grouped_statistics(
        GROUPED_STATS_DATASET_CLINICAL_ACC_PATH,
        include_diary_covariates=True,
        results_root=CLASSIFICATION_RESULTS_WITH_COVARIATES_ROOT,
    )


def classification_grouped_statistics_rfe_dataset_clinical():
    return run_classification_grouped_statistics(
        GROUPED_STATS_DATASET_CLINICAL_PATH,
        feature_selector_mode=FEATURE_SELECTOR_MODE_RFE,
    )


def classification_grouped_statistics_rfe_dataset_clinical_acc():
    return run_classification_grouped_statistics(
        GROUPED_STATS_DATASET_CLINICAL_ACC_PATH,
        feature_selector_mode=FEATURE_SELECTOR_MODE_RFE,
    )


def classification_grouped_statistics_with_covariates_rfe_dataset_clinical():
    return run_classification_grouped_statistics(
        GROUPED_STATS_DATASET_CLINICAL_PATH,
        include_diary_covariates=True,
        results_root=CLASSIFICATION_RESULTS_WITH_COVARIATES_ROOT,
        feature_selector_mode=FEATURE_SELECTOR_MODE_RFE,
    )


def classification_grouped_statistics_with_covariates_rfe_dataset_clinical_acc():
    return run_classification_grouped_statistics(
        GROUPED_STATS_DATASET_CLINICAL_ACC_PATH,
        include_diary_covariates=True,
        results_root=CLASSIFICATION_RESULTS_WITH_COVARIATES_ROOT,
        feature_selector_mode=FEATURE_SELECTOR_MODE_RFE,
    )


def classification_grouped_statistics_ablation_dataset_clinical():
    return run_classification_grouped_statistics_ablation(GROUPED_STATS_DATASET_CLINICAL_PATH)


def classification_grouped_statistics_ablation_dataset_clinical_acc():
    return run_classification_grouped_statistics_ablation(GROUPED_STATS_DATASET_CLINICAL_ACC_PATH)


def run_classification_grouped_statistics(
        grouped_stats_path,
        feature_block_key=FEATURE_BLOCK_ALL,
        include_diary_covariates=False,
        results_root=CLASSIFICATION_RESULTS_ROOT,
        feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST,
):
    grouped_stats_path = Path(grouped_stats_path)
    if not grouped_stats_path.exists():
        raise FileNotFoundError(
            f"Grouped statistics dataset not found: {grouped_stats_path}. "
            f"Run grouped clinical data first."
        )

    if feature_block_key not in FEATURE_BLOCKS:
        raise ValueError(
            f"Unknown feature block {feature_block_key}. "
            f"Available blocks: {', '.join(FEATURE_BLOCKS.keys())}"
        )
    if feature_selector_mode not in FEATURE_SELECTOR_MODES:
        raise ValueError(
            f"Unknown feature selector mode {feature_selector_mode}. "
            f"Available: {', '.join(FEATURE_SELECTOR_MODES)}"
        )

    dataset_name = grouped_stats_path.parents[1].name
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_dataset_name = (
        dataset_name
        if feature_selector_mode == FEATURE_SELECTOR_MODE_KBEST
        else f"{dataset_name}-{feature_selector_mode}"
    )
    run_dir = Path(results_root) / mode_dataset_name / feature_block_key / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting grouped-statistics classification for {dataset_name} "
        f"from {grouped_stats_path} using feature block {feature_block_key} "
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
    pca_output = _save_dataset_pca_projection(
        prepared_df=prepared_df,
        run_dir=run_dir,
        feature_block_key=feature_block_key,
    )

    _save_json(
        {
            "dataset_name": dataset_name,
            "source_path": str(grouped_stats_path),
            "run_dir": str(run_dir),
            "seed": SEED,
            "feature_block_key": feature_block_key,
            "feature_block_label": FEATURE_BLOCKS[feature_block_key]["label"],
            "include_diary_covariates": bool(include_diary_covariates),
            "feature_selector_mode": feature_selector_mode,
            "diary_covariates": _json_ready_dict(covariate_info),
            "feature_block_description": FEATURE_BLOCKS[feature_block_key]["description"],
            "available_feature_blocks": _json_ready_dict(FEATURE_BLOCKS),
            "stats_prefixes": list(STATS_PREFIXES),
            "feature_coverage_threshold": FEATURE_COVERAGE_THRESHOLD,
            "feature_selection": _feature_selection_metadata(feature_selector_mode),
            "tuning_cv_strategy": "StratifiedKFold",
            "tuning_cv_splits_cap": TUNING_MAX_CV_SPLITS,
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
            "search_settings": _json_ready_dict(_search_settings_for_selector_mode(feature_selector_mode)),
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
            feature_block_key=feature_block_key,
            feature_selector_mode=feature_selector_mode,
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
        "feature_block_key": feature_block_key,
        "feature_block_label": FEATURE_BLOCKS[feature_block_key]["label"],
        "feature_selector_mode": feature_selector_mode,
        "include_diary_covariates": bool(include_diary_covariates),
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "prepared_dataset_path": str(run_dir / "prepared_dataset.xlsx"),
        "pca_dir": str(pca_output["pca_dir"]),
    }


def run_classification_grouped_statistics_ablation(grouped_stats_path):
    grouped_stats_path = Path(grouped_stats_path)
    dataset_name = grouped_stats_path.parents[1].name
    run_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_dir = ABLATION_RESULTS_ROOT / dataset_name / run_label
    ablation_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting grouped-statistics feature-block ablation for {dataset_name} "
        f"from {grouped_stats_path}"
    )

    block_results = []
    for feature_block_key in FEATURE_BLOCKS.keys():
        result = run_classification_grouped_statistics(
            grouped_stats_path=grouped_stats_path,
            feature_block_key=feature_block_key,
            results_root=ablation_dir,
        )
        block_results.append(result)

    ablation_summary_path = _save_ablation_summary(
        ablation_dir=ablation_dir,
        block_results=block_results,
    )
    logger.info(
        f"Grouped-statistics feature-block ablation finished for {dataset_name}. "
        f"Results saved to {ablation_dir}"
    )
    return {
        "dataset_name": dataset_name,
        "run_dir": str(ablation_dir),
        "summary_path": str(ablation_summary_path),
        "block_results": block_results,
    }


def _save_ablation_summary(ablation_dir, block_results):
    default_rows = []
    tuned_rows = []
    block_rows = []

    for result in block_results:
        feature_block_key = result["feature_block_key"]
        summary_path = Path(result["summary_path"])
        if summary_path.exists():
            default_df = pd.read_excel(summary_path, sheet_name="default_metrics")
            tuned_df = pd.read_excel(summary_path, sheet_name="tuned_metrics")
            default_rows.append(default_df)
            tuned_rows.append(tuned_df)

        block_rows.append(
            {
                "feature_block_key": feature_block_key,
                "feature_block_label": result["feature_block_label"],
                "run_dir": result["run_dir"],
                "summary_path": result["summary_path"],
                "pca_dir": result["pca_dir"],
            }
        )

    summary_path = ablation_dir / "ablation_summary.xlsx"
    with pd.ExcelWriter(summary_path) as writer:
        if default_rows:
            pd.concat(default_rows, ignore_index=True).to_excel(
                writer,
                sheet_name="default_metrics",
                index=False,
            )
        if tuned_rows:
            pd.concat(tuned_rows, ignore_index=True).to_excel(
                writer,
                sheet_name="tuned_metrics",
                index=False,
            )
        pd.DataFrame(block_rows).to_excel(
            writer,
            sheet_name="block_runs",
            index=False,
        )
    return summary_path


def _prepare_dataset(df, include_diary_covariates=False):
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

    covariate_info = {
        "enabled": bool(include_diary_covariates),
        "columns_requested": list(DIARY_COVARIATE_COLUMNS),
        "rows_with_diary_data": 0,
        "subject_count_with_any_covariate": 0,
        "subject_count_total": int(prepared["#Subject"].nunique()),
        "non_missing_ratio": {},
    }
    covariate_columns = []
    if include_diary_covariates:
        covariates_df, covariate_summary = _subject_diary_covariates(
            prepared["#Subject"].astype(str).dropna().unique().tolist()
        )
        prepared = prepared.merge(covariates_df, on="#Subject", how="left")
        covariate_columns = [column for column in DIARY_COVARIATE_COLUMNS if column in prepared.columns]
        covariate_info.update(covariate_summary)

    prepared[stats_columns] = prepared[stats_columns].replace([np.inf, -np.inf], np.nan)
    ordered_columns = [
                          column
                          for column in (*IDENTITY_COLUMNS, TARGET_COLUMN, TARGET_LABEL_COLUMN)
                          if column in prepared.columns
                      ] + covariate_columns + stats_columns
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

    return prepared, excluded_labels_df, pd.DataFrame(dataset_overview_rows), covariate_info


def _subject_diary_covariates(subject_codes):
    subject_codes = [str(code) for code in subject_codes if str(code)]
    covariates_df = pd.DataFrame({"#Subject": sorted(set(subject_codes))})
    for column in DIARY_COVARIATE_COLUMNS:
        covariates_df[column] = np.nan

    rows = list(
        SleepDiaryDay.objects.filter(subject__code__in=subject_codes).values(
            "subject__code",
            "day_sleep_count",
            "day_sleep_time",
            "alcohol_count",
            "sleeping_pill",
            "caffeine_count",
            "sleep_quality",
            "rest_quality",
            "caffeine_time",
            "alcohol_time",
        )
    )
    if not rows:
        return covariates_df, {
            "rows_with_diary_data": 0,
            "subject_count_with_any_covariate": 0,
            "subject_count_total": int(len(covariates_df)),
            "non_missing_ratio": {column: 0.0 for column in DIARY_COVARIATE_COLUMNS},
        }

    diary_df = pd.DataFrame(rows)
    diary_df["day_sleep_count_num"] = pd.to_numeric(diary_df["day_sleep_count"], errors="coerce")
    diary_df["day_sleep_time_num"] = pd.to_numeric(diary_df["day_sleep_time"], errors="coerce")
    diary_df["alcohol_count_num"] = pd.to_numeric(diary_df["alcohol_count"], errors="coerce")
    diary_df["sleeping_pill_num"] = pd.to_numeric(diary_df["sleeping_pill"], errors="coerce")
    diary_df["caffeine_count_num"] = pd.to_numeric(diary_df["caffeine_count"], errors="coerce")
    diary_df["sleep_quality_num"] = pd.to_numeric(diary_df["sleep_quality"], errors="coerce")
    diary_df["rest_quality_num"] = pd.to_numeric(diary_df["rest_quality"], errors="coerce")
    diary_df["caffeine_time_min"] = diary_df["caffeine_time"].apply(_time_to_minutes)
    diary_df["alcohol_time_min"] = diary_df["alcohol_time"].apply(_time_to_minutes)

    aggregated = diary_df.groupby("subject__code", as_index=False).agg(
        day_sleep_count_mean=("day_sleep_count_num", "mean"),
        day_sleep_time_mean=("day_sleep_time_num", "mean"),
        alcohol_count_mean=("alcohol_count_num", "mean"),
        sleeping_pill_rate=("sleeping_pill_num", "mean"),
        caffeine_count_mean=("caffeine_count_num", "mean"),
        sleep_quality_mean=("sleep_quality_num", "mean"),
        rest_quality_mean=("rest_quality_num", "mean"),
        caffeine_time_mean=("caffeine_time_min", "mean"),
        alcohol_time_mean=("alcohol_time_min", "mean"),
        caffeine_time_std=("caffeine_time_min", "std"),
        alcohol_time_std=("alcohol_time_min", "std"),
    )
    aggregated = aggregated.rename(columns={"subject__code": "#Subject"})
    covariates_df = covariates_df.merge(aggregated, on="#Subject", how="left", suffixes=("", "_agg"))

    # Fill canonical columns from aggregated results and keep the expected names only.
    for column in DIARY_COVARIATE_COLUMNS:
        agg_column = f"{column}_agg"
        if agg_column in covariates_df.columns:
            covariates_df[column] = covariates_df[agg_column]
            covariates_df = covariates_df.drop(columns=[agg_column])

    any_covariate = covariates_df[list(DIARY_COVARIATE_COLUMNS)].notna().any(axis=1)
    summary = {
        "rows_with_diary_data": int(len(diary_df)),
        "subject_count_with_any_covariate": int(any_covariate.sum()),
        "subject_count_total": int(len(covariates_df)),
        "non_missing_ratio": {
            column: float(covariates_df[column].notna().mean())
            for column in DIARY_COVARIATE_COLUMNS
        },
    }
    return covariates_df, summary


def _time_to_minutes(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if hasattr(value, "hour"):
        return float(value.hour * 60 + value.minute + value.second / 60.0)

    parsed = pd.to_datetime(value, format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float(parsed.hour * 60 + parsed.minute + parsed.second / 60.0)


def _save_dataset_pca_projection(
        prepared_df,
        run_dir,
        feature_block_key=FEATURE_BLOCK_ALL,
):
    pca_dir = run_dir / "pca_projection"
    pca_dir.mkdir(parents=True, exist_ok=True)

    stats_columns = [
        column
        for column in prepared_df.columns
        if str(column).startswith(STATS_PREFIXES)
           and _feature_belongs_to_block(column, feature_block_key)
    ]
    if len(prepared_df) < 2 or len(stats_columns) < 2:
        logger.warning(
            f"Skipping PCA projection for {run_dir.name} [{feature_block_key}]: "
            f"need at least 2 subjects and 2 stats features in this block"
        )
        return {"pca_dir": pca_dir}

    pca_df = prepared_df.copy()
    pca_df["subject_prefix"] = (
        pca_df["#Subject"]
        .astype(str)
        .str.extract(r"^([A-Za-z]+(?:-[A-Za-z]+)?|[A-Za-z]+)", expand=False)
        .fillna("NO_PREFIX")
    )

    X = pca_df[stats_columns].replace([np.inf, -np.inf], np.nan)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=SEED)),
        ]
    )
    components = pipeline.fit_transform(X)
    pca_model = pipeline.named_steps["pca"]

    projection_df = pca_df[
        [
            column
            for column in (
            "#Subject",
            "#Age",
            "#Gender",
            "#Disease",
            TARGET_COLUMN,
            TARGET_LABEL_COLUMN,
            "subject_prefix",
        )
            if column in pca_df.columns
        ]
    ].copy()
    projection_df["PC1"] = components[:, 0]
    projection_df["PC2"] = components[:, 1]
    projection_df.to_excel(pca_dir / "pca_projection.xlsx", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "component": "PC1",
                "explained_variance_ratio": float(pca_model.explained_variance_ratio_[0]),
            },
            {
                "component": "PC2",
                "explained_variance_ratio": float(pca_model.explained_variance_ratio_[1]),
            },
        ]
    )
    summary_df.to_excel(pca_dir / "pca_summary.xlsx", index=False)

    _save_pca_scatter_plot(
        projection_df=projection_df,
        color_column=TARGET_LABEL_COLUMN,
        title=f"PCA projection colored by diagnosis ({feature_block_key})",
        explained_variance_ratio=pca_model.explained_variance_ratio_,
        output_path=pca_dir / "pca_by_diagnosis.png",
    )
    _save_pca_scatter_plot(
        projection_df=projection_df,
        color_column="subject_prefix",
        title=f"PCA projection colored by subject prefix ({feature_block_key})",
        explained_variance_ratio=pca_model.explained_variance_ratio_,
        output_path=pca_dir / "pca_by_subject_prefix.png",
    )

    logger.info(f"PCA projection diagnostics saved to {pca_dir}")
    return {
        "pca_dir": pca_dir,
        "projection_path": pca_dir / "pca_projection.xlsx",
        "summary_path": pca_dir / "pca_summary.xlsx",
        "diagnosis_plot_path": pca_dir / "pca_by_diagnosis.png",
        "prefix_plot_path": pca_dir / "pca_by_subject_prefix.png",
    }


def _save_pca_scatter_plot(
        projection_df,
        color_column,
        title,
        explained_variance_ratio,
        output_path,
):
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, group_df in projection_df.groupby(color_column, dropna=False):
        ax.scatter(
            group_df["PC1"],
            group_df["PC2"],
            s=36,
            alpha=0.8,
            label=str(label),
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}% var)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _run_scenario_analysis(
        prepared_df,
        positive_codes,
        negative_codes,
        run_dir,
        feature_block_key=FEATURE_BLOCK_ALL,
        feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST,
):
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
        feature_block_key=feature_block_key,
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
    if min(class_counts.values()) < 2:
        reason = "Need at least 2 subjects in each class for stratified tuning"
        logger.warning(f"Skipping {scenario_label}: {reason}")
        default_summary.update({"status": "skipped", "skip_reason": reason})
        tuned_summary.update({"status": "skipped", "skip_reason": reason})
        return {
            "default_summary": default_summary,
            "tuned_summary": tuned_summary,
        }

    stats_columns = [column for column in scenario_df.columns if str(column).startswith(STATS_PREFIXES)]
    stats_columns = [
        column
        for column in stats_columns
        if _feature_belongs_to_block(column, feature_block_key)
    ]
    covariate_columns = [column for column in DIARY_COVARIATE_COLUMNS if column in scenario_df.columns]
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

    _save_json(
        {"candidate_feature_labels": feature_columns},
        scenario_dir / "candidate_feature_labels.json",
    )
    np.save(scenario_dir / "X_original.npy", X)
    np.save(scenario_dir / "y_original.npy", y)

    best_estimator, random_search = _run_hyperparameter_search(
        X,
        y,
        feature_selector_mode=feature_selector_mode,
    )
    _save_pickle(best_estimator, scenario_dir / "trained_model.pkl")
    _save_json(
        _pipeline_params_for_json(best_estimator),
        scenario_dir / "trained_model_hyper_parameters.json",
    )
    selected_feature_columns = _selected_feature_columns(best_estimator, feature_columns)
    _save_json({"feature_labels": selected_feature_columns}, scenario_dir / "feature_labels.json")
    _save_selected_feature_scores(
        estimator=best_estimator,
        selected_feature_columns=selected_feature_columns,
        output_path=scenario_dir / "selected_feature_scores.xlsx",
        candidate_feature_columns=feature_columns,
    )
    pd.DataFrame(random_search.cv_results_).sort_values(
        by="rank_test_score"
    ).to_excel(scenario_dir / "hyperparameter_search_results.xlsx", index=False)

    feature_importance_df = _feature_importances_dataframe(
        best_estimator.named_steps["clf"],
        selected_feature_columns,
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
        feature_columns=selected_feature_columns,
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


def _prepare_scenario_features(scenario_df, stats_columns, additional_feature_columns=()):
    filtered = scenario_df.copy()
    stats_columns = [column for column in stats_columns if column in filtered.columns]
    additional_feature_columns = [
        column for column in additional_feature_columns if column in filtered.columns
    ]
    candidate_columns = list(dict.fromkeys(stats_columns + additional_feature_columns))
    if not candidate_columns:
        metadata_only = filtered.copy()
        return metadata_only, pd.DataFrame(), pd.DataFrame()

    numeric_features = filtered[candidate_columns].apply(pd.to_numeric, errors="coerce")
    coverage = numeric_features.notna().mean()

    kept_columns = []
    for column in stats_columns:
        if coverage.get(column, 0.0) >= FEATURE_COVERAGE_THRESHOLD:
            kept_columns.append(column)
    for column in additional_feature_columns:
        # Keep requested covariates whenever they contain at least one value for this scenario.
        if coverage.get(column, 0.0) > 0:
            kept_columns.append(column)

    numeric_after_coverage = numeric_features[kept_columns]
    keep_by_nonzero = numeric_after_coverage.fillna(0).abs().sum(axis=0) > 0
    kept_columns = keep_by_nonzero[keep_by_nonzero].index.tolist()

    feature_mapping_rows = []
    coverage_rows = []
    for column in candidate_columns:
        display_name = _feature_display_name(column)
        kept = column in kept_columns
        is_covariate = column in additional_feature_columns
        coverage_rows.append(
            {
                "original_feature": column,
                "display_feature": display_name,
                "feature_source": "covariate" if is_covariate else "statistics",
                "non_missing_ratio": float(coverage.get(column, 0)),
                "kept": kept,
                "drop_reason": (
                    ""
                    if kept
                    else "all_missing"
                    if is_covariate and coverage.get(column, 0) <= 0
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
                    "feature_source": "covariate" if is_covariate else "statistics",
                }
            )

    metadata_columns = [column for column in filtered.columns if column not in candidate_columns]
    filtered = filtered[metadata_columns + kept_columns].copy()
    filtered = filtered.rename(columns={column: _feature_display_name(column) for column in kept_columns})

    return (
        filtered,
        pd.DataFrame(feature_mapping_rows),
        pd.DataFrame(coverage_rows),
    )


def _run_hyperparameter_search(X, y, feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST):
    cv = _build_tuning_cv(y)
    search_settings = _search_settings_for_selector_mode(feature_selector_mode)
    search = RandomizedSearchCV(
        estimator=_build_pipeline_with_selector(feature_selector_mode),
        cv=cv,
        **search_settings,
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


def _search_settings_for_selector_mode(feature_selector_mode):
    if feature_selector_mode not in FEATURE_SELECTOR_MODES:
        raise ValueError(
            f"Unknown feature selector mode {feature_selector_mode}. "
            f"Available: {', '.join(FEATURE_SELECTOR_MODES)}"
        )

    search_settings = SEARCH_SETTINGS.copy()
    param_distributions = {
        key: value
        for key, value in SEARCH_SETTINGS.get("param_distributions", {}).items()
    }

    if feature_selector_mode == FEATURE_SELECTOR_MODE_RFE:
        param_distributions.pop("feature_selector__k", None)
        param_distributions["feature_selector__n_features_to_select"] = list(FEATURE_SELECTION_RFE_OPTIONS)
        param_distributions["feature_selector__step"] = [0.05, 0.1, 0.2]
        search_settings["n_iter"] = RFE_TUNING_N_ITER
    else:
        param_distributions["feature_selector__k"] = list(FEATURE_SELECTION_K_OPTIONS)

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
            "search_iterations": RFE_TUNING_N_ITER,
            "note": (
                "RFE is tuned inside CV; n_features_to_select is clipped "
                "if larger than current feature count."
            ),
        }
    return {
        "pipeline_steps": [
            "median imputation",
            "constant-feature removal",
            "min-max scaling",
            "ANOVA SelectKBest",
        ],
        "k_options": list(FEATURE_SELECTION_K_OPTIONS),
        "k_note": "k is tuned inside CV; when k exceeds current feature count it is clipped safely.",
    }


def _build_tuning_cv(y):
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError(
            "Stratified tuning requires at least 2 samples in each class, "
            f"got {class_counts.to_dict()}"
        )
    n_splits = min(TUNING_MAX_CV_SPLITS, min_class_count)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


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
    return _build_pipeline_with_selector(FEATURE_SELECTOR_MODE_KBEST)


def _build_pipeline_with_selector(feature_selector_mode=FEATURE_SELECTOR_MODE_KBEST):
    if feature_selector_mode not in FEATURE_SELECTOR_MODES:
        raise ValueError(
            f"Unknown feature selector mode {feature_selector_mode}. "
            f"Available: {', '.join(FEATURE_SELECTOR_MODES)}"
        )

    feature_selector = (
        AdaptiveSelectKBest(k="all")
        if feature_selector_mode == FEATURE_SELECTOR_MODE_KBEST
        else AdaptiveRFESelector(n_features_to_select=40, step=0.1)
    )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_filter", VarianceThreshold(threshold=0.0)),
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            ("feature_selector", feature_selector),
            ("clf", xgb.XGBClassifier(**_resolved_model_params())),
        ]
    )


def _resolved_model_params():
    params = MODEL_PARAMS.copy()
    params["device"] = _default_xgb_device()
    return params


def _build_rfe_estimator():
    ranking_params = _resolved_model_params().copy()
    ranking_params.update(
        {
            "booster": "gbtree",
            "n_estimators": max(80, min(160, int(ranking_params.get("n_estimators", 100)))),
            "max_depth": min(5, int(ranking_params.get("max_depth", 10))),
            "learning_rate": min(0.2, float(ranking_params.get("learning_rate", 0.2))),
            "subsample": min(0.9, float(ranking_params.get("subsample", 1.0))),
            "colsample_bytree": min(0.9, float(ranking_params.get("colsample_bytree", 1.0))),
            "gamma": min(0.2, float(ranking_params.get("gamma", 1.0))),
            "min_child_weight": max(1.0, min(5.0, float(ranking_params.get("min_child_weight", 5.0)))),
        }
    )
    return xgb.XGBClassifier(**ranking_params)


def _default_xgb_device():
    requested = os.environ.get("GENEACTIV_XGB_DEVICE")
    if requested:
        return requested
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def _scenario_label(positive_codes, negative_codes):
    return f"scenario-{_codes_to_label(positive_codes)}_vs_{_codes_to_label(negative_codes)}"


def _codes_to_label(codes):
    return "+".join(LABEL_MAPPING[code] for code in codes)


def _base_summary_row(
        scenario_label,
        feature_block_key,
        positive_codes,
        negative_codes,
        subject_count,
        positive_count,
        negative_count,
):
    return {
        "scenario": scenario_label,
        "feature_block_key": feature_block_key,
        "feature_block_label": FEATURE_BLOCKS[feature_block_key]["label"],
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


def _feature_belongs_to_block(feature_name, feature_block_key):
    if feature_block_key == FEATURE_BLOCK_ALL:
        return True

    raw_name = str(feature_name)
    stat_prefix, payload = _split_stats_feature_name(raw_name)
    if not payload:
        return False

    if feature_block_key == "diary-only":
        return payload.startswith(("diary.", "diary_norm."))
    if feature_block_key == "actigraphy-only":
        return payload.startswith(("actigraphy.", "actigraphy_norm."))
    if feature_block_key == "activity-only":
        return payload.startswith("activity.")
    if feature_block_key == "norm-only":
        return "_norm." in payload
    if feature_block_key == "non-norm-only":
        return "_norm." not in payload
    if feature_block_key == "level-only":
        return stat_prefix in {"Mean.", "Median.", "Min.", "Max."}
    if feature_block_key == "trend-only":
        return stat_prefix == "Slope."
    if feature_block_key == "variability-only":
        return stat_prefix in {"SD.", "MAD.", "Range.", "IQR.", "CV."}

    raise ValueError(
        f"Unknown feature block {feature_block_key}. "
        f"Available blocks: {', '.join(FEATURE_BLOCKS.keys())}"
    )


def _split_stats_feature_name(feature_name):
    for prefix in STATS_PREFIXES:
        if str(feature_name).startswith(prefix):
            return prefix, str(feature_name)[len(prefix):]
    return "", str(feature_name)


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


def _selected_feature_columns(estimator, feature_columns):
    selected_columns = np.array(feature_columns, dtype=object)
    for step_name in ("variance_filter", "feature_selector"):
        step = estimator.named_steps.get(step_name)
        if step is None or not hasattr(step, "get_support"):
            continue
        selected_columns = selected_columns[step.get_support()]
    return selected_columns.tolist()


def _save_selected_feature_scores(
        estimator,
        selected_feature_columns,
        output_path,
        candidate_feature_columns=None,
):
    selector = estimator.named_steps.get("feature_selector")
    if selector is None:
        pd.DataFrame(columns=["feature", "score", "p_value", "rank", "selected"]).to_excel(
            output_path, index=False
        )
        return

    if candidate_feature_columns is None:
        candidate_feature_columns = selected_feature_columns
    feature_after_variance = np.array(candidate_feature_columns, dtype=object)
    variance_filter = estimator.named_steps.get("variance_filter")
    if variance_filter is not None and hasattr(variance_filter, "get_support"):
        variance_mask = variance_filter.get_support()
        if len(variance_mask) == len(feature_after_variance):
            feature_after_variance = feature_after_variance[variance_mask]

    selected_mask = selector.get_support()
    if len(selected_mask) != len(feature_after_variance):
        feature_after_variance = np.array(
            [f"feature_{idx}" for idx in range(len(selected_mask))],
            dtype=object,
        )

    if hasattr(selector, "scores_") and selector.scores_ is not None:
        selected_feature_names = feature_after_variance[selected_mask]
        score_df = pd.DataFrame(
            {
                "feature": selected_feature_names,
                "score": np.asarray(selector.scores_)[selected_mask],
                "p_value": np.asarray(selector.pvalues_)[selected_mask],
                "rank": 1,
                "selected": True,
            }
        ).sort_values(by="score", ascending=False).reset_index(drop=True)
        score_df.to_excel(output_path, index=False)
        return

    if hasattr(selector, "ranking_") and selector.ranking_ is not None:
        ranking_df = pd.DataFrame(
            {
                "feature": feature_after_variance,
                "rank": np.asarray(selector.ranking_, dtype=int),
                "selected": np.asarray(selected_mask, dtype=bool),
            }
        ).sort_values(by=["rank", "feature"]).reset_index(drop=True)
        ranking_df.to_excel(output_path, index=False)
        return

    pd.DataFrame(columns=["feature", "score", "p_value", "rank", "selected"]).to_excel(
        output_path, index=False
    )


def _safe_f_classif(X, y):
    scores, p_values = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
    return scores, p_values


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
    transformed = estimator[:-1].transform(X)
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

        min_max_evals = max(2 * len(feature_columns) + 1, 500)
        logger.info(
            f"Using SHAP generic explainer fallback with max_evals={min_max_evals} "
            f"for {len(feature_columns)} features"
        )
        explainer = shap.Explainer(
            predict_positive,
            transformed_df,
            max_evals=min_max_evals,
        )
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


def _pipeline_params_for_json(estimator):
    if hasattr(estimator, "named_steps"):
        return {
            step_name: (
                _json_ready_dict(step.get_params())
                if hasattr(step, "get_params")
                else repr(step)
            )
            for step_name, step in estimator.named_steps.items()
        }
    if hasattr(estimator, "get_params"):
        return _json_ready_dict(estimator.get_params())
    return _json_ready_dict(estimator)


def _json_ready_dict(data):
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    if isinstance(data, dict):
        return {str(key): _json_ready_dict(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_ready_dict(value) for value in data]
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, Path):
        return str(data)
    return repr(data)


def _is_gpu_error(exc):
    message = str(exc).lower()
    return any(keyword in message for keyword in ("cuda", "gpu", "device"))
