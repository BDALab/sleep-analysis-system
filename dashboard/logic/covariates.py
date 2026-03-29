import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

from mysite.settings import BASE_DIR, MEDIA_ROOT

logger = logging.getLogger(__name__)

DATASET_CLINICAL_PATH = Path(BASE_DIR) / "dataset-clinical.xlsx"
DATASET_CLINICAL_ACC_DREAMT_PATH = Path(BASE_DIR) / "dataset-clinical-acc.xlsx"
DEFAULT_COVARIATES = ("gender", "age")
CORRELATION_TYPES = ("pearson", "spearman", "kendall")
EXCLUDED_NORM_COLUMNS = (
    "actigraphy_norm.Sleep onset latency",
    "actigraphy_norm.Wake after sleep onset",
    "actigraphy_norm.Awakening > 5 minutes",
    "actigraphy_norm.Sleep efficiency",
    "diary_norm.Sleep onset latency",
    "diary_norm.Wake after sleep onset",
    "diary_norm.Awakening > 5 minutes",
    "diary_norm.Sleep efficiency",
    "rbdsq.RBDSQ",
)


class CovariateController(BaseEstimator, TransformerMixin):
    def __init__(self, inline=False):
        self.inline = inline
        self.regressors = {}
        self.covariates = None

    def fit(self, X, y, **params):
        assert isinstance(X, pd.DataFrame), f"X must be pandas dataframe, got {type(X)}"
        assert isinstance(y, pd.DataFrame), f"y must be pandas dataframe, got {type(y)}"
        assert X.shape[0] == y.shape[0], f"X and y size mismatch: {X.shape[0]} vs {y.shape[0]}"

        self.regressors = {
            column: LinearRegression(**params).fit(y.values, X[column].values)
            for column in X.columns
        }
        self.covariates = y
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), f"X must be pandas dataframe, got {type(X)}"

        transformed = X if self.inline else X.copy()
        for column in transformed.columns:
            transformed[column] = (
                    transformed[column].values
                    - self.regressors[column].predict(self.covariates.values)
            )
        return transformed


def calculate_covariates_dataset_clinical():
    return calculate_covariates_for_excel(DATASET_CLINICAL_PATH)


def calculate_covariates_dataset_clinical_acc_dreamt():
    return calculate_covariates_for_excel(DATASET_CLINICAL_ACC_DREAMT_PATH)


def calculate_covariates_for_excel(source_path, covariates=DEFAULT_COVARIATES):
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    logger.info(f"Calculating controlled covariates for {source_path.name}")
    df = pd.read_excel(source_path, index_col=0)
    df = _prepare_dataset(df)

    output_dir = Path(MEDIA_ROOT) / "covariates" / source_path.stem
    data_dir = output_dir / "data"
    results_dir = output_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    computation_settings = _build_computation_settings()
    _save_covariate_correlations(
        df=df,
        computation_settings=computation_settings,
        selected_covariates=covariates,
        output_dir=results_dir,
    )

    fieldnames_covariates = [
        setting["fieldname"]
        for setting in computation_settings
        if setting["scale"] in covariates
    ]
    df_covariates = df[fieldnames_covariates].fillna(0)
    df_features = df[[column for column in df.columns if not column.startswith("#")]].fillna(0)
    df_feat_nocovars = CovariateController().fit_transform(df_features, df_covariates)

    feat_data_columns = df_feat_nocovars.columns.to_list()
    metadata_columns = [column for column in ("#Subject", "#Date", "#Age", "#Gender", "#Disease") if
                        column in df.columns]
    merged_df = pd.merge(
        df_feat_nocovars,
        df[metadata_columns],
        left_index=True,
        right_index=True,
        how="inner",
    )
    feature_matrix_df = merged_df[feat_data_columns + ["#Disease"]]

    feature_matrix_path = data_dir / "feature_matrix.xlsx"
    controlled_path = data_dir / f"feature_matrix_controlled_{'_'.join(covariates)}.xlsx"
    merged_path = data_dir / "clinical_data.xlsx"

    feature_matrix_df.to_excel(feature_matrix_path, index=True)
    df_feat_nocovars.to_excel(controlled_path, index=True)
    merged_df.to_excel(merged_path, index=True)

    logger.info(
        f"Controlled covariates calculated for {source_path.name}. "
        f"Outputs saved to {data_dir}"
    )
    return {
        "source": str(source_path),
        "output_dir": str(output_dir),
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "feature_matrix_path": str(feature_matrix_path),
        "controlled_features_path": str(controlled_path),
        "merged_path": str(merged_path),
    }


def _prepare_dataset(df):
    prepared = df.copy()
    if "#Gender" in prepared.columns:
        prepared["#Gender"] = prepared["#Gender"].apply(_normalize_gender)

    numeric_candidates = [column for column in prepared.columns if column.startswith("#") or not column.startswith("#")]
    for column in numeric_candidates:
        if column in ("#Subject", "#Date"):
            continue
        try:
            prepared[column] = pd.to_numeric(prepared[column])
        except (ValueError, TypeError):
            continue

    return prepared


def _normalize_gender(value):
    if pd.isna(value):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized == "F":
            return 1
        if normalized == "M":
            return 2
    return value


def _build_computation_settings():
    return [
        {
            "scale": "age",
            "fieldname": "#Age",
            "excluded": (
                "#Subject",
                "#Date",
                "#Disease",
                "#Gender",
                *EXCLUDED_NORM_COLUMNS,
            ),
            "correlation": CORRELATION_TYPES,
        },
        {
            "scale": "gender",
            "fieldname": "#Gender",
            "excluded": (
                "#Subject",
                "#Date",
                "#Disease",
                "#Age",
                *EXCLUDED_NORM_COLUMNS,
            ),
            "correlation": CORRELATION_TYPES,
        },
    ]


def _save_covariate_correlations(df, computation_settings, selected_covariates, output_dir):
    for setting in computation_settings:
        if setting["scale"] not in selected_covariates:
            continue

        results = []
        for feature in df.columns:
            if feature in setting["excluded"]:
                continue

            clin_data = pd.to_numeric(df[setting["fieldname"]], errors="coerce")
            feat_data = pd.to_numeric(df[feature], errors="coerce")
            valid = ~(clin_data.isna() | feat_data.isna())
            if valid.sum() < 3:
                continue

            correlations = {"feature": feature}
            for corr_type in setting["correlation"]:
                r_value, p_value = _compute_correlation(
                    feat_data.loc[valid].values,
                    clin_data.loc[valid].values,
                    corr_type=corr_type,
                )
                correlations[f"r ({corr_type})"] = round(float(r_value), 4) if pd.notna(r_value) else np.nan
                correlations[f"p ({corr_type})"] = round(float(p_value), 4) if pd.notna(p_value) else np.nan
            results.append(correlations)

        pd.DataFrame(results).to_excel(
            output_dir / f"corr_covars_{setting['scale']}.xlsx",
            index=False,
        )


def _compute_correlation(x, y, corr_type):
    try:
        if corr_type == "pearson":
            return pearsonr(x, y)
        if corr_type == "spearman":
            return spearmanr(x, y)
        if corr_type == "kendall":
            return kendalltau(x, y)
    except Exception:
        logger.warning(f"Failed to compute {corr_type} correlation", exc_info=True)
    return np.nan, np.nan
