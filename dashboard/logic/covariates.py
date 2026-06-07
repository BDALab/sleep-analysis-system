import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kendalltau, pearsonr, spearmanr, ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

from dashboard.models import Subject
from mysite.settings import BASE_DIR, MEDIA_ROOT

logger = logging.getLogger(__name__)

DATASET_CLINICAL_PATH = Path(BASE_DIR) / "dataset-clinical.xlsx"
DATASET_CLINICAL_ACC_DREAMT_PATH = Path(BASE_DIR) / "dataset-clinical-acc.xlsx"
CANDIDATE_COVARIATES = ("age", "gender", "education")
COVARIATE_SCENARIOS = (
    ((3,), (0,)),
    ((3, 2), (0,)),
    ((2,), (0,)),
)
COVARIATE_ALPHA = 0.05
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


def calculate_covariates_for_excel(source_path, covariates=None):
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    logger.info(f"Calculating controlled covariates for {source_path.name}")
    df = pd.read_excel(source_path, index_col=0)
    df = _attach_subject_covariates(df)
    df = _prepare_dataset(df)

    output_dir = Path(MEDIA_ROOT) / "covariates" / source_path.stem
    data_dir = output_dir / "data"
    results_dir = output_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    verification = verify_covariates_for_excel(
        source_path,
        alpha=COVARIATE_ALPHA,
        output_dir=results_dir,
    )
    selected_covariates = (
        tuple(covariates)
        if covariates is not None
        else tuple(verification["selected_covariates"])
    )

    computation_settings = _build_computation_settings()
    _save_covariate_correlations(
        df=df,
        computation_settings=computation_settings,
        selected_covariates=selected_covariates,
        output_dir=results_dir,
    )

    fieldnames_covariates = [
        setting["fieldname"]
        for setting in computation_settings
        if setting["scale"] in selected_covariates
    ]
    df_features = df[[column for column in df.columns if not column.startswith("#")]].fillna(0)
    if fieldnames_covariates:
        df_covariates = _impute_covariates(df[fieldnames_covariates])
        df_feat_nocovars = CovariateController().fit_transform(df_features, df_covariates)
    else:
        df_feat_nocovars = df_features.copy()

    feat_data_columns = df_feat_nocovars.columns.to_list()
    metadata_columns = [column for column in ("#Subject", "#Date", "#Age", "#Gender", "#Education", "#Disease") if
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
    covariate_suffix = "_".join(selected_covariates) if selected_covariates else "none"
    controlled_path = data_dir / f"feature_matrix_controlled_{covariate_suffix}.xlsx"
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
        "selected_covariates": list(selected_covariates),
        "verification_path": verification["output_path"],
        "verification": verification,
    }


def verify_covariates_for_excel(source_path, alpha=COVARIATE_ALPHA, output_dir=None):
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    source_df = pd.read_excel(source_path)
    if "#Subject" not in source_df.columns:
        raise KeyError(f"Dataset {source_path.name} is missing #Subject")

    subject_codes = (
        source_df["#Subject"].dropna().astype(str).str.strip().drop_duplicates().tolist()
    )
    subject_rows = list(
        Subject.objects.filter(code__in=subject_codes).values(
            "code",
            "age",
            "sex",
            "education_years",
            "diagnosis_code",
        )
    )
    subject_df = pd.DataFrame(subject_rows).rename(
        columns={
            "code": "#Subject",
            "age": "age",
            "sex": "gender",
            "education_years": "education",
        }
    )

    test_rows = []
    group_rows = []
    for positive_codes, negative_codes in COVARIATE_SCENARIOS:
        scenario_label = _scenario_label(positive_codes, negative_codes)
        scenario_codes = set(positive_codes) | set(negative_codes)
        scenario_df = subject_df[subject_df["diagnosis_code"].isin(scenario_codes)].copy()
        scenario_df["group"] = scenario_df["diagnosis_code"].apply(
            lambda value: "positive" if value in positive_codes else "negative"
        )

        for covariate in ("age", "education"):
            test_rows.append(
                _welch_test_row(
                    scenario_df=scenario_df,
                    scenario_label=scenario_label,
                    covariate=covariate,
                    alpha=alpha,
                )
            )

        sex_test, sex_groups = _chi_squared_sex_rows(
            scenario_df=scenario_df,
            scenario_label=scenario_label,
            alpha=alpha,
        )
        test_rows.append(sex_test)

        for group_name in ("negative", "positive"):
            group_df = scenario_df[scenario_df["group"] == group_name]
            row = {
                "scenario": scenario_label,
                "group": group_name,
                "n_subjects": int(len(group_df)),
                "age_n": int(group_df["age"].notna().sum()),
                "age_mean": _safe_mean(group_df["age"]),
                "age_sd": _safe_std(group_df["age"]),
                "education_n": int(group_df["education"].notna().sum()),
                "education_mean": _safe_mean(group_df["education"]),
                "education_sd": _safe_std(group_df["education"]),
            }
            row.update(sex_groups.get(group_name, {}))
            group_rows.append(row)

    tests_df = pd.DataFrame(test_rows)
    groups_df = pd.DataFrame(group_rows)
    selected_covariates = [
        covariate
        for covariate in CANDIDATE_COVARIATES
        if bool(
            tests_df.loc[
                (tests_df["covariate"] == covariate)
                & (tests_df["control_recommended"]),
                "control_recommended",
            ].any()
        )
    ]
    scenario_covariates = [
        {
            "scenario": scenario_label,
            "selected_covariates": [
                covariate
                for covariate in CANDIDATE_COVARIATES
                if bool(
                    tests_df.loc[
                        (tests_df["scenario"] == scenario_label)
                        & (tests_df["covariate"] == covariate)
                        & (tests_df["control_recommended"]),
                        "control_recommended",
                    ].any()
                )
            ],
        }
        for scenario_label in tests_df["scenario"].drop_duplicates().tolist()
    ]

    if output_dir is None:
        output_dir = Path(MEDIA_ROOT) / "covariates" / source_path.stem / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "covariate_verification.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        tests_df.to_excel(writer, sheet_name="tests", index=False)
        groups_df.to_excel(writer, sheet_name="group_descriptives", index=False)
        pd.DataFrame(
            {
                "setting": [
                    "alpha",
                    "selection_rule",
                    "selected_covariates_union",
                    *[
                        f"selected_covariates_{item['scenario']}"
                        for item in scenario_covariates
                    ],
                ],
                "value": [
                    alpha,
                    "Control separately within each scenario when p < alpha",
                    ", ".join(selected_covariates) or "none",
                    *[
                        ", ".join(item["selected_covariates"]) or "none"
                        for item in scenario_covariates
                    ],
                ],
            }
        ).to_excel(writer, sheet_name="settings", index=False)

    return {
        "source": str(source_path),
        "output_path": str(output_path),
        "alpha": alpha,
        "subject_count": int(subject_df["#Subject"].nunique()) if not subject_df.empty else 0,
        "selected_covariates": selected_covariates,
        "scenario_covariates": scenario_covariates,
        "tests": tests_df.replace({np.nan: None}).to_dict("records"),
        "group_descriptives": groups_df.replace({np.nan: None}).to_dict("records"),
    }


def verify_all_covariate_datasets():
    return [
        verify_covariates_for_excel(DATASET_CLINICAL_PATH),
        verify_covariates_for_excel(DATASET_CLINICAL_ACC_DREAMT_PATH),
    ]


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


def _attach_subject_covariates(df):
    if "#Subject" not in df.columns:
        raise KeyError("Dataset is missing #Subject")

    enriched = df.copy()
    subject_codes = enriched["#Subject"].dropna().astype(str).str.strip().unique().tolist()
    education_by_subject = {
        row["code"]: row["education_years"]
        for row in Subject.objects.filter(code__in=subject_codes).values(
            "code",
            "education_years",
        )
    }
    imported_education = enriched["#Subject"].astype(str).str.strip().map(education_by_subject)
    if "#Education" in enriched.columns:
        existing_education = pd.to_numeric(enriched["#Education"], errors="coerce")
        enriched["#Education"] = existing_education.combine_first(imported_education)
    else:
        enriched["#Education"] = imported_education
    return enriched


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


def _impute_covariates(df):
    imputed = df.copy()
    for column in imputed.columns:
        numeric = pd.to_numeric(imputed[column], errors="coerce")
        if numeric.notna().any():
            imputed[column] = numeric.fillna(numeric.median())
        else:
            raise ValueError(f"Selected covariate {column} has no usable values")
    return imputed


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
                "#Education",
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
                "#Education",
                *EXCLUDED_NORM_COLUMNS,
            ),
            "correlation": CORRELATION_TYPES,
        },
        {
            "scale": "education",
            "fieldname": "#Education",
            "excluded": (
                "#Subject",
                "#Date",
                "#Disease",
                "#Age",
                "#Gender",
                *EXCLUDED_NORM_COLUMNS,
            ),
            "correlation": CORRELATION_TYPES,
        },
    ]


def _welch_test_row(scenario_df, scenario_label, covariate, alpha):
    negative = pd.to_numeric(
        scenario_df.loc[scenario_df["group"] == "negative", covariate],
        errors="coerce",
    ).dropna()
    positive = pd.to_numeric(
        scenario_df.loc[scenario_df["group"] == "positive", covariate],
        errors="coerce",
    ).dropna()

    statistic = np.nan
    p_value = np.nan
    if len(negative) >= 2 and len(positive) >= 2:
        statistic, p_value = ttest_ind(negative, positive, equal_var=False)

    return {
        "scenario": scenario_label,
        "covariate": covariate,
        "test": "Welch t-test",
        "negative_n": int(len(negative)),
        "positive_n": int(len(positive)),
        "statistic": _safe_float(statistic),
        "p_value": _safe_float(p_value),
        "alpha": alpha,
        "control_recommended": bool(pd.notna(p_value) and p_value < alpha),
        "note": "",
    }


def _chi_squared_sex_rows(scenario_df, scenario_label, alpha):
    valid = scenario_df[scenario_df["gender"].isin(("F", "M"))].copy()
    contingency = pd.crosstab(valid["group"], valid["gender"]).reindex(
        index=["negative", "positive"],
        columns=["F", "M"],
        fill_value=0,
    )

    statistic = np.nan
    p_value = np.nan
    expected_min = np.nan
    note = ""
    if (contingency.sum(axis=1) > 0).all() and (contingency.sum(axis=0) > 0).all():
        statistic, p_value, _, expected = chi2_contingency(
            contingency.values,
            correction=False,
        )
        expected_min = float(np.min(expected))
        if expected_min < 5:
            note = "At least one expected cell count is below 5; interpret chi-squared cautiously."
    else:
        note = "Chi-squared test requires both groups and both sex categories."

    group_counts = {
        group: {
            "female_n": int(contingency.loc[group, "F"]),
            "male_n": int(contingency.loc[group, "M"]),
        }
        for group in contingency.index
    }
    return {
        "scenario": scenario_label,
        "covariate": "gender",
        "test": "Pearson chi-squared",
        "negative_n": int(contingency.loc["negative"].sum()),
        "positive_n": int(contingency.loc["positive"].sum()),
        "statistic": _safe_float(statistic),
        "p_value": _safe_float(p_value),
        "alpha": alpha,
        "control_recommended": bool(pd.notna(p_value) and p_value < alpha),
        "note": note,
        "expected_min": _safe_float(expected_min),
    }, group_counts


def _scenario_label(positive_codes, negative_codes):
    labels = dict(Subject.DIAGNOSIS_CODE)
    positive = "+".join(labels[code] for code in positive_codes)
    negative = "+".join(labels[code] for code in negative_codes)
    return f"{positive} vs {negative}"


def _safe_float(value):
    return float(value) if pd.notna(value) else None


def _safe_mean(values):
    numeric = pd.to_numeric(values, errors="coerce")
    return _safe_float(numeric.mean())


def _safe_std(values):
    numeric = pd.to_numeric(values, errors="coerce")
    return _safe_float(numeric.std(ddof=1))


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
