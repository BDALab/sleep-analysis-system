import logging
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.stats import mannwhitneyu, shapiro, spearmanr

from dashboard.models import Subject
from mysite.settings import MEDIA_ROOT

logger = logging.getLogger(__name__)

GROUPED_DATASET_CLINICAL_PATH = (
        Path(MEDIA_ROOT)
        / "covariates"
        / "dataset-clinical"
        / "data"
        / "grouped_clinical_matrix_with_stats.xlsx"
)
GROUPED_DATASET_CLINICAL_ACC_PATH = (
        Path(MEDIA_ROOT)
        / "covariates"
        / "dataset-clinical-acc"
        / "data"
        / "grouped_clinical_matrix_with_stats.xlsx"
)
RESULTS_ROOT = Path(MEDIA_ROOT) / "feature-correlation-analysis"

HC_CODE = 0
MCI_CODE = 2
PRE_DLB_CODE = 3
ALPHA = 0.05
ANALYSIS_SCENARIOS = (
    {
        "key": "predlb-vs-hc",
        "label": "preDLB vs HC",
        "reference_label": "HC",
        "reference_codes": (HC_CODE,),
        "case_label": "preDLB",
        "case_codes": (PRE_DLB_CODE,),
    },
    {
        "key": "predlb-mci-vs-hc",
        "label": "preDLB + MCI-AD vs HC",
        "reference_label": "HC",
        "reference_codes": (HC_CODE,),
        "case_label": "preDLB + MCI-AD",
        "case_codes": (PRE_DLB_CODE, MCI_CODE),
    },
    {
        "key": "mci-vs-hc",
        "label": "MCI-AD vs HC",
        "reference_label": "HC",
        "reference_codes": (HC_CODE,),
        "case_label": "MCI-AD",
        "case_codes": (MCI_CODE,),
    },
)
STATISTIC_PREFIXES = (
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
CLINICAL_OUTCOMES = {
    "RBDq": "rbdq",
    "UPDRS": "updrs",
    "MFS": "mfs",
    "Visuospatial": "visuospatial",
    "Attention": "attention",
    "Executive": "executive",
}
CORRELATION_COLUMNS = (
    "Features",
    "Clinical outcome",
    "n",
    "Spearman rho",
    "p",
    "adj p",
)


def analyze_all_feature_datasets():
    results = []
    for source_path, dataset_name in (
            (GROUPED_DATASET_CLINICAL_PATH, "dataset-clinical"),
            (GROUPED_DATASET_CLINICAL_ACC_PATH, "dataset-clinical-acc"),
    ):
        for scenario in ANALYSIS_SCENARIOS:
            results.append(
                analyze_feature_dataset(
                    source_path,
                    dataset_name=dataset_name,
                    scenario=scenario,
                )
            )
    return results


def analyze_feature_dataset(
        source_path,
        dataset_name=None,
        scenario=None,
        output_path=None,
):
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Grouped feature dataset not found: {source_path}")

    dataset_name = dataset_name or source_path.stem
    scenario = scenario or ANALYSIS_SCENARIOS[-1]
    output_path = (
        Path(output_path)
        if output_path
        else (
                RESULTS_ROOT
                / dataset_name
                / scenario["key"]
                / "feature_clinical_correlation_matrix.xlsx"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped_df = pd.read_excel(source_path)
    if "#Subject" not in grouped_df.columns:
        raise KeyError(f"Dataset {source_path.name} is missing #Subject")

    analysis_df = _prepare_subject_level_data(grouped_df, scenario)
    feature_columns = [
        column
        for column in analysis_df.columns
        if str(column).startswith(STATISTIC_PREFIXES)
    ]
    if not feature_columns:
        raise ValueError(f"No feature columns found in {source_path}")

    result_rows = []
    normality_rows = []
    for feature in feature_columns:
        row, normality = _analyze_feature(analysis_df, feature, scenario)
        result_rows.append(row)
        normality_rows.extend(normality)

    results_df = pd.DataFrame(result_rows)
    output_columns = _group_output_columns(scenario)
    results_df["adj p"] = _benjamini_hochberg(
        results_df[output_columns[-2]]
    )
    results_df = results_df[output_columns]

    correlations_df = _analyze_correlations(analysis_df, feature_columns)
    correlation_matrices = {
        "correlation_rho": _correlation_matrix(correlations_df, "Spearman rho"),
        "correlation_p": _correlation_matrix(correlations_df, "p"),
        "correlation_adj_p": _correlation_matrix(correlations_df, "adj p"),
        "correlation_n": _correlation_matrix(correlations_df, "n"),
    }
    normality_df = pd.DataFrame(normality_rows)
    significant_group_df = results_df[
        pd.to_numeric(results_df["adj p"], errors="coerce") < ALPHA
        ].copy()
    significant_correlations_df = correlations_df[
        pd.to_numeric(correlations_df["adj p"], errors="coerce") < ALPHA
        ].copy()
    significant_by_outcome = {
        outcome: int(
            (
                    (correlations_df["Clinical outcome"] == outcome)
                    & (pd.to_numeric(correlations_df["adj p"], errors="coerce") < ALPHA)
            ).sum()
        )
        for outcome in CLINICAL_OUTCOMES
    }

    settings_df = pd.DataFrame(
        [
            {"setting": "source", "value": str(source_path)},
            {"setting": "scenario", "value": scenario["label"]},
            {
                "setting": "cohort",
                "value": (
                    f"{scenario['reference_label']} "
                    f"(diagnosis_code={list(scenario['reference_codes'])}) vs "
                    f"{scenario['case_label']} "
                    f"(diagnosis_code={list(scenario['case_codes'])})"
                ),
            },
            {"setting": "unit_of_analysis", "value": "one row per database Subject"},
            {
                "setting": "normality_test",
                "value": (
                    "Shapiro-Wilk, separately in "
                    f"{scenario['reference_label']} and {scenario['case_label']}"
                ),
            },
            {"setting": "group_test", "value": "two-sided Mann-Whitney U"},
            {
                "setting": "correlation",
                "value": (
                    "Spearman rho for each calculated feature x clinical outcome "
                    f"pair in the {scenario['label']} cohort"
                ),
            },
            {
                "setting": "clinical_outcomes",
                "value": ", ".join(CLINICAL_OUTCOMES),
            },
            {
                "setting": "multiple_testing",
                "value": "Benjamini-Hochberg FDR separately across features for each clinical outcome",
            },
            {"setting": "alpha", "value": ALPHA},
            {"setting": "feature_count", "value": len(feature_columns)},
            {"setting": "clinical_outcome_count", "value": len(CLINICAL_OUTCOMES)},
            {
                "setting": "correlation_pair_count",
                "value": int(len(correlations_df)),
            },
            {
                "setting": f"{scenario['reference_label']}_subject_count",
                "value": int((analysis_df["analysis_group"] == "reference").sum()),
            },
            {
                "setting": f"{scenario['case_label']}_subject_count",
                "value": int((analysis_df["analysis_group"] == "case").sum()),
            },
            {
                "setting": "significant_group_features_FDR",
                "value": int(len(significant_group_df)),
            },
            {
                "setting": "significant_clinical_correlations_FDR",
                "value": int(len(significant_correlations_df)),
            },
            *[
                {
                    "setting": f"significant_{outcome}_correlations_FDR",
                    "value": count,
                }
                for outcome, count in significant_by_outcome.items()
            ],
        ]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="analysis", index=False)
        significant_group_df.to_excel(writer, sheet_name="significant_group", index=False)
        for sheet_name, matrix in correlation_matrices.items():
            matrix.to_excel(writer, sheet_name=sheet_name, index=False)
        significant_correlations_df.to_excel(
            writer,
            sheet_name="significant_correlations",
            index=False,
        )
        correlations_df.to_excel(writer, sheet_name="correlations_long", index=False)
        normality_df.to_excel(writer, sheet_name="normality_details", index=False)
        settings_df.to_excel(writer, sheet_name="settings", index=False)
        _style_workbook(writer.book)

    logger.info(
        "Feature correlation analysis completed for %s / %s: features=%d, "
        "outcomes=%d, significant_group=%d, significant_correlations=%d, output=%s",
        dataset_name,
        scenario["label"],
        len(feature_columns),
        len(CLINICAL_OUTCOMES),
        len(significant_group_df),
        len(significant_correlations_df),
        output_path,
    )
    return {
        "dataset_name": dataset_name,
        "scenario_key": scenario["key"],
        "scenario_label": scenario["label"],
        "reference_label": scenario["reference_label"],
        "case_label": scenario["case_label"],
        "source_path": str(source_path),
        "output_path": str(output_path),
        "feature_count": int(len(feature_columns)),
        "reference_subject_count": int(
            (analysis_df["analysis_group"] == "reference").sum()
        ),
        "case_subject_count": int((analysis_df["analysis_group"] == "case").sum()),
        "significant_group_count": int(len(significant_group_df)),
        "clinical_outcome_count": int(len(CLINICAL_OUTCOMES)),
        "correlation_pair_count": int(len(correlations_df)),
        "significant_correlation_count": int(len(significant_correlations_df)),
        "significant_by_outcome": significant_by_outcome,
        "preview": results_df.head(20).replace({np.nan: None}).to_dict("records"),
    }


def _prepare_subject_level_data(grouped_df, scenario):
    prepared = grouped_df.copy()
    prepared["#Subject"] = prepared["#Subject"].astype(str).str.strip()
    subject_codes = prepared["#Subject"].dropna().unique().tolist()
    subject_rows = list(
        Subject.objects.filter(code__in=subject_codes).values(
            "code",
            "diagnosis_code",
            *CLINICAL_OUTCOMES.values(),
        )
    )
    subject_df = pd.DataFrame(subject_rows).rename(columns={"code": "#Subject"})
    if subject_df.empty:
        raise ValueError("No Subject rows matched the grouped dataset")

    prepared = prepared.merge(subject_df, on="#Subject", how="inner")
    reference_codes = set(scenario["reference_codes"])
    case_codes = set(scenario["case_codes"])
    allowed_codes = reference_codes | case_codes
    prepared = prepared[prepared["diagnosis_code"].isin(allowed_codes)].copy()
    prepared["analysis_group"] = np.where(
        prepared["diagnosis_code"].isin(reference_codes),
        "reference",
        "case",
    )
    for model_field in CLINICAL_OUTCOMES.values():
        prepared[model_field] = pd.to_numeric(prepared[model_field], errors="coerce")
    return prepared


def _group_output_columns(scenario):
    reference_label = scenario["reference_label"]
    case_label = scenario["case_label"]
    return [
        "Features",
        "Test normality",
        f"median ({reference_label})",
        f"median ({case_label})",
        f"MAD ({reference_label})",
        f"MAD ({case_label})",
        f"Mann-whitney U test ({reference_label} vs. {case_label})",
        "adj p",
    ]


def _analyze_feature(df, feature, scenario):
    reference = _numeric_values(
        df.loc[df["analysis_group"] == "reference", feature]
    )
    case = _numeric_values(df.loc[df["analysis_group"] == "case", feature])

    reference_normality = _shapiro_result(reference)
    case_normality = _shapiro_result(case)
    normality_summary = _normality_summary(
        reference_normality,
        case_normality,
        scenario["reference_label"],
        scenario["case_label"],
    )

    u_statistic = np.nan
    u_p_value = np.nan
    if len(reference) >= 1 and len(case) >= 1:
        try:
            u_statistic, u_p_value = mannwhitneyu(
                reference,
                case,
                alternative="two-sided",
                method="auto",
            )
        except ValueError:
            logger.warning("Mann-Whitney U failed for %s", feature, exc_info=True)

    output_columns = _group_output_columns(scenario)
    return {
        "Features": feature,
        "Test normality": normality_summary,
        output_columns[2]: _median(reference),
        output_columns[3]: _median(case),
        output_columns[4]: _mad(reference),
        output_columns[5]: _mad(case),
        output_columns[6]: _safe_float(u_p_value),
        "adj p": np.nan,
    }, [
        _normality_detail(
            feature,
            scenario["reference_label"],
            reference_normality,
        ),
        _normality_detail(feature, scenario["case_label"], case_normality),
    ]


def _analyze_correlations(df, feature_columns):
    rows = []
    for outcome_name, model_field in CLINICAL_OUTCOMES.items():
        outcome_rows = []
        for feature in feature_columns:
            pair_data = df[[feature, model_field]].copy()
            pair_data[feature] = pd.to_numeric(pair_data[feature], errors="coerce")
            pair_data[model_field] = pd.to_numeric(
                pair_data[model_field],
                errors="coerce",
            )
            pair_data = pair_data.replace([np.inf, -np.inf], np.nan).dropna()

            rho = np.nan
            p_value = np.nan
            if (
                    len(pair_data) >= 3
                    and pair_data[feature].nunique() > 1
                    and pair_data[model_field].nunique() > 1
            ):
                rho, p_value = spearmanr(
                    pair_data[feature],
                    pair_data[model_field],
                )

            outcome_rows.append(
                {
                    "Features": feature,
                    "Clinical outcome": outcome_name,
                    "n": int(len(pair_data)),
                    "Spearman rho": _safe_float(rho),
                    "p": _safe_float(p_value),
                    "adj p": np.nan,
                }
            )

        outcome_df = pd.DataFrame(outcome_rows)
        outcome_df["adj p"] = _benjamini_hochberg(outcome_df["p"])
        rows.append(outcome_df)

    return pd.concat(rows, ignore_index=True)[list(CORRELATION_COLUMNS)]


def _correlation_matrix(correlations_df, value_column):
    matrix = correlations_df.pivot(
        index="Features",
        columns="Clinical outcome",
        values=value_column,
    )
    matrix = matrix.reindex(columns=list(CLINICAL_OUTCOMES))
    return matrix.reset_index()


def _numeric_values(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _shapiro_result(values):
    if len(values) < 3:
        return {
            "n": int(len(values)),
            "statistic": None,
            "p_value": None,
            "normal": None,
            "status": "insufficient",
        }
    if values.nunique() < 2:
        return {
            "n": int(len(values)),
            "statistic": None,
            "p_value": None,
            "normal": None,
            "status": "constant",
        }
    statistic, p_value = shapiro(values)
    return {
        "n": int(len(values)),
        "statistic": _safe_float(statistic),
        "p_value": _safe_float(p_value),
        "normal": bool(p_value >= ALPHA),
        "status": "tested",
    }


def _normality_summary(
        reference_result,
        case_result,
        reference_label,
        case_label,
):
    def describe(label, result):
        if result["status"] == "constant":
            return f"{label}: constant (n={result['n']})"
        if result["p_value"] is None:
            return f"{label}: insufficient n={result['n']}"
        distribution = "normal" if result["normal"] else "non-normal"
        return f"{label}: {distribution} (p={result['p_value']:.4g})"

    return (
        f"{describe(reference_label, reference_result)}; "
        f"{describe(case_label, case_result)}"
    )


def _normality_detail(feature, group, result):
    return {
        "Features": feature,
        "group": group,
        "n": result["n"],
        "Shapiro-Wilk W": result["statistic"],
        "p": result["p_value"],
        "normal_at_alpha_0.05": result["normal"],
        "status": result["status"],
    }


def _median(values):
    return _safe_float(values.median()) if len(values) else None


def _mad(values):
    if not len(values):
        return None
    median = values.median()
    return _safe_float((values - median).abs().median())


def _benjamini_hochberg(p_values):
    numeric = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return adjusted

    ordered = valid.sort_values()
    count = len(ordered)
    raw_adjusted = ordered.to_numpy(dtype=float) * count / np.arange(1, count + 1)
    monotonic = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted.loc[ordered.index] = np.clip(monotonic, 0.0, 1.0)
    return adjusted


def _style_workbook(workbook):
    header_fill = PatternFill("solid", fgColor="1F4E78")
    significant_fill = PatternFill("solid", fgColor="E2F0D9")
    header_font = Font(color="FFFFFF", bold=True)
    analysis_widths = {
        1: 42,
        2: 48,
        3: 14,
        4: 14,
        5: 14,
        6: 14,
        7: 20,
        8: 14,
    }

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.sheet_view.showGridLines = False
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for column_index, column_cells in enumerate(worksheet.columns, start=1):
            max_length = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            worksheet.column_dimensions[get_column_letter(column_index)].width = min(
                max(max_length + 2, 12),
                48,
            )

    for sheet_name in ("analysis", "significant_group"):
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "B2"
        worksheet.row_dimensions[1].height = 38
        for column, width in analysis_widths.items():
            worksheet.column_dimensions[get_column_letter(column)].width = width

        for row in range(2, worksheet.max_row + 1):
            worksheet.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="top")
            for column in range(3, 9):
                worksheet.cell(row, column).alignment = Alignment(
                    horizontal="right",
                    vertical="top",
                )
            for column in (3, 4, 5, 6):
                worksheet.cell(row, column).number_format = "0.0000"
            for column in (7, 8):
                worksheet.cell(row, column).number_format = "0.000E+00"
            value = worksheet.cell(row, 8).value
            if isinstance(value, (int, float)) and value < ALPHA:
                worksheet.cell(row, 8).fill = significant_fill
                worksheet.cell(row, 8).font = Font(bold=True)

    for sheet_name in (
            "correlation_rho",
            "correlation_p",
            "correlation_adj_p",
            "correlation_n",
    ):
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "B2"
        worksheet.column_dimensions["A"].width = 48
        for column in range(2, worksheet.max_column + 1):
            worksheet.column_dimensions[get_column_letter(column)].width = 16
            for row in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row, column)
                cell.alignment = Alignment(horizontal="right")
                if sheet_name == "correlation_n":
                    cell.number_format = "0"
                elif sheet_name == "correlation_rho":
                    cell.number_format = "0.0000"
                else:
                    cell.number_format = "0.000E+00"
                if (
                        sheet_name == "correlation_adj_p"
                        and isinstance(cell.value, (int, float))
                        and cell.value < ALPHA
                ):
                    cell.fill = significant_fill
                    cell.font = Font(bold=True)

    for sheet_name in ("significant_correlations", "correlations_long"):
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        widths = (48, 18, 10, 16, 16, 16)
        for column, width in enumerate(widths, start=1):
            worksheet.column_dimensions[get_column_letter(column)].width = width
        for row in range(2, worksheet.max_row + 1):
            worksheet.cell(row, 3).number_format = "0"
            worksheet.cell(row, 4).number_format = "0.0000"
            worksheet.cell(row, 5).number_format = "0.000E+00"
            worksheet.cell(row, 6).number_format = "0.000E+00"
            value = worksheet.cell(row, 6).value
            if isinstance(value, (int, float)) and value < ALPHA:
                worksheet.cell(row, 6).fill = significant_fill
                worksheet.cell(row, 6).font = Font(bold=True)

    settings = workbook["settings"]
    settings.column_dimensions["A"].width = 42
    settings.column_dimensions["B"].width = 85
    for row in range(2, settings.max_row + 1):
        settings.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="top")
        if len(str(settings.cell(row, 2).value or "")) > 70:
            settings.row_dimensions[row].height = 32


def _safe_float(value):
    return float(value) if value is not None and pd.notna(value) else None
