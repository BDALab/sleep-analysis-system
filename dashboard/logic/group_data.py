import logging
from pathlib import Path

import pandas as pd

from mysite.settings import MEDIA_ROOT

logger = logging.getLogger(__name__)

DATASET_CLINICAL_PATH = (
        Path(MEDIA_ROOT) / "covariates" / "dataset-clinical" / "data" / "clinical_data.xlsx"
)
DATASET_CLINICAL_ACC_PATH = (
        Path(MEDIA_ROOT) / "covariates" / "dataset-clinical-acc" / "data" / "clinical_data.xlsx"
)
IDENTITY_COLUMNS = ("#Subject", "#Gender", "#Age", "#Disease")


def group_covariates_dataset_clinical():
    return group_clinical_data_excel(DATASET_CLINICAL_PATH)


def group_covariates_dataset_clinical_acc():
    return group_clinical_data_excel(DATASET_CLINICAL_ACC_PATH)


def group_all_covariate_datasets():
    return [
        group_covariates_dataset_clinical(),
        group_covariates_dataset_clinical_acc(),
    ]


def group_clinical_data_excel(source_path, output_path=None):
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Clinical dataset not found: {source_path}")

    logger.info(f"Grouping clinical data by subject for {source_path.name}")
    df = pd.read_excel(source_path)
    _validate_columns(df, source_path)

    if df["#Subject"].isna().any():
        missing_count = int(df["#Subject"].isna().sum())
        logger.warning(
            f"Skipping {missing_count} row(s) without #Subject in {source_path.name}"
        )
        df = df[df["#Subject"].notna()].copy()

    df = _sort_rows_for_grouping(df)
    grouped_df = _group_by_subject(df)

    output_path = Path(output_path) if output_path else source_path.parent / "grouped_clinical_matrix.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_df.to_excel(output_path, index=False)

    logger.info(
        f"Grouped clinical data saved to {output_path} "
        f"({len(grouped_df)} subjects, {len(grouped_df.columns)} columns)"
    )
    return {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "subject_count": int(grouped_df["#Subject"].nunique()) if not grouped_df.empty else 0,
        "column_count": int(len(grouped_df.columns)),
    }


def _validate_columns(df, source_path):
    missing_columns = [column for column in IDENTITY_COLUMNS if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Clinical dataset {source_path.name} is missing required column(s): "
            f"{', '.join(missing_columns)}"
        )


def _sort_rows_for_grouping(df):
    sorted_df = df.copy()
    sorted_df["_row_order"] = range(len(sorted_df))

    if "#Date" in sorted_df.columns:
        sorted_df["_parsed_date"] = pd.to_datetime(sorted_df["#Date"], errors="coerce")
        sorted_df = sorted_df.sort_values(
            by=["#Subject", "_parsed_date", "_row_order"],
            kind="mergesort",
        )
        sorted_df = sorted_df.drop(columns=["_parsed_date"])
    else:
        sorted_df = sorted_df.sort_values(by=["#Subject", "_row_order"], kind="mergesort")

    return sorted_df.drop(columns=["_row_order"])


def _group_by_subject(df):
    feature_columns = [
        column
        for column in df.columns
        if column not in IDENTITY_COLUMNS and not _is_auxiliary_column(column)
    ]

    grouped_rows = []
    max_days = 0

    for subject, group in df.groupby("#Subject", sort=True):
        _log_inconsistent_identity(subject, group)

        demographics = group.iloc[0][list(IDENTITY_COLUMNS)].to_dict()
        daily_data = {}
        for day_index, (_, row) in enumerate(group.iterrows(), start=1):
            for column in feature_columns:
                daily_data[f"day{day_index}.{column}"] = row[column]

        grouped_rows.append({**demographics, **daily_data})
        max_days = max(max_days, len(group))

    grouped_df = pd.DataFrame(grouped_rows)
    ordered_day_columns = [
        f"day{day_index}.{column}"
        for day_index in range(1, max_days + 1)
        for column in feature_columns
    ]
    ordered_columns = list(IDENTITY_COLUMNS) + [
        column for column in ordered_day_columns if column in grouped_df.columns
    ]
    return grouped_df.reindex(columns=ordered_columns)


def _is_auxiliary_column(column):
    normalized = str(column).strip()
    return normalized.startswith("Unnamed:") or normalized.startswith("#")


def _log_inconsistent_identity(subject, group):
    for column in IDENTITY_COLUMNS[1:]:
        if group[column].nunique(dropna=False) > 1:
            logger.warning(
                f"Subject {subject} has inconsistent {column} values in grouped clinical data"
            )
