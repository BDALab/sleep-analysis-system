import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from dashboard.logic.covariates import (
    COVARIATE_ALPHA,
    CovariateController,
    _attach_subject_covariates,
    _impute_covariates,
    _prepare_dataset,
    verify_covariates_for_excel,
)
from dashboard.logic.group_data import group_clinical_data_excel
from dashboard.models import Subject
from mysite.settings import BASE_DIR, MEDIA_ROOT

logger = logging.getLogger(__name__)

PREPARATION_ROOT = Path(MEDIA_ROOT) / "analysis-preparation"
DATASET_SOURCES = {
    "dataset-clinical": Path(BASE_DIR) / "dataset-clinical.xlsx",
    "dataset-clinical-acc": Path(BASE_DIR) / "dataset-clinical-acc.xlsx",
}
SCENARIOS = (
    {
        "key": "predlb-vs-hc",
        "label": "preDLB vs HC",
        "positive_codes": (3,),
        "negative_codes": (0,),
    },
    {
        "key": "predlb-mci-vs-hc",
        "label": "preDLB+MCI-AD vs HC",
        "positive_codes": (3, 2),
        "negative_codes": (0,),
    },
    {
        "key": "mci-vs-hc",
        "label": "MCI-AD vs HC",
        "positive_codes": (2,),
        "negative_codes": (0,),
    },
)
COVARIATE_FIELDS = {
    "age": "#Age",
    "gender": "#Gender",
    "education": "#Education",
}


def prepare_all_analysis_datasets(dataset_names=None):
    names = dataset_names or tuple(DATASET_SOURCES)
    return [prepare_analysis_dataset(name) for name in names]


def prepare_analysis_dataset(dataset_name):
    if dataset_name not in DATASET_SOURCES:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    source_path = DATASET_SOURCES[dataset_name]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = PREPARATION_ROOT / dataset_name / run_id
    verification_dir = run_dir / "verification"
    raw_dir = run_dir / "raw"
    verification_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    verification = verify_covariates_for_excel(
        source_path,
        alpha=COVARIATE_ALPHA,
        output_dir=verification_dir,
    )
    source_df = pd.read_excel(source_path, index_col=0)
    source_df = _prepare_dataset(_attach_subject_covariates(source_df))
    source_df = _attach_diagnosis_code(source_df)

    raw_clinical_path = raw_dir / "clinical_data.xlsx"
    source_df.drop(columns=["_diagnosis_code"]).to_excel(raw_clinical_path, index=False)
    raw_grouping = group_clinical_data_excel(
        raw_clinical_path,
        output_path=raw_dir / "grouped_clinical_matrix.xlsx",
    )

    scenario_results = []
    for scenario in SCENARIOS:
        selected_covariates = _selected_covariates_for_scenario(
            verification,
            scenario["label"],
        )
        scenario_result = _prepare_scenario_dataset(
            source_df=source_df,
            run_dir=run_dir,
            scenario=scenario,
            selected_covariates=selected_covariates,
        )
        scenario_results.append(scenario_result)

    manifest = {
        "dataset_name": dataset_name,
        "source_path": str(source_path),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "verification_path": verification["output_path"],
        "raw_grouped_stats_path": raw_grouping["stats_output_path"],
        "scenarios": scenario_results,
    }
    with open(run_dir / "preparation_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    logger.info("Prepared %s analysis datasets in %s", dataset_name, run_dir)
    return manifest


def _prepare_scenario_dataset(
        source_df,
        run_dir,
        scenario,
        selected_covariates,
):
    scenario_codes = set(scenario["positive_codes"]) | set(scenario["negative_codes"])
    scenario_df = source_df[source_df["_diagnosis_code"].isin(scenario_codes)].copy()
    scenario_df = scenario_df.drop(columns=["_diagnosis_code"])

    feature_columns = [
        column for column in scenario_df.columns if not str(column).startswith("#")
    ]
    adjusted_features = scenario_df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    covariate_fields = [
        COVARIATE_FIELDS[covariate]
        for covariate in selected_covariates
        if COVARIATE_FIELDS[covariate] in scenario_df.columns
    ]
    if covariate_fields:
        covariates_df = _impute_covariates(scenario_df[covariate_fields])
        feature_medians = adjusted_features.median(axis=0, skipna=True).fillna(0)
        adjusted_features = CovariateController().fit_transform(
            adjusted_features.fillna(feature_medians),
            covariates_df,
        )

    metadata_columns = [
        column
        for column in (
            "#Subject",
            "#Date",
            "#Age",
            "#Gender",
            "#Education",
            "#Disease",
        )
        if column in scenario_df.columns
    ]
    adjusted_df = pd.concat(
        [
            scenario_df[metadata_columns].reset_index(drop=True),
            adjusted_features.reset_index(drop=True),
        ],
        axis=1,
    )

    scenario_dir = run_dir / "scenarios" / scenario["key"]
    data_dir = scenario_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    clinical_path = data_dir / "clinical_data_adjusted.xlsx"
    adjusted_df.to_excel(clinical_path, index=False)
    grouping = group_clinical_data_excel(
        clinical_path,
        output_path=data_dir / "grouped_clinical_matrix.xlsx",
    )

    settings = {
        "scenario_key": scenario["key"],
        "scenario_label": scenario["label"],
        "positive_codes": list(scenario["positive_codes"]),
        "negative_codes": list(scenario["negative_codes"]),
        "selected_covariates": list(selected_covariates),
        "subject_count": int(adjusted_df["#Subject"].nunique()),
        "clinical_path": str(clinical_path),
        "grouped_stats_path": grouping["stats_output_path"],
    }
    with open(scenario_dir / "scenario_settings.json", "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
    return settings


def _attach_diagnosis_code(df):
    enriched = df.copy()
    codes = enriched["#Subject"].dropna().astype(str).str.strip().unique().tolist()
    diagnosis_by_subject = {
        row["code"]: row["diagnosis_code"]
        for row in Subject.objects.filter(code__in=codes).values(
            "code",
            "diagnosis_code",
        )
    }
    enriched["_diagnosis_code"] = (
        enriched["#Subject"].astype(str).str.strip().map(diagnosis_by_subject)
    )
    return enriched


def _selected_covariates_for_scenario(verification, scenario_label):
    return [
        covariate
        for covariate in COVARIATE_FIELDS
        if any(
            test["scenario"] == scenario_label
            and test["covariate"] == covariate
            and test["control_recommended"]
            for test in verification["tests"]
        )
    ]
