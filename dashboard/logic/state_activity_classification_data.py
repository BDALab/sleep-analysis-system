"""Wearable-only, matched-night inputs for the state activity classifier audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.logic.classification_grouped_statistics import STATS_PREFIXES
from dashboard.logic.classification_grouped_statistics_strict import _filter_columns_by_feature_family
from dashboard.logic.classification_validity_checks import (
    subject_person_id, subject_source_cohort, subject_visit_index,
)
from dashboard.logic.feature_families import (
    ACTIVITY_EXTENSION_STABLE_FAMILY_IDS, PRIMARY_SLEEP_STABLE_FAMILY_IDS,
)
from dashboard.logic.group_data import _add_grouped_statistics, _group_by_subject, _sort_rows_for_grouping
from dashboard.logic.state_activity_export import _fingerprint
from dashboard.logic.state_activity_features import (
    METRICS, STATES, VISIT_FEATURES, StateActivityConfig, aggregate_visits,
)
from dashboard.logic.state_activity_rbdq import (
    COHORT_RELATIVE_PATH, EXTENDED_RUN, validate_exported_aggregates,
)
from dashboard.logic.state_activity_targeted_features import (
    QC_COLUMNS, TARGETS, aggregate_target_nights, audit_legacy_nightly, target_feature,
)

ADJUSTMENT_COLUMNS = ("#Gender", "#Education")
TARGET_STATE_FEATURES = tuple(target_feature(target, state) for target in TARGETS for state in STATES)
ADDED_FEATURES = (*VISIT_FEATURES, *TARGET_STATE_FEATURES)
BASELINE = "whole_interval"
AUGMENTED = "whole_interval_plus_states"
FEATURE_SETS = (BASELINE, AUGMENTED)


def _dated(table):
    table = table.copy()
    table["#Date"] = pd.to_datetime(table["#Date"], errors="raise").dt.strftime("%Y-%m-%d")
    if table["#Subject"].isna().any() or table["#Date"].isna().any():
        raise ValueError("Missing subject/date identity")
    if table.duplicated(["#Subject", "#Date"]).any():
        raise ValueError("Duplicate subject/date nights")
    return table.reset_index(drop=True)


def prepare_matched_features(nightly, visits, targeted, frozen_nightly, frozen_cohort, diagnoses, config):
    """No clinical score is consulted, filtered on, or returned as a predictor.

    The baseline is recomputed on exactly the nights supporting all 20 added
    features. Missing baseline predictors/covariates remain for foldwise imputation.
    """
    if config != StateActivityConfig():
        raise ValueError("Use the primary state extraction settings for this fixed experiment")
    nightly, targeted, frozen_nightly = map(_dated, (nightly, targeted, frozen_nightly))
    validate_exported_aggregates(nightly, visits, config)
    if frozen_cohort["#Subject"].duplicated().any():
        raise ValueError("Duplicate frozen cohort visits")
    if nightly.groupby("#Subject").size().max() > 7:
        raise ValueError("More than seven nights per visit")
    if nightly["night_id"].duplicated().any() or targeted["night_id"].duplicated().any():
        raise ValueError("Duplicate night IDs")
    keys = ["#Subject", "#Date", "night_id", "recording_id", "#DiagnosisCode"]
    aligned = nightly[keys + ["status", *QC_COLUMNS]].merge(
        targeted, on=keys, how="outer", suffixes=("_source", ""), validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("State and targeted extractions contain different night identities")
    for column in ("status", *QC_COLUMNS):
        a, b = aligned[f"{column}_source"], aligned[column]
        equal = a.fillna("").eq(b.fillna("")) if column.endswith("status") or column == "status" else np.isclose(
            a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-10, atol=1e-12, equal_nan=True)
        if not np.all(equal):
            raise ValueError(f"Targeted extraction support differs: {column}")

    codes = set(visits["#Subject"]) | set(frozen_cohort["#Subject"])
    if codes - set(diagnoses) or any(diagnoses[code] not in (0, 3) for code in codes):
        raise ValueError("Missing or non-HC/MCI-LB numeric diagnosis codes")
    if not np.array_equal(visits["#Subject"].map(diagnoses), visits["#DiagnosisCode"]):
        raise ValueError("Numeric diagnoses changed since extraction")
    if not np.array_equal(nightly["#Subject"].map(diagnoses), nightly["#DiagnosisCode"]):
        raise ValueError("Nightly numeric diagnoses disagree with the database")
    demographics = frozen_cohort[["#Subject", "#Age", *ADJUSTMENT_COLUMNS]].merge(
        visits, on="#Subject", suffixes=("_frozen", ""), how="left", validate="one_to_one")
    if demographics["#DiagnosisCode"].isna().any():
        raise ValueError("Frozen extended cohort has visits absent from the state extraction")
    for column in ("#Age", *ADJUSTMENT_COLUMNS):
        if not np.allclose(demographics[column], demographics[f"{column}_frozen"], equal_nan=True):
            raise ValueError(f"Demographic drift from the frozen cohort: {column}")
    sleep_codes = set(visits.loc[visits.sleep_valid_nights >= config.min_nights, "#Subject"])
    if sleep_codes != set(frozen_cohort["#Subject"]):
        raise ValueError("Sleep eligibility does not reproduce the frozen extended cohort")
    legacy_audit = audit_legacy_nightly(targeted, frozen_nightly)
    if legacy_audit["status"].eq("mismatch").any():
        raise ValueError("Targeted legacy metrics do not reproduce the frozen nightly values")

    # One common support mask for all new features, independent of UPDRS/RBDq/etc.
    paired = nightly["status"].eq("ok") & nightly["sleep_status"].eq("ok") & nightly["wake_status"].eq("ok")
    state_fields = [f"{state}_{metric}" for state in STATES for metric in METRICS]
    paired &= np.isfinite(nightly[state_fields].to_numpy(dtype=float)).all(axis=1)
    target_by_id = targeted.set_index("night_id").loc[nightly["night_id"]].reset_index()
    target_fields = [f"{context}_{target['metric']}" for target in TARGETS
                     for context in ("whole", "sleep", "wake", "legacy_whole")]
    paired &= np.isfinite(target_by_id[target_fields].to_numpy(dtype=float)).all(axis=1)
    matched_nights = nightly.loc[paired & nightly["#Subject"].isin(frozen_cohort["#Subject"])].copy()
    counts = matched_nights.groupby("#Subject").size()
    eligible = set(counts[counts >= config.min_nights].index)
    matched_nights = matched_nights.loc[matched_nights["#Subject"].isin(eligible)].copy()
    if not eligible:
        raise ValueError("No visits with five jointly valid nights")
    raw = matched_nights[["#Subject", "#Date"]].merge(
        frozen_nightly, on=["#Subject", "#Date"], how="left", validate="one_to_one", indicator=True)
    if not raw["_merge"].eq("both").all():
        raise ValueError("Matched state nights missing from frozen baseline nightly data")
    # Explicit wearable namespace, not whatever extra numeric columns exist in an export.
    nightly_predictors = [column for column in frozen_nightly if str(column).startswith(
        ("actigraphy.", "actigraphy_norm.", "diary.", "diary_norm.", "activity."))]
    baseline = _add_grouped_statistics(_group_by_subject(_sort_rows_for_grouping(
        raw[["#Subject", "#Date", "#Age", "#Gender", "#Education", "#Disease", *nightly_predictors]])))
    stats = [column for column in baseline if str(column).startswith(STATS_PREFIXES)]
    baseline_features, _, family_audit = _filter_columns_by_feature_family(
        stats, [], PRIMARY_SLEEP_STABLE_FAMILY_IDS | ACTIVITY_EXTENSION_STABLE_FAMILY_IDS)
    if not baseline_features:
        raise ValueError("No stable sleep/activity baseline predictors")
    state = aggregate_visits(matched_nights, config)[["#Subject", *VISIT_FEATURES]]
    target_rows = targeted.loc[targeted["night_id"].isin(matched_nights["night_id"])]
    for target in TARGETS:
        fields = [target_feature(target, state) for state in STATES]
        state = state.merge(aggregate_target_nights(target_rows, target, config)[["#Subject", *fields]],
                            on="#Subject", validate="one_to_one")
    table = baseline[["#Subject", *ADJUSTMENT_COLUMNS, *baseline_features]].merge(
        state, on="#Subject", validate="one_to_one").sort_values("#Subject").reset_index(drop=True)
    if not np.isfinite(table[list(ADDED_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Matched state features are not finite")
    table["diagnosis_code"] = table["#Subject"].map(diagnoses)
    table["y_true"] = table["diagnosis_code"].map({0: 0, 3: 1})
    table["person_group"] = table["#Subject"].map(subject_person_id)
    table["source_cohort"] = table["#Subject"].map(subject_source_cohort)
    table["visit_index"] = table["#Subject"].map(subject_visit_index)
    if set(table.y_true) != {0, 1}:
        raise ValueError("Both diagnosis groups are required")
    if (table.groupby("person_group").source_cohort.nunique() != 1).any():
        raise ValueError("A participant crosses sources; source holdout would leak participants")
    # Check raw nightly adjustment values, not just the saved visit-level copy.
    for column in ADJUSTMENT_COLUMNS:
        for code, group in raw.groupby("#Subject"):
            expected = visits.set_index("#Subject").loc[code, column]
            if not np.allclose(group[column], expected, equal_nan=True):
                raise ValueError(f"Frozen nightly demographics differ: {code}, {column}")
    audit = pd.DataFrame({"#Subject": sorted(codes)})
    audit["diagnosis_code"] = audit["#Subject"].map(diagnoses)
    audit["in_frozen_extended_cohort"] = audit["#Subject"].isin(frozen_cohort["#Subject"])
    audit["matched_valid_nights"] = audit["#Subject"].map(counts).fillna(0).astype(int)
    audit["eligible"] = audit["#Subject"].isin(eligible)
    audit["exclusion_reason"] = np.where(audit.eligible, "", np.where(
        audit.in_frozen_extended_cohort, "fewer than five jointly valid nights", "outside frozen extended cohort"))
    return {"table": table, "baseline_features": baseline_features, "family_audit": family_audit,
            "cohort_audit": audit, "legacy_audit": legacy_audit,
            "matched_nights": matched_nights[["#Subject", "#Date", "night_id", "recording_id"]]}


def load_experiment_inputs(state_dir, target_dir):
    from dashboard.models import Subject

    state_dir, target_dir = Path(state_dir).resolve(), Path(target_dir).resolve()
    state_manifest = state_dir / "manifest.json"
    target_manifest = target_dir / "analysis_manifest.json"
    source = json.loads(state_manifest.read_text())
    target = json.loads(target_manifest.read_text())
    if source["status"] != "complete" or target["status"] != "complete":
        raise ValueError("Both input extractions must be complete")
    if source["config"] != target["config"]:
        raise ValueError("Extraction settings differ")
    paths = [state_manifest, state_dir / "nightly_state_activity.csv", state_dir / "visit_state_activity.csv",
             EXTENDED_RUN / COHORT_RELATIVE_PATH, EXTENDED_RUN / "raw/clinical_data.xlsx"]
    expected = {str(Path(item["path"]).resolve()): item for item in target["input_files"]}
    fingerprints = []
    for path in paths:
        actual = _fingerprint(path)
        saved = expected.get(str(path))
        if not saved or any(actual[key] != saved[key] for key in ("sha256", "size_bytes")):
            raise ValueError(f"Input differs from the completed targeted extraction: {path}")
        fingerprints.append(actual)
    fingerprints.extend(_fingerprint(path) for path in
                        (target_manifest, target_dir / "nightly_target_features.csv"))
    nightly, visits = (pd.read_csv(path) for path in paths[1:3])
    cohort, raw = pd.read_excel(paths[3]), pd.read_excel(paths[4])
    targeted = pd.read_csv(target_dir / "nightly_target_features.csv")
    codes = set(cohort["#Subject"]) | set(visits["#Subject"])
    diagnoses = dict(Subject.objects.filter(code__in=codes).values_list("code", "diagnosis_code"))
    prepared = prepare_matched_features(nightly, visits, targeted, raw, cohort, diagnoses,
                                        StateActivityConfig(**source["config"]))
    return prepared, fingerprints, source["config"]
