"""Exact manuscript activity metrics, partitioned by predicted state.

Separate from the original 16-feature/RBDq extraction to preserve its schema.
"""

import logging

import numpy as np
import pandas as pd

from dashboard.logic.state_activity_export import _load_signals
from dashboard.logic.state_activity_features import align_activity_states, summarize_night, utc_timestamp

logger = logging.getLogger(__name__)
CONTEXTS = ("whole", "sleep", "wake")
TARGETS = (
    {"outcome": "UPDRS", "field": "updrs", "metric": "mad", "aggregation": "IQR",
     "canonical_feature": "IQR.activity.Median Absolute Deviation",
     "canonical_nightly": "activity.Median Absolute Deviation"},
    {"outcome": "Attention", "field": "attention", "metric": "relative_interdecile", "aggregation": "SD",
     "canonical_feature": "SD.activity.Relative Interdencile Range",
     "canonical_nightly": "activity.Relative Interdencile Range"},
)
QC_COLUMNS = (
    "complete_activity_windows", "sleep_minutes", "wake_minutes", "mixed_minutes",
    "missing_prediction_minutes", "invalid_activity_minutes", "sleep_status", "wake_status",
    "usable_signal_coverage_fraction",
)


def target_distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mad": np.nan, "relative_interdecile": np.nan}
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Target metrics require finite nonnegative activity values")
    q10, q90 = np.quantile(values, [.1, .9], method="linear")
    maximum = float(np.max(values))
    return {"mad": float(np.median(np.abs(values - np.median(values)))),
            "relative_interdecile": float((q90 - q10) / maximum) if maximum else np.nan}


def target_feature(target, context):
    return f"{target['aggregation']}.activity_target_{context}.{target['metric']}"


def aggregate_target_nights(nightly, target, config):
    """Match nights *within each target*, not by availability of another scale."""
    if nightly.duplicated(["#Subject", "#Date"]).any():
        raise ValueError("Duplicate target subject/date nights")
    fields = [f"{context}_{target['metric']}" for context in (*CONTEXTS, "legacy_whole")]
    paired = (nightly["status"] == "ok") & (nightly["sleep_status"] == "ok") & (nightly["wake_status"] == "ok")
    paired &= np.isfinite(nightly[fields].to_numpy(dtype=float)).all(axis=1)
    rows = []
    for subject, group in nightly.groupby("#Subject", sort=True):
        valid = group.loc[paired.loc[group.index]]
        enough = len(valid) >= config.min_nights
        row = {"#Subject": subject, "target_valid_nights": len(valid),
               "target_night_dates": ";".join(sorted(valid["#Date"].astype(str)))}
        for context, field in zip((*CONTEXTS, "legacy_whole"), fields):
            values = valid[field].to_numpy(dtype=float)
            if not enough:
                value = np.nan
            elif target["aggregation"] == "IQR":
                q25, q75 = np.quantile(values, [.25, .75], method="linear")
                value = float(q75 - q25)
            elif target["aggregation"] == "SD":
                value = float(np.std(values, ddof=0))
            else:
                raise ValueError("Unsupported target aggregation")
            row[target_feature(target, context)] = value
        rows.append(row)
    return pd.DataFrame(rows)


def validate_source_fingerprints(actual, expected):
    # Paths, hashes, and sizes are bound to the completed extraction. Do not
    # silently use predictions regenerated with a different sleep/wake model.
    a = {str(item["path"]): (item["sha256"], item["size_bytes"]) for item in actual}
    b = {str(item["path"]): (item["sha256"], item["size_bytes"]) for item in expected}
    if a != b:
        raise ValueError("Cached signal inputs changed since the completed state extraction")


def extract_targeted_nights(source_nightly, source_manifest, config):
    from dashboard.models import SleepNight

    if source_nightly["night_id"].duplicated().any():
        raise ValueError("Duplicate source night IDs")
    by_id = source_nightly.set_index("night_id")
    nights = list(SleepNight.objects.filter(pk__in=by_id.index.tolist()).select_related(
        "data", "subject", "diary_day").order_by("data_id", "pk"))
    if {night.pk for night in nights} != set(by_id.index):
        raise ValueError("Source night IDs no longer match the database")
    expected_sources = {item["recording_id"]: item for item in source_manifest["sources"]}
    groups = {}
    for night in nights:
        groups.setdefault(night.data_id, []).append(night)
    if set(groups) != set(expected_sources):
        raise ValueError("Source recording IDs do not reproduce")
    rows, sources = [], []
    for position, (recording_id, group) in enumerate(groups.items(), 1):
        if position == 1 or position % 10 == 0 or position == len(groups):
            logger.info("Target activity extraction: recording %s/%s", position, len(groups))
        activity, predictions, fingerprints = _load_signals(group[0].data)
        validate_source_fingerprints(fingerprints, expected_sources[recording_id]["files"])
        activity_utc = activity.copy()
        activity_utc.index = pd.to_datetime(activity_utc.index, utc=True)
        activity_utc = activity_utc.sort_index()
        for night in group:
            saved = by_id.loc[night.pk]
            if (night.subject.code != saved["#Subject"] or night.data_id != saved["recording_id"]
                    or night.diary_day.date.isoformat() != saved["#Date"]
                    or night.subject_id != night.data.subject_id or night.subject_id != night.diary_day.subject_id):
                raise ValueError(f"Night identity changed: {night.pk}")
            onset, offset = utc_timestamp(night.sleep_onset), utc_timestamp(night.sleep_end)
            start, end = onset - pd.Timedelta(seconds=15), offset + pd.Timedelta(seconds=15)
            if start != utc_timestamp(saved["interval_start_utc"]) or end != utc_timestamp(saved["interval_end_utc"]):
                raise ValueError(f"Night interval changed: {night.pk}")
            aligned = align_activity_states(activity, predictions, start, end, config)
            qc = summarize_night(aligned, config)
            for column in QC_COLUMNS:
                left, right = qc[column], saved[column]
                equal = left == right if isinstance(left, str) else np.isclose(left, right, rtol=1e-10, atol=1e-12,
                                                                               equal_nan=True)
                if not equal:
                    raise ValueError(f"Original state support no longer reproduces: night {night.pk}, {column}")
            row = {"#Subject": saved["#Subject"], "#Date": saved["#Date"],
                   "#DiagnosisCode": saved["#DiagnosisCode"], "night_id": night.pk,
                   "recording_id": recording_id, "status": saved["status"],
                   **{column: qc[column] for column in QC_COLUMNS}}
            for context in CONTEXTS:
                if context == "whole":
                    values = aligned.loc[aligned.state.isin((*CONTEXTS[1:], "mixed")), "activity_index"]
                else:
                    values = aligned.loc[aligned.state == context, "activity_index"]
                    if qc[f"{context}_status"] != "ok":
                        values = []
                row.update({f"{context}_{name}": value for name, value in target_distribution(values).items()})
            # Original code slices activity timestamps inclusively at the
            # stored onset/offset centres; audit this against frozen exports.
            legacy = target_distribution(activity_utc.loc[onset:offset].to_numpy())
            row.update({f"legacy_whole_{name}": value for name, value in legacy.items()})
            rows.append(row)
        sources.append({"recording_id": recording_id, "files": fingerprints, "status": "verified"})
    return pd.DataFrame(rows), sources


def audit_legacy_nightly(extracted, frozen_nightly):
    frozen = frozen_nightly.copy()
    frozen["#Date"] = pd.to_datetime(frozen["#Date"]).dt.strftime("%Y-%m-%d")
    keys = ["#Subject", "#Date"]
    if frozen.duplicated(keys).any():
        raise ValueError("Duplicate frozen nightly feature rows")
    joined = extracted.merge(frozen, on=keys, how="left", validate="one_to_one", indicator=True)
    rows = []
    for target in TARGETS:
        for row in joined.to_dict("records"):
            observed = row[f"legacy_whole_{target['metric']}"]
            expected = row[target["canonical_nightly"]]
            status = "no_frozen_reference" if row["_merge"] == "left_only" else (
                "pass" if np.isclose(observed, expected, rtol=1e-9, atol=1e-12, equal_nan=True) else "mismatch"
            )
            rows.append({"#Subject": row["#Subject"], "#Date": row["#Date"], "metric": target["metric"],
                         "recomputed": observed, "frozen": expected, "status": status})
    return pd.DataFrame(rows)
