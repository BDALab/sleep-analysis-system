"""Read cached signals and export a separate, versioned exploratory dataset."""

import hashlib
import json
import logging
import pickle
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.logic.state_activity_features import (
    PREDICTION_COLUMN,
    StateActivityConfig,
    aggregate_visits,
    align_activity_states,
    summarize_night,
    utc_timestamp,
)

logger = logging.getLogger(__name__)


def select_nights(diagnosis_codes=(0, 3), subjects=(), wrist="L", limit_recordings=None, exclude_recordings=()):
    from dashboard.models import SleepNight

    queryset = SleepNight.objects.filter(
        data__training_data=False,
        data__dreamt_data=False,
        data__body_location=wrist,
        subject__diagnosis_code__in=diagnosis_codes,
    ).select_related("data", "subject", "diary_day").order_by("data_id", "diary_day__date", "pk")
    if subjects:
        queryset = queryset.filter(subject__code__in=subjects)
    nights = list(queryset)
    if subjects and set(subjects) != {n.subject.code for n in nights}:
        raise ValueError("Some requested subjects have no eligible nights under the selected filters")
    available_ids = {n.data_id for n in nights}
    if set(exclude_recordings) - available_ids:
        raise ValueError("Some excluded recording IDs are not in the selected cohort")
    nights = [n for n in nights if n.data_id not in exclude_recordings]
    if limit_recordings is not None:
        if limit_recordings < 1:
            raise ValueError("limit_recordings must be positive")
        ids = list(dict.fromkeys(n.data_id for n in nights))[:limit_recordings]
        nights = [n for n in nights if n.data_id in ids]
    if not nights:
        raise ValueError("No eligible SleepNight records found")
    if any(n.subject_id != n.data.subject_id or n.subject_id != n.diary_day.subject_id for n in nights):
        raise ValueError("Subject mismatch between a night, recording, or diary")
    keys = [(n.subject_id, n.diary_day.date) for n in nights]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate subject/date nights; select the intended recording before extraction")
    return nights


def recording_paths(recording):
    # Do not use model path properties: some create directories as a side effect.
    raw = Path(recording.data.path).resolve()
    media = raw.parent.parent
    prediction = media / "predictions" / f"{raw.name}.pkl"
    activity_dir = media / "sleeppy" / raw.stem / "activity_index_days"
    activity = sorted(p for p in activity_dir.glob("*.h5") if not p.name.startswith("._"))
    return prediction, activity


def check_inputs(nights):
    recordings = {night.data_id: night.data for night in nights}
    rows = []
    for recording_id, recording in recordings.items():
        prediction, activity = recording_paths(recording)
        missing = []
        if not prediction.is_file():
            missing.append("prediction cache")
        if not activity:
            missing.append("activity-index day files")
        rows.append({
            "recording_id": recording_id,
            "prediction_cache": str(prediction),
            "activity_day_files": len(activity),
            "status": "missing " + ", ".join(missing) if missing else "ready",
        })
    return rows


def _fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def _load_signals(recording):
    prediction_path, activity_paths = recording_paths(recording)
    if not prediction_path.is_file() or not activity_paths:
        raise FileNotFoundError(
            "Existing prediction cache and activity-index day files are required; no regeneration is performed")
    predictions = pd.read_pickle(prediction_path)
    if not isinstance(predictions, pd.DataFrame) or PREDICTION_COLUMN not in predictions:
        raise ValueError(f"Prediction cache must contain {PREDICTION_COLUMN}")
    activity_frames = []
    for path in activity_paths:
        frame = pd.read_hdf(path)
        if not isinstance(frame, pd.DataFrame) or "activity_index" not in frame:
            raise ValueError(f"Missing activity_index column: {path}")
        activity_frames.append(frame["activity_index"])
    provenance = [_fingerprint(path) for path in (prediction_path, *activity_paths)]
    return pd.concat(activity_frames), predictions[PREDICTION_COLUMN], provenance


def _metadata(night):
    subject = night.subject
    return {
        "#Subject": subject.code,
        "#Date": night.diary_day.date.isoformat(),
        "#DiagnosisCode": subject.diagnosis_code,
        "#Age": subject.age,
        # Same coding as covariates._normalize_gender, without importing the
        # feature-screening/statistical pipeline into this extraction command.
        "#Gender": {"F": 1, "M": 2}.get(subject.sex, np.nan),
        "#Education": subject.education_years,
        "RBDq": subject.rbdq,
        "night_id": night.pk,
        "recording_id": night.data_id,
        "sleeppy_broken_flag": bool(night.broken),
    }


def export_state_activity(nights, output_dir, config=None, selection=None):
    """Only write new result files. A pre-existing output directory is refused."""
    config = config or StateActivityConfig()
    output_dir = Path(output_dir).resolve()
    if not nights:
        raise ValueError("No nights supplied")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    recordings = {}
    for night in nights:
        recordings.setdefault(night.data_id, []).append(night)
    rows, sources = [], []
    failed_recordings = 0
    for index, (recording_id, recording_nights) in enumerate(recordings.items(), 1):
        logger.info("State activity: recording %d/%d", index, len(recordings))
        source = {"recording_id": recording_id}
        try:
            activity, predictions, fingerprints = _load_signals(recording_nights[0].data)
            source.update(status="ok", files=fingerprints)
            recording_rows = []
            for night in recording_nights:
                # SleepNight endpoints are prediction *centres*. Recover the
                # actual first/last epoch edges before selecting full minutes.
                start = utc_timestamp(night.sleep_onset) - pd.Timedelta(seconds=15)
                end = utc_timestamp(night.sleep_end) + pd.Timedelta(seconds=15)
                aligned = align_activity_states(activity, predictions, start, end, config)
                row = {
                    **_metadata(night),
                    "interval_start_utc": start.isoformat(),
                    "interval_end_utc": end.isoformat(),
                    "status": "ok" if not aligned.empty else "no_complete_activity_windows",
                    "error": "",
                    **summarize_night(aligned, config),
                }
                recording_rows.append(row)
            rows.extend(recording_rows)
        except (OSError, ValueError, KeyError, TypeError, ImportError, EOFError, pickle.UnpicklingError,
                RuntimeError) as error:
            failed_recordings += 1
            source.update(status="input_error", error=f"{type(error).__name__}: {error}")
            logger.warning("State activity: recording %s failed: %s", recording_id, error)
            empty = align_activity_states(
                pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float),
                pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=float),
                "2000-01-01", "2000-01-02", config,
            )
            for night in recording_nights:
                rows.append({**_metadata(night), "status": "input_error", "error": source["error"],
                             **summarize_night(empty, config)})
        sources.append(source)

    nightly = pd.DataFrame(rows)
    visits = aggregate_visits(nightly, config)
    nightly.to_csv(output_dir / "nightly_state_activity.csv", index=False, na_rep="")
    visits.to_csv(output_dir / "visit_state_activity.csv", index=False, na_rep="")
    summary = {
        "recordings": len(recordings), "failed_recordings": failed_recordings,
        "nights_selected": len(nights),
        "nights_processed": int((nightly["status"] == "ok").sum()),
        "visits_selected": len(visits),
        "visits_with_sleep_features": int((visits["sleep_valid_nights"] >= config.min_nights).sum()),
        "visits_with_wake_features": int((visits["wake_valid_nights"] >= config.min_nights).sum()),
        "visits_with_both_states": int(((visits["sleep_valid_nights"] >= config.min_nights) & (
                    visits["wake_valid_nights"] >= config.min_nights)).sum()),
    }
    manifest = {
        "schema_version": 1, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "incomplete_inputs" if failed_recordings else
            "no_usable_nights" if not summary["nights_processed"] else "complete"
        ),
        "config": config.to_dict(), "selection": selection or {}, "summary": summary,
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "definition": {
            "state_labels": "predicted, not PSG-confirmed; probability >= threshold is sleep",
            "interval": "first sleep epoch start to last sleep epoch end; no daytime or presleep wake",
            "activity_windows": "60 seconds, start timestamps; only complete in-interval windows",
            "prediction_epochs": "30 seconds, centre timestamps; naive cache times interpreted as UTC",
            "missing_predictions": "exclude incomplete windows; no interpolation or forward fill",
            "mixed_windows": "exclude" if config.mixed_policy == "exclude" else "strict temporal majority; 50/50 remains mixed",
            "aggregation": "one row per database Subject/visit; median and IQR across state-valid nights",
            "minimum_support": "exploratory configurable thresholds, not clinically validated cutoffs",
            "nonwear": "no additional nonwear mask applied; activity zeros are not automatically nonwear",
            "broken_flag": "exported for QC, not automatically excluded",
            "whole_interval_reference": "nightly QC reference only; does not overwrite or exactly reproduce canonical endpoint conventions",
            "inference": "no correlations, GEE models, FDR tests, or classifier fitting performed",
        },
        "implementation": [_fingerprint(Path(__file__)),
                           _fingerprint(Path(__file__).with_name("state_activity_features.py"))],
        "sources": sources,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {"output_dir": str(output_dir), **summary}
