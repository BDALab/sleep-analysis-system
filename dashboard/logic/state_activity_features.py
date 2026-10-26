"""Exploratory activity summaries conditional on *predicted* sleep/wake.

No model fitting, database writes, or modification of the canonical features.
Activity timestamps mark 60-second window starts; prediction timestamps mark
30-second window centres (see preprocess_data._preprocess_prediction_data).
Naive cached timestamps are UTC, matching the existing feature extractor.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

STATES = ("sleep", "wake")
METRICS = ("mean", "median", "p90", "iqr")
PREDICTION_COLUMN = "SLEEP_PREDICTION"
ACTIVITY_SECONDS = 60
PREDICTION_SECONDS = 30


@dataclass(frozen=True)
class StateActivityConfig:
    mixed_policy: str = "exclude"
    min_state_minutes: float = 5.0
    min_nights: int = 5
    min_night_coverage: float = 0.9
    sleep_probability_threshold: float = 0.5

    def __post_init__(self):
        if self.mixed_policy not in ("exclude", "majority"):
            raise ValueError("mixed_policy must be exclude or majority")
        if not np.isfinite(self.min_state_minutes) or self.min_state_minutes <= 0:
            raise ValueError("min_state_minutes must be finite and positive")
        if isinstance(self.min_nights, bool) or int(self.min_nights) != self.min_nights or self.min_nights < 1:
            raise ValueError("min_nights must be a positive integer")
        if not 0 < self.sleep_probability_threshold < 1:
            raise ValueError("sleep_probability_threshold must be between 0 and 1")
        if not 0 <= self.min_night_coverage <= 1:
            raise ValueError("min_night_coverage must be between 0 and 1")

    def to_dict(self):
        return asdict(self)


def utc_timestamp(value):
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("Missing interval boundary")
    return value.tz_localize("UTC") if value.tz is None else value.tz_convert("UTC")


def _validated_series(series, duration_seconds, name):
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a Series with a DatetimeIndex")
    if series.index.hasnans:
        raise ValueError(f"{name} has missing timestamps")
    result = series.copy()
    result.index = pd.to_datetime(result.index, utc=True)
    result = result.sort_index()
    if result.index.has_duplicates:
        raise ValueError(f"{name} has duplicate timestamps")
    if len(result) > 1 and (result.index.to_series().diff().dropna() < pd.Timedelta(seconds=duration_seconds)).any():
        raise ValueError(f"{name} has overlapping windows")
    # Cache corruption must not silently turn text or impossible probabilities
    # into apparent sleep. Genuine missing numeric values are handled below.
    return pd.to_numeric(result, errors="raise").astype(float)


def align_activity_states(activity, predictions, start, end, config=None):
    """Classify complete activity windows by exact temporal overlap.

    Intervals are half-open [start, end). Never forward-fill across missing
    prediction epochs or duplicate a minute into both state distributions.
    Majority is a sensitivity policy, not an alternative sleep/wake model.
    """
    config = config or StateActivityConfig()
    activity = _validated_series(activity, ACTIVITY_SECONDS, "Activity index")
    predictions = _validated_series(predictions, PREDICTION_SECONDS, "Predictions")
    start, end = utc_timestamp(start), utc_timestamp(end)
    if end <= start:
        raise ValueError("Night end must be after night start")
    invalid_probability = predictions.notna() & (
            ~np.isfinite(predictions) | (predictions < 0) | (predictions > 1)
    )
    if invalid_probability.any():
        raise ValueError("Prediction probabilities must lie within [0, 1]")

    # Convert explicitly to nanoseconds: newer pandas may store datetime64[us].
    p_start = (predictions.index - pd.Timedelta(seconds=15)).as_unit("ns").asi8
    p_end = p_start + PREDICTION_SECONDS * 10 ** 9
    p_value = predictions.to_numpy()
    rows = []
    activity = activity.loc[
        (activity.index >= start) & (activity.index <= end - pd.Timedelta(seconds=ACTIVITY_SECONDS))]
    for timestamp, value in activity.items():
        window_end = timestamp + pd.Timedelta(seconds=ACTIVITY_SECONDS)
        if timestamp < start or window_end > end:
            continue
        a, b = timestamp.value, window_end.value
        left = np.searchsorted(p_end, a, side="right")
        right = np.searchsorted(p_start, b, side="left")
        sleep_seconds = wake_seconds = 0.0
        for i in range(left, right):
            if not np.isfinite(p_value[i]):
                continue
            overlap = (min(b, p_end[i]) - max(a, p_start[i])) / 10 ** 9
            if p_value[i] >= config.sleep_probability_threshold:
                sleep_seconds += overlap
            else:
                wake_seconds += overlap
        coverage = sleep_seconds + wake_seconds
        if not np.isfinite(value) or value < 0:
            state = "invalid_activity"
        elif not np.isclose(coverage, ACTIVITY_SECONDS, rtol=0, atol=1e-6):
            state = "missing_prediction"
        elif np.isclose(sleep_seconds, ACTIVITY_SECONDS, rtol=0, atol=1e-6):
            state = "sleep"
        elif np.isclose(wake_seconds, ACTIVITY_SECONDS, rtol=0, atol=1e-6):
            state = "wake"
        elif config.mixed_policy == "majority" and sleep_seconds > ACTIVITY_SECONDS / 2 + 1e-6:
            state = "sleep"
        elif config.mixed_policy == "majority" and wake_seconds > ACTIVITY_SECONDS / 2 + 1e-6:
            state = "wake"
        else:
            state = "mixed"
        rows.append((timestamp, value, sleep_seconds, wake_seconds, coverage, state))
    result = pd.DataFrame(rows, columns=(
        "window_start", "activity_index", "sleep_seconds", "wake_seconds",
        "prediction_coverage_seconds", "state",
    ))
    for column in ("activity_index", "sleep_seconds", "wake_seconds", "prediction_coverage_seconds"):
        result[column] = result[column].astype(float)
    result.attrs["interval_seconds"] = (end - start).total_seconds()
    return result


def _distribution(values):
    if len(values) == 0:
        return dict.fromkeys(METRICS, np.nan)
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    }


def summarize_night(aligned, config=None):
    config = config or StateActivityConfig()
    result = {"complete_activity_windows": len(aligned)}
    for state in (*STATES, "mixed", "missing_prediction", "invalid_activity"):
        result[f"{state}_minutes"] = int((aligned["state"] == state).sum())
    interval_seconds = aligned.attrs["interval_seconds"]
    result["night_interval_minutes"] = interval_seconds / 60
    result["activity_coverage_fraction"] = (
            float((np.isfinite(aligned["activity_index"]) & (
                        aligned["activity_index"] >= 0)).sum()) * 60 / interval_seconds
    )
    # Count mixed minutes as covered even though they are not assigned to a
    # pure state. Missing activity periods must not vanish from the denominator.
    result["usable_signal_coverage_fraction"] = float(
        aligned["state"].isin((*STATES, "mixed")).sum()) * 60 / interval_seconds
    coverage_ok = result["usable_signal_coverage_fraction"] >= config.min_night_coverage
    result["prediction_coverage_fraction"] = (
        float(aligned["prediction_coverage_seconds"].sum() / (len(aligned) * ACTIVITY_SECONDS))
        if len(aligned) else np.nan
    )
    for state in STATES:
        values = aligned.loc[aligned["state"] == state, "activity_index"]
        sufficient = len(values) >= config.min_state_minutes and coverage_ok
        result[f"{state}_status"] = (
            "insufficient_coverage" if not coverage_ok else "ok" if sufficient else "insufficient_minutes"
        )
        distribution = _distribution(values if sufficient else [])
        result.update({f"{state}_{metric}": value for metric, value in distribution.items()})
    # A diagnostic reference, not a replacement for the historical whole-window
    # features. Uses full in-boundary minute windows, including mixed windows.
    valid_activity = aligned.loc[
        np.isfinite(aligned["activity_index"]) & (aligned["activity_index"] >= 0),
        "activity_index",
    ]
    result.update({f"whole_interval_{k}": v for k, v in _distribution(valid_activity).items()})
    return result


VISIT_METADATA = (
    "#Subject", "#DiagnosisCode", "#Age", "#Gender", "#Education", "RBDq",
)
VISIT_FEATURES = tuple(
    f"{aggregation}.activity_{state}.{metric}"
    for state in STATES for metric in METRICS for aggregation in ("Median", "IQR")
)


def aggregate_visits(nightly, config=None):
    """One row per database Subject (visit), never pool repeat visits.

    Each state must independently have min_nights with sufficient minutes.
    Keep excluded visits and missing features for an auditable denominator.
    """
    config = config or StateActivityConfig()
    columns = [*VISIT_METADATA, "nights_selected", "nights_processed", "sleep_valid_nights", "wake_valid_nights",
               *VISIT_FEATURES]
    if nightly.empty:
        return pd.DataFrame(columns=columns)
    if nightly.duplicated(["#Subject", "#Date"]).any():
        raise ValueError("Duplicate subject/date nights; resolve recordings before aggregation")
    rows = []
    for subject, group in nightly.groupby("#Subject", sort=True):
        for column in VISIT_METADATA:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Inconsistent visit metadata: {subject}, {column}")
        row = group.iloc[0][list(VISIT_METADATA)].to_dict()
        row["nights_selected"] = len(group)
        row["nights_processed"] = int((group["status"] == "ok").sum())
        for state in STATES:
            valid = group.loc[(group["status"] == "ok") & (group[f"{state}_status"] == "ok")]
            row[f"{state}_valid_nights"] = len(valid)
            for metric in METRICS:
                values = valid[f"{state}_{metric}"].dropna().to_numpy(dtype=float)
                enough = len(values) >= config.min_nights
                row[f"Median.activity_{state}.{metric}"] = float(np.median(values)) if enough else np.nan
                row[f"IQR.activity_{state}.{metric}"] = float(
                    np.quantile(values, .75) - np.quantile(values, .25)) if enough else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
