import importlib.util
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from dashboard.logic.state_activity_export import check_inputs, export_state_activity, recording_paths
from dashboard.logic.state_activity_features import (
    METRICS, STATES, VISIT_FEATURES, StateActivityConfig,
    aggregate_visits, align_activity_states, summarize_night,
)


def series(values, start="2026-01-01 00:00:00", seconds=60, tz="UTC"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=f"{seconds}s", tz=tz), dtype=float)


class StateActivityAlignmentTest(unittest.TestCase):
    def align(self, activity, predictions, config=None, start="2026-01-01", end="2026-01-02"):
        return align_activity_states(activity, predictions, start, end, config)

    def test_centred_predictions_and_start_timestamp_activity(self):
        result = self.align(series([2, 8, 5]), series([1, 1, 0, 0, 1, 0], "2026-01-01 00:00:15", 30))
        self.assertEqual(result.state.tolist(), ["sleep", "wake", "mixed"])
        self.assertEqual(result.sleep_seconds.tolist(), [60, 0, 30])
        self.assertEqual(result.prediction_coverage_seconds.tolist(), [60, 60, 60])

    def test_overlap_alignment_not_exact_timestamp_join_or_nearest_match(self):
        # A window starting :10 spans 50 seconds sleep and 10 seconds wake.
        activity = series([4], "2026-01-01 00:00:10")
        predictions = series([1, 1, 0], "2026-01-01 00:00:15", 30)
        pure = self.align(activity, predictions)
        self.assertEqual(pure.state.iloc[0], "mixed")
        self.assertEqual(pure.sleep_seconds.iloc[0], 50)
        majority = self.align(activity, predictions, StateActivityConfig(mixed_policy="majority"))
        self.assertEqual(majority.state.iloc[0], "sleep")

    def test_majority_keeps_ties_mixed(self):
        result = self.align(series([1]), series([1, 0], "2026-01-01 00:00:15", 30),
                            StateActivityConfig(mixed_policy="majority"))
        self.assertEqual(result.state.iloc[0], "mixed")

    def test_missing_epoch_is_not_forward_filled(self):
        predictions = series([1, 1, 1, 1], "2026-01-01 00:00:15", 30).drop(
            pd.Timestamp("2026-01-01 00:00:45", tz="UTC"))
        result = self.align(series([1, 2]), predictions)
        self.assertEqual(result.state.tolist(), ["missing_prediction", "sleep"])
        self.assertEqual(result.prediction_coverage_seconds.tolist(), [30, 60])

    def test_nan_probability_is_missing_not_wake(self):
        result = self.align(series([1]), series([1, np.nan], "2026-01-01 00:00:15", 30))
        self.assertEqual(result.state.iloc[0], "missing_prediction")

    def test_probability_threshold_matches_existing_point_five_rule(self):
        result = self.align(series([1, 2]), series([.5, .7, .49, 0], "2026-01-01 00:00:15", 30))
        self.assertEqual(result.state.tolist(), ["sleep", "wake"])

    def test_invalid_probabilities_fail_loudly(self):
        for invalid in (-.01, 1.1, np.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "within"):
                self.align(series([1]), series([1, invalid], "2026-01-01 00:00:15", 30))

    def test_zero_activity_is_valid_negative_or_missing_is_not(self):
        result = self.align(series([0, -1, np.nan, np.inf]), series([1] * 8, "2026-01-01 00:00:15", 30))
        self.assertEqual(result.state.tolist(), ["sleep", "invalid_activity", "invalid_activity", "invalid_activity"])

    def test_windows_crossing_night_boundary_are_excluded(self):
        result = self.align(series([1, 2, 3]), series([1] * 6, "2026-01-01 00:00:15", 30), start="2026-01-01 00:00:15",
                            end="2026-01-01 00:02:15")
        self.assertEqual(result.activity_index.tolist(), [2])

    def test_half_open_boundary_does_not_include_next_epoch(self):
        result = self.align(series([2]), series([1, 1, 0], "2026-01-01 00:00:15", 30))
        self.assertEqual(result.state.iloc[0], "sleep")

    def test_naive_utc_and_aware_timezones_align(self):
        activity = series([2], tz=None)
        predictions = series([1, 1], "2026-01-01 01:00:15", 30, "Europe/Prague")
        self.assertEqual(self.align(activity, predictions).state.iloc[0], "sleep")

    def test_datetime_microseconds_are_converted_before_overlap_math(self):
        activity = series([2])
        predictions = series([1, 1], "2026-01-01 00:00:15", 30)
        activity.index = activity.index.as_unit("us")
        predictions.index = predictions.index.as_unit("us")
        self.assertEqual(self.align(activity, predictions).state.iloc[0], "sleep")

    def test_duplicate_and_overlapping_epochs_are_rejected(self):
        for bad in (
                pd.Series([1., 1.], index=pd.to_datetime(["2026-01-01 00:00:15"] * 2)),
                series([1, 1], "2026-01-01 00:00:15", 20),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate|overlapping"):
                self.align(series([2]), bad)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.align(series([1, 2], seconds=30), series([1, 1], "2026-01-01 00:00:15", 30))

    def test_invalid_configs_are_rejected(self):
        for kwargs in ({"min_state_minutes": 0}, {"min_state_minutes": np.nan}, {"min_nights": 0},
                       {"mixed_policy": "nearest"}, {"sleep_probability_threshold": 1}, {"min_night_coverage": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StateActivityConfig(**kwargs)


class StateActivitySummaryTest(unittest.TestCase):
    def test_four_distribution_features_and_no_wake_is_missing_not_zero(self):
        config = StateActivityConfig(min_state_minutes=2)
        aligned = align_activity_states(series([0, 2]), series([1] * 4, "2026-01-01 00:00:15", 30), "2026-01-01",
                                        "2026-01-01 00:02:00", config)
        result = summarize_night(aligned, config)
        self.assertEqual(result["sleep_mean"], 1)
        self.assertEqual(result["sleep_median"], 1)
        self.assertAlmostEqual(result["sleep_p90"], 1.8)
        self.assertEqual(result["sleep_iqr"], 1)
        self.assertEqual(result["wake_status"], "insufficient_minutes")
        self.assertTrue(np.isnan(result["wake_mean"]))

    def test_minimum_minutes_is_checked_per_state(self):
        config = StateActivityConfig(min_state_minutes=2)
        aligned = align_activity_states(series([1, 2, 9]), series([1, 1, 1, 1, 0, 0], "2026-01-01 00:00:15", 30),
                                        "2026-01-01", "2026-01-01 00:03:00", config)
        result = summarize_night(aligned, config)
        self.assertEqual(result["sleep_status"], "ok")
        self.assertEqual(result["wake_minutes"], 1)
        self.assertTrue(np.isnan(result["wake_p90"]))

    def test_empty_interval_has_explicit_support_and_missing_features(self):
        empty = series([])
        result = summarize_night(align_activity_states(empty, empty, "2026-01-01", "2026-01-02"))
        self.assertEqual(result["complete_activity_windows"], 0)
        self.assertTrue(np.isnan(result["sleep_mean"]))
        self.assertTrue(np.isnan(result["prediction_coverage_fraction"]))

    def test_missing_activity_remains_in_night_coverage_denominator(self):
        config = StateActivityConfig(min_state_minutes=1)
        aligned = align_activity_states(series([1, 2]), series([1] * 4, "2026-01-01 00:00:15", 30), "2026-01-01",
                                        "2026-01-01 00:10:00", config)
        result = summarize_night(aligned, config)
        self.assertAlmostEqual(result["usable_signal_coverage_fraction"], .2)
        self.assertEqual(result["prediction_coverage_fraction"], 1)
        self.assertEqual(result["sleep_status"], "insufficient_coverage")
        self.assertTrue(np.isnan(result["sleep_mean"]))

    def test_aggregate_keeps_repeat_visits_separate_and_checks_state_nights(self):
        rows = []
        for subject in ("HC-1", "HC2-1"):
            for day in range(1, 6):
                row = {"#Subject": subject, "#Date": f"2026-01-{day:02d}", "#DiagnosisCode": 0, "#Age": 70,
                       "#Gender": 1, "#Education": 12, "RBDq": 3, "status": "ok", "sleep_status": "ok",
                       "wake_status": "ok" if day < 5 else "insufficient_minutes"}
                for state in STATES:
                    for metric in METRICS:
                        row[f"{state}_{metric}"] = day if state == "sleep" or day < 5 else np.nan
                rows.append(row)
        result = aggregate_visits(pd.DataFrame(rows))
        self.assertEqual(len(result), 2)
        self.assertEqual(len(VISIT_FEATURES), 16)
        self.assertEqual(result["Median.activity_sleep.mean"].tolist(), [3, 3])
        self.assertEqual(result["IQR.activity_sleep.mean"].tolist(), [2, 2])
        self.assertTrue(result["Median.activity_wake.mean"].isna().all())
        self.assertEqual(result["wake_valid_nights"].tolist(), [4, 4])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_visits(pd.DataFrame([rows[0], rows[0]]))


class StateActivityExportTest(unittest.TestCase):
    def night(self, root):
        raw = root / "media/data/example.csv"
        recording = SimpleNamespace(data=SimpleNamespace(path=str(raw)))
        subject = SimpleNamespace(code="HC-1", diagnosis_code=0, age=70, sex="F", education_years=12, rbdq=3)
        return SimpleNamespace(data=recording, data_id=1, subject=subject,
                               diary_day=SimpleNamespace(date=date(2026, 1, 1)), pk=1, broken=False,
                               sleep_onset=pd.Timestamp("2026-01-01 00:00:15", tz="UTC"),
                               sleep_end=pd.Timestamp("2026-01-01 00:02:45", tz="UTC"))

    def test_paths_do_not_create_source_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            night = self.night(root)
            prediction, activity = recording_paths(night.data)
            self.assertEqual(prediction, (root / "media/predictions/example.csv.pkl").resolve())
            self.assertEqual(activity, [])
            self.assertEqual(list(root.iterdir()), [])
            self.assertIn("missing", check_inputs([night])[0]["status"])

    def test_export_recovers_epoch_edges_and_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            night = self.night(root)
            activity = series([1, 2, 3])
            predictions = series([1] * 6, "2026-01-01 00:00:15", 30)
            output = root / "result"
            with patch("dashboard.logic.state_activity_export._load_signals", return_value=(activity, predictions, [])):
                result = export_state_activity([night], output, StateActivityConfig(min_state_minutes=1, min_nights=1))
            self.assertEqual(result["nights_processed"], 1)
            self.assertEqual(result["visits_with_sleep_features"], 1)
            self.assertEqual(result["visits_with_wake_features"], 0)
            nightly = pd.read_csv(output / "nightly_state_activity.csv")
            self.assertEqual(nightly.sleep_minutes.iloc[0], 3)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["config"]["min_nights"], 1)
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse((root / "media").exists())
            with self.assertRaises(FileExistsError):
                export_state_activity([night], output)

    def test_input_failures_are_audited_and_cannot_appear_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = export_state_activity([self.night(root)], root / "result")
            self.assertEqual(result["failed_recordings"], 1)
            self.assertEqual(result["nights_processed"], 0)
            manifest = json.loads((root / "result/manifest.json").read_text())
            self.assertEqual(manifest["status"], "incomplete_inputs")
            visits = pd.read_csv(root / "result/visit_state_activity.csv")
            self.assertTrue(visits[list(VISIT_FEATURES)].isna().all().all())

    def test_real_hdf_and_pickle_inputs(self):
        try:
            import tables  # noqa: F401
        except ImportError:
            self.skipTest("PyTables is required for the HDF integration test")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            night = self.night(root)
            prediction, _ = recording_paths(night.data)
            prediction.parent.mkdir(parents=True)
            series([1] * 6, "2026-01-01 00:00:15", 30).to_frame("SLEEP_PREDICTION").to_pickle(prediction)
            activity_dir = root / "media/sleeppy/example/activity_index_days"
            activity_dir.mkdir(parents=True)
            series([1, 2, 3]).to_frame("activity_index").to_hdf(activity_dir / "day.h5", key="activity_index_data_24hr")
            self.assertEqual(check_inputs([night])[0]["status"], "ready")
            result = export_state_activity([night], root / "result",
                                           StateActivityConfig(min_state_minutes=1, min_nights=1))
            self.assertEqual(result["failed_recordings"], 0)
            manifest = json.loads((root / "result/manifest.json").read_text())
            self.assertEqual(len(manifest["sources"][0]["files"]), 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["sources"][0]["files"]))


@unittest.skipUnless(importlib.util.find_spec("django"), "Django is required for command tests")
class StateActivityCommandTest(unittest.TestCase):
    def command(self, extra=()):
        from dashboard.management.commands.extract_state_activity import Command
        command = Command(stdout=io.StringIO(), stderr=io.StringIO())
        options = vars(command.create_parser("manage.py", "extract_state_activity").parse_args(
            ["--output-dir", "/unused-test-output", *extra]))
        return command, options

    def test_invalid_config_is_rejected_before_database_selection(self):
        from django.core.management.base import CommandError
        command, options = self.command(["--min-night-coverage", "2"])
        with patch("dashboard.management.commands.extract_state_activity.select_nights") as select:
            with self.assertRaises(CommandError):
                command.handle(**options)
            select.assert_not_called()

    def test_check_inputs_never_starts_extraction(self):
        command, options = self.command(["--check-inputs"])
        with patch("dashboard.management.commands.extract_state_activity.select_nights",
                   return_value=[object()]), patch("dashboard.management.commands.extract_state_activity.check_inputs",
                                                   return_value=[{"status": "ready"}]), patch(
                "dashboard.management.commands.extract_state_activity.export_state_activity") as export:
            command.handle(**options)
            export.assert_not_called()

    def test_partial_run_raises_command_error_after_reporting_audit_path(self):
        from django.core.management.base import CommandError
        command, options = self.command()
        result = {"output_dir": "/example-audit", "nights_processed": 0, "nights_selected": 1,
                  "visits_with_sleep_features": 0, "visits_with_wake_features": 0, "visits_with_both_states": 0,
                  "failed_recordings": 1}
        with patch("dashboard.management.commands.extract_state_activity.select_nights",
                   return_value=[object()]), patch(
                "dashboard.management.commands.extract_state_activity.export_state_activity", return_value=result):
            with self.assertRaisesRegex(CommandError, "incomplete"):
                command.handle(**options)
        self.assertIn("/example-audit", command.stdout._out.getvalue())


if __name__ == "__main__":
    unittest.main()
