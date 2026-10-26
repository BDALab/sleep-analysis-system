from datetime import datetime, timezone
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from dashboard.logic.state_activity_export import check_inputs, export_state_activity, select_nights
from dashboard.logic.state_activity_features import StateActivityConfig


class Command(BaseCommand):
    help = "Export separate predicted-sleep/wake activity summaries from existing caches; no training or database changes."
    # Avoid importing the web UI and its unrelated ML/plotting dependencies.
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument("--output-dir",
                            help="New directory only; default: timestamped media/state-activity directory.")
        parser.add_argument("--check-inputs", action="store_true",
                            help="Check cached-file availability without extracting or writing results.")
        parser.add_argument("--diagnosis-codes", nargs="+", type=int, choices=(0, 1, 2, 3), default=[0, 3])
        parser.add_argument("--subject", action="append", default=[],
                            help="Exact database Subject code; repeat to select visits.")
        parser.add_argument("--wrist", choices=("L", "R", "O"), default="L")
        parser.add_argument("--limit-recordings", type=int,
                            help="Process only the first N eligible recordings for a smoke test.")
        parser.add_argument("--exclude-recording", type=int, action="append", default=[],
                            help="Explicitly omit a recording ID; repeat as needed. Exclusions are recorded in the manifest.")
        parser.add_argument("--mixed-policy", choices=("exclude", "majority"), default="exclude")
        parser.add_argument("--min-state-minutes", type=float, default=5,
                            help="Minimum assigned minutes per state per night (default 5; exploratory support rule).")
        parser.add_argument("--min-nights", type=int, default=5,
                            help="Minimum usable nights independently per state per visit (default 5).")
        parser.add_argument("--min-night-coverage", type=float, default=.9,
                            help="Minimum fraction of the nightly interval covered by valid activity and predictions (default 0.9).")
        parser.add_argument("--sleep-threshold", type=float, default=.5)

    def handle(self, *args, **options):
        try:
            config = StateActivityConfig(
                mixed_policy=options["mixed_policy"],
                min_state_minutes=options["min_state_minutes"],
                min_nights=options["min_nights"],
                min_night_coverage=options["min_night_coverage"],
                sleep_probability_threshold=options["sleep_threshold"],
            )
            selection = {
                "diagnosis_codes": options["diagnosis_codes"],
                "subjects": options["subject"], "wrist": options["wrist"],
                "limit_recordings": options["limit_recordings"],
                "excluded_recording_ids": options["exclude_recording"],
                "training_data_excluded": True, "dreamt_data_excluded": True,
            }
            nights = select_nights(
                diagnosis_codes=options["diagnosis_codes"], subjects=options["subject"],
                wrist=options["wrist"], limit_recordings=options["limit_recordings"],
                exclude_recordings=options["exclude_recording"],
            )
            if options["check_inputs"]:
                checked = check_inputs(nights)
                missing = [row for row in checked if row["status"] != "ready"]
                self.stdout.write(
                    f"Selected {len(nights)} nights, {len(checked)} recordings; {len(missing)} recordings lack cached inputs.")
                for row in missing:
                    self.stdout.write(f"Recording {row['recording_id']}: {row['status']}")
                if missing:
                    raise CommandError(
                        "Missing caches; nothing was regenerated or written. See report/state_activity_features.md.")
                return
            output = options["output_dir"] or (
                    Path(settings.MEDIA_ROOT) / "state-activity"
                    / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            )
            result = export_state_activity(nights, output, config, selection)
        except (OSError, ValueError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(f"Results: {result['output_dir']}")
        self.stdout.write(
            f"Processed {result['nights_processed']}/{result['nights_selected']} nights. "
            f"Visits with sleep/wake/both features: {result['visits_with_sleep_features']}/"
            f"{result['visits_with_wake_features']}/{result['visits_with_both_states']}."
        )
        if result["failed_recordings"]:
            raise CommandError(
                "Some recordings failed. QC outputs were saved, but this is an incomplete run; inspect manifest.json before analysis.")
        if not result["nights_processed"]:
            raise CommandError("No usable nightly intervals were found. Inspect the QC outputs before analysis.")
