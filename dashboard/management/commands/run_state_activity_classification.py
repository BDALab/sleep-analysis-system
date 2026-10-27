from django.core.management.base import BaseCommand, CommandError

from dashboard.logic.classification_person_grouped_validation import MODEL_FAMILIES
from dashboard.logic.state_activity_classification import run_state_activity_classification


class Command(BaseCommand):
    help = "Compare stable sleep/whole-interval activity features with the same features plus predicted sleep/wake activity."
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument("--state-input-dir", required=True, help="Completed extract_state_activity directory.")
        parser.add_argument("--target-input-dir", required=True, help="Completed run_state_activity_targets directory.")
        parser.add_argument("--output-dir",
                            help="New directory only; otherwise timestamped media/classification/state-activity.")
        parser.add_argument("--prepare-only", action="store_true",
                            help="Validate inputs and save matched tables/folds; no models fitted.")
        parser.add_argument("--models", nargs="+", choices=MODEL_FAMILIES, default=list(MODEL_FAMILIES))
        parser.add_argument("--bootstrap-resamples", type=int, default=10000)

    def handle(self, *args, **options):
        try:
            result = run_state_activity_classification(
                options["state_input_dir"], options["target_input_dir"], options["output_dir"],
                prepare_only=options["prepare_only"], models=options["models"],
                n_bootstrap=options["bootstrap_resamples"])
        except (OSError, ValueError, KeyError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(self.style.SUCCESS(f"Results: {result['output_dir']}"))
        self.stdout.write(f"Status: {result['status']}; {result['visits']} visits / {result['people']} people.")
