from django.core.management.base import BaseCommand, CommandError

from dashboard.logic.state_activity_targeted_analysis import run_targeted_followups


class Command(BaseCommand):
    help = "Run the manuscript-targeted UPDRS/MAD and attention/interdecile-range sleep/wake follow-ups."
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument("--input-dir", required=True, help="Completed extract_state_activity result directory.")
        parser.add_argument("--output-dir",
                            help="New directory only; defaults to timestamped media/state-activity-targeted.")

    def handle(self, *args, **options):
        try:
            result = run_targeted_followups(options["input_dir"], options["output_dir"])
        except (OSError, ValueError, KeyError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(self.style.SUCCESS(f"Results: {result['output_dir']}"))
        for outcome, cohort in result["cohorts"].items():
            self.stdout.write(f"{outcome}: {cohort['visits']} matched visits / {cohort['people']} people.")
