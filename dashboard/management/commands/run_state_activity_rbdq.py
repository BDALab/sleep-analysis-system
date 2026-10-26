from django.core.management.base import BaseCommand, CommandError

from dashboard.logic.state_activity_rbdq import run_state_activity_rbdq


class Command(BaseCommand):
    help = "Run the separate manuscript-aligned exploratory RBDq analysis of predicted-state activity."
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument("--input-dir", required=True, help="A completed extract_state_activity result directory.")
        parser.add_argument("--output-dir",
                            help="New directory only; defaults to timestamped media/state-activity-rbdq.")

    def handle(self, *args, **options):
        try:
            result = run_state_activity_rbdq(options["input_dir"], options["output_dir"])
        except (OSError, ValueError, KeyError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(self.style.SUCCESS(f"Results: {result['output_dir']}"))
        self.stdout.write(f"Common cohort: {result['visits']} visits / {result['people']} people.")
        self.stdout.write(
            f"Primary FDR-significant main effects: {result['primary_main_fdr']}/16; diagnosis interactions: {result['primary_interaction_fdr']}/16.")
