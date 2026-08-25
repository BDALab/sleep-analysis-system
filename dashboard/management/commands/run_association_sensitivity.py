from django.core.management.base import BaseCommand, CommandError

from dashboard.logic.association_sensitivity_analysis import (
    run_hc_vs_predlb_association_sensitivity,
)


class Command(BaseCommand):
    help = "Run frozen-candidate age and source sensitivity GEE analyses."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Optional output directory; a timestamped media directory is used by default.",
        )

    def handle(self, *args, **options):
        result = run_hc_vs_predlb_association_sensitivity(
            output_dir=options["output_dir"],
        )
        self.stdout.write(self.style.SUCCESS(result["workbook_path"]))
        if not result["primary_reproduction_passed"]:
            raise CommandError(
                "Primary-model reproduction check failed; do not interpret results."
            )
