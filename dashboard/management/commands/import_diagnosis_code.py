from __future__ import annotations

from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from dashboard.models import Subject

DEFAULT_EXCEL_PATH = Path(
    r"E:\geneactiv-processing-data\grouped_clinical_matrix_with_stats_relabeled_with_new_labels.xlsx"
)
SUBJECT_COLUMN = "#Subject"
DIAGNOSIS_COLUMN = "#DiseaseNew"
ALLOWED_CODES = {0, 1, 2, 3}


class Command(BaseCommand):
    help = "Import diagnosis_code values for Subject from an Excel sheet."

    def add_arguments(self, parser):
        parser.add_argument(
            "--excel",
            dest="excel_path",
            default=str(DEFAULT_EXCEL_PATH),
            help="Path to the Excel file with #Subject and #DiseaseNew columns.",
        )

    def handle(self, *args, **options):
        excel_path = Path(options["excel_path"]).expanduser()
        if not excel_path.exists():
            raise CommandError(f"Excel file not found: {excel_path}")

        df = pd.read_excel(excel_path, usecols=[SUBJECT_COLUMN, DIAGNOSIS_COLUMN])
        if SUBJECT_COLUMN not in df.columns or DIAGNOSIS_COLUMN not in df.columns:
            raise CommandError(
                f"Expected columns {SUBJECT_COLUMN} and {DIAGNOSIS_COLUMN} in {excel_path}"
            )

        df = df.dropna(subset=[DIAGNOSIS_COLUMN])
        df[SUBJECT_COLUMN] = df[SUBJECT_COLUMN].astype(str).str.strip()

        code_map = {}
        invalid_rows = 0
        for _, row in df.iterrows():
            raw_code = row[DIAGNOSIS_COLUMN]
            try:
                diagnosis_code = int(raw_code)
            except (TypeError, ValueError):
                invalid_rows += 1
                continue

            if diagnosis_code not in ALLOWED_CODES:
                invalid_rows += 1
                continue

            subject_code = str(row[SUBJECT_COLUMN]).strip()
            if not subject_code:
                invalid_rows += 1
                continue

            code_map[subject_code] = diagnosis_code

        if not code_map:
            self.stdout.write(self.style.WARNING("No valid diagnosis codes found."))
            return

        subjects = Subject.objects.filter(code__in=code_map.keys()).only("id", "code", "diagnosis_code")
        subjects_by_code = {subject.code: subject for subject in subjects}

        missing_subjects = sorted(set(code_map.keys()) - set(subjects_by_code.keys()))
        to_update = []
        for code, diagnosis_code in code_map.items():
            subject = subjects_by_code.get(code)
            if not subject:
                continue
            if subject.diagnosis_code != diagnosis_code:
                subject.diagnosis_code = diagnosis_code
                to_update.append(subject)

        if to_update:
            Subject.objects.bulk_update(to_update, ["diagnosis_code"])

        self.stdout.write(
            self.style.SUCCESS(
                "Import complete: "
                f"rows={len(df)}, "
                f"unique_subjects={len(code_map)}, "
                f"updated={len(to_update)}, "
                f"missing_subjects={len(missing_subjects)}, "
                f"invalid_rows={invalid_rows}"
            )
        )

        if missing_subjects:
            preview = ", ".join(missing_subjects[:10])
            self.stdout.write(
                self.style.WARNING(
                    f"Missing subjects (first {min(len(missing_subjects), 10)}): {preview}"
                )
            )
