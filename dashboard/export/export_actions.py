import logging
from datetime import datetime

from dashboard.export.export import Export
from dashboard.logic.multithread import parallel_for_with_param
from dashboard.logic.reults_visualization.sleep_graph import create_graph
from dashboard.models import Subject, SleepNight, SleepDiaryDay

logger = logging.getLogger(__name__)


def export_all(export_user):
    if export_user is None:
        return
    total_start = datetime.now()
    subjects = Subject.objects.all()
    logger.info(f'{len(subjects)} subjects will be exported')
    results = parallel_for_with_param(subjects, export_subject, export_user)

    total_end = datetime.now()
    logger.info(f'For {len(subjects)} subjects export took {total_end - total_start}')

    for r in results:
        if not r.result():
            return False
    return True


def export_subject(subject, export_user):
    if SleepNight.objects.filter(subject=subject).exists() or SleepDiaryDay.objects.filter(subject=subject).exists():
        export = Export(subject, export_user)
        export.prepare_export_folder()
        sleep_nights = SleepNight.objects.filter(subject=subject)
        if sleep_nights.exists():
            for night in sleep_nights:
                _, fig = create_graph(night)
                export.export_figure(fig, night.diary_day.date, night)
            export.export_sleep_nights(sleep_nights)
        elif SleepDiaryDay.objects.filter(subject=subject).exists():
            diary = SleepDiaryDay.objects.filter(subject=subject)
            export.export_sleep_nights(diary)
        export.export_to_pdf_and_cleanup()
        logger.info(f'Details of subject {subject.code} exported for user {export_user}')
    else:
        logger.warning(f'There is nothing to export for subject {subject.code}, sleep diary is missing')
