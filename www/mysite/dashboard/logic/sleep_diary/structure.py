import logging

from dashboard.models import SleepDiaryDay, CsvData

logger = logging.getLogger(__name__)


def create_structure():
    structure = []
    sleep_days_cluster = []
    ordered_days = SleepDiaryDay.objects.order_by('subject')
    previous_subject = ordered_days.first().subject
    for sleepDay in ordered_days.all():
        # Save cluster into dict by subject and find relevant csv data
        if previous_subject != sleepDay.subject:
            data = CsvData.objects.filter(subject=previous_subject).first()
            if data is None:
                logger.warning(
                    f'Missing csv data for subject {previous_subject} with {len(sleep_days_cluster)} sleep diary days')
            else:
                structure.append((previous_subject, data, sleep_days_cluster))
                logger.info(
                    f'Subject {previous_subject} added to validation structure with {len(sleep_days_cluster)} sleep diary days')
            sleep_days_cluster = []
            previous_subject = sleepDay.subject

        # Cluster sleepDays by subject
        sleep_days_cluster.append(sleepDay)
    data = CsvData.objects.filter(subject=previous_subject).first()
    structure.append((previous_subject, data, sleep_days_cluster))
    logger.info(
        f'Subject {previous_subject} added to structure with {len(sleep_days_cluster)} sleep diary days (Last subject)')
    logger.info('Validation structure created')
    return structure
