import logging
from datetime import datetime

import pandas
from django.db.models import Avg

from dashboard.logic.features_extraction.utils import safe_div
from dashboard.models import SleepNight, SleepDiaryDay, Subject

logger = logging.getLogger(__name__)


def export_all_features_avg():
    total_start = datetime.now()
    logger.info('Starting features export')

    columns = ['[1] Subject',
               '[1] Time in bed',
               '[1] Sleep onset latency',
               '[1] Sleep onset latency - norm',
               '[1] Wake after sleep onset',
               '[1] Wake after sleep onset - norm',
               '[1] Wake after sleep offset',
               '[1] Total sleep time',
               '[1] Wake bouts',
               '[1] Awakening > 5 minutes',
               '[1] Awakening > 5 minutes - norm',
               '[1] Sleep efficiency',
               '[1] Sleep efficiency - norm',
               '[1] Sleep fragmentation'
               ]

    df = pandas.DataFrame(gather_data_actigraph(), columns=columns)
    if df is None:
        return False
    df.to_excel('dataset-avg.xlsx')

    df = pandas.DataFrame(gather_data_sleep_diary(), columns=columns)
    if df is None:
        return False
    df.to_excel('dataset_diary-avg.xlsx')

    total_end = datetime.now()
    logger.info(f'Export took {total_end - total_start}')
    return True


def gather_data_actigraph():
    export_list = []
    for subject in Subject.objects.all():
        sleep_nights = SleepNight.objects.filter(subject=subject).all()
        if not sleep_nights:
            continue
        _create_row(export_list, sleep_nights, subject)
    return export_list


def gather_data_sleep_diary():
    export_list = []
    for subject in Subject.objects.all():
        sleep_nights = SleepDiaryDay.objects.filter(subject=subject).all()
        if not sleep_nights:
            continue
        _create_row(export_list, sleep_nights, subject)
    return export_list


def _create_row(export_list, sleep_nights, subject):
    row = [
        subject.code,
        _get_avg_property(sleep_nights, 'tib'),
        _get_avg_property(sleep_nights, 'sol'),
        _get_avg_norm(sleep_nights, 'sol_norm'),
        _get_avg_property(sleep_nights, 'waso'),
        _get_avg_norm(sleep_nights, 'waso_norm'),
        _get_avg_property(sleep_nights, 'wasf'),
        _get_avg_property(sleep_nights, 'tst'),
        _get_avg_property(sleep_nights, 'wb'),
        _get_avg_property(sleep_nights, 'awk5plus'),
        _get_avg_norm(sleep_nights, 'awk5plus_norm'),
        _get_avg_property(sleep_nights, 'se'),
        _get_avg_norm(sleep_nights, 'se_norm'),
        _get_avg_property(sleep_nights, 'sf'),
    ]
    export_list.append(row)


def _get_avg_property(sleep_nights, name):
    sum_num = 0
    for night in sleep_nights:
        sum_num = sum_num + getattr(night, name)
    return safe_div(sum_num, len(sleep_nights))


def _get_avg_norm(sleep_nights, name):
    sum_num = 0
    for night in sleep_nights:
        sum_num = sum_num + int(getattr(night, name))
    return safe_div(sum_num, len(sleep_nights))
