import logging
from datetime import datetime

import pandas

from dashboard.logic.features_extraction.utils import safe_div
from dashboard.models import SleepNight, Subject

logger = logging.getLogger(__name__)


def export_all_features_avg_clinic():
    total_start = datetime.now()
    logger.info('Starting features export')

    columns = ['#Subject',
               '#Age',
               '#Gender',
               '#Disease',
               'Time in bed (A)',
               'Sleep onset latency (A)',
               'Sleep onset latency - norm (A)',
               'Wake after sleep onset (A)',
               'Wake after sleep onset - norm (A)',
               'Wake after sleep offset (A)',
               'Total sleep time (A)',
               'Wake bouts (A)',
               'Awakening > 5 minutes (A)',
               'Awakening > 5 minutes - norm (A)',
               'Sleep efficiency (A)',
               'Sleep efficiency - norm (A)',
               'Sleep fragmentation (A)',
               'Time in bed (D)',
               'Sleep onset latency (D)',
               'Sleep onset latency - norm (D)',
               'Wake after sleep onset (D)',
               'Wake after sleep onset - norm (D)',
               'Wake after sleep offset (D)',
               'Total sleep time (D)',
               'Wake bouts (D)',
               'Awakening > 5 minutes (D)',
               'Awakening > 5 minutes - norm (D)',
               'Sleep efficiency (D)',
               'Sleep efficiency - norm (D)',
               'Sleep fragmentation (D)',
               ]

    df = pandas.DataFrame(gather_data(), columns=columns)
    if df is None:
        return False
    df.to_excel('dataset-avg-clinical.xlsx')

    total_end = datetime.now()
    logger.info(f'Export took {total_end - total_start}')
    return True


def gather_data():
    export_list = []
    for subject in Subject.objects.all():
        sleep_nights = SleepNight.objects.filter(subject=subject).all()
        if not sleep_nights:
            continue
        _create_row(export_list, sleep_nights, subject)
    return export_list


def _create_row(export_list, sleep_nights, subject):
    row = [
        subject.code,
        subject.age,
        subject.sex,
        subject.pPD,
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
        _get_avg_property_diary(sleep_nights, 'tib'),
        _get_avg_property_diary(sleep_nights, 'sol'),
        _get_avg_norm_diary(sleep_nights, 'sol_norm'),
        _get_avg_property_diary(sleep_nights, 'waso'),
        _get_avg_norm_diary(sleep_nights, 'waso_norm'),
        _get_avg_property_diary(sleep_nights, 'wasf'),
        _get_avg_property_diary(sleep_nights, 'tst'),
        _get_avg_property_diary(sleep_nights, 'wb'),
        _get_avg_property_diary(sleep_nights, 'awk5plus'),
        _get_avg_norm_diary(sleep_nights, 'awk5plus_norm'),
        _get_avg_property_diary(sleep_nights, 'se'),
        _get_avg_norm_diary(sleep_nights, 'se_norm'),
        _get_avg_property_diary(sleep_nights, 'sf'),
    ]
    export_list.append(row)


def _get_avg_property(sleep_nights, name):
    sum_num = 0
    for night in sleep_nights:
        sum_num = sum_num + getattr(night, name)
    return safe_div(sum_num, len(sleep_nights))


def _get_avg_property_diary(sleep_nights, name):
    sum_num = 0
    for night in sleep_nights:
        sum_num = sum_num + getattr(night.diary_day, name)
    return safe_div(sum_num, len(sleep_nights))


def _get_avg_norm(sleep_nights, name):
    sum_num = 0
    for night in sleep_nights:
        sum_num = sum_num + int(getattr(night, name))
    return safe_div(sum_num, len(sleep_nights))


def _get_avg_norm_diary(sleep_nights, name):
    sum_num = 0
    for night in sleep_nights:
        sum_num = sum_num + int(getattr(night.diary_day, name))
    return safe_div(sum_num, len(sleep_nights))
