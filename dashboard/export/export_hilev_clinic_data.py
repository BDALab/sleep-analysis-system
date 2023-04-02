import logging
from datetime import datetime

import pandas

from dashboard.logic.features_extraction.utils import safe_div
from dashboard.models import SleepNight, Subject

logger = logging.getLogger(__name__)


def export_all_features_clinic(avg=True):
    total_start = datetime.now()
    logger.info('Starting features export')

    columns = ['#Subject',
               '#Age',
               '#Gender',
               '#Disease',
               'actigraphy.Time in bed',
               'actigraphy.Sleep onset latency',
               'actigraphy_norm.Sleep onset latency',
               'actigraphy.Wake after sleep onset',
               'actigraphy_norm.Wake after sleep onset',
               'actigraphy.Wake after sleep offset',
               'actigraphy.Total sleep time',
               'actigraphy.Wake bouts',
               'actigraphy.Awakening > 5 minutes',
               'actigraphy_norm.Awakening > 5 minutes',
               'actigraphy.Sleep efficiency',
               'actigraphy_norm.Sleep efficiency',
               'actigraphy.Sleep fragmentation',
               'diary.Time in bed',
               'diary.Sleep onset latency',
               'diary_norm.Sleep onset latency',
               'diary.Wake after sleep onset',
               'diary_norm.Wake after sleep onset',
               'diary.Wake after sleep offset',
               'diary.Total sleep time',
               'diary.Wake bouts',
               'diary.Awakening > 5 minutes',
               'diary_norm.Awakening > 5 minutes',
               'diary.Sleep efficiency',
               'diary_norm.Sleep efficiency',
               'diary.Sleep fragmentation',
               ]

    if not avg:
        columns.insert(1, '#Date')
    df = pandas.DataFrame(gather_data(avg), columns=columns)
    if df is None:
        return False
    if avg:
        df.to_excel('dataset-avg-clinical.xlsx')
    else:
        df.to_excel('dataset-clinical.xlsx')

    total_end = datetime.now()
    logger.info(f'Export took {total_end - total_start}')
    return True


def gather_data(avg):
    export_list = []
    for subject in Subject.objects.order_by('code').all():
        sleep_nights = SleepNight.objects.filter(subject=subject).all()
        if not sleep_nights:
            continue
        if avg:
            _create_row(export_list, sleep_nights, subject)
        else:
            _create_rows(export_list, sleep_nights, subject)
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


def _create_rows(export_list, sleep_nights, subject):
    for night in sleep_nights:
        row = [
            subject.code,
            night.date,
            subject.age,
            subject.sex,
            subject.get_diagnosis_display(),
            getattr(night, 'tib'),
            getattr(night, 'sol'),
            int(getattr(night, 'sol_norm')),
            getattr(night, 'waso'),
            int(getattr(night, 'waso_norm')),
            getattr(night, 'wasf'),
            getattr(night, 'tst'),
            getattr(night, 'wb'),
            getattr(night, 'awk5plus'),
            int(getattr(night, 'awk5plus_norm')),
            getattr(night, 'se'),
            int(getattr(night, 'se_norm')),
            getattr(night, 'sf'),
            getattr(night.diary_day, 'tib'),
            getattr(night.diary_day, 'sol'),
            int(getattr(night.diary_day, 'sol_norm')),
            getattr(night.diary_day, 'waso'),
            int(getattr(night.diary_day, 'waso_norm')),
            getattr(night.diary_day, 'wasf'),
            getattr(night.diary_day, 'tst'),
            getattr(night.diary_day, 'wb'),
            getattr(night.diary_day, 'awk5plus'),
            int(getattr(night.diary_day, 'awk5plus_norm')),
            getattr(night.diary_day, 'se'),
            int(getattr(night.diary_day, 'se_norm')),
            getattr(night.diary_day, 'sf'),
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
