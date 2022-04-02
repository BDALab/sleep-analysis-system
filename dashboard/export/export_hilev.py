import logging
from datetime import datetime

import pandas

from dashboard.models import SleepNight, SleepDiaryDay

logger = logging.getLogger(__name__)


def export_all_features():
    total_start = datetime.now()
    logger.info('Starting features export')

    columns = ['Subject',
               'Date',
               'Probable Parkinson Disease',
               'Probable Mild Cognitive Impairment',
               'Healthy control',
               'Sleep Apnea',
               'Time in bed',
               'Sleep onset latency',
               'Sleep onset latency - norm',
               'Wake after sleep onset',
               'Wake after sleep onset - norm',
               'Wake after sleep offset',
               'Total sleep time',
               'Wake bouts',
               'Awakening > 5 minutes',
               'Awakening > 5 minutes - norm',
               'Sleep efficiency',
               'Sleep efficiency - norm',
               'Sleep fragmentation'
               ]

    columns_all = ['Subject',
                   'Date',
                   'Probable Parkinson Disease',
                   'Probable Mild Cognitive Impairment',
                   'Healthy control',
                   'Sleep Apnea',
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

    df = pandas.DataFrame(gather_data(SleepNight.objects.all()), columns=columns)
    if df is None:
        return False
    df.to_excel('dataset.xlsx')

    df = pandas.DataFrame(gather_data(SleepDiaryDay.objects.all()), columns=columns)
    if df is None:
        return False
    df.to_excel('dataset_diary.xlsx')

    df = pandas.DataFrame(gather_all_data(SleepNight.objects.all()), columns=columns_all)
    if df is None:
        return False
    df.to_excel('dataset_hilev_all.xlsx')

    total_end = datetime.now()
    logger.info(f'Export took {total_end - total_start}')
    return True


def gather_data(nights_or_diary_days):
    export_list = []
    for entry in nights_or_diary_days:
        row = [
            entry.subject.code,
            entry.date,
            entry.subject.pPD,
            entry.subject.pMCI,
            entry.subject.HC,
            entry.subject.SA,
            entry.tib,
            entry.sol,
            int(entry.sol_norm),
            entry.waso,
            int(entry.waso_norm),
            entry.wasf,
            entry.tst,
            entry.wb,
            entry.awk5plus,
            int(entry.awk5plus_norm),
            entry.se,
            int(entry.se_norm),
            entry.sf
        ]
        export_list.append(row)
    return export_list


def gather_all_data(nights):
    export_list = []
    for entry in nights:
        if entry.diary_day is not None:
            day = entry.diary_day
            row = [
                entry.subject.code,
                entry.date,

                entry.subject.pPD,
                entry.subject.pMCI,
                entry.subject.HC,
                entry.subject.SA,

                entry.tib,
                entry.sol,
                int(entry.sol_norm),
                entry.waso,
                int(entry.waso_norm),
                entry.wasf,
                entry.tst,
                entry.wb,
                entry.awk5plus,
                int(entry.awk5plus_norm),
                entry.se,
                int(entry.se_norm),
                entry.sf,

                day.tib,
                day.sol,
                int(day.sol_norm),
                day.waso,
                int(day.waso_norm),
                day.wasf,
                day.tst,
                day.wb,
                day.awk5plus,
                int(day.awk5plus_norm),
                day.se,
                int(day.se_norm),
                day.sf
            ]
            export_list.append(row)
    return export_list
