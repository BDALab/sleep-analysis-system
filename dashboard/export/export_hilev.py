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
               'Hemicrania Continua',
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

    df = pandas.DataFrame(gather_data(SleepNight.objects.all()), columns=columns)
    if df is None:
        return False
    df.to_excel('dataset.xlsx')

    df = pandas.DataFrame(gather_data(SleepDiaryDay.objects.all()), columns=columns)
    if df is None:
        return False
    df.to_excel('dataset_diary.xlsx')

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
