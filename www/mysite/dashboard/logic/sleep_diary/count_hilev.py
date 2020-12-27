import logging
from datetime import timedelta
from os import path

import numpy
from pandas import DataFrame

from dashboard.logic import cache
from dashboard.logic.machine_learning.settings import prediction_name
from dashboard.logic.sleep_diary.structure import create_structure
from dashboard.models import CsvData, SleepDiaryDay, SleepNight

logger = logging.getLogger(__name__)


def hilev():
    structure = create_structure()
    res = True
    for subject, data, sleepDays in structure:
        if not isinstance(data, CsvData) and path.exists(data.cached_prediction_path):
            res = False
            continue
        logger.info(f'Validate subject {subject}')
        df = cache.load_obj(data.cached_prediction_path)
        for day in sleepDays:
            if not isinstance(day, SleepDiaryDay):
                res = False
                continue
            if not isinstance(df, DataFrame):
                res = False
                continue
            nights = SleepNight.objects.filter(diary_day=day).filter(data=data).filter(subject=subject)
            if not nights.exists():
                night = SleepNight()
                night.diary_day = day
                night.data = data
                night.subject = subject
            else:
                night = nights.first()
            s = day.with_date(day.sleep_time)
            night.start = s - timedelta(hours=1)
            e = day.with_date(day.get_up_time)
            night.end = e + timedelta(hours=1)
            interval = df.loc[night.start:night.end, [prediction_name]]
            rolling_10 = interval.rolling('300s').sum()
            rolling_10['PREDICTION'] = numpy.where(rolling_10[prediction_name] > 6, 'S', 'W')
            sleep = rolling_10.index[rolling_10['PREDICTION'] == 'S'].tolist()
            if not sleep:
                logger.warning(f'No sleep found for {night.subject.code} {night.diary_day.date} {night.data.filename}')
                res = False
                continue
            night.sleep_onset = sleep[0]
            night.sleep_end = sleep[-1]
            pred = rolling_10.loc[night.sleep_onset:night.sleep_end, ['PREDICTION']]
            if not isinstance(pred, DataFrame):
                res = False
                continue
            pred.to_excel(night.name)
            night.tst = len(sleep) * 30
            wake = pred.index[pred['PREDICTION'] == 'W'].tolist()
            night.waso = len(wake) * 30
            night.se = len(sleep) / (len(sleep) + len(wake)) * 100
            logger.info(night)
            night.save()
    return res
