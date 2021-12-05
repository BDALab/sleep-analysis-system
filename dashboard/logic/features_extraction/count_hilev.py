import logging
from datetime import timedelta
from os import path

import numpy
import pandas
import pytz
from pandas import DataFrame

from dashboard.logic import cache
from dashboard.logic.machine_learning.settings import prediction_name, Algorithm, algorithm
from dashboard.logic.sleep_diary.structure import create_structure
from dashboard.logic.zangle.helper_functions import is_cached, get_split_path
from dashboard.models import CsvData, SleepDiaryDay, SleepNight, WakeInterval

logger = logging.getLogger(__name__)


def hilev(algrithm=None):
    structure = create_structure()
    res = True
    ls = structure[0][0]
    for subject, data, day in structure:
        if not isinstance(data, CsvData):
            res = False
            continue
        if algrithm == Algorithm.XGBoost and not path.exists(data.cached_prediction_path):
            res = False
            continue
        if algrithm == Algorithm.ZAngle and (not is_cached(data, day, subject) or not path.exists(data.z_data_path)):
            res = False
            continue
        if not isinstance(day, SleepDiaryDay):
            res = False
            continue

        df = _get_dataframe(data, day, subject)
        if not isinstance(df, DataFrame):
            res = False
            continue
        nights = SleepNight.objects.filter(diary_day=day).filter(data=data).filter(subject=subject)
        if not nights.exists():
            night = _create_night(data, day, subject)
        else:
            night = nights.first()
        s = day.t1 - timedelta(minutes=0)
        e = day.t4 + timedelta(minutes=0)
        tib_interval = df.loc[s:e, [prediction_name]]
        sleep = tib_interval.index[tib_interval[prediction_name] == 1].tolist()
        if not sleep:
            logger.warning(f'No sleep found for {night.subject.code} {night.diary_day.date} {night.data.filename}')
            res = False
            continue
        sleep_interval = df.loc[sleep[0]:sleep[-1], [prediction_name]]
        wake = sleep_interval.index[sleep_interval[prediction_name] == 0].tolist()
        night.sleep_onset = pytz.timezone("UTC").localize(sleep[0])
        night.sleep_end = pytz.timezone("UTC").localize(sleep[-1])
        tst_interval = df.loc[sleep[0]:sleep[-1], [prediction_name]]
        _count_hilevs(day, night, tst_interval, sleep, wake)
        logger.info(night)
        night.save()
        tib_interval.to_excel(night.name)

    return res


def _get_dataframe(data, day, subject):
    if algorithm == Algorithm.XGBoost:
        return cache.load_obj(data.cached_prediction_path)
    elif algorithm == Algorithm.ZAngle and data.training_data:
        df = pandas.read_excel(data.z_data_path, index_col='time stamp')
    elif algorithm == Algorithm.ZAngle and not data.training_data:
        if is_cached(data, day, subject):
            df = pandas.read_excel(get_split_path(data, day, subject), index_col='time stamp')
        else:
            return None
    df[prediction_name] = numpy.where(df[prediction_name] == 'S', 1, 0)
    return df


def _count_hilevs(day, night, tst_interval, sleep, wake):
    night.tib = (day.t4 - day.t1).seconds
    night.sol = (sleep[0] - day.t1).seconds
    night.waso = len(wake) * 30
    night.wasf = (day.t4 - sleep[-1]).seconds
    night.wb = (tst_interval[prediction_name].diff() == -1).sum()
    night.awk5plus = _count_awk5plus(tst_interval)


def _count_awk5plus(pred):
    awk5p = 0
    wake_counter = 0
    sleep_counter = 0
    for v in pred[prediction_name]:
        if v == 0:  # wake
            wake_counter += 1
            if wake_counter == 10 and sleep_counter >= 10:
                # 10 * 30s = 5minutes -> 5 minutes of wake without short disruption of sleep shorter then 5 minutes
                awk5p += 1
                sleep_counter = 0
        else:  # sleep
            sleep_counter += 1
            wake_counter = 0
    return awk5p


def _count_dtst(day):
    wake = 0
    for interval in day.wake_intervals:
        if isinstance(interval, WakeInterval):
            wake += (interval.end_with_date - interval.start_with_date).seconds
    return (day.t3 - day.t2).seconds - wake


def _create_night(data, day, subject):
    night = SleepNight()
    night.diary_day = day
    night.data = data
    night.subject = subject
    return night
