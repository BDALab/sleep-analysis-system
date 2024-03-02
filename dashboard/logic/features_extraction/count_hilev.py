import logging
from datetime import timedelta
from os import path

import pytz
from pandas import DataFrame

from dashboard.logic import cache
from dashboard.logic.machine_learning.predict import predict
from dashboard.logic.machine_learning.settings import prediction_name
from dashboard.logic.sleep_diary.structure import create_structure_all
from dashboard.models import CsvData, SleepDiaryDay, SleepNight, WakeInterval

logger = logging.getLogger(__name__)


def hilev_all():
    structure = create_structure_all()
    hilev(structure)


def hilev(structure):
    res = True
    for subject, data, day in structure:
        if not isinstance(data, CsvData):
            res = False
            continue
        if not isinstance(day, SleepDiaryDay):
            res = False
            continue
        if not path.exists(data.cached_prediction_path):
            df = predict(data)
        else:
            df = _get_dataframe(data)
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
        try:
            night.save()
        except:
            logger.error(f'Could not save {night}')
        tib_interval.to_excel(night.name)

    return res


def _get_dataframe(data):
    return cache.load_obj(data.cached_prediction_path)


def _count_hilevs(day, night, tst_interval, sleep, wake):
    night.tib = (day.t4 - day.t1).seconds
    night.sol = (sleep[0] - day.t1).seconds
    night.waso = len(wake) * 30
    night.wasf = (day.t4 - sleep[-1]).seconds
    night.wb = ((tst_interval[prediction_name].diff() == -1).sum() - 1)
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
