import logging
from datetime import timedelta
from os import path
from statistics import mean, median

import numpy
import pandas
import pytz
from pandas import DataFrame

from dashboard.logic import cache
from dashboard.logic.highlevel_features.highlevel_features_lists import HilevLists
from dashboard.logic.highlevel_features.highlevel_features_norms_lists import HilevNormLists
from dashboard.logic.machine_learning.settings import prediction_name, hilev_prediction, Algorithm, algorithm
from dashboard.logic.sleep_diary.structure import create_structure
from dashboard.logic.zangle.helper_functions import is_cached, get_split_path
from dashboard.models import CsvData, SleepDiaryDay, SleepNight, WakeInterval

logger = logging.getLogger(__name__)


def hilev(algrithm=None):
    structure = create_structure()
    res = True
    all_hilev = HilevLists()
    average_hilev = HilevLists()
    median_hilev = HilevLists()
    subject_hilev = HilevLists()
    all_norm_hilev = HilevNormLists()
    average_norm_hilev = HilevNormLists()
    median_norm_hilev = HilevNormLists()
    subject_norm_hilev = HilevNormLists()
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
        s = day.t1 - timedelta(minutes=30)
        e = day.t4 + timedelta(minutes=30)
        interval = df.loc[s:e, [prediction_name]]
        rolling_10 = interval.rolling('300s').sum()
        rolling_10['strict'] = numpy.where(rolling_10[prediction_name] <= 5, 'W', 'S')
        sleep = rolling_10.index[rolling_10['strict'] == 'S'].tolist()
        if not sleep:
            logger.warning(f'No sleep found for {night.subject.code} {night.diary_day.date} {night.data.filename}')
            res = False
            continue
        night.sleep_onset = pytz.timezone("Europe/Prague").localize(sleep[0])
        night.sleep_end = pytz.timezone("Europe/Prague").localize(sleep[-1])
        rolling_10[hilev_prediction] = numpy.where(rolling_10[prediction_name] <= 2, 'W', 'S')
        pred = rolling_10.loc[sleep[0]:sleep[-1], [hilev_prediction]]
        if not isinstance(pred, DataFrame):
            res = False
            continue
        pred.to_excel(night.name)
        wake = pred.index[pred[hilev_prediction] == 'W'].tolist()

        _count_hilevs(day, night, pred, sleep, wake)
        logger.info(night)
        night.save()

        all_hilev.IDs.append(f'{subject.code}_{day.date}')
        _assign_hilev(night, day, all_hilev)

        all_norm_hilev.IDs.append(f'{subject.code}_{day.date}')
        _assign_norm_hilev(night, all_norm_hilev)

        if ls != subject:
            _assign_average(average_hilev, subject, subject_hilev)
            _assign_median(median_hilev, subject, subject_hilev)
            _assign_norm_average(average_norm_hilev, subject, subject_norm_hilev)
            _assign_norm_median(median_norm_hilev, subject, subject_norm_hilev)
            ls = subject
            subject_hilev.clear()
            subject_norm_hilev.clear()

        _assign_hilev(night, day, subject_hilev)
        _assign_norm_hilev(night, subject_norm_hilev)

    _assign_average(average_hilev, subject, subject_hilev)
    _assign_median(median_hilev, subject, subject_hilev)
    _assign_norm_average(average_norm_hilev, subject, subject_norm_hilev)
    _assign_norm_median(median_norm_hilev, subject, subject_norm_hilev)

    all_hilev.to_data_frame().to_excel("hilevs.xlsx")
    average_hilev.to_data_frame().to_excel("average_hilevs.xlsx")
    median_hilev.to_data_frame().to_excel("median_hilevs.xlsx")

    all_norm_hilev.to_data_frame().to_excel("hilevs_norm.xlsx")
    average_norm_hilev.to_data_frame().to_excel("average_hilevs_norm.xlsx")
    median_norm_hilev.to_data_frame().to_excel("median_hilevs_norm.xlsx")

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


def _count_hilevs(day, night, pred, sleep, wake):
    night.tst = (night.sleep_end - night.sleep_onset).seconds
    night.waso = len(wake) * 30
    night.se = ((night.tst - night.waso) / night.tst) * 100
    night.sf = _count_sf(night, pred)
    night.sol = _count_sol(day, sleep)
    night.awk5plus = _count_awk5plus(pred)


def _count_sol(day, sleep):
    onset_latency = sleep[0] - day.t1 if sleep[0] > day.t1 else timedelta(seconds=0)
    return onset_latency.seconds


def _count_sf(night, pred):
    pred["number_prediction"] = numpy.where(pred[hilev_prediction] == 'S', 1, 0)
    wakes_counts = (pred["number_prediction"].diff() == -1).sum()
    sf = wakes_counts / (night.convert(night.tst).seconds / 3600)
    return sf


def _count_awk5plus(pred):
    awk5p = 0
    sleep_counter = 0
    for v in pred[hilev_prediction]:
        if v == 'S':
            sleep_counter += 1
            if sleep_counter == 10:
                awk5p += 1
        else:
            sleep_counter = 0
    return awk5p


def _count_dtst(day):
    wake = 0
    for interval in day.wake_intervals:
        if isinstance(interval, WakeInterval):
            wake += (interval.end_with_date - interval.start_with_date).seconds
    return (day.t3 - day.t2).seconds - wake


def _assign_hilev(night, day, subject_hilev):
    subject_hilev.TSTs.append(night.tst)
    subject_hilev.WASOs.append(night.waso)
    subject_hilev.SEs.append(night.se)
    subject_hilev.SFs.append(night.sf)
    subject_hilev.SOLs.append(night.sol)
    subject_hilev.WKS5.append(night.awk5plus)
    subject_hilev.DTSTs.append(_count_dtst(day))


def _assign_norm_hilev(night, subject_hilev):
    subject_hilev.SOLs.append(night.sol_norm.value)
    subject_hilev.WKS5.append(night.awk5plus_norm.value)
    subject_hilev.WASOs.append(night.waso_norm.value)
    subject_hilev.SEs.append(night.se_norm.value)


def _assign_average(average_hilev, subject, subject_hilev):
    average_hilev.IDs.append(subject.code)
    average_hilev.TSTs.append(mean(subject_hilev.TSTs))
    average_hilev.WASOs.append(mean(subject_hilev.WASOs))
    average_hilev.SEs.append(mean(subject_hilev.SEs))
    average_hilev.SFs.append(mean(subject_hilev.SFs))
    average_hilev.SOLs.append(mean(subject_hilev.SOLs))
    average_hilev.WKS5.append(mean(subject_hilev.WKS5))
    average_hilev.DTSTs.append(mean(subject_hilev.DTSTs))


def _assign_norm_average(average_hilev, subject, subject_hilev):
    average_hilev.IDs.append(subject.code)
    average_hilev.SOLs.append(mean(subject_hilev.SOLs))
    average_hilev.WKS5.append(mean(subject_hilev.WKS5))
    average_hilev.WASOs.append(mean(subject_hilev.WASOs))
    average_hilev.SEs.append(mean(subject_hilev.SEs))


def _assign_median(median_hilev, subject, subject_hilev):
    median_hilev.IDs.append(subject.code)
    median_hilev.TSTs.append(median(subject_hilev.TSTs))
    median_hilev.WASOs.append(median(subject_hilev.WASOs))
    median_hilev.SEs.append(median(subject_hilev.SEs))
    median_hilev.SFs.append(median(subject_hilev.SFs))
    median_hilev.SOLs.append(median(subject_hilev.SOLs))
    median_hilev.WKS5.append(median(subject_hilev.WKS5))
    median_hilev.DTSTs.append(median(subject_hilev.DTSTs))


def _assign_norm_median(median_hilev, subject, subject_hilev):
    median_hilev.IDs.append(subject.code)
    median_hilev.SOLs.append(median(subject_hilev.SOLs))
    median_hilev.WKS5.append(median(subject_hilev.WKS5))
    median_hilev.WASOs.append(median(subject_hilev.WASOs))
    median_hilev.SEs.append(median(subject_hilev.SEs))


def _create_night(data, day, subject):
    night = SleepNight()
    night.diary_day = day
    night.data = data
    night.subject = subject
    return night
