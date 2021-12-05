import logging

import pandas
import pandas as pd

from dashboard.logic.features_extraction.utils import safe_div
from dashboard.logic.machine_learning.settings import hilev_prediction, Algorithm, algorithm, prediction_name
from dashboard.logic.zangle.helper_functions import is_cached, get_split_path
from dashboard.models import SleepDiaryDay, WakeInterval, SleepNight

logger = logging.getLogger(__name__)


def validate_sleep_wake():
    nights = SleepNight.objects.all()
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0
    for night in nights:
        assert isinstance(night, SleepNight)
        df = _get_df(night)
        day = night.diary_day
        assert isinstance(day, SleepDiaryDay)
        s = day.t1
        e = day.t4
        prediction = df[s:e]
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # in bed before sleep: sleep_time -> sleep_duration
        sleep, remaining_values = _select_interval(prediction,
                                                   s,
                                                   day.t2)
        TN += sleep.count('W')
        FP += sleep.count('S')

        # wakeup intervals during night
        for wake_interval in WakeInterval.objects.filter(sleep_diary_day=day).all():
            assert isinstance(wake_interval, WakeInterval)
            sleep, remaining_values = _select_interval(remaining_values,
                                                       wake_interval.start_with_date,
                                                       wake_interval.end_with_date)
            TN += sleep.count('W')
            FP += sleep.count('S')

        # after wake in bed: wake_time -> get_up_time
        sleep, remaining_values = _select_interval(remaining_values,
                                                   day.t3,
                                                   day.t4)
        TN += sleep.count('W')
        FP += sleep.count('S')

        # the rest of the night, so the sleep time
        sleep, remaining_values = _select_interval(remaining_values,
                                                   s,
                                                   e)
        TP += sleep.count('S')
        FN += sleep.count('W')

        logger.info(
            f'day {day.date}  || TP: {TP} | FN: {FN} | FP: {FP} | TN: {FN} || '
            f'ACC: {safe_div(TN + TP, TP + TN + FP + FN) * 100}% | '
            f'SEN: {safe_div(TP, TP + FN) * 100}% | '
            f'SPE: {safe_div(TN, TN + FP) * 100}%')
        total_TP += TP
        total_FN += FN
        total_FP += FP
        total_TN += TN

    logger.info(
        f'Total results || TP: {total_TP} | FN: {total_FN} | FP: {total_FP} | TN: {total_TN} || '
        f'ACC: {safe_div(total_TN + total_TP, total_TN + total_TP + total_FN + total_FP) * 100}% | '
        f'SEN: {safe_div(total_TP, total_TP + total_FN) * 100}% | '
        f'SPE: {safe_div(total_TN, total_TN + total_FP) * 100}%')


def _get_df(night):
    if algorithm == Algorithm.XGBoost:
        return pd.read_excel(night.name, index_col=0)
    elif algorithm == Algorithm.ZAngle and night.data.training_data:
        df = pandas.read_excel(night.data.z_data_path, index_col='time stamp')
    elif algorithm == Algorithm.ZAngle and not night.data.training_data:
        if is_cached(night.data, night.diary_day, night.subject):
            df = pandas.read_excel(get_split_path(night.data, night.diary_day, night.subject), index_col='time stamp')
        else:
            return None
    return df


def _select_interval(prediction, start, end):
    sleep_time_duration = prediction[start:end]
    remaining_values = pd.concat([prediction[:start], prediction[end:]])
    sleep = sleep_time_duration[hilev_prediction].values.tolist() if algorithm == Algorithm.XGBoost \
        else sleep_time_duration[prediction_name].values.tolist()
    return sleep, remaining_values
