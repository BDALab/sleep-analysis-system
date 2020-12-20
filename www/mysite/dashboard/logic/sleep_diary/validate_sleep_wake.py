import logging
from os import path

import pandas as pd

from dashboard.logic import cache
from dashboard.logic.features_extraction.data_entry import safe_div
from dashboard.models import SleepDiaryDay, CsvData, WakeInterval

logger = logging.getLogger(__name__)


def validate_sleep_wake():
    structure = _create_validation_structure()
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0
    for subject, data, sleepDays in structure:
        if isinstance(data, CsvData) and path.exists(data.cached_prediction_path):
            logger.info(f'Validate subject {subject}')
            df = cache.load_obj(data.cached_prediction_path)
            for day in sleepDays:
                assert isinstance(day, SleepDiaryDay)
                s = day.with_date(day.sleep_time)
                e = day.with_date(day.get_up_time)
                prediction = df[s:e]
                TP = 0
                TN = 0
                FP = 0
                FN = 0

                # in bed before sleep: sleep_time -> sleep_duration
                sleep, remaining_values = _select_interval(prediction,
                                                           s,
                                                           day.with_date(day.sleep_duration))
                TN += sleep.count(0)
                FP += sleep.count(1)

                # wakeup intervals during night
                for wake_interval in WakeInterval.objects.filter(sleep_diary_day=day).all():
                    sleep, remaining_values = _select_interval(remaining_values,
                                                               day.with_date(wake_interval.start),
                                                               day.with_date(wake_interval.end))
                    TN += sleep.count(0)
                    FP += sleep.count(1)

                # after wake in bed: wake_time -> get_up_time
                sleep, remaining_values = _select_interval(remaining_values,
                                                           day.with_date(day.wake_time),
                                                           day.with_date(day.get_up_time))
                TN += sleep.count(0)
                FP += sleep.count(1)

                # the rest of the night, so the sleep time
                sleep, remaining_values = _select_interval(remaining_values,
                                                           s,
                                                           e)
                TP += sleep.count(1)
                FN += sleep.count(0)

                logger.info(
                    f'day {day.date}  || TP: {TP} | FN: {FN} | FP: {FP} | TN: {FN} || '
                    f'ACC: {safe_div(TN + TP, TP + TN + FP + FN) * 100}% | '
                    f'SEN: {safe_div(TP, TP + FN) * 100}% | '
                    f'SPE: {safe_div(TN, TN + FP) * 100}%')
                assert len(remaining_values) == 0
                total_TP += TP
                total_FN += FN
                total_FP += FP
                total_TN += TN
        else:
            logger.warning(f'Prediction data are not cached for subject {subject}')

    logger.info(
        f'Total results || TP: {total_TP} | FN: {total_FN} | FP: {total_FP} | TN: {total_TN} || '
        f'ACC: {safe_div(total_TN + total_TP, total_TN + total_TP + total_FN + total_FP) * 100}% | '
        f'SEN: {safe_div(total_TP, total_TP + total_FN) * 100}% | '
        f'SPE: {safe_div(total_TN, total_TN + total_FP) * 100}%')


def _select_interval(prediction, start, end):
    sleep_time_duration = prediction[start:end]
    remaining_values = pd.concat([prediction[:start], prediction[end:]])
    sleep = sleep_time_duration['SLEEP'].values.tolist()
    return sleep, remaining_values


def _create_validation_structure():
    structure = []
    sleep_days_cluster = []
    ordered_days = SleepDiaryDay.objects.order_by('subject')
    previous_subject = ordered_days.first().subject
    for sleepDay in ordered_days.all():
        # Save cluster into dict by subject and find relevant csv data
        if previous_subject != sleepDay.subject:
            data = CsvData.objects.filter(subject=previous_subject).first()
            if data is None:
                logger.warning(
                    f'Missing csv data for subject {previous_subject} with {len(sleep_days_cluster)} sleep diary days')
            else:
                structure.append((previous_subject, data, sleep_days_cluster))
                logger.info(
                    f'Subject {previous_subject} added to validation structure with {len(sleep_days_cluster)} sleep diary days')
            sleep_days_cluster = []
            previous_subject = sleepDay.subject

        # Cluster sleepDays by subject
        sleep_days_cluster.append(sleepDay)
    data = CsvData.objects.filter(subject=previous_subject).first()
    structure.append((previous_subject, data, sleep_days_cluster))
    logger.info(
        f'Subject {previous_subject} added to structure with {len(sleep_days_cluster)} sleep diary days (Last subject)')
    logger.info('Validation structure created')
    return structure
