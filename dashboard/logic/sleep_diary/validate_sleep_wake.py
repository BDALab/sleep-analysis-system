import logging
import os.path

import pandas as pd

from dashboard.logic import cache
from dashboard.logic.features_extraction.utils import safe_div
from dashboard.logic.machine_learning.settings import prediction_name
from dashboard.models import SleepDiaryDay, WakeInterval, SleepNight, Subject, CsvData
from mysite.settings import BASE_DIR

logger = logging.getLogger(__name__)

THRESHOLDS = [round(th, 1) for th in [i / 10 for i in range(1, 11)]]


def validate_sleep_wake():
    nights = SleepNight.objects.all()

    validation_entries = []
    for night in nights:
        assert isinstance(night, SleepNight)
        df = _get_df(night)
        if df is None:
            logger.warning(f"Prediction data cannot be found for {night.date} of subject {night.subject.code}")
            continue
        day = night.diary_day
        if not isinstance(day, SleepDiaryDay):
            logger.warning(f"Sleep diary day cannot be resolved for night {night.id} (subject {night.subject.code})")
            continue
        validation_entries.append((night, day, df))

    if not validation_entries:
        logger.warning('No nights available for sleep/wake validation.')
        return None

    results = []
    for threshold in THRESHOLDS:
        metrics = _evaluate_threshold(validation_entries, threshold)
        if metrics['evaluated_nights'] == 0:
            logger.warning(f'Unable to evaluate threshold {threshold:.1f}: no matching diary data.')
            continue
        results.append(metrics)
        logger.info(
            f'Threshold {threshold:.1f} || TP: {metrics["TP"]} | FN: {metrics["FN"]} | '
            f'FP: {metrics["FP"]} | TN: {metrics["TN"]} || ACC: {metrics["accuracy"]:.2f}% | '
            f'SEN: {metrics["sensitivity"]:.2f}% | SPE: {metrics["specificity"]:.2f}% | '
            f'Nights: {metrics["evaluated_nights"]}')

    if not results:
        logger.warning('Sleep/wake validation failed for all thresholds.')
        return None

    result_table = pd.DataFrame(
        [
            {
                'threshold': res['threshold'],
                'accuracy': res['accuracy'],
                'sensitivity': res['sensitivity'],
                'specificity': res['specificity'],
                'true_positive': res['TP'],
                'false_positive': res['FP'],
                'true_negative': res['TN'],
                'false_negative': res['FN'],
                'evaluated_nights': res['evaluated_nights'],
            }
            for res in results
        ]
    ).sort_values('threshold')
    output_path = os.path.join(BASE_DIR, 'threshold_against_sleep_diary.xlsx')
    result_table.to_excel(output_path, index=False)
    logger.info(f'Threshold comparison exported to {output_path}')

    best_result = max(results, key=lambda res: res['accuracy'])
    logger.info(
        f'Best threshold {best_result["threshold"]:.1f} with ACC: {best_result["accuracy"]:.2f}% | '
        f'SEN: {best_result["sensitivity"]:.2f}% | SPE: {best_result["specificity"]:.2f}% | '
        f'Nights evaluated: {best_result["evaluated_nights"]}')

    _evaluate_threshold(validation_entries, best_result['threshold'], log_details=True)
    _report_subject_night_counts()

    return best_result


def _evaluate_threshold(entries, threshold, log_details=False):
    map_values = {1: 'S', 0: 'W'}
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0
    evaluated_nights = 0

    for night, day, df in entries:
        s = day.t1
        e = day.t4
        prediction = df.loc[s:e, [prediction_name]].copy()
        if prediction.empty:
            continue

        labelled_prediction = prediction.copy()
        labelled_prediction[prediction_name] = (
            (prediction[prediction_name] >= threshold).astype(int).map(map_values)
        )

        TP = TN = FP = FN = 0

        # in bed before sleep: sleep_time -> sleep_duration
        sleep, remaining_values = _select_interval(labelled_prediction, s, day.t2)
        TN += sleep.count('W')
        FP += sleep.count('S')

        # wakeup intervals during night
        for wake_interval in WakeInterval.objects.filter(sleep_diary_day=day).all():
            assert isinstance(wake_interval, WakeInterval)
            sleep, remaining_values = _select_interval(
                remaining_values,
                wake_interval.start_with_date,
                wake_interval.end_with_date
            )
            TN += sleep.count('W')
            FP += sleep.count('S')

        # after wake in bed: wake_time -> get_up_time
        sleep, remaining_values = _select_interval(remaining_values, day.t3, day.t4)
        TN += sleep.count('W')
        FP += sleep.count('S')

        # the rest of the night, so the sleep time
        sleep, remaining_values = _select_interval(remaining_values, s, e)
        TP += sleep.count('S')
        FN += sleep.count('W')

        if log_details:
            logger.info(
                f'Day {day.date} (threshold {threshold:.1f}) || TP: {TP} | FN: {FN} | FP: {FP} | TN: {TN} || '
                f'ACC: {safe_div(TN + TP, TP + TN + FP + FN) * 100:.2f}% | '
                f'SEN: {safe_div(TP, TP + FN) * 100:.2f}% | '
                f'SPE: {safe_div(TN, TN + FP) * 100:.2f}%'
            )

        total_TP += TP
        total_FN += FN
        total_FP += FP
        total_TN += TN
        evaluated_nights += 1

    accuracy = safe_div(total_TN + total_TP, total_TN + total_TP + total_FN + total_FP) * 100
    sensitivity = safe_div(total_TP, total_TP + total_FN) * 100
    specificity = safe_div(total_TN, total_TN + total_FP) * 100

    if log_details and evaluated_nights > 0:
        logger.info(
            f'Total results for threshold {threshold:.1f} || TP: {total_TP} | FN: {total_FN} | FP: {total_FP} | '
            f'TN: {total_TN} || ACC: {accuracy:.2f}% | SEN: {sensitivity:.2f}% | SPE: {specificity:.2f}%'
        )

    return {
        'threshold': threshold,
        'TP': total_TP,
        'FP': total_FP,
        'TN': total_TN,
        'FN': total_FN,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'evaluated_nights': evaluated_nights,
    }


def _report_subject_night_counts():
    logger.info('==================================')
    subjects = Subject.objects.filter().all()
    seven_nigh_subjects = []
    less_night_subjects = []
    for subject in subjects:
        data = CsvData.objects.filter(subject=subject).first()
        if data is not None and not data.training_data:
            nights = SleepNight.objects.filter(subject=subject).all()
            if len(nights) != 7:
                logger.info(f'Subject {subject.code} has {len(nights)} nights')
                less_night_subjects.append(subject)
            else:
                seven_nigh_subjects.append(subject)
    logger.info('==================================')
    logger.info(f'Seven nights subjects: {len(seven_nigh_subjects)}')
    logger.info(f'Less nights subjects: {len(less_night_subjects)}')


def _get_df(night):
    if os.path.exists(night.name):
        return pd.read_excel(night.name, index_col=0)
    if not os.path.exists(night.data.excel_prediction_path):
        if os.path.exists(night.data.cached_prediction_path):
            df = cache.load_obj(night.data.cached_prediction_path)
            df.to_excel(night.data.excel_prediction_path)
            _get_df(night)
        return None
    return pd.read_excel(night.data.excel_prediction_path, index_col=0, usecols='A,EE')


def _select_interval(prediction, start, end):
    sleep_time_duration = prediction[start:end]
    remaining_values = pd.concat([prediction[:start], prediction[end:]])
    sleep = sleep_time_duration[prediction_name].values.tolist()
    return sleep, remaining_values
