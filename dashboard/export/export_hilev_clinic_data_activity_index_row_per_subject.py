import logging
from datetime import datetime

import pandas

from dashboard.models import SleepNight, Subject, SleepNightActivityIndexFeatures

logger = logging.getLogger(__name__)


def export_all_features_clinic_activity_index():
    total_start = datetime.now()
    logger.info('Starting features export')

    columns = ['#Subject',
               '#Date',
               '#Age',
               '#Gender',
               '#Disease',
               ]

    for i in range(1, 8):
        columns.append()

    df = pandas.DataFrame(gather_data(), columns=columns)
    if df is None:
        return False

    df.to_excel('dataset-clinical-acc.xlsx')

    total_end = datetime.now()
    logger.info(f'Export took {total_end - total_start}')
    return True


def gather_data():
    export_list = []
    for subject in Subject.objects.order_by('code').all():
        sleep_nights = SleepNight.objects.filter(subject=subject).all()
        if not sleep_nights:
            continue
        _create_rows(export_list, sleep_nights, subject)
    return export_list


def _create_rows(export_list, sleep_nights, subject):
    for night in sleep_nights:
        activity = SleepNightActivityIndexFeatures.objects.filter(sleep_night=night).first()
        if activity is None:
            logger.warning(
                f'Activity index features not found for {subject.code} {night.date}')
            continue
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

            activity.max,
            activity.min,
            activity.relative_position_of_max,
            activity.relative_position_of_min,
            activity.range,
            activity.relative_range,
            activity.relative_variation_range,
            activity.interquartile_range,
            activity.relative_interquartile_range,
            activity.interdencile_range,
            activity.relative_interdencile_range,
            activity.interpercentile_range,
            activity.relative_interpercentile_range,
            activity.studentized_range,
            activity.mean,
            activity.harmonic_mean,
            activity.mean_excluding_outliers_10,
            activity.mean_excluding_outliers_20,
            activity.mean_excluding_outliers_30,
            activity.mean_excluding_outliers_40,
            activity.median,
            activity.mode,
            activity.variance,
            activity.standard_deviation,
            activity.median_absolute_deviation,
            activity.relative_standard_deviation,
            activity.index_of_dispersion,
            activity.kurtosis,
            activity.skewness,
            activity.pearson_1st_skewness_coefficient,
            activity.pearson_2st_skewness_coefficient,
            activity.percentile_1,
            activity.percentile_5,
            activity.percentile_10,
            activity.percentile_20,
            activity.percentile_80,
            activity.percentile_90,
            activity.percentile_95,
            activity.percentile_99,
            activity.shannon_entropy,
            activity.modulation,
            activity.tkeo_max,
            activity.tkeo_min,
        ]
        export_list.append(row)
