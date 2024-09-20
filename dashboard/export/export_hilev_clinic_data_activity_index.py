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
               'actigraphy.Time in bed',
               'actigraphy.Sleep onset latency',
               'actigraphy_norm.Sleep onset latency',
               'actigraphy.Wake after sleep onset',
               'actigraphy_norm.Wake after sleep onset',
               'actigraphy.Wake after sleep offset',
               'actigraphy.Total sleep time',
               'actigraphy.Wake bouts',
               'actigraphy.Awakening > 5 minutes',
               'actigraphy_norm.Awakening > 5 minutes',
               'actigraphy.Sleep efficiency',
               'actigraphy_norm.Sleep efficiency',
               'actigraphy.Sleep fragmentation',
               'diary.Time in bed',
               'diary.Sleep onset latency',
               'diary_norm.Sleep onset latency',
               'diary.Wake after sleep onset',
               'diary_norm.Wake after sleep onset',
               'diary.Wake after sleep offset',
               'diary.Total sleep time',
               'diary.Wake bouts',
               'diary.Awakening > 5 minutes',
               'diary_norm.Awakening > 5 minutes',
               'diary.Sleep efficiency',
               'diary_norm.Sleep efficiency',
               'diary.Sleep fragmentation',
               'activity.Max',
               'activity.Min',
               'activity.Relative Position of Max',
               'activity.Relative Position of Min',
               'activity.Range',
               'activity.Relative Range',
               'activity.Relative Variation Range',
               'activity.Interquartile Range',
               'activity.Relative Interquartile Range',
               'activity.Interdencile Range',
               'activity.Relative Interdencile Range',
               'activity.Interpercentile Range',
               'activity.Relative Interpercentile Range',
               'activity.Studentized Range',
               'activity.Mean',
               'activity.Harmonic Mean',
               'activity.Mean Excluding Outliers (10)',
               'activity.Mean Excluding Outliers (20)',
               'activity.Mean Excluding Outliers (30)',
               'activity.Mean Excluding Outliers (40)',
               'activity.Median',
               'activity.Mode',
               'activity.Variance',
               'activity.Standard Deviation',
               'activity.Median Absolute Deviation',
               'activity.Relative Standard Deviation',
               'activity.Index of Dispersion',
               'activity.Kurtosis',
               'activity.Skewness',
               'activity.Pearson 1st Skewness Coefficient',
               'activity.Pearson 2nd Skewness Coefficient',
               'activity.1st Percentile',
               'activity.5th Percentile',
               'activity.10th Percentile',
               'activity.20th Percentile',
               'activity.80th Percentile',
               'activity.90th Percentile',
               'activity.95th Percentile',
               'activity.99th Percentile',
               'activity.Shannon Entropy',
               'activity.Modulation',
               'activity.Teager Kaiser Energy Operator Max',
               'activity.Teager Kaiser Energy Operator Min'
               ]

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
