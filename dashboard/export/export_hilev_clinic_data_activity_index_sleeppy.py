import logging
from datetime import datetime

import pandas

from dashboard.models import SleepNight, Subject, SleeppyData, \
    SleeppyActivityIndexFeatures

logger = logging.getLogger(__name__)


def export_all_features_clinic_activity_index_sleepy():
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
               'sleeppy.Sleep onset latency',
               'sleeppy_norm.Sleep onset latency',
               'sleeppy.Wake after sleep onset',
               'sleeppy_norm.Wake after sleep onset',
               'sleeppy.Total sleep time',
               'sleeppy.Wake bouts',
               'sleeppy.Awakening > 5 minutes',
               'sleeppy_norm.Awakening > 5 minutes',
               'sleeppy.Sleep efficiency',
               'sleeppy_norm.Sleep efficiency',
               'sleeppy.Sleep fragmentation',
               'sleeppy_activity.Max',
               'sleeppy_activity.Min',
               'sleeppy_activity.Relative Position of Max',
               'sleeppy_activity.Relative Position of Min',
               'sleeppy_activity.Range',
               'sleeppy_activity.Relative Range',
               'sleeppy_activity.Relative Variation Range',
               'sleeppy_activity.Interquartile Range',
               'sleeppy_activity.Relative Interquartile Range',
               'sleeppy_activity.Interdencile Range',
               'sleeppy_activity.Relative Interdencile Range',
               'sleeppy_activity.Interpercentile Range',
               'sleeppy_activity.Relative Interpercentile Range',
               'sleeppy_activity.Studentized Range',
               'sleeppy_activity.Mean',
               'sleeppy_activity.Harmonic Mean',
               'sleeppy_activity.Mean Excluding Outliers (10)',
               'sleeppy_activity.Mean Excluding Outliers (20)',
               'sleeppy_activity.Mean Excluding Outliers (30)',
               'sleeppy_activity.Mean Excluding Outliers (40)',
               'sleeppy_activity.Median',
               'sleeppy_activity.Mode',
               'sleeppy_activity.Variance',
               'sleeppy_activity.Standard Deviation',
               'sleeppy_activity.Median Absolute Deviation',
               'sleeppy_activity.Relative Standard Deviation',
               'sleeppy_activity.Index of Dispersion',
               'sleeppy_activity.Kurtosis',
               'sleeppy_activity.Skewness',
               'sleeppy_activity.Pearson 1st Skewness Coefficient',
               'sleeppy_activity.Pearson 2nd Skewness Coefficient',
               'sleeppy_activity.1st Percentile',
               'sleeppy_activity.5th Percentile',
               'sleeppy_activity.10th Percentile',
               'sleeppy_activity.20th Percentile',
               'sleeppy_activity.80th Percentile',
               'sleeppy_activity.90th Percentile',
               'sleeppy_activity.95th Percentile',
               'sleeppy_activity.99th Percentile',
               'sleeppy_activity.Shannon Entropy',
               'sleeppy_activity.Modulation',
               'sleeppy_activity.Teager Kaiser Energy Operator Max',
               'sleeppy_activity.Teager Kaiser Energy Operator Min'
               ]

    df = pandas.DataFrame(gather_data(), columns=columns)
    if df is None:
        return False

    df.to_excel('dataset-clinical-sleeppy.xlsx')

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
        sleeppy_data = SleeppyData.objects.filter(sleep_night=night).all()
        for sleeppy in sleeppy_data:
            activity = SleeppyActivityIndexFeatures.objects.filter(sleeppy_data=sleeppy).first()
            if activity is None:
                logger.warning(
                    f'Activity index features not found for sleeppy data {sleeppy.data.filename} {sleeppy.date}')
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

                sleeppy.sol,
                int(sleeppy.sol_norm),
                sleeppy.waso,
                int(sleeppy.waso_norm),
                sleeppy.tst,
                sleeppy.wb,
                sleeppy.awk5plus,
                int(sleeppy.awk5plus),
                sleeppy.se,
                int(sleeppy.se_norm),
                sleeppy.sf,

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
