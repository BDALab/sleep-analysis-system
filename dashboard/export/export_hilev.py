import logging
from datetime import datetime

import pandas

from dashboard.models import SleepNight, SleepDiaryDay, SleeppyActivityIndexFeatures, SleepNightActivityIndexFeatures, \
    SleeppyData

logger = logging.getLogger(__name__)


def export_all_features():
    total_start = datetime.now()
    logger.info('Starting features export')

    columns = ['Subject',
               'Date',
               'Probable Parkinson Disease',
               'Probable Mild Cognitive Impairment',
               'Healthy control',
               'Sleep Apnea',
               'Time in bed',
               'Sleep onset latency',
               'Sleep onset latency - norm',
               'Wake after sleep onset',
               'Wake after sleep onset - norm',
               'Wake after sleep offset',
               'Total sleep time',
               'Wake bouts',
               'Awakening > 5 minutes',
               'Awakening > 5 minutes - norm',
               'Sleep efficiency',
               'Sleep efficiency - norm',
               'Sleep fragmentation'
               ]

    columns_all = ['Subject',
                   'Date',
                   'Probable Parkinson Disease',
                   'Probable Mild Cognitive Impairment',
                   'Healthy control',
                   'Sleep Apnea',

                   'Time in bed (A)',
                   'Sleep onset latency (A)',
                   'Sleep onset latency - norm (A)',
                   'Wake after sleep onset (A)',
                   'Wake after sleep onset - norm (A)',
                   'Wake after sleep offset (A)',
                   'Total sleep time (A)',
                   'Wake bouts (A)',
                   'Awakening > 5 minutes (A)',
                   'Awakening > 5 minutes - norm (A)',
                   'Sleep efficiency (A)',
                   'Sleep efficiency - norm (A)',
                   'Sleep fragmentation (A)',

                   'Time in bed (D)',
                   'Sleep onset latency (D)',
                   'Sleep onset latency - norm (D)',
                   'Wake after sleep onset (D)',
                   'Wake after sleep onset - norm (D)',
                   'Wake after sleep offset (D)',
                   'Total sleep time (D)',
                   'Wake bouts (D)',
                   'Awakening > 5 minutes (D)',
                   'Awakening > 5 minutes - norm (D)',
                   'Sleep efficiency (D)',
                   'Sleep efficiency - norm (D)',
                   'Sleep fragmentation (D)',
                   ]

    columns_hilev_act_all = ['Subject',
                             'Date',
                             'Probable Parkinson Disease',
                             'Probable Mild Cognitive Impairment',
                             'Healthy control',
                             'Sleep Apnea',

                             'Time in bed (A)',
                             'Sleep onset latency (A)',
                             'Sleep onset latency - norm (A)',
                             'Wake after sleep onset (A)',
                             'Wake after sleep onset - norm (A)',
                             'Wake after sleep offset (A)',
                             'Total sleep time (A)',
                             'Wake bouts (A)',
                             'Awakening > 5 minutes (A)',
                             'Awakening > 5 minutes - norm (A)',
                             'Sleep efficiency (A)',
                             'Sleep efficiency - norm (A)',
                             'Sleep fragmentation (A)',

                             'Time in bed (D)',
                             'Sleep onset latency (D)',
                             'Sleep onset latency - norm (D)',
                             'Wake after sleep onset (D)',
                             'Wake after sleep onset - norm (D)',
                             'Wake after sleep offset (D)',
                             'Total sleep time (D)',
                             'Wake bouts (D)',
                             'Awakening > 5 minutes (D)',
                             'Awakening > 5 minutes - norm (D)',
                             'Sleep efficiency (D)',
                             'Sleep efficiency - norm (D)',
                             'Sleep fragmentation (D)',

                             'Max',
                             'Min',
                             'Relative Position of Max',
                             'Relative Position of Min',
                             'Range',
                             'Relative Range',
                             'Relative Variation Range',
                             'Interquartile Range',
                             'Relative Interquartile Range',
                             'Interdencile Range',
                             'Relative Interdencile Range',
                             'Interpercentile Range',
                             'Relative Interpercentile Range',
                             'Studentized Range',
                             'Mean',
                             'Harmonic Mean',
                             'Mean Excluding Outliers (10)',
                             'Mean Excluding Outliers (20)',
                             'Mean Excluding Outliers (30)',
                             'Mean Excluding Outliers (40)',
                             'Median',
                             'Mode',
                             'Variance',
                             'Standard Deviation',
                             'Median Absolute Deviation',
                             'Relative Standard Deviation',
                             'Index of Dispersion',
                             'Kurtosis',
                             'Skewness',
                             'Pearson 1st Skewness Coefficient',
                             'Pearson 2nd Skewness Coefficient',
                             '1st Percentile',
                             '5th Percentile',
                             '10th Percentile',
                             '20th Percentile',
                             '80th Percentile',
                             '90th Percentile',
                             '95th Percentile',
                             '99th Percentile',
                             'Shannon Entropy',
                             'Modulation',
                             'Teager Kaiser Energy Operator Max',
                             'Teager Kaiser Energy Operator Min'
                             ]

    columns_hilev_sleeppy_all = ['Subject',
                                 'Date',
                                 'Probable Parkinson Disease',
                                 'Probable Mild Cognitive Impairment',
                                 'Healthy control',
                                 'Sleep Apnea',

                                 'Time in bed (A)',
                                 'Sleep onset latency (A)',
                                 'Sleep onset latency - norm (A)',
                                 'Wake after sleep onset (A)',
                                 'Wake after sleep onset - norm (A)',
                                 'Wake after sleep offset (A)',
                                 'Total sleep time (A)',
                                 'Wake bouts (A)',
                                 'Awakening > 5 minutes (A)',
                                 'Awakening > 5 minutes - norm (A)',
                                 'Sleep efficiency (A)',
                                 'Sleep efficiency - norm (A)',
                                 'Sleep fragmentation (A)',

                                 'Time in bed (D)',
                                 'Sleep onset latency (D)',
                                 'Sleep onset latency - norm (D)',
                                 'Wake after sleep onset (D)',
                                 'Wake after sleep onset - norm (D)',
                                 'Wake after sleep offset (D)',
                                 'Total sleep time (D)',
                                 'Wake bouts (D)',
                                 'Awakening > 5 minutes (D)',
                                 'Awakening > 5 minutes - norm (D)',
                                 'Sleep efficiency (D)',
                                 'Sleep efficiency - norm (D)',
                                 'Sleep fragmentation (D)',

                                 'Sleep onset latency (S)',
                                 'Sleep onset latency - norm (S)',
                                 'Wake after sleep onset (S)',
                                 'Wake after sleep onset - norm (S)',
                                 'Total sleep time (S)',
                                 'Wake bouts (S)',
                                 'Awakening > 5 minutes (S)',
                                 'Awakening > 5 minutes - norm (S)',
                                 'Sleep efficiency (S)',
                                 'Sleep efficiency - norm (S)',
                                 'Sleep fragmentation (S)',

                                 'Max',
                                 'Min',
                                 'Relative Position of Max',
                                 'Relative Position of Min',
                                 'Range',
                                 'Relative Range',
                                 'Relative Variation Range',
                                 'Interquartile Range',
                                 'Relative Interquartile Range',
                                 'Interdencile Range',
                                 'Relative Interdencile Range',
                                 'Interpercentile Range',
                                 'Relative Interpercentile Range',
                                 'Studentized Range',
                                 'Mean',
                                 'Harmonic Mean',
                                 'Mean Excluding Outliers (10)',
                                 'Mean Excluding Outliers (20)',
                                 'Mean Excluding Outliers (30)',
                                 'Mean Excluding Outliers (40)',
                                 'Median',
                                 'Mode',
                                 'Variance',
                                 'Standard Deviation',
                                 'Median Absolute Deviation',
                                 'Relative Standard Deviation',
                                 'Index of Dispersion',
                                 'Kurtosis',
                                 'Skewness',
                                 'Pearson 1st Skewness Coefficient',
                                 'Pearson 2nd Skewness Coefficient',
                                 '1st Percentile',
                                 '5th Percentile',
                                 '10th Percentile',
                                 '20th Percentile',
                                 '80th Percentile',
                                 '90th Percentile',
                                 '95th Percentile',
                                 '99th Percentile',
                                 'Shannon Entropy',
                                 'Modulation',
                                 'Teager Kaiser Energy Operator Max',
                                 'Teager Kaiser Energy Operator Min'
                                 ]

    df = pandas.DataFrame(gather_data(SleepNight.objects.all()), columns=columns)
    if df is None:
        return False
    df.to_excel('dataset.xlsx')

    df = pandas.DataFrame(gather_data(SleepDiaryDay.objects.all()), columns=columns)
    if df is None:
        return False
    df.to_excel('dataset_diary.xlsx')

    df = pandas.DataFrame(gather_all_data(SleepNight.objects.all()), columns=columns_all)
    if df is None:
        return False
    df.to_excel('dataset_hilev_all.xlsx')

    df = pandas.DataFrame(gather_all_data_acc(SleepNight.objects.all()), columns=columns_hilev_act_all)
    if df is None:
        return False
    df.to_excel('dataset_hilev_all_acc_act.xlsx')

    df = pandas.DataFrame(gather_all_data_sleeppy(SleepNight.objects.all()), columns=columns_hilev_sleeppy_all)
    if df is None:
        return False
    df.to_excel('dataset_hilev_all_acc_sleeppy.xlsx')

    total_end = datetime.now()
    logger.info(f'Export took {total_end - total_start}')
    return True


def gather_data(nights_or_diary_days):
    export_list = []
    for entry in nights_or_diary_days:
        row = [
            entry.subject.code,
            entry.date,
            entry.subject.pPD,
            entry.subject.pMCI,
            entry.subject.HC,
            entry.subject.SA,
            entry.tib,
            entry.sol,
            int(entry.sol_norm),
            entry.waso,
            int(entry.waso_norm),
            entry.wasf,
            entry.tst,
            entry.wb,
            entry.awk5plus,
            int(entry.awk5plus_norm),
            entry.se,
            int(entry.se_norm),
            entry.sf
        ]
        export_list.append(row)
    return export_list


def gather_all_data(nights):
    export_list = []
    for entry in nights:
        if entry.diary_day is not None:
            day = entry.diary_day
            row = [
                entry.subject.code,
                entry.date,

                entry.subject.pPD,
                entry.subject.pMCI,
                entry.subject.HC,
                entry.subject.SA,

                entry.tib,
                entry.sol,
                int(entry.sol_norm),
                entry.waso,
                int(entry.waso_norm),
                entry.wasf,
                entry.tst,
                entry.wb,
                entry.awk5plus,
                int(entry.awk5plus_norm),
                entry.se,
                int(entry.se_norm),
                entry.sf,

                day.tib,
                day.sol,
                int(day.sol_norm),
                day.waso,
                int(day.waso_norm),
                day.wasf,
                day.tst,
                day.wb,
                day.awk5plus,
                int(day.awk5plus_norm),
                day.se,
                int(day.se_norm),
                day.sf
            ]
            export_list.append(row)
    return export_list


def gather_all_data_acc(nights):
    export_list = []
    for entry in nights:
        if entry.diary_day is not None:
            day = entry.diary_day
            activity_index = SleepNightActivityIndexFeatures.objects.filter(sleep_night=entry).all()
            for activity in activity_index:
                row = _create_activity_row(activity, day, entry)
                export_list.append(row)
    return export_list


def gather_all_data_sleeppy(nights):
    export_list = []
    for entry in nights:
        if entry.diary_day is not None:
            day = entry.diary_day
            sleeppy_data = SleeppyData.objects.filter(sleep_night=entry).all()
            for sleeppy in sleeppy_data:
                activity = SleeppyActivityIndexFeatures.objects.filter(sleeppy_data=sleeppy).first()
                if activity is None:
                    logger.warning(
                        f'Activity index features not found for sleeppy data {sleeppy.data.filename} {sleeppy.date}')
                    continue
                row = _create_activity_row_sleeppy(activity, day, sleeppy, entry)
                export_list.append(row)
    return export_list


def _create_activity_row(activity, day, entry):
    row = [
        entry.subject.code,
        entry.date,

        entry.subject.pPD,
        entry.subject.pMCI,
        entry.subject.HC,
        entry.subject.SA,

        entry.tib,
        entry.sol,
        int(entry.sol_norm),
        entry.waso,
        int(entry.waso_norm),
        entry.wasf,
        entry.tst,
        entry.wb,
        entry.awk5plus,
        int(entry.awk5plus_norm),
        entry.se,
        int(entry.se_norm),
        entry.sf,

        day.tib,
        day.sol,
        int(day.sol_norm),
        day.waso,
        int(day.waso_norm),
        day.wasf,
        day.tst,
        day.wb,
        day.awk5plus,
        int(day.awk5plus_norm),
        day.se,
        int(day.se_norm),
        day.sf,

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
    return row


def _create_activity_row_sleeppy(activity, day, sleeppy, entry):
    row = [
        entry.subject.code,
        entry.date,

        entry.subject.pPD,
        entry.subject.pMCI,
        entry.subject.HC,
        entry.subject.SA,

        entry.tib,
        entry.sol,
        int(entry.sol_norm),
        entry.waso,
        int(entry.waso_norm),
        entry.wasf,
        entry.tst,
        entry.wb,
        entry.awk5plus,
        int(entry.awk5plus_norm),
        entry.se,
        int(entry.se_norm),
        entry.sf,

        day.tib,
        day.sol,
        int(day.sol_norm),
        day.waso,
        int(day.waso_norm),
        day.wasf,
        day.tst,
        day.wb,
        day.awk5plus,
        int(day.awk5plus_norm),
        day.se,
        int(day.se_norm),
        day.sf,

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
    return row
