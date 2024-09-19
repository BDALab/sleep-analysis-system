"""

z activity indexu high level features - jen z doby člověk spal

průměr
směrodatná odchylka
activity index -> tiger kaiser activity operator -> minimum
shanonova entropie

další high level feature co můžou vstupovat do dalších algoritmů

"""

import logging
import os.path
from datetime import datetime, timezone
from statistics import variance, stdev, mean, mode, median, harmonic_mean

import numpy as np
import pandas as pd
from numpy import quantile, percentile
from scipy.stats import iqr, trim_mean, median_abs_deviation, kurtosis, skew, entropy

from dashboard.logic.features_extraction.utils import safe_div
from dashboard.models import CsvData, SleepNight, SleeppyData, SleepNightActivityIndexFeatures, \
    SleeppyActivityIndexFeatures

logger = logging.getLogger(__name__)


def sleeppy_new_hilev_all():
    start = datetime.now()
    data = CsvData.objects.filter(training_data=False).all()
    logger.info(f'{len(data)} csv data objects will be used for Sleeppy new hilev features')
    result = True
    for d in data:
        if not sleeppy_hilev(d):
            result = False
    end = datetime.now()
    logger.info(f'Sleeppy new hilev features extraction of all the {len(data)} data took {end - start}')
    return result


def sleeppy_hilev(csv_data, force=False):
    if isinstance(csv_data, CsvData):
        start = datetime.now()
        logger.info(f'SleepPy hilev for {csv_data.filename}')
        try:
            # Check if we have sleeppy data or skip
            if not os.path.exists(csv_data.sleeppy_dir):
                logger.info(
                    f'Skipping hilev extraction for {csv_data.filename}, sleeppy results folder is not available')
                return True

            # get days
            src_name = csv_data.data.path.split("/")[-1][0:-4]  # save naming convention
            sub_dst = (csv_data.sleeppy_dir + "/" + src_name)
            days = sorted(
                [
                    sub_dst + "/activity_index_days/" + i
                    for i in os.listdir(sub_dst + "/activity_index_days/")
                    if ".DS_Store" not in i
                ]
            )

            count = 0

            # Get sleep nights for the data
            nights = SleepNight.objects.filter(data=csv_data).all()

            for day in days:
                count += 1
                df = pd.read_hdf(day)  # Time | activity_index

                # Skip the day if the activity index is missing
                if df is None:
                    logger.warning(f'Cannot load activity index for {day} of {csv_data.filename}')
                    continue

                # Searching for the right night from sleep nights
                for night in nights:

                    # Convert start and end of sleep to naive date
                    night_start_naive = night.sleep_onset.astimezone(timezone.utc).replace(tzinfo=None)
                    night_end_naive = night.sleep_end.astimezone(timezone.utc).replace(tzinfo=None)

                    # Get the sleep window
                    night_df = df.loc[night_start_naive:night_end_naive]

                    # If there was sleep window, so the sleep night is matching with the data, we will continue
                    if not night_df.empty:
                        # Prepare SleepNightActivityIndexFeatures object
                        night_model = SleepNightActivityIndexFeatures(
                            day_index=count + 1,
                            sleep_night=night
                        )

                        # Calculate signal features for the sleep window, the core logic
                        values_list = night_df['activity_index'].tolist()
                        night_model = _calculate_signal_features(values_list, night_model)

                        # Delete data from database if there are already, we do not want duplicities
                        (SleepNightActivityIndexFeatures.objects
                         .filter(day_index=count + 1, sleep_night=night)
                         .delete())

                        # Save the data
                        night_model.save()
                        logger.info(f'Created features from sleep night for night {count + 1}: {night.date}')

                        # We will do the same as for sleep night, but now with the window from sleeppy data
                        sleeppy_data = SleeppyData.objects.filter(sleep_night=night).first()
                        if sleeppy_data:

                            # Convert start and end of sleep to naive date
                            sp_start_naive = sleeppy_data.sleep_onset.astimezone(timezone.utc).replace(tzinfo=None)
                            sp_end_naive = sleeppy_data.sleep_end.astimezone(timezone.utc).replace(tzinfo=None)

                            # Get the sleep window
                            sleeppy_df = df.loc[sp_start_naive:sp_end_naive]

                            # If there was sleep window, so the sleeppy window is matching with the data, we will continue
                            if not sleeppy_df.empty:
                                # Prepare SleeppyActivityIndexFeatures object
                                sleeppy_model = SleeppyActivityIndexFeatures(
                                    day_index=count + 1,
                                    sleeppy_data=sleeppy_data
                                )

                                # Calculate signal features for the sleep window, the core logic
                                values_list_sp = sleeppy_df['activity_index'].tolist()
                                sleeppy_model = _calculate_signal_features(values_list_sp, sleeppy_model)

                                # Delete data from database if there are already, we do not want duplicities
                                (SleeppyActivityIndexFeatures.objects
                                 .filter(day_index=count + 1, sleeppy_data=sleeppy_data)
                                 .delete())

                                # Save the data
                                sleeppy_model.save()
                                logger.info(f'Created features from sleeppy for night {count + 1}: {night.date}')
                        # Continue with new data file, there could be only one correct night, so if it was found, move on
                        break

        except Exception as e:
            logger.error(f'{csv_data.filename} failed due to {e}')
            return False
        end = datetime.now()
        logger.info(f'SleepPy for {csv_data.filename} made in {end - start}')
        return True


def _tiger_kaiser_activity_operator_vector(time_series):
    # Initialize field for results
    KA = np.zeros_like(time_series)

    for t in range(1, len(time_series) - 1):
        KA[t] = abs(time_series[t]) ** 2 - abs(time_series[t + 1] * time_series[t - 1])

    return KA


def _calculate_signal_features(vec, model):
    # Prepare reusable features
    _len = len(vec)
    _max = max(vec)
    _pos_max = vec.index(_max)
    _min = min(vec)
    _pos_min = vec.index(_min)
    _range = abs(_max - _min)
    _var = variance(vec)
    _std = stdev(vec)
    _mean = mean(vec)
    _mode = mode(vec)
    _median = median(vec)
    _tkeo = _tiger_kaiser_activity_operator_vector(vec)

    # Fill model fields
    model.max = _max
    model.min = _min
    model.relative_position_of_max = safe_div(_pos_max, _len)
    model.relative_position_of_min = safe_div(_pos_min, _len)
    model.range = _range
    model.relative_range = safe_div(_range, _max)
    model.relative_variation_range = safe_div(_range, _mean)
    model.interquartile_range = iqr(vec)
    model.relative_interquartile_range = safe_div(iqr(vec), _max)
    model.interdencile_range = quantile(vec, 0.9) - quantile(vec, 0.1)
    model.relative_interdencile_range = safe_div(quantile(vec, 0.9) - quantile(vec, 0.1), _max)
    model.interpercentile_range = quantile(vec, 0.99) - quantile(vec, 0.01)
    model.relative_interpercentile_range = safe_div(quantile(vec, 0.99) - quantile(vec, 0.01), _max)
    model.studentized_range = safe_div(_range, _var)
    model.mean = _mean
    model.harmonic_mean = harmonic_mean(vec)
    model.mean_excluding_outliers_10 = trim_mean(vec, 0.1)
    model.mean_excluding_outliers_20 = trim_mean(vec, 0.2)
    model.mean_excluding_outliers_30 = trim_mean(vec, 0.3)
    model.mean_excluding_outliers_40 = trim_mean(vec, 0.4)
    model.median = _median
    model.mode = _mode
    model.variance = _var
    model.standard_deviation = _std
    model.median_absolute_deviation = median_abs_deviation(vec)
    model.relative_standard_deviation = safe_div(_std, _mean)
    model.index_of_dispersion = safe_div(_var, _mean)
    model.kurtosis = kurtosis(vec)
    model.skewness = skew(vec)
    model.pearson_1st_skewness_coefficient = safe_div((3 * (_mean - _mode)), _std)
    model.pearson_2st_skewness_coefficient = safe_div(3 * (_mean - _median), _std)
    model.percentile_1 = percentile(vec, 1)
    model.percentile_5 = percentile(vec, 5)
    model.percentile_10 = percentile(vec, 10)
    model.percentile_20 = percentile(vec, 20)
    model.percentile_80 = percentile(vec, 80)
    model.percentile_90 = percentile(vec, 90)
    model.percentile_95 = percentile(vec, 95)
    model.percentile_99 = percentile(vec, 99)
    model.shannon_entropy = entropy(vec)
    model.modulation = _range / (_max + _min)
    model.tkeo_max = np.max(_tkeo)
    model.tkeo_min = np.min(_tkeo)

    return model
