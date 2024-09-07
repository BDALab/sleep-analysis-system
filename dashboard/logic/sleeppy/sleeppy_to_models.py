import logging
import os
from datetime import datetime

import pandas as pd

from dashboard.models import CsvData, SleeppyData, SleepNight

logger = logging.getLogger(__name__)


def sleeppy_to_models():
    data = CsvData.objects.filter(training_data=False).all()
    for d in data:
        src_name = d.data.path.split("/")[-1][0:-4]
        sub_dir = (d.sleeppy_dir + "/" + src_name)
        if os.path.exists(sub_dir):
            results_dir = sub_dir + "/results"
            if not os.path.exists(results_dir):
                continue
            major_rest_periods_df = pd.read_csv(results_dir + f"/{src_name}_major_rest_periods.csv")
            sleep_endpoints_summary_df = pd.read_csv(results_dir + "/sleep_endpoints_summary.csv")
            df = pd.merge(major_rest_periods_df, sleep_endpoints_summary_df, on='day')
            for index, row in df.iterrows():
                available_hours = row['available_hours']
                if index == 0 and available_hours < 12:
                    continue  # Skip first day if it is just small timeframe
                major_rest_period = row['major_rest_period']

                if major_rest_period == '[]':
                    continue  # Skip days with empty major rest period

                # Remove the 'Timestamp' and parentheses from the string
                major_rest_period_cleaned = major_rest_period.replace("Timestamp('", "").replace("')", "")

                # Split the string into a list of datetime strings
                datetime_str_list = major_rest_period_cleaned.strip("[]").split(", ")

                # Convert each string to a pandas Timestamp object
                timestamps = [pd.Timestamp(ts_str, tz=datetime.now().astimezone().tzinfo) for ts_str in
                              datetime_str_list]

                if len(timestamps) != 2:
                    logger.warning(f"No sleep window found for {src_name}, day {row['day']}")
                    continue
                sleep_onset = timestamps[0]
                sleep_end = timestamps[1]
                sleep_night = _get_closest_to_sleep_onset(d, sleep_onset)

                sleeppydata = SleeppyData(
                    sleep_night=sleep_night,
                    data=d,
                    subject=d.subject,
                    sleep_onset=sleep_onset,
                    sleep_end=sleep_end,
                    sol=row['sleep_onset_latency'],
                    waso=row['waso'],
                    wb=row['number_wake_bouts'],
                    awk5plus=row['number_wake_bouts_5min'],
                    tst=row['total_sleep_time'],
                    se=row['percent_time_asleep'],
                )
                logger.debug(sleeppydata.info)
                if SleeppyData.objects.filter(sleep_night=sleep_night).first():
                    logger.info(f'Sleeppy data for {sleep_night.date}, {d.filename}, {d.subject.code} already exists')
                else:
                    sleeppydata.save()
                    logger.info(f'Sleeppy data for {sleep_night.date}, {d.filename}, {d.subject.code} created')


def _get_closest_to_sleep_onset(data, target):
    closest_greater_qs = SleepNight.objects.filter(data=data).filter(sleep_onset__gt=target).order_by('sleep_onset')
    closest_less_qs = SleepNight.objects.filter(data=data).filter(sleep_onset__lt=target).order_by('-sleep_onset')

    try:
        try:
            closest_greater = closest_greater_qs[0]
        except IndexError:
            return closest_less_qs[0]

        try:
            closest_less = closest_less_qs[0]
        except IndexError:
            return closest_greater_qs[0]
    except IndexError:
        raise SleepNight.objects.model.DoesNotExist("There is no closest object"
                                                    " because there are no objects.")

    if closest_greater.sleep_onset - target > target - closest_less.sleep_onset:
        return closest_less
    else:
        return closest_greater
