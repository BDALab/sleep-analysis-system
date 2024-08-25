import logging
import os

import pandas as pd

from dashboard.models import CsvData

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

                # Remove the 'Timestamp' and parentheses from the string
                major_rest_period_cleaned = major_rest_period.replace("Timestamp('", "").replace("')", "")

                # Split the string into a list of datetime strings
                datetime_str_list = major_rest_period_cleaned.strip("[]").split(", ")

                # Convert each string to a pandas Timestamp object
                timestamps = [pd.Timestamp(ts_str) for ts_str in datetime_str_list]

                if len(timestamps) != 2:
                    logger.warning(f"No sleep window found for {src_name}, day {row['day']}")
                    continue

                sleep_onset = timestamps[0]
                sleep_end = timestamps[1]
                sol = row['sleep_onset_latency']
                waso = row['waso']
