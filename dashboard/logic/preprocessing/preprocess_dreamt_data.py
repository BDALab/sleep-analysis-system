import csv
import logging
import math
from datetime import datetime, timedelta

import pandas as pd

from dashboard.logic.features_extraction.data_entry import DataEntry
from dashboard.logic.preprocessing.preprocess_csv_data import fix_csv_data

logger = logging.getLogger(__name__)


def _proprocess_dreamt_training_data(csv_object):
    """
    Preprocess DREAMT-format training CSV files.

    Expected columns (header row):
    processed_time,ACC_X_g,ACC_Y_g,ACC_Z_g,TEMP,sleep_binary

    - processed_time: "%Y-%m-%d %H:%M:%S.%f"
    - ACC_*_g: acceleration in g
    - TEMP: temperature in °C
    - sleep_binary: 0 (wake) / 1 (sleep)

    We aggregate samples into 15-second epochs, compute magnitude and z-angle
    per-sample arrays for each epoch (to match legacy processing), and label each
    epoch by majority vote of sleep_binary within the epoch (ties resolved by 0).
    The epoch "time" stored is the start timestamp of the 15-second window.
    """
    logger.info(f'DREAMT data will be preprocessed for {csv_object.filename}')
    # Fix potential null-bytes as we do for classic CSVs as a safety net
    fix_csv_data(csv_object)

    data_list = []
    start_time = datetime.now()

    def parse_time(s):
        try:
            return datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            # fallback if no microseconds
            return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

    with open(csv_object.data.path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='|')

        header = next(reader, None)
        if not header:
            logger.warning(f'Empty DREAMT CSV: {csv_object.filename}')
            return False

        # Map columns by name to indexes to be robust to order
        header_lower = [h.strip() for h in header]
        try:
            idx_time = header_lower.index('processed_time')
            idx_x = header_lower.index('acc_x_g')
            idx_y = header_lower.index('acc_y_g')
            idx_z = header_lower.index('acc_z_g')
            idx_temp = header_lower.index('temp')
            idx_sleep = header_lower.index('sleep_binary')
        except ValueError:
            # Try case-sensitive alternative (as provided in the example)
            try:
                idx_time = header.index('processed_time')
                idx_x = header.index('ACC_X_g')
                idx_y = header.index('ACC_Y_g')
                idx_z = header.index('ACC_Z_g')
                idx_temp = header.index('TEMP')
                idx_sleep = header.index('sleep_binary')
            except ValueError as e:
                logger.error(f'Unexpected DREAMT CSV header for {csv_object.filename}: {header}')
                return False

        # Initialize first window
        first_row = next(reader, None)
        if not first_row:
            logger.warning(f'No rows after header in DREAMT CSV: {csv_object.filename}')
            return False

        current_start = parse_time(first_row[idx_time])
        current_end = current_start + timedelta(seconds=15)

        acc_mag = []
        acc_z_angle = []
        temp = []
        sleep_flags = []

        def flush_epoch():
            nonlocal acc_mag, acc_z_angle, temp, sleep_flags, current_start
            if not acc_mag:
                return
            # Majority vote for sleep label (0/1). Tie -> 0 (wake)
            ones = sum(1 for sleep_flag in sleep_flags if sleep_flag == 1)
            zeros = len(sleep_flags) - ones
            sleep = 1 if ones > zeros else 0
            entry = DataEntry(
                time=current_start,
                sleep=sleep,
                acc=acc_mag,
                acc_z=acc_z_angle,
                temp=temp
            )
            data_list.append(entry.to_dic())
            # reset buffers
            acc_mag = []
            acc_z_angle = []
            temp = []
            sleep_flags = []

        # Process the first row together with the loop
        rows_iter = [first_row]
        rows_iter.extend(reader)

        for row in rows_iter:
            try:
                ts = parse_time(row[idx_time])
                x = float(row[idx_x])
                y = float(row[idx_y])
                z = float(row[idx_z])
                t = float(row[idx_temp])
                s = int(float(row[idx_sleep]))  # robust to '0.0'/'1.0'
            except Exception as e:
                logger.warning(f'Skipping malformed row in {csv_object.filename}: {row} | {e}')
                continue

            # Advance window(s) until the timestamp fits
            while ts >= current_end:
                flush_epoch()
                current_start = current_end
                current_end = current_start + timedelta(seconds=15)

            # Append sample to current buffers
            acc_mag.append(math.sqrt(x ** 2 + y ** 2 + z ** 2))
            # z-angle computed the same as legacy (_process_csv_data_core)
            denom = (x ** 2 + y ** 2) ** 0.5
            acc_z_angle.append(math.degrees(math.atan(z / denom)) if denom != 0 else 0.0)
            temp.append(t)
            sleep_flags.append(1 if s >= 1 else 0)

        # Flush the last epoch
        flush_epoch()

    if not data_list:
        logger.warning(f'No data aggregated for DREAMT CSV: {csv_object.filename}')
        return False

    df = pd.DataFrame.from_dict(data_list, orient='columns')
    df = df.set_index('Date')
    df.to_excel(csv_object.x_data_path)
    end_time = datetime.now()
    logger.info(f'DREAMT data {csv_object.filename} preprocessed in {end_time - start_time}')
    return True
