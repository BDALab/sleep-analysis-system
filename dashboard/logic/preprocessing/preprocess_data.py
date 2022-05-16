import csv
import logging
import math
import os
from datetime import timedelta, datetime

import pandas as pd

from dashboard.logic.features_extraction.data_entry import DataEntry
from dashboard.models import PsData, CsvData, SleepDiaryDay
from .preprocess_csv_data import fix_csv_data, get_csv_start, convert_csv_time
from .preprocess_ps_data import get_ps_start, convert_ps_timestamp, convert_sleep
from ..machine_learning.predict_core import predict_core

logger = logging.getLogger(__name__)


def preprocess_all_data():
    total_start = datetime.now()

    data = CsvData.objects.all()
    for d in data:
        preprocess_data(d)
    logger.info(f'{len(data)} training csv data objects preprocessed in {datetime.now() - total_start}')
    return True


def preprocess_data(csv_object):
    if isinstance(csv_object, CsvData):
        if os.path.exists(csv_object.x_data_path):
            return True
        elif csv_object.training_data:
            return _preprocess_training_data(csv_object)
        return _preprocess_prediction_data(csv_object)

    else:
        logger.warning(f'Wrong data type {type(csv_object)} was passed into preprocessing method.')
        return False


def _preprocess_training_data(csv_object):
    logger.info(f'Data will be preprocessed for {csv_object.filename}')
    ps_object = PsData.objects.filter(csv_data=csv_object).first()
    if not isinstance(ps_object, PsData) or not fix_csv_data(csv_object):
        return False
    else:
        start_time = datetime.now()
        start = _find_start(csv_object, ps_object)
        data_list = []
        with open(ps_object.data.path, 'r') as ps_file:
            ps_reader = csv.reader(ps_file, delimiter='\t', quotechar='|')

            frequency_modulo = 0
            modulo_reminder = 0

            with open(csv_object.data.path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

                header_end = False
                after_midnight = False

                for csv_row in csv_reader:
                    if frequency_modulo == 0:
                        frequency_modulo = _assign_frequency_modulo(csv_row)
                    # now I care just about data, which starts with timestamp starting with 20 --> begin of year
                    if len(csv_row) > 0 and csv_row[0].startswith('20'):
                        if modulo_reminder == 0:
                            if convert_csv_time(csv_row) >= start:
                                break
                        modulo_reminder = (modulo_reminder + 1) % frequency_modulo

                for ps_row in ps_reader:
                    if header_end and ps_row != []:
                        if ps_row[2].startswith('00'):
                            after_midnight = True
                        if convert_ps_timestamp(date, ps_row[2], after_midnight) >= start - timedelta(seconds=15):
                            time = convert_ps_timestamp(date, ps_row[2], after_midnight)
                            end = time + timedelta(seconds=15)
                            magnitude_data, z_angle_data, temp = _process_csv_data_core(csv_reader,
                                                                                        end,
                                                                                        frequency_modulo,
                                                                                        modulo_reminder)
                            if not magnitude_data:
                                break
                            entry = DataEntry(
                                time=time,
                                sleep=convert_sleep(ps_row[0]),
                                acc=magnitude_data,
                                acc_z=z_angle_data,
                                temp=temp
                            )
                            data_list.append(entry.to_dic())
                    if ps_row == ['Sleep Stage', 'Position', 'Time [hh:mm:ss]', 'Event', 'Duration[s]']:
                        header_end = True
                    if next(iter(ps_row or []), None) == 'Recording Date:':
                        date = ps_row[1].split()[0]
        df = pd.DataFrame.from_dict(data_list, orient='columns')
        df = df.set_index('Date')
        df.to_excel(csv_object.x_data_path)
        end_time = datetime.now()
        logger.info(f'Data {csv_object.filename} preprocessed in {end_time - start_time}')
    return True


def _assign_frequency_modulo(csv_row):
    frequency_modulo = 0
    if len(csv_row) > 0 and csv_row[0].startswith('Measurement Frequency'):
        frequency = csv_row[1]
        if '25.0 Hz' in frequency:
            frequency_modulo = 1
        elif '50.0 Hz' in frequency:
            frequency_modulo = 2
        elif '85.7 Hz' in frequency:
            frequency_modulo = 3
    return frequency_modulo


def _process_csv_data_core(csv_reader, end, frequency_modulo, modulo_reminder):
    magnitude_data = []
    z_angle_data = []
    temp = []
    for csv_row in csv_reader:
        if len(csv_row) > 0 and csv_row[0].startswith('20'):
            if modulo_reminder == 0:
                csv_date = convert_csv_time(csv_row)
                acc_magnitude = math.sqrt(
                    float(csv_row[1]) ** 2 + float(csv_row[2]) ** 2 + float(csv_row[3]) ** 2)
                acc_z_angle = math.degrees(
                    math.atan(float(csv_row[3]) / (
                            (float(csv_row[1]) ** 2 + float(csv_row[2]) ** 2) ** 0.5)))
                magnitude_data.append(acc_magnitude)
                z_angle_data.append(acc_z_angle)
                temp.append(float(csv_row[6]))
            modulo_reminder = (modulo_reminder + 1) % frequency_modulo
            if csv_date >= end:
                break
    return magnitude_data, z_angle_data, temp


def _find_start(csv_object, ps_object):
    if isinstance(ps_object, PsData) and isinstance(csv_object, CsvData):
        ps_start = get_ps_start(ps_object) - timedelta(seconds=15)
        csv_start = get_csv_start(csv_object)
        start = ps_start if ps_start > csv_start else csv_start
        return start
    else:
        return None


def _preprocess_prediction_data(csv_object, predict=True):
    logger.info(f'Data will be preprocessed for {csv_object.filename}')
    if not fix_csv_data(csv_object):
        return False
    data_list = []
    start_time = datetime.now()
    nights = _get_diary_nights(csv_object)
    if nights:
        total_end = nights[-1][1]
    with open(csv_object.data.path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

        frequency_modulo = 0
        modulo_reminder = 0
        for csv_row in csv_reader:
            if frequency_modulo == 0:
                frequency_modulo = _assign_frequency_modulo(csv_row)
            # now I care just about data, which starts with timestamp starting with 20 --> begin of year
            if len(csv_row) > 0 and csv_row[0].startswith('20') and frequency_modulo != 0:
                csv_date = convert_csv_time(csv_row)
                if not _is_in_any_range(csv_date, nights):
                    if csv_date > total_end:
                        break
                    else:
                        continue
                time = csv_date + timedelta(seconds=15)
                end = csv_date + timedelta(seconds=30)
                magnitude_data, z_angle_data, temp = _process_csv_data_core(csv_reader,
                                                                            end,
                                                                            frequency_modulo,
                                                                            modulo_reminder)
                if not magnitude_data:
                    break
                data_list.append(
                    DataEntry(
                        time=time,
                        acc=magnitude_data,
                        acc_z=z_angle_data,
                        temp=temp
                    ).to_dic()
                )
        if not data_list:
            logger.warning(f'No data to preprocess {csv_object.filename}')
            return False
        df = pd.DataFrame.from_dict(data_list, orient='columns')
        df = df.set_index('Date')
        df.to_excel(csv_object.x_data_path)
        end_time = datetime.now()
        logger.info(f'Data {csv_object.filename} preprocessed in {end_time - start_time}')
        if predict:
            predict_core(csv_object, df)
        return True


def _is_in_any_range(csv_date, nights):
    for night in nights:
        if night[0] < csv_date < night[1]:
            return True
    return False


def _get_diary_nights(csv_object):
    diary = SleepDiaryDay.objects.filter(subject=csv_object.subject)
    nights = []
    if diary.exists():
        for day in diary:
            nights.append((day.t1 - timedelta(minutes=30), day.t4 + timedelta(minutes=30)))
    return nights
