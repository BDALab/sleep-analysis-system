import csv
import logging
import math
from datetime import timedelta, datetime

from dashboard.logic.cache import save_obj
from dashboard.logic.features_extraction.data_entry import DataEntry
from dashboard.models import PsData, CsvData
from .preprocess_csv_data import fix_csv_data, get_csv_start, convert_csv_time
from .preprocess_ps_data import get_ps_start, convert_ps_timestamp, convert_sleep
from ..machine_learning.settings import algorithm, Algorithm
from ..multithread import parallel_for
from ..zangle.predict import preprocess_prediction_data_z, preprocess_training_data_z

logger = logging.getLogger(__name__)


def preprocess_all_data():
    total_start = datetime.now()

    data = CsvData.objects.all()
    results = parallel_for(data, preprocess_data)

    logger.info(f'{len(data)} training csv data objects preprocessed in {datetime.now() - total_start}')
    for r in results:
        if not r.result():
            return False
    return True


def preprocess_data(csv_object):
    if isinstance(csv_object, CsvData):
        if csv_object.data_cached and algorithm == Algorithm.XGBoost:
            return True
        elif csv_object.training_data:
            return _preprocess_training_data(csv_object) \
                if algorithm == Algorithm.XGBoost \
                else preprocess_training_data_z(csv_object)
        return _preprocess_prediction_data(csv_object) \
            if algorithm == Algorithm.XGBoost \
            else preprocess_prediction_data_z(csv_object)

    else:
        logger.warning(f'Wrong data type {type(csv_object)} was passed into preprocessing method.')
        return False


def _preprocess_training_data(csv_object):
    ps_object = PsData.objects.filter(csv_data=csv_object).first()
    if not isinstance(ps_object, PsData) or not fix_csv_data(csv_object):
        return False
    else:
        start_time = datetime.now()
        start = _find_start(csv_object, ps_object)
        data_list = []
        with open(ps_object.data.path, 'r') as ps_file:
            ps_reader = csv.reader(ps_file, delimiter='\t', quotechar='|')

            frequency_modulo = 3
            modulo_reminder = 0

            with open(csv_object.data.path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

                header_end = False
                after_midnight = False

                for csv_row in csv_reader:
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
                            accelerometer_data = []
                            temperature_data = []
                            for csv_row in csv_reader:
                                if len(csv_row) > 0 and csv_row[0].startswith('Measurement Frequency'):
                                    frequency = csv_row[1]
                                    if '25.0 Hz' in frequency:
                                        frequency_modulo = 1
                                    elif '85.7 Hz' in frequency:
                                        frequency_modulo = 3
                                elif len(csv_row) > 0 and csv_row[0].startswith('20'):
                                    if modulo_reminder == 0:
                                        csv_date = convert_csv_time(csv_row)
                                        acc_magnitude = math.sqrt(
                                            float(csv_row[1]) ** 2 + float(csv_row[2]) ** 2 + float(csv_row[3]) ** 2)
                                        accelerometer_data.append(acc_magnitude)
                                        temperature_data.append(float(csv_row[6]))  # Temperature
                                    modulo_reminder = (modulo_reminder + 1) % frequency_modulo
                                    if csv_date >= end:
                                        break
                            if not accelerometer_data:
                                break
                            data_list.append(
                                DataEntry(
                                    time=time,
                                    sleep=convert_sleep(ps_row[0]),
                                    accelerometer=accelerometer_data,
                                    temperature=temperature_data
                                )
                            )

                    if ps_row == ['Sleep Stage', 'Position', 'Time [hh:mm:ss]', 'Event', 'Duration[s]']:
                        header_end = True
                    if next(iter(ps_row or []), None) == 'Recording Date:':
                        date = ps_row[1].split()[0]
        save_obj(data_list, csv_object.cached_data_path)
        csv_object.data_cached = True
        csv_object.save()
        end_time = datetime.now()
        logger.info(f'Data {csv_object.filename} preprocessed in {end_time - start_time}')
    return True


def _find_start(csv_object, ps_object):
    if isinstance(ps_object, PsData) and isinstance(csv_object, CsvData):
        ps_start = get_ps_start(ps_object) - timedelta(seconds=15)
        csv_start = get_csv_start(csv_object)
        start = ps_start if ps_start > csv_start else csv_start
        return start
    else:
        return None


def _preprocess_prediction_data(csv_object):
    if not fix_csv_data(csv_object):
        return False
    data_list = []
    start_time = datetime.now()
    with open(csv_object.data.path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')

        frequency_modulo = 3
        modulo_reminder = 0
        accelerometer_data = []
        temperature_data = []
        timestamp = 0
        for csv_row in csv_reader:
            # now I care just about data, which starts with timestamp starting with 20 --> begin of year
            if len(csv_row) > 0 and csv_row[0].startswith('20'):
                if modulo_reminder == 0:
                    csv_date = convert_csv_time(csv_row)
                    if timestamp == 0:  # first time
                        timestamp = csv_date + timedelta(seconds=15)
                    elif csv_date >= timestamp + timedelta(seconds=15):  # end of 30s window
                        data_list.append(
                            DataEntry(timestamp, accelerometer_data, temperature_data, None))
                        accelerometer_data = []
                        temperature_data = []
                        timestamp = csv_date + timedelta(seconds=15)
                    accelerometer_data.append(float(csv_row[1]))  # X_axis
                    accelerometer_data.append(float(csv_row[2]))  # Y_axis
                    accelerometer_data.append(float(csv_row[3]))  # Z_axis
                    temperature_data.append(float(csv_row[6]))  # Temperature
                modulo_reminder = (modulo_reminder + 1) % frequency_modulo
        save_obj(data_list, csv_object.cached_data_path)
        csv_object.data_cached = True
        csv_object.save()
        end_time = datetime.now()
        logger.info(f'Data {csv_object.filename} preprocessed in {end_time - start_time}')
        return True
