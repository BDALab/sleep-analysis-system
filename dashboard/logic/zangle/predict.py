import logging
from datetime import datetime

import numpy as np
import pandas as pd

from dashboard.logic.machine_learning import settings
from dashboard.logic.preprocessing.preprocess_csv_data import fix_csv_data
from dashboard.logic.zangle.helper_functions import f_comp_angle, f_cut_start, f_cut_end, f_inactiv
from dashboard.models import PsData

logger = logging.getLogger(__name__)


def preprocess_training_data_z(csv_object):
    ps_object = PsData.objects.filter(csv_data=csv_object).first()
    if not isinstance(ps_object, PsData) or not fix_csv_data(csv_object):
        return False
    else:
        start_time = datetime.now()

        df = pd.read_csv(csv_object.data.path,
                         names=['time stamp', 'x axis [g]', 'y axis [g]', 'z axis [g]',
                                'light level [lux]', 'button [1/0]', 'temperature [°C]'],
                         skiprows=100,
                         # might be slightly faster:
                         infer_datetime_format=True, memory_map=True)
        df['time stamp'] = pd.to_datetime(df['time stamp'], format='%Y-%m-%d %H:%M:%S:%f')
        # drop not used columns from ACG
        df.drop(columns=['light level [lux]', 'button [1/0]', 'temperature [°C]'], inplace=True, axis=1)

        # 1st PSG read -> get recording date
        df_PSG = pd.read_csv(ps_object.data.path,
                             infer_datetime_format=True, memory_map=True)
        PSG_date = df_PSG["RemLogic Event Export"][2].split("\t")[1]
        # 2nd PSG read -> parse to datetime
        df_PSG = pd.read_csv(ps_object.data.path, sep='\t', skiprows=17,
                             infer_datetime_format=True, memory_map=True)
        df_PSG['Time [hh:mm:ss]'] = PSG_date + " " + df_PSG['Time [hh:mm:ss]']
        df_PSG['Time [hh:mm:ss]'] = pd.to_datetime(df_PSG['Time [hh:mm:ss]'], format='%d/%m/%Y %H:%M:%S')
        PSG_date = pd.to_datetime(PSG_date, format='%d/%m/%Y')
        # add a day if PSG crosses 00:00
        midnight = df_PSG[(df_PSG['Time [hh:mm:ss]'].dt.hour == 0) &
                          (df_PSG['Time [hh:mm:ss]'].dt.minute == 0)]
        mid_idx = midnight.index[0]
        if ~midnight.empty & mid_idx != 0:
            for index, value in df_PSG['Time [hh:mm:ss]'].items():
                if index >= mid_idx:
                    df_PSG['Time [hh:mm:ss]'][index] += pd.to_timedelta(1, unit='d')

        # Drop values so that the ACG and PSG recording is starting and ending at the same time
        f_cut_start(df, df_PSG, csv_object, ps_object)
        f_cut_end(df, df_PSG, csv_object, ps_object)

        # Set origin from PSG to start resampling
        orig = (df_PSG['Time [hh:mm:ss]'].dt.hour[0] * 3600 +
                (df_PSG['Time [hh:mm:ss]'].dt.minute[0] * 60) +
                (df_PSG['Time [hh:mm:ss]'].dt.second[0]))
        orig = pd.Timestamp(orig, unit='s')

        # Resample by 5 second epoch and compute median of x,y,z
        df = df.resample('5S', on='time stamp', kind='timestamp').median()  # .round(decimals=4)

        # Apply func f_comp_angle
        df['angle'] = df.apply(f_comp_angle, axis=1)  # .round(decimals=4)

        # Return absolute difference in angle per column
        df['abs_angle_change'] = df['angle'].diff().abs()  # .round(decimals=4)

        # New column with all "W" values
        df[settings.prediction_name] = "W"

        # Decide inactivity based on the angle
        f_inactiv(settings.first_threshold, settings.time_window, settings.angle, df)

        # Resample ACG to have same number of columns as PSG (30s epochs)
        df = df.resample('30S').interpolate()

        # If PSG has extra value at the end
        if len(df_PSG) > len(df):
            df_PSG.drop(df_PSG.index[len(df_PSG) - 1], inplace=True)

        # Overwrite State PSG to bi-state
        df[settings.scale_name] = np.where((df_PSG['Sleep Stage'] == 'N1') | (df_PSG['Sleep Stage'] == 'N2') |
                                           (df_PSG['Sleep Stage'] == 'N3') | (df_PSG['Sleep Stage'] == 'R'), 'S',
                                           'W')

        df.to_excel(csv_object.z_data_path)
        end_time = datetime.now()
        logger.info(f'Data {csv_object.filename} preprocessed in {end_time - start_time}')

        return True


def preprocess_prediction_data_z(csv_object):
    pass
