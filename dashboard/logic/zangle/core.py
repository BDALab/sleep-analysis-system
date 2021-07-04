import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dashboard.logic.machine_learning import settings
from dashboard.logic.preprocessing.preprocess_csv_data import fix_csv_data
from dashboard.logic.zangle.helper_functions import f_cut_start, f_cut_end, load_df_from_csv, \
    load_df_from_ps_data, is_not_cached, get_split_path, f_comp_angle, f_inactiv, is_cached
from dashboard.models import PsData, Subject, SleepDiaryDay

logger = logging.getLogger(__name__)


def training_data_z(csv_object):
    ps_object = PsData.objects.filter(csv_data=csv_object).first()
    if not isinstance(ps_object, PsData) or not fix_csv_data(csv_object):
        return False
    else:
        start_time = datetime.now()

        df = load_df_from_csv(csv_object)

        df_PSG = load_df_from_ps_data(ps_object)

        # Drop values so that the ACG and PSG recording is starting and ending at the same time
        f_cut_start(df, df_PSG, csv_object, ps_object)
        f_cut_end(df, df_PSG, csv_object, ps_object)

        # Set origin from PSG to start resampling
        orig = (df_PSG['Time [hh:mm:ss]'].dt.hour[0] * 3600 +
                (df_PSG['Time [hh:mm:ss]'].dt.minute[0] * 60) +
                (df_PSG['Time [hh:mm:ss]'].dt.second[0]))
        orig = pd.Timestamp(orig, unit='s')

        df = zangle_core(df)

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


def prediction_data_z(csv_object):
    if is_not_cached(csv_object):
        df = load_df_from_csv(csv_object)
        subject = csv_object.subject
        if isinstance(subject, Subject):
            days = SleepDiaryDay.objects.filter(subject=subject)
            for day in days:
                if isinstance(day, SleepDiaryDay):
                    start_time = datetime.now()
                    if is_cached(csv_object, day, subject):
                        continue
                    s = day.t1 - timedelta(minutes=30)
                    e = day.t4 + timedelta(minutes=30)

                    df_night = df[s < df['time stamp']]
                    df_night = df_night[df_night['time stamp'] < e]
                    if not df_night.empty:
                        df_night = zangle_core(df_night)
                        df_night.to_excel(get_split_path(csv_object, day, subject))
                        end_time = datetime.now()
                        logger.info(f'Data subject:{subject.code} day:{day.date} data:{csv_object.filename} '
                                    f'preprocessed in {end_time - start_time}')


def zangle_core(df):
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
    return df
