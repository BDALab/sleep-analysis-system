from statistics import variance, stdev, mean, median

from numpy import quantile, percentile
from scipy.stats import iqr, trim_mean, median_absolute_deviation, kurtosis, skew, mode

from dashboard.logic.features_extraction.utils import safe_div
from dashboard.logic.machine_learning.settings import scale_name


def _get_features_for_vector(vec, prefix):
    # Prepare frequently used values
    _len = len(vec)
    _max = max(vec)
    _pos_max = vec.index(_max)
    _min = min(vec)
    _pos_min = vec.index(_min)
    _range = abs(_max - _min)
    _var = variance(vec)
    _std = stdev(vec)
    _mean = mean(vec)
    _mode = mode(vec, axis=None)[0][0]
    _median = median(vec)
    features = {
        f'{prefix} | MAX': _max,
        f'{prefix} | MIN': _min,
        # f'{prefix} | POSITION OF MAX': _pos_max, # TOO BIG
        # f'{prefix} | POSITION OF MIN': _pos_min, # TOO BIG
        f'{prefix} | RELATIVE POSITION OF MAX': safe_div(_pos_max, _len),
        f'{prefix} | RELATIVE POSITION OF MIN': safe_div(_pos_min, _len),
        f'{prefix} | RANGE': _range,
        f'{prefix} | RELATIVE RANGE': safe_div(_range, _max),
        f'{prefix} | RELATIVE VARIATION RANGE': safe_div(_range, _mean),
        f'{prefix} | INTERQUARTILE RANGE': iqr(vec),
        f'{prefix} | RELATIVE INTERQUARTILE RANGE': safe_div(iqr(vec), _max),
        f'{prefix} | INTERDECILE RANGE': quantile(vec, 0.9) - quantile(vec, 0.1),
        f'{prefix} | RELATIVE INTERDECILE RANGE': safe_div(quantile(vec, 0.9) - quantile(vec, 0.1), _max),
        f'{prefix} | INTERPERCENTILE RANGE': quantile(vec, 0.99) - quantile(vec, 0.01),
        f'{prefix} | RELATIVE INTERPERCENTILE RANGE': safe_div(quantile(vec, 0.99) - quantile(vec, 0.01), _max),
        f'{prefix} | STUDENTIZED RANGE': safe_div(_range, _var),
        f'{prefix} | MEAN': _mean,
        # f'{prefix} | GEOMETRIC MEAN': gmean(vec), # always NaN
        # f'{prefix} | HARMONIC MEAN': harmonic_mean(vec), # harmonic mean does not support negative values
        f'{prefix} | MEAN EXCLUDING OUTLIERS (10)': trim_mean(vec, 0.1),
        f'{prefix} | MEAN EXCLUDING OUTLIERS (20)': trim_mean(vec, 0.2),
        f'{prefix} | MEAN EXCLUDING OUTLIERS (30)': trim_mean(vec, 0.3),
        f'{prefix} | MEAN EXCLUDING OUTLIERS (40)': trim_mean(vec, 0.4),
        # f'{prefix} | MEAN EXCLUDING OUTLIERS (50)': trim_mean(vec, 0.5),
        f'{prefix} | MEDIAN': _median,
        f'{prefix} | MODE': _mode,
        f'{prefix} | VARIANCE': _var,
        f'{prefix} | STANDARD DEVIATION': _std,
        f'{prefix} | MEDIAN ABSOLUTE DEVIATION': median_absolute_deviation(vec),
        # f'{prefix} | GEOMETRIC STANDARD DEVIATION': gstd(vec), # The geometric standard deviation is defined for
        # strictly positive values only.
        f'{prefix} | RELATIVE STANDARD DEVIATION': safe_div(_std, _mean),
        f'{prefix} | INDEX OF DISPERSION': safe_div(_var, _mean),
        # f'{prefix} | 3rd MOMENT': moment(_var, 3), always 0
        # f'{prefix} | 4th MOMENT': moment(_var, 4), always 0
        # f'{prefix} | 5th MOMENT': moment(_var, 5), always 0
        # f'{prefix} | 6th MOMENT': moment(_var, 6), always 0
        f'{prefix} | KURTOSIS': kurtosis(vec),
        f'{prefix} | SKEWNESS': skew(vec),
        f'{prefix} | PEARSONS 1st SKEWNESS COEFFICIENT': safe_div((3 * (_mean - _mode)), _std),
        f'{prefix} | PEARSONS 2nd SKEWNESS COEFFICIENT': safe_div(3 * (_mean - _median), _std),
        f'{prefix} | 1st PERCENTILE': percentile(vec, 1),
        f'{prefix} | 5th PERCENTILE': percentile(vec, 5),
        f'{prefix} | 10th PERCENTILE': percentile(vec, 10),
        f'{prefix} | 20th PERCENTILE': percentile(vec, 20),
        f'{prefix} | 1st QUARTILE': percentile(vec, 25),
        f'{prefix} | 30th PERCENTILE': percentile(vec, 30),
        f'{prefix} | 40th PERCENTILE': percentile(vec, 40),
        f'{prefix} | 60th PERCENTILE': percentile(vec, 60),
        f'{prefix} | 70th PERCENTILE': percentile(vec, 70),
        f'{prefix} | 3th QUARTILE': percentile(vec, 75),
        f'{prefix} | 80th PERCENTILE': percentile(vec, 80),
        f'{prefix} | 90th PERCENTILE': percentile(vec, 90),
        f'{prefix} | 95th PERCENTILE': percentile(vec, 95),
        f'{prefix} | 99th PERCENTILE': percentile(vec, 99),
        # f'{prefix} | SHANNON ENTROPY': entropy(vec), # MAKE NO SENSE HERE
        # f'{prefix} | MODULATION': _range / (_max + _min) # MAKE NO SENSE HERE
    }
    return features


class DataEntry(object):
    def __init__(self, time, acc, acc_z, sleep=-1):
        self.time = time
        self.acc = acc
        self.acc_z = acc_z
        self.sleep = sleep

    def __str__(self):
        return f'DataEntry[Date: {self.time} | ' \
               f'Sleep: {self.sleep} | ' \
               f'Accelerometer magnitude entries: {len(self.acc)} | ' \
               f'Accelerometer z-angle entries: {len(self.acc_z)}]'

    def get_features(self):
        features = _get_features_for_vector(self.acc, 'MAGNITUDE')
        features.update(_get_features_for_vector(self.acc_z, 'Z_ANGLE'))
        return features

    def to_dic(self):
        data = {
            'Date': self.time,
            scale_name: self.sleep
        }
        data.update(self.get_features())
        return data
