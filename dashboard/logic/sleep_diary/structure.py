import logging
import os.path

from dashboard.logic import cache
from dashboard.models import SleepDiaryDay, CsvData, Subject

logger = logging.getLogger(__name__)


def create_structure_all():
    structure = []
    for subject in Subject.objects.all():
        _structure_for_subject(structure, subject)
    return structure


def create_structure(subject):
    structure = []
    _structure_for_subject(structure, subject)
    return structure


def _structure_for_subject(structure, subject):
    sleep_days = SleepDiaryDay.objects.filter(subject=subject)
    if sleep_days.exists():
        for sleep_day in sleep_days:
            assert isinstance(sleep_day, SleepDiaryDay)
            data = CsvData.objects.filter(subject=subject)
            if not data.exists():  # no CSV data
                logger.warning(
                    f'Missing csv data for subject {subject} with {len(sleep_days)} sleep diary days')
            else:
                if len(data) == 1:  # single CSV data file
                    matching_data = data.first()
                else:  # data need to be found
                    s = sleep_day.t1
                    e = sleep_day.t4
                    for d in data:
                        assert isinstance(d, CsvData)
                        if not os.path.exists(d.cached_prediction_path):
                            continue
                        pred = cache.load_obj(d.cached_prediction_path)
                        interval = pred[s:e]
                        if len(interval) > 0:  # matchin data found
                            matching_data = d
                            break
                if matching_data is None:
                    continue
                structure.append((subject, matching_data, sleep_day))
                logger.debug(
                    f'{subject.code} - {data.first().filename} - {sleep_day.date} added to validation structure ')
