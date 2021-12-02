import logging
from datetime import timedelta, date, datetime, time

import pandas as pd
from pandas._libs.tslibs.nattype import NaT

from dashboard.models import Subject, SleepDiaryDay, WakeInterval
from mysite.settings import METADATA_PATH

logger = logging.getLogger(__name__)

REST_QUALITY = {
    'not at all rested': 1,
    'slightly rested': 2,
    'somehow rested': 3,
    'well-rested': 4,
    'very well-rested': 5
}

SLEEP_QUALITY = {
    'very poor': 1,
    'poor': 2,
    'fair': 3,
    'good': 4,
    'very good': 5
}

INVALID_TIME = '0:00:13'


def parse_metadata():
    result = True
    df = pd.read_excel(METADATA_PATH, index_col=0)
    for index, row in df.iterrows():
        personal_id = row['#personalID']
        if personal_id is None:
            logger.error(f'Subject without personal ID detected! Skip...')
            result = False
            continue
        subject = Subject.objects.filter(code=personal_id).first()
        if subject is None:
            try:
                subject = Subject(
                    code=personal_id,
                    age=datetime.now().year - (row['#dateOfBirth']).year
                )
                subject.save()
            except Exception as e:
                logger.error(f'Failed to parse subject; exception {str(e)}')
                result = False
                continue
        for i in range(1, 8):
            day = f'day{i}'
            if subject is None or day is None or row[f'{day}-date'] is NaT:
                logger.error(f'Day without subject or date detected! Skip...')
                result = False
                continue
            dd = SleepDiaryDay.objects.filter(subject=subject).filter(date=row[f'{day}-date']).first()
            if dd is None:
                try:
                    logger.info(f'Create new diary day: subject {personal_id} | day {i}')
                    dd = SleepDiaryDay(
                        subject=subject,
                        date=row[f'{day}-date'],
                    )
                except Exception as e:
                    logger.error(f'Failed to parse day for subject {subject.code}; exception {str(e)}')
                    result = False
                    continue
            try:
                dd.day_sleep_count = row[f'{day}-1a']
                dd.day_sleep_time = row[f'{day}-1b']
                dd.alcohol_count = row[f'{day}-2a']
                dd.alcohol_time = _get_time_safe(day, row, '2b')
                dd.caffeine_count = row[f'{day}-3a']
                dd.caffeine_time = _get_time_safe(day, row, '3b')
                dd.sleeping_pill = row[f'{day}-4']
                dd.sleep_time = _get_time_safe(day, row, '5')
                dd.sleep_duration = (datetime.combine(date.today(), _get_time_safe(day, row, '5')) + timedelta(
                    minutes=row[f'{day}-6'])).time()
                dd.wake_count = _get_wake_count(day, row)
                dd.wake_time = _get_time_safe(day, row, '8')
                dd.get_up_time = (datetime.combine(date.today(), _get_time_safe(day, row, '8')) + timedelta(
                    minutes=row[f'{day}-9'])).time()
                dd.sleep_quality = SLEEP_QUALITY.get(row[f'{day}-10'])
                dd.rest_quality = REST_QUALITY.get(row[f'{day}-11'])
                dd.note = row[f'{day}-12']

                dd.save()

                if not _parse_wake_intervals(day, row, dd):
                    result = False
            except Exception as e:
                logger.error(f'Failed to parse metadata: subject {personal_id} | day {i}; exception {str(e)}')
                result = False
    return result


def _get_wake_count(day, row):
    val = row[f'{day}-7']
    return val if isinstance(val, int) else val[0]


def _get_time_safe(day, row, field):
    val = row[f'{day}-{field}']
    return INVALID_TIME if not isinstance(val, time) else val


def _parse_wake_intervals(day, row, diary_day):
    val = row[f'{day}-7']
    result = True
    if isinstance(val, str):
        intervals = val.split(';')
        for interval in intervals[1:]:
            try:
                values = interval.split('-')
                start = values[0].strip()
                if len(values) > 1 and values[1].strip() != "":
                    end = values[1].strip()
                else:
                    end = start
                interval_model = WakeInterval.objects.filter(sleep_diary_day=diary_day).filter(start=start).first()
                if interval_model is None:
                    interval_model = WakeInterval()
                interval_model.sleep_diary_day = diary_day
                interval_model.start = start
                interval_model.end = end

                interval_model.save()

            except Exception as e:
                logger.error(
                    f'Failed to parse metadata (wake interval): subject {diary_day.subject} | day {diary_day.date}; exception {str(e)}')
                result = False
    return result
