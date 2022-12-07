from datetime import datetime

import pandas as pd

from dashboard.models import Subject, CsvData, SleepDiaryDay, WakeInterval


def export_metadata_to_xlsx():
    df = pd.DataFrame(columns=[
        'filename',
        '#id',
        '#personalID',
        '#firstName',
        '#surname',
        '#dateOfBirth',

        'day1-date', 'day1-1a', 'day1-1b', 'day1-2a', 'day1-2b', 'day1-3a', 'day1-3b', 'day1-4', 'day1-5', 'day1-6',
        'day1-7', 'day1-8', 'day1-9', 'day1-10', 'day1-11', 'day1-12',

        'day2-date', 'day2-1a', 'day2-1b', 'day2-2a', 'day2-2b', 'day2-3a', 'day2-3b', 'day2-4', 'day2-5', 'day2-6',
        'day2-7', 'day2-8', 'day2-9', 'day2-10', 'day2-11', 'day2-12',

        'day3-date', 'day3-1a', 'day3-1b', 'day3-2a', 'day3-2b', 'day3-3a', 'day3-3b', 'day3-4', 'day3-5', 'day3-6',
        'day3-7', 'day3-8', 'day3-9', 'day3-10', 'day3-11', 'day3-12',

        'day4-date', 'day4-1a', 'day4-1b', 'day4-2a', 'day4-2b', 'day4-3a', 'day4-3b', 'day4-4', 'day4-5', 'day4-6',
        'day4-7', 'day4-8', 'day4-9', 'day4-10', 'day4-11', 'day4-12',

        'day5-date', 'day5-1a', 'day5-1b', 'day5-2a', 'day5-2b', 'day5-3a', 'day5-3b', 'day5-4', 'day5-5',
        'day5-6', 'day5-7', 'day5-8', 'day5-9', 'day5-10', 'day5-11', 'day5-12',

        'day6-date', 'day6-1a', 'day6-1b', 'day6-2a', 'day6-2b', 'day6-3a', 'day6-3b', 'day6-4', 'day6-5', 'day6-6',
        'day6-7', 'day6-8', 'day6-9', 'day6-10', 'day6-11', 'day6-12',

        'day7-date', 'day7-1a', 'day7-1b', 'day7-2a', 'day7-2b', 'day7-3a', 'day7-3b', 'day7-4', 'day7-5', 'day7-6',
        'day7-7', 'day7-8', 'day7-9', 'day7-10', 'day7-11', 'day7-12'
    ])
    # column names mentioned explicitly, because we want this specific order
    subjects = Subject.objects.all()
    for subject in subjects:
        file = CsvData.objects.filter(subject=subject).first()
        filename = file.filename if file is not None else 'Missing data file!'
        data = {
            'filename': filename,
            '#id': filename,  # fix
            '#personalID': subject.code,
            '#firstName': subject.sex,  # fix
            '#surname': ' ',  # fix
            '#dateOfBirth': subject.age,  # fix
        }

        for i, day in enumerate(SleepDiaryDay.objects.filter(subject=subject).all(), start=1):
            wakes_string = f'{day.wake_count}'
            for wake in WakeInterval.objects.filter(sleep_diary_day=day).all():
                wakes_string += f'; {wake.start}-{wake.end}'

            data[f'day{i}-date'] = day.date
            data[f'day{i}-1a'] = day.day_sleep_count
            data[f'day{i}-1b'] = str(_get_time_safe(day.day_sleep_time))[:5]
            data[f'day{i}-2a'] = day.alcohol_count
            data[f'day{i}-2b'] = str(_get_time_safe(day.alcohol_time))[:5]
            data[f'day{i}-3a'] = day.caffeine_count
            data[f'day{i}-3b'] = str(_get_time_safe(day.caffeine_time))[:5]
            data[f'day{i}-4'] = int(day.sleeping_pill)
            data[f'day{i}-5'] = str(_get_time_safe(day.sleep_time))[:5]
            data[f'day{i}-6'] = _get_timedelta_safe(day.t1, day.t2)
            data[f'day{i}-7'] = wakes_string
            data[f'day{i}-8'] = str(_get_time_safe(day.wake_time))[:5]
            data[f'day{i}-9'] = _get_timedelta_safe(day.t3, day.t4)
            data[f'day{i}-10'] = _get_name_lower_safe(day.get_sleep_quality_display())
            data[f'day{i}-11'] = _get_name_lower_safe(day.get_rest_quality_display())
            data[f'day{i}-12'] = day.note

        df = df.append(data, ignore_index=True)

    df = df.replace('nan', '')
    df.to_excel(f'export-{datetime.now().date()}.xlsx', index=False)
    return True


def _get_timedelta_safe(t1, t2):
    return (t2 - t1).seconds / 60


def _get_name_lower_safe(name):
    return name.lower() if name is not None else ''


def _get_time_safe(time):
    if isinstance(time, int):
        return time
    elif not time:
        return 0
    return time if time.second != 13 else 0
