# HOW TO CHANGE MODEL:
# Change your models (in models.py).
# Run python manage.py makemigrations to create migrations for those changes
# --> python manage.py makemigrations dashboard
# Run python manage.py migrate to apply those changes to the database.
# --> python manage.py sqlmigrate dashboard 0001
# --> python manage.py migrate

import os
from datetime import timedelta, datetime
from os import path

import pytz
from django.core.validators import FileExtensionValidator
from django.db import models

from dashboard.logic.features_extraction.norms import sol, awk5plus, waso, se
from dashboard.logic.features_extraction.utils import safe_div
from mysite.settings import MEDIA_ROOT


class Subject(models.Model):
    code = models.CharField('subject code', max_length=50, unique=True)
    age = models.SmallIntegerField('age (years)')
    SEX = [
        ('M', 'Male'),
        ('F', 'Female')
    ]
    sex = models.CharField('sex', max_length=1, choices=SEX)
    creation_date = models.DateField('creation date', auto_now_add=True)
    pPD = models.BooleanField('probable parkinson disease', default=False)
    pMCI = models.BooleanField('probable mild cognitive impairment', default=False)
    HC = models.BooleanField('healthy control', default=False)
    SA = models.BooleanField('sleep apnea', default=False)
    predPDorMCI = models.BooleanField('probable parkinson disease', default=False)
    DIAGNOSIS = [
        ('D', 'preDLB'),
        ('N', 'NonHC'),
        ('H', 'HC'),
        ('U', 'Unknown')
    ]
    diagnosis = models.CharField('diagnosis', max_length=1, choices=DIAGNOSIS, default='U')
    diagnosis2024 = models.CharField('diagnosis', max_length=1, choices=DIAGNOSIS, default='U')

    def __str__(self):
        return self.code

    def is_test(self):
        return CsvData.objects.filter(subject=self).filter(training_data=True).exists()

    @property
    def export_path(self):
        return path.join(MEDIA_ROOT, 'export', self.code)

    @property
    def export_name(self):
        return path.join(self.export_path, f'{self.code}_report')


class CsvData(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    BODY_LOCATIONS = [
        ('L', 'Left wrist'),
        ('R', 'Right wrist'),
        ('O', 'Other')
    ]
    body_location = models.CharField('body location', max_length=1, choices=BODY_LOCATIONS, default='L')
    data = models.FileField('data', upload_to='data/', validators=[FileExtensionValidator(["csv"])])
    description = models.CharField('description', max_length=255, blank=True)
    creation_date = models.DateField('date of upload', auto_now_add=True)
    training_data = models.BooleanField('training data', default=False)
    end_date = models.DateTimeField('end date to process', null=True, blank=True)
    dreamt_data = models.BooleanField('dreamt data', default=False)

    @property
    def filename(self):
        return path.basename(self.data.name)

    @property
    def cached_prediction_path(self):
        split = path.split(self.data.path)
        folder = f'{split[0]}/../predictions'
        name = f'{split[1]}.pkl'
        if not path.exists(folder):
            os.mkdir(folder)
        return f'{folder}/{name}'

    @property
    def x_data_path(self):
        split = path.split(self.data.path)
        folder = f'{split[0]}/../cache'
        name = f'{split[1]}.xlsx'
        if not path.exists(folder):
            os.mkdir(folder)
        return f'{folder}/{name}'

    @property
    def excel_prediction_path(self):
        split = path.split(self.data.path)
        folder = f'{split[0]}/../predictions-excel'
        name = f'{split[1]}.xlsx'
        if not path.exists(folder):
            os.mkdir(folder)
        return f'{folder}/{name}'

    @property
    def excel_prediction_url(self):
        if path.exists(self.excel_prediction_path):
            return self.data.storage.url(self.excel_prediction_path)
        else:
            return ''

    @property
    def sleeppy_dir(self):
        split = path.split(self.data.path)
        folder = f'{split[0]}/../sleeppy'
        if not path.exists(folder):
            os.mkdir(folder)
        return folder

    def __str__(self):
        return f'CSV data {self.filename} from subject {self.subject.code}'


class PsData(models.Model):
    csv_data = models.OneToOneField(CsvData, on_delete=models.CASCADE, primary_key=True)
    data = models.FileField('data', upload_to='ps/', validators=[FileExtensionValidator(["txt"])])
    creation_date = models.DateField('date of upload', auto_now_add=True)

    @property
    def filename(self):
        return path.basename(self.data.name)

    def __str__(self):
        return f'Polysomnography data {self.filename} for CSV data {self.csv_data.filename}'


class SleepDiaryDay(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    creation_date = models.DateField('date of upload', auto_now_add=True)
    date = models.DateField('date')

    day_sleep_count = models.PositiveSmallIntegerField('count of sleep during day', blank=True)
    day_sleep_time = models.PositiveSmallIntegerField('duration of sleep during day (min)', blank=True, null=True)
    alcohol_count = models.PositiveSmallIntegerField('number of alcoholic beverages', blank=True)
    alcohol_time = models.TimeField('time of the last alcoholic beverage', blank=True, null=True)
    caffeine_count = models.PositiveSmallIntegerField('number of caffeine drinks', blank=True)
    caffeine_time = models.TimeField('time of the last caffeine drink', blank=True, null=True)
    sleeping_pill = models.BooleanField('sleeping pill')

    sleep_time = models.TimeField('try to fall asleep time')
    sleep_duration = models.TimeField('expected fall asleep time')
    wake_count = models.PositiveSmallIntegerField('number of wake ups during night')
    wake_time = models.TimeField('wake up time')
    get_up_time = models.TimeField('get up time')

    SLEEP_QUALITY = [
        (1, 'Very poor'),
        (2, 'Poor'),
        (3, 'Fair'),
        (4, 'Good'),
        (5, 'Very good'),
    ]
    sleep_quality = models.PositiveSmallIntegerField('sleep quality', choices=SLEEP_QUALITY, blank=True, null=True)
    REST_QUALITY = [
        (1, 'Not at all rested'),
        (2, 'Slightly rested'),
        (3, 'Somehow rested'),
        (4, 'Well-rested'),
        (5, 'Very well-rested')
    ]
    rest_quality = models.PositiveSmallIntegerField('rest quality', choices=REST_QUALITY, blank=True, null=True)
    note = models.CharField('note', max_length=255, blank=True)

    def _with_date(self, time):
        if 0 <= time.hour < 12:
            return datetime.combine(self.date + timedelta(days=1), time)
        return datetime.combine(self.date, time)

    def with_date_base_on_previous(self, time, previous):
        maybe = self._with_date(time)
        if maybe < previous:
            return datetime.combine(maybe.date() + timedelta(days=1), time)
        else:
            return maybe

    @property
    def t1(self):
        return self._with_date(self.sleep_time)

    @property
    def t2(self):
        return self.with_date_base_on_previous(self.sleep_duration, self.t1)

    @property
    def t3(self):
        return self.with_date_base_on_previous(self.wake_time, self.t2)

    @property
    def t4(self):
        return self.with_date_base_on_previous(self.get_up_time, self.t3)

    def __str__(self):
        return f'Subject: {self.subject.code} | Date: {self.date} | {self.info}'

    @property
    def wake_intervals(self):
        return WakeInterval.objects.filter(sleep_diary_day=self)

    @property
    def wake_intervals_str(self):
        s = ''
        for w in self.wake_intervals:
            s += f'{w} | '
        return s

    @property
    def info(self):
        return f'Bed time: {self.sleep_time} ' \
               f'| Sleep onset: {self.sleep_duration} ' \
               f'| Wake count: {self.wake_count} ' \
               f'| Wake ups: {self.wake_intervals_str}' \
               f'| Sleep end: {self.wake_time} ' \
               f'| Get up time: {self.get_up_time}'

    @staticmethod
    def convert(n):
        return timedelta(seconds=n)

    @property
    def tib(self):
        return (self.t4 - self.t1).seconds

    @property
    def sol(self):
        return (self.t2 - self.t1).seconds

    @property
    def waso(self):
        wake = 0
        for i in self.wake_intervals:
            wake += i.duration.seconds
        return wake

    @property
    def wasf(self):
        return (self.t4 - self.t3).seconds

    @property
    def tst(self):
        return self.tib - (self.sol + self.waso + self.wasf)

    @property
    def wb(self):
        return len(self.wake_intervals)

    @property
    def awk5plus(self):
        count = 0
        for i in self.wake_intervals:
            if i.duration.seconds >= 300:
                count += 1
        return count

    @property
    def se(self):
        return safe_div(self.tst, self.tib) * 100

    @property
    def sf(self):
        return safe_div(self.wb, (self.tst / 3600))

    @property
    def sol_norm(self):
        return sol(self.subject.age, self.sol)

    @property
    def awk5plus_norm(self):
        return awk5plus(self.subject.age, self.awk5plus)

    @property
    def waso_norm(self):
        return waso(self.subject.age, self.waso)

    @property
    def se_norm(self):
        return se(self.subject.age, self.se)


class WakeInterval(models.Model):
    sleep_diary_day = models.ForeignKey(SleepDiaryDay, on_delete=models.CASCADE)
    creation_date = models.DateField('creation date', auto_now_add=True)
    start = models.TimeField('start')
    end = models.TimeField('stop')

    @property
    def duration(self):
        t1 = timedelta(hours=self.start.hour, minutes=self.start.minute)
        t2 = timedelta(hours=self.end.hour, minutes=self.end.minute)
        duration = t2 - t1
        return duration if duration.seconds > 60 else timedelta(minutes=1)

    @property
    def start_with_date(self):
        return self.sleep_diary_day.with_date_base_on_previous(self.start, self.sleep_diary_day.t2)

    @property
    def end_with_date(self):
        return self.sleep_diary_day.with_date_base_on_previous(self.end, self.start_with_date)

    def __str__(self):
        return f'{self.start}-{self.end} ({self.duration})'


class RBDSQ(models.Model):
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    creation_date = models.DateField('creation date', auto_now_add=True)
    q1 = models.BooleanField('1. I sometimes have very vivid dreams.')
    q2 = models.BooleanField('2. My dreams frequently have aggressive or action packed content.')
    q3 = models.BooleanField('3. The dream contents mostly match my nocturnal behaviour.')
    q4 = models.BooleanField('4. I know that my arms or legs move when I sleep.')
    q5 = models.BooleanField('5. It thereby happened that I (almost) hurt my bed partner or myself.')
    q61 = models.BooleanField(
        '6.1. I have or had the following phenomena during my dreams: speaking, shouting, swearing, laughing loudly.')
    q62 = models.BooleanField(
        '6.2. I have or had the following phenomena during my dreams: sudden limb movements, \"fights\".')
    q63 = models.BooleanField(
        '6.3. I have or had the following phenomena during my dreams: gestures, complex movements, that are useless '
        'during sleep, e.g., to wave, to salute, to frighten mosquitoes, falls off the bed.')
    q64 = models.BooleanField(
        '6.4 I have or had the following phenomena during my dreams: things that fell down around the bed, '
        'e.g., bedside lamp, book, glasses.')
    q7 = models.BooleanField('7. It happens that my movements awake me.')
    q8 = models.BooleanField('8. After awakening I mostly remember the content of my dreams well.')
    q9 = models.BooleanField('9. My sleep is frequently disturbed.')
    q10 = models.BooleanField(
        '10. I have/had a disease of the nervous system (e.g., stroke, head trauma, parkinsonism, RLS, narcolepsy, '
        'depression, epilepsy, inflammatory disease of the brain),')
    q10comment = models.CharField('which?', max_length=255, blank=True)

    def score(self):
        return [self.q1,
                self.q2,
                self.q3,
                self.q4,
                self.q5,
                self.q61,
                self.q62,
                self.q63,
                self.q64,
                self.q7,
                self.q8,
                self.q9,
                self.q10,
                ].count(True)


class SleepNight(models.Model):
    diary_day = models.ForeignKey(SleepDiaryDay, on_delete=models.CASCADE)
    data = models.ForeignKey(CsvData, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    sleep_onset = models.DateTimeField('sleep onset')
    sleep_end = models.DateTimeField('sleep end')
    tib = models.PositiveIntegerField('time in bed')
    sol = models.PositiveIntegerField('sleep onset latency')
    waso = models.PositiveIntegerField('wake after sleep onset')
    wasf = models.PositiveIntegerField('wake after sleep offset')
    wb = models.PositiveIntegerField('wake bouts')
    awk5plus = models.PositiveSmallIntegerField('awakenings > 5 minutes')
    broken = models.BooleanField('broken according to sleepy', default=False)

    @property
    def date(self):
        return self.diary_day.date if self.diary_day else self.sleep_onset.date()

    @property
    def tst(self):
        return self.tib - (self.sol + self.waso + self.wasf)

    @property
    def se(self):
        return safe_div(self.tst, self.tib) * 100

    @property
    def sf(self):
        return safe_div(self.wb, (self.tst / 3600))

    @property
    def name(self):
        split = path.split(self.data.data.path)
        folder = f'{split[0]}/../predictions-fin'
        name = f'{split[1]}_day_{self.date}.xlsx'
        if not path.exists(folder):
            os.mkdir(folder)
        return f'{folder}/{name}'

    @staticmethod
    def convert(n):
        return timedelta(seconds=n)

    @property
    def name_url(self):
        return self.data.data.storage.url(self.name)

    @property
    def sol_norm(self):
        return sol(self.subject.age, self.sol)

    @property
    def awk5plus_norm(self):
        return awk5plus(self.subject.age, self.awk5plus)

    @property
    def waso_norm(self):
        return waso(self.subject.age, self.waso)

    @property
    def se_norm(self):
        return se(self.subject.age, self.se)

    def __str__(self):
        return f'Subject: {self.subject.code} | Day:{self.date} | Data:{self.data.filename} | {self.info}'

    @property
    def info(self):
        return f'Sleep onset: {self.sleep_onset.astimezone(pytz.timezone("Europe/Prague")).time()} ' \
               f'| Sleep offset: {self.sleep_end.astimezone(pytz.timezone("Europe/Prague")).time()} ' \
               f'| Time in bed: {self.convert(self.tib)} ' \
               f'| Total sleep time: {self.convert(self.tst)} ' \
               f'| Sleep onset latency: {self.convert(self.sol)}' \
               f'| Wake after sleep onset: {self.convert(self.waso)} ' \
               f'| Wake after sleep offset: {self.convert(self.wasf)} ' \
               f'| Wake bouts: {self.wb} ' \
               f'| Sleep efficiency: {self.se:.1f}% ' \
               f'| Sleep fragmentation: {self.sf:.2f} ' \
               f'| Awakenings > 5 minutes: {self.awk5plus}' \
               f'| Sleep onset latency norm: {self.sol_norm}' \
               f'| Awakenings > 5 minutes norm: {self.awk5plus_norm}' \
               f'| Wake after sleep onset norm: {self.waso_norm}' \
               f'| Sleep efficiency norm: {self.se_norm}'


class SleeppyData(models.Model):
    sleep_night = models.ForeignKey(SleepNight, on_delete=models.DO_NOTHING, blank=True, null=True)
    data = models.ForeignKey(CsvData, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    sleep_onset = models.DateTimeField('sleep onset')  # rest-periods
    sleep_end = models.DateTimeField('sleep end')  # rest-periods
    # tib = models.PositiveIntegerField('time in bed') # cannot be obtained, we do not have wasf
    sol = models.PositiveIntegerField('sleep onset latency')  # sleep_onset_latency in minutes
    waso = models.PositiveIntegerField('wake after sleep onset')  # waso
    # wasf = models.PositiveIntegerField('wake after sleep offset') unfortunately cannot be obtained
    wb = models.PositiveIntegerField('wake bouts')  # number_wake_bouts
    awk5plus = models.PositiveSmallIntegerField('awakenings > 5 minutes')  # number_wake_bouts_5min

    # Modified properties
    tst = models.PositiveIntegerField('total sleep time')  # total_sleep_time
    se = models.PositiveIntegerField('sleep efficiency')  # percent_time_asleep
    broken = models.BooleanField('broken according to sleep nights', default=False)

    @property
    def date(self):
        return self.sleep_night.date if self.sleep_night else self.sleep_onset.date()

    @property
    def sf(self):
        return safe_div(self.wb, (self.tst / 60))

    @property
    def name(self):
        split = path.split(self.data.data.path)
        folder = f'{split[0]}/../predictions-fin'
        name = f'{split[1]}_day_{self.date}.xlsx'
        if not path.exists(folder):
            os.mkdir(folder)
        return f'{folder}/{name}'

    @staticmethod
    def convert(n):
        return timedelta(seconds=n)

    @property
    def name_url(self):
        return self.data.data.storage.url(self.name)

    @property
    def sol_norm(self):
        return sol(self.subject.age, self.sol)

    @property
    def awk5plus_norm(self):
        return awk5plus(self.subject.age, self.awk5plus)

    @property
    def waso_norm(self):
        return waso(self.subject.age, self.waso)

    @property
    def se_norm(self):
        return se(self.subject.age, self.se)

    def __str__(self):
        return f'Subject: {self.subject.code} | Day:{self.date} | Data:{self.data.filename} | {self.info}'

    @property
    def info(self):
        return f'Sleep onset: {self.sleep_onset.astimezone(pytz.timezone("Europe/Prague")).time()} ' \
               f'| Sleep offset: {self.sleep_end.astimezone(pytz.timezone("Europe/Prague")).time()} ' \
               f'| Total sleep time: {self.convert(self.tst)} ' \
               f'| Sleep onset latency: {self.convert(self.sol)}' \
               f'| Wake after sleep onset: {self.convert(self.waso)} ' \
               f'| Wake bouts: {self.wb} ' \
               f'| Sleep efficiency: {self.se:.1f}% ' \
               f'| Sleep fragmentation: {self.sf:.2f} ' \
               f'| Awakenings > 5 minutes: {self.awk5plus}' \
               f'| Sleep onset latency norm: {self.sol_norm}' \
               f'| Awakenings > 5 minutes norm: {self.awk5plus_norm}' \
               f'| Wake after sleep onset norm: {self.waso_norm}' \
               f'| Sleep efficiency norm: {self.se_norm}'


class SignalFeatures(models.Model):
    max = models.FloatField('max')
    min = models.FloatField('min')
    relative_position_of_max = models.FloatField('relative position of max')
    relative_position_of_min = models.FloatField('relative position of min')
    range = models.FloatField('range')
    relative_range = models.FloatField('relative range')
    relative_variation_range = models.FloatField('relative variation range')
    interquartile_range = models.FloatField('interquartile range')
    relative_interquartile_range = models.FloatField('relative interquartile range')
    interdencile_range = models.FloatField('interdencile range')
    relative_interdencile_range = models.FloatField('relative interdencile range')
    interpercentile_range = models.FloatField('interpercentile range')
    relative_interpercentile_range = models.FloatField('relative interpercentile range')
    studentized_range = models.FloatField('studentized range')
    mean = models.FloatField('mean')
    harmonic_mean = models.FloatField('harmonic mean')
    mean_excluding_outliers_10 = models.FloatField('mean excluding outliers (10)')
    mean_excluding_outliers_20 = models.FloatField('mean excluding outliers (20)')
    mean_excluding_outliers_30 = models.FloatField('mean excluding outliers (30)')
    mean_excluding_outliers_40 = models.FloatField('mean excluding outliers (40)')
    median = models.FloatField('median')
    mode = models.FloatField('mode')
    variance = models.FloatField('variance')
    standard_deviation = models.FloatField('standard deviation')
    median_absolute_deviation = models.FloatField('median absolute deviation')
    relative_standard_deviation = models.FloatField('relative standard deviation')
    index_of_dispersion = models.FloatField('index of dispersion')
    kurtosis = models.FloatField('kurtosis')
    skewness = models.FloatField('skewness')
    pearson_1st_skewness_coefficient = models.FloatField('pearson 1st skewness coefficient')
    pearson_2st_skewness_coefficient = models.FloatField('pearson 2st skewness coefficient')
    percentile_1 = models.FloatField('1st percentile')
    percentile_5 = models.FloatField('5th percentile')
    percentile_10 = models.FloatField('10th percentile')
    percentile_20 = models.FloatField('20th percentile')
    percentile_80 = models.FloatField('80th percentile')
    percentile_90 = models.FloatField('90th percentile')
    percentile_95 = models.FloatField('95th percentile')
    percentile_99 = models.FloatField('99th percentile')
    shannon_entropy = models.FloatField('shannon entropy')
    modulation = models.FloatField('modulation')
    tkeo_max = models.FloatField('teager kaiser energy operator max')
    tkeo_min = models.FloatField('teager kaiser energy operator min')


class SleepNightActivityIndexFeatures(SignalFeatures):
    day_index = models.PositiveSmallIntegerField('sleeppy index')
    sleep_night = models.ForeignKey(SleepNight, on_delete=models.CASCADE)


class SleeppyActivityIndexFeatures(SignalFeatures):
    day_index = models.PositiveSmallIntegerField('sleeppy index')
    sleeppy_data = models.ForeignKey(SleeppyData, on_delete=models.CASCADE)
