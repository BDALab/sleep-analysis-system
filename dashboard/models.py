import os
from datetime import timedelta, datetime
from os import path

import pytz
from django.core.validators import FileExtensionValidator
from django.db import models

from dashboard.logic import cache
# HOW TO CHANGE MODEL:
# Change your models (in models.py).
# Run python manage.py makemigrations to create migrations for those changes
# --> python manage.py makemigrations dashboard
# Run python manage.py migrate to apply those changes to the database.
# --> python manage.py sqlmigrate dashboard 0001
# --> python manage.py migrate
from dashboard.logic.highlevel_features.norms import sol, awk5plus, waso, se


class Subject(models.Model):
    code = models.CharField('subject code', max_length=50, unique=True)
    age = models.SmallIntegerField('age (years)')
    SEX = [
        ('M', 'Male'),
        ('F', 'Female')
    ]
    sex = models.CharField('sex', max_length=1, choices=SEX)
    creation_date = models.DateField('creation date', auto_now_add=True)
    sleep_disorder = models.BooleanField('sleep disorder', default=False)
    diagnosis = models.CharField('diagnosis', max_length=255, blank=True)

    def __str__(self):
        return self.code

    def is_test(self):
        return CsvData.objects.filter(subject=self).filter(training_data=True).exists()


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
    prediction_cached = models.BooleanField('graph image created', editable=False, default=0)
    data_cached = models.BooleanField('data cached', editable=False, default=0)
    training_data = models.BooleanField('training data', default=False)
    features_extracted = models.BooleanField('features extracted', editable=False, default=0)

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
    def cached_data_path(self):
        split = path.split(self.data.path)
        folder = f'{split[0]}/../cache'
        name = f'{split[1]}.pkl'
        if not path.exists(folder):
            os.mkdir(folder)
        return f'{folder}/{name}'

    @property
    def z_data_path(self):
        split = path.split(self.data.path)
        folder = f'{split[0]}/../z_prediction'
        name = f'{split[1]}.xlsx'
        if not path.exists(folder):
            os.mkdir(folder)
        return f'{folder}/{name}'

    @property
    def features_data_path(self):
        split = path.split(self.data.path)
        folder = f'{split[0]}/../features'
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
        if not path.exists(self.excel_prediction_path) and path.exists(self.cached_prediction_path):
            df = cache.load_obj(self.cached_prediction_path)
            df.to_excel(self.excel_prediction_path)
        elif path.exists(self.excel_prediction_path):
            return self.data.storage.url(self.excel_prediction_path)
        else:
            return ''

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
    tst = models.PositiveIntegerField('total sleep time')
    waso = models.PositiveIntegerField('wake after sleep onset')
    se = models.FloatField('sleep efficiency')
    sf = models.FloatField('sleep fragmentation')
    sol = models.PositiveIntegerField('sleep onset latency')
    awk5plus = models.PositiveSmallIntegerField('awakenings > 5 minutes')

    @property
    def name(self):
        split = path.split(self.data.data.path)
        folder = f'{split[0]}/../predictions-fin'
        name = f'{split[1]}_day_{self.diary_day.date}.xlsx'
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
        return f'Subject: {self.subject.code} | Day:{self.diary_day.date} | Data:{self.data.filename} | {self.info}'

    @property
    def info(self):
        return f'Sleep onset: {self.sleep_onset.astimezone(pytz.timezone("Europe/Prague")).time()} ' \
               f'| Sleep end: {self.sleep_end.astimezone(pytz.timezone("Europe/Prague")).time()} ' \
               f'| Total sleep time: {self.convert(self.tst)} ' \
               f'| Wake after sleep onset: {self.convert(self.waso)} ' \
               f'| Sleep efficiency: {self.se:.1f}% ' \
               f'| Sleep fragmentation: {self.sf:.2f} ' \
               f'| Sleep onset latency: {self.convert(self.sol)}' \
               f'| Awakenings > 5 minutes: {self.awk5plus}' \
               f'| Sleep onset latency norm: {self.sol_norm}' \
               f'| Awakenings > 5 minutes norm: {self.awk5plus_norm}' \
               f'| Wake after sleep onset norm: {self.waso_norm}' \
               f'| Sleep efficiency norm: {self.se_norm}'
