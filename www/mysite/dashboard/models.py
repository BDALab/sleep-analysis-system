import os
from datetime import timedelta, datetime
from os import path

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
        return CsvData.objects.filter(subject=self).fiter(training_data=True).exists()


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
        else:
            return ''
        return self.data.storage.url(self.excel_prediction_path)

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
        return f'Polysomnography data {self.filename()} for CSV data {self.csv_data.filename}'


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

    def with_date(self, time):
        if 0 <= time.hour < 12:
            return datetime.combine(self.date + timedelta(days=1), time)
        return datetime.combine(self.date, time)


class WakeInterval(models.Model):
    sleep_diary_day = models.ForeignKey(SleepDiaryDay, on_delete=models.CASCADE)
    creation_date = models.DateField('creation date', auto_now_add=True)
    start = models.TimeField('start')
    end = models.TimeField('stop')

    def duration(self):
        t1 = timedelta(hours=self.start.hour, minutes=self.start.minute)
        t2 = timedelta(hours=self.end.hour, minutes=self.end.minute)
        duration = t2 - t1
        return duration if duration.seconds > 60 else timedelta(minutes=1)


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
