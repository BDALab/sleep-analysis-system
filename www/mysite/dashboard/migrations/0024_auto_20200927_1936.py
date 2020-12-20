# Generated by Django 3.1.1 on 2020-09-27 17:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0023_sleepdiaryday_wakeinterval'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='sleepdiaryday',
            name='bed_time',
        ),
        migrations.AlterField(
            model_name='sleepdiaryday',
            name='sleep_duration',
            field=models.TimeField(verbose_name='expected fall asleep time'),
        ),
    ]
