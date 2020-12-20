# Generated by Django 3.1.1 on 2020-09-28 21:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0026_auto_20200928_0015'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sleepdiaryday',
            name='day_sleep_time',
            field=models.PositiveSmallIntegerField(blank=True, null=True, verbose_name='duration of sleep during day (min)'),
        ),
    ]
