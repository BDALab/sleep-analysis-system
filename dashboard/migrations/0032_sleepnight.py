# Generated by Django 3.1.1 on 2020-12-20 20:21

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0031_auto_20201217_2137'),
    ]

    operations = [
        migrations.CreateModel(
            name='SleepNight',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start', models.DateTimeField(verbose_name='start')),
                ('end', models.DateField(verbose_name='end')),
                ('sleep_onset', models.DateField(verbose_name='sleep onset')),
                ('sleep_end', models.DateField(verbose_name='sleep end')),
                ('tst', models.PositiveIntegerField(verbose_name='total sleep time')),
                ('waso', models.PositiveIntegerField(verbose_name='wake after sleep onset')),
                ('se', models.PositiveIntegerField(verbose_name='sleep efficiency')),
                ('data', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dashboard.csvdata')),
            ],
        ),
    ]
