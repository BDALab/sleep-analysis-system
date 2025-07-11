# Generated by Django 3.1.14 on 2024-09-07 15:13

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0050_auto_20240707_1909'),
    ]

    operations = [
        migrations.CreateModel(
            name='SleeppyData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sleep_onset', models.DateTimeField(verbose_name='sleep onset')),
                ('sleep_end', models.DateTimeField(verbose_name='sleep end')),
                ('sol', models.PositiveIntegerField(verbose_name='sleep onset latency')),
                ('waso', models.PositiveIntegerField(verbose_name='wake after sleep onset')),
                ('wb', models.PositiveIntegerField(verbose_name='wake bouts')),
                ('awk5plus', models.PositiveSmallIntegerField(verbose_name='awakenings > 5 minutes')),
                ('tst', models.PositiveIntegerField(verbose_name='total sleep time')),
                ('se', models.PositiveIntegerField(verbose_name='sleep efficiency')),
                ('data', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dashboard.csvdata')),
                ('sleep_night', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.DO_NOTHING,
                                                  to='dashboard.sleepnight')),
                ('subject', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dashboard.subject')),
            ],
        ),
    ]
