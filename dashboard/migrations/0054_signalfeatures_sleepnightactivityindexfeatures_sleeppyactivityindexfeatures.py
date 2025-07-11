# Generated by Django 3.1.14 on 2024-09-18 16:27

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0053_sleeppydata_broken'),
    ]

    operations = [
        migrations.CreateModel(
            name='SignalFeatures',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('max', models.FloatField(verbose_name='max')),
                ('min', models.FloatField(verbose_name='min')),
                ('relative_position_of_max', models.FloatField(verbose_name='relative position of max')),
                ('relative_position_of_min', models.FloatField(verbose_name='relative position of min')),
                ('range', models.FloatField(verbose_name='range')),
                ('relative_range', models.FloatField(verbose_name='relative range')),
                ('relative_variation_range', models.FloatField(verbose_name='relative variation range')),
                ('interquartile_range', models.FloatField(verbose_name='interquartile range')),
                ('relative_interquartile_range', models.FloatField(verbose_name='relative interquartile range')),
                ('interdencile_range', models.FloatField(verbose_name='interdencile range')),
                ('relative_interdencile_range', models.FloatField(verbose_name='relative interdencile range')),
                ('interpercentile_range', models.FloatField(verbose_name='interpercentile range')),
                ('relative_interpercentile_range', models.FloatField(verbose_name='relative interpercentile range')),
                ('studentized_range', models.FloatField(verbose_name='studentized range')),
                ('mean', models.FloatField(verbose_name='mean')),
                ('harmonic_mean', models.FloatField(verbose_name='harmonic mean')),
                ('mean_excluding_outliers_10', models.FloatField(verbose_name='mean excluding outliers (10)')),
                ('mean_excluding_outliers_20', models.FloatField(verbose_name='mean excluding outliers (20)')),
                ('mean_excluding_outliers_30', models.FloatField(verbose_name='mean excluding outliers (30)')),
                ('mean_excluding_outliers_40', models.FloatField(verbose_name='mean excluding outliers (40)')),
                ('median', models.FloatField(verbose_name='median')),
                ('mode', models.FloatField(verbose_name='mode')),
                ('variance', models.FloatField(verbose_name='variance')),
                ('standard_deviation', models.FloatField(verbose_name='standard deviation')),
                ('median_absolute_deviation', models.FloatField(verbose_name='median absolute deviation')),
                ('relative_standard_deviation', models.FloatField(verbose_name='relative standard deviation')),
                ('index_of_dispersion', models.FloatField(verbose_name='index of dispersion')),
                ('kurtosis', models.FloatField(verbose_name='kurtosis')),
                ('skewness', models.FloatField(verbose_name='skewness')),
                (
                'pearson_1st_skewness_coefficient', models.FloatField(verbose_name='pearson 1st skewness coefficient')),
                (
                'pearson_2st_skewness_coefficient', models.FloatField(verbose_name='pearson 2st skewness coefficient')),
                ('percentile_1', models.FloatField(verbose_name='1st percentile')),
                ('percentile_5', models.FloatField(verbose_name='5th percentile')),
                ('percentile_10', models.FloatField(verbose_name='10th percentile')),
                ('percentile_20', models.FloatField(verbose_name='20th percentile')),
                ('percentile_80', models.FloatField(verbose_name='80th percentile')),
                ('percentile_90', models.FloatField(verbose_name='90th percentile')),
                ('percentile_95', models.FloatField(verbose_name='95th percentile')),
                ('percentile_99', models.FloatField(verbose_name='99th percentile')),
                ('shannon_entropy', models.FloatField(verbose_name='shannon entropy')),
                ('modulation', models.FloatField(verbose_name='modulation')),
                ('tkeo_max', models.FloatField(verbose_name='teager kaiser energy operator max')),
                ('tkeo_min', models.FloatField(verbose_name='teager kaiser energy operator min')),
            ],
        ),
        migrations.CreateModel(
            name='SleeppyActivityIndexFeatures',
            fields=[
                ('signalfeatures_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='dashboard.signalfeatures')),
                ('day_index', models.PositiveSmallIntegerField(verbose_name='sleeppy index')),
                ('sleeppy_data',
                 models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dashboard.sleeppydata')),
            ],
            bases=('dashboard.signalfeatures',),
        ),
        migrations.CreateModel(
            name='SleepNightActivityIndexFeatures',
            fields=[
                ('signalfeatures_ptr',
                 models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True,
                                      primary_key=True, serialize=False, to='dashboard.signalfeatures')),
                ('day_index', models.PositiveSmallIntegerField(verbose_name='sleeppy index')),
                ('sleep_night',
                 models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dashboard.sleepnight')),
            ],
            bases=('dashboard.signalfeatures',),
        ),
    ]
