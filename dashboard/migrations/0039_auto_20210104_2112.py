# Generated by Django 3.1.1 on 2021-01-04 20:12

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0038_auto_20210104_1850'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sleepnight',
            name='se',
            field=models.FloatField(verbose_name='sleep efficiency'),
        ),
        migrations.AlterField(
            model_name='sleepnight',
            name='sf',
            field=models.FloatField(verbose_name='sleep fragmentation'),
        ),
    ]
