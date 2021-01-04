# Generated by Django 3.1.1 on 2021-01-04 17:50

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0037_auto_20201227_2334'),
    ]

    operations = [
        migrations.AddField(
            model_name='sleepnight',
            name='sf',
            field=models.PositiveIntegerField(default=0, verbose_name='sleep fragmentation'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='sleepnight',
            name='sol',
            field=models.PositiveIntegerField(default=0, verbose_name='sleep onset latency'),
            preserve_default=False,
        ),
    ]