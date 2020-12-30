# Generated by Django 3.1.1 on 2020-12-26 21:50

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0034_auto_20201226_2150'),
    ]

    operations = [
        migrations.AddField(
            model_name='sleepnight',
            name='data',
            field=models.ForeignKey(default=0, on_delete=django.db.models.deletion.CASCADE, to='dashboard.csvdata'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='sleepnight',
            name='subject',
            field=models.ForeignKey(default=0, on_delete=django.db.models.deletion.CASCADE, to='dashboard.subject'),
            preserve_default=False,
        ),
    ]