# Generated by Django 3.1.1 on 2020-09-29 21:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0027_auto_20200928_2312'),
    ]

    operations = [
        migrations.AddField(
            model_name='sleepdiaryday',
            name='note',
            field=models.CharField(blank=True, max_length=255, verbose_name='note'),
        ),
    ]
