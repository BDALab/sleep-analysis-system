# Generated by Django 3.1.1 on 2021-06-20 21:47

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0039_auto_20210104_2112'),
    ]

    operations = [
        migrations.AddField(
            model_name='sleepnight',
            name='awk5plus',
            field=models.PositiveSmallIntegerField(default=0, verbose_name='awakenings > 5 minutes'),
            preserve_default=False,
        ),
    ]
