# Generated by Django 2.2.5 on 2020-02-26 21:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0017_auto_20200224_2129'),
    ]

    operations = [
        migrations.AddField(
            model_name='csvdata',
            name='data_cached',
            field=models.BooleanField(default=0, editable=False, verbose_name='data cached'),
        ),
        migrations.AddField(
            model_name='psdata',
            name='data_cached',
            field=models.BooleanField(default=0, editable=False, verbose_name='data cached'),
        ),
    ]
