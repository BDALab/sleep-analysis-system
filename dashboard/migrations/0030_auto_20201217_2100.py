# Generated by Django 3.1.1 on 2020-12-17 20:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0029_rbdsq'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rbdsq',
            name='q63',
            field=models.BooleanField(verbose_name='6.3. i have or had the following phenomena during my dreams: gestures, complex movements, that are useless during sleep, e.g., to wave, to salute, to frighten mosquitoes, falls off the bed'),
        ),
        migrations.AlterField(
            model_name='rbdsq',
            name='q64',
            field=models.BooleanField(verbose_name='6.4 i have or had the following phenomena during my dreams: things that fell down around the bed, e.g., bedside lamp, book, glasses'),
        ),
    ]