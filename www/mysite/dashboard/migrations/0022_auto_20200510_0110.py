# Generated by Django 3.0.3 on 2020-05-09 23:10

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('dashboard', '0021_csvdata_features_extracted'),
    ]

    operations = [
        migrations.RenameField(
            model_name='csvdata',
            old_name='graph_created',
            new_name='prediction_cached',
        ),
        migrations.RemoveField(
            model_name='csvdata',
            name='full_data',
        ),
        migrations.RemoveField(
            model_name='psdata',
            name='graph_created',
        ),
        migrations.AddField(
            model_name='csvdata',
            name='training_data',
            field=models.BooleanField(default=False, verbose_name='training data'),
        ),
    ]
