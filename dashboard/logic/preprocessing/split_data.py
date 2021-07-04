import csv
import logging
from datetime import datetime
from os import path

from dashboard.logic.machine_learning.settings import algorithm, Algorithm
from dashboard.logic.preprocessing.preprocess_csv_data import fix_csv_data, convert_csv_time
from dashboard.logic.sleep_diary.structure import create_structure
from dashboard.models import CsvData, SleepDiaryDay

logger = logging.getLogger(__name__)


def split_data():
    if algorithm == Algorithm.ZAngle:
        return True
    structure = create_structure()
    res = True
    for subject, data, day in structure:
        start_time = datetime.now()

        if not isinstance(data, CsvData) and path.exists(data.data.path):
            res = False
            continue
        if not isinstance(day, SleepDiaryDay):
            res = False
            continue
        export_path = f"{path.split(data.data.path)[0]}//..//split//{subject.code}_{day.date}.csv"
        if path.exists(export_path):
            continue
        if not fix_csv_data(data):
            res = False
            continue

        with open(data.data.path, 'r') as import_file:
            csv_reader = csv.reader(import_file, delimiter=',', quotechar='|')

            with open(export_path, 'w', newline='') as export_file:
                csv_writer = csv.writer(export_file, delimiter=',', quotechar='|')

                export_rows = []
                for csv_row in csv_reader:
                    # now I care just about data, which starts with timestamp starting with 20 --> begin of year
                    if len(csv_row) > 0 and csv_row[0].startswith('20'):
                        csv_date = convert_csv_time(csv_row)
                        if day.t1 <= csv_date <= day.t4:
                            export_rows.append(csv_row)
                        if convert_csv_time(csv_row) > day.t4:
                            break
                csv_writer.writerows(export_rows)
                export_rows.clear()

        end_time = datetime.now()
        logger.info(f'File {export_path} created in {end_time - start_time}')

    return res
