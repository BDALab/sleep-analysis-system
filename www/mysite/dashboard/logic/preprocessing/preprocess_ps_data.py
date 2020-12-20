import csv
from datetime import datetime, timedelta

from dashboard.models import PsData


def get_ps_start(ps_data):
    if isinstance(ps_data, PsData):
        with open(ps_data.data.path, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter='\t', quotechar='|')
            header_end = False
            after_midnight = False
            for row in reader:
                if header_end and row != []:
                    if row[2].startswith('00'):
                        after_midnight = True
                    return convert_ps_timestamp(date, row[2], after_midnight)
                if row == ['Sleep Stage', 'Position', 'Time [hh:mm:ss]', 'Event', 'Duration[s]']:
                    header_end = True
                if next(iter(row or []), None) == 'Recording Date:':
                    date = row[1].split()[0]
        return None
    return None


def convert_ps_timestamp(date, time, after_midnight):
    dt = datetime.strptime(f'{date} {time}', '%d/%m/%Y %H:%M:%S')
    if after_midnight:
        return dt + timedelta(days=1)
    return dt


# Sleep status (0 = awake, 1 = sleep)
def convert_sleep(string):
    sleep_strings = ['R', 'N1', 'N2', 'N3']
    if string in sleep_strings:
        return 1
    return 0
