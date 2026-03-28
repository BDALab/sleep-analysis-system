import csv
import logging
import os
import re
from datetime import datetime

from dashboard.models import CsvData

logger = logging.getLogger(__name__)
CSV_TIMESTAMP_RE = re.compile(r'^\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')


def fix_csv_data(csv_data):
    if isinstance(csv_data, CsvData):
        path = csv_data.data.path
        tmp_path = _create_tmp_path(path)

        try:
            _fix_csv_data(path, tmp_path)
            os.remove(path)
            os.rename(tmp_path, path)
            return True
        except OSError:
            return False

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return False


def _create_tmp_path(path):
    split = os.path.split(path)
    tmp_path = f'{split[0]}/tmp_{split[1]}'
    return tmp_path


def _fix_csv_data(orig_path, fix_path):
    fi = open(orig_path, 'rb')
    content = fi.read()
    fi.close()
    fo = open(fix_path, 'wb')
    fo.write(content.replace(b'\x00', b''))
    fo.close()


def get_csv_start(csv_data):
    if isinstance(csv_data, CsvData):
        with open(csv_data.data.path, 'r', encoding='latin-1', errors='ignore', newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            for row in reader:
                # now I care just about data, which starts with timestamp starting with 20 --> begin of year
                if len(row) > 0 and row[0].startswith('20'):
                    try:
                        return convert_csv_time(row)
                    except ValueError:
                        continue
        return None
    return None


def convert_csv_time(row):
    timestamp = row[0].replace('\x00', '').strip()
    match = CSV_TIMESTAMP_RE.match(timestamp)
    if match is None:
        raise ValueError(f'Invalid csv timestamp: {timestamp!r}')
    return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
