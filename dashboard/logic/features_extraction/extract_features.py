import logging
from datetime import datetime

from pandas import DataFrame

from dashboard.logic.cache import load_obj
from dashboard.logic.multithread import parallel_for
from dashboard.models import CsvData
from .data_entry import DataEntry
from ..machine_learning.settings import algorithm, Algorithm

logger = logging.getLogger(__name__)


def extract_features_all():
    if algorithm == Algorithm.ZAngle:
        return True
    total_start = datetime.now()

    cached = CsvData.objects.filter(data_cached=True).filter(features_extracted=False)
    logger.info(f'{len(cached)} cached csv data object will be used')
    results = parallel_for(cached, extract_features)

    total_end = datetime.now()
    logger.info(f'For {len(cached)} csv data objects were features extracted in {total_end - total_start}')

    for r in results:
        if not r.result():
            return False
    return True


def extract_features(csv_object):
    start = datetime.now()
    features = []
    times = []
    if isinstance(csv_object, CsvData):
        if csv_object.features_extracted:
            return True
        data = load_obj(csv_object.cached_data_path)
        for entry in data:
            if not isinstance(entry, DataEntry):
                logger.error(f'Preprocessed data for {csv_object} on path {csv_object.cached_data_path} are broken!')
                return False
            if not entry.accelerometer:
                break
            entry_features = entry.get_features()
            if csv_object.training_data:
                entry_features['SLEEP'] = entry.sleep
            features.append(entry_features)
            times.append(entry.time)
        df = DataFrame(features, index=times)
        df.to_excel(csv_object.features_data_path)
        csv_object.features_extracted = True
        csv_object.save()
        end = datetime.now()
        logger.info(f'Features for {csv_object.filename} extracted in {end - start}')
        return True
    else:
        return False
