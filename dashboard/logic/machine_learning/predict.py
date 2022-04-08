import logging
import os
from datetime import datetime

import pandas as pd

from dashboard.logic import cache
from dashboard.logic.machine_learning.predict_core import predict_core
from dashboard.logic.preprocessing.preprocess_data import preprocess_data
from dashboard.models import CsvData

logger = logging.getLogger(__name__)


def predict_all():
    start = datetime.now()
    data = CsvData.objects.all()
    logger.info(f'{len(data)} csv data objects will be used for prediction')
    result = True
    for d in data:
        if predict(d) is None:
            result = False
    end = datetime.now()
    logger.info(f'Prediction of all the {len(data)} data took {end - start}')
    return result


def predict(csv_data, force=False):
    if isinstance(csv_data, CsvData):
        start = datetime.now()

        if os.path.exists(csv_data.cached_prediction_path) and not force:
            logger.info(f'Prediction features data for {csv_data.filename} will be loaded from cache')
            df = cache.load_obj(csv_data.cached_prediction_path)
            return df

        else:
            result = preprocess_data(csv_data)
            if not result:
                logger.warning(f'Data {csv_data.filename} cannot be preprocessed')
                return None

            logger.info(f'Prediction need to be done for {csv_data.filename}')
            df = pd.read_excel(csv_data.x_data_path, index_col=0)
            predict_core(csv_data, df)

            end = datetime.now()
            logger.info(f'Prediction for {csv_data.filename} made in {end - start}')
            return df
    else:
        return None


