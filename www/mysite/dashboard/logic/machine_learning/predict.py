import logging
import os
from datetime import datetime

import pandas as pd

from dashboard.logic import cache
from dashboard.logic.features_extraction.extract_features import extract_features
from dashboard.logic.machine_learning.settings import scale_name, prediction_name
from dashboard.logic.multithread import parallel_for
from dashboard.logic.preprocessing.preprocess_data import preprocess_data
from dashboard.models import CsvData
from mysite.settings import TRAINED_MODEL_PATH

logger = logging.getLogger(__name__)


def predict_all():
    start = datetime.now()
    data = CsvData.objects.all()
    logger.info(f'{len(data)} csv data objects will be used for prediction')
    results = parallel_for(data, predict)
    end = datetime.now()
    logger.info(f'Prediction of all the {len(data)} data took {end - start}')
    for r in results:
        if r.result() is None:
            return False
    return True


def predict(csv_data, force=False):
    if isinstance(csv_data, CsvData):
        start = datetime.now()

        if os.path.exists(csv_data.cached_prediction_path) and not force:
            logger.info(f'Prediction features data for {csv_data.filename} will be loaded from cache')
            df = cache.load_obj(csv_data.cached_prediction_path)
            return df

        else:
            logger.info(f'Data {csv_data.filename} need to be preprocessed')
            result = preprocess_data(csv_data)
            if not result:
                logger.warning(f'Data {csv_data.filename} cannot be preprocessed')
                return None

            logger.info(f'Features for {csv_data.filename} need to be extracted')
            result = extract_features(csv_data)
            if not result:
                logger.warning(f'Features cannot be extracted for {csv_data.filename}')
                return None

            df = pd.read_excel(csv_data.features_data_path, index_col=0)
            logger.info(f'Prediction need to be done for {csv_data.filename}')
            predictions = _predict(df)
            df[prediction_name] = predictions
            cache.save_obj(df, csv_data.cached_prediction_path)
            df.to_excel(csv_data.excel_prediction_path)
            csv_data.prediction_cached = True
            csv_data.save()

            end = datetime.now()
            logger.info(f'Prediction for {csv_data.filename} made in {end - start}')
            return df
    else:
        return None


def _predict(df):
    x = df[[c for c in df.columns if c != scale_name]].values
    model = cache.load_obj(TRAINED_MODEL_PATH)
    predictions = model.predict(x)
    return predictions
