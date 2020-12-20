import logging
from os import path, remove

from dashboard.models import CsvData
from mysite.settings import HYPER_PARAMS_PATH, MODEL_PATH, DATASET_PATH, TRAINED_MODEL_PATH

logger = logging.getLogger(__name__)


def delete_all_data():
    try:
        for csv_data in CsvData.objects.all():
            _delete_cached_prediction(csv_data)
            _delete_cached_data(csv_data)
            _delete_extracted_features(csv_data)
        delete_model()
        return True
    except OSError:
        return False


def delete_predicted_data():
    try:
        for csv_data in CsvData.objects.all():
            _delete_cached_prediction(csv_data)
        return True
    except OSError:
        return False


def delete_cached_data():
    try:
        for csv_data in CsvData.objects.all():
            _delete_cached_data(csv_data)
        return True
    except OSError:
        return False


def delete_extracted_features():
    try:
        for csv_data in CsvData.objects.all():
            _delete_extracted_features(csv_data)
        return True
    except OSError:
        return False


def delete_model():
    try:
        safe_delete(HYPER_PARAMS_PATH)
        safe_delete(MODEL_PATH)
        safe_delete(DATASET_PATH)
        safe_delete(TRAINED_MODEL_PATH)
        return True
    except OSError:
        return False


def _delete_cached_data(data):
    if data.data_cached:
        safe_delete(data.cached_data_path)
        data.data_cached = False
        data.save()


def _delete_cached_prediction(data):
    if data.prediction_cached:
        safe_delete(data.cached_prediction_path)
        data.prediction_cached = False
        data.save()


def _delete_extracted_features(data):
    if data.features_extracted:
        safe_delete(data.features_data_path)
        data.features_extracted = False
        data.save()


def safe_delete(data):
    if path.exists(data):
        remove(data)
