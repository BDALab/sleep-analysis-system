import logging
from os import path

from pandas import read_excel
from sklearn.metrics import accuracy_score

from dashboard.logic.features_extraction.data_entry import DataEntry
from mysite.settings import HYPER_PARAMS_PATH, MODEL_PATH, TRAINED_MODEL_PATH, DATASET_PATH
from .cache import load_obj
from .machine_learning.settings import scale_name
from ..models import Subject, CsvData, PsData

logger = logging.getLogger(__name__)


def _check_files(data):
    for entry in data:
        broken = False
        if not path.exists(entry.data.path):
            logger.error(f"File {data} on path {data.data.path} does not exists!")
            broken = True
        return broken


def _check_prediction(data):
    for entry in data:
        broken = False
        if entry.prediction_cached and not path.exists(entry.cached_prediction_path):
            logger.error(f"Prediction file {data} on path {entry.cached_prediction_path} does not exists!")
            broken = True
        return broken


def check_all_data():
    broken = False
    if _check_files(CsvData.objects.all()):
        broken = True
    if _check_files(PsData.objects.all()):
        broken = True

    if _check_prediction(CsvData.objects.all()):
        broken = True
    if _check_prediction(PsData.objects.all()):
        broken = True

    subjects_len = len(Subject.objects.all())
    csv_len = len(CsvData.objects.all())
    ps_len = len(PsData.objects.all())
    logger.info(f'subjects count: {subjects_len} | csv data count: {csv_len} | ps data count: {ps_len}')
    if ps_len != csv_len:
        logger.warning('Lens are not same, some data may missing')
        broken = True

    csv_full = len(CsvData.objects.filter(full_data=True))
    csv_epoch = len(CsvData.objects.filter(full_data=False))
    logger.info(f'csv full count: {csv_full} | csv epoch count: {csv_epoch}')
    if csv_full != csv_epoch:
        logger.warning('Lens are not same, some data may missing')
        broken = True

    if not check_cached_data():
        broken = True

    if not check_extracted_features():
        broken = True

    if not check_model():
        broken = True

    return not broken


def check_extracted_features():
    broken = False
    data = CsvData.objects.filter(features_extracted=True)
    logger.info(f'{len(data)} csv objects claim to have features extracted')
    for d in data:
        broken = _check_extracted_features_entry(broken, d.features_data_path, number_of_features=91)
    return not broken


def check_predicted_features():
    broken = False
    data = CsvData.objects.filter(prediction_cached=True)
    logger.info(f'{len(data)} csv objects claim to have predictions cached')
    for d in data:
        broken = _check_extracted_features_entry(broken, d.cached_prediction_path, number_of_features=91)
    return not broken


def _check_extracted_features_entry(broken, data_path, number_of_features):
    if path.exists(data_path):
        df = read_excel(data_path, index_col=0)
        (X, Y) = df.shape
        if not X > 0:
            logger.warning(f'There are no data in file {data_path}')
            broken = True
        elif not Y == number_of_features:
            logger.warning(f'Some features are missing in file {data_path}, '
                           f'there is {Y} features, expected was {number_of_features} features')
            broken = True

    else:
        logger.warning(f'File {data_path} missing!')
        broken = True

    return broken


def check_cached_data():
    broken = False
    data = CsvData.objects.filter(data_cached=True)
    logger.info(f'{len(data)} csv objects claim to have cached data')
    for d in data:
        broken = _check_cached_data_entry(broken, d)
    return not broken


def _check_cached_data_entry(broken, d):
    if isinstance(d, CsvData):
        if path.exists(d.cached_data_path):
            entry_list = load_obj(d.cached_data_path)
            if not entry_list:
                logger.warning(f'There are no data in file {d.cached_data_path}')
                broken = True
            elif not isinstance(entry_list, list):
                logger.warning(f'Data in file {d.cached_data_path} are broken')
                broken = True
            else:
                first = next(iter(entry_list), None)
                if first is None or not isinstance(first, DataEntry):
                    logger.warning(f'Data in file {d.cached_data_path} are broken')
                    broken = True
                elif isinstance(first, DataEntry):
                    if not first.accelerometer:
                        logger.warning(f'Accelerometer data in file {d.cached_data_path} are broken')
                        broken = True
                    elif not first.temperature:
                        logger.warning(f'Temperature data in file {d.cached_data_path} are broken')
                        broken = True
                else:
                    logger.warning(f'Data in file {d.cached_data_path} are broken')
                    broken = True
        else:
            logger.warning(f'File {d.cached_data_path} missing!')
            broken = True
    else:
        broken = True
    return broken


def check_model():
    if path.exists(HYPER_PARAMS_PATH) \
            and path.exists(MODEL_PATH) \
            and path.exists(TRAINED_MODEL_PATH) \
            and path.exists(DATASET_PATH):
        df = read_excel(DATASET_PATH, index_col=0)
        x = df[[c for c in df.columns if c != scale_name]].values
        model = load_obj(TRAINED_MODEL_PATH)
        predictions = model.predict(x)
        y_test = df[scale_name].values
        accuracy = accuracy_score(y_test, predictions)
        logger.info("Trained model Accuracy: %.2f%%" % (accuracy * 100.0))
        if accuracy > 0.8:
            return True
        else:
            logger.warning('Accuracy is low!')
            return False
    else:
        logger.warning(f'Model or some of the data are missing. '
                       f'| Model: {path.exists(MODEL_PATH)}'
                       f'| Trained model: {path.exists(TRAINED_MODEL_PATH)}'
                       f'| Cached Dataset: {path.exists(DATASET_PATH)}'
                       f'| Hyper parameters: {path.exists(HYPER_PARAMS_PATH)}')
        return False
