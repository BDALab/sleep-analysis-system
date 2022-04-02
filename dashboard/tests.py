import unittest
from os import path

import pandas
import xgboost as xgb
from parameterized import parameterized
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

from dashboard.logic import cache
from dashboard.logic.machine_learning.learn import load_data, results_to_print_cv
from dashboard.logic.machine_learning.settings import scale_name
from dashboard.logic.machine_learning.visualisation import plot_logloss_and_error, plot_fi, plot_cross_validation
from mysite.settings import BASE_DIR, TRAINED_MODEL_PATH, TRAINED_MODEL_EXPORT_PATH


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = cache.load_obj(TRAINED_MODEL_PATH)

    @parameterized.expand(
        [
            [path.join(BASE_DIR, 'media/features/MECSLEEP01_left_wrist_012870_2013-06-12_11-40-37.csv.xlsx')],
            [path.join(BASE_DIR, 'media/features/MECSLEEP01_right_wrist_012855_2013-06-11_12-08-25.csv.xlsx')],
            [path.join(BASE_DIR, 'media/features/MECSLEEP39_left_wrist_018145_2014-12-19_16-51-20.csv.xlsx')],
            [path.join(BASE_DIR, 'media/features/MECSLEEP48_right_wrist_018140_2015-04-09_16-45-53.csv.xlsx')],
            [path.join(BASE_DIR, 'media/features/MECSLEEP50_right_wrist_018141_2015-04-21_14-18-26.csv.xlsx')],
            [path.join(BASE_DIR, 'media/features/MECSLEEP60_right_wrist_018141_2015-05-21_15-32-02.csv.xlsx')],
        ]
    )
    def test_model_accuracy(self, data_path):
        df = pandas.read_excel(data_path, index_col=0)
        x = df[[c for c in df.columns if c != scale_name]].values
        predictions = self.model.predict(x)
        y_test = df[scale_name].values
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        self.assertTrue(accuracy > 0.8)

    @parameterized.expand(
        [
            [path.join(BASE_DIR, 'media/predictions/185936__044351_2019-02-11_15-15-05.csv.pkl')],
            [path.join(BASE_DIR, 'media/predictions/MECSLEEP01_left_wrist_012870_2013-06-12_11-40-37.csv.pkl')],
            [path.join(BASE_DIR, 'media/predictions/MECSLEEP01_right_wrist_012855_2013-06-11_12-08-25.csv.pkl')],
            [path.join(BASE_DIR, 'media/predictions/MECSLEEP39_left_wrist_018145_2014-12-19_16-51-20.csv.pkl')],
            [path.join(BASE_DIR, 'media/predictions/MECSLEEP48_right_wrist_018140_2015-04-09_16-45-53.csv.pkl')],
            [path.join(BASE_DIR, 'media/predictions/MECSLEEP50_right_wrist_018141_2015-04-21_14-18-26.csv.pkl')],
            [path.join(BASE_DIR, 'media/predictions/MECSLEEP60_right_wrist_018141_2015-05-21_15-32-02.csv.pkl')],
        ]
    )
    def test_model_on_unknown_data(self, data_path):
        df = cache.load_obj(data_path)
        x = df[[c for c in df.columns if c != scale_name]].values
        predictions = self.model.predict(x)
        sleep = [x for x in predictions if x == 1]
        wake = [x for x in predictions if x == 0]
        r_sleep = len(sleep) / len(predictions)
        r_wake = len(wake) / len(predictions)
        print("Sleep: %.2f%%" % (r_sleep * 100.0))
        print("Wake: %.2f%%" % (r_wake * 100.0))
        self.assertTrue(1 > r_sleep > 0)
        self.assertTrue(1 > r_wake > 0)


class ModelTuningsTest(unittest.TestCase):
    @parameterized.expand(
        [
            [path.join(BASE_DIR, 'www/mysite/ml/trained/29-5-modA'), 'Model A'],
            [path.join(BASE_DIR, 'www/mysite/ml/trained/24-5-modB'), 'Model B'],
            [path.join(BASE_DIR, 'www/mysite/ml/trained/27-5-modC'), 'Model C'],
        ]
    )
    def test_make_results_for_models(self, model_dir, name):
        print(f'Model: {name}')
        model = cache.load_obj(f'{model_dir}/trained_model.pkl')
        plot_logloss_and_error(model, name=name, save_dir=model_dir)

        x, y, names = load_data()
        results = cache.load_obj(f'{model_dir}/cv_results.pkl')
        plot_fi(model, names, scale_name, sort=True, save_dir=model_dir)
        plot_cross_validation(results, name=name, save_dir=model_dir)
        print(results_to_print_cv(results))

        _test_prediction(model, x, y)

        self.assertTrue(True)

    def test_dataset_percents(self):
        x, y, names = load_data()
        y = y.ravel()
        sleep = [e for e in y if e == 1]
        wake = [e for e in y if e == 0]
        print('Percentage of sleep/wake of dataset:')
        print(f'Sleep: {len(sleep)} entries, {(len(sleep) / len(y)) * 100:.2f}%')
        print(f'Sleep: {len(wake)} entries, {(len(wake) / len(y)) * 100:.2f}%')
        self.assertEquals(len(sleep) + len(wake), len(y))


def _test_prediction(model, x, y):
    predict = model.predict(x)
    print(
        f'ACC: {accuracy_score(y, predict):.2f} | F1: {f1_score(y, predict):.2f} | MCC: {matthews_corrcoef(y, predict)}')
    print(confusion_matrix(y, predict))
    print(classification_report(y, predict))


class SaveModelTest(unittest.TestCase):
    def test_save(self):
        model = cache.load_obj(TRAINED_MODEL_PATH)
        x, y, names = load_data()
        if isinstance(model, xgb.sklearn.XGBClassifier):
            model._Booster.save_model(TRAINED_MODEL_EXPORT_PATH)
            _test_prediction(model, x, y)
        import_model = xgb.sklearn.XGBClassifier()
        import_model.load_model(TRAINED_MODEL_PATH)
        _test_prediction(import_model, x, y)

    def _test_prediction(model, x, y):
        predict = model.predict(x)
        print(
            f'ACC: {accuracy_score(y, predict):.2f} | F1: {f1_score(y, predict):.2f} | MCC: {matthews_corrcoef(y, predict)}')
        print(confusion_matrix(y, predict))
        print(classification_report(y, predict))
