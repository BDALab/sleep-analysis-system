import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, RepeatedStratifiedKFold, cross_validate, \
    train_test_split

from dashboard.logic.cache import save_obj, load_obj
from dashboard.logic.machine_learning.classification_metrics import scoring, sensitivity_score, specificity_score
from dashboard.logic.machine_learning.settings import scale_name, model_params, search_settings, model_name
from dashboard.logic.machine_learning.visualisation import plot_fi, df_into_to_sting, \
    plot_logloss_and_error, plot_cross_validation, shap_summary_plot, shap_beeswarm_plot
from dashboard.models import CsvData
from mysite.settings import ML_DIR, HYPER_PARAMS_PATH, DATASET_PATH, TRAINED_MODEL_PATH, \
    BEST_ESTIMATOR_PATH, CV_RESULTS_PATH

logger = logging.getLogger(__name__)


def prepare_model():
    start = datetime.now()
    learn()
    end = datetime.now()
    logger.info(f'Whole learning process took {end - start}')
    return True


def learn():
    logger.info('Load the data')
    x, y, names = load_data()
    y = y.reshape((len(y),))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    y_train = y_train.reshape((len(y_train),))

    logger.info('Original train data:}')
    log_data_info(y_train)

    # Add NaN according to K-nearest neighbours
    imputer = KNNImputer(n_neighbors=4, weights="uniform")
    x_train = imputer.fit_transform(x_train)

    # Add synthetic values to balance dataset
    # sm = SMOTE(random_state=42)
    # x_train, y_train = sm.fit_sample(x_train, y_train)
    # logger.info('Data after SMOTE synthesis:}')
    # log_data_info(y_train)

    if os.path.exists(TRAINED_MODEL_PATH):
        logger.info('Load model')
        model = load_obj(TRAINED_MODEL_PATH)
    else:
        if os.path.exists(HYPER_PARAMS_PATH):
            params = load_obj(HYPER_PARAMS_PATH)
        else:
            logger.info('Hyper-parameters tuning')
            params = _search_best_hyper_parameters(x_train, y_train)

        logger.info('Cross-validation of params')
        y_train = y_train.ravel()
        model = xgb.sklearn.XGBClassifier(**params)
        cv_results = evaluate_cross_validation(
            model=model,
            x_train=x_train,
            y_train=y_train,
            save_path=CV_RESULTS_PATH)
        logger.info(results_to_print_cv(cv_results))

        plot_cross_validation(cv_results, 'Model binary:logistic')

        train_model_test_train_data(model, x_test, x_train, y_test, y_train)
        save_obj(model, TRAINED_MODEL_PATH)

    # Plot the feature importances
    plot_fi(model, names, scale_name, sort=True, save_dir=ML_DIR)
    plot_logloss_and_error(model, model_name)

    predict = model.predict(x_test)
    logger.info('After training results on test data: ')
    logger.info(results_to_print(y_test, predict))
    logger.info('Confusion matrix: ')
    logger.info(confusion_matrix(y_test, predict))

    predict = model.predict(x)
    logger.info('After training results on whole dataset: ')
    logger.info(results_to_print(y, predict))
    logger.info('Confusion matrix: ')
    logger.info(confusion_matrix(y, predict))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    shap_summary_plot(names, shap_values, x, ML_DIR)
    shap_beeswarm_plot(explainer, names, x, ML_DIR)

    return model


def log_data_info(y_train):
    y_tmp = y_train.ravel()
    logger.info(f'Data len: {len(y_tmp)}')
    sleep = [e for e in y_tmp if e == 1]
    wake = [e for e in y_tmp if e == 0]
    logger.info('Percentage of sleep/wake:')
    logger.info(f'Sleep: {len(sleep)} entries, {(len(sleep) / len(y_tmp)) * 100:.2f}%')
    logger.info(f'Wake: {len(wake)} entries, {(len(wake) / len(y_tmp)) * 100:.2f}%')


def train_model(model, x, y, eval_metrics=["error", "logloss", "auc"]):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35)
    train_model_test_train_data(eval_metrics, model, x_test, x_train, y_test, y_train)


def train_model_test_train_data(model, x_test, x_train, y_test, y_train, eval_metrics=["error", "logloss", "auc"]):
    eval_set = [(x_train, y_train), (x_test, y_test)]
    model.fit(x_train, y_train, early_stopping_rounds=10, eval_metric=eval_metrics, eval_set=eval_set,
              verbose=True)


def load_data():
    if os.path.exists(DATASET_PATH):
        logger.info(f'Load cached dataset')
        super_df = pd.read_excel(DATASET_PATH, index_col=0)
    else:
        data = CsvData.objects.all()
        start = datetime.now()
        frames = []
        for d in data:
            if d.training_data and os.path.exists(d.x_data_path):
                # Load the feature matrix and the label(s)
                df = pd.read_excel(d.x_data_path, index_col=0)
                frames.append(df)

        super_df = pd.concat(frames)
        end = datetime.now()
        logger.info(f'Dataset merged in {end - start}, info: {df_into_to_sting(super_df)}')
        super_df.to_excel(DATASET_PATH)

    x = super_df[[c for c in super_df.columns if c != scale_name]].values
    y = super_df[scale_name].values
    names = [c for c in super_df.columns if c != scale_name]
    return x, y, names


def _search_best_hyper_parameters(x, y):
    start = datetime.now()
    # Create the classifier
    model = xgb.sklearn.XGBClassifier(**model_params)

    # Get the cross-validation indices
    kfolds = StratifiedKFold(n_splits=10, shuffle=True)

    # Employ the hyper-parameter tuning
    y = y.astype(bool)
    random_search = RandomizedSearchCV(model, cv=kfolds.split(x, y), **search_settings)
    random_search.fit(x, y)

    logger.info(f'Estimator: score = {random_search.best_score_:.4f} | model = {random_search.best_estimator_}')
    params = random_search.best_params_
    save_obj(params, HYPER_PARAMS_PATH)
    save_obj(random_search.best_estimator_, BEST_ESTIMATOR_PATH)
    end = datetime.now()
    logger.info(f'Best hyper parameters found and cached in {end - start}')
    return params


def results_to_print_cv(cv_results):
    # Compute the mean and std of the metrics
    cls_report = {
        "acc_avg": round(float(np.mean(cv_results["test_acc"])), 4),
        "acc_std": round(float(np.std(cv_results["test_acc"])), 4),
        "sen_avg": round(float(np.mean(cv_results["test_sen"])), 4),
        "sen_std": round(float(np.std(cv_results["test_sen"])), 4),
        "spe_avg": round(float(np.mean(cv_results["test_spe"])), 4),
        "spe_std": round(float(np.std(cv_results["test_spe"])), 4),
        "f1_avg": round(float(np.mean(cv_results["test_f1"])), 4),
        "f1_std": round(float(np.std(cv_results["test_f1"])), 4),
        "mcc_avg": round(float(np.mean(cv_results["test_mcc"])), 4),
        "mcc_std": round(float(np.std(cv_results["test_mcc"])), 4)
    }
    acc = f"{cls_report['acc_avg']:.2f} ± {cls_report['acc_std']:.2f}"
    sen = f"{cls_report['sen_avg']:.2f} ± {cls_report['sen_std']:.2f}"
    spe = f"{cls_report['spe_avg']:.2f} ± {cls_report['spe_std']:.2f}"
    f1 = f"{cls_report['f1_avg']:.2f} ± {cls_report['f1_std']:.2f}"
    mcc = f"{cls_report['mcc_avg']:.2f} ± {cls_report['mcc_std']:.2f}"
    return f" ACC = {acc} | SEN = {sen} | SPE = {spe} | F1 = {f1} | MCC = {mcc}\n"


def results_to_print(y_test, predict):
    acc = f"{accuracy_score(y_test, predict):.2f}"
    sen = f"{sensitivity_score(y_test, predict):.2f}"
    spe = f"{specificity_score(y_test, predict):.2f}"
    f1 = f"{f1_score(y_test, predict):.2f}"
    mcc = f"{matthews_corrcoef(y_test, predict):.2f}"
    return f" ACC = {acc} | SEN = {sen} | SPE = {spe} | F1 = {f1} | MCC = {mcc}\n"


def evaluate_cross_validation(model, x_train, y_train, save_path):
    start = datetime.now()
    # Prepare the cross-validation scheme
    kfolds = RepeatedStratifiedKFold(n_splits=10, n_repeats=20)

    # Cross-validate the classifier
    cv_results = cross_validate(model, x_train, y_train, scoring=scoring, cv=kfolds,
                                fit_params={"eval_metric": ["rmse", "error", "logloss", "auc"]})
    if save_path:
        save_obj(cv_results, save_path)
    end = datetime.now()
    logger.info(f'Cross validation performed in {end - start}')
    return cv_results
