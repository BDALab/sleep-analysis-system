import logging
import os.path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from dashboard.logic.cache import save_obj
from dashboard.logic.machine_learning.learn import evaluate_cross_validation, results_to_print, \
    train_model_test_train_data, results_to_print_cv
from dashboard.logic.machine_learning.settings import model_params, search_settings
from dashboard.logic.machine_learning.visualisation import plot_cross_validation, plot_fi, plot_logloss_and_error, \
    set_visual_styles
from mysite.settings import HILEV_FNUSA, HILEV_CV_RESULTS_PATH, HILEV_DIR, HILEV_TRAINED_MODEL_PATH

logger = logging.getLogger(__name__)


def train_parkinson_classifier():
    df = pd.read_excel(HILEV_FNUSA)

    names = ['Time in bed (A)',
             'Sleep onset latency (A)',
             'Sleep onset latency - norm (A)',
             'Wake after sleep onset (A)',
             'Wake after sleep onset - norm (A)',
             'Wake after sleep offset (A)',
             'Total sleep time (A)',
             'Wake bouts (A)',
             'Awakening > 5 minutes (A)',
             'Awakening > 5 minutes - norm (A)',
             'Sleep efficiency (A)',
             'Sleep efficiency - norm (A)',
             'Sleep fragmentation (A)',

             'Time in bed (D)',
             'Sleep onset latency (D)',
             'Sleep onset latency - norm (D)',
             'Wake after sleep onset (D)',
             'Wake after sleep onset - norm (D)',
             'Wake after sleep offset (D)',
             'Total sleep time (D)',
             'Wake bouts (D)',
             'Awakening > 5 minutes (D)',
             'Awakening > 5 minutes - norm (D)',
             'Sleep efficiency (D)',
             'Sleep efficiency - norm (D)',
             'Sleep fragmentation (D)',
             ]

    na = df.columns
    x = df[names].values
    y = (df['Probable Parkinson Disease'] | df['Probable Mild Cognitive Impairment']).values
    y = y.reshape((len(y),))

    if not os.path.exists(HILEV_DIR):
        os.mkdir(HILEV_DIR)
    model = learn_model(x, y, names)
    # model = load_obj(HILEV_TRAINED_MODEL_PATH)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    set_visual_styles()
    f = plt.figure()
    shap.summary_plot(shap_values, x, plot_type="bar", feature_names=names)
    f.savefig(f'{HILEV_DIR}/summary_plot.png', bbox_inches='tight', dpi=600)
    f.clf()

    # summarize the effects of all the features
    f = plt.figure()
    shap_obj = explainer(x)
    shap_obj.feature_names = names
    shap.plots.beeswarm(shap_obj, max_display=len(names))
    f.savefig(f'{HILEV_DIR}/beeswarm.png', bbox_inches='tight', dpi=600)

    return True


def learn_model(x, y, names):
    start = datetime.now()
    logger.info('Whole dataset:}')
    log_data_info(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=13)
    y_train = y_train.reshape((len(y_train),))
    logger.info('Original train data:}')
    log_data_info(y_train)

    # Add NaN according to K-nearest neighbours
    imputer = KNNImputer(n_neighbors=4, weights="uniform")
    x_train = imputer.fit_transform(x_train)

    # Add synthetic values to balance dataset
    # sm = SMOTE(random_state=27)
    # x_train, y_train = sm.fit_sample(x_train, y_train)

    logger.info('Data after SMOTE synthesis:}')
    log_data_info(y_train)

    # search hyper params
    params = _search_best_hyper_parameters(x_train, y_train)

    logger.info('Cross-validation of params')
    y_train = y_train.ravel()
    model = xgboost.sklearn.XGBClassifier(**params)
    cv_results = evaluate_cross_validation(
        model=model,
        x_train=x_train,
        y_train=y_train,
        save_path=HILEV_CV_RESULTS_PATH)

    logger.info(results_to_print_cv(cv_results))
    plot_cross_validation(cv_results, 'Model binary:logistic', HILEV_DIR)

    # final training of model
    train_model_test_train_data(model, x_test, x_train, y_test, y_train)
    save_obj(model, HILEV_TRAINED_MODEL_PATH)

    # Plot the feature importances
    plot_fi(model, names, 'Probable Parkinson Disease', sort=True, save_dir=HILEV_DIR)
    plot_logloss_and_error(model, 'first', HILEV_DIR)

    # predict on test data
    predict = model.predict(x_test)
    logger.info('After training results on test data: ')
    logger.info(results_to_print(y_test, predict))
    logger.info('Confusion matrix: ')
    logger.info(confusion_matrix(y_test, predict))

    # predict on whole dataset
    predict = model.predict(x)
    logger.info('After training results on whole dataset: ')
    logger.info(results_to_print(y_test, predict))
    logger.info('Confusion matrix: ')
    logger.info(confusion_matrix(y, predict))

    end = datetime.now()
    logger.info(f'Learning process took {end - start}')

    return model


def log_data_info(y_train):
    y_tmp = y_train.ravel()
    logger.info(f'Data len: {len(y_tmp)}')
    parkinson = [e for e in y_tmp if e]
    healthy = [e for e in y_tmp if not e]
    logger.info('Percentage of parkinson/healthy people:')
    logger.info(f'Parkinson: {len(parkinson)} entries, {(len(parkinson) / len(y_tmp)) * 100:.2f}%')
    logger.info(f'Healthy: {len(healthy)} entries, {(len(healthy) / len(y_tmp)) * 100:.2f}%')


def _search_best_hyper_parameters(x, y):
    start = datetime.now()
    # Create the classifier
    model = xgboost.sklearn.XGBClassifier(**model_params)

    # Get the cross-validation indices
    kfolds = StratifiedKFold(n_splits=10, shuffle=True)

    # Employ the hyper-parameter tuning
    y = y.astype(bool)
    random_search = RandomizedSearchCV(model, cv=kfolds.split(x, y), **search_settings)
    random_search.fit(x, y)

    logger.info(f'Estimator: score = {random_search.best_score_:.4f} | model = {random_search.best_estimator_}')
    params = random_search.best_params_
    end = datetime.now()
    logger.info(f'Best hyper parameters found and cached in {end - start}')
    return params
