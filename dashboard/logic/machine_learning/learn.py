import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import (
    StratifiedKFold,
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_validate,
)

from dashboard.logic.cache import save_obj, load_obj
from dashboard.logic.machine_learning.classification_metrics import scoring, sensitivity_score, specificity_score
from dashboard.logic.machine_learning.settings import scale_name, model_params, search_settings, model_name
from dashboard.logic.machine_learning.visualisation import plot_fi, df_into_to_sting, \
    plot_logloss_and_error, plot_cross_validation, shap_summary_plot, shap_beeswarm_plot
from dashboard.models import CsvData
from mysite.settings import ML_DIR, HYPER_PARAMS_PATH, DATASET_PATH, DATASET_PARQUET_PATH, TRAINED_MODEL_PATH, \
    BEST_ESTIMATOR_PATH, CV_RESULTS_PATH, TRAINED_MODEL_EXPORT_PATH

logger = logging.getLogger(__name__)


def prepare_model():
    start = datetime.now()
    learn()
    end = datetime.now()
    logger.info(f'Whole learning process took {end - start}')
    return True


def learn():
    logger.info('Load the data')
    x, y, names, groups = load_data()
    # Use compact dtypes for faster GPU transfer and lower memory
    x = x.astype(np.float32)
    y = y.astype(np.int32).reshape((len(y),))

    # Group-aware split to avoid subject leakage
    gss = GroupShuffleSplit(test_size=0.4, random_state=42)
    train_idx, test_idx = next(gss.split(x, y, groups))
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    y_train = y_train.reshape((len(y_train),))

    logger.info('Original train data:}')
    log_data_info(y_train)

    # Impute missing values efficiently and consistently (fit on train, apply to all)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

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
            params = _search_best_hyper_parameters(x_train, y_train, groups_train)

        # Ensure GPU-friendly params and class imbalance handling
        params = _ensure_gpu_params(params)
        spw = _compute_scale_pos_weight(y_train)
        if spw is not None:
            params["scale_pos_weight"] = spw
        # Ensure evaluation metrics are set on the model (sklearn API no longer accepts eval_metric in fit)
        params.setdefault("eval_metric", ["error", "logloss", "auc"])

        logger.info('Cross-validation of params')
        y_train = y_train.ravel()
        # Tune best number of boosting rounds using xgb.cv with group-aware folds
        dtrain = xgb.DMatrix(x_train, label=y_train)
        folds = list(GroupKFold(n_splits=10).split(x_train, y_train, groups_train))
        cv_params = dict(params)
        # Remove sklearn-only params
        cv_params.pop('n_estimators', None)
        # Ensure eval metrics are set
        cv_params.setdefault('eval_metric', ['auc', 'logloss', 'error'])
        logger.info('Running xgb.cv to find best iteration (n_estimators)')
        cv_results = xgb.cv(
            params=cv_params,
            dtrain=dtrain,
            num_boost_round=2000,
            folds=folds,
            early_stopping_rounds=50,
            metrics=['auc', 'logloss', 'error'],
            verbose_eval=False,
            seed=42,
        )
        best_n_estimators = cv_results.shape[0]
        logger.info(f'Best n_estimators from xgb.cv: {best_n_estimators}')
        params['n_estimators'] = int(best_n_estimators)

        model = xgb.sklearn.XGBClassifier(**params)
        cv_results = evaluate_cross_validation(
            model=model,
            x_train=x_train,
            y_train=y_train,
            groups=groups_train,
            save_path=CV_RESULTS_PATH,
        )
        logger.info(results_to_print_cv(cv_results))

        plot_cross_validation(cv_results, 'Model binary:logistic')

        train_model_test_train_data(model, x_test, x_train, y_test, y_train)
        save_obj(model, TRAINED_MODEL_PATH)
        model.save_model(TRAINED_MODEL_EXPORT_PATH)

    # Plot the feature importances
    plot_fi(model, names, scale_name, sort=True, save_dir=ML_DIR)
    plot_logloss_and_error(model, model_name)

    # Use Booster + DMatrix for predictions to avoid device-mismatch warnings
    booster = model.get_booster()
    predict = (booster.predict(xgb.DMatrix(x_test)) >= 0.5).astype(int)
    logger.info('After training results on test data: ')
    logger.info(results_to_print(y_test, predict))
    logger.info('Confusion matrix: ')
    logger.info(confusion_matrix(y_test, predict))

    # Transform the whole dataset for global evaluation and SHAP
    x_full = imputer.transform(x)
    predict = (booster.predict(xgb.DMatrix(x_full)) >= 0.5).astype(int)
    logger.info('After training results on whole dataset: ')
    logger.info(results_to_print(y, predict))
    logger.info('Confusion matrix: ')
    logger.info(confusion_matrix(y, predict))

    # SHAP on a sample to keep runtime reasonable
    sample_size = min(20000, x_full.shape[0])
    if sample_size < x_full.shape[0]:
        idx = np.random.RandomState(42).choice(x_full.shape[0], size=sample_size, replace=False)
        x_shap = x_full[idx]
    else:
        x_shap = x_full

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_shap)

    shap_summary_plot(names, shap_values, x_shap, ML_DIR)
    shap_beeswarm_plot(explainer, names, x_shap, ML_DIR)

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
    model.fit(
        x_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )


def load_data():
    # Prefer Parquet cache if available
    if os.path.exists(DATASET_PARQUET_PATH):
        logger.info('Load cached dataset (parquet)')
        try:
            super_df = pd.read_parquet(DATASET_PARQUET_PATH)
        except Exception as e:
            logger.info(f'Parquet read failed ({e}); falling back to Excel cache if present')
            if os.path.exists(DATASET_PATH):
                super_df = pd.read_excel(DATASET_PATH, index_col=0)
            else:
                super_df = None
    elif os.path.exists(DATASET_PATH):
        logger.info('Load cached dataset (excel)')
        super_df = pd.read_excel(DATASET_PATH, index_col=0)
    else:
        super_df = None

    if super_df is None:
        data = CsvData.objects.all()
        start = datetime.now()
        frames = []
        for d in data:
            # Include both standard training and DREAMT datasets
            if d.training_data and os.path.exists(d.x_data_path):
                df = pd.read_excel(d.x_data_path, index_col=0)
                # Only use files that contain the required label column
                if scale_name not in df.columns:
                    logger.info(f'Skipping {d.x_data_path}: missing label column {scale_name}')
                    continue
                # Attach group (subject) to avoid leakage
                df["GROUP"] = d.subject.code
                frames.append(df)

        if not frames:
            raise RuntimeError('No training dataframes with labels found to build dataset')

        super_df = pd.concat(frames)
        end = datetime.now()
        logger.info(f'Dataset merged in {end - start}, info: {df_into_to_sting(super_df)}')
        # Cache as Parquet primarily (fast IO, preserves dtypes)
        try:
            super_df.to_parquet(DATASET_PARQUET_PATH, index=True)
            logger.info(f'Cached dataset to Parquet at {DATASET_PARQUET_PATH}')
        except Exception as e:
            logger.info(f'Parquet write failed ({e}); caching to Excel at {DATASET_PATH}')
            super_df.to_excel(DATASET_PATH)

    # If cached dataset exists but lacks GROUP, rebuild to include it
    if "GROUP" not in super_df.columns:
        data = CsvData.objects.all()
        start = datetime.now()
        frames = []
        for d in data:
            # Include both standard training and DREAMT datasets
            if d.training_data and os.path.exists(d.x_data_path):
                df = pd.read_excel(d.x_data_path, index_col=0)
                if scale_name not in df.columns:
                    logger.info(f'Skipping {d.x_data_path}: missing label column {scale_name}')
                    continue
                df["GROUP"] = d.subject.code
                frames.append(df)
        if frames:
            super_df = pd.concat(frames)
            logger.info(f'Rebuilt dataset with GROUP in {datetime.now() - start}')
            try:
                super_df.to_parquet(DATASET_PARQUET_PATH, index=True)
                logger.info(f'Cached dataset to Parquet at {DATASET_PARQUET_PATH}')
            except Exception as e:
                logger.info(f'Parquet write failed ({e}); caching to Excel at {DATASET_PATH}')
                super_df.to_excel(DATASET_PATH)

    x = super_df[[c for c in super_df.columns if c not in (scale_name, "GROUP")]].values
    y = super_df[scale_name].values
    names = [c for c in super_df.columns if c not in (scale_name, "GROUP")]
    groups = super_df["GROUP"].values
    return x, y, names, groups


def _search_best_hyper_parameters(x, y, groups=None):
    start = datetime.now()
    # Create the classifier
    # Ensure GPU params for search
    base_params = _ensure_gpu_params(dict(model_params))
    spw = _compute_scale_pos_weight(y)
    if spw is not None:
        base_params["scale_pos_weight"] = spw
    model = xgb.sklearn.XGBClassifier(**base_params)

    # Get the cross-validation indices
    if groups is not None:
        cv = GroupKFold(n_splits=10)
    else:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Employ the hyper-parameter tuning
    y = y.astype(bool)
    random_search = RandomizedSearchCV(model, cv=cv, **search_settings)
    if groups is not None:
        random_search.fit(x, y, groups=groups)
    else:
        random_search.fit(x, y)

    logger.info(f'Estimator: score = {random_search.best_score_:.4f} | model = {random_search.best_estimator_}')
    params = random_search.best_params_
    save_obj(params, HYPER_PARAMS_PATH)
    save_obj(random_search.best_estimator_, BEST_ESTIMATOR_PATH)
    end = datetime.now()
    logger.info(f'Best hyper parameters found and cached in {end - start}')
    return params


def _ensure_gpu_params(params: dict) -> dict:
    # Force GPU-backed training where available
    params = dict(params)
    # XGBoost 2.x style GPU config
    params["tree_method"] = "hist"
    params["device"] = "cuda"
    # Remove deprecated/unused keys if present
    params.pop("gpu_id", None)
    params.pop("predictor", None)
    return params


def _compute_scale_pos_weight(y: np.ndarray):
    # Compute neg/pos ratio for class imbalance handling
    y = np.ravel(y)
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0:
        return None
    return float(neg) / float(pos)


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


def evaluate_cross_validation(model, x_train, y_train, groups, save_path):
    start = datetime.now()
    # Prepare the cross-validation scheme (group-aware)
    kfolds = GroupKFold(n_splits=10)

    # Cross-validate the classifier
    cv_results = cross_validate(model, x_train, y_train, scoring=scoring, cv=kfolds, groups=groups)
    if save_path:
        save_obj(cv_results, save_path)
    end = datetime.now()
    logger.info(f'Cross validation performed in {end - start}')
    return cv_results
