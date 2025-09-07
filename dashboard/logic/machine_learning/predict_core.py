import xgboost as xgb

from dashboard.logic import cache
from dashboard.logic.machine_learning.settings import prediction_name, scale_name
from mysite.settings import TRAINED_MODEL_EXPORT_PATH


def predict_core(csv_data, df):
    predictions = _predict(df)
    df[prediction_name] = predictions
    cache.save_obj(df, csv_data.cached_prediction_path)
    df.to_excel(csv_data.excel_prediction_path)


def _predict(df):
    x = df[[c for c in df.columns if c != scale_name]].values
    booster = xgb.Booster()
    model_path = TRAINED_MODEL_EXPORT_PATH
    print(f'Loading model from {model_path}')
    booster.load_model(model_path)
    dmat = xgb.DMatrix(x)
    preds = booster.predict(dmat)
    return preds
