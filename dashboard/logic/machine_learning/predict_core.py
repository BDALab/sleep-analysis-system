from dashboard.logic import cache
from dashboard.logic.machine_learning.settings import prediction_name, scale_name
from mysite.settings import TRAINED_MODEL_PATH


def predict_core(csv_data, df):
    predictions = _predict(df)
    df[prediction_name] = predictions
    cache.save_obj(df, csv_data.cached_prediction_path)
    df.to_excel(csv_data.excel_prediction_path)


def _predict(df):
    x = df[[c for c in df.columns if c != scale_name]].values
    model = cache.load_obj(TRAINED_MODEL_PATH)
    predictions = model.predict(x)
    return predictions
