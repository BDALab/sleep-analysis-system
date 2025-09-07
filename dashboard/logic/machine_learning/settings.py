# ---------------------
# XGBoost
# ---------------------

# Set the random generator seed
seed = 42

# ------------------------------
# Define the classifier settings
# ------------------------------
model_params = {
    # General parameters
    "booster": "gbtree",
    "verbosity": 1,
    "n_jobs": -1,

    # Learning task parameters
    "objective": "binary:logistic",  # https://xgboost.readthedocs.io/en/latest/parameter.html
    "eval_metric": ["rmse", "error", "logloss", "auc"],
    "seed": seed,
    # Tree Booster parameters
    "n_estimators": 1000,
    "learning_rate": 0.20,
    "gamma": 1.0,
    "max_depth": 6,
    "subsample": 1.0,
    "colsample_bylevel": 1,
    "colsample_bytree": 1.0,
    "min_child_weight": 5,

    # XGBoost 2.x GPU settings
    # Use histogram algorithm on GPU by setting device to CUDA.
    # (tree_method "hist" is recommended with device="cuda").
    "tree_method": "hist",
    "device": "cuda",
    # Reasonable defaults for speed/VRAM balance
    "max_bin": 512,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# ------------------------------------------
# Define the hyper-parameter tuning settings
# ------------------------------------------
param_grid = {
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
    "gamma": [0, 0.10, 0.15, 0.25, 0.5],
    "max_depth": [6, 8, 10, 12, 15],
    "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "scale_pos_weight": [1, 3, 5, 6, 7, 9]  # https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
}

search_settings = {
    "param_distributions": param_grid,
    "scoring": "f1_micro",  # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    # Avoid concurrent GPU fits that contend for VRAM
    "n_jobs": 1,
    "n_iter": 100,
    "verbose": 1
}

# Set the scale name
scale_name = "SLEEP"

prediction_name = "SLEEP_PREDICTION"

hilev_prediction = 'PREDICTION'

model_name = "Model Z"
