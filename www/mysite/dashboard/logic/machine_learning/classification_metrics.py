from sklearn.metrics import recall_score, confusion_matrix, make_scorer, accuracy_score, matthews_corrcoef


def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


# Prepare the scoring
scoring = {
    "acc": make_scorer(accuracy_score, greater_is_better=True),
    "sen": make_scorer(sensitivity_score, greater_is_better=True),
    "spe": make_scorer(specificity_score, greater_is_better=True),
    "mcc": make_scorer(matthews_corrcoef, greater_is_better=True)
}
