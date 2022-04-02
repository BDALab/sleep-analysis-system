from sklearn.metrics import recall_score, confusion_matrix, make_scorer, accuracy_score, matthews_corrcoef, f1_score


def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


# Prepare the scoring
scoring = {
    "acc": make_scorer(accuracy_score),
    "sen": make_scorer(sensitivity_score),
    "spe": make_scorer(specificity_score),
    "mcc": make_scorer(matthews_corrcoef),
    "f1": make_scorer(f1_score)
}
