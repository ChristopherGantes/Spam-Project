# evaluation/cross_validation.py
from sklearn.model_selection import cross_val_score


def perform_cross_validation(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv)
    return cv_scores.mean()
