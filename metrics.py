"""@author: Ben Hasenson & Léa Chaccour"""

import pandas as pd
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


def evaluate_model(y_true: pd.Series, y_pred: pd.Series):
    """Evaluates the model predictions and returns a dictionary of metrics."""
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def get_scorers():
    """Returns a dictionary of scoring functions for model evaluation."""
    return {
        "R2": make_scorer(r2_score),
        "RMSE": make_scorer(
            lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)
        ),
        "MAE": make_scorer(mean_absolute_error),
        "MAPE": make_scorer(mean_absolute_percentage_error),
    }
