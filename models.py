"""@author: Léa chaccour & Ben Hasenson"""

from enum import Enum

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin


class TuningMode(Enum):
    """Enum for tuning mode, either TEST or HPC.
    TEST is for quick runs with limited hyperparameter space.
    HPC is for full hyperparameter tuning on a high-performance cluster.
    """

    TEST = "test"
    HPC = "hpc"


def get_model_dict():
    return {
        "LR": LinearRegression(),
        "RF": RandomForestRegressor(random_state=42),
        "GB": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0),
        "MLP": MLPRegressor(random_state=42, max_iter=2000),

    }

def apply_model_specific_scaling(model_name, X_train, X_test, y_train):
    """
    Prepares training and test data for a specific model, applying scaling if required. 
    Currently, scaling is applied only for MLPRegressor since it is sensitive to feature and target distributions.
    """
    if model_name == "MLPRegressor":
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        return X_train_scaled, X_test_scaled, y_train_scaled, scaler_y
    else:
        return X_train, X_test, y_train, None

#hybrid models 
class HybridRFMLP(BaseEstimator, RegressorMixin):
    """
    Hybrid model that first fits a Random Forest and then trains an MLP
    on the concatenation of (X, RF predictions). The MLP refines or corrects
    the RF outputs to capture complex residual patterns.
    """

    def __init__(self, rf_params=None, mlp_params=None, random_state=42):
        self.rf_params = rf_params if rf_params is not None else {}
        self.mlp_params = mlp_params if mlp_params is not None else {}
        self.random_state = random_state
        self.rf_model = RandomForestRegressor(random_state=random_state, **self.rf_params)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.mlp_model = MLPRegressor(random_state=random_state, max_iter=2000, **self.mlp_params)

    def fit(self, X, y):
        # Fitting the Random Forest
        self.rf_model.fit(X, y)
        rf_pred = self.rf_model.predict(X).reshape(-1, 1)

        #Combining X and RF predictions as input to the MLP
        X_hybrid = np.hstack((X, rf_pred))

        # Scaling features and target for the MLP
        X_scaled = self.scaler_X.fit_transform(X_hybrid)
        y_array = np.array(y).reshape(-1, 1)
        y_scaled = self.scaler_y.fit_transform(y_array).ravel()


        # Fitting the MLP on the hybrid features
        self.mlp_model.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        rf_pred = self.rf_model.predict(X).reshape(-1, 1)
        X_hybrid = np.hstack((X, rf_pred))
        X_scaled = self.scaler_X.transform(X_hybrid)
        y_pred_scaled = self.mlp_model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

class HybridXGBMLP(BaseEstimator, RegressorMixin):
    """
    Hybrid model combining XGBoost and MLP.
    Trains XGBoost first, then uses its predictions as an extra feature for the MLP.
    """

    def __init__(self, xgb_params=None, mlp_params=None, random_state=42):
        self.xgb_params = xgb_params if xgb_params is not None else {}
        self.mlp_params = mlp_params if mlp_params is not None else {}
        self.random_state = random_state
        self.xgb_model = XGBRegressor(random_state=random_state, verbosity=0, **self.xgb_params)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.mlp_model = MLPRegressor(random_state=random_state, max_iter=2000, **self.mlp_params)

    def fit(self, X, y):
        y = np.array(y).ravel()
        self.xgb_model.fit(X, y)
        xgb_pred = self.xgb_model.predict(X).reshape(-1, 1)
        X_hybrid = np.hstack((X, xgb_pred))
        X_scaled = self.scaler_X.fit_transform(X_hybrid)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self.mlp_model.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        xgb_pred = self.xgb_model.predict(X).reshape(-1, 1)
        X_hybrid = np.hstack((X, xgb_pred))
        X_scaled = self.scaler_X.transform(X_hybrid)
        y_pred_scaled = self.mlp_model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

class HybridGBMLP(BaseEstimator, RegressorMixin):
    """
    Hybrid model combining Gradient Boosting and MLP.
    Trains GradientBoosting first, then uses its predictions as an extra feature for the MLP.
    """

    def __init__(self, gb_params=None, mlp_params=None, random_state=42):
        self.gb_params = gb_params if gb_params is not None else {}
        self.mlp_params = mlp_params if mlp_params is not None else {}
        self.random_state = random_state

        # Define base and meta learners
        self.gb_model = GradientBoostingRegressor(random_state=random_state, **self.gb_params)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.mlp_model = MLPRegressor(random_state=random_state, max_iter=2000, **self.mlp_params)

    def fit(self, X, y):
        y = np.array(y).ravel()

        #  Fitting Gradient Boosting on raw data
        self.gb_model.fit(X, y)
        gb_pred = self.gb_model.predict(X).reshape(-1, 1)

        # Combining original features + GB predictions
        X_hybrid = np.hstack((X, gb_pred))

        #  Scaling for the MLP
        X_scaled = self.scaler_X.fit_transform(X_hybrid)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        #  Training the MLP on scaled hybrid data
        self.mlp_model.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        #  Getting Gradient Boosting predictions
        gb_pred = self.gb_model.predict(X).reshape(-1, 1)

        #  Concatenating and scale for MLP
        X_hybrid = np.hstack((X, gb_pred))
        X_scaled = self.scaler_X.transform(X_hybrid)

        #  Predicting using MLP and inverse scale
        y_pred_scaled = self.mlp_model.predict(X_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
def get_param_grid(model_name: str, tuning_mode: TuningMode) -> dict:
    """Get the parameter grid for hyperparameter tuning based on the model name and tuning mode."""

    if model_name == "GradientBoosting":
        if tuning_mode == TuningMode.TEST:
            return {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05],
                "max_depth": [3, 4],
                "min_samples_split": [2, 4],
                "subsample": [0.8, 1.0],
            }
        else:  # HPC mode
            return {
                "n_estimators": list(range(100, 1501, 100)),
                "learning_rate": np.round(np.linspace(0.005, 0.2, 20), 4).tolist(),
                "max_depth": list(range(3, 15)),
                "min_samples_split": list(range(2, 11)),
                "subsample": np.round(np.arange(0.6, 1.05, 0.1), 2).tolist(),
            }
    elif model_name == "XGBoost":
        if tuning_mode == TuningMode.TEST:
            return {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05],
                "max_depth": [3, 4],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }
        else:  # HPC mode
            return {
                "n_estimators": list(range(100, 1001, 100)),
                "learning_rate": np.round(np.linspace(0.01, 0.3, 10), 4).tolist(),
                "max_depth": list(range(3, 11)),
                "subsample": np.round(np.arange(0.6, 1.05, 0.1), 2).tolist(),
                "colsample_bytree": np.round(np.arange(0.6, 1.05, 0.1), 2).tolist(),
            }
    elif model_name == "RandomForest":
        if tuning_mode == TuningMode.TEST:
            return {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "min_samples_split": [2, 4],
            }
        else:  # HPC mode
            return {
                "n_estimators": list(range(100, 1001, 100)),
                "max_depth": list(range(3, 11)),
                "min_samples_split": list(range(2, 11)),
                "max_features": ["log2", "sqrt"],
            }
    elif model_name == "ANN":
        if tuning_mode == TuningMode.TEST:
            return {
                "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "activation": ["relu", "tanh"],
                "solver": ["adam"],
                "alpha": [0.0001, 0.001],
                "learning_rate_init": [0.001, 0.01],
            }
   