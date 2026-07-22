from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import lightgbm as lgb
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from tensorflow.keras.layers import GRU, LSTM, Dense
from tensorflow.keras.models import Sequential


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_rnn_model(model_type: str = "LSTM", input_shape: Tuple[int, int] = (60, 1)):
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
    else:
        model.add(GRU(50, return_sequences=True, input_shape=input_shape))
        model.add(GRU(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def get_tree_models(random_seed: int | None = None) -> Dict[str, object]:
    seed = 42 if random_seed is None else int(random_seed)
    return {
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed),
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel="rbf", C=1e3, gamma=0.1),
        "LightGBM": lgb.LGBMRegressor(objective="regression", n_estimators=100, random_state=seed, verbose=-1),
    }


@dataclass(slots=True)
class TrainedModel:
    name: str
    model: object
    train_seconds: float
    predict_seconds: float
    predictions: np.ndarray
