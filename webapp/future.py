from __future__ import annotations

import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import RNN_MODELS, TREE_MODELS


def build_future_forecast(
    data: pd.DataFrame,
    scaled_data,
    scaler,
    time_step: int,
    future_days: int,
    models: dict,
):
    last_sequence_scaled = scaled_data[-time_step:].copy()
    model_names = [name for name in ["LSTM", "GRU", "XGBoost", "Random Forest", "Linear Regression", "SVR", "LightGBM"] if name in models]
    future_preds_dict = {name: [] for name in model_names}
    curr_inputs = {name: last_sequence_scaled.copy() for name in model_names}

    for _ in range(future_days):
        for name in ["LSTM", "GRU"]:
            if name not in models:
                continue
            model = models[name]
            p = model.predict(curr_inputs[name].reshape(1, time_step, 1), verbose=0)[0, 0]
            future_preds_dict[name].append(p)
            curr_inputs[name] = np.append(curr_inputs[name][1:], [[p]], axis=0)
        for name in ["XGBoost", "Random Forest", "Linear Regression", "SVR", "LightGBM"]:
            if name not in models:
                continue
            model = models[name]
            p = model.predict(curr_inputs[name].reshape(1, time_step))[0]
            future_preds_dict[name].append(p)
            curr_inputs[name] = np.append(curr_inputs[name][1:], [[p]], axis=0)

    full_series = data["close"]
    future_stats = {}
    if "ARIMA" in models:
        arima_f_model = ARIMA(full_series, order=(5, 1, 0)).fit()
        future_stats["ARIMA"] = arima_f_model.forecast(steps=future_days).values
    if "SARIMA" in models:
        sarima_f_model = SARIMAX(full_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
        future_stats["SARIMA"] = sarima_f_model.forecast(steps=future_days).values

    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=future_days)
    future_df = pd.DataFrame(index=future_dates)

    for name in model_names:
        future_df[f"{name} Future Predictions"] = scaler.inverse_transform(
            np.array(future_preds_dict[name]).reshape(-1, 1)
        ).flatten()

    for name, values in future_stats.items():
        future_df[f"{name} Future Predictions"] = values

    return future_df
