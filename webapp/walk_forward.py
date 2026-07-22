from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WALK_FORWARD_MODELS
from .metrics import compute_regression_metrics


def run_walk_forward_validation(
    data: pd.DataFrame,
    scaled_data,
    scaler,
    time_step: int,
    walk_forward_days: int,
    models: dict,
):
    supported_models = [name for name in WALK_FORWARD_MODELS if name in models]
    if not supported_models:
        return pd.DataFrame(), pd.DataFrame()

    training_data_len = len(scaled_data) - walk_forward_days
    initial_input_sequence = scaled_data[training_data_len - time_step : training_data_len].copy()

    walk_forward_predictions = {name: [] for name in supported_models}

    for name in supported_models:
        current_input = initial_input_sequence.copy()
        for _ in range(walk_forward_days):
            if name in {"LSTM", "GRU"}:
                model = models[name]
                inp = current_input.reshape(1, time_step, 1)
                pred = model.predict(inp, verbose=0)[0, 0]
            else:
                model = models[name]
                inp = current_input.reshape(1, time_step)
                pred = model.predict(inp)[0]
                if isinstance(pred, np.ndarray):
                    pred = pred.item()

            walk_forward_predictions[name].append(pred)
            current_input = np.append(current_input[1:], np.array([[pred]]), axis=0)

    actual_prices = data["close"].values[-walk_forward_days:]
    walk_forward_df = pd.DataFrame(index=data.index[-walk_forward_days:])
    walk_forward_df["Actual Price"] = actual_prices

    for name in supported_models:
        unscaled_preds = scaler.inverse_transform(np.array(walk_forward_predictions[name]).reshape(-1, 1))
        walk_forward_df[f"{name} Prediction"] = unscaled_preds.flatten()

    wf_metrics = []
    actual = walk_forward_df["Actual Price"]
    for col in walk_forward_df.columns:
        if "Prediction" in col:
            model_name = col.replace(" Prediction", "")
            preds = walk_forward_df[col]
            metrics = compute_regression_metrics(actual.values, preds.values)
            wf_metrics.append(
                {
                    "Model": model_name,
                    "MSE": metrics["MSE"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MAPE (%)": metrics["MAPE (%)"],
                    "R²": metrics["R²"],
                }
            )

    wf_metrics_df = pd.DataFrame(wf_metrics).sort_values("RMSE") if wf_metrics else pd.DataFrame()
    return walk_forward_df, wf_metrics_df
