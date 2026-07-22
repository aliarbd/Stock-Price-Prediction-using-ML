from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100


def calculate_directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)

    non_zero_actual_movement_indices = np.where(y_true_diff != 0)
    y_true_diff_filtered = y_true_diff[non_zero_actual_movement_indices]
    y_pred_diff_filtered = y_pred_diff[non_zero_actual_movement_indices]

    if len(y_true_diff_filtered) == 0:
        return np.nan

    correct_direction_count = np.sum(np.sign(y_true_diff_filtered) == np.sign(y_pred_diff_filtered))
    return (correct_direction_count / len(y_true_diff_filtered)) * 100


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    da = calculate_directional_accuracy(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE (%)": float(mape),
        "R²": float(r2),
        "Directional Accuracy (%)": float(da),
    }
