from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-9)
    if not np.any(valid_mask):
        return None
    return _finite_or_none(np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100)


def calculate_directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    if len(y_true) < 2:
        return None

    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)

    non_zero_actual_movement_indices = np.where(y_true_diff != 0)
    y_true_diff_filtered = y_true_diff[non_zero_actual_movement_indices]
    y_pred_diff_filtered = y_pred_diff[non_zero_actual_movement_indices]

    if len(y_true_diff_filtered) == 0:
        return None

    correct_direction_count = np.sum(np.sign(y_true_diff_filtered) == np.sign(y_pred_diff_filtered))
    return _finite_or_none((correct_direction_count / len(y_true_diff_filtered)) * 100)


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {
            "MSE": None,
            "MAE": None,
            "RMSE": None,
            "MAPE (%)": None,
            "R²": None,
            "R2": None,
            "Directional Accuracy (%)": None,
        }

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = calculate_mape(y_true, y_pred)
    r2 = None if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0) else r2_score(y_true, y_pred)
    da = calculate_directional_accuracy(y_true, y_pred)

    r2_value = _finite_or_none(r2)
    return {
        "MSE": _finite_or_none(mse),
        "MAE": _finite_or_none(mae),
        "RMSE": _finite_or_none(rmse),
        "MAPE (%)": _finite_or_none(mape),
        "R²": r2_value,
        "R2": r2_value,
        "Directional Accuracy (%)": _finite_or_none(da),
    }
