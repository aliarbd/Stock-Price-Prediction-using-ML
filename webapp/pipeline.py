from __future__ import annotations

import json
import time
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .config import MODEL_ORDER, PipelineConfig, RNN_MODELS, STATISTICAL_MODELS, TREE_MODELS
from .data import create_sequences, load_data, preprocess_data
from .future import build_future_forecast
from .metrics import compute_regression_metrics
from .models import get_rnn_model, get_tree_models, set_random_seed
from .strategy import build_strategy_frame
from .visuals import (
    build_equity_curve_figure,
    build_future_forecast_figure,
    build_prediction_comparison_figure,
    build_strategy_signals_figure,
    build_walk_forward_figure,
    figure_to_json,
)
from .walk_forward import run_walk_forward_validation


ProgressCallback = Callable[[str, float], None]


def _emit(progress: ProgressCallback | None, stage: str, percent: float) -> None:
    if progress is not None:
        progress(stage, percent)


def _safe_records(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def run_pipeline(config: PipelineConfig, progress: ProgressCallback | None = None) -> Dict:
    set_random_seed(config.random_seed)

    selected_models = config.normalized_models()
    if not selected_models:
        raise ValueError("At least one model must be selected.")

    _emit(progress, "Loading historical data", 5)
    df = load_data(config.stock_symbol, config.exchange, config.dataset_length)
    data, scaler, scaled_data = preprocess_data(df)

    _emit(progress, "Preparing sequences", 10)
    total_points = len(scaled_data)
    train_size = int(total_points * config.train_split)
    if train_size <= config.time_step or (total_points - train_size) <= config.time_step:
        raise ValueError(
            "Dataset length and train split must leave enough rows on both sides of the split "
            "for the selected time step."
        )
    train_data = scaled_data[0:train_size, :]
    x_train, y_train = create_sequences(train_data, config.time_step)
    x_train_rnn = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train_xgb = x_train.reshape(x_train.shape[0], x_train.shape[1])

    test_data_seq = scaled_data[train_size - config.time_step :, :]
    x_test_split, y_test_split = create_sequences(test_data_seq, config.time_step)
    x_test_rnn = np.reshape(x_test_split, (x_test_split.shape[0], x_test_split.shape[1], 1))
    x_test_tree = x_test_split
    y_actual = scaler.inverse_transform(y_test_split.reshape(-1, 1))

    tree_models = get_tree_models(config.random_seed)
    model_objects: Dict[str, object] = {}
    model_predictions_split: Dict[str, np.ndarray] = {}
    performance_rows = []

    _emit(progress, "Training models", 20)
    for model_name in selected_models:
        train_seconds = 0.0
        predict_seconds = 0.0
        predictions = None
        model = None

        if model_name in RNN_MODELS:
            model = get_rnn_model(model_type=model_name, input_shape=(config.time_step, 1))
            start = time.perf_counter()
            model.fit(x_train_rnn, y_train, batch_size=32, epochs=1, verbose=0)
            train_seconds = time.perf_counter() - start

            start = time.perf_counter()
            predictions_scaled = model.predict(x_test_rnn, verbose=0)
            predictions = scaler.inverse_transform(predictions_scaled)
            predict_seconds = time.perf_counter() - start
        elif model_name in TREE_MODELS:
            model = tree_models[model_name]
            start = time.perf_counter()
            model.fit(x_train_xgb, y_train)
            train_seconds = time.perf_counter() - start

            start = time.perf_counter()
            raw_predictions = model.predict(x_test_tree)
            if model_name in {"SVR"}:
                predictions = scaler.inverse_transform(raw_predictions.reshape(-1, 1))
            elif model_name in {"LightGBM"}:
                predictions = scaler.inverse_transform(raw_predictions.reshape(-1, 1))
            else:
                predictions = scaler.inverse_transform(raw_predictions.reshape(-1, 1))
            predict_seconds = time.perf_counter() - start
        elif model_name in STATISTICAL_MODELS:
            train_series = data["close"].iloc[:train_size]
            if model_name == "ARIMA":
                from statsmodels.tsa.arima.model import ARIMA

                start = time.perf_counter()
                model = ARIMA(train_series, order=(5, 2, 1)).fit()
                train_seconds = time.perf_counter() - start
                start = time.perf_counter()
                forecast = model.predict(start=len(train_series), end=len(train_series) + len(y_actual) - 1, dynamic=False)
                predictions = forecast.values.reshape(-1, 1)
                predict_seconds = time.perf_counter() - start
            else:
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                start = time.perf_counter()
                model = SARIMAX(train_series, order=(1, 2, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
                train_seconds = time.perf_counter() - start
                start = time.perf_counter()
                forecast = model.predict(start=len(train_series), end=len(train_series) + len(y_actual) - 1, dynamic=False)
                predictions = forecast.values.reshape(-1, 1)
                predict_seconds = time.perf_counter() - start

        if model is None or predictions is None:
            continue

        model_objects[model_name] = model
        model_predictions_split[model_name] = predictions
        metrics = compute_regression_metrics(y_actual, predictions)
        performance_rows.append(
            {
                "Model": model_name,
                **metrics,
                "Training Time (s)": round(float(train_seconds), 4),
                "Prediction Time (s)": round(float(predict_seconds), 4),
            }
        )

    if not performance_rows:
        raise ValueError("No model predictions were generated.")

    summary_stats_split = pd.DataFrame(performance_rows).sort_values("RMSE").reset_index(drop=True)
    best_model_name = summary_stats_split.iloc[0]["Model"]
    best_model_metrics = {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in summary_stats_split.iloc[0].to_dict().items()
    }

    plot_df_split = pd.DataFrame(y_actual, columns=["Actual Price"], index=data.index[train_size:])
    for model_name, predictions_array in model_predictions_split.items():
        if len(predictions_array) == len(plot_df_split):
            plot_df_split[f"{model_name} Prediction"] = predictions_array.flatten()

    _emit(progress, "Building comparison visuals", 45)
    comparison_fig = build_prediction_comparison_figure(plot_df_split, selected_models, config.stock_symbol, train_size)

    _emit(progress, "Running strategy analysis", 60)
    auto_strat_df, individual_trades_df, strategy_metrics = build_strategy_frame(
        data=data,
        plot_df_split=plot_df_split,
        best_model_name=best_model_name,
        ema_period=config.ema_period,
    )
    strategy_fig = build_strategy_signals_figure(auto_strat_df, best_model_name, config.ema_period)
    equity_fig = build_equity_curve_figure(auto_strat_df, config.stock_symbol)

    _emit(progress, "Running walk-forward validation", 75)
    wf_models = {name: model_objects[name] for name in model_objects if name in set(["LSTM", "GRU", "XGBoost", "Random Forest", "Linear Regression", "SVR", "LightGBM"])}
    walk_forward_df, wf_metrics_df = run_walk_forward_validation(
        data=data,
        scaled_data=scaled_data,
        scaler=scaler,
        time_step=config.time_step,
        walk_forward_days=config.walk_forward_days,
        models=wf_models,
    )
    walk_forward_fig = build_walk_forward_figure(walk_forward_df) if not walk_forward_df.empty else None

    _emit(progress, "Generating future forecast", 88)
    future_df = build_future_forecast(
        data=data,
        scaled_data=scaled_data,
        scaler=scaler,
        time_step=config.time_step,
        future_days=config.forecast_days,
        models=model_objects,
    )
    future_fig = build_future_forecast_figure(future_df, config.stock_symbol)

    dataset_summary = {
        "stock": config.stock_symbol,
        "exchange": config.exchange,
        "total_rows": int(len(df)),
        "start_date": df.index.min().strftime("%Y-%m-%d"),
        "end_date": df.index.max().strftime("%Y-%m-%d"),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test_split)),
    }

    result = {
        "dataset_summary": dataset_summary,
        "selected_models": selected_models,
        "comparison": {
            "figure": figure_to_json(comparison_fig),
            "table": _safe_records(summary_stats_split),
        },
        "best_model": {
            "name": best_model_name,
            "metrics": best_model_metrics,
        },
        "strategy": {
            "figure": figure_to_json(strategy_fig),
            "equity_figure": figure_to_json(equity_fig),
            "metrics": strategy_metrics,
            "trades": _safe_records(individual_trades_df),
        },
        "walk_forward": {
            "figure": figure_to_json(walk_forward_fig) if walk_forward_fig is not None else None,
            "table": _safe_records(wf_metrics_df),
            "series": _safe_records(walk_forward_df.reset_index().rename(columns={"index": "Date"}))
            if not walk_forward_df.empty
            else [],
        },
        "future_prediction": {
            "figure": figure_to_json(future_fig),
            "table": _safe_records(future_df.reset_index().rename(columns={"index": "Date"})),
        },
    }

    _emit(progress, "Complete", 100)
    return result
