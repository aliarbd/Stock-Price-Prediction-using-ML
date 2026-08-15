"""
Live Prediction & Saved Model Monitoring Service.
Executes zero-retrain inference on fresh market data using exact saved preprocessing scalers.
Maintains strict dataset separation:
- Training/Test data (original experiment snapshot)
- Original Future Forecast
- Live/Post-Save Unseen Data (predictions generated AFTER live_monitoring_boundary)
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import load_data
from .metrics import compute_regression_metrics
from .model_registry import (
    get_model_predictions,
    get_model_record,
    record_prediction,
    update_actual_values,
)
from .model_store import load_saved_model
from .serialization import find_non_finite_values, sanitize_for_json
from .strategy import build_strategy_frame
from .visuals import build_strategy_signals_figure, figure_to_json

logger = logging.getLogger(__name__)


def _ensure_finite_prediction(value: float, model_id: str, model_type: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        logger.error("Invalid live prediction for model_id=%s model_type=%s: %r", model_id, model_type, value)
        raise ValueError("Model returned an invalid prediction value.")
    return value


def _predict_single_step(
    model_obj: Any,
    scaler_obj: Any,
    model_type: str,
    close_series: pd.Series,
    timestep: int,
) -> float:
    """Execute zero-retrain single-step prediction for the next observation."""
    raw_input_vals = close_series.tail(timestep).values.reshape(-1, 1)
    if len(raw_input_vals) < timestep:
        return float(close_series.iloc[-1])

    if model_type in ["LSTM", "GRU"]:
        scaled_input = scaler_obj.transform(raw_input_vals)
        rnn_input = scaled_input.reshape(1, timestep, 1)
        pred_scaled = model_obj.predict(rnn_input, verbose=0)[0, 0]
        return float(scaler_obj.inverse_transform(np.array([[pred_scaled]]))[0, 0])
    elif model_type in ["XGBoost", "Random Forest", "Linear Regression", "SVR", "LightGBM"]:
        scaled_input = scaler_obj.transform(raw_input_vals)
        tree_input = scaled_input.reshape(1, timestep)
        pred_raw = model_obj.predict(tree_input)
        pred_scaled = float(pred_raw[0] if isinstance(pred_raw, (list, np.ndarray)) else pred_raw)
        return float(scaler_obj.inverse_transform(np.array([[pred_scaled]]))[0, 0])
    elif model_type in ["ARIMA", "SARIMA"]:
        try:
            forecast = model_obj.forecast(steps=1)
            return float(forecast.values[0] if hasattr(forecast, "values") else forecast[0])
        except Exception:
            return float(close_series.iloc[-1])
    return float(close_series.iloc[-1])


def get_post_save_predictions_analysis(model_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate Post-Save Actual vs Predicted metrics and Plotly graph (Graph #1).
    STRICT RULE: Uses ONLY predictions generated after live_monitoring_boundary with evaluated actual prices.
    Never uses historical test-set predictions.
    """
    model_obj, scaler_obj, metadata = load_saved_model(model_id)
    live_boundary = metadata.get("train_end")

    # Match any pending predictions with fresh market actuals
    if df is not None and not df.empty:
        update_actual_values(model_id, df)

    history = get_model_predictions(model_id, limit=5000)

    # Filter to evaluated live predictions generated strictly after live_monitoring_boundary
    evaluated_live = []
    for h in history:
        p_date = str(h["prediction_date"])[:10]
        if h.get("actual_value") is not None and (not live_boundary or p_date > live_boundary):
            evaluated_live.append(h)

    evaluated_live.sort(key=lambda x: str(x["prediction_date"]))

    if not evaluated_live:
        return {
            "has_data": False,
            "message": "Waiting for new unseen market data.",
            "metrics": None,
            "figure": None,
            "post_save_samples": 0,
        }

    dates = [str(h["prediction_date"])[:10] for h in evaluated_live]
    preds = np.array([float(h["predicted_value"]) for h in evaluated_live])
    actuals = np.array([float(h["actual_value"]) for h in evaluated_live])

    metrics = compute_regression_metrics(actuals.reshape(-1, 1), preds.reshape(-1, 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actuals,
            mode="lines+markers",
            name="Actual Price",
            line=dict(color="#0F172A", width=2.5),
            marker=dict(size=6),
            hovertemplate="<b>Date:</b> %{x}<br><b>Actual:</b> $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=preds,
            mode="lines+markers",
            name="Predicted Price",
            line=dict(color="#4338CA", width=2, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="<b>Date:</b> %{x}<br><b>Predicted:</b> $%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Actual vs Predicted Price Since Model Was Saved ({metadata['symbol']})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return {
        "has_data": True,
        "post_save_samples": len(evaluated_live),
        "metrics": metrics,
        "figure": figure_to_json(fig),
        "series": [
            {
                "date": dates[i],
                "actual": float(actuals[i]),
                "predicted": float(preds[i]),
                "error": float(abs(actuals[i] - preds[i])),
            }
            for i in range(len(dates))
        ],
    }


def _evaluate_live_strategy(
    evaluated_live: List[Dict[str, Any]],
    df: pd.DataFrame,
    ema_period: int = 9,
    initial_investment: float = 100.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate trading strategy strictly over post-save evaluated predictions.
    Starts with fresh portfolio value (initial_investment = 100.0) and 0 historical trades.
    """
    if not evaluated_live or df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    df_close = df["close"].copy()
    df_close.index = pd.to_datetime(df_close.index).strftime("%Y-%m-%d")
    df_ema = df_close.ewm(span=ema_period, adjust=False).mean()

    records = []
    in_position = False
    trades = []
    entry_date = None
    entry_price = None

    for h in evaluated_live:
        target_date = str(h["prediction_date"])[:10]
        input_date = str(h["input_data_timestamp"])[:10]
        pred_val = float(h["predicted_value"])
        actual_val = float(h["actual_value"])

        prev_close = float(df_close.loc[input_date]) if input_date in df_close.index else actual_val
        ema_val = (
            float(df_ema.loc[target_date])
            if target_date in df_ema.index
            else (float(df_ema.loc[input_date]) if input_date in df_ema.index else actual_val)
        )

        pred_cond = pred_val > prev_close
        ema_cond = actual_val > ema_val
        signal_active = pred_cond and ema_cond

        if not in_position:
            if signal_active:
                signal_code = 1  # BUY
                in_position = True
                entry_date = target_date
                entry_price = actual_val
            else:
                signal_code = 0  # HOLD / CASH
        else:
            if not signal_active:
                signal_code = 0  # CLOSE
                in_position = False
                exit_date = target_date
                exit_price = actual_val
                pnl_pct = ((exit_price - entry_price) / (entry_price + 1e-8)) * 100
                trades.append(
                    {
                        "Entry Date": entry_date,
                        "Entry Price": entry_price,
                        "Exit Date": exit_date,
                        "Exit Price": exit_price,
                        "Profit/Loss (%)": pnl_pct,
                    }
                )
                entry_date = None
                entry_price = None
            else:
                signal_code = 1  # IN POSITION

        records.append(
            {
                "Date": target_date,
                "Actual": actual_val,
                "Predicted": pred_val,
                "EMA_Filter": ema_val,
                "Signal": signal_code,
            }
        )

    if in_position and entry_date is not None and len(records) > 0:
        last_rec = records[-1]
        exit_date = last_rec["Date"]
        exit_price = last_rec["Actual"]
        pnl_pct = ((exit_price - entry_price) / (entry_price + 1e-8)) * 100
        trades.append(
            {
                "Entry Date": entry_date,
                "Entry Price": entry_price,
                "Exit Date": exit_date,
                "Exit Price": exit_price,
                "Profit/Loss (%)": pnl_pct,
            }
        )

    strat_df = pd.DataFrame(records)
    trades_df = pd.DataFrame(trades)

    if strat_df.empty:
        return strat_df, trades_df, {}

    curr_portfolio = initial_investment
    portfolio_vals = []
    strat_returns = []

    trade_pnl_map = {t["Exit Date"]: t["Profit/Loss (%)"] for t in trades}
    for idx, row in strat_df.iterrows():
        d = row["Date"]
        if d in trade_pnl_map:
            pnl_ret = trade_pnl_map[d] / 100.0
            curr_portfolio *= 1.0 + pnl_ret
            strat_returns.append(pnl_ret)
        else:
            strat_returns.append(0.0)
        portfolio_vals.append(curr_portfolio)

    strat_df["Portfolio_Value"] = portfolio_vals
    strat_df["Cum_Growth"] = np.array(portfolio_vals) / initial_investment
    strat_df["Strategy_Return_Pct"] = (strat_df["Cum_Growth"] - 1.0) * 100.0

    total_return_pct = float(strat_df["Strategy_Return_Pct"].iloc[-1])
    bh_return_pct = float(
        ((strat_df["Actual"].iloc[-1] - strat_df["Actual"].iloc[0]) / (strat_df["Actual"].iloc[0] + 1e-8)) * 100.0
    )

    total_trades = len(trades)
    winning_trades = int((trades_df["Profit/Loss (%)"] > 0).sum()) if not trades_df.empty else 0
    losing_trades = int((trades_df["Profit/Loss (%)"] <= 0).sum()) if not trades_df.empty else 0
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    ret_series = pd.Series(strat_returns)
    std_ret = ret_series.std()
    sharpe = float((ret_series.mean() / std_ret) * np.sqrt(252)) if len(ret_series) > 1 and std_ret > 0 else None

    peak = strat_df["Cum_Growth"].cummax()
    drawdown = (strat_df["Cum_Growth"] - peak) / peak
    max_dd = float(drawdown.min() * 100.0) if not drawdown.empty else None

    current_pos_str = "In Position (Long)" if strat_df["Signal"].iloc[-1] == 1 else "Cash (Neutral)"

    summary = {
        "total_days": len(strat_df),
        "strategy_total_return_pct": round(total_return_pct, 2),
        "buy_hold_return_pct": round(bh_return_pct, 2),
        "win_rate_pct": round(win_rate, 2) if total_trades > 0 else None,
        "sharpe_ratio": round(sharpe, 2) if sharpe is not None and math.isfinite(sharpe) else None,
        "max_drawdown_pct": round(max_dd, 2) if max_dd is not None and math.isfinite(max_dd) else None,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "initial_investment": initial_investment,
        "final_portfolio_value": round(curr_portfolio, 2),
        "current_position": current_pos_str,
    }

    return strat_df, trades_df, summary


def _build_live_strategy_figure(
    strat_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    symbol: str,
    model_type: str,
) -> go.Figure:

    fig = go.Figure()
    if strat_df.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=strat_df["Date"],
            y=strat_df["Actual"],
            mode="lines+markers",
            name="Actual Close",
            line=dict(color="#0F172A", width=2),
            marker=dict(size=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=strat_df["Date"],
            y=strat_df["Predicted"],
            mode="lines+markers",
            name=f"{model_type} Prediction",
            line=dict(color="#4338CA", width=1.5, dash="dash"),
            marker=dict(size=4, symbol="diamond"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=strat_df["Date"],
            y=strat_df["EMA_Filter"],
            mode="lines",
            name="EMA Filter",
            line=dict(color="#F59E0B", width=1.5),
        )
    )

    if not trades_df.empty:
        buy_entries = trades_df[["Entry Date", "Entry Price"]].drop_duplicates()
        fig.add_trace(
            go.Scatter(
                x=buy_entries["Entry Date"],
                y=buy_entries["Entry Price"],
                mode="markers",
                name="BUY Signal",
                marker=dict(color="#059669", size=10, symbol="triangle-up"),
            )
        )
        exit_entries = trades_df[["Exit Date", "Exit Price"]].drop_duplicates()
        fig.add_trace(
            go.Scatter(
                x=exit_entries["Exit Date"],
                y=exit_entries["Exit Price"],
                mode="markers",
                name="CLOSE Signal",
                marker=dict(color="#E11D48", size=10, symbol="triangle-down"),
            )
        )

    fig.update_layout(
        title=f"Trading Strategy Performance Since Model Was Saved ({symbol})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def get_post_save_strategy_analysis(model_id: str, df: pd.DataFrame, ema_period: int = 9) -> Dict[str, Any]:
    """
    Evaluate trading strategy performance strictly on post-save live predictions (Graph #2).
    STRICT RULE: Starts fresh from zero/initial equity on live monitoring boundary.
    Does NOT use original backtest trades or test-set predictions.
    """
    model_obj, scaler_obj, metadata = load_saved_model(model_id)
    live_boundary = metadata.get("train_end")
    model_type = metadata["model_type"]

    if df is not None and not df.empty:
        update_actual_values(model_id, df)

    history = get_model_predictions(model_id, limit=5000)

    evaluated_live = [
        h
        for h in history
        if h.get("actual_value") is not None and (not live_boundary or str(h["prediction_date"])[:10] > live_boundary)
    ]
    evaluated_live.sort(key=lambda x: str(x["prediction_date"]))

    if not evaluated_live:
        return {
            "has_data": False,
            "message": "Live strategy will begin when the first new prediction is generated.",
            "metrics": None,
            "figure": None,
            "trades": [],
        }

    strat_df, trades_df, strategy_metrics = _evaluate_live_strategy(evaluated_live, df, ema_period=ema_period)

    if strat_df.empty:
        return {
            "has_data": False,
            "message": "Live strategy will begin when the first new prediction is generated.",
            "metrics": None,
            "figure": None,
            "trades": [],
        }

    fig = _build_live_strategy_figure(strat_df, trades_df, metadata["symbol"], model_type)
    trades_list = json.loads(trades_df.to_json(orient="records", date_format="iso")) if not trades_df.empty else []

    return {
        "has_data": True,
        "metrics": strategy_metrics,
        "figure": figure_to_json(fig),
        "trades": trades_list,
    }


def get_current_signal_analysis(model_id: str, df: pd.DataFrame, ema_period: int = 9) -> Dict[str, Any]:
    """
    Determine current trading signal (BUY, CLOSE, HOLD) using latest live prediction & EMA filter.
    """
    model_obj, scaler_obj, metadata = load_saved_model(model_id)
    timestep = metadata.get("timestep", 60)
    model_type = metadata["model_type"]
    symbol = metadata["symbol"]
    live_boundary = metadata.get("train_end")

    last_candle_date = df.index[-1].strftime("%Y-%m-%d")
    next_bday = (pd.to_datetime(last_candle_date) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    last_close_val = float(df["close"].iloc[-1])

    # Check if we have live records generated strictly after live_boundary
    history = get_model_predictions(model_id, limit=50)
    has_live_records = any(str(h["prediction_date"])[:10] > live_boundary for h in history) if live_boundary else len(history) > 0

    predicted_val = _ensure_finite_prediction(
        _predict_single_step(model_obj, scaler_obj, model_type, df["close"], timestep),
        model_id,
        model_type,
    )

    ema_series = df["close"].ewm(span=ema_period, adjust=False).mean()
    latest_ema = float(ema_series.iloc[-1])

    pred_condition = predicted_val > last_close_val
    ema_condition = last_close_val > latest_ema
    current_active = pred_condition and ema_condition

    prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else last_close_val
    prev_ema = float(ema_series.iloc[-2]) if len(df) >= 2 else latest_ema
    prev_active = (last_close_val > prev_close) and (prev_close > prev_ema)

    signal_code = "HOLD"
    if not prev_active and current_active:
        signal_code = "BUY"
    elif prev_active and not current_active:
        signal_code = "CLOSE"
    else:
        signal_code = "HOLD"

    position_status = "In Position (Long)" if current_active else "Cash (Neutral)"

    rationale_parts = []
    if pred_condition:
        rationale_parts.append(f"Forecast higher (${predicted_val:.2f} > ${last_close_val:.2f})")
    else:
        rationale_parts.append(f"Forecast lower/flat (${predicted_val:.2f} <= ${last_close_val:.2f})")

    if ema_condition:
        rationale_parts.append(f"Price above EMA({ema_period}) (${last_close_val:.2f} > ${latest_ema:.2f})")
    else:
        rationale_parts.append(f"Price below EMA({ema_period}) (${last_close_val:.2f} <= ${latest_ema:.2f})")

    return {
        "model_id": model_id,
        "symbol": symbol,
        "signal": signal_code if has_live_records else "HOLD",
        "position_status": position_status if has_live_records else "Waiting for Data",
        "predicted_value": round(predicted_val, 4),
        "last_close": round(last_close_val, 4),
        "ema_filter": round(latest_ema, 4),
        "input_date": last_candle_date,
        "target_date": next_bday,
        "rationale": " & ".join(rationale_parts) if has_live_records else "Model saved. Waiting for new market data.",
    }


def run_live_prediction(model_id: str) -> Dict[str, Any]:
    """
    Execute zero-retrain live prediction for fresh market data.
    Creates a new prediction_history record ONLY when a new prediction for target date is requested.
    Prevents duplicate prediction records for (model_id, target_date).
    """
    model_obj, scaler_obj, metadata = load_saved_model(model_id)

    symbol = metadata["symbol"]
    exchange = metadata["exchange"]
    model_type = metadata["model_type"]
    timestep = metadata.get("timestep", 60)
    live_boundary = metadata.get("train_end")

    # 1. Fetch latest market data
    df = load_data(symbol, exchange, dataset_length=500)
    if df is None or len(df) < timestep:
        raise ValueError(f"Insufficient historical data fetched for {symbol} on {exchange}.")

    # 2. Match any existing pending predictions with actual prices
    update_actual_values(model_id, df)

    last_candle_date = df.index[-1].strftime("%Y-%m-%d")
    next_bday = (pd.to_datetime(last_candle_date) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    last_close_val = float(df["close"].iloc[-1])

    # 3. Generate 1-step prediction for next target date
    predicted_val = _ensure_finite_prediction(
        _predict_single_step(model_obj, scaler_obj, model_type, df["close"], timestep),
        model_id,
        model_type,
    )

    # 4. Determine signal for current prediction
    current_signal_info = get_current_signal_analysis(model_id, df)

    # 5. Record prediction if target date is strictly after live_boundary & not already in database
    existing_preds = get_model_predictions(model_id, limit=500)
    existing_dates = {str(p["prediction_date"])[:10] for p in existing_preds}

    if (not live_boundary or next_bday > live_boundary) and next_bday not in existing_dates:
        record_prediction(
            model_id=model_id,
            symbol=symbol,
            input_data_timestamp=last_candle_date,
            prediction_date=next_bday,
            predicted_value=predicted_val,
            prediction_horizon=1,
            signal=current_signal_info.get("signal"),
        )

    # Re-run update actuals in case target_date candle is already present in df
    update_actual_values(model_id, df)

    pred_change = predicted_val - last_close_val
    pred_change_pct = (pred_change / (last_close_val + 1e-8)) * 100

    # 6. Fetch updated prediction history
    history = get_model_predictions(model_id, limit=50)

    # 7. Build Live Prediction Chart (Next-day forecast focus)
    chart_fig = _build_live_prediction_chart(df, history, predicted_val, next_bday, symbol, model_type)

    # 8. Post-Save Monitoring Analysis (STRICT separation from test set!)
    post_save_pred_analysis = get_post_save_predictions_analysis(model_id, df)
    post_save_strategy_analysis = get_post_save_strategy_analysis(model_id, df)
    updated_signal = get_current_signal_analysis(model_id, df)

    # 9. Extract original training snapshot
    training_snapshot = metadata.get("training_snapshot", {})

    result = {
        "model_id": model_id,
        "name": metadata["name"],
        "symbol": symbol,
        "exchange": exchange,
        "model_type": model_type,
        "input_data_timestamp": last_candle_date,
        "prediction_date": next_bday,
        "last_close": round(last_close_val, 4),
        "predicted_value": round(predicted_val, 4),
        "predicted_change": round(pred_change, 4),
        "predicted_change_pct": round(pred_change_pct, 2),
        "live_metrics": post_save_pred_analysis.get("metrics") or {},
        "figure": figure_to_json(chart_fig),
        "predictions_history": history,
        "training_snapshot": training_snapshot,
        "post_save_predictions": post_save_pred_analysis,
        "post_save_strategy": post_save_strategy_analysis,
        "current_signal": updated_signal,
    }
    non_finite = find_non_finite_values(result, "live_monitoring")
    if non_finite:
        logger.warning(
            "Live monitoring result contained non-finite values before JSON sanitization: %s",
            ", ".join(f"{path}={value!r}" for path, value in non_finite),
        )
    return sanitize_for_json(result)


def _build_live_prediction_chart(
    df: pd.DataFrame,
    history: List[Dict[str, Any]],
    latest_predicted_val: float,
    latest_pred_date: str,
    symbol: str,
    model_type: str,
) -> go.Figure:
    """Build Plotly chart comparing actual close history vs predictions."""
    df_recent = df.tail(90).copy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_recent.index.strftime("%Y-%m-%d"),
            y=df_recent["close"],
            mode="lines+markers",
            name="Actual Close",
            line=dict(color="#0F172A", width=2.5),
            marker=dict(size=4),
        )
    )

    if history:
        hist_dates = [h["prediction_date"] for h in history if h.get("actual_value") is not None]
        hist_preds = [h["predicted_value"] for h in history if h.get("actual_value") is not None]

        if hist_dates:
            fig.add_trace(
                go.Scatter(
                    x=hist_dates,
                    y=hist_preds,
                    mode="markers",
                    name="Past Live Predictions",
                    marker=dict(color="#4338CA", size=8, symbol="diamond"),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=[latest_pred_date],
            y=[latest_predicted_val],
            mode="markers+text",
            name="Next Prediction",
            text=[f"{latest_predicted_val:.2f}"],
            textposition="top center",
            marker=dict(color="#059669", size=14, symbol="star"),
        )
    )

    fig.update_layout(
        title=f"Live Prediction for {symbol} ({model_type}) — Target Date: {latest_pred_date}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig
