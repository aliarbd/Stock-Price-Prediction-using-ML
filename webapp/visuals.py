from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go


def figure_to_json(fig: go.Figure):
    return json.loads(fig.to_json())


def build_prediction_comparison_figure(plot_df_split: pd.DataFrame, model_names, symbol: str, train_size: int):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df_split.index,
            y=plot_df_split["Actual Price"],
            mode="lines",
            name="Actual Price",
            line=dict(color="darkblue", width=1),
        )
    )

    custom_colors = {
        "LSTM": "darkblue",
        "GRU": "darkgreen",
        "XGBoost": "darkorange",
        "Random Forest": "indigo",
        "Linear Regression": "darkred",
        "SVR": "darkmagenta",
        "LightGBM": "teal",
        "ARIMA": "darkgoldenrod",
        "SARIMA": "purple",
    }

    for model_name in model_names:
        col = f"{model_name} Prediction"
        if col in plot_df_split.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df_split.index,
                    y=plot_df_split[col],
                    mode="lines",
                    name=model_name,
                    line=dict(color=custom_colors.get(model_name, "black"), width=1.5),
                    opacity=0.9,
                )
            )

    fig.update_layout(
        title=dict(
            text=f"Model Performance (Training: {train_size} days, Test: {len(plot_df_split)} days) on Test Set ({symbol})",
            pad_b=40,
        ),
        xaxis_title="Date",
        yaxis_title="Close Price",
        hovermode="x unified",
        template="plotly_white",
        autosize=True,
        margin=dict(l=40, r=20, t=70, b=40),
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_strategy_signals_figure(auto_strat_df: pd.DataFrame, best_model_name: str, ema_period: int):
    pred_col = f"{best_model_name} Prediction"
    entry_points = auto_strat_df[auto_strat_df["Signal_Change"] == 1]
    exit_points = auto_strat_df[auto_strat_df["Signal_Change"] == -1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=auto_strat_df.index,
            y=auto_strat_df["close"],
            mode="lines",
            name="Actual Price",
            line=dict(color="blue", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=auto_strat_df.index,
            y=auto_strat_df["EMA_Filter"],
            mode="lines",
            name=f"EMA {ema_period} (Filter)",
            line=dict(color="green", width=1, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=auto_strat_df.index,
            y=auto_strat_df[pred_col],
            mode="lines",
            name=f"{best_model_name} Prediction",
            line=dict(color="red", width=1.5, dash="solid"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=entry_points.index,
            y=entry_points["close"],
            mode="markers",
            name="New Entry (BUY)",
            marker=dict(color="green", size=12, symbol="triangle-up", line=dict(width=1, color="black")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exit_points.index,
            y=exit_points["close"],
            mode="markers",
            name="Exit to Cash (SELL)",
            marker=dict(color="red", size=12, symbol="triangle-down", line=dict(width=1, color="black")),
        )
    )
    fig.update_layout(
        title=f"Strategy Signals & EMA Filter ({best_model_name})",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=40, r=20, t=70, b=40),
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_equity_curve_figure(auto_strat_df: pd.DataFrame, symbol: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=auto_strat_df.index,
            y=auto_strat_df["Strategy_Return_Pct"],
            name="Strategy Return",
            line=dict(color="darkgreen", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=auto_strat_df.index,
            y=auto_strat_df["BH_Return_Pct"],
            name="Buy & Hold Return",
            line=dict(color="royalblue", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Equity Curve Comparison ({symbol})",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
        template="plotly_white",
        autosize=True,
        margin=dict(l=40, r=20, t=70, b=40),
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_walk_forward_figure(walk_forward_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=walk_forward_df.index,
            y=walk_forward_df["Actual Price"],
            name="Actual Price",
            line=dict(color="black", width=3),
        )
    )
    colors_map = {
        "LSTM": "red",
        "GRU": "green",
        "XGBoost": "orange",
        "Random Forest": "purple",
        "Linear Regression": "brown",
        "SVR": "pink",
        "LightGBM": "gray",
        "ARIMA": "magenta",
        "SARIMA": "cyan",
    }
    for col in walk_forward_df.columns:
        if "Prediction" in col:
            model_name = col.replace(" Prediction", "")
            fig.add_trace(
                go.Scatter(
                    x=walk_forward_df.index,
                    y=walk_forward_df[col],
                    name=model_name,
                    line=dict(color=colors_map.get(model_name, "black"), dash="dash"),
                )
            )
    fig.update_layout(
        title="Walk-Forward Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        autosize=True,
        margin=dict(l=40, r=20, t=70, b=40),
        height=700,
    )
    return fig


def build_future_forecast_figure(future_df: pd.DataFrame, symbol: str):
    fig = go.Figure()
    colors_map = {
        "LSTM": "red",
        "GRU": "green",
        "XGBoost": "orange",
        "Random Forest": "purple",
        "Linear Regression": "brown",
        "SVR": "pink",
        "LightGBM": "gray",
        "ARIMA": "magenta",
        "SARIMA": "cyan",
    }
    for col in future_df.columns:
        if "Future Predictions" in col:
            model_name = col.replace(" Future Predictions", "")
            fig.add_trace(
                go.Scatter(
                    x=future_df.index,
                    y=future_df[col],
                    mode="lines+markers",
                    name=f"{model_name} Future",
                    line=dict(color=colors_map.get(model_name, "black"), width=2.5),
                )
            )
    fig.update_layout(
        title=f"Future Forecast: {symbol} (Next {len(future_df)} Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        autosize=True,
        margin=dict(l=40, r=20, t=70, b=40),
        height=700,
        template="plotly_white",
    )
    return fig
