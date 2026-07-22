from __future__ import annotations

import numpy as np
import pandas as pd


def build_strategy_frame(
    data: pd.DataFrame,
    plot_df_split: pd.DataFrame,
    best_model_name: str,
    ema_period: int,
    signal_threshold: float = 0.0,
    initial_investment: float = 100.0,
):
    strat_df = data.copy()
    strat_df["EMA_Filter"] = strat_df["close"].ewm(span=ema_period, adjust=False).mean()

    test_start_date = plot_df_split.index[0]
    auto_strat_df = strat_df.loc[test_start_date:].copy()

    pred_col = f"{best_model_name} Prediction"
    auto_strat_df[pred_col] = plot_df_split[pred_col]

    auto_strat_df["Prev_Close"] = auto_strat_df["close"].shift(1)
    auto_strat_df["Daily_Return"] = auto_strat_df["close"].pct_change().fillna(0)

    auto_strat_df["Signal"] = 0
    condition_prediction = auto_strat_df[pred_col] > auto_strat_df["Prev_Close"] * (1 + signal_threshold)
    condition_ema = auto_strat_df["close"] > auto_strat_df["EMA_Filter"]

    in_position = False
    for i in range(len(auto_strat_df)):
        current_prediction_condition = condition_prediction.iloc[i]
        current_ema_condition = condition_ema.iloc[i]
        if not in_position:
            if current_prediction_condition and current_ema_condition:
                auto_strat_df.iloc[i, auto_strat_df.columns.get_loc("Signal")] = 1
                in_position = True
        else:
            if not (current_prediction_condition and current_ema_condition):
                auto_strat_df.iloc[i, auto_strat_df.columns.get_loc("Signal")] = 0
                in_position = False
            else:
                auto_strat_df.iloc[i, auto_strat_df.columns.get_loc("Signal")] = 1

    auto_strat_df["Signal_Change"] = auto_strat_df["Signal"].diff().fillna(0).astype(int)

    individual_trades = []
    entry_price = None
    entry_date = None
    in_trade = False
    for index, row in auto_strat_df.iterrows():
        if row["Signal_Change"] == 1 and not in_trade:
            entry_price = row["close"]
            entry_date = index
            in_trade = True
        elif (row["Signal_Change"] == -1 and in_trade) or (in_trade and index == auto_strat_df.index[-1]):
            exit_price = row["close"]
            exit_date = index
            profit_loss = (exit_price - entry_price) / entry_price * 100
            individual_trades.append(
                {
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Exit Date": exit_date,
                    "Exit Price": exit_price,
                    "Profit/Loss (%)": profit_loss,
                }
            )
            in_trade = False
            entry_price = None
            entry_date = None

    individual_trades_df = pd.DataFrame(individual_trades)

    original_index_name = auto_strat_df.index.name if auto_strat_df.index.name is not None else "index"
    auto_strat_df = auto_strat_df.reset_index()

    if not individual_trades_df.empty:
        auto_strat_df = pd.merge(
            auto_strat_df,
            individual_trades_df[["Exit Date", "Profit/Loss (%)"]],
            left_on=original_index_name,
            right_on="Exit Date",
            how="left",
        )
    else:
        auto_strat_df["Profit/Loss (%)"] = np.nan

    auto_strat_df.rename(columns={"Profit/Loss (%)": "Trade_Profit/Loss (%)"}, inplace=True)
    if "Exit Date" in auto_strat_df.columns:
        auto_strat_df.drop(columns=["Exit Date"], inplace=True)
    auto_strat_df.set_index(original_index_name, inplace=True)

    current_portfolio_value = float(initial_investment)
    portfolio_values_list = []
    cum_growth_factors_list = []
    strategy_daily_returns_list = []

    trade_profit_loss_pct_clean = auto_strat_df["Trade_Profit/Loss (%)"].fillna(0)
    for idx in auto_strat_df.index:
        trade_return_on_exit = trade_profit_loss_pct_clean.loc[idx] / 100
        if trade_return_on_exit != 0:
            current_portfolio_value *= 1 + trade_return_on_exit
            strategy_daily_returns_list.append(trade_return_on_exit)
        else:
            strategy_daily_returns_list.append(0.0)
        portfolio_values_list.append(current_portfolio_value)
        cum_growth_factors_list.append(current_portfolio_value / initial_investment)

    auto_strat_df["Portfolio_Value_TK"] = portfolio_values_list
    auto_strat_df["Cum_Growth_Factor"] = cum_growth_factors_list
    auto_strat_df["Strategy_Daily_Return"] = strategy_daily_returns_list
    auto_strat_df["Strategy_Daily_Return_Formatted"] = trade_profit_loss_pct_clean.apply(
        lambda x: f"{x:.2f}%" if x != 0 else "0.00%"
    )
    auto_strat_df["Strategy_Return_Pct"] = (auto_strat_df["Cum_Growth_Factor"] - 1) * 100
    auto_strat_df["BH_Return_Pct"] = ((1 + auto_strat_df["Daily_Return"]).cumprod() - 1) * 100

    total_days = len(auto_strat_df)
    strategy_final_return = auto_strat_df["Strategy_Return_Pct"].iloc[-1] / 100
    bh_final_return = auto_strat_df["BH_Return_Pct"].iloc[-1] / 100

    trades_in_market = auto_strat_df[auto_strat_df["Signal"].shift(1) == 1]
    winning_days = trades_in_market[trades_in_market["Daily_Return"] > 0]
    win_rate = (len(winning_days) / len(trades_in_market) * 100) if len(trades_in_market) > 0 else 0

    buy_signals = int((auto_strat_df["Signal_Change"] == 1).sum())
    exit_signals = int((auto_strat_df["Signal_Change"] == -1).sum())

    strategy_daily_returns = auto_strat_df["Strategy_Daily_Return"]
    sharpe_ratio = (
        (strategy_daily_returns.mean() / strategy_daily_returns.std()) * np.sqrt(252)
        if strategy_daily_returns.std() != 0
        else 0
    )
    rolling_peak = auto_strat_df["Cum_Growth_Factor"].cummax()
    drawdown = (auto_strat_df["Cum_Growth_Factor"] - rolling_peak) / rolling_peak
    max_drawdown = float(drawdown.min())

    winning_trades = int((individual_trades_df["Profit/Loss (%)"] > 0).sum()) if not individual_trades_df.empty else 0
    losing_trades = int((individual_trades_df["Profit/Loss (%)"] <= 0).sum()) if not individual_trades_df.empty else 0
    total_trades = int(len(individual_trades_df))

    metrics_summary = {
        "total_days": total_days,
        "buy_signals": buy_signals,
        "exit_signals": exit_signals,
        "strategy_total_return_pct": float(strategy_final_return * 100),
        "buy_hold_return_pct": float(bh_final_return * 100),
        "win_rate_pct": float(win_rate),
        "outperformance_pct": float((strategy_final_return - bh_final_return) * 100),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown_pct": float(max_drawdown * 100),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "initial_investment": float(initial_investment),
        "final_portfolio_value": float(auto_strat_df["Portfolio_Value_TK"].iloc[-1]),
    }

    return auto_strat_df, individual_trades_df, metrics_summary
