from __future__ import annotations

import pandas as pd
from tvDatafeed import Interval, TvDatafeed


def load_data(stock_symbol: str, exchange: str, dataset_length: int) -> pd.DataFrame:
    fetch_length = max(int(dataset_length), 10000)
    tv = TvDatafeed()
    hist_df = tv.get_hist(
        symbol=stock_symbol,
        exchange=exchange,
        interval=Interval.in_daily,
        n_bars=fetch_length,
    )

    if hist_df is None or hist_df.empty:
        raise ValueError(f"No historical data returned for {stock_symbol} on {exchange}.")

    df = hist_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)
    df = df[(df["close"].diff() != 0) | (df.index == df.index[0])].copy()

    if dataset_length and len(df) > dataset_length:
        df = df.tail(int(dataset_length)).copy()

    return df


def preprocess_data(df: pd.DataFrame):
    from sklearn.preprocessing import MinMaxScaler

    data = df[["close"]].copy()
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    return data, scaler, scaled_data


def create_sequences(data, time_step: int):
    import numpy as np

    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i : (i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)
