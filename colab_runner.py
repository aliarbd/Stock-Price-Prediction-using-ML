"""Colab-friendly wrapper around the FastAPI pipeline."""

from __future__ import annotations

from webapp.config import MODEL_ORDER, PipelineConfig
from webapp.pipeline import run_pipeline


def run_analysis(
    stock_symbol: str = "TSLA",
    exchange: str = "NASDAQ",
    dataset_length: int = 1000,
    train_split: float = 0.8,
    time_step: int = 60,
    ema_period: int = 9,
    forecast_days: int = 5,
    walk_forward_days: int = 5,
    random_seed: int | None = None,
    selected_models=None,
):
    config = PipelineConfig(
        stock_symbol=stock_symbol,
        exchange=exchange,
        dataset_length=dataset_length,
        train_split=train_split,
        time_step=time_step,
        ema_period=ema_period,
        forecast_days=forecast_days,
        walk_forward_days=walk_forward_days,
        random_seed=random_seed,
        selected_models=selected_models or MODEL_ORDER.copy(),
    )
    res = run_pipeline(config)
    return res[0] if isinstance(res, tuple) else res


if __name__ == "__main__":
    run_analysis()
