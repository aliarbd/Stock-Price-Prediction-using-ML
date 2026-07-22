from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


MODEL_ORDER = [
    "Linear Regression",
    "SVR",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "LSTM",
    "GRU",
    "ARIMA",
    "SARIMA",
]

RNN_MODELS = {"LSTM", "GRU"}
TREE_MODELS = {
    "XGBoost",
    "Random Forest",
    "Linear Regression",
    "SVR",
    "LightGBM",
}
STATISTICAL_MODELS = {"ARIMA", "SARIMA"}
WALK_FORWARD_MODELS = [
    "LSTM",
    "GRU",
    "XGBoost",
    "Random Forest",
    "Linear Regression",
    "SVR",
    "LightGBM",
]


@dataclass(slots=True)
class PipelineConfig:
    stock_symbol: str
    exchange: str
    dataset_length: int = 1000
    train_split: float = 0.8
    time_step: int = 60
    ema_period: int = 9
    forecast_days: int = 5
    walk_forward_days: int = 5
    random_seed: Optional[int] = None
    selected_models: List[str] = field(default_factory=lambda: MODEL_ORDER.copy())

    def normalized_models(self) -> List[str]:
        selected = [model for model in MODEL_ORDER if model in set(self.selected_models)]
        return selected or MODEL_ORDER.copy()
