"""
Model store layer providing disk persistence (models, scalers, metadata)
integrated with SQLite database registry.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf

from .db import SAVED_MODELS_DIR
from .model_registry import (
    delete_model_record,
    get_model_predictions,
    get_model_record,
    list_model_records,
    save_model_record,
    update_model_status,
)


def save_trained_model(
    model_id: str,
    name: str,
    symbol: str,
    exchange: str,
    model_type: str,
    model_obj: Any,
    scaler_obj: Any,
    metrics: Dict[str, Any],
    timestep: int = 60,
    timeframe: str = "1D",
    feature_config: Optional[List[str]] = None,
    target_column: str = "close",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    train_split: float = 0.8,
    hyperparameters: Optional[Dict[str, Any]] = None,
    version: int = 1,
    parent_model_id: Optional[str] = None,
    training_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist trained model, scaler, metadata JSON, and database record.
    """
    model_dir = SAVED_MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Model Object according to model_type
    if model_type in ["LSTM", "GRU"]:
        model_path = model_dir / "model.keras"
        try:
            model_obj.save(model_path)
        except Exception:
            # Fallback to .h5 if native keras fails
            model_path = model_dir / "model.h5"
            model_obj.save(model_path)
    else:
        # Scikit-learn, XGBoost, LightGBM, ARIMA, SARIMA
        model_path = model_dir / "model.joblib"
        joblib.dump(model_obj, model_path)

    # 2. Save Scaler Object (Fitted MinMaxScaler)
    scaler_path = model_dir / "scaler.joblib"
    joblib.dump(scaler_obj, scaler_path)

    # 3. Save Metadata JSON
    feature_config = feature_config or ["close"]
    hyperparameters = hyperparameters or {}
    training_snapshot = training_snapshot or {}

    metadata = {
        "id": model_id,
        "name": name,
        "symbol": symbol.upper(),
        "exchange": exchange.upper(),
        "timeframe": timeframe,
        "model_type": model_type,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "metadata_path": str(model_dir / "metadata.json"),
        "feature_config": feature_config,
        "timestep": timestep,
        "target_column": target_column,
        "train_start": train_start,
        "train_end": train_end,
        "train_split": train_split,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "status": "Active",
        "is_active": 1,
        "auto_update": 0,
        "version": version,
        "parent_model_id": parent_model_id,
        "training_snapshot": training_snapshot,
    }

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # 4. Save Database Record
    save_model_record(metadata)

    return metadata


def load_saved_model(model_id: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load saved model object, scaler object, and metadata record.
    """
    model_record = get_model_record(model_id)
    if not model_record:
        # Fallback to loading metadata directly from disk if record missing in DB
        metadata_path = SAVED_MODELS_DIR / model_id / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model {model_id} not found.")
        with open(metadata_path, "r") as f:
            model_record = json.load(f)

    model_type = model_record["model_type"]
    model_path = Path(model_record["model_path"])
    scaler_path = Path(model_record["scaler_path"])

    if not model_path.exists():
        # Fallback check inside directory
        model_dir = SAVED_MODELS_DIR / model_id
        if (model_dir / "model.keras").exists():
            model_path = model_dir / "model.keras"
        elif (model_dir / "model.h5").exists():
            model_path = model_dir / "model.h5"
        elif (model_dir / "model.joblib").exists():
            model_path = model_dir / "model.joblib"
        else:
            raise FileNotFoundError(f"Model artifact file missing for model {model_id}.")

    if not scaler_path.exists():
        scaler_path = SAVED_MODELS_DIR / model_id / "scaler.joblib"

    # Load Model Object
    if model_type in ["LSTM", "GRU"]:
        model_obj = tf.keras.models.load_model(model_path)
    else:
        model_obj = joblib.load(model_path)

    # Load Scaler Object
    scaler_obj = joblib.load(scaler_path)

    return model_obj, scaler_obj, model_record


def delete_saved_model(model_id: str) -> bool:
    """
    Delete saved model files and database record.
    """
    model_dir = SAVED_MODELS_DIR / model_id
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)

    delete_model_record(model_id)
    return True


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve metadata and predictions for a saved model."""
    record = get_model_record(model_id)
    if not record:
        return None
    if not record.get("training_snapshot"):
        metadata_path = SAVED_MODELS_DIR / model_id / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    disk_meta = json.load(f)
                    record["training_snapshot"] = disk_meta.get("training_snapshot", {})
            except Exception:
                pass
    record["predictions"] = get_model_predictions(model_id)
    return record


def list_models(symbol: Optional[str] = None, is_active: Optional[int] = None) -> List[Dict[str, Any]]:
    """List all saved models."""
    return list_model_records(symbol=symbol, is_active=is_active)