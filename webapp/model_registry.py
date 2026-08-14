"""
Model registry module managing CRUD operations and prediction history in SQLite.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .db import get_db_connection


def save_model_record(data: Dict[str, Any]) -> str:
    """Save or update a model record in SQLite database."""
    now_iso = datetime.utcnow().isoformat()
    model_id = data["id"]

    metrics_json = json.dumps(data.get("metrics", {})) if isinstance(data.get("metrics"), dict) else data.get("metrics_json", "{}")
    hyper_json = json.dumps(data.get("hyperparameters", {})) if isinstance(data.get("hyperparameters"), dict) else data.get("hyperparameters_json", "{}")
    feature_config = json.dumps(data.get("feature_config", ["close"])) if isinstance(data.get("feature_config"), (list, dict)) else str(data.get("feature_config", '["close"]'))
    snapshot_json = json.dumps(data.get("training_snapshot", {})) if isinstance(data.get("training_snapshot"), dict) else data.get("training_snapshot_json", "{}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO saved_models (
                id, name, symbol, exchange, timeframe, model_type,
                model_path, scaler_path, metadata_path, feature_config,
                timestep, target_column, train_start, train_end, train_split,
                created_at, updated_at, status, is_active, auto_update,
                update_interval_minutes, version, parent_model_id,
                metrics_json, hyperparameters_json, last_prediction_time, latest_prediction_val,
                training_snapshot_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                symbol=excluded.symbol,
                exchange=excluded.exchange,
                status=excluded.status,
                is_active=excluded.is_active,
                auto_update=excluded.auto_update,
                updated_at=excluded.updated_at,
                metrics_json=excluded.metrics_json,
                last_prediction_time=excluded.last_prediction_time,
                latest_prediction_val=excluded.latest_prediction_val,
                training_snapshot_json=excluded.training_snapshot_json;
            """,
            (
                model_id,
                data["name"],
                data["symbol"],
                data["exchange"],
                data.get("timeframe", "1D"),
                data["model_type"],
                data["model_path"],
                data["scaler_path"],
                data["metadata_path"],
                feature_config,
                data.get("timestep", 60),
                data.get("target_column", "close"),
                data.get("train_start"),
                data.get("train_end"),
                data.get("train_split", 0.8),
                data.get("created_at", now_iso),
                now_iso,
                data.get("status", "Active"),
                data.get("is_active", 1),
                data.get("auto_update", 0),
                data.get("update_interval_minutes", 1440),
                data.get("version", 1),
                data.get("parent_model_id"),
                metrics_json,
                hyper_json,
                data.get("last_prediction_time"),
                data.get("latest_prediction_val"),
                snapshot_json,
            ),
        )
        conn.commit()
    return model_id


def get_model_record(model_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single model record by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM saved_models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        res["metrics"] = json.loads(res.get("metrics_json") or "{}")
        res["hyperparameters"] = json.loads(res.get("hyperparameters_json") or "{}")
        res["feature_config"] = json.loads(res.get("feature_config") or '["close"]')
        res["training_snapshot"] = json.loads(res.get("training_snapshot_json") or "{}")
        return res


def list_model_records(symbol: Optional[str] = None, is_active: Optional[int] = None) -> List[Dict[str, Any]]:
    """List saved models with optional filtering."""
    query = "SELECT * FROM saved_models"
    params: List[Any] = []
    conditions = []

    if symbol:
        conditions.append("symbol = ?")
        params.append(symbol.upper())
    if is_active is not None:
        conditions.append("is_active = ?")
        params.append(is_active)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY created_at DESC"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        models = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d.get("metrics_json") or "{}")
            d["hyperparameters"] = json.loads(d.get("hyperparameters_json") or "{}")
            d["feature_config"] = json.loads(d.get("feature_config") or '["close"]')
            d["training_snapshot"] = json.loads(d.get("training_snapshot_json") or "{}")
            models.append(d)
        return models


def update_model_status(model_id: str, status: str, is_active: int) -> bool:
    """Update active status of a model."""
    now_iso = datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE saved_models SET status = ?, is_active = ?, updated_at = ? WHERE id = ?",
            (status, is_active, now_iso, model_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def toggle_auto_update(model_id: str, auto_update: int) -> bool:
    """Toggle auto_update setting for a model."""
    now_iso = datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE saved_models SET auto_update = ?, updated_at = ? WHERE id = ?",
            (auto_update, now_iso, model_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_model_record(model_id: str) -> bool:
    """Delete model database record."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM saved_models WHERE id = ?", (model_id,))
        conn.commit()
        return cursor.rowcount > 0


def record_prediction(
    model_id: str,
    symbol: str,
    input_data_timestamp: str,
    prediction_date: str,
    predicted_value: float,
    prediction_horizon: int = 1,
    signal: Optional[str] = None,
) -> int:
    """
    Insert a prediction entry and update model record latest prediction.
    Prevents duplicate predictions for the same (model_id, prediction_date).
    """
    now_iso = datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Duplicate check: check if prediction for model_id + prediction_date already exists
        cursor.execute(
            "SELECT id FROM predictions WHERE model_id = ? AND prediction_date = ?",
            (model_id, str(prediction_date)),
        )
        existing = cursor.fetchone()
        if existing:
            return existing["id"]

        cursor.execute(
            """
            INSERT INTO predictions (
                model_id, symbol, prediction_timestamp, input_data_timestamp,
                prediction_date, predicted_value, prediction_horizon, created_at,
                status, signal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                symbol,
                now_iso,
                str(input_data_timestamp),
                str(prediction_date),
                float(predicted_value),
                int(prediction_horizon),
                now_iso,
                "pending",
                signal,
            ),
        )
        prediction_id = cursor.lastrowid

        cursor.execute(
            """
            UPDATE saved_models
            SET last_prediction_time = ?, latest_prediction_val = ?, updated_at = ?
            WHERE id = ?
            """,
            (now_iso, float(predicted_value), now_iso, model_id),
        )
        conn.commit()
        return prediction_id


def get_model_predictions(model_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve prediction history for a model."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM predictions WHERE model_id = ? ORDER BY prediction_date DESC, id DESC LIMIT ?",
            (model_id, limit),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def update_actual_values(model_id: str, df_latest: pd.DataFrame) -> Dict[str, Any]:
    """
    Check pending predictions (where actual_value is NULL) against df_latest.
    Compute errors and directional accuracy for unseen predictions.
    """
    if df_latest is None or df_latest.empty:
        return {}

    now_iso = datetime.utcnow().isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM predictions WHERE model_id = ? AND actual_value IS NULL",
            (model_id,),
        )
        pending = [dict(r) for r in cursor.fetchall()]

        if pending:
            df_close = df_latest["close"].copy()
            df_close.index = pd.to_datetime(df_close.index).strftime("%Y-%m-%d")

            for p in pending:
                p_date = str(p["prediction_date"])[:10]
                if p_date in df_close.index:
                    actual = float(df_close.loc[p_date])
                    pred = float(p["predicted_value"])
                    err = abs(actual - pred)
                    pct_err = (err / (abs(actual) + 1e-8)) * 100

                    # Direction check
                    inp_date = str(p["input_data_timestamp"])[:10]
                    direction = None
                    if inp_date in df_close.index:
                        prev_close = float(df_close.loc[inp_date])
                        pred_dir = pred >= prev_close
                        actual_dir = actual >= prev_close
                        direction = 1 if pred_dir == actual_dir else 0

                    cursor.execute(
                        """
                        UPDATE predictions
                        SET actual_value = ?, error = ?, percentage_error = ?, direction_correct = ?, status = 'evaluated', actual_available_at = ?
                        WHERE id = ?
                        """,
                        (actual, err, pct_err, direction, now_iso, p["id"]),
                    )

            conn.commit()
        return _compute_live_metrics(conn, model_id)


def _compute_live_metrics(conn: Any, model_id: str) -> Dict[str, Any]:
    """Compute performance metrics strictly on evaluated post-save live predictions."""
    cursor = conn.cursor()
    cursor.execute("SELECT train_end FROM saved_models WHERE id = ?", (model_id,))
    model_row = cursor.fetchone()
    live_boundary = model_row["train_end"] if model_row else None

    cursor.execute(
        "SELECT prediction_date, predicted_value, actual_value, direction_correct FROM predictions WHERE model_id = ? AND actual_value IS NOT NULL",
        (model_id,),
    )
    rows = cursor.fetchall()
    
    # Filter strictly to predictions post live boundary
    eval_rows = []
    for r in rows:
        p_date = str(r["prediction_date"])[:10]
        if not live_boundary or p_date > live_boundary:
            eval_rows.append(r)

    if not eval_rows:
        return {
            "live_samples": 0,
            "live_rmse": None,
            "live_mae": None,
            "live_mape": None,
            "live_directional_accuracy": None,
        }

    preds = np.array([r["predicted_value"] for r in eval_rows])
    actuals = np.array([r["actual_value"] for r in eval_rows])
    dirs = [r["direction_correct"] for r in eval_rows if r["direction_correct"] is not None]

    mse = np.mean((actuals - preds) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actuals - preds)))
    mape = float(np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100)
    da = float(np.mean(dirs) * 100) if dirs else None

    return {
        "live_samples": len(eval_rows),
        "live_rmse": round(rmse, 4),
        "live_mae": round(mae, 4),
        "live_mape": round(mape, 4),
        "live_directional_accuracy": round(da, 2) if da is not None else None,
    }
