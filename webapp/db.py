"""
Database setup and connection management for Stock Prediction Lab.
Uses SQLite for lightweight, zero-dependency model metadata and prediction persistence.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)

DB_PATH = SAVED_MODELS_DIR / "app.db"


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection configured with Row factory."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Initialize database tables if they do not exist."""
    SAVED_MODELS_DIR.mkdir(exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Table: saved_models
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '1D',
            model_type TEXT NOT NULL,
            model_path TEXT NOT NULL,
            scaler_path TEXT NOT NULL,
            metadata_path TEXT NOT NULL,
            feature_config TEXT NOT NULL,
            timestep INTEGER NOT NULL,
            target_column TEXT NOT NULL DEFAULT 'close',
            train_start TEXT,
            train_end TEXT,
            train_split REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Active',
            is_active INTEGER NOT NULL DEFAULT 1,
            auto_update INTEGER NOT NULL DEFAULT 0,
            update_interval_minutes INTEGER NOT NULL DEFAULT 1440,
            version INTEGER NOT NULL DEFAULT 1,
            parent_model_id TEXT,
            metrics_json TEXT,
            hyperparameters_json TEXT,
            last_prediction_time TEXT,
            latest_prediction_val REAL,
            training_snapshot_json TEXT
        );
        """)

        # Migration: Ensure training_snapshot_json exists on pre-existing tables
        try:
            cursor.execute("ALTER TABLE saved_models ADD COLUMN training_snapshot_json TEXT;")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Table: predictions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            prediction_timestamp TEXT NOT NULL,
            input_data_timestamp TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            actual_value REAL,
            prediction_horizon INTEGER NOT NULL DEFAULT 1,
            error REAL,
            direction_correct INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (model_id) REFERENCES saved_models (id) ON DELETE CASCADE
        );
        """)

        # Migration: Ensure new prediction tracking columns exist
        for col_def in [
            "status TEXT DEFAULT 'pending'",
            "actual_available_at TEXT",
            "signal TEXT",
            "percentage_error REAL",
        ]:
            try:
                cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_def};")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Indices for speed
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_saved_models_symbol ON saved_models(symbol);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_saved_models_status ON saved_models(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);")

        conn.commit()
