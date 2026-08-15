from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any, List, Tuple

import numpy as np
import pandas as pd


def is_non_finite_number(value: Any) -> bool:
    """Return True for NaN, +Infinity, or -Infinity numeric values."""
    if isinstance(value, (float, np.floating)):
        return not math.isfinite(float(value))
    return False


def find_non_finite_values(value: Any, path: str = "root") -> List[Tuple[str, Any]]:
    """Recursively identify non-finite numeric values before JSON serialization."""
    if is_non_finite_number(value):
        return [(path, value)]
    if isinstance(value, dict):
        issues: List[Tuple[str, Any]] = []
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            issues.extend(find_non_finite_values(item, next_path))
        return issues
    if isinstance(value, (list, tuple, set)):
        issues = []
        for index, item in enumerate(value):
            issues.extend(find_non_finite_values(item, f"{path}[{index}]"))
        return issues
    if isinstance(value, np.ndarray):
        return find_non_finite_values(value.tolist(), path)
    if isinstance(value, pd.Series):
        return find_non_finite_values(value.to_dict(), path)
    if isinstance(value, pd.DataFrame):
        return find_non_finite_values(value.to_dict(orient="records"), path)
    return []


def sanitize_for_json(value: Any) -> Any:
    """Convert API payloads into standards-compliant JSON-safe objects."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (float, np.floating)):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Decimal):
        value_float = float(value)
        return value_float if math.isfinite(value_float) else None
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, pd.Series):
        return sanitize_for_json(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return sanitize_for_json(value.to_dict(orient="records"))
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    return value
