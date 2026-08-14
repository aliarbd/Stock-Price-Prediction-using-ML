"""
Background scheduler service for auto_update enabled models.
Runs non-blocking periodic checks for new candles and updates predictions.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .model_store import list_models
from .prediction_service import run_live_prediction

logger = logging.getLogger(__name__)

_scheduler_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_CHECK_INTERVAL_SECONDS = 300  # Check every 5 minutes


def _scheduler_loop() -> None:
    """Background loop inspecting active auto_update models."""
    logger.info("Auto-prediction background scheduler started.")
    while not _stop_event.is_set():
        try:
            active_models = list_models(is_active=1)
            auto_models = [m for m in active_models if m.get("auto_update") == 1]

            for m in auto_models:
                if _stop_event.is_set():
                    break
                try:
                    model_id = m["id"]
                    logger.info(f"Running scheduled prediction for model {model_id} ({m['name']})...")
                    run_live_prediction(model_id)
                except Exception as exc:
                    logger.error(f"Scheduled prediction error for model {m.get('id')}: {exc}")

        except Exception as exc:
            logger.error(f"Error in scheduler loop: {exc}")

        # Sleep in short chunks to respond quickly to stop_event
        for _ in range(_CHECK_INTERVAL_SECONDS):
            if _stop_event.is_set():
                break
            time.sleep(1)


def start_scheduler() -> None:
    """Start the background scheduler thread if not already running."""
    global _scheduler_thread, _stop_event
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        return

    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True, name="ModelSchedulerThread")
    _scheduler_thread.start()


def stop_scheduler() -> None:
    """Stop the background scheduler thread gracefully."""
    global _scheduler_thread, _stop_event
    _stop_event.set()
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        _scheduler_thread.join(timeout=2.0)
    _scheduler_thread = None
