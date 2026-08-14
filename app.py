from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request
import uvicorn

from webapp.config import MODEL_ORDER, PipelineConfig
from webapp.data import load_data
from webapp.db import init_db
from webapp.model_registry import (
    get_model_predictions,
    toggle_auto_update,
    update_model_status,
)
from webapp.model_store import (
    delete_saved_model,
    get_model,
    list_models,
    save_trained_model,
)
from webapp.pipeline import run_pipeline
from webapp.prediction_service import (
    get_current_signal_analysis,
    get_post_save_predictions_analysis,
    get_post_save_strategy_analysis,
    run_live_prediction,
)
from webapp.scheduler import start_scheduler, stop_scheduler

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Stock Prediction Web App", version="1.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "status": "error",
            "error": str(exc) or "Internal Server Error",
            "detail": traceback.format_exc(),
        },
    )

executor = ThreadPoolExecutor(max_workers=2)
job_store: Dict[str, Dict] = {}
job_lock = threading.Lock()


class RunRequest(BaseModel):
    stock_symbol: str = Field(..., min_length=1)
    exchange: str = Field(..., min_length=1)
    dataset_length: int = Field(1000, ge=120)
    train_split: float = Field(0.8, ge=0.5, le=0.95)
    time_step: int = Field(60, ge=5, le=365)
    ema_period: int = Field(9, ge=2, le=365)
    forecast_days: int = Field(5, ge=1, le=30)
    walk_forward_days: int = Field(5, ge=1, le=30)
    random_seed: Optional[int] = Field(default=None)
    selected_models: List[str] = Field(default_factory=lambda: MODEL_ORDER.copy())


class SaveModelRequest(BaseModel):
    job_id: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)


@app.on_event("startup")
def on_startup():
    """Initialize SQLite database and background scheduler on application startup."""
    init_db()
    start_scheduler()


@app.on_event("shutdown")
def on_shutdown():
    """Shutdown background scheduler gracefully."""
    stop_scheduler()


def _update_job(job_id: str, **kwargs) -> None:
    with job_lock:
        job = job_store.setdefault(job_id, {})
        job.update(kwargs)


def _progress_callback(job_id: str):
    def _callback(stage: str, percent: float) -> None:
        _update_job(job_id, stage=stage, progress=round(float(percent), 2))

    return _callback


def _run_job(job_id: str, payload: RunRequest) -> None:
    try:
        _update_job(job_id, status="running", progress=0, stage="Queued")
        config = PipelineConfig(
            stock_symbol=payload.stock_symbol.upper().strip(),
            exchange=payload.exchange.upper().strip(),
            dataset_length=payload.dataset_length,
            train_split=payload.train_split,
            time_step=payload.time_step,
            ema_period=payload.ema_period,
            forecast_days=payload.forecast_days,
            walk_forward_days=payload.walk_forward_days,
            random_seed=payload.random_seed,
            selected_models=payload.selected_models,
        )
        result, trained_models, scaler, train_dates = run_pipeline(config, progress=_progress_callback(job_id))
        _update_job(
            job_id,
            status="complete",
            progress=100,
            stage="Complete",
            result=result,
            config=config,
            trained_models=trained_models,
            scaler=scaler,
            train_dates=train_dates,
        )
    except Exception as exc:  # pragma: no cover - surfaced to UI
        import traceback
        traceback.print_exc()
        _update_job(
            job_id,
            status="error",
            stage="Error",
            error=str(exc),
            detail=traceback.format_exc(),
        )


# ============================================================================
# PAGE ROUTES (HTML)
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Main Training Dashboard Page."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "active_page": "dashboard",
            "model_order": MODEL_ORDER,
        },
    )


@app.get("/models", response_class=HTMLResponse)
async def saved_models_page(request: Request):
    """Saved Models Registry Page."""
    return templates.TemplateResponse(
        request=request,
        name="models.html",
        context={
            "active_page": "models",
        },
    )


@app.get("/models/{model_id}", response_class=HTMLResponse)
async def model_detail_page(request: Request, model_id: str):
    """Model Details & History Page."""
    return templates.TemplateResponse(
        request=request,
        name="model_detail.html",
        context={
            "active_page": "models",
            "model_id": model_id,
        },
    )


@app.get("/predict", response_class=HTMLResponse)
async def live_prediction_page(request: Request):
    """Live Prediction Page."""
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "active_page": "predict",
        },
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.post("/api/runs")
async def create_run(payload: RunRequest):
    """Trigger background model training pipeline run."""
    job_id = uuid.uuid4().hex
    with job_lock:
        job_store[job_id] = {
            "status": "queued",
            "progress": 0,
            "stage": "Queued",
            "result": None,
            "error": None,
        }
    executor.submit(_run_job, job_id, payload)
    return {"job_id": job_id}


@app.get("/api/runs/{job_id}")
async def get_run(job_id: str):
    """Get status and result for a training run job."""
    with job_lock:
        job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Run not found")
    # Return clean job response without binary objects
    clean_job = {
        "status": job.get("status"),
        "progress": job.get("progress"),
        "stage": job.get("stage"),
        "result": job.get("result"),
        "error": job.get("error"),
    }
    return JSONResponse(clean_job)


@app.post("/api/models/save")
async def save_model_endpoint(payload: SaveModelRequest):
    """
    Save a trained model instance from a completed training run into SQLite and disk storage.
    """
    with job_lock:
        job = job_store.get(payload.job_id)

    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=400, detail="Completed run job not found.")

    trained_models = job.get("trained_models") or {}
    scaler = job.get("scaler")
    config: Optional[PipelineConfig] = job.get("config")
    train_dates = job.get("train_dates") or {}
    result = job.get("result") or {}

    model_name = payload.model_name
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' was not trained in run {payload.job_id}.")

    model_obj = trained_models[model_name]
    if model_obj is None or scaler is None or config is None:
        raise HTTPException(status_code=400, detail="Model runtime artifacts missing from job memory.")

    # Find metrics row for this model from summary stats table
    table_rows = result.get("comparison", {}).get("table", [])
    model_metrics = {}
    for r in table_rows:
        if r.get("Model") == model_name:
            model_metrics = r
            break

    model_id = uuid.uuid4().hex
    display_name = f"{config.stock_symbol}_{model_name.replace(' ', '')}_V1"

    # Complete snapshot of training experiment
    training_snapshot = {
        "dataset_summary": result.get("dataset_summary", {}),
        "comparison": result.get("comparison", {}),
        "best_model": result.get("best_model", {}),
        "strategy": result.get("strategy", {}),
        "walk_forward": result.get("walk_forward", {}),
        "future_prediction": result.get("future_prediction", {}),
        "selected_models": result.get("selected_models", []),
        "config": {
            "time_step": config.time_step,
            "ema_period": config.ema_period,
            "forecast_days": config.forecast_days,
            "walk_forward_days": config.walk_forward_days,
            "dataset_length": config.dataset_length,
            "train_split": config.train_split,
        },
        "train_dates": train_dates,
    }

    saved_metadata = save_trained_model(
        model_id=model_id,
        name=display_name,
        symbol=config.stock_symbol,
        exchange=config.exchange,
        model_type=model_name,
        model_obj=model_obj,
        scaler_obj=scaler,
        metrics=model_metrics,
        timestep=config.time_step,
        timeframe="1D",
        feature_config=["close"],
        target_column="close",
        train_start=train_dates.get("start"),
        train_end=train_dates.get("end"),
        train_split=config.train_split,
        version=1,
        training_snapshot=training_snapshot,
    )

    return {
        "status": "success",
        "model_id": model_id,
        "name": display_name,
        "symbol": config.stock_symbol,
        "model_type": model_name,
        "metadata": saved_metadata,
    }


@app.get("/api/models")
async def list_models_endpoint(symbol: Optional[str] = None):
    """List all saved models."""
    models = list_models(symbol=symbol)
    return JSONResponse(models)


@app.get("/api/models/{model_id}")
async def get_model_endpoint(model_id: str):
    """Get metadata, prediction history, and monitoring data for a saved model."""
    model_info = get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Saved model {model_id} not found.")
    
    # Attempt to attach live monitoring summaries
    try:
        df = load_data(model_info["symbol"], model_info["exchange"], dataset_length=500)
        if df is not None:
            model_info["post_save_predictions"] = get_post_save_predictions_analysis(model_id, df)
            model_info["post_save_strategy"] = get_post_save_strategy_analysis(model_id, df)
            model_info["current_signal"] = get_current_signal_analysis(model_id, df)
    except Exception:
        pass

    return JSONResponse(model_info)


@app.get("/api/models/{model_id}/training-summary")
async def get_training_summary_endpoint(model_id: str):
    """Retrieve original training experiment snapshot for a saved model."""
    model_info = get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Saved model {model_id} not found.")
    return JSONResponse(model_info.get("training_snapshot", {}))


@app.get("/api/models/{model_id}/post-save-predictions")
async def get_post_save_predictions_endpoint(model_id: str):
    """Retrieve post-save actual vs predicted performance metrics and chart."""
    model_info = get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Saved model {model_id} not found.")
    try:
        df = load_data(model_info["symbol"], model_info["exchange"], dataset_length=500)
        res = get_post_save_predictions_analysis(model_id, df)
        return JSONResponse(res)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/models/{model_id}/post-save-strategy")
async def get_post_save_strategy_endpoint(model_id: str):
    """Retrieve post-save trading strategy performance metrics and chart."""
    model_info = get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Saved model {model_id} not found.")
    try:
        df = load_data(model_info["symbol"], model_info["exchange"], dataset_length=500)
        res = get_post_save_strategy_analysis(model_id, df)
        return JSONResponse(res)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/models/{model_id}/current-signal")
async def get_current_signal_endpoint(model_id: str):
    """Retrieve current trading signal for a saved model."""
    model_info = get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Saved model {model_id} not found.")
    try:
        df = load_data(model_info["symbol"], model_info["exchange"], dataset_length=500)
        res = get_current_signal_analysis(model_id, df)
        return JSONResponse(res)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/models/{model_id}")
async def delete_model_endpoint(model_id: str):
    """Delete a saved model and its artifacts."""
    success = delete_saved_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    return {"status": "success", "deleted_model_id": model_id}


@app.post("/api/models/{model_id}/predict")
async def live_predict_endpoint(model_id: str):
    """Execute zero-retrain live prediction using fresh market data."""
    try:
        res = run_live_prediction(model_id)
        return JSONResponse(res)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/models/{model_id}/retrain")
async def retrain_model_endpoint(model_id: str):
    """
    Retrain an existing saved model on fresh market data and save as a new version.
    """
    existing = get_model(model_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")

    new_version = int(existing.get("version", 1)) + 1
    new_model_id = uuid.uuid4().hex

    config = PipelineConfig(
        stock_symbol=existing["symbol"],
        exchange=existing["exchange"],
        dataset_length=1000,
        train_split=existing.get("train_split", 0.8),
        time_step=existing.get("timestep", 60),
        selected_models=[existing["model_type"]],
    )

    # Run pipeline for single model
    pipeline_res, trained_models, scaler, train_dates = run_pipeline(config)
    model_obj = trained_models.get(existing["model_type"])

    table_rows = pipeline_res.get("comparison", {}).get("table", [])
    metrics = table_rows[0] if table_rows else {}

    display_name = f"{existing['symbol']}_{existing['model_type'].replace(' ', '')}_V{new_version}"

    retrain_snapshot = {
        "dataset_summary": pipeline_res.get("dataset_summary", {}),
        "comparison": pipeline_res.get("comparison", {}),
        "best_model": pipeline_res.get("best_model", {}),
        "strategy": pipeline_res.get("strategy", {}),
        "walk_forward": pipeline_res.get("walk_forward", {}),
        "future_prediction": pipeline_res.get("future_prediction", {}),
        "train_dates": train_dates,
    }

    save_trained_model(
        model_id=new_model_id,
        name=display_name,
        symbol=existing["symbol"],
        exchange=existing["exchange"],
        model_type=existing["model_type"],
        model_obj=model_obj,
        scaler_obj=scaler,
        metrics=metrics,
        timestep=existing.get("timestep", 60),
        train_start=train_dates.get("start"),
        train_end=train_dates.get("end"),
        train_split=config.train_split,
        version=new_version,
        parent_model_id=model_id,
        training_snapshot=retrain_snapshot,
    )

    return {
        "status": "success",
        "model_id": new_model_id,
        "version": new_version,
        "name": display_name,
    }


@app.get("/api/models/{model_id}/predictions")
async def get_predictions_endpoint(model_id: str, limit: int = 100):
    """Retrieve prediction history for a specific model."""
    predictions = get_model_predictions(model_id, limit=limit)
    return JSONResponse(predictions)


@app.post("/api/models/{model_id}/activate")
async def activate_model_endpoint(model_id: str):
    """Activate a model."""
    update_model_status(model_id, status="Active", is_active=1)
    return {"status": "success", "is_active": 1}


@app.post("/api/models/{model_id}/deactivate")
async def deactivate_model_endpoint(model_id: str):
    """Deactivate a model."""
    update_model_status(model_id, status="Deactivated", is_active=0)
    return {"status": "success", "is_active": 0}


@app.post("/api/models/{model_id}/toggle-autoupdate")
async def toggle_autoupdate_endpoint(model_id: str, auto_update: int = Query(..., ge=0, le=1)):
    """Toggle auto_update flag for a model."""
    toggle_auto_update(model_id, auto_update=auto_update)
    return {"status": "success", "auto_update": auto_update}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
