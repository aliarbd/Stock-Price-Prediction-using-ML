from __future__ import annotations

import threading
import uuid
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request
import uvicorn

from webapp.config import MODEL_ORDER, PipelineConfig
from webapp.pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Stock Prediction Web App", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

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
        result = run_pipeline(config, progress=_progress_callback(job_id))
        _update_job(job_id, status="complete", progress=100, stage="Complete", result=result)
    except Exception as exc:  # pragma: no cover - surfaced to UI
        _update_job(job_id, status="error", stage="Error", error=str(exc))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_order": MODEL_ORDER,
        },
    )


@app.post("/api/runs")
async def create_run(payload: RunRequest):
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
    with job_lock:
        job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return JSONResponse(job)


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
