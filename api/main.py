# FastAPI service wrapping src/score_pipeline.py's single scoring function: /score, /health, /metrics.

import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from src import feature_engineering as fe
from src import model as m
from src import score_pipeline as sp

DATA_PATH = "data/5g_kpi_dataset.csv"

# module-level state: loaded model + in-memory history buffer, populated at startup and appended to per request
_state: dict = {}


def _load_latest_model():
    # same local file-based MLflow store src/model.py's train_isolation_forest writes to
    tracking_uri = (Path(__file__).resolve().parent.parent / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(m.EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"no MLflow experiment {m.EXPERIMENT_NAME!r} found under {tracking_uri}; "
            "run src/model.py's train_isolation_forest (e.g. via notebooks/02) first"
        )
    runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError(f"no MLflow runs found in experiment {m.EXPERIMENT_NAME!r} under {tracking_uri}")
    return mlflow.sklearn.load_model(f"runs:/{runs[0].info.run_id}/model")


def _load_train_history() -> pd.DataFrame:
    # train-side raw rows only; preloading test rows here would duplicate them into history on their first /score call
    df = fe.load_raw(DATA_PATH)
    X, meta = fe.build_feature_matrix(df.copy())
    _, _, meta_train, _ = fe.temporal_train_test_split(X, meta)
    return meta_train[["timestamp", "cell_id", "slice_type"]].merge(
        df, on=["timestamp", "cell_id", "slice_type"], how="left"
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _state["model"] = _load_latest_model()
    _state["history"] = _load_train_history()
    yield
    _state.clear()


app = FastAPI(title="5G KPI Anomaly Detection API", lifespan=_lifespan)

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status_code"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API request latency in seconds", ["endpoint"])
REQUEST_ERRORS = Counter("api_request_errors_total", "Total API request errors", ["endpoint"])


@app.middleware("http")
async def _instrument(request: Request, call_next):
    endpoint = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        REQUEST_ERRORS.labels(endpoint=endpoint).inc()
        REQUEST_COUNT.labels(endpoint=endpoint, method=request.method, status_code="500").inc()
        raise
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    REQUEST_COUNT.labels(endpoint=endpoint, method=request.method, status_code=str(response.status_code)).inc()
    if response.status_code >= 500:
        REQUEST_ERRORS.labels(endpoint=endpoint).inc()
    return response


class KPIRow(BaseModel):
    timestamp: str
    cell_id: str
    cell_type: str
    slice_type: str
    throughput_mbps: float
    latency_ms: float
    packet_loss_pct: float
    handover_count: int
    rsrp_dbm: float
    rsrq_db: float
    prb_utilization_pct: float
    active_users: int


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/score")
def score(row: KPIRow):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw_row = row.model_dump()
    ts = pd.to_datetime(raw_row["timestamp"])
    history = _state["history"]
    # pre-filter to this row's own (cell_id, slice_type) group, strictly-before rows only
    group_history = history[
        (history["cell_id"] == raw_row["cell_id"])
        & (history["slice_type"] == raw_row["slice_type"])
        & (history["timestamp"] < ts)
    ]

    try:
        result = sp.score_row(raw_row, group_history, _state["model"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    new_row = pd.DataFrame([raw_row])
    new_row["timestamp"] = ts
    _state["history"] = pd.concat([history, new_row], ignore_index=True)

    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
