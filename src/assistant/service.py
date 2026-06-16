"""FastAPI service for the travel assistant.

Endpoints:
- POST /chat           — main path; returns the response and updates Prometheus.
- POST /admin/reload   — re-resolve current alias/config and atomically swap the
                         live pipeline. No downtime; in-flight /chat requests
                         finish on the previous pipeline.
- GET  /metrics        — Prometheus exposition (mounted ASGI app).
- GET  /health         — liveness check.

Lifecycle: on startup, resolve the deployment config (either from a local
file in dev mode or via the MLflow Model Registry alias in production) and
spawn the async judge worker. On shutdown, cancel the worker.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException
from prometheus_client import make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

from src.assistant import AssistantResponse, Pipeline, build_pipeline
from src.config import get_settings
from src.configs import load_config, load_config_from_registry
from src.pricing import cost_usd
from src.monitoring.judge_worker import JudgeWorker
from src.monitoring.metrics import (
    assistant_info,
    chat_cost_usd_total,
    chat_input_tokens,
    chat_output_tokens,
    chat_request_duration_seconds,
    chat_requests_total,
    deep_judge_queue_depth,
    in_flight_requests,
    judge_sample_rate,
    llm_api_errors_total,
)

log = logging.getLogger(__name__)


# Order of assistant_info labels. Must match the declaration in
# src/monitoring/metrics.py — used to call gauge.remove(*values) on swap.
_ASSISTANT_INFO_LABEL_ORDER = (
    "config_id",
    "model",
    "guardrail_type",
    "model_name",
    "model_alias",
    "model_version",
)


# --- Reload machinery -------------------------------------------------------

async def _resolve_and_build() -> tuple[Pipeline, dict[str, str]]:
    """Resolve current config (Registry alias if set, else local dev config)
    and build a new Pipeline. Either both succeed (return them) or this raises;
    in the latter case the caller must NOT touch the running state."""
    settings = get_settings()
    if settings.assistant_model_alias:
        config, version = await asyncio.to_thread(
            load_config_from_registry,
            settings.mlflow_registered_model_name,
            settings.assistant_model_alias,
        )
        labels = {
            "config_id": config.config_id or "unknown",
            "model": config.model.name,
            "guardrail_type": config.guardrail.type,
            "model_name": settings.mlflow_registered_model_name,
            "model_alias": settings.assistant_model_alias,
            "model_version": str(version),
        }
    else:
        config = await asyncio.to_thread(
            load_config, settings.assistant_config, settings.configs_dir
        )
        labels = {
            "config_id": settings.assistant_config,
            "model": config.model.name,
            "guardrail_type": config.guardrail.type,
            "model_name": "local",
            "model_alias": "dev",
            "model_version": "n/a",
        }
    pipeline = build_pipeline(config)
    return pipeline, labels


def _swap(app: FastAPI, pipeline: Pipeline, labels: dict[str, str]) -> None:
    """Atomic swap: clear the old assistant_info series, set the new, and
    reassign app.state.{pipeline, config_id, assistant_info_labels}. Each
    individual write is one bytecode op; concurrent /chat handlers that
    captured the old pipeline reference earlier finish against the old one,
    new requests pick up the new one."""
    old_labels: dict[str, str] | None = getattr(app.state, "assistant_info_labels", None)
    if old_labels:
        try:
            assistant_info.remove(*[old_labels[k] for k in _ASSISTANT_INFO_LABEL_ORDER])
        except KeyError:
            pass  # series already gone
    assistant_info.labels(**labels).set(1)
    app.state.pipeline = pipeline
    app.state.config_id = labels["config_id"]
    app.state.assistant_info_labels = labels


# --- Admin-token auth dependency --------------------------------------------

async def _require_admin_token(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    """If settings.admin_token is configured, require the X-Admin-Token header
    to match. If admin_token is unset (dev default), the endpoint is open."""
    settings = get_settings()
    if settings.admin_token is None:
        return
    if x_admin_token != settings.admin_token.get_secret_value():
        raise HTTPException(status_code=403, detail="invalid admin token")


# --- Lifecycle --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    pipeline, labels = await _resolve_and_build()
    _swap(app, pipeline, labels)
    log.info("startup: loaded config %s", labels)

    judge_sample_rate.set(settings.judge_sample_rate)

    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.judge_queue_maxsize)
    deep_judge_queue_depth.set(0)

    worker = JudgeWorker(queue)
    worker_task = asyncio.create_task(worker.run())

    app.state.judge_queue = queue
    app.state.judge_sample_rate = settings.judge_sample_rate

    try:
        yield
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


# --- App + schemas ----------------------------------------------------------

app = FastAPI(lifespan=lifespan, title="Travel assistant")
app.mount("/metrics", make_asgi_app())


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class ModelCallSchema(BaseModel):
    # `model` clashes with pydantic's protected namespace; opt out for this schema.
    model_config = ConfigDict(protected_namespaces=())

    model: str
    role: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class ChatResponse(BaseModel):
    text: str
    refused: bool
    input_category: str | None
    output_verdict: str | None
    model_calls: list[ModelCallSchema]


class ReloadResponse(BaseModel):
    status: str
    previous: dict[str, str]
    current: dict[str, str]


# --- Endpoints --------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/admin/reload",
    response_model=ReloadResponse,
    dependencies=[Depends(_require_admin_token)],
)
async def admin_reload() -> ReloadResponse:
    """Re-resolve current config (Registry alias if `ASSISTANT_MODEL_ALIAS` set,
    else local dev config) and swap the live pipeline atomically. On any
    failure during resolution/build, returns 500 with the existing pipeline
    intact."""
    try:
        pipeline, labels = await _resolve_and_build()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"reload failed: {type(exc).__name__}: {exc}",
        ) from exc

    previous: dict[str, str] = dict(
        getattr(app.state, "assistant_info_labels", {}) or {}
    )
    _swap(app, pipeline, labels)
    log.info("/admin/reload swapped: previous=%s current=%s", previous, labels)
    return ReloadResponse(status="ok", previous=previous, current=labels)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # Snapshot at request start so a concurrent /admin/reload that swaps
    # app.state.pipeline mid-request doesn't change the binding for this call.
    pipeline: Pipeline = app.state.pipeline
    config_id: str = app.state.config_id
    judge_queue: asyncio.Queue = app.state.judge_queue
    sample_rate: float = app.state.judge_sample_rate

    in_flight_requests.inc()
    start = time.perf_counter()
    try:
        try:
            response: AssistantResponse = await asyncio.to_thread(
                pipeline.respond, req.message
            )
        except Exception as exc:  # noqa: BLE001
            llm_api_errors_total.labels(
                config_id=config_id, error_type=type(exc).__name__
            ).inc()
            raise

        # Per-ModelCall cost increment (worked example). A /chat may produce
        # multiple model calls (v4 = 2: classifier + main; v5 = 3: classifier +
        # main + output validator); emit per call, not per request.
        for call in response.model_calls:
            chat_cost_usd_total.labels(
                config_id=config_id, model=call.model
            ).inc(cost_usd(call.model, call.input_tokens, call.output_tokens))
            chat_input_tokens.labels(
                config_id=config_id, model=call.model
            ).observe(call.input_tokens)
            chat_output_tokens.labels(
                config_id=config_id, model=call.model
            ).observe(call.output_tokens)

        chat_requests_total.labels(
            config_id=config_id,
            refused=str(response.refused).lower(),
            input_category=response.input_category or "unmonitored",
        ).inc()

        # Sample into the deep-judge queue (non-blocking).
        if random.random() < sample_rate:
            try:
                judge_queue.put_nowait((config_id, req.message, response.text))
                deep_judge_queue_depth.set(judge_queue.qsize())
            except asyncio.QueueFull:
                # Drop on overflow. Add a dedicated counter if drops become
                # operationally meaningful.
                pass

        return ChatResponse(
            text=response.text,
            refused=response.refused,
            input_category=response.input_category,
            output_verdict=response.output_verdict,
            model_calls=[
                ModelCallSchema(
                    model=c.model,
                    role=c.role,
                    input_tokens=c.input_tokens,
                    output_tokens=c.output_tokens,
                    latency_seconds=c.latency_seconds,
                )
                for c in response.model_calls
            ],
        )
    finally:
        chat_request_duration_seconds.labels(config_id=config_id).observe(
            time.perf_counter() - start
        )
        in_flight_requests.dec()
