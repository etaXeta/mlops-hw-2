"""Prometheus metric definitions.

All metrics live on the default registry, defined at module-import time.
Importing this module is idempotent; instantiation happens once.

Three signal classes (see STUDENT_README.md):
- Cheap signals: emitted from /chat on every request.
- State signals (gauges): point-in-time view of the service.
- Sampled signals: emitted by the async judge worker.

NOTE FOR STUDENTS (Task 4): several metric definitions have been removed —
you'll find them as `# TODO (Task 4)` comments below. Define each Counter
or Histogram following the existing `chat_requests_total` example, then
wire them up at the right call sites in src/assistant/service.py and
src/monitoring/judge_worker.py (also marked with TODO comments there).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- Cheap signals -----------------------------------------------------------

chat_requests_total = Counter(
    "chat_requests_total",
    "Total /chat invocations that returned a response (not counting LLM errors).",
    ["config_id", "refused", "input_category"],
)

llm_api_errors_total = Counter(
    "llm_api_errors_total",
    "Errors raised by the LLM client during a /chat invocation.",
    ["config_id", "error_type"],
)

chat_cost_usd_total = Counter(
    "chat_cost_usd_total",
    "Cumulative USD cost of /chat completions, sliced by model.",
    ["config_id", "model"],
)

chat_request_duration_seconds = Histogram(
    "chat_request_duration_seconds",
    "End-to-end latency of /chat requests.",
    ["config_id"],
    buckets=(
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        float("inf"),
    ),
)

chat_input_tokens = Histogram(
    "chat_input_tokens",
    "Number of input tokens per model call in /chat.",
    ["config_id", "model"],
)

chat_output_tokens = Histogram(
    "chat_output_tokens",
    "Number of output tokens per model call in /chat.",
    ["config_id", "model"],
)

# --- State signals (gauges) --------------------------------------------------

in_flight_requests = Gauge(
    "in_flight_requests",
    "Number of /chat requests currently being processed.",
)

deep_judge_queue_depth = Gauge(
    "deep_judge_queue_depth",
    "Pending (input, response) pairs waiting for sampled judge evaluation.",
)

judge_sample_rate = Gauge(
    "judge_sample_rate",
    "Configured fraction of /chat traffic forwarded to the deep judge.",
)

assistant_info = Gauge(
    "assistant_info",
    "Info metric carrying deployment identity in labels. Always 1.",
    [
        "config_id",
        "model",
        "guardrail_type",
        "model_name",       # MLflow registered model name; "local" in dev mode
        "model_alias",      # which Registry alias the service resolved; "dev" in dev mode
        "model_version",    # registered version number; "n/a" in dev mode
    ],
)

# --- Sampled signals (async worker emits) ------------------------------------

judge_evaluations_total = Counter(
    "judge_evaluations_total",
    "Total deep judge evaluations, sliced by verdict.",
    ["config_id", "verdict"],
)

judge_latency_seconds = Histogram(
    "judge_latency_seconds",
    "Latency of judge evaluations.",
    ["config_id"],
    buckets=(
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        float("inf"),
    ),
)
