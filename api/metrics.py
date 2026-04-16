"""Prometheus metrics for DigitalTwinSim."""

import os

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

# Custom metrics
simulations_total = Counter(
    "dts_simulations_total",
    "Total simulations launched",
    ["tenant_id", "domain", "status"],
)

llm_cost_total = Counter(
    "dts_llm_cost_usd_total",
    "Total LLM cost in USD",
    ["tenant_id", "provider"],
)

active_simulations = Gauge(
    "dts_active_simulations",
    "Currently running simulations",
)

api_request_duration = Histogram(
    "dts_api_request_duration_seconds",
    "API request duration",
    ["method", "endpoint", "status"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


def setup_metrics(app):
    """Attach Prometheus instrumentator to FastAPI app."""
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/api/health"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
