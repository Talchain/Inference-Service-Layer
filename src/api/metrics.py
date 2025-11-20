"""
Prometheus metrics endpoint and instrumentation.

Exposes key application metrics for monitoring:
- Request counts and latencies
- Error rates
- Service health
- Business metrics (queries generated, beliefs updated, etc.)
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import APIRouter, Response

router = APIRouter()

# Request metrics
http_requests_total = Counter(
    "isl_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "isl_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)

# Error metrics
http_errors_total = Counter(
    "isl_http_errors_total",
    "Total HTTP errors",
    ["method", "endpoint", "error_code"],
)

# Business metrics - Preference Learning
preference_queries_generated = Counter(
    "isl_preference_queries_generated_total",
    "Total preference queries generated",
    ["strategy"],
)

belief_updates_total = Counter(
    "isl_belief_updates_total",
    "Total belief updates performed",
    ["user_id_hash"],
)

# Business metrics - Causal Inference
causal_validations_total = Counter(
    "isl_causal_validations_total",
    "Total causal model validations",
    ["status"],
)

counterfactual_analyses_total = Counter(
    "isl_counterfactual_analyses_total",
    "Total counterfactual analyses performed",
)

# Business metrics - Teaching
teaching_examples_generated = Counter(
    "isl_teaching_examples_generated_total",
    "Total teaching examples generated",
    ["concept"],
)

# Business metrics - Validation
model_validations_total = Counter(
    "isl_model_validations_total",
    "Total model validations performed",
    ["quality_level"],
)

# Cache metrics
redis_operations_total = Counter(
    "isl_redis_operations_total",
    "Total Redis operations",
    ["operation", "status"],
)

cache_hits_total = Counter(
    "isl_cache_hits_total",
    "Total cache hits",
)

cache_misses_total = Counter(
    "isl_cache_misses_total",
    "Total cache misses",
)

# Service health metrics
service_up = Gauge(
    "isl_service_up",
    "Service is up (1) or down (0)",
)

redis_connected = Gauge(
    "isl_redis_connected",
    "Redis is connected (1) or disconnected (0)",
)

active_requests = Gauge(
    "isl_active_requests",
    "Number of currently active requests",
)

# Computation metrics
activa_computation_duration_seconds = Histogram(
    "isl_activa_computation_duration_seconds",
    "ActiVA algorithm computation time",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
)

bayesian_update_duration_seconds = Histogram(
    "isl_bayesian_update_duration_seconds",
    "Bayesian belief update computation time",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0),
)

counterfactual_computation_duration_seconds = Histogram(
    "isl_counterfactual_computation_duration_seconds",
    "Counterfactual analysis computation time",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """
    Expose Prometheus metrics.

    Returns metrics in Prometheus text format for scraping.
    This endpoint is not included in the OpenAPI schema.
    """
    # Set service up gauge
    service_up.set(1)

    # Generate and return metrics
    metrics_output = generate_latest()
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST,
    )
