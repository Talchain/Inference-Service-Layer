"""
Business-level metrics for ISL observability.

Tracks business KPIs beyond technical metrics.
"""

from prometheus_client import Counter, Histogram, Gauge


# Business metrics
assumptions_validated_total = Counter(
    'isl_assumptions_validated_total',
    'Total assumptions validated',
    ['evidence_quality']
)

models_analyzed_total = Counter(
    'isl_models_analyzed_total',
    'Total models analyzed',
    ['analysis_type']
)

model_complexity = Histogram(
    'isl_model_complexity',
    'Distribution of model complexity',
    ['metric'],
    buckets=[5, 10, 15, 20, 30, 40, 50]
)

active_users = Gauge(
    'isl_active_users_current',
    'Current number of active users'
)

cache_fingerprint_matches = Counter(
    'isl_cache_fingerprint_matches_total',
    'Determinism verifications (fingerprint matches)'
)


def track_assumption_validated(evidence_quality: str) -> None:
    """Track assumption validation by evidence quality."""
    assumptions_validated_total.labels(evidence_quality=evidence_quality).inc()


def track_model_analyzed(analysis_type: str) -> None:
    """Track model analysis by type."""
    models_analyzed_total.labels(analysis_type=analysis_type).inc()


def track_model_complexity(node_count: int, edge_count: int) -> None:
    """Track model complexity distribution."""
    model_complexity.labels(metric="nodes").observe(node_count)
    model_complexity.labels(metric="edges").observe(edge_count)


def track_cache_fingerprint_match() -> None:
    """Track determinism verification (fingerprint match)."""
    cache_fingerprint_matches.inc()
