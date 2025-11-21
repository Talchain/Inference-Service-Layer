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

# ActiVA preference learning metrics
activa_queries_generated_total = Counter(
    'isl_activa_queries_generated_total',
    'Total preference queries generated'
)

activa_queries_answered_total = Counter(
    'isl_activa_queries_answered_total',
    'Total preference queries answered',
    ['choice']  # A or B
)

activa_convergence_total = Counter(
    'isl_activa_convergence_total',
    'Users reached convergence',
    ['num_queries_bucket']  # 1-5, 6-10, 11+
)

activa_information_gain = Histogram(
    'isl_activa_information_gain',
    'Expected information gain of generated queries',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
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


def track_activa_query_generated() -> None:
    """Track ActiVA preference query generation."""
    activa_queries_generated_total.inc()


def track_activa_query_answered(choice: str) -> None:
    """Track ActiVA preference query answered."""
    activa_queries_answered_total.labels(choice=choice).inc()


def track_activa_convergence(num_queries: int) -> None:
    """Track ActiVA preference learning convergence."""
    if num_queries <= 5:
        bucket = "1-5"
    elif num_queries <= 10:
        bucket = "6-10"
    else:
        bucket = "11+"
    activa_convergence_total.labels(num_queries_bucket=bucket).inc()


def track_activa_information_gain(info_gain: float) -> None:
    """Track ActiVA expected information gain."""
    activa_information_gain.observe(info_gain)
