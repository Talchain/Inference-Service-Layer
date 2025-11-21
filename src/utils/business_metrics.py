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

# FACET robustness metrics
facet_analyses_total = Counter(
    'isl_facet_analyses_total',
    'Total robustness analyses performed',
    ['status']  # robust, fragile, failed
)

facet_robustness_score = Histogram(
    'isl_facet_robustness_score',
    'Robustness scores distribution',
    buckets=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
)

facet_fragile_recommendations_total = Counter(
    'isl_facet_fragile_recommendations_total',
    'Fragile recommendations flagged'
)

facet_robust_regions_found = Histogram(
    'isl_facet_robust_regions_found',
    'Number of robust regions per analysis',
    buckets=[0, 1, 2, 3, 5, 10]
)

# Habermas Machine deliberation metrics
habermas_deliberations_total = Counter(
    'isl_habermas_deliberations_total',
    'Total deliberation rounds conducted',
    ['status']  # active, converged, diverged
)

habermas_rounds_per_session = Histogram(
    'isl_habermas_rounds_per_session',
    'Number of rounds per deliberation session',
    buckets=[1, 2, 3, 5, 7, 10, 15]
)

habermas_agreement_level = Histogram(
    'isl_habermas_agreement_level',
    'Agreement levels achieved',
    buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

habermas_convergence_total = Counter(
    'isl_habermas_convergence_total',
    'Successful convergences achieved'
)

# LLM usage and cost metrics (Phase 4A)
llm_requests_total = Counter(
    'isl_llm_requests_total',
    'Total LLM requests made',
    ['model', 'endpoint', 'cached']  # Track by model, endpoint, and cache status
)

llm_cost_dollars = Histogram(
    'isl_llm_cost_dollars',
    'LLM cost per request in dollars',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

llm_cache_hits_total = Counter(
    'isl_llm_cache_hits_total',
    'Total LLM cache hits'
)

llm_tokens_total = Counter(
    'isl_llm_tokens_total',
    'Total LLM tokens used',
    ['type', 'model']  # input/output, model name
)

llm_budget_exceeded_total = Counter(
    'isl_llm_budget_exceeded_total',
    'Times session budget was exceeded',
    ['session_type']  # deliberation, preference, etc.
)

llm_fallback_to_rules_total = Counter(
    'isl_llm_fallback_to_rules_total',
    'Times LLM failed and fell back to rules',
    ['reason']  # error, budget, timeout
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


def track_robustness_analysis(
    status: str,
    robustness_score: float,
    is_fragile: bool,
    regions_found: int,
) -> None:
    """
    Track FACET robustness analysis metrics.

    Args:
        status: Analysis status (robust, fragile, failed)
        robustness_score: Overall robustness score (0-1)
        is_fragile: Whether recommendation is fragile
        regions_found: Number of robust regions found
    """
    facet_analyses_total.labels(status=status).inc()
    facet_robustness_score.observe(robustness_score)

    if is_fragile:
        facet_fragile_recommendations_total.inc()

    facet_robust_regions_found.observe(regions_found)


def track_habermas_deliberation(
    status: str,
    agreement_level: float,
    round_number: int,
    converged: bool,
) -> None:
    """
    Track Habermas Machine deliberation metrics.

    Args:
        status: Deliberation status (active, converged, diverged)
        agreement_level: Agreement level achieved (0-1)
        round_number: Current round number
        converged: Whether convergence was reached
    """
    habermas_deliberations_total.labels(status=status).inc()
    habermas_agreement_level.observe(agreement_level)

    if converged:
        habermas_convergence_total.inc()
        habermas_rounds_per_session.observe(round_number)


def track_llm_request(
    model: str,
    endpoint: str,
    cost: float,
    input_tokens: int,
    output_tokens: int,
    cached: bool,
) -> None:
    """
    Track LLM request metrics.

    Args:
        model: LLM model name (e.g., gpt-4, gpt-3.5-turbo)
        endpoint: Endpoint that made the request
        cost: Cost in dollars
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached: Whether response was cached
    """
    cached_str = "true" if cached else "false"
    llm_requests_total.labels(model=model, endpoint=endpoint, cached=cached_str).inc()
    llm_cost_dollars.observe(cost)
    llm_tokens_total.labels(type="input", model=model).inc(input_tokens)
    llm_tokens_total.labels(type="output", model=model).inc(output_tokens)

    if cached:
        llm_cache_hits_total.inc()


def track_llm_budget_exceeded(session_type: str) -> None:
    """Track LLM budget exceeded event."""
    llm_budget_exceeded_total.labels(session_type=session_type).inc()


def track_llm_fallback(reason: str) -> None:
    """Track LLM fallback to rule-based system."""
    llm_fallback_to_rules_total.labels(reason=reason).inc()
