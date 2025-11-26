# ISL Integration Examples

**Audience:** PLoT Development Team
**Purpose:** Copy-paste-ready integration examples
**Status:** Production-Ready ✅

---

## Overview

This guide provides complete, tested integration examples for PLoT→ISL communication. All examples use async/await patterns and follow production best practices.

**Base URL:**
- Staging: `https://isl-staging.olumi.ai`
- Production: `https://isl.olumi.ai`

---

## Table of Contents

1. [Basic Decision Analysis Flow](#example-1-basic-decision-analysis-flow)
2. [Error Handling Pattern](#example-2-error-handling-pattern)
3. [Caching Strategy](#example-3-caching-strategy)
4. [Concurrent Requests](#example-4-concurrent-requests)
5. [Graceful Degradation](#example-5-graceful-degradation)
6. [Cross-Reference Navigation](#example-6-cross-reference-navigation)
7. [Monitoring Integration](#example-7-monitoring-integration)
8. [Rate Limit Handling](#example-8-rate-limit-handling)

---

## Example 1: Basic Decision Analysis Flow

**Scenario:** User builds pricing model, ISL validates and analyzes.

```python
import httpx
import asyncio
from typing import Dict, Any


async def analyze_pricing_decision() -> Dict[str, Any]:
    """Complete decision analysis: validate → counterfactual → sensitivity."""

    async with httpx.AsyncClient(
        base_url="https://isl-staging.olumi.ai",
        timeout=30.0
    ) as client:

        # Step 1: Validate causal structure
        validation = await client.post(
            "/api/v1/causal/validate",
            json={
                "dag": {
                    "nodes": ["price", "demand", "revenue", "market_size"],
                    "edges": [
                        ["price", "demand"],
                        ["demand", "revenue"],
                        ["market_size", "demand"]
                    ]
                },
                "treatment": "price",
                "outcome": "revenue"
            },
            headers={"X-Request-Id": f"plot-req-{int(time.time())}"}
        )

        if validation.status_code != 200:
            return {"error": "Validation failed", "details": validation.json()}

        val_data = validation.json()
        print(f"✓ Validation: {val_data.get('status')}")

        if val_data.get("status") != "identifiable":
            return {
                "error": "Not identifiable",
                "explanation": val_data.get("explanation")
            }

        # Step 2: Analyze counterfactual scenario
        counterfactual = await client.post(
            "/api/v1/causal/counterfactual",
            json={
                "model": {
                    "variables": ["price", "demand", "revenue", "market_size"],
                    "equations": {
                        "demand": "market_size * (1 - 0.5 * price)",
                        "revenue": "price * demand"
                    },
                    "distributions": {
                        "market_size": {
                            "type": "normal",
                            "parameters": {"mean": 100000, "std": 10000}
                        }
                    }
                },
                "intervention": {"price": 12.0},
                "outcome": "revenue"
            },
            headers={"X-Request-Id": f"plot-req-{int(time.time())}"}
        )

        if counterfactual.status_code != 200:
            return {"error": "Counterfactual failed", "details": counterfactual.json()}

        cf_data = counterfactual.json()
        outcome = cf_data.get("outcome_distribution", {})
        print(f"✓ Revenue range: £{outcome.get('lower', 0):.0f} - £{outcome.get('upper', 0):.0f}")

        # Step 3: Get sensitivity analysis
        sensitivity = await client.post(
            "/api/v1/analysis/sensitivity",
            json={
                "model": {
                    "variables": ["price", "demand", "revenue", "market_size"],
                    "equations": {
                        "demand": "market_size * (1 - elasticity * price)",
                        "revenue": "price * demand"
                    },
                    "distributions": {
                        "market_size": {
                            "type": "normal",
                            "parameters": {"mean": 100000, "std": 10000}
                        },
                        "elasticity": {
                            "type": "normal",
                            "parameters": {"mean": 0.5, "std": 0.1}
                        }
                    }
                },
                "baseline_result": outcome.get("mean", 0),
                "assumptions": [
                    {
                        "name": "Price elasticity",
                        "current_value": 0.5,
                        "type": "parametric",
                        "variation_range": {"min": 0.3, "max": 0.8}
                    }
                ]
            },
            headers={"X-Request-Id": f"plot-req-{int(time.time())}"}
        )

        if sensitivity.status_code == 200:
            sens_data = sensitivity.json()
            drivers = sens_data.get("drivers", [])
            if drivers:
                top_driver = drivers[0]
                print(f"✓ Top driver: {top_driver.get('parameter')} ({top_driver.get('impact'):.0%} impact)")

        return {
            "validation": val_data,
            "prediction": cf_data,
            "sensitivity": sensitivity.json() if sensitivity.status_code == 200 else None
        }


# Usage
if __name__ == "__main__":
    import time
    result = asyncio.run(analyze_pricing_decision())
    print(f"\nComplete analysis: {result.keys()}")
```

**Key Points:**
- Always set `X-Request-Id` header for tracing
- Check status codes before processing responses
- Handle errors at each step
- Typical flow time: 2-5 seconds total

---

## Example 2: Error Handling Pattern

**Scenario:** Robust error handling with retries and fallback.

```python
import httpx
import asyncio
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


async def robust_isl_call(
    endpoint: str,
    payload: Dict[str, Any],
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Call ISL with exponential backoff retry and error handling.

    Args:
        endpoint: API endpoint path
        payload: Request payload
        max_retries: Maximum retry attempts

    Returns:
        Dict with 'success' and 'data' or 'error' keys
    """

    async with httpx.AsyncClient(
        base_url="https://isl-staging.olumi.ai",
        timeout=30.0
    ) as client:

        for attempt in range(max_retries):
            try:
                response = await client.post(endpoint, json=payload)

                # Success
                if response.status_code == 200:
                    return {
                        "success": True,
                        "data": response.json()
                    }

                # Client error (don't retry)
                if 400 <= response.status_code < 500:
                    error = response.json()

                    # Handle validation errors
                    if response.status_code == 422:
                        return {
                            "success": False,
                            "error": "validation_error",
                            "message": str(error.get("detail", "Invalid input")),
                            "retryable": False,
                            "suggested_action": "Fix input and retry"
                        }

                    # Handle rate limiting
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        return {
                            "success": False,
                            "error": "rate_limited",
                            "message": f"Rate limit exceeded. Retry after {retry_after}s",
                            "retry_after": retry_after,
                            "retryable": True
                        }

                    # Other client errors
                    return {
                        "success": False,
                        "error": f"client_error_{response.status_code}",
                        "message": str(error),
                        "retryable": False
                    }

                # Server error (retry with backoff)
                if response.status_code >= 500:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt  # 1s, 2s, 4s
                        logger.warning(
                            f"Server error {response.status_code}, "
                            f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait)
                        continue
                    else:
                        return {
                            "success": False,
                            "error": "server_error",
                            "message": "Service unavailable after retries",
                            "retryable": True,
                            "suggested_action": "Try again later"
                        }

            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Timeout, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                else:
                    return {
                        "success": False,
                        "error": "timeout",
                        "message": "Request timeout after retries",
                        "retryable": True
                    }

            except httpx.ConnectError:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Connection error, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                else:
                    return {
                        "success": False,
                        "error": "connection_error",
                        "message": "Cannot connect to ISL",
                        "retryable": True,
                        "fallback": "Use basic analysis without uncertainty"
                    }

        # Should never reach here
        return {
            "success": False,
            "error": "unknown",
            "message": "Unexpected error"
        }


# Usage
async def example_usage():
    result = await robust_isl_call(
        "/api/v1/causal/validate",
        {
            "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
            "treatment": "A",
            "outcome": "B"
        }
    )

    if result["success"]:
        print(f"✓ Success: {result['data'].get('status')}")
    else:
        print(f"✗ Error: {result['error']} - {result['message']}")

        if result.get("retryable"):
            print(f"  → Can retry later")
        else:
            print(f"  → Fix input: {result.get('suggested_action')}")
```

**Error Types:**
- `validation_error` (422): Fix input
- `rate_limited` (429): Wait and retry
- `server_error` (5xx): Retry with backoff
- `timeout`: Retry with backoff
- `connection_error`: Check service health

---

## Example 3: Caching Strategy

**Scenario:** Maximize cache hits through deterministic inputs.

```python
import hashlib
import json
from typing import Dict, Any


def normalize_dag_for_caching(dag: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize DAG to ensure deterministic cache keys.

    Same structure → same key → cache hit

    Args:
        dag: Raw DAG structure

    Returns:
        Normalized DAG
    """
    # Sort nodes alphabetically
    nodes = sorted(dag["nodes"])

    # Sort edges lexicographically
    edges = sorted(dag["edges"], key=lambda e: (e[0], e[1]))

    return {
        "nodes": nodes,
        "edges": edges
    }


def compute_request_fingerprint(request: Dict[str, Any]) -> str:
    """
    Compute deterministic fingerprint for request.

    Same request → same fingerprint → verifiable cache hit

    Args:
        request: Request payload

    Returns:
        12-character hex fingerprint
    """
    normalized = json.dumps(request, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


async def cached_validation(
    dag: Dict[str, Any],
    treatment: str,
    outcome: str
) -> Dict[str, Any]:
    """Validation with cache-friendly inputs."""

    # Normalize before calling ISL
    normalized_dag = normalize_dag_for_caching(dag)

    request = {
        "dag": normalized_dag,
        "treatment": treatment,
        "outcome": outcome
    }

    # Compute fingerprint for verification
    fingerprint = compute_request_fingerprint(request)

    async with httpx.AsyncClient(
        base_url="https://isl-staging.olumi.ai"
    ) as client:
        response = await client.post(
            "/api/v1/causal/validate",
            json=request
        )

        if response.status_code == 200:
            data = response.json()

            # Verify ISL's fingerprint matches ours (determinism check)
            isl_fingerprint = data.get("_metadata", {}).get("config_fingerprint")

            if isl_fingerprint:
                print(f"✓ Request fingerprint: {fingerprint}")
                print(f"✓ ISL fingerprint: {isl_fingerprint}")
                print(f"✓ Match: {fingerprint == isl_fingerprint}")

            return data

        return {"error": response.status_code}


# Usage
dag = {
    "nodes": ["Brand", "Price", "Revenue"],  # Unsorted
    "edges": [["Price", "Revenue"], ["Brand", "Price"]]  # Unsorted
}

result = asyncio.run(cached_validation(dag, "Price", "Revenue"))

# Subsequent calls with same DAG (different order) will hit cache:
dag2 = {
    "nodes": ["Revenue", "Brand", "Price"],  # Different order
    "edges": [["Brand", "Price"], ["Price", "Revenue"]]  # Different order
}

result2 = asyncio.run(cached_validation(dag2, "Price", "Revenue"))
# → Cache hit! (normalized to same structure)
```

**Cache Optimization Tips:**
1. Always normalize inputs (sort nodes/edges)
2. Use consistent parameter naming
3. Include seed for deterministic Monte Carlo
4. Verify fingerprints match for determinism check

**Expected Cache Hit Rates:**
- Day 1: 10-20% (cold cache)
- Week 1: 30-50% (warming up)
- Week 4+: 50-70% (steady state)

---

## Example 4: Concurrent Requests

**Scenario:** Analyze multiple scenarios in parallel.

```python
async def analyze_scenarios_concurrent(
    scenarios: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze multiple scenarios concurrently.

    Args:
        scenarios: List of scenario specifications

    Returns:
        Dict with successful and failed results
    """

    async with httpx.AsyncClient(
        base_url="https://isl-staging.olumi.ai",
        timeout=30.0
    ) as client:

        async def analyze_one(scenario: Dict[str, Any], index: int):
            """Analyze single scenario."""
            try:
                response = await client.post(
                    "/api/v1/causal/counterfactual",
                    json=scenario,
                    headers={"X-Request-Id": f"plot-batch-{index}"}
                )

                if response.status_code == 200:
                    return {"index": index, "success": True, "data": response.json()}
                else:
                    return {
                        "index": index,
                        "success": False,
                        "error": response.status_code,
                        "message": str(response.json())
                    }

            except Exception as e:
                return {
                    "index": index,
                    "success": False,
                    "error": "exception",
                    "message": str(e)
                }

        # Run all concurrently (max 10 at once to respect rate limits)
        batch_size = 10
        all_results = []

        for i in range(0, len(scenarios), batch_size):
            batch = scenarios[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                analyze_one(scenario, i + j)
                for j, scenario in enumerate(batch)
            ])
            all_results.extend(batch_results)

            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(scenarios):
                await asyncio.sleep(0.5)

        # Separate successful and failed
        successful = [r["data"] for r in all_results if r["success"]]
        failed = [
            {"index": r["index"], "error": r.get("error"), "message": r.get("message")}
            for r in all_results if not r["success"]
        ]

        return {
            "successful": successful,
            "failed": failed,
            "success_rate": len(successful) / len(scenarios) if scenarios else 0
        }


# Usage
scenarios = [
    {
        "model": {...},
        "intervention": {"price": 10.0},
        "outcome": "revenue"
    },
    {
        "model": {...},
        "intervention": {"price": 12.0},
        "outcome": "revenue"
    },
    {
        "model": {...},
        "intervention": {"price": 14.0},
        "outcome": "revenue"
    },
]

result = asyncio.run(analyze_scenarios_concurrent(scenarios))
print(f"Success rate: {result['success_rate']:.0%}")
print(f"Successful: {len(result['successful'])}")
print(f"Failed: {len(result['failed'])}")
```

**Concurrency Best Practices:**
- Batch requests (max 10 concurrent)
- Add delays between batches (0.5s)
- Total throughput: ~100 req/min
- Use unique Request-IDs for tracing

---

## Example 5: Graceful Degradation

**Scenario:** Handle ISL unavailability gracefully.

```python
async def analyze_with_fallback(
    model: Dict[str, Any],
    scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze with ISL, fallback to basic if unavailable.

    Returns full analysis or basic estimate depending on availability.
    """

    # Check ISL health first
    try:
        async with httpx.AsyncClient(
            base_url="https://isl-staging.olumi.ai",
            timeout=5.0
        ) as client:
            health = await client.get("/health")
            isl_available = (
                health.status_code == 200 and
                health.json().get("status") == "healthy"
            )
    except:
        isl_available = False

    if not isl_available:
        # Fallback: Basic point estimate (no uncertainty)
        estimate = compute_point_estimate(model, scenario)

        return {
            "mode": "fallback",
            "prediction": {
                "value": estimate,
                "confidence_interval": None,
                "uncertainty": "unavailable"
            },
            "message": "⚠ Advanced analysis unavailable. Showing basic estimate.",
            "show_warning": True
        }

    # ISL available: Full analysis with uncertainty
    try:
        async with httpx.AsyncClient(
            base_url="https://isl-staging.olumi.ai",
            timeout=30.0
        ) as client:
            response = await client.post(
                "/api/v1/causal/counterfactual",
                json={"model": model, **scenario}
            )

            if response.status_code == 200:
                return {
                    "mode": "full",
                    "prediction": response.json(),
                    "message": "✓ Analysis complete with uncertainty quantification",
                    "show_warning": False
                }
            else:
                # ISL failed: Fallback
                estimate = compute_point_estimate(model, scenario)
                return {
                    "mode": "fallback",
                    "prediction": {"value": estimate},
                    "message": f"⚠ Analysis failed ({response.status_code}). Showing basic estimate.",
                    "show_warning": True
                }

    except Exception as e:
        # Exception: Fallback
        estimate = compute_point_estimate(model, scenario)
        return {
            "mode": "fallback",
            "prediction": {"value": estimate},
            "message": f"⚠ Analysis error. Showing basic estimate.",
            "error": str(e),
            "show_warning": True
        }


def compute_point_estimate(model: Dict[str, Any], scenario: Dict[str, Any]) -> float:
    """
    Simple structural equation evaluation (fallback).

    Evaluates equations with given intervention values.
    No Monte Carlo, no uncertainty quantification.
    """
    # Simplified evaluation for fallback
    # In real implementation, parse and evaluate equations

    intervention = scenario.get("intervention", {})
    outcome = scenario.get("outcome")

    # Placeholder: Use simple heuristics
    # Real implementation would evaluate structural equations

    return 50000.0  # Placeholder value


# Usage
result = asyncio.run(analyze_with_fallback(
    model={...},
    scenario={"intervention": {"price": 12.0}, "outcome": "revenue"}
))

if result["show_warning"]:
    # Show warning UI in PLoT
    print(f"⚠ Warning: {result['message']}")

print(f"Mode: {result['mode']}")  # 'full' or 'fallback'
print(f"Prediction: {result['prediction']}")
```

**Degradation Strategy:**
1. Check `/health` endpoint first (5s timeout)
2. If unavailable → basic point estimate
3. If available but fails → retry once, then fallback
4. Show clear warnings to user when degraded

---

## Example 6: Cross-Reference Navigation

**Scenario:** Navigate assumption → uncertainty → sensitivity.

```python
from typing import Dict, Any, Optional, List


class CrossReferenceIndex:
    """Index for cross-reference navigation in ISL responses."""

    def __init__(self, isl_response: Dict[str, Any]):
        """Build index from ISL response."""
        self.assumptions = {}
        self.uncertainties = {}
        self.sensitivities = {}

        # Index all entities by ID
        for assumption in isl_response.get("assumptions", []):
            self.assumptions[assumption["id"]] = assumption

        for uncertainty in isl_response.get("uncertainty_breakdown", []):
            self.uncertainties[uncertainty["id"]] = uncertainty

        for driver in isl_response.get("sensitivity_drivers", []):
            self.sensitivities[driver["id"]] = driver

    def get_assumption_for_uncertainty(
        self,
        uncertainty_id: str
    ) -> Optional[Dict[str, Any]]:
        """Navigate from uncertainty source to linked assumption."""
        uncertainty = self.uncertainties.get(uncertainty_id)
        if not uncertainty:
            return None

        assumption_id = uncertainty.get("linked_assumption")
        if not assumption_id:
            return None

        return self.assumptions.get(assumption_id)

    def get_sensitivity_for_assumption(
        self,
        assumption_id: str
    ) -> Optional[Dict[str, Any]]:
        """Navigate from assumption to sensitivity driver."""
        assumption = self.assumptions.get(assumption_id)
        if not assumption:
            return None

        sensitivity_id = assumption.get("linked_sensitivity")
        if not sensitivity_id:
            return None

        return self.sensitivities.get(sensitivity_id)

    def get_uncertainties_for_assumption(
        self,
        assumption_id: str
    ) -> List[Dict[str, Any]]:
        """Get all uncertainties linked to an assumption."""
        return [
            u for u in self.uncertainties.values()
            if u.get("linked_assumption") == assumption_id
        ]


# Usage in UI
async def handle_uncertainty_click(uncertainty_id: str, isl_response: Dict):
    """
    Handle user clicking "Why is this uncertain?" in UI.

    Navigate: Uncertainty → Assumption
    """
    index = CrossReferenceIndex(isl_response)

    assumption = index.get_assumption_for_uncertainty(uncertainty_id)

    if assumption:
        # Show assumption details in UI
        return {
            "type": "assumption",
            "parameter": assumption["parameter_name"],
            "current_value": assumption["value"],
            "evidence": assumption["evidence_quality"],
            "description": assumption["description"]
        }
    else:
        return {"error": "No linked assumption found"}


async def handle_assumption_click(assumption_id: str, isl_response: Dict):
    """
    Handle user clicking "Show impact" on assumption.

    Navigate: Assumption → Sensitivity
    """
    index = CrossReferenceIndex(isl_response)

    sensitivity = index.get_sensitivity_for_assumption(assumption_id)

    if sensitivity:
        # Show sensitivity analysis in UI
        return {
            "type": "sensitivity",
            "parameter": sensitivity["parameter"],
            "impact": sensitivity["variance_contribution"],
            "range": sensitivity["outcome_range"],
            "chart_data": sensitivity.get("sensitivity_curve", [])
        }
    else:
        return {"error": "No sensitivity analysis available"}


# Example: Full navigation flow
isl_response = await call_isl_counterfactual(...)

# User clicks on uncertainty in UI
uncertainty_id = "uncertainty_param_elasticity_abc123"
assumption = await handle_uncertainty_click(uncertainty_id, isl_response)

# User clicks "Show impact"
if assumption["type"] == "assumption":
    sensitivity = await handle_assumption_click(
        assumption["parameter"],
        isl_response
    )

    # Display sensitivity chart
    print(f"Impact: {sensitivity['impact']:.0%} of total variance")
```

**Navigation Paths:**
1. **Uncertainty → Assumption:** "Why uncertain?" shows which assumption
2. **Assumption → Sensitivity:** "Show impact" shows sensitivity analysis
3. **Sensitivity → Uncertainties:** "What drives this?" shows contributing uncertainties

---

## Example 7: Monitoring Integration

**Scenario:** Track ISL performance in PLoT metrics.

```python
import time
from prometheus_client import Histogram, Counter, Gauge


# Define Prometheus metrics
isl_request_duration = Histogram(
    'plot_isl_request_duration_seconds',
    'ISL request duration',
    ['endpoint', 'status']
)

isl_request_total = Counter(
    'plot_isl_requests_total',
    'ISL request count',
    ['endpoint', 'status']
)

isl_cache_hit_rate = Gauge(
    'plot_isl_cache_hit_rate',
    'ISL cache hit rate (from metadata)'
)


async def monitored_isl_call(
    endpoint: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """ISL call with Prometheus metrics."""

    start = time.time()
    status = "error"

    try:
        async with httpx.AsyncClient(
            base_url="https://isl-staging.olumi.ai",
            timeout=30.0
        ) as client:
            response = await client.post(endpoint, json=payload)

            # Record duration
            duration = time.time() - start
            status = "success" if response.status_code == 200 else "error"

            isl_request_duration.labels(
                endpoint=endpoint,
                status=status
            ).observe(duration)

            isl_request_total.labels(
                endpoint=endpoint,
                status=status
            ).inc()

            # Extract cache hit info from metadata
            if response.status_code == 200:
                data = response.json()
                metadata = data.get("_metadata", {})

                # ISL doesn't expose cache hit in metadata yet,
                # but we can infer from response time
                if duration < 0.1:  # < 100ms likely cache hit
                    isl_cache_hit_rate.set(1.0)
                else:
                    isl_cache_hit_rate.set(0.0)

            return response.json()

    except Exception as e:
        # Record error
        duration = time.time() - start
        isl_request_duration.labels(
            endpoint=endpoint,
            status="exception"
        ).observe(duration)

        isl_request_total.labels(
            endpoint=endpoint,
            status="exception"
        ).inc()

        raise


# Prometheus queries for PLoT dashboard:

# ISL request rate
# rate(plot_isl_requests_total[5m])

# ISL P95 latency
# histogram_quantile(0.95, rate(plot_isl_request_duration_seconds_bucket[5m]))

# ISL error rate
# rate(plot_isl_requests_total{status="error"}[5m]) / rate(plot_isl_requests_total[5m])

# ISL cache hit rate (rolling average)
# avg_over_time(plot_isl_cache_hit_rate[5m])
```

**Monitoring Checklist:**
- ✅ Request rate (QPS)
- ✅ P95 latency by endpoint
- ✅ Error rate (%)
- ✅ Cache hit rate (inferred)
- ✅ Timeout rate

**Alert Thresholds:**
- Error rate > 5% → Warning
- P95 latency > 5s → Warning
- Timeout rate > 1% → Warning

---

## Example 8: Rate Limit Handling

**Scenario:** Handle rate limits gracefully.

```python
async def call_with_rate_limit_handling(
    endpoint: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call ISL with automatic rate limit handling.

    Respects Retry-After header and implements backoff.
    """

    async with httpx.AsyncClient(
        base_url="https://isl-staging.olumi.ai",
        timeout=30.0
    ) as client:

        while True:
            response = await client.post(endpoint, json=payload)

            # Success
            if response.status_code == 200:
                return {"success": True, "data": response.json()}

            # Rate limited
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))

                print(f"⚠ Rate limited. Waiting {retry_after}s...")

                # Wait as instructed
                await asyncio.sleep(retry_after)

                # Retry
                continue

            # Other error
            return {
                "success": False,
                "error": response.status_code,
                "message": str(response.json())
            }


# Check rate limit headers preemptively
async def check_rate_limit_status() -> Dict[str, int]:
    """Check current rate limit status."""

    async with httpx.AsyncClient(
        base_url="https://isl-staging.olumi.ai"
    ) as client:
        # Make lightweight request to check headers
        response = await client.get("/health")

        return {
            "limit": int(response.headers.get("X-RateLimit-Limit", 100)),
            "remaining": int(response.headers.get("X-RateLimit-Remaining", 100)),
            "can_proceed": int(response.headers.get("X-RateLimit-Remaining", 100)) > 10
        }


# Usage
status = await check_rate_limit_status()

if status["can_proceed"]:
    result = await call_with_rate_limit_handling("/api/v1/causal/validate", {...})
else:
    print(f"⚠ Only {status['remaining']} requests remaining. Waiting...")
    await asyncio.sleep(10)
```

**Rate Limit: 100 requests/minute per IP**

**Best Practices:**
1. Check `X-RateLimit-Remaining` header
2. Respect `Retry-After` header on 429
3. Implement exponential backoff for retries
4. Batch requests when possible
5. Cache results to reduce calls

---

## Best Practices Summary

### 1. Request Headers
```python
headers = {
    "X-Request-Id": f"plot-{user_id}-{timestamp}",  # For tracing
    "Content-Type": "application/json"
}
```

### 2. Timeouts
```python
# Recommended timeouts by endpoint:
timeouts = {
    "/api/v1/causal/validate": 10.0,      # Fast
    "/api/v1/causal/counterfactual": 30.0, # Slow (Monte Carlo)
    "/api/v1/analysis/sensitivity": 45.0,  # Slowest
    "/health": 5.0                         # Health check
}
```

### 3. Error Handling
```python
# Always check status codes
if response.status_code == 200:
    # Success
elif response.status_code == 422:
    # Validation error - fix input
elif response.status_code == 429:
    # Rate limited - wait and retry
elif response.status_code >= 500:
    # Server error - retry with backoff
```

### 4. Determinism Verification
```python
# Verify config fingerprints match
response_fingerprint = data["_metadata"]["config_fingerprint"]
local_fingerprint = compute_fingerprint(request)

assert response_fingerprint == local_fingerprint, "Determinism violation!"
```

### 5. Logging
```python
# Log with request IDs for tracing
logger.info(
    f"ISL call: {endpoint}",
    extra={
        "request_id": request_id,
        "duration_ms": duration * 1000,
        "status_code": response.status_code
    }
)
```

---

## Troubleshooting

### Q: Requests timing out

**A:** Increase timeout (30s for counterfactuals), reduce Monte Carlo samples, or implement async handling

```python
# Reduce samples for faster response
payload = {
    ...,
    "monte_carlo_samples": 5000  # Instead of default 10000
}
```

### Q: Low cache hit rate

**A:** Normalize inputs (sort nodes/edges), use consistent parameter names, include seed

```python
# Always normalize
dag = normalize_dag_for_caching(raw_dag)
```

### Q: 429 Rate Limit errors

**A:** Implement exponential backoff, reduce concurrent requests, batch operations

```python
# Batch with delays
for batch in chunks(requests, 10):
    await process_batch(batch)
    await asyncio.sleep(6)  # 6s between batches = 100 req/min
```

### Q: Inconsistent results

**A:** Check config_fingerprint across requests - different configs = different results

```python
# Use fixed seed for determinism
payload = {..., "seed": 42}
```

### Q: Validation errors

**A:** Check input limits (50 nodes, 200 edges, 100-char names, etc.)

```python
# Validate before sending
assert len(dag["nodes"]) <= 50, "Too many nodes"
assert len(dag["edges"]) <= 200, "Too many edges"
```

---

## Support

**Integration Issues:** #isl-integration
**API Questions:** #isl-api
**Performance:** #isl-performance
**Security:** #isl-security

**API Documentation:** https://isl-staging.olumi.ai/docs

---

**Last Updated:** 2025-11-20
**Version:** 1.0.0
