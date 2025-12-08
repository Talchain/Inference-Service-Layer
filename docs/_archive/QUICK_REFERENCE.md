# ISL API Quick Reference

**Quick lookup guide for PLoT developers**

---

## Base URLs

| Environment | URL | Purpose |
|-------------|-----|---------|
| Staging | `https://isl-staging.olumi.ai` | Testing, integration |
| Production | `https://isl.olumi.ai` | Live traffic |

---

## Endpoints

| Endpoint | Method | Purpose | Typical Latency |
|----------|--------|---------|-----------------|
| `/health` | GET | Service health check | ~2ms |
| `/metrics` | GET | Prometheus metrics | ~5ms |
| `/api/v1/causal/validate` | POST | Validate causal identifiability | ~10-50ms |
| `/api/v1/causal/counterfactual` | POST | Analyze scenarios with uncertainty | ~500ms-2s |
| `/api/v1/analysis/sensitivity` | POST | Sensitivity analysis | ~1-3s |
| `/api/v1/team/align` | POST | Multi-stakeholder alignment | ~200-500ms |

---

## Common Payloads

### Causal Validation

```json
{
  "dag": {
    "nodes": ["A", "B", "C"],
    "edges": [["A", "B"], ["B", "C"]]
  },
  "treatment": "A",
  "outcome": "C"
}
```

**Response:**
```json
{
  "status": "identifiable",
  "adjustment_sets": [["B"]],
  "explanation": {...},
  "_metadata": {"config_fingerprint": "a1b2c3d4e5f6", "request_id": "req_xyz"}
}
```

---

### Counterfactual Analysis

```json
{
  "model": {
    "variables": ["X", "Y"],
    "equations": {"Y": "10 + 2*X"},
    "distributions": {
      "X": {"type": "normal", "parameters": {"mean": 0, "std": 1}}
    }
  },
  "intervention": {"X": 1.5},
  "outcome": "Y"
}
```

**Response:**
```json
{
  "outcome_distribution": {
    "mean": 13.0,
    "lower": 12.5,
    "upper": 13.5,
    "p10": 12.6,
    "p50": 13.0,
    "p90": 13.4
  },
  "uncertainty_breakdown": [...],
  "_metadata": {...}
}
```

---

### Sensitivity Analysis

```json
{
  "model": {...},
  "baseline_result": 50000,
  "assumptions": [
    {
      "name": "Price elasticity",
      "current_value": 0.5,
      "type": "parametric",
      "variation_range": {"min": 0.3, "max": 0.8}
    }
  ]
}
```

**Response:**
```json
{
  "drivers": [
    {
      "parameter": "elasticity",
      "variance_contribution": 0.65,
      "impact": "high"
    }
  ],
  "_metadata": {...}
}
```

---

## HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 422 | Validation Error | Fix input, check limits |
| 429 | Rate Limited | Wait, use `Retry-After` header |
| 500 | Internal Error | Retry with exponential backoff |
| 503 | Service Unavailable | Check `/health`, use fallback |
| 504 | Timeout | Simplify model, retry |

---

## Error Response Format

All errors use structured `error.v1` schema:

```json
{
  "schema": "error.v1",
  "code": "INVALID_INPUT",
  "message": "DAG cannot exceed 50 nodes",
  "suggested_action": "reduce_model_size"
}
```

**Error Codes:**
- `INVALID_INPUT` - Fix input parameters
- `RATE_LIMIT_EXCEEDED` - Wait and retry
- `TIMEOUT` - Reduce complexity
- `INTERNAL_ERROR` - Retry or contact support

---

## Request Headers

| Header | Required | Purpose | Example |
|--------|----------|---------|---------|
| `Content-Type` | Yes | Request format | `application/json` |
| `X-Request-Id` | No (recommended) | Distributed tracing | `plot-req-12345` |

---

## Response Headers

| Header | Always Present | Purpose |
|--------|----------------|---------|
| `Content-Type` | Yes | Response format |
| `X-Request-Id` | Yes | Request ID (yours or generated) |
| `Retry-After` | If 429 | Seconds to wait before retry |
| `X-RateLimit-Limit` | If implemented | Requests per minute limit |
| `X-RateLimit-Remaining` | If implemented | Remaining requests |

---

## Response Metadata

Every successful response includes `_metadata`:

```json
{
  "_metadata": {
    "isl_version": "1.0.0",
    "config_fingerprint": "a1b2c3d4e5f6",
    "request_id": "req_abc123",
    "computed_at": "2025-11-20T10:30:00Z"
  }
}
```

**Fields:**
- `isl_version` - ISL version for compatibility
- `config_fingerprint` - Deterministic hash (verify matches your request)
- `request_id` - For tracing and debugging
- `computed_at` - UTC timestamp

---

## Input Validation Limits

### DAG Limits

| Field | Limit | Reason |
|-------|-------|--------|
| Nodes | 50 max | Performance, complexity management |
| Edges | 200 max | Performance, complexity management |
| Node name length | 100 chars | Memory, readability |

### String Limits

| Field | Limit |
|-------|-------|
| Variable names | 100 chars |
| Descriptions | 10,000 chars |
| Equations | 1,000 chars |
| User IDs | 100 chars (alphanumeric + underscore/hyphen only) |

### List Limits

| Field | Limit |
|-------|-------|
| Team perspectives | 20 max |
| Decision options | 50 max |
| Assumptions | 30 max |
| Priorities/constraints | 20 max each |

### Numeric Limits

| Field | Range |
|-------|-------|
| Monte Carlo samples | 1,000 - 100,000 |
| Confidence | 0.0 - 1.0 |
| Weights | -1,000 - 1,000 |

---

## Rate Limiting

**Limit:** 100 requests per minute per IP address

**Response when exceeded:**
```json
{
  "schema": "error.v1",
  "code": "RATE_LIMIT_EXCEEDED",
  "message": "Too many requests. Please wait before trying again.",
  "retry_after": 45
}
```

**Headers:**
- `Retry-After: 45` (seconds to wait)
- `X-RateLimit-Limit: 100`
- `X-RateLimit-Remaining: 0`

**Best Practices:**
1. Check `X-RateLimit-Remaining` before bursts
2. Respect `Retry-After` header
3. Batch requests when possible
4. Cache results to reduce calls

---

## Recommended Timeouts

```python
timeouts = {
    "health_check": 5.0,        # Quick
    "validate": 10.0,           # Fast validation
    "counterfactual": 30.0,     # Monte Carlo sampling
    "sensitivity": 45.0,        # Multiple scenarios
    "default": 30.0             # Safe default
}
```

---

## Performance Tips

### 1. Normalize Inputs for Caching

```python
# Sort nodes and edges for deterministic cache keys
dag = {
    "nodes": sorted(raw_dag["nodes"]),
    "edges": sorted(raw_dag["edges"])
}
```

### 2. Use Appropriate Sample Sizes

```python
# Trade-off: Accuracy vs Speed
samples = {
    "quick_preview": 1000,      # ~100ms
    "standard": 10000,          # ~1s (default)
    "high_accuracy": 50000      # ~5s
}
```

### 3. Batch Concurrent Requests

```python
# Max 10 concurrent to respect rate limits
batch_size = 10
for batch in chunks(requests, batch_size):
    await asyncio.gather(*[process(r) for r in batch])
    await asyncio.sleep(6)  # 6s delay = 100 req/min
```

### 4. Implement Fallback

```python
# Check health first
healthy = await check_isl_health()

if not healthy:
    return basic_fallback_analysis()
```

---

## Determinism Verification

**ISL guarantees:** Identical inputs → Identical outputs

**Verify:**
```python
# 1. Compute your request fingerprint
local_fp = compute_fingerprint(request)

# 2. Check ISL's fingerprint in response
isl_fp = response["_metadata"]["config_fingerprint"]

# 3. Verify match
assert local_fp == isl_fp, "Determinism violation!"
```

**Fingerprint Algorithm:**
```python
import hashlib
import json

def compute_fingerprint(request):
    normalized = json.dumps(request, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]
```

---

## Common Error Scenarios

### "DAG cannot exceed 50 nodes"

**Cause:** Too many nodes in DAG
**Fix:** Simplify model, combine related variables

### "Self-loops not allowed"

**Cause:** Edge where source == target (e.g., `["A", "A"]`)
**Fix:** Remove self-loops, DAGs must be acyclic

### "Equation contains unsafe characters"

**Cause:** Non-alphanumeric characters in equation
**Fix:** Use only `a-zA-Z0-9_+-*/() .` in equations

### "Rate limit exceeded"

**Cause:** More than 100 requests per minute
**Fix:** Wait for `Retry-After` seconds, implement backoff

### "Monte Carlo samples must be >= 1000"

**Cause:** Too few samples for accuracy
**Fix:** Use at least 1000 samples (default: 10,000)

---

## Health Check

```bash
curl https://isl-staging.olumi.ai/health
```

**Healthy Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "redis": {
    "connected": true
  }
}
```

**Degraded Response:**
```json
{
  "status": "degraded",
  "version": "1.0.0",
  "redis": {
    "connected": false
  },
  "message": "Redis unavailable, operating with in-memory cache"
}
```

---

## Monitoring Queries

### Prometheus

```promql
# Request rate
rate(isl_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(isl_request_duration_seconds_bucket[5m]))

# Error rate
rate(isl_requests_total{status="error"}[5m]) / rate(isl_requests_total[5m])

# Cache hit rate
rate(isl_cache_hits_total[1h]) / rate(isl_cache_requests_total[1h])
```

---

## Support Channels

| Issue Type | Channel |
|------------|---------|
| Integration help | `#isl-integration` |
| API questions | `#isl-api` |
| Performance issues | `#isl-performance` |
| Security concerns | `#isl-security` |
| Bugs/outages | `#isl-incidents` |

---

## Useful Links

- **API Docs:** https://isl-staging.olumi.ai/docs
- **Integration Examples:** [INTEGRATION_EXAMPLES.md](./INTEGRATION_EXAMPLES.md)
- **Cross-Reference Schema:** [CROSS_REFERENCE_SCHEMA.md](./CROSS_REFERENCE_SCHEMA.md)
- **Operations Runbook:** [../operations/PILOT_MONITORING_RUNBOOK.md](../operations/PILOT_MONITORING_RUNBOOK.md)

---

**Last Updated:** 2025-11-20
**Version:** 1.0.0
