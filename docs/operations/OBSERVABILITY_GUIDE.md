# ISL Observability Guide

**Purpose:** Production observability and debugging workflows
**Audience:** Operations and Platform teams

---

## Structured Logging

**Format:** JSON (machine-readable)

**Example:**
```json
{
  "timestamp": "2025-11-20T10:30:15.123Z",
  "level": "INFO",
  "logger": "src.api.causal",
  "message": "Request completed",
  "request_id": "req_abc123",
  "endpoint": "/api/v1/causal/validate",
  "duration_ms": 4.52,
  "status_code": 200
}
```

**Query Logs (Kubernetes):**
```bash
# Get logs for specific request
kubectl logs -l app=isl | jq 'select(.request_id == "req_abc123")'

# Get all errors
kubectl logs -l app=isl | jq 'select(.level == "ERROR")'

# Get slow requests (>2s)
kubectl logs -l app=isl | jq 'select(.duration_ms > 2000)'
```

---

## Business Metrics

**Prometheus Metrics:**

```promql
# Assumptions validated by evidence quality
rate(isl_assumptions_validated_total{evidence_quality="high"}[1h])

# Model complexity distribution
histogram_quantile(0.95, rate(isl_model_complexity_bucket{metric="nodes"}[1h]))

# Active users
isl_active_users_current

# Cache determinism verification
rate(isl_cache_fingerprint_matches_total[5m])
```

---

## Debugging Workflows

### Slow Request Investigation

1. Find slow requests:
```bash
kubectl logs -l app=isl --since=1h | jq 'select(.duration_ms > 5000)'
```

2. Check operation breakdown:
```bash
kubectl logs -l app=isl | jq 'select(.request_id == "req_slow") | select(.operation)'
```

3. Identify bottleneck from operation timings

### High Error Rate

1. Get error distribution:
```bash
kubectl logs -l app=isl --since=1h | jq 'select(.level == "ERROR") | .error_type' | sort | uniq -c
```

2. Distinguish user vs system errors:
- 422 errors → User fixes needed
- 500 errors → ISL fixes needed

---

## Alerts

**Critical:**
- ISL Down (up{job="isl"} == 0)
- High Error Rate (>5%)

**Warning:**
- High Latency (P95 > 2.5s)
- Low Cache Hit Rate (<20%)

---

**For full details, see:** [PILOT_MONITORING_RUNBOOK.md](./PILOT_MONITORING_RUNBOOK.md)
