# ISL Observability Guide

## Overview

ISL provides comprehensive observability through:
- **Request tracing** — Unique IDs for every request
- **Structured logging** — JSON logs with correlation
- **Metrics** — Prometheus endpoint
- **Error tracking** — Sentry integration

## Request Tracing

### Request ID Generation

Every request receives a unique identifier in the format:
```
req_{16 alphanumeric characters}
```

Examples:
- `req_a1b2c3d4e5f6g7h8`
- `req_xyz123abc456def0`

The format `req_{uuid16}` aligns with the Olumi platform standard.

### Header Flow

```
Client Request                    ISL                         Response
─────────────────────────────────────────────────────────────────────────
X-Request-Id: req_abc123  ──►  Accepts provided ID  ──►  X-Request-Id: req_abc123
(or no header)            ──►  Generates new ID     ──►  X-Request-Id: req_xyz789
X-Trace-Id: trace_123     ──►  Maps to X-Request-Id ──►  X-Request-Id: trace_123
```

### Header Priority

When multiple headers are present, ISL uses this priority:
1. `X-Request-Id` (platform standard)
2. `X-Trace-Id` (legacy, deprecated)
3. Auto-generated if neither provided

### Headers Reference

| Header | Direction | Required | Description |
|--------|-----------|----------|-------------|
| `X-Request-Id` | Request/Response | No | Primary request identifier |
| `X-Trace-Id` | Request/Response | No | Legacy identifier (deprecated, maps to X-Request-Id) |
| `X-User-Id` | Request | No | Optional user context for audit logs |
| `X-API-Key` | Request | Yes* | API authentication key |
| `X-RateLimit-Limit` | Response | — | Rate limit ceiling for endpoint |
| `X-RateLimit-Remaining` | Response | — | Remaining requests in current window |
| `Retry-After` | Response | — | Seconds to wait when rate limited (429) |

*Required unless `ISL_AUTH_DISABLED=true`

### Using Request IDs for Debugging

**Client-side:** Store the `X-Request-Id` from responses
```python
import requests

response = requests.post(
    'https://isl.example.com/api/v1/causal/validate',
    json={'dag': {...}},
    headers={'X-API-Key': 'your-key'}
)

request_id = response.headers.get('X-Request-Id')
print(f"Request ID: {request_id}")  # Save for debugging

if response.status_code != 200:
    print(f"Error - report this ID: {request_id}")
```

**When reporting issues:** Include the request ID

**Server-side:** Search logs by request ID
```bash
grep "req_a1b2c3d4e5f6g7h8" /var/log/isl/*.log
```

**Sentry:** Filter events by request_id tag
```
tags[request_id]:req_a1b2c3d4e5f6g7h8
```

## Structured Logging

### Log Format

All logs are JSON-formatted for easy parsing:
```json
{
  "timestamp": "2024-12-11T10:30:00.123Z",
  "level": "INFO",
  "logger": "isl.api",
  "request_id": "req_a1b2c3d4e5f6g7h8",
  "message": "Request completed",
  "extra": {
    "method": "POST",
    "path": "/api/v1/causal/validate",
    "status_code": 200,
    "duration_ms": 45
  }
}
```

### Log Levels

| Level | Usage |
|-------|-------|
| DEBUG | Detailed debugging (disabled in production) |
| INFO | Normal operations, request lifecycle |
| WARNING | Recoverable issues, deprecations |
| ERROR | Failures requiring attention |
| CRITICAL | System-level failures |

### Configuration

```bash
# Set log level
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Enable JSON formatting (default in production)
LOG_FORMAT=json  # or "text" for development
```

## Prometheus Metrics

### Endpoint

```
GET /metrics
```

No authentication required. Returns Prometheus-formatted metrics.

### Available Metrics

#### Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_requests_total` | Counter | method, endpoint, status | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | method, endpoint | Request latency |

#### Rate Limiting Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `isl_rate_limit_hits_total` | Counter | identifier_type | Rate limit rejections |
| `isl_rate_limit_checks_total` | Counter | result | All rate limit checks |

### Example Prometheus Queries

```promql
# Request rate by endpoint
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Rate limit rejection rate
rate(isl_rate_limit_hits_total[5m])
```

### Grafana Dashboard

Import the ISL dashboard from `docs/grafana/isl-dashboard.json` (if available) or create panels using the queries above.

## Sentry Integration

### Configuration

```bash
SENTRY_ENABLED=true
SENTRY_DSN=https://xxx@sentry.io/xxx
SENTRY_ENVIRONMENT=production          # or staging, development
SENTRY_TRACES_SAMPLE_RATE=0.1          # 10% of transactions traced
SENTRY_PROFILES_SAMPLE_RATE=0.1        # 10% of traces profiled
```

### Context Attached to Events

Every Sentry event includes:

| Tag/Context | Value |
|-------------|-------|
| `request_id` | The unique request identifier |
| `environment` | production/staging/development |
| `path` | Request URL path |
| `method` | HTTP method |

### Before-Send Filtering

ISL filters sensitive data before sending to Sentry:
- Authorization headers are removed
- API keys are masked
- Request bodies are truncated to 10KB

### Searching Sentry

Find events by request ID:
```
tags[request_id]:req_a1b2c3d4e5f6g7h8
```

Find events by environment:
```
environment:production
```

## Tracing Middleware Implementation

The `TracingMiddleware` handles request ID propagation:

```python
# Priority: X-Request-Id > X-Trace-Id > generated
request_id = (
    request.headers.get('X-Request-Id') or
    request.headers.get('X-Trace-Id') or
    generate_trace_id()  # req_{uuid16}
)

# Both headers returned for backward compatibility
response.headers['X-Request-Id'] = request_id
response.headers['X-Trace-Id'] = request_id  # Deprecated
```

## Health Checks

### Endpoints

| Endpoint | Purpose | Auth Required |
|----------|---------|---------------|
| `/health` | Basic health check | No |
| `/health/ready` | Readiness probe (K8s) | No |
| `/health/live` | Liveness probe (K8s) | No |

### Response Format

```json
{
  "status": "healthy",
  "version": "2.1.0",
  "timestamp": "2024-12-11T10:30:00Z"
}
```

## Troubleshooting

### Common Issues

**Missing request ID in response**
- Check that `TracingMiddleware` is registered in middleware stack
- Verify middleware order (tracing should be early in chain)

**Logs not appearing**
- Check `LOG_LEVEL` configuration
- Verify structured logging is enabled
- Check log output destination

**Metrics endpoint returns 404**
- Ensure `/metrics` route is registered
- Check if metrics collection is enabled

**Sentry events not appearing**
- Verify `SENTRY_ENABLED=true`
- Check `SENTRY_DSN` is correct
- Confirm network connectivity to Sentry
- Check sample rates aren't set to 0

**Rate limit headers missing**
- Rate limit middleware may be disabled
- Check `RATE_LIMITING_ENABLED` setting

### Debug Checklist

1. Capture the `X-Request-Id` from response
2. Search logs for that request ID
3. Check Sentry for matching events
4. Review metrics for error patterns
5. Verify middleware chain in startup logs
