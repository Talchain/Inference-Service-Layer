# Service Level Objectives (SLOs) and Indicators (SLIs)

## Overview

This document defines the Service Level Objectives (SLOs) and Service Level Indicators (SLIs) for the Inference Service Layer (ISL). These metrics help ensure the service meets reliability and security requirements.

## Definitions

- **SLI (Service Level Indicator):** A quantitative measure of service behavior
- **SLO (Service Level Objective):** A target value or range for an SLI
- **Error Budget:** The allowed amount of unreliability (100% - SLO)

---

## Core SLIs and SLOs

### 1. Availability

**SLI Definition:**
```
Availability = (Successful Requests / Total Requests) × 100%

Where:
- Successful = HTTP status codes 2xx, 3xx, 4xx (client errors excluded from failures)
- Failed = HTTP status codes 5xx
```

**Prometheus Query:**
```promql
(
  sum(rate(isl_http_requests_total{status!~"5.."}[5m]))
  /
  sum(rate(isl_http_requests_total[5m]))
) * 100
```

**SLO Target:** 99.9% (Three nines)

**Error Budget:** 0.1% = 43.2 minutes/month of downtime

| Time Period | Max Downtime |
|-------------|--------------|
| Daily       | 1.44 minutes |
| Weekly      | 10.08 minutes |
| Monthly     | 43.2 minutes |
| Quarterly   | 2.16 hours |

---

### 2. Latency

**SLI Definition:**
```
Latency SLI = Percentage of requests completing within threshold

- p50 (median): 50th percentile response time
- p95: 95th percentile response time
- p99: 99th percentile response time
```

**Prometheus Queries:**
```promql
# p50 latency
histogram_quantile(0.50, sum(rate(isl_http_request_duration_seconds_bucket[5m])) by (le))

# p95 latency
histogram_quantile(0.95, sum(rate(isl_http_request_duration_seconds_bucket[5m])) by (le))

# p99 latency
histogram_quantile(0.99, sum(rate(isl_http_request_duration_seconds_bucket[5m])) by (le))

# Percentage of requests under 500ms
(
  sum(rate(isl_http_request_duration_seconds_bucket{le="0.5"}[5m]))
  /
  sum(rate(isl_http_request_duration_seconds_count[5m]))
) * 100
```

**SLO Targets:**

| Metric | Target | Threshold |
|--------|--------|-----------|
| p50 | 99% of requests | < 200ms |
| p95 | 95% of requests | < 500ms |
| p99 | 90% of requests | < 1000ms |

---

### 3. Error Rate

**SLI Definition:**
```
Error Rate = (5xx Responses / Total Responses) × 100%
```

**Prometheus Query:**
```promql
(
  sum(rate(isl_http_requests_total{status=~"5.."}[5m]))
  /
  sum(rate(isl_http_requests_total[5m]))
) * 100
```

**SLO Target:** < 0.1% error rate

---

## Security SLIs and SLOs

### 4. Authentication Success Rate

**SLI Definition:**
```
Auth Success Rate = (Successful Authentications / Total Auth Attempts) × 100%

Note: Excludes legitimate 401s from missing API keys on protected endpoints
```

**Prometheus Query:**
```promql
# Successful auth (requests that passed authentication)
(
  sum(rate(isl_http_requests_total{status!~"401|403"}[5m]))
  /
  sum(rate(isl_http_requests_total[5m]))
) * 100
```

**SLO Target:** > 99% for authenticated clients

**Alert Threshold:** < 95% triggers investigation

---

### 5. Rate Limiting Accuracy

**SLI Definition:**
```
Rate Limit Accuracy = (Correctly Limited Requests / Total Rate Limit Events) × 100%

"Correctly Limited" = Requests that exceeded configured limits
```

**Prometheus Query:**
```promql
# Rate of rate-limited requests
sum(rate(isl_rate_limit_checks_total{result="blocked"}[5m]))

# Ratio of blocked to total
(
  sum(rate(isl_rate_limit_checks_total{result="blocked"}[5m]))
  /
  sum(rate(isl_rate_limit_checks_total[5m]))
) * 100
```

**SLO Target:** 100% enforcement (no bypass)

**Alert Threshold:** Any bypass detected

---

### 6. Redis Availability (for Rate Limiting)

**SLI Definition:**
```
Redis Availability = Time Redis is healthy / Total Time × 100%
```

**Prometheus Query:**
```promql
avg_over_time(isl_redis_client_connected[5m]) * 100
```

**SLO Target:** 99.9%

**Fallback Behavior:** In-memory rate limiting when Redis unavailable

---

### 7. Request Size Rejection Accuracy

**SLI Definition:**
```
Oversized Request Rejection Rate = (Rejected Oversized / Total Oversized) × 100%
```

**SLO Target:** 100% (all oversized requests rejected)

---

## Operational SLIs and SLOs

### 8. Throughput

**SLI Definition:**
```
Throughput = Requests per second handled successfully
```

**Prometheus Query:**
```promql
sum(rate(isl_http_requests_total{status=~"2.."}[1m]))
```

**SLO Target:** Handle 1000 RPS at p99 < 1s latency

---

### 9. Active Requests

**SLI Definition:**
```
Active Requests = Current number of in-flight requests
```

**Prometheus Query:**
```promql
isl_active_requests
```

**SLO Target:** < 100 concurrent requests per replica

**Alert Threshold:** > 80 concurrent requests

---

## SLO Burn Rate Alerts

### Fast Burn (High Severity)
Consuming error budget rapidly - immediate action required.

```yaml
# Prometheus Alert Rule
- alert: ISLHighErrorBurnRate
  expr: |
    (
      sum(rate(isl_http_requests_total{status=~"5.."}[5m]))
      /
      sum(rate(isl_http_requests_total[5m]))
    ) > 0.001 * 14.4
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "ISL error rate consuming error budget too fast"
    description: "At current rate, monthly error budget will be exhausted in < 2 hours"
```

### Slow Burn (Medium Severity)
Consuming error budget steadily - investigation needed.

```yaml
- alert: ISLMediumErrorBurnRate
  expr: |
    (
      sum(rate(isl_http_requests_total{status=~"5.."}[30m]))
      /
      sum(rate(isl_http_requests_total[30m]))
    ) > 0.001 * 6
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "ISL error rate elevated"
    description: "At current rate, monthly error budget will be exhausted in < 1 week"
```

---

## Security Alert Rules

### Authentication Failures Spike

```yaml
- alert: ISLAuthFailureSpike
  expr: |
    sum(rate(isl_http_requests_total{status="401"}[5m])) > 10
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High rate of authentication failures"
    description: "Possible brute force attack or misconfigured client"
```

### Rate Limit Exhaustion

```yaml
- alert: ISLRateLimitExhaustion
  expr: |
    sum(rate(isl_rate_limit_hits_total[5m])) > 50
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High rate of rate limit violations"
    description: "Clients are hitting rate limits frequently"
```

### Redis Connection Lost

```yaml
- alert: ISLRedisDown
  expr: isl_redis_client_connected == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Redis connection lost"
    description: "Rate limiting falling back to in-memory mode"
```

---

## Dashboard Queries

### Error Budget Remaining (Monthly)

```promql
# Monthly error budget remaining (as percentage)
(
  1 - (
    sum(increase(isl_http_requests_total{status=~"5.."}[30d]))
    /
    sum(increase(isl_http_requests_total[30d]))
  ) / 0.001
) * 100
```

### Current Availability (Rolling 7 days)

```promql
(
  sum(increase(isl_http_requests_total{status!~"5.."}[7d]))
  /
  sum(increase(isl_http_requests_total[7d]))
) * 100
```

---

## SLO Review Schedule

| Review Type | Frequency | Participants |
|-------------|-----------|--------------|
| SLO Metrics Review | Weekly | Engineering |
| Error Budget Review | Monthly | Engineering + Product |
| SLO Target Adjustment | Quarterly | Engineering + Product + SRE |
| Comprehensive SLO Audit | Annually | All stakeholders |

---

## Appendix: Metric Collection

### Required Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `isl_http_requests_total` | Counter | method, endpoint, status | Total HTTP requests |
| `isl_http_request_duration_seconds` | Histogram | method, endpoint | Request latency |
| `isl_http_errors_total` | Counter | method, endpoint, error_code | HTTP errors |
| `isl_rate_limit_hits_total` | Counter | identifier_type | Rate limit violations |
| `isl_rate_limit_checks_total` | Counter | result | Rate limit decisions |
| `isl_redis_client_connected` | Gauge | - | Redis connection status |
| `isl_redis_client_ops_total` | Counter | operation | Redis operations |
| `isl_active_requests` | Gauge | - | In-flight requests |

### Scrape Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'isl'
    scrape_interval: 15s
    static_configs:
      - targets: ['isl:8000']
    metrics_path: '/metrics'
```

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-26 | 1.0 | Initial SLO/SLI definitions |
