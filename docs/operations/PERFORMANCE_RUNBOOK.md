# ISL Performance & Observability Runbook

Operational guide for monitoring, troubleshooting, and optimizing ISL performance.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Monitoring Setup](#monitoring-setup)
3. [Performance Profiling](#performance-profiling)
4. [Load Testing](#load-testing)
5. [Alert Response](#alert-response)
6. [Common Issues](#common-issues)
7. [Scaling Recommendations](#scaling-recommendations)

---

## Quick Start

### Starting Monitoring Stack

```bash
# Start Prometheus, Grafana, and Alertmanager
docker-compose -f docker-compose.monitoring.yml up -d

# Check status
docker-compose -f docker-compose.monitoring.yml ps

# View Grafana dashboard
open http://localhost:3000
# Login: admin / admin

# View Prometheus
open http://localhost:9090

# View Alertmanager
open http://localhost:9093
```

### Running Performance Tests

```bash
# Profile endpoints
python scripts/performance/profile_endpoints.py

# Load test (Web UI)
locust -f tests/load/locustfile.py --host http://localhost:8000
# Open http://localhost:8089

# Load test (headless)
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 10m --headless
```

---

## Monitoring Setup

### Metrics Available

**Request Metrics**:
- `http_requests_total` - Total request count by endpoint/status
- `http_request_duration_seconds` - Request latency histogram
- `http_errors_total` - Error count by type/endpoint
- `active_requests` - Currently active requests

**Error Recovery Metrics**:
- `isl_circuit_breaker_state` - Circuit breaker states (0=CLOSED, 1=HALF_OPEN, 2=OPEN)
- `isl_service_health_status` - Service health (1=HEALTHY, 0.5=DEGRADED, 0=FAILING)
- `isl_fallback_triggered_total` - Fallback trigger count

**Cache Metrics**:
- `isl_cache_hit_rate` - Cache hit percentage
- `isl_cache_size` - Current cache size
- `isl_cache_evictions_total` - Cache eviction count

### Grafana Dashboards

**Main Dashboard**: `ISL Performance & Observability`

Panels:
1. **Request Overview** - RPS, latency percentiles, error rate
2. **Endpoint Health** - Per-endpoint latency and success rates
3. **Error Recovery** - Circuit breaker states, fallback rates
4. **Cache Performance** - Hit rate, size, evictions
5. **Resource Usage** - Memory, active requests

---

## Performance Profiling

### Endpoint Profiling

**Profile all endpoints**:
```bash
python scripts/performance/profile_endpoints.py
```

**Output**: `scripts/performance/performance_report.json`

**Key Metrics**:
- P50/P95/P99 latency per endpoint
- Memory usage per endpoint
- CPU profile (top 20 functions)
- Success rate

**Interpreting Results**:

```json
{
  "endpoint": "Causal Validation",
  "latency": {
    "mean": 145.3,
    "median": 132.1,
    "p95": 234.5,
    "p99": 312.7
  },
  "memory": {
    "mean_mb": 12.3,
    "max_mb": 15.7
  },
  "success_rate": 1.0
}
```

**Targets**:
- ✅ P50 <500ms (PASS)
- ⚠️ P95 <1000ms (WARNING if exceeded)
- ❌ P95 >3000ms (FAIL if exceeded)

### Identifying Bottlenecks

**Check CPU profile**:
Look for functions with high cumulative time in `cpu_profile_top_20` output.

**Common bottlenecks**:
1. **NOTEARS discovery** - O(d³) complexity
2. **Path analysis** - Exponential in DAG size
3. **Monte Carlo sampling** - Linear in sample count
4. **JSON serialization** - Large response payloads

**Optimization strategies**:
- Enable caching for repeated queries
- Reduce sample counts for Monte Carlo
- Use simpler algorithms for large DAGs
- Implement pagination for large results

---

## Load Testing

### Test Scenarios

**1. Ramp Test** - Gradually increase load
```bash
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 15m --headless
```

**Purpose**: Find breaking point  
**Watch for**: When P95 > 3s or error rate > 5%

**2. Sustained Load** - Constant moderate load
```bash
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 30m --headless
```

**Purpose**: Check stability  
**Watch for**: Memory leaks, gradual degradation

**3. Spike Test** - Sudden high load
```bash
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --users 200 --spawn-rate 50 --run-time 5m --headless
```

**Purpose**: Test resilience  
**Watch for**: Circuit breakers, error recovery

### Interpreting Load Test Results

**Good Performance**:
```
Total RPS: 45.2
Failure rate: 0.5%
P50: 285ms
P95: 687ms
P99: 1243ms
```

**Degraded Performance**:
```
Total RPS: 38.7
Failure rate: 3.2%
P50: 523ms  ⚠️
P95: 2341ms ⚠️
P99: 4567ms ❌
```

**Breaking Point Indicators**:
- Failure rate >5%
- P95 >3s sustained
- RPS plateaus (can't scale higher)
- Circuit breakers opening
- Memory usage increasing unbounded

---

## Alert Response

### HighLatency

**Symptom**: P95 latency >3s for 5 minutes

**Diagnosis**:
1. Check Grafana "Endpoint Latency Heatmap"
2. Identify slow endpoint
3. Check if specific to one endpoint or all

**Response**:
```bash
# Check service health
curl http://localhost:8000/health/services

# Check circuit breakers
curl http://localhost:8000/health/circuit-breakers

# If NOTEARS causing issues:
# - Reduce max_iter parameter
# - Use simpler algorithm (correlation)
# - Enable circuit breaker
```

**Escalation**: If P95 >5s for >2 min, CRITICAL

---

### HighErrorRate

**Symptom**: Error rate >5% for 5 minutes

**Diagnosis**:
1. Check error distribution (Grafana pie chart)
2. Check logs for error patterns
3. Identify common error types

**Response**:
```bash
# Check error breakdown
curl http://localhost:8000/health/services | jq '.services'

# Check recent errors in logs
tail -n 100 logs/app.log | grep ERROR

# Common fixes:
# - 400 errors: Input validation issues
# - 500 errors: Check service health, circuit breakers
# - Timeout errors: Reduce load or scale up
```

---

### CircuitBreakerOpen

**Symptom**: Circuit breaker in OPEN state

**Diagnosis**:
1. Identify which circuit: `GET /health/circuit-breakers`
2. Check failure count and last failure time
3. Review logs for underlying errors

**Response**:
```bash
# Check circuit breaker status
curl http://localhost:8000/health/circuit-breakers | jq

# Common circuits:
# - notears_discovery: NOTEARS algorithm failing
# - path_analysis: Path finding timeout
# - strategy_generation: Complex strategy computation

# Actions:
# - Wait for auto-recovery (timeout period)
# - Fix underlying issue (data quality, parameters)
# - Manual reset if needed (requires code change)
```

**Prevention**:
- Validate input data before expensive operations
- Set reasonable timeout/iteration limits
- Monitor fallback trigger rates

---

### ServiceDegraded

**Symptom**: Service health status = DEGRADED

**Diagnosis**:
1. Check success rate: `GET /health/services`
2. Identify which service (conformal, discovery, validation)
3. Check fallback rate

**Response**:
```bash
# Example response:
{
  "conformal_prediction": {
    "status": "DEGRADED",
    "success_rate_percent": 65.0,
    "fallbacks": 350
  }
}

# Degraded is expected if:
# - Users sending insufficient calibration data
# - Large fallback usage (Monte Carlo instead of conformal)

# Action required only if:
# - Success rate <50% (becomes FAILING)
# - Fallback rate suddenly increases
# - User complaints about quality
```

---

## Common Issues

### Issue: High P95 Latency on Causal Discovery

**Symptoms**:
- `/api/v1/causal/discover` P95 >5s
- NOTEARS circuit breaker opening

**Root Causes**:
1. Too many variables (d >10)
2. Too many samples (n >1000)
3. max_iter too high

**Solutions**:
```python
# Reduce max_iter
payload = {
    "data": data,
    "variable_names": vars,
    "algorithm": "notears",
    "max_iter": 50,  # Reduce from default 100
}

# Or use simpler algorithm
payload = {
    "algorithm": "correlation",  # Much faster
    "threshold": 0.3
}
```

**Prevention**:
- Document complexity limits (d ≤10)
- Add input validation
- Enable circuit breaker

---

### Issue: Memory Usage Increasing

**Symptoms**:
- Memory >2GB sustained
- Gradual increase over time

**Root Causes**:
1. Cache growing unbounded
2. Memory leak in NOTEARS
3. Large response objects not GC'd

**Solutions**:
```bash
# Check cache stats
curl http://localhost:8000/cache/stats

# If cache too large:
# - Reduce CACHE_MAX_SIZE in config
# - Reduce CACHE_TTL
# - Clear cache (restart service)

# Monitor with:
docker stats isl-api
```

---

### Issue: Cache Hit Rate <50%

**Symptoms**:
- `isl_cache_hit_rate` <0.5
- High latency despite caching

**Root Causes**:
1. Unique queries (no repeats)
2. Cache TTL too short
3. Cache eviction too aggressive

**Solutions**:
```python
# Increase cache size
CACHE_MAX_SIZE = 1000  # From 500

# Increase TTL
CACHE_TTL = 3600  # From 1800

# Check if queries are cacheable
# - Same DAG structure
# - Same parameters
# - Deterministic results (use seed)
```

---

## Scaling Recommendations

### Vertical Scaling

**Current Capacity** (single instance):
- ~50 concurrent users
- ~45 RPS sustained
- P95 <1s

**Scale Up If**:
- Users >50 sustained
- P95 >1s consistently
- Memory >80%

**Recommendations**:
- 2 CPU cores → 4 CPU cores
- 2GB RAM → 4GB RAM
- Enable connection pooling

---

### Horizontal Scaling

**When to Scale Horizontally**:
- Vertical scaling exhausted
- Need high availability
- Users >100 concurrent

**Architecture**:
```
Load Balancer (nginx)
  ├── ISL Instance 1
  ├── ISL Instance 2
  └── ISL Instance 3

Shared:
  ├── Redis (cache)
  └── Prometheus (metrics)
```

**Configuration**:
```yaml
# docker-compose.scale.yml
services:
  isl-api:
    deploy:
      replicas: 3
    
  redis:
    image: redis:alpine
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
```

---

### Caching Strategy

**What to Cache**:
- ✅ Causal validation (DAG structure deterministic)
- ✅ Path analysis (expensive, deterministic)
- ✅ Conformal intervals (with same calibration data)
- ❌ Counterfactuals (unless seeded)

**Cache Tiers**:
1. **In-memory** (current) - Fast, limited size
2. **Redis** (future) - Shared across instances
3. **CDN** (future) - For static/public results

---

## Performance Targets

### Latency Targets

| Percentile | Target | Warning | Critical |
|------------|--------|---------|----------|
| P50        | <500ms | >500ms  | >1000ms  |
| P95        | <1000ms| >1500ms | >3000ms  |
| P99        | <2000ms| >3000ms | >5000ms  |

### Throughput Targets

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| RPS    | >40    | <30     | <20      |
| Error Rate | <1% | >2%     | >5%      |
| Success Rate | >99% | <98%  | <95%     |

### Resource Targets

| Resource | Target | Warning | Critical |
|----------|--------|---------|----------|
| Memory   | <1.5GB | >2GB    | >2.5GB   |
| CPU      | <70%   | >85%    | >95%     |
| Cache Hit | >70% | <50%    | <30%     |

---

## Maintenance

### Daily
- ✅ Check Grafana dashboard
- ✅ Review alerts (if any)
- ✅ Check error logs

### Weekly
- ✅ Run performance profiling
- ✅ Review cache hit rates
- ✅ Check for memory leaks

### Monthly
- ✅ Run full load test
- ✅ Review performance trends
- ✅ Update capacity plan

---

## Troubleshooting Checklist

**Slow Performance**:
- [ ] Check Grafana latency dashboard
- [ ] Identify slow endpoint
- [ ] Profile endpoint
- [ ] Check circuit breakers
- [ ] Review recent code changes
- [ ] Check resource usage

**High Error Rate**:
- [ ] Check error distribution
- [ ] Review error logs
- [ ] Check service health
- [ ] Verify input validation
- [ ] Check circuit breakers
- [ ] Review recent deployments

**System Issues**:
- [ ] Check memory usage
- [ ] Check CPU usage
- [ ] Check active connections
- [ ] Review logs for exceptions
- [ ] Check disk space
- [ ] Verify dependencies

---

**Last Updated**: 2025-11-23  
**Version**: 1.0  
**Maintained By**: ISL Operations Team
