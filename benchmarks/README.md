# ISL Performance Benchmarks

## Overview

This directory contains performance benchmark scripts for validating ISL meets production performance targets.

## Performance Targets (Phase 1D)

| Endpoint Type | P95 Latency Target | Notes |
|---------------|-------------------|-------|
| **Causal Validation** | < 2.0s | DAG validation, adjustment sets |
| **Counterfactual Analysis** | < 2.0s | Monte Carlo sampling, heavier computation |
| **Preference Elicitation** | < 1.5s | ActiVA algorithm, user-facing |
| **Teaching Examples** | < 1.5s | Bayesian teaching, user-facing |
| **Advanced Validation** | < 2.0s | Comprehensive model checks |

**Additional Targets**:
- Support 100+ concurrent users
- Cache hit rate > 40% (after warm-up)
- Success rate > 99.5%

## Quick Start

### 1. Start ISL

```bash
# Terminal 1: Start ISL
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Run Benchmark

```bash
# Terminal 2: Run benchmark (default: 60s, 10 concurrent users)
poetry run python benchmarks/performance_benchmark.py

# Custom duration and concurrency
poetry run python benchmarks/performance_benchmark.py --duration 120 --concurrency 20

# Test against remote instance
poetry run python benchmarks/performance_benchmark.py --host https://isl.example.com
```

### 3. Review Results

Results are printed to console and saved to `benchmark_results.json`.

Example output:
```
================================================================================
BENCHMARK RESULTS
================================================================================

Endpoint: /api/v1/causal/validate
  Total Requests:     120
  Successful:         120 (100.0%)
  Failed:             0
  Average Latency:    0.542s
  P50 Latency:        0.498s
  P95 Latency:        0.856s (target: <2.0s) ✓ PASS
  P99 Latency:        1.120s
  Max Latency:        1.245s

...

================================================================================
TARGET EVALUATION
================================================================================

Causal Validation                P95: 0.856s  Target: <2.0s  ✓ PASS
Counterfactual Analysis          P95: 1.234s  Target: <2.0s  ✓ PASS
Preference Elicitation           P95: 0.678s  Target: <1.5s  ✓ PASS
Teaching Examples                P95: 0.891s  Target: <1.5s  ✓ PASS

Concurrency:                     10 users  Target: 100+  ⚠ NOTE: Test with 100+ for production

================================================================================
OVERALL: ✓ ALL TARGETS MET
================================================================================
```

## Benchmark Options

```bash
python benchmarks/performance_benchmark.py --help
```

**Options**:
- `--host URL`: ISL base URL (default: `http://localhost:8000`)
- `--duration N`: Benchmark duration in seconds (default: `60`)
- `--concurrency N`: Concurrent users (default: `10`)
- `--output FILE`: Output JSON file (default: `benchmark_results.json`)

## Interpreting Results

### Latency Metrics

- **Average**: Mean latency across all requests
- **P50 (Median)**: 50% of requests faster than this
- **P95**: 95% of requests faster than this (our target metric)
- **P99**: 99% of requests faster than this
- **Max**: Slowest request observed

### Success Rate

Target: **> 99.5%** success rate

If success rate is low:
1. Check ISL logs for errors
2. Verify Redis is running
3. Check system resources (CPU, memory)
4. Reduce concurrency to identify bottlenecks

### Concurrency Testing

For pilot: Test with 10-20 concurrent users
For production: Test with 100+ concurrent users

```bash
# Production stress test
poetry run python benchmarks/performance_benchmark.py \
  --duration 300 \
  --concurrency 100 \
  --output production_benchmark.json
```

## Monitoring During Benchmarks

### 1. Watch Prometheus Metrics

Open `http://localhost:9090` and query:

```promql
# Request rate
rate(isl_http_requests_total[1m])

# P95 latency
histogram_quantile(0.95, rate(isl_http_request_duration_seconds_bucket[1m]))

# Error rate
rate(isl_http_errors_total[1m])

# Active requests
isl_active_requests
```

### 2. Watch Grafana Dashboard

Open `http://localhost:3000` → ISL Overview dashboard

Watch real-time:
- Request rate
- Latency percentiles
- Error rates
- Active requests

### 3. Watch System Resources

```bash
# CPU and memory
docker stats isl-api

# Redis memory
docker exec isl-redis redis-cli INFO memory
```

## Troubleshooting

### High Latency

**Symptoms**: P95 latency exceeds targets

**Common causes**:
1. **Redis not available**: Check `docker ps | grep redis`
2. **CPU throttling**: Check `docker stats`
3. **Memory pressure**: Check system memory usage
4. **Cold start**: Run warm-up requests first
5. **Network latency**: Test localhost vs. remote

**Solutions**:
```bash
# 1. Start Redis if not running
cd deployment/redis
docker-compose -f docker-compose.redis.yml up -d

# 2. Warm up cache (run small benchmark first)
poetry run python benchmarks/performance_benchmark.py --duration 10

# 3. Then run full benchmark
poetry run python benchmarks/performance_benchmark.py --duration 60
```

### High Failure Rate

**Symptoms**: Success rate < 99%

**Common causes**:
1. ISL not running
2. Concurrency too high for system
3. Timeout errors

**Solutions**:
```bash
# 1. Verify ISL is running
curl http://localhost:8000/health

# 2. Reduce concurrency
poetry run python benchmarks/performance_benchmark.py --concurrency 5

# 3. Check logs
docker logs isl-api --tail 100
```

### Timeout Errors

**Symptoms**: Requests timing out

**Solutions**:
1. Increase timeout in benchmark script (default: 30s)
2. Check for long-running computations
3. Review slow query logs

## Performance Optimization Tips

### 1. Redis Caching

- Warm up cache before benchmarking
- Monitor cache hit rate: Target > 40%
- Check Redis memory: `docker exec isl-redis redis-cli INFO memory`

### 2. Resource Allocation

**Minimum (Pilot)**:
- CPU: 2 cores
- Memory: 4 GB
- Redis: 512 MB

**Recommended (Production)**:
- CPU: 4 cores
- Memory: 8 GB
- Redis: 2 GB

### 3. Tuning Parameters

Edit `src/config.py`:
```python
# Increase worker processes
WORKERS = 4  # 2x CPU cores

# Adjust Redis pool size
REDIS_MAX_CONNECTIONS = 50
```

## Continuous Performance Testing

### CI/CD Integration

Add to your CI pipeline:

```yaml
# .github/workflows/performance.yml
- name: Run Performance Benchmarks
  run: |
    poetry run python benchmarks/performance_benchmark.py \
      --duration 30 \
      --concurrency 10 \
      --output ci_benchmark.json

    # Fail if targets not met
    python scripts/check_performance_targets.py ci_benchmark.json
```

### Scheduled Testing

Run daily benchmarks to track performance trends:

```cron
# Daily performance test at 2 AM
0 2 * * * cd /app && poetry run python benchmarks/performance_benchmark.py --duration 300 --output daily_$(date +\%Y\%m\%d).json
```

## Historical Results

Store benchmark results in `benchmarks/results/` directory:

```
benchmarks/results/
  └── 2025-01-20_baseline.json
  └── 2025-01-21_after_optimization.json
  └── 2025-01-22_production.json
```

Compare results over time to track performance regression/improvement.

## References

- [ISL Performance Targets](/docs/PHASE_1D_BRIEF.md)
- [Monitoring Setup](/docs/MONITORING_SETUP.md)
- [Redis Deployment](/docs/REDIS_DEPLOYMENT.md)

---

**Performance Testing Complete!** 🎉

Run benchmarks regularly to ensure ISL continues meeting performance targets as the system evolves.
