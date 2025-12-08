# Error Recovery & Resilience

Enterprise-grade error recovery patterns for the Inference Service Layer.

## Overview

The ISL implements comprehensive error recovery patterns to ensure production resilience:

- **Graceful Degradation**: Services return partial results rather than complete failures
- **Circuit Breakers**: Prevent cascading failures from expensive operations
- **Multi-level Fallbacks**: Progressive fallback chains ensure availability
- **Health Monitoring**: Track service health and degradation in real-time
- **Never 500 Errors**: Always return actionable results or helpful error messages

## Architecture

### Error Recovery Framework

Located in `src/utils/error_recovery.py`, provides:

```python
from src.utils.error_recovery import (
    CircuitBreaker,
    RetryStrategy,
    FallbackStrategy,
    HealthMonitor,
    with_fallback,
    with_circuit_breaker,
    with_retry,
    health_monitor,  # Global singleton
)
```

### Core Components

#### 1. Circuit Breaker

Prevents repeated calls to failing expensive operations.

**States**:
- `CLOSED`: Normal operation (all calls proceed)
- `OPEN`: Too many failures (calls rejected immediately)
- `HALF_OPEN`: Testing recovery (limited calls allowed)

**Configuration**:
```python
breaker = CircuitBreaker(
    name="operation_name",
    failure_threshold=5,      # Open after 5 failures
    success_threshold=2,      # Close after 2 successes in HALF_OPEN
    timeout=60               # Try recovery after 60 seconds
)
```

**Usage**:
```python
try:
    result = breaker.call(expensive_function, arg1, arg2)
except RecoveryError:
    # Circuit is OPEN - use fallback
    result = fallback_function()
```

#### 2. Retry Strategy

Exponential backoff for transient failures.

**Configuration**:
```python
retry = RetryStrategy(
    max_retries=3,
    initial_delay=0.1,      # 100ms initial delay
    backoff_factor=2.0,     # Double delay each retry
    max_delay=10.0          # Cap at 10 seconds
)
```

**Usage**:
```python
result = retry.execute(flaky_operation, arg1, arg2)
```

#### 3. Health Monitor

Tracks service health and degradation.

**Health States**:
- `HEALTHY`: Success rate ≥ 80%
- `DEGRADED`: Success rate 50-79%
- `FAILING`: Success rate < 50%

**Usage**:
```python
# Record operations
health_monitor.record_success("service_name")
health_monitor.record_failure("service_name")
health_monitor.record_fallback("service_name")

# Check health
health = health_monitor.get_health("service_name")
print(f"Status: {health.status.value}")
print(f"Success rate: {health.success_rate:.2%}")
```

## Service Implementations

### 1. Conformal Prediction

**Fallback Chain**: Normal → Degraded → Monte Carlo → Error

#### Normal Mode (10+ calibration points)
- Full conformal prediction
- Split calibration for efficiency
- Finite-sample valid guarantees

#### Degraded Mode (5-9 calibration points)
- Conformal with all calibration data
- Still finite-sample valid
- Warning in assumptions

#### Fallback Mode (0-4 calibration points)
- Monte Carlo intervals (percentile-based)
- Not finite-sample valid
- Clear warning to user

#### Ultimate Fallback
- Helpful 400 error message
- Suggests collecting more calibration data
- Never returns 500 error

**Code Location**: `src/services/conformal_predictor.py:47-145`

**Monitoring**:
```python
health = health_monitor.get_health("conformal_prediction")
```

### 2. Causal Discovery

**Fallback Chain**: NOTEARS/PC → Simple Correlation → Minimal DAG

#### Advanced Algorithms (if enabled)
- NOTEARS: Gradient-based continuous optimization
- PC: Constraint-based discovery
- Circuit breaker protected (threshold=3, timeout=120s)

#### Simple Discovery Fallback
- Correlation-based edge creation
- Applies prior knowledge constraints
- Always succeeds with reasonable defaults

#### Ultimate Fallback
- Returns DAG with nodes only (no edges)
- Very low confidence score (0.1)
- Always succeeds

**Code Locations**:
- `src/services/causal_discovery_engine.py:223-371` (advanced)
- `src/services/causal_discovery_engine.py:373-413` (fallback)
- `src/services/causal_discovery_engine.py:415-455` (minimal)

**Circuit Breakers**:
```python
from src.services.causal_discovery_engine import _notears_breaker, _pc_breaker

# Check states
print(_notears_breaker.state)  # CLOSED, OPEN, or HALF_OPEN
print(_notears_breaker.failure_count)
```

### 3. Validation Suggester

**Fallback Chain**: Complex Analysis → Simple Strategies → Manual Guidance

#### Normal Mode
- Comprehensive path analysis (backdoor, frontdoor, instrumental)
- Multiple adjustment strategies ranked by complexity
- Circuit breaker protected (threshold=3, timeout=60s)

#### Simple Strategy Fallback
- Basic backdoor adjustment (control all confounders)
- Limited to 5 variables for manageability
- Always returns at least one strategy

#### Ultimate Fallback
- Returns manual guidance strategy
- Low confidence score (0.3)
- Never fails completely

**Code Locations**:
- `src/services/advanced_validation_suggester.py:153-224` (normal)
- `src/services/advanced_validation_suggester.py:784-866` (fallback)

**Circuit Breakers**:
```python
from src.services.advanced_validation_suggester import (
    _path_analysis_breaker,
    _strategy_generation_breaker,
)
```

## Monitoring

### Health Endpoints

#### GET /health/services

Returns health status of all monitored services.

**Response**:
```json
{
  "overall_status": "healthy",
  "timestamp": "2025-11-23T10:30:00Z",
  "services": {
    "conformal_prediction": {
      "status": "HEALTHY",
      "success_rate_percent": 95.5,
      "total_requests": 1000,
      "successes": 955,
      "failures": 20,
      "fallbacks": 25,
      "uptime_seconds": 3600.5,
      "last_check": "2025-11-23T10:29:55Z"
    },
    "advanced_discovery": {
      "status": "DEGRADED",
      "success_rate_percent": 65.0,
      ...
    }
  }
}
```

**Health States**:
- `HEALTHY`: All services operating normally (≥80% success rate)
- `DEGRADED`: Some services experiencing issues (50-79% success rate)

#### GET /health/circuit-breakers

Returns circuit breaker status for expensive operations.

**Response**:
```json
{
  "overall_status": "operational",
  "timestamp": "2025-11-23T10:30:00Z",
  "circuit_breakers": {
    "notears_discovery": {
      "state": "CLOSED",
      "failure_count": 0,
      "success_count": 45,
      "failure_threshold": 3,
      "timeout_seconds": 120,
      "last_failure_time": null
    },
    "path_analysis": {
      "state": "OPEN",
      "failure_count": 5,
      "failure_threshold": 3,
      "timeout_seconds": 60,
      "last_failure_time": "2025-11-23T10:25:30Z"
    }
  }
}
```

**Circuit States**:
- `CLOSED`: Normal operation (green)
- `OPEN`: Circuit tripped due to repeated failures (red) - calls rejected
- `HALF_OPEN`: Testing recovery (yellow) - limited calls allowed

## Operational Runbook

### Monitoring Checklist

**Daily**:
- ✅ Check `/health/services` - all services HEALTHY?
- ✅ Check `/health/circuit-breakers` - all circuits CLOSED?
- ✅ Review service health success rates (should be >90%)

**Weekly**:
- ✅ Review fallback usage trends
- ✅ Investigate any persistent DEGRADED services
- ✅ Analyze circuit breaker trip patterns

### Common Scenarios

#### Scenario 1: Service Shows DEGRADED

**Symptoms**:
```json
{
  "conformal_prediction": {
    "status": "DEGRADED",
    "success_rate_percent": 65.0,
    "fallbacks": 350
  }
}
```

**Diagnosis**:
1. Check if this is expected (e.g., users sending insufficient calibration data)
2. Review logs for `conformal_fallback_to_monte_carlo` events
3. Check if this is a data quality issue vs. system issue

**Actions**:
- If data quality: Consider user education about calibration requirements
- If system issue: Investigate underlying errors before fallback

**Resolution**:
- DEGRADED is often expected behavior (graceful degradation working)
- Only escalate if success rate drops below 50% (FAILING)

#### Scenario 2: Circuit Breaker OPEN

**Symptoms**:
```json
{
  "notears_discovery": {
    "state": "OPEN",
    "failure_count": 5,
    "last_failure_time": "2025-11-23T10:25:30Z"
  }
}
```

**Diagnosis**:
1. Check logs for repeated failures in NOTEARS algorithm
2. Identify common patterns in failing requests
3. Check if this is a resource issue (memory, CPU) or algorithmic issue

**Actions**:
- Review logs: `grep "notears_failed_fallback" logs/app.log`
- Check resource utilization during failures
- Verify data characteristics (sample size, dimensionality)

**Resolution**:
- Circuit will automatically transition to HALF_OPEN after timeout (120s)
- If failures persist, consider:
  - Adjusting algorithm parameters
  - Increasing resource limits
  - Improving data validation

**Manual Reset** (if needed):
```python
from src.services.causal_discovery_engine import _notears_breaker
_notears_breaker.reset()
```

#### Scenario 3: High Fallback Rate

**Symptoms**:
- Service health shows many fallbacks but still HEALTHY
- Users receiving degraded results frequently

**Diagnosis**:
1. Determine if fallbacks are due to expected conditions (e.g., insufficient data)
2. Check if fallback quality is acceptable for use cases

**Actions**:
- Review fallback trigger conditions
- Assess impact on user experience
- Consider adjusting thresholds if appropriate

**Resolution**:
- High fallback rate is acceptable if:
  - Triggered by expected user behavior (e.g., small datasets)
  - Fallback quality meets requirements
  - Success rate remains high
- Investigate further if:
  - Fallback rate suddenly increases
  - Coincides with system changes

### Resetting Health Monitoring

**Reset a single service**:
```python
from src.utils.error_recovery import health_monitor
health_monitor.reset_service("conformal_prediction")
```

**Reset all services**:
```python
for service in health_monitor.get_all_services():
    health_monitor.reset_service(service)
```

### Adjusting Thresholds

**Circuit Breaker**:
```python
# Edit src/services/causal_discovery_engine.py
_notears_breaker = CircuitBreaker(
    "notears_discovery",
    failure_threshold=5,  # Increase if too sensitive
    timeout=180           # Increase if recovery takes longer
)
```

**Health Monitor** (edit `src/utils/error_recovery.py`):
```python
# Adjust health state thresholds
if success_rate >= 0.8:  # Change from 0.8 to 0.7 if too strict
    return ServiceHealthStatus.HEALTHY
```

## Performance Impact

### Memory Overhead
- Circuit breakers: ~1 KB per breaker (4 total)
- Health monitor: ~5 KB per service (4 total)
- **Total**: <50 KB overhead

### Latency Impact
- Circuit breaker check: <0.1ms
- Health monitoring: <0.1ms
- **Fallback operations**:
  - Conformal → Monte Carlo: +2-5ms
  - Advanced → Simple discovery: -800ms (faster!)
  - Complex → Simple strategies: -200ms (faster!)

### Throughput Impact
- Minimal (<1%) in normal operation
- **Improved** during degradation (fast fallbacks better than slow failures)

## Best Practices

### For Developers

1. **Always use fallbacks**:
   ```python
   try:
       result = expensive_operation()
   except Exception as e:
       logger.warning("fallback", extra={"error": str(e)})
       health_monitor.record_fallback("service_name")
       result = cheap_fallback()
   ```

2. **Record all operations**:
   ```python
   try:
       result = operation()
       health_monitor.record_success("service_name")
       return result
   except Exception as e:
       health_monitor.record_failure("service_name")
       raise
   ```

3. **Use circuit breakers for expensive ops**:
   ```python
   # Operations that take >1 second or consume significant resources
   breaker = CircuitBreaker("operation", failure_threshold=3)
   result = breaker.call(expensive_function)
   ```

4. **Provide helpful error messages**:
   ```python
   # ✅ Good
   raise HTTPException(
       status_code=400,
       detail="Insufficient calibration data (3 points). "
              "Provide at least 10 points for conformal prediction."
   )

   # ❌ Bad
   raise HTTPException(status_code=500, detail="Error")
   ```

### For Operations

1. **Monitor health endpoints daily**
2. **Set up alerts**:
   - Any service FAILING (success rate <50%)
   - Any circuit breaker OPEN for >10 minutes
   - Fallback rate >30% sustained for >1 hour

3. **Review logs for degradation events**:
   - `conformal_fallback_to_monte_carlo`
   - `notears_failed_fallback`
   - `strategy_generation_failed_fallback`

4. **Validate fallback quality**:
   - Monte Carlo intervals should have reasonable width
   - Simple discovery should find basic structures
   - Simple strategies should be actionable

## Testing

### Unit Tests

Located in `tests/unit/test_error_recovery.py`:

```bash
pytest tests/unit/test_error_recovery.py -v
```

**Coverage**:
- Circuit breaker state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- Retry with exponential backoff
- Health monitoring and status calculation
- Decorator patterns (@with_fallback, @with_circuit_breaker, @with_retry)

### Integration Tests

Located in `tests/integration/test_error_recovery_integration.py`:

```bash
pytest tests/integration/test_error_recovery_integration.py -v
```

**Coverage**:
- Conformal prediction fallback chains
- Causal discovery fallback chains
- Validation suggester fallback chains
- End-to-end recovery scenarios

### Run All Error Recovery Tests

```bash
pytest tests/ -k "error_recovery" -v
```

## Metrics & Observability

### Prometheus Metrics (available at `/metrics`)

```
# Service health
isl_service_health_status{service="conformal_prediction"} 1  # 1=HEALTHY, 0.5=DEGRADED, 0=FAILING
isl_service_success_rate{service="conformal_prediction"} 0.95
isl_service_fallback_total{service="conformal_prediction"} 25

# Circuit breakers
isl_circuit_breaker_state{name="notears_discovery"} 0  # 0=CLOSED, 1=OPEN, 2=HALF_OPEN
isl_circuit_breaker_failures{name="notears_discovery"} 0
```

### Structured Logging

All fallback events are logged with structured data:

```python
logger.warning(
    "conformal_fallback_to_monte_carlo",
    extra={
        "n_calibration": 3,
        "min_required": 10,
        "alpha": 0.1
    }
)
```

**Query examples**:
```bash
# Find all fallback events
grep "fallback" logs/app.log | jq .

# Count fallbacks by service
grep "fallback" logs/app.log | jq -r .service | sort | uniq -c

# Find circuit breaker trips
grep "circuit_state.*OPEN" logs/app.log | jq .
```

## Future Enhancements

1. **Adaptive Thresholds**: Adjust circuit breaker thresholds based on historical success rates
2. **Fallback Quality Metrics**: Track quality of fallback results vs. primary results
3. **Automatic Recovery**: Self-healing for known failure patterns
4. **Graceful Mode Switching**: Smooth transitions between normal/degraded modes
5. **Distributed Circuit Breakers**: Share circuit state across multiple instances

## References

- Circuit Breaker Pattern: [Martin Fowler](https://martinfowler.com/bliki/CircuitBreaker.html)
- Retry Pattern: [Microsoft Cloud Patterns](https://learn.microsoft.com/en-us/azure/architecture/patterns/retry)
- Graceful Degradation: [AWS Well-Architected](https://aws.amazon.com/architecture/well-architected/)

---

**Last Updated**: 2025-11-23
**Version**: 1.0
**Author**: ISL Team
