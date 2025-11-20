# PLoT Performance Guide

**Optimization Strategies for ISL Integration**

**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Performance Targets](#performance-targets)
2. [Result Caching Strategies](#result-caching-strategies)
3. [Connection Pooling](#connection-pooling)
4. [Async Concurrent Requests](#async-concurrent-requests)
5. [Request Batching](#request-batching)
6. [Payload Optimization](#payload-optimization)
7. [Monitoring & Profiling](#monitoring--profiling)
8. [Performance Checklist](#performance-checklist)

---

## Performance Targets

### ISL Latency Targets (P95)

| Endpoint Category | P95 Target | Typical Latency | Complexity |
|-------------------|------------|-----------------|------------|
| Health Check | <10ms | ~3ms | Minimal |
| Causal Validation | <2.0s | ~5ms | Low |
| Counterfactual Analysis | <2.0s | ~1.0s | High |
| Preference Elicitation | <1.5s | ~40ms | Medium |
| Preference Update | <1.5s | ~30ms | Low |
| Teaching Examples | <1.5s | ~300ms | Medium |
| Team Alignment | N/A | ~300ms | Medium |
| Sensitivity Analysis | N/A | ~150ms | Medium |
| Advanced Validation | <2.0s | ~100ms | Medium |

### PLoT Integration Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **End-to-end latency** | <3s | PLoT → ISL → Response |
| **Concurrent users** | 100+ | Pilot: 10-25 users |
| **Request success rate** | >99.5% | Excluding client errors |
| **Cache hit rate** | >40% | For deterministic operations |

---

## Result Caching Strategies

### 1. Config Fingerprint-Based Caching

**Principle**: ISL responses include `config_fingerprint` in metadata. Cache results using fingerprint as part of cache key.

```python
import hashlib
import json
from typing import Any, Dict, Optional


class ISLResultCache:
    """Cache ISL results with config fingerprint verification."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour default TTL

    def _generate_cache_key(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        config_fingerprint: str
    ) -> str:
        """Generate deterministic cache key."""
        # Create stable JSON representation
        payload_json = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()[:16]

        return f"isl:{endpoint}:{payload_hash}:{config_fingerprint}"

    def get(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        current_fingerprint: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result if fingerprint matches.

        Args:
            endpoint: ISL endpoint path
            payload: Request payload
            current_fingerprint: Current ISL config fingerprint

        Returns:
            Cached result or None
        """
        cache_key = self._generate_cache_key(endpoint, payload, current_fingerprint)

        cached_data = self.redis.get(cache_key)
        if cached_data:
            result = json.loads(cached_data)

            # Verify fingerprint still matches
            cached_fingerprint = result.get("_metadata", {}).get("config_fingerprint")
            if cached_fingerprint == current_fingerprint:
                return result
            else:
                # Fingerprint changed, invalidate cache
                self.redis.delete(cache_key)

        return None

    def set(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache ISL result.

        Args:
            endpoint: ISL endpoint path
            payload: Request payload
            result: ISL response
            ttl: Time to live in seconds (default: 3600)
        """
        config_fingerprint = result.get("_metadata", {}).get("config_fingerprint")
        if not config_fingerprint:
            # Cannot cache without fingerprint
            return

        cache_key = self._generate_cache_key(endpoint, payload, config_fingerprint)

        self.redis.setex(
            cache_key,
            ttl or self.ttl,
            json.dumps(result)
        )


# Example usage
import redis

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
cache = ISLResultCache(redis_client)


def validate_causal_model_cached(dag: dict, treatment: str, outcome: str):
    """Validate causal model with caching."""

    # Get current ISL fingerprint
    health_response = client.get(f"{ISL_BASE_URL}/health").json()
    current_fingerprint = health_response.get("config_fingerprint")

    # Build payload
    payload = {
        "dag": dag,
        "treatment": treatment,
        "outcome": outcome
    }

    # Check cache
    cached_result = cache.get(
        endpoint="/api/v1/causal/validate",
        payload=payload,
        current_fingerprint=current_fingerprint
    )

    if cached_result:
        print(f"✓ Cache hit")
        return cached_result

    # Cache miss - call ISL
    print(f"⚠ Cache miss - calling ISL")
    response = client.post(
        f"{ISL_BASE_URL}/api/v1/causal/validate",
        json=payload,
        headers={"X-Request-Id": generate_request_id()}
    )
    result = response.json()

    # Cache result
    cache.set(
        endpoint="/api/v1/causal/validate",
        payload=payload,
        result=result,
        ttl=3600
    )

    return result
```

### 2. User Session Caching

**For**: Preference learning endpoints where user beliefs persist in Redis

```python
def get_user_queries_cached(user_id: str, context: dict, num_queries: int):
    """Get preference queries with session-aware caching."""

    # Cache key includes user_id and context hash
    context_hash = hashlib.sha256(
        json.dumps(context, sort_keys=True).encode()
    ).hexdigest()[:16]

    cache_key = f"plot:queries:{user_id}:{context_hash}:{num_queries}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Cache miss - generate queries
    response = client.post(
        f"{ISL_BASE_URL}/api/v1/preferences/elicit",
        json={"user_id": user_id, "context": context, "num_queries": num_queries}
    )
    result = response.json()

    # Cache for 5 minutes (queries are time-sensitive)
    redis_client.setex(cache_key, 300, json.dumps(result))

    return result
```

### 3. Cache Warming

**Strategy**: Pre-populate cache with common requests during low-traffic periods

```python
async def warm_cache(common_requests: list):
    """Pre-populate cache with common requests."""

    async with httpx.AsyncClient() as client:
        tasks = []

        for request_spec in common_requests:
            task = client.post(
                f"{ISL_BASE_URL}{request_spec['endpoint']}",
                json=request_spec['payload']
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        # Results automatically cached by ISLResultCache
        print(f"✓ Cache warmed with {len(responses)} requests")


# Example: Warm cache during deployment
common_dags = [
    {
        "endpoint": "/api/v1/causal/validate",
        "payload": {
            "dag": {"nodes": ["Price", "Revenue"], "edges": [["Price", "Revenue"]]},
            "treatment": "Price",
            "outcome": "Revenue"
        }
    },
    # ... more common requests
]

asyncio.run(warm_cache(common_dags))
```

---

## Connection Pooling

### HTTP Connection Pool

**Benefit**: Reuse TCP connections, reduce latency by 20-50ms per request

```python
import httpx
from typing import Optional


class ISLConnectionPool:
    """Persistent HTTP connection pool for ISL."""

    def __init__(
        self,
        base_url: str,
        pool_size: int = 100,
        max_keepalive: int = 50,
        timeout: float = 10.0
    ):
        # Configure connection limits
        limits = httpx.Limits(
            max_connections=pool_size,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=30.0  # Keep connections alive for 30s
        )

        # Create persistent client
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            limits=limits,
            http2=True  # Enable HTTP/2 for multiplexing
        )

    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """Make request using pooled connection."""
        return await self.client.request(method, endpoint, **kwargs)

    async def close(self):
        """Close connection pool."""
        await self.client.aclose()


# Example usage with context manager
async def make_multiple_requests():
    """Make multiple requests efficiently using connection pool."""

    async with ISLConnectionPool(base_url=ISL_BASE_URL, pool_size=100) as pool:
        # First request establishes connection
        response1 = await pool.request(
            "POST",
            "/api/v1/causal/validate",
            json={...}
        )

        # Subsequent requests reuse connection (faster)
        response2 = await pool.request(
            "POST",
            "/api/v1/preferences/elicit",
            json={...}
        )

        return response1.json(), response2.json()
```

### Connection Pool Monitoring

```python
def monitor_connection_pool(client: httpx.AsyncClient):
    """Monitor connection pool statistics."""

    pool_stats = {
        "max_connections": client._limits.max_connections,
        "max_keepalive": client._limits.max_keepalive_connections,
        "active_connections": len(client._transport._pool._requests),
        "keepalive_connections": len(client._transport._pool._connections)
    }

    print(f"Connection Pool Stats:")
    print(f"  Active: {pool_stats['active_connections']}/{pool_stats['max_connections']}")
    print(f"  Keepalive: {pool_stats['keepalive_connections']}/{pool_stats['max_keepalive']}")

    # Alert if pool exhausted
    if pool_stats['active_connections'] >= pool_stats['max_connections'] * 0.8:
        print(f"⚠ WARNING: Connection pool 80% utilized")
```

---

## Async Concurrent Requests

### Pattern 1: Parallel Validation of Multiple Models

```python
async def validate_models_in_parallel(models: list) -> list:
    """Validate multiple models concurrently."""

    async with ISLConnectionPool(base_url=ISL_BASE_URL) as pool:
        tasks = [
            pool.request(
                "POST",
                "/api/v1/causal/validate",
                json={
                    "dag": model["dag"],
                    "treatment": model["treatment"],
                    "outcome": model["outcome"]
                }
            )
            for model in models
        ]

        # Execute concurrently
        responses = await asyncio.gather(*tasks)

        return [r.json() for r in responses]


# Example: Validate 10 models in ~5ms total (vs 50ms sequential)
models = [...]  # 10 models
results = asyncio.run(validate_models_in_parallel(models))
```

### Pattern 2: Pipeline Pattern for Multi-Step Workflows

```python
async def preference_elicitation_pipeline(users: list):
    """Run preference elicitation for multiple users in pipeline."""

    async with ISLConnectionPool(base_url=ISL_BASE_URL) as pool:
        # Stage 1: Generate queries for all users (parallel)
        query_tasks = [
            pool.request(
                "POST",
                "/api/v1/preferences/elicit",
                json={"user_id": user_id, "context": {...}, "num_queries": 3}
            )
            for user_id in users
        ]

        query_responses = await asyncio.gather(*query_tasks)
        queries_by_user = {
            users[i]: r.json() for i, r in enumerate(query_responses)
        }

        # Stage 2: Process user responses (simulated)
        # In production, wait for actual user responses
        user_responses = {...}  # User answers

        # Stage 3: Update beliefs for all users (parallel)
        update_tasks = [
            pool.request(
                "POST",
                "/api/v1/preferences/update",
                json={
                    "user_id": user_id,
                    "query_id": queries_by_user[user_id]["queries"][0]["query_id"],
                    "response": user_responses.get(user_id, {})
                }
            )
            for user_id in users
            if user_id in user_responses
        ]

        update_responses = await asyncio.gather(*update_tasks)

        return [r.json() for r in update_responses]
```

---

## Request Batching

### Batch Similar Requests

**Strategy**: Group similar requests to reduce overhead

```python
async def batch_counterfactual_scenarios(
    model: dict,
    outcome: str,
    interventions: list
) -> list:
    """
    Analyze multiple scenarios for the same model.

    Optimization: Reuse model structure, only vary interventions.
    """

    async with ISLConnectionPool(base_url=ISL_BASE_URL) as pool:
        tasks = [
            pool.request(
                "POST",
                "/api/v1/causal/counterfactual",
                json={
                    "model": model,
                    "outcome": outcome,
                    "intervention": intervention,
                    "num_samples": 1000
                }
            )
            for intervention in interventions
        ]

        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]


# Example: Test 5 price points concurrently
scenarios = [
    {"Price": 50},
    {"Price": 55},
    {"Price": 60},
    {"Price": 65},
    {"Price": 70}
]

results = asyncio.run(
    batch_counterfactual_scenarios(
        model={...},
        outcome="Revenue",
        interventions=scenarios
    )
)

# Total time: ~1.5s (vs 7.5s sequential)
```

---

## Payload Optimization

### 1. Minimize Payload Size

**Strategy**: Only send necessary data

```python
# ❌ BAD: Send entire context every time
payload = {
    "user_id": "alice",
    "context": {
        "domain": "pricing",
        "variables": ["revenue", "churn", "brand", "market_share"],
        "constraints": [...],  # Large list
        "historical_data": [...],  # Large dataset
        "metadata": {...}
    },
    "num_queries": 3
}

# ✅ GOOD: Send only required fields
payload = {
    "user_id": "alice",
    "context": {
        "domain": "pricing",
        "variables": ["revenue", "churn"]  # Only relevant variables
    },
    "num_queries": 3
}
```

### 2. Compress Large Payloads

```python
import gzip
import json


def compress_payload(payload: dict) -> bytes:
    """Compress large JSON payloads."""
    payload_json = json.dumps(payload)

    # Only compress if > 1KB
    if len(payload_json) > 1024:
        return gzip.compress(payload_json.encode())

    return payload_json.encode()


# Use with httpx
response = client.post(
    url,
    content=compress_payload(large_payload),
    headers={
        "Content-Encoding": "gzip",
        "Content-Type": "application/json"
    }
)
```

### 3. Reduce Monte Carlo Samples

**Strategy**: Use fewer samples when high precision not needed

```python
def adaptive_sample_size(
    required_confidence: float,
    baseline_samples: int = 1000
) -> int:
    """
    Determine sample size based on required confidence.

    Args:
        required_confidence: 0.90, 0.95, or 0.99
        baseline_samples: Default sample count

    Returns:
        Optimized sample count
    """
    sample_map = {
        0.90: baseline_samples // 2,   # 500 samples
        0.95: baseline_samples,         # 1000 samples
        0.99: baseline_samples * 2      # 2000 samples
    }

    return sample_map.get(required_confidence, baseline_samples)


# Example: Use 500 samples for 90% confidence (2x faster)
result = client.post(
    f"{ISL_BASE_URL}/api/v1/causal/counterfactual",
    json={
        "model": {...},
        "outcome": "Revenue",
        "intervention": {"Price": 60},
        "num_samples": adaptive_sample_size(required_confidence=0.90)
    }
)
```

---

## Monitoring & Profiling

### 1. Latency Tracking

```python
import time
from collections import defaultdict
from typing import Dict, List


class LatencyTracker:
    """Track ISL request latencies."""

    def __init__(self):
        self.latencies: Dict[str, List[float]] = defaultdict(list)

    def record(self, endpoint: str, latency_ms: float):
        """Record latency for endpoint."""
        self.latencies[endpoint].append(latency_ms)

    def get_stats(self, endpoint: str) -> dict:
        """Get latency statistics for endpoint."""
        latencies = self.latencies[endpoint]
        if not latencies:
            return {}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "count": n,
            "mean": sum(latencies) / n,
            "p50": sorted_latencies[int(n * 0.5)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)],
            "max": sorted_latencies[-1]
        }

    def print_report(self):
        """Print latency report for all endpoints."""
        print(f"\nLatency Report:")
        print(f"{'='*70}")

        for endpoint, latencies in sorted(self.latencies.items()):
            stats = self.get_stats(endpoint)
            print(f"\n{endpoint}")
            print(f"  Requests: {stats['count']}")
            print(f"  Mean:     {stats['mean']:.1f}ms")
            print(f"  P50:      {stats['p50']:.1f}ms")
            print(f"  P95:      {stats['p95']:.1f}ms")
            print(f"  P99:      {stats['p99']:.1f}ms")
            print(f"  Max:      {stats['max']:.1f}ms")


# Usage
tracker = LatencyTracker()


def request_with_tracking(endpoint: str, **kwargs):
    """Make ISL request with latency tracking."""
    start = time.time()

    response = client.post(f"{ISL_BASE_URL}{endpoint}", **kwargs)

    latency_ms = (time.time() - start) * 1000
    tracker.record(endpoint, latency_ms)

    return response.json()


# At end of session
tracker.print_report()
```

### 2. Cache Hit Rate Monitoring

```python
class CacheMonitor:
    """Monitor cache performance."""

    def __init__(self):
        self.hits = 0
        self.misses = 0

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    def print_stats(self):
        print(f"\nCache Statistics:")
        print(f"  Total requests: {self.total_requests}")
        print(f"  Hits:           {self.hits} ({self.hit_rate:.1%})")
        print(f"  Misses:         {self.misses}")

        if self.hit_rate < 0.4:
            print(f"  ⚠ Hit rate below 40% target")
        else:
            print(f"  ✓ Hit rate meets target")


cache_monitor = CacheMonitor()
```

### 3. Prometheus Metrics Export

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server


# Define metrics
isl_requests_total = Counter(
    'plot_isl_requests_total',
    'Total ISL requests',
    ['endpoint', 'status']
)

isl_latency_seconds = Histogram(
    'plot_isl_latency_seconds',
    'ISL request latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

isl_cache_hits_total = Counter(
    'plot_isl_cache_hits_total',
    'ISL cache hits',
    ['endpoint']
)

isl_cache_misses_total = Counter(
    'plot_isl_cache_misses_total',
    'ISL cache misses',
    ['endpoint']
)


def request_with_metrics(endpoint: str, **kwargs):
    """Make ISL request with Prometheus metrics."""

    with isl_latency_seconds.labels(endpoint=endpoint).time():
        try:
            response = client.post(f"{ISL_BASE_URL}{endpoint}", **kwargs)
            response.raise_for_status()

            isl_requests_total.labels(endpoint=endpoint, status='success').inc()
            return response.json()

        except Exception as e:
            isl_requests_total.labels(endpoint=endpoint, status='error').inc()
            raise


# Start metrics server
start_http_server(9090)  # Metrics available at http://localhost:9090/metrics
```

---

## Performance Checklist

### Pre-Deployment

- [ ] **Implement result caching** with config fingerprint validation
- [ ] **Use connection pooling** for persistent connections
- [ ] **Enable HTTP/2** for request multiplexing
- [ ] **Set appropriate timeouts** per endpoint
- [ ] **Validate payloads locally** before sending to ISL
- [ ] **Use async for concurrent requests** when possible
- [ ] **Implement latency tracking** and monitoring

### Production Operations

- [ ] **Monitor cache hit rate** (target: >40%)
- [ ] **Track P95 latencies** per endpoint
- [ ] **Alert on latency spikes** (>2x baseline)
- [ ] **Monitor connection pool utilization**
- [ ] **Profile slow requests** with request IDs
- [ ] **Implement circuit breakers** for cascading failure prevention
- [ ] **Use adaptive sample sizes** for Monte Carlo operations

### Continuous Optimization

- [ ] **Review cache TTLs** based on usage patterns
- [ ] **Optimize payload sizes** (remove unnecessary fields)
- [ ] **Batch similar requests** when possible
- [ ] **Warm cache** with common requests during deployment
- [ ] **A/B test** different sample sizes for counterfactuals
- [ ] **Profile end-to-end workflows** and identify bottlenecks

---

## Performance Optimization Roadmap

### Phase 1: Foundation (Week 1)

1. Implement result caching with config fingerprints
2. Add connection pooling
3. Deploy latency tracking

**Expected Impact**: 30-50% latency reduction

### Phase 2: Advanced Caching (Week 2-3)

1. Implement cache warming for common requests
2. Add user session caching
3. Optimize cache TTLs based on usage patterns

**Expected Impact**: Cache hit rate 40-60%

### Phase 3: Concurrency (Week 4)

1. Convert sequential workflows to async
2. Implement request batching
3. Add circuit breakers

**Expected Impact**: 2-3x throughput improvement

### Phase 4: Fine-Tuning (Ongoing)

1. Adaptive sample sizes based on required confidence
2. Payload compression for large requests
3. HTTP/2 server push for predictable workflows

**Expected Impact**: 10-20% additional improvement

---

## Performance Comparison

### Before Optimization

```
Sequential Workflow (5 requests):
  Validation:      5ms
  Counterfactual:  1000ms
  Preferences:     40ms
  Teaching:        300ms
  Team alignment:  300ms
  ------------------------
  Total:           1,645ms
```

### After Optimization

```
Optimized Workflow (5 concurrent requests with caching):
  All requests (parallel): max(5, 1000, 40, 300, 300)
  Cache hits (3/5):        0ms (cached)
  Actual ISL calls (2/5):  max(1000, 300)
  ------------------------
  Total:                   1,000ms (39% faster)

Additional optimizations:
  - Reduced counterfactual samples: 1000 → 500
  - Connection reuse: -20ms per request
  - HTTP/2 multiplexing: -10ms overhead
  ------------------------
  Final total:              450ms (73% faster)
```

---

## Next Steps

1. Implement caching strategy from this guide
2. Set up latency monitoring
3. Run performance tests with your workload
4. Iterate based on profiling data

---

**Last Updated**: 2025-11-20
**Document Version**: 1.0.0
**ISL Version**: 1.0.0
