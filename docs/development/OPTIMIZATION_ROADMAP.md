# ISL Performance Optimization Roadmap

**Document Version:** 1.0
**Last Updated:** 2025-01-20
**Owner:** ISL Development Team
**Status:** Planning

---

## Executive Summary

This roadmap outlines a 4-phase performance optimization strategy for the Inference Service Layer (ISL). Each phase builds on the previous, targeting different optimization opportunities from quick wins to advanced techniques.

**Current Baseline Performance:**
- Causal Validation P95: ~0.5-1.0s
- Counterfactual Analysis P95: ~1.0-2.0s
- Preference Elicitation P95: ~1.5-3.0s

**Optimization Targets:**
- Phase 1: 20-30% latency reduction
- Phase 2: 40-50% latency reduction
- Phase 3: 60-70% latency reduction
- Phase 4: 80%+ latency reduction (for cached requests)

---

## Phase 1: Quick Wins (1-2 weeks)

**Goal:** Low-hanging fruit optimizations requiring minimal code changes.

**Target:** 20-30% latency reduction for common operations.

### 1.1 Response Serialization Optimization

**Problem:** Pydantic v2 model serialization overhead for large response objects.

**Solution:**
```python
# Current (implicit serialization):
return CausalValidationResponse(**result)

# Optimized (explicit serialization with exclude_none):
return CausalValidationResponse(**result).model_dump(
    mode='json',
    exclude_none=True,
    by_alias=True
)
```

**Expected Impact:** 10-15% reduction in response time for large objects.

**Implementation:**
- Add `exclude_none=True` to all response models
- Use `model_dump_json()` directly where possible
- Benchmark before/after

---

### 1.2 JSON Parsing Optimization

**Problem:** Standard library `json` module is slower than alternatives.

**Solution:**
```python
# Replace standard library json with orjson
import orjson

# Fast serialization
def serialize_response(data: dict) -> str:
    return orjson.dumps(data).decode()

# Fast deserialization
def parse_request(payload: str) -> dict:
    return orjson.loads(payload)
```

**Expected Impact:** 5-10% reduction in JSON handling time.

**Implementation:**
- Add `orjson` dependency
- Replace `json.dumps/loads` in hot paths
- Benchmark with large payloads

---

### 1.3 Connection Pool Tuning

**Problem:** Creating new HTTP/Redis connections for each request.

**Solution:**
```python
# Redis connection pool (already implemented, tune limits)
REDIS_CONFIG = {
    "max_connections": 100,  # Increase from 50
    "socket_keepalive": True,
    "socket_keepalive_options": {
        1: 1,   # TCP_KEEPIDLE
        2: 1,   # TCP_KEEPINTVL
        3: 3    # TCP_KEEPCNT
    }
}

# HTTP client connection pool (if applicable)
httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20
    )
)
```

**Expected Impact:** 5-10% reduction for high-concurrency scenarios.

**Implementation:**
- Monitor connection pool utilization
- Increase limits based on load testing
- Add metrics for pool exhaustion

---

### 1.4 Lazy Imports

**Problem:** Importing heavy dependencies at module level slows startup.

**Solution:**
```python
# Current (eager imports):
import numpy as np
import scipy.stats

# Optimized (lazy imports):
def compute_statistics():
    import numpy as np
    import scipy.stats
    # ... computation
```

**Expected Impact:** 30-50% reduction in cold start time.

**Implementation:**
- Profile import times with `python -X importtime`
- Move heavy imports to function scope
- Consider `importlib.util.LazyLoader` for frequently-called functions

---

### 1.5 Cache Key Optimization

**Problem:** Complex cache key computation with JSON serialization.

**Solution:**
```python
# Current (JSON-based hash):
def compute_cache_key(data: dict) -> str:
    normalized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]

# Optimized (direct hash):
def compute_cache_key(data: dict) -> str:
    # Use xxhash for faster hashing
    import xxhash
    normalized = json.dumps(data, sort_keys=True)
    return xxhash.xxh64(normalized).hexdigest()[:16]
```

**Expected Impact:** 2-5% reduction in cache overhead.

**Implementation:**
- Add `xxhash` dependency
- Benchmark hash computation
- Ensure backward compatibility with existing cache keys

---

**Phase 1 Success Metrics:**
- [ ] P95 latency reduced by 20-30%
- [ ] Cold start time reduced by 30-50%
- [ ] No regressions in correctness
- [ ] All 119 tests still passing

**Estimated Effort:** 1-2 weeks
**Risk:** Low

---

## Phase 2: Algorithmic Improvements (2-4 weeks)

**Goal:** Optimize core algorithms and mathematical computations.

**Target:** 40-50% total latency reduction (building on Phase 1).

### 2.1 Monte Carlo Sampling Optimization

**Problem:** Monte Carlo sampling is CPU-intensive for large sample sizes.

**Current Implementation:**
```python
# Monte Carlo with 10,000 samples
samples = np.random.normal(loc=mean, scale=std, size=10000)
```

**Optimization Strategy:**

**A. Adaptive Sampling:**
```python
def adaptive_monte_carlo(distribution, target_error=0.01, max_samples=10000):
    """Use adaptive sampling to reduce samples while maintaining accuracy."""
    min_samples = 1000
    current_samples = min_samples

    while current_samples < max_samples:
        samples = distribution.sample(current_samples)
        error_estimate = samples.std() / np.sqrt(current_samples)

        if error_estimate < target_error:
            break

        current_samples = min(current_samples * 2, max_samples)

    return samples
```

**B. Importance Sampling:**
```python
def importance_sampling(target_distribution, proposal_distribution, n_samples=5000):
    """Use importance sampling to focus on high-probability regions."""
    # Sample from proposal
    samples = proposal_distribution.sample(n_samples)

    # Compute importance weights
    weights = target_distribution.pdf(samples) / proposal_distribution.pdf(samples)
    weights /= weights.sum()

    # Weighted statistics
    return samples, weights
```

**Expected Impact:** 30-50% reduction in Monte Carlo computation time.

---

### 2.2 DAG Analysis Caching

**Problem:** Re-computing d-separation and graph properties for similar DAGs.

**Solution:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def compute_d_separation(dag_hash: str, treatment: str, outcome: str) -> bool:
    """Cache d-separation results for DAG structures."""
    dag = deserialize_dag(dag_hash)
    return is_d_separated(dag, treatment, outcome)

def analyze_dag(dag: Dict) -> Dict:
    """Use cached d-separation computation."""
    dag_hash = compute_dag_hash(dag)  # Deterministic hash
    result = compute_d_separation(dag_hash, treatment, outcome)
    return result
```

**Expected Impact:** 20-40% reduction for repeated DAG structures.

---

### 2.3 Vectorized Belief Operations

**Problem:** Iterating over beliefs one-by-one instead of vectorized operations.

**Current Implementation:**
```python
# Scalar operations
for belief_name, belief_data in beliefs.items():
    mean = belief_data['mean']
    std = belief_data['std']
    samples = np.random.normal(mean, std, n_samples)
    results[belief_name] = samples
```

**Optimized Implementation:**
```python
# Vectorized operations
belief_names = list(beliefs.keys())
means = np.array([beliefs[k]['mean'] for k in belief_names])
stds = np.array([beliefs[k]['std'] for k in belief_names])

# Single vectorized sampling call
all_samples = np.random.normal(
    loc=means[:, None],
    scale=stds[:, None],
    size=(len(beliefs), n_samples)
)

results = dict(zip(belief_names, all_samples))
```

**Expected Impact:** 15-25% reduction in belief processing time.

---

### 2.4 Parallel Question Generation

**Problem:** Generating preference questions sequentially.

**Solution:**
```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def generate_questions_parallel(beliefs, dag, num_questions=5):
    """Generate multiple questions in parallel."""
    loop = asyncio.get_event_loop()

    with ProcessPoolExecutor(max_workers=4) as executor:
        # Generate questions in parallel
        tasks = [
            loop.run_in_executor(
                executor,
                generate_single_question,
                beliefs, dag, i
            )
            for i in range(num_questions)
        ]

        questions = await asyncio.gather(*tasks)

    return questions
```

**Expected Impact:** 20-40% reduction in preference elicitation time.

---

### 2.5 Sensitivity Analysis Caching

**Problem:** Computing sensitivity multiple times for same parameters.

**Solution:**
```python
# Cache sensitivity results by parameter hash
sensitivity_cache = {}

def compute_sensitivity(params: Dict, dag: Dict) -> Dict:
    """Compute sensitivity with caching."""
    cache_key = compute_cache_key({"params": params, "dag": dag})

    if cache_key in sensitivity_cache:
        return sensitivity_cache[cache_key]

    result = _compute_sensitivity_uncached(params, dag)
    sensitivity_cache[cache_key] = result

    return result
```

**Expected Impact:** 50-70% reduction for repeated sensitivity queries.

---

**Phase 2 Success Metrics:**
- [ ] P95 latency reduced by 40-50% total
- [ ] Monte Carlo sampling time reduced by 30-50%
- [ ] Vectorized operations cover 80%+ of belief processing
- [ ] All correctness tests passing
- [ ] Numerical accuracy within 1% of baseline

**Estimated Effort:** 2-4 weeks
**Risk:** Medium (requires careful validation of numerical accuracy)

---

## Phase 3: Infrastructure Optimization (3-6 weeks)

**Goal:** Leverage infrastructure for performance gains.

**Target:** 60-70% total latency reduction (building on Phases 1-2).

### 3.1 Redis Result Caching (PRIORITY)

**Problem:** Re-computing identical requests.

**Solution:** Full Redis integration with TTL-based caching.

```python
# Cache structure
CACHE_KEYS = {
    "beliefs": "isl:beliefs:{hash}",      # TTL: 24 hours
    "ident": "isl:ident:{hash}",          # TTL: 6 hours
    "result": "isl:result:{hash}",        # TTL: 2 hours
    "sensitivity": "isl:sensitivity:{hash}" # TTL: 4 hours
}

async def get_or_compute_result(request_hash: str, compute_fn):
    """Get from cache or compute and cache."""
    cache_key = f"isl:result:{request_hash}"

    # Try cache first
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute if not cached
    result = await compute_fn()

    # Cache with TTL
    await redis_client.setex(
        cache_key,
        7200,  # 2 hours
        json.dumps(result)
    )

    return result
```

**Expected Impact:** 80-95% latency reduction for cached requests (P95: <100ms).

**Cache Hit Rate Projections:**
- Pilot (10 users): 20-30% hit rate
- Production (100 users): 40-60% hit rate
- Heavy usage: 60-80% hit rate

---

### 3.2 Async Database/Cache Operations

**Problem:** Synchronous I/O blocking request processing.

**Solution:**
```python
# Replace sync Redis client with async
import aioredis

redis_client = aioredis.from_url(
    REDIS_URL,
    encoding="utf-8",
    decode_responses=True
)

# Use async operations throughout
async def causal_validate(request: CausalValidationRequest):
    # Parallel cache checks
    cache_results = await asyncio.gather(
        redis_client.get(f"isl:ident:{dag_hash}"),
        redis_client.get(f"isl:beliefs:{belief_hash}"),
        return_exceptions=True
    )

    # Process results
    ...
```

**Expected Impact:** 10-20% reduction through parallel I/O.

---

### 3.3 Response Streaming

**Problem:** Large responses buffered entirely before sending.

**Solution:**
```python
from fastapi.responses import StreamingResponse

async def stream_sensitivity_analysis(request):
    """Stream large sensitivity analysis results."""

    async def generate():
        yield '{"drivers": ['

        for i, driver in enumerate(compute_drivers()):
            if i > 0:
                yield ','
            yield json.dumps(driver)

        yield ']}'

    return StreamingResponse(
        generate(),
        media_type="application/json"
    )
```

**Expected Impact:** 30-50% reduction in time-to-first-byte for large responses.

---

### 3.4 Request Batching

**Problem:** Processing multiple requests independently when they could be batched.

**Solution:**
```python
class RequestBatcher:
    """Batch similar requests for processing."""

    def __init__(self, max_batch_size=10, max_wait_ms=50):
        self.batch = []
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

    async def add_request(self, request):
        """Add request to batch and process when ready."""
        self.batch.append(request)

        if len(self.batch) >= self.max_batch_size:
            return await self.process_batch()

        # Wait for more requests (up to max_wait_ms)
        await asyncio.sleep(self.max_wait_ms / 1000)

        if self.batch:
            return await self.process_batch()

    async def process_batch(self):
        """Process all requests in batch."""
        results = await compute_batch(self.batch)
        self.batch = []
        return results
```

**Expected Impact:** 15-30% throughput improvement for high-concurrency scenarios.

---

### 3.5 CDN/Edge Caching

**Problem:** Repeated identical requests from same geographic region.

**Solution:**
```python
# Add cache-control headers for edge caching
@app.get("/api/v1/causal/validate")
async def validate_causal(request: Request, response: Response):
    # Compute result
    result = await compute_validation(request)

    # Add cache headers (for deterministic results)
    response.headers["Cache-Control"] = "public, max-age=3600"
    response.headers["ETag"] = compute_etag(result)

    return result
```

**Expected Impact:** 50-80% latency reduction for edge-cached requests.

---

**Phase 3 Success Metrics:**
- [ ] P95 latency reduced by 60-70% total
- [ ] Cache hit rate >30% within 1 week of pilot
- [ ] Redis availability >99.9%
- [ ] No cache coherency issues
- [ ] All 119 tests passing

**Estimated Effort:** 3-6 weeks
**Risk:** Medium-High (requires Redis provisioning, careful cache invalidation)

---

## Phase 4: Advanced Optimizations (6-12 weeks)

**Goal:** Cutting-edge optimizations for maximum performance.

**Target:** 80%+ latency reduction for cached requests, 70%+ for uncached.

### 4.1 JIT Compilation with Numba

**Problem:** Python interpreter overhead for tight loops.

**Solution:**
```python
from numba import jit

@jit(nopython=True)
def monte_carlo_sample(mean, std, n_samples):
    """JIT-compiled Monte Carlo sampling."""
    samples = np.empty(n_samples)
    for i in range(n_samples):
        samples[i] = np.random.normal(mean, std)
    return samples

# 10-100x speedup for numerical loops
```

**Expected Impact:** 50-90% reduction in numerical computation time.

---

### 4.2 GPU Acceleration

**Problem:** CPU-bound for large-scale Monte Carlo simulations.

**Solution:**
```python
import cupy as cp  # GPU-accelerated NumPy

def monte_carlo_gpu(distributions, n_samples=10000):
    """GPU-accelerated Monte Carlo sampling."""
    # Transfer to GPU
    means_gpu = cp.array([d.mean for d in distributions])
    stds_gpu = cp.array([d.std for d in distributions])

    # Sample on GPU
    samples_gpu = cp.random.normal(
        loc=means_gpu[:, None],
        scale=stds_gpu[:, None],
        size=(len(distributions), n_samples)
    )

    # Transfer back to CPU
    return cp.asnumpy(samples_gpu)
```

**Expected Impact:** 10-100x speedup for large Monte Carlo simulations (100k+ samples).

**Requirements:**
- NVIDIA GPU
- CUDA toolkit
- cupy library

---

### 4.3 Predictive Caching

**Problem:** Cache miss on first request for common patterns.

**Solution:**
```python
class PredictiveCacheWarmer:
    """Pre-compute and cache likely requests."""

    async def warm_cache_for_user(self, user_history):
        """Predict next requests based on user history."""
        # Analyze historical patterns
        likely_next = predict_next_requests(user_history)

        # Pre-compute and cache
        for request in likely_next[:5]:
            await self.compute_and_cache(request)
```

**Expected Impact:** 20-40% increase in cache hit rate.

---

### 4.4 Model Distillation

**Problem:** Complex models requiring expensive computation.

**Solution:**
```python
# Train lightweight surrogate model
def train_surrogate_model(training_data):
    """Train fast approximation of expensive computation."""
    # Use neural network to approximate complex causal inference
    model = train_neural_surrogate(training_data)
    return model

# Use surrogate for fast approximation
def fast_inference(request, use_surrogate=True):
    if use_surrogate:
        # Fast neural network inference (~10ms)
        return surrogate_model.predict(request)
    else:
        # Slow exact computation (~1000ms)
        return exact_computation(request)
```

**Expected Impact:** 90-99% latency reduction with acceptable accuracy trade-off.

**Accuracy Target:** Within 5% of exact computation.

---

### 4.5 Query Planning & Optimization

**Problem:** Inefficient execution order for complex queries.

**Solution:**
```python
class QueryOptimizer:
    """Optimize execution plan for complex requests."""

    def optimize_execution_plan(self, request):
        """Reorder operations for optimal performance."""
        operations = parse_operations(request)

        # Build dependency graph
        dep_graph = build_dependency_graph(operations)

        # Find optimal execution order
        optimal_order = topological_sort_with_parallelization(dep_graph)

        return optimal_order

    async def execute_optimized(self, request):
        """Execute operations in optimal order."""
        plan = self.optimize_execution_plan(request)

        # Execute independent operations in parallel
        results = {}
        for batch in plan.parallel_batches:
            batch_results = await asyncio.gather(*[
                execute_operation(op) for op in batch
            ])
            results.update(batch_results)

        return combine_results(results)
```

**Expected Impact:** 20-50% reduction for complex multi-step requests.

---

**Phase 4 Success Metrics:**
- [ ] P95 latency <500ms for uncached requests
- [ ] P95 latency <50ms for cached requests
- [ ] Throughput >1000 req/sec sustained
- [ ] GPU utilization >70% during peak load (if applicable)
- [ ] Surrogate model accuracy within 5% of exact

**Estimated Effort:** 6-12 weeks
**Risk:** High (requires specialized hardware, careful validation)

---

## Performance Targets Summary

| Endpoint | Baseline P95 | Phase 1 Target | Phase 2 Target | Phase 3 Target | Phase 4 Target |
|----------|--------------|----------------|----------------|----------------|----------------|
| Causal Validation | 1.0s | 0.7-0.8s | 0.5-0.6s | 0.3-0.4s (cached: 0.05s) | 0.2s (cached: 0.02s) |
| Counterfactual | 2.0s | 1.4-1.6s | 1.0-1.2s | 0.6-0.8s (cached: 0.08s) | 0.4s (cached: 0.03s) |
| Preference Elicitation | 3.0s | 2.1-2.4s | 1.5-1.8s | 0.9-1.2s (cached: 0.1s) | 0.6s (cached: 0.05s) |

**Cache Hit Rate Targets:**
- Week 1: 10-20%
- Week 2: 20-30%
- Week 4: 30-50%
- Month 2: 50-70%

---

## Implementation Priority

**Immediate (Phase 1):**
1. ✅ Redis caching strategy documented
2. Response serialization optimization
3. JSON parsing with orjson
4. Connection pool tuning

**Short-term (Phase 2):**
1. Monte Carlo adaptive sampling
2. DAG analysis caching
3. Vectorized belief operations

**Medium-term (Phase 3):**
1. 🚧 Redis integration (IN PROGRESS)
2. Async I/O throughout
3. Response streaming

**Long-term (Phase 4):**
1. JIT compilation exploration
2. GPU acceleration (if workload justifies)
3. Predictive caching

---

## Monitoring & Validation

**Performance Metrics:**
```python
# Add to Prometheus metrics
isl_optimization_phase = Gauge("isl_optimization_phase", "Current optimization phase")
isl_cache_hit_rate = Gauge("isl_cache_hit_rate", "Cache hit rate percentage")
isl_computation_time = Histogram("isl_computation_time_seconds", "Computation time by phase")
```

**Validation Checklist:**
- [ ] All 119 tests pass after each optimization
- [ ] Numerical accuracy within 1% of baseline
- [ ] No memory leaks (profile with memory-profiler)
- [ ] No cache coherency issues
- [ ] Performance gains measured and documented

---

## Rollback Plan

**If optimization causes issues:**
1. Feature flags for each optimization
2. Gradual rollout (10% → 50% → 100%)
3. Automatic rollback on error rate >5%
4. Manual rollback via environment variable

```python
# Feature flags
ENABLE_ORJSON = os.getenv("ENABLE_ORJSON", "true") == "true"
ENABLE_ADAPTIVE_SAMPLING = os.getenv("ENABLE_ADAPTIVE_SAMPLING", "false") == "true"
ENABLE_GPU = os.getenv("ENABLE_GPU", "false") == "true"

# Use flags in code
if ENABLE_ORJSON:
    import orjson as json
else:
    import json
```

---

## Cost-Benefit Analysis

| Phase | Effort (weeks) | Latency Reduction | Risk | ROI |
|-------|----------------|-------------------|------|-----|
| Phase 1 | 1-2 | 20-30% | Low | ⭐⭐⭐⭐⭐ Very High |
| Phase 2 | 2-4 | 40-50% | Medium | ⭐⭐⭐⭐ High |
| Phase 3 | 3-6 | 60-70% | Medium-High | ⭐⭐⭐⭐ High |
| Phase 4 | 6-12 | 80%+ | High | ⭐⭐⭐ Medium (depends on scale) |

**Recommendation:** Prioritize Phases 1-3. Phase 4 only if scale justifies investment.

---

## Next Steps

1. **Immediate:**
   - ✅ Complete Redis strategy documentation
   - Run `profile_performance.py` to establish baseline
   - Implement Phase 1 optimizations

2. **Week 1-2:**
   - Profile improvements after Phase 1
   - Plan Phase 2 algorithm optimizations
   - Validate numerical accuracy

3. **Week 3-6:**
   - Complete Redis integration (Phase 3)
   - Monitor cache hit rates
   - Measure overall performance improvement

4. **Month 2-3:**
   - Evaluate Phase 4 ROI based on pilot usage
   - Consider GPU acceleration if Monte Carlo dominates
   - Continuous profiling and optimization

---

**For optimization questions, contact: #isl-performance**
