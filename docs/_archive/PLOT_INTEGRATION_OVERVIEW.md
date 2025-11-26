# PLoT Integration Overview

**Version**: 1.0.0
**Last Updated**: 2025-11-20
**Status**: Production Ready for Pilot Phase

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Integration Checklist](#integration-checklist)
3. [Quick Start Guide](#quick-start-guide)
4. [Key Concepts](#key-concepts)
5. [API Endpoint Overview](#api-endpoint-overview)
6. [Authentication & Headers](#authentication--headers)
7. [Response Format](#response-format)
8. [Next Steps](#next-steps)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     PLoT (Preference Learning               │
│                     and Optimization Tool)                   │
└────────────┬────────────────────────────────────────────────┘
             │
             │ HTTPS/REST
             │
┌────────────▼────────────────────────────────────────────────┐
│              ISL (Inference Service Layer)                  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Causal     │  │ Preference   │  │   Teaching   │     │
│  │  Inference   │  │  Learning    │  │   Engine     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Team         │  │ Sensitivity  │  │  Validation  │     │
│  │ Alignment    │  │  Analysis    │  │   Engine     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────┬────────────────────────────────────────────────┘
             │
             │ Redis Protocol
             │
┌────────────▼────────────────────────────────────────────────┐
│                    Redis (User State)                       │
│              User beliefs • Session data                    │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Stateless RESTful API**: Each request is independent
2. **Async-First**: FastAPI with full async support for high concurrency
3. **Deterministic**: Same inputs + same config = identical outputs
4. **Observable**: Request IDs, config fingerprints, comprehensive logging
5. **Resilient**: Comprehensive error handling with actionable error messages

### Technology Stack

- **Framework**: FastAPI 0.104+ with Pydantic v2
- **Server**: Uvicorn with uvloop for performance
- **Cache/State**: Redis 7.x
- **Validation**: Pydantic models with strict typing
- **Algorithms**: NetworkX (causal graphs), NumPy/SciPy (inference)

---

## Integration Checklist

### Pre-Integration Phase

- [ ] **Environment Setup**
  - [ ] ISL deployed and accessible
  - [ ] Redis instance running and accessible
  - [ ] Network connectivity verified (port 8000 for ISL, 6379 for Redis)

- [ ] **API Discovery**
  - [ ] Access Swagger docs at `http://<isl-host>:8000/docs`
  - [ ] Review available endpoints
  - [ ] Test health endpoint: `GET /health`

- [ ] **Authentication** (if applicable)
  - [ ] Obtain API keys or tokens
  - [ ] Configure request headers
  - [ ] Test authentication flow

### Integration Phase

- [ ] **Client Implementation**
  - [ ] Choose HTTP client library (httpx, requests, aiohttp)
  - [ ] Implement request/response models matching ISL schemas
  - [ ] Add request ID generation (`X-Request-Id` header)
  - [ ] Implement timeout handling (recommended: 5-10s)

- [ ] **Core Workflows**
  - [ ] Causal model validation workflow
  - [ ] Counterfactual analysis workflow
  - [ ] Preference elicitation workflow
  - [ ] (Optional) Teaching examples workflow
  - [ ] (Optional) Team alignment workflow

- [ ] **Error Handling**
  - [ ] Handle 400 (validation errors)
  - [ ] Handle 422 (Pydantic validation errors)
  - [ ] Handle 500 (internal server errors)
  - [ ] Implement retry logic with exponential backoff
  - [ ] Log errors with request IDs

### Post-Integration Phase

- [ ] **Testing**
  - [ ] Unit tests for client code
  - [ ] Integration tests against ISL
  - [ ] Performance tests (latency, throughput)
  - [ ] Error scenario tests

- [ ] **Monitoring**
  - [ ] Track request latencies
  - [ ] Monitor error rates
  - [ ] Set up alerts for degraded performance
  - [ ] Verify config fingerprints for reproducibility

- [ ] **Documentation**
  - [ ] Document integration patterns used
  - [ ] Create runbooks for common issues
  - [ ] Document configuration requirements

---

## Quick Start Guide

### 1. Verify ISL is Running

```bash
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-11-20T16:30:00Z",
  "config_fingerprint": "a1b2c3d4e5f6"
}
```

### 2. Your First API Call (Causal Validation)

```python
import httpx
import uuid

# Configuration
ISL_BASE_URL = "http://localhost:8000"
REQUEST_ID = f"req_{uuid.uuid4().hex[:12]}"

# Payload
payload = {
    "dag": {
        "nodes": ["Price", "Revenue", "Brand"],
        "edges": [
            ["Price", "Revenue"],
            ["Brand", "Price"],
            ["Brand", "Revenue"]
        ]
    },
    "treatment": "Price",
    "outcome": "Revenue"
}

# Make request
with httpx.Client(timeout=10.0) as client:
    response = client.post(
        f"{ISL_BASE_URL}/api/v1/causal/validate",
        json=payload,
        headers={"X-Request-Id": REQUEST_ID}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Adjustment sets: {result['adjustment_sets']}")
        print(f"Minimal set: {result['minimal_set']}")
    else:
        print(f"Error {response.status_code}: {response.text}")
```

### 3. Next Steps

- Explore [Integration Patterns](PLOT_INTEGRATION_PATTERNS.md) for complete examples
- Review [Error Handling](PLOT_ERROR_HANDLING.md) for robust error management
- Consult [Performance Guide](PLOT_PERFORMANCE_GUIDE.md) for optimization

---

## Key Concepts

### Request IDs

Every request should include an `X-Request-Id` header for distributed tracing:

```python
headers = {"X-Request-Id": f"req_{uuid.uuid4().hex[:12]}"}
```

ISL will generate one if not provided, but explicit IDs help with:
- End-to-end request tracing across PLoT → ISL → Redis
- Debugging and troubleshooting
- Log correlation

### Config Fingerprints

Every response includes a `_metadata` field with a `config_fingerprint`:

```json
{
  "status": "identifiable",
  "adjustment_sets": [...],
  "_metadata": {
    "isl_version": "1.0.0",
    "config_fingerprint": "a1b2c3d4e5f6",
    "config_details": {...},
    "request_id": "req_abc123"
  }
}
```

**Purpose**: Verify that reruns use identical ISL configuration for reproducibility.

**Usage**:
- Store fingerprint alongside results
- Verify fingerprint matches when rerunning analysis
- Alert if fingerprint changes unexpectedly

### Deterministic Computation

ISL is designed for deterministic results:
- Same inputs + same config → identical outputs
- Enables result caching
- Supports reproducible research

**Caveats**:
- Random seed must be set (`RANDOM_SEED` env var)
- Monte Carlo operations use fixed iteration counts
- Ensure `ENABLE_DETERMINISTIC_MODE=true` in production

### User Sessions

For preference learning, ISL maintains user state in Redis:

```python
# First request - initializes user beliefs
response1 = client.post("/api/v1/preferences/elicit", json={
    "user_id": "user_alice",
    "context": {...},
    "num_queries": 3
})

# Second request - updates based on responses
response2 = client.post("/api/v1/preferences/update", json={
    "user_id": "user_alice",
    "query_id": response1.json()["queries"][0]["query_id"],
    "response": {"selected_option": "A"}
})
```

**Key Points**:
- `user_id` must be consistent across requests for the same session
- Beliefs persist in Redis with configurable TTL (default: 24 hours)
- Use `flush_session` endpoint to clear user state if needed

---

## API Endpoint Overview

### Causal Inference (`/api/v1/causal/*`)

| Endpoint | Method | Purpose | P95 Target |
|----------|--------|---------|------------|
| `/validate` | POST | Validate causal model (Y₀) | <2.0s |
| `/counterfactual` | POST | Run counterfactual analysis (FACET) | <2.0s |

**Use Cases**:
- Validate DAG before decision modeling
- Estimate intervention effects
- Test "what if" scenarios

### Preference Learning (`/api/v1/preferences/*`)

| Endpoint | Method | Purpose | P95 Target |
|----------|--------|---------|------------|
| `/elicit` | POST | Generate preference queries | <1.5s |
| `/update` | POST | Update beliefs from responses | <1.5s |

**Use Cases**:
- Learn user value weights
- Refine decision criteria
- Personalize recommendations

### Teaching (`/api/v1/teaching/*`)

| Endpoint | Method | Purpose | P95 Target |
|----------|--------|---------|------------|
| `/teach` | POST | Generate pedagogical examples | <1.5s |

**Use Cases**:
- Explain trade-offs to users
- Build mental models
- Accelerate learning

### Validation (`/api/v1/validation/*`)

| Endpoint | Method | Purpose | P95 Target |
|----------|--------|---------|------------|
| `/validate-model` | POST | Comprehensive model validation | <2.0s |

**Use Cases**:
- Validate DAG structure
- Check statistical assumptions
- Get improvement suggestions

### Team Alignment (`/api/v1/team/*`)

| Endpoint | Method | Purpose | P95 Target |
|----------|--------|---------|------------|
| `/align` | POST | Find common ground across stakeholders | N/A |

**Use Cases**:
- Multi-stakeholder decision making
- Conflict resolution
- Consensus building

### Sensitivity Analysis (`/api/v1/analysis/*`)

| Endpoint | Method | Purpose | P95 Target |
|----------|--------|---------|------------|
| `/sensitivity` | POST | Test assumption robustness | N/A |

**Use Cases**:
- Identify critical assumptions
- Assess conclusion stability
- Find breakpoints

### Health & Monitoring (`/health`)

| Endpoint | Method | Purpose | Latency |
|----------|--------|---------|---------|
| `/health` | GET | Service health check | <10ms |

**Use Cases**:
- Load balancer health checks
- Monitoring and alerting
- Config fingerprint verification

---

## Authentication & Headers

### Required Headers

None currently - ISL is designed for internal service-to-service communication.

### Recommended Headers

```python
headers = {
    "X-Request-Id": "req_abc123",  # For tracing
    "Content-Type": "application/json",  # Always JSON
}
```

### Optional Headers

Future versions may support:
- `Authorization`: API key or JWT token
- `X-Client-Version`: PLoT version for compatibility tracking
- `X-Timeout-Ms`: Per-request timeout override

---

## Response Format

All ISL responses follow this structure:

### Success Response (200 OK)

```json
{
  // Endpoint-specific data
  "status": "identifiable",
  "adjustment_sets": [...],
  "confidence": "high",

  // Explanation (always present)
  "explanation": {
    "summary": "Price → Revenue is identifiable...",
    "reasoning": "...",
    "technical_basis": "...",
    "assumptions": [...]
  },

  // Metadata (always present)
  "_metadata": {
    "isl_version": "1.0.0",
    "config_fingerprint": "a1b2c3d4e5f6",
    "config_details": {
      "max_monte_carlo_iterations": 1000,
      "confidence_level": 0.95
    },
    "request_id": "req_abc123"
  }
}
```

### Error Response (4xx/5xx)

```json
{
  "detail": "Node 'InvalidNode' not found in DAG",
  "error_code": "INVALID_INPUT",
  "request_id": "req_abc123"
}
```

**Common HTTP Status Codes**:
- `200`: Success
- `400`: Invalid input (missing required fields, invalid DAG)
- `422`: Validation error (Pydantic schema mismatch)
- `500`: Internal server error (computation failure)

---

## Next Steps

### For PLoT Developers

1. **Read Integration Patterns**: [PLOT_INTEGRATION_PATTERNS.md](PLOT_INTEGRATION_PATTERNS.md)
   - 5+ complete working examples
   - Copy-paste ready code
   - Common workflows covered

2. **Implement Error Handling**: [PLOT_ERROR_HANDLING.md](PLOT_ERROR_HANDLING.md)
   - Retry strategies
   - Error code reference
   - Circuit breaker patterns

3. **Optimize Performance**: [PLOT_PERFORMANCE_GUIDE.md](PLOT_PERFORMANCE_GUIDE.md)
   - Caching strategies
   - Request batching
   - Connection pooling

### For ISL Operators

1. **Deploy ISL**: Follow deployment guide
2. **Configure Redis**: Set up persistent storage
3. **Monitor Health**: Set up `/health` endpoint monitoring
4. **Review Logs**: Ensure structured logging is captured

### Testing Your Integration

```bash
# 1. Run integration tests
pytest tests/integration/test_plot_integration.py

# 2. Performance validation
python benchmarks/performance_benchmark.py --host http://localhost:8000 --duration 60

# 3. Error scenario tests
python tests/integration/test_error_scenarios.py
```

---

## Support & Resources

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Integration Patterns**: [PLOT_INTEGRATION_PATTERNS.md](PLOT_INTEGRATION_PATTERNS.md)
- **Error Reference**: [PLOT_ERROR_HANDLING.md](PLOT_ERROR_HANDLING.md)
- **Performance Guide**: [PLOT_PERFORMANCE_GUIDE.md](PLOT_PERFORMANCE_GUIDE.md)
- **Source Code**: https://github.com/your-org/Inference-Service-Layer

---

**Last Updated**: 2025-11-20
**Document Version**: 1.0.0
**ISL Version**: 1.0.0
