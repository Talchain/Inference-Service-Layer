# ISL Technical Specification

**Version:** 2.1.0
**Updated:** 2025-11-26
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [API Reference](#3-api-reference)
4. [Data Models](#4-data-models)
5. [Middleware](#5-middleware)
6. [Configuration](#6-configuration)
7. [Security](#7-security)
8. [Performance](#8-performance)
9. [Deployment](#9-deployment)

---

## 1. Overview

### Purpose

The Inference Service Layer (ISL) provides causal inference capabilities for:

- **PLoT Engine** - Causal validation, counterfactuals, sensitivity analysis
- **TAE** - Assumption validation, robustness assessment
- **CEE** - Contrastive explanations, causal discovery

### Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Runtime | Python | 3.11+ |
| Framework | FastAPI | 0.104+ |
| Validation | Pydantic | 2.5+ |
| Cache | Redis | 7.2+ |
| Metrics | Prometheus | 2.48+ |

### Capabilities

| Feature | Description | Latency (P95) |
|---------|-------------|---------------|
| Causal Validation | DAG validation, adjustment sets | 13ms |
| Counterfactual | What-if scenario generation | 245ms |
| Sensitivity | Assumption robustness testing | 180ms |
| Explanations | Progressive disclosure (3 levels) | 120ms |
| Discovery | Factor extraction from text | 850ms |

---

## 2. Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Applications                     │
│                  (PLoT, TAE, CEE, UI)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│  Middleware Stack (in order):                                │
│  1. GZip Compression                                         │
│  2. Request Size Limit (10MB default)                        │
│  3. Request Timeout (60s default)                            │
│  4. Memory Circuit Breaker (85% threshold)                   │
│  5. Distributed Tracing (X-Trace-Id)                         │
│  6. API Key Authentication                                   │
│  7. CORS                                                     │
│  8. Rate Limiting (100 req/min)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌──────────────────┐   ┌──────────────────┐
        │   Redis Cache    │   │   Prometheus     │
        │   (TTL: 5min)    │   │   Metrics        │
        └──────────────────┘   └──────────────────┘
```

### Module Structure

```
src/
├── api/                    # HTTP endpoints
│   ├── main.py             # App setup, middleware
│   ├── causal.py           # /api/v1/causal/*
│   ├── validation.py       # /api/v1/validation/*
│   ├── counterfactual.py   # /api/v1/counterfactual/*
│   ├── sensitivity.py      # /api/v1/sensitivity/*
│   ├── explanations.py     # /api/v1/explanations/*
│   └── discovery.py        # /api/v1/discovery/*
│
├── services/               # Business logic
│   ├── causal_validator.py
│   ├── counterfactual_engine.py
│   ├── sensitivity_analyzer.py
│   ├── explanation_generator.py
│   └── causal_representation_learner.py
│
├── models/                 # Pydantic schemas
│   ├── requests.py
│   ├── responses.py
│   └── shared.py
│
├── middleware/             # Cross-cutting concerns
│   ├── auth.py             # API key authentication
│   ├── rate_limiting.py    # Rate limiting
│   └── request_limits.py   # Timeout, size limits
│
└── config/                 # Configuration
    └── __init__.py         # Settings class
```

---

## 3. API Reference

### Authentication

All `/api/v1/*` endpoints require the `X-API-Key` header:

```http
POST /api/v1/validation/assumptions
Content-Type: application/json
X-API-Key: your_api_key_here
```

**Public endpoints (no auth required):**
- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`

### Endpoints

#### Validation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/validation/assumptions` | POST | Validate causal assumptions |
| `/api/v1/validation/identifiability` | POST | Check effect identifiability |
| `/api/v1/validation/validate` | POST | Advanced model validation |

**Request Example:**
```json
{
  "dag": {
    "nodes": ["Marketing", "Price", "Sales"],
    "edges": [["Marketing", "Sales"], ["Price", "Sales"]]
  },
  "treatment": "Marketing",
  "outcome": "Sales"
}
```

**Response Example:**
```json
{
  "schema": "validation.v1",
  "is_valid": true,
  "assumptions": [
    {
      "name": "unconfoundedness",
      "status": "satisfied",
      "confidence": 0.92
    }
  ],
  "metadata": {
    "isl_version": "2.1.0",
    "request_id": "req_abc123",
    "timestamp": "2025-11-26T10:00:00Z"
  }
}
```

#### Counterfactual

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/counterfactual/generate` | POST | Generate counterfactual scenarios |
| `/api/v1/counterfactual/goal-seek` | POST | Find interventions for target outcome |
| `/api/v1/counterfactual/batch` | POST | Batch generation |

#### Sensitivity

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sensitivity/analyze` | POST | Full sensitivity analysis |
| `/api/v1/sensitivity/elasticity` | POST | Elasticity calculation |

**Response includes:**
- `elasticity` - How outcome changes per unit violation
- `robustness_score` - 0-1 score (higher = more robust)
- `critical` - Boolean flag for sensitive assumptions

#### Explanations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/explanations/progressive` | POST | Multi-level explanations |
| `/api/v1/explanations/quality` | POST | Readability assessment |

**Explanation Levels:**
- `simple` - Non-technical, plain language
- `intermediate` - Some technical terms
- `technical` - Full statistical notation

#### Discovery

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/discovery/extract-factors` | POST | Extract factors from text |
| `/api/v1/discovery/suggest-dag` | POST | Suggest DAG structure |

### Response Schema

All responses follow this structure:

```json
{
  "schema": "<endpoint>.v1",
  "data": { ... },
  "metadata": {
    "isl_version": "2.1.0",
    "request_id": "req_...",
    "timestamp": "...",
    "processing_time_ms": 45
  }
}
```

### Error Responses

```json
{
  "schema": "error.v1",
  "code": "VALIDATION_ERROR",
  "message": "Human-readable error description",
  "details": { ... },
  "retryable": false,
  "suggested_action": "fix_input"
}
```

**Error Codes:**
| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| `VALIDATION_ERROR` | 400 | No | Invalid input |
| `UNAUTHORIZED` | 401 | No | Missing/invalid API key |
| `RATE_LIMITED` | 429 | Yes | Rate limit exceeded |
| `REQUEST_TIMEOUT` | 504 | Yes | Processing timeout |
| `INTERNAL_ERROR` | 500 | Yes | Server error |

---

## 4. Data Models

### DAG Structure

```python
class DAGStructure(BaseModel):
    nodes: List[str]  # Variable names
    edges: List[Tuple[str, str]]  # (parent, child) pairs

    # Constraints:
    # - Max 50 nodes
    # - Max 200 edges
    # - Must be acyclic
```

### Causal Model

```python
class CausalModel(BaseModel):
    dag: DAGStructure
    treatment: str  # Must be in nodes
    outcome: str    # Must be in nodes
    confounders: Optional[List[str]] = None
```

### Sensitivity Metric

```python
class SensitivityMetric(BaseModel):
    assumption: str
    elasticity: float
    robustness_score: float  # 0-1
    critical: bool
    violation_details: List[ViolationDetail]
```

---

## 5. Middleware

### Middleware Stack Order

```python
# Applied in reverse order (last added = first executed)
1. RateLimitMiddleware      # Check rate limits
2. CORSMiddleware           # CORS headers
3. APIKeyAuthMiddleware     # Authenticate
4. TracingMiddleware        # Add trace ID
5. MemoryCircuitBreaker     # Memory protection
6. RequestTimeoutMiddleware # Timeout enforcement
7. RequestSizeLimitMiddleware # Size limits
8. GZipMiddleware           # Compression
```

### API Key Authentication

```python
# Supports both environment variables:
ISL_API_KEYS = "key1,key2,key3"  # Preferred (comma-separated)
ISL_API_KEY = "single_key"       # Legacy (still supported)

# ISL_API_KEYS takes precedence if both are set
```

### Rate Limiting

- **Default:** 100 requests/minute per client
- **Identification:** API key (if present) or IP address
- **Headers returned:**
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`
  - `Retry-After` (when limited)

### Request Timeout

- **Default:** 60 seconds
- **Configurable via:** `REQUEST_TIMEOUT_SECONDS`
- **Exempt paths:** `/health`, `/ready`, `/metrics`
- **Response on timeout:** 504 with `REQUEST_TIMEOUT` code

---

## 6. Configuration

### Environment Variables

#### Required (Production)

| Variable | Description |
|----------|-------------|
| `ISL_API_KEYS` | Comma-separated API keys |
| `CORS_ORIGINS` | Comma-separated allowed origins |

#### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `ISL_API_KEY` | - | Legacy single API key |
| `ENVIRONMENT` | development | development/staging/production |
| `LOG_LEVEL` | INFO | DEBUG/INFO/WARNING/ERROR |
| `REQUEST_TIMEOUT_SECONDS` | 60 | Max request duration |
| `MAX_REQUEST_SIZE_MB` | 10 | Max request body size |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | 100 | Rate limit |
| `REDIS_HOST` | localhost | Redis server |
| `REDIS_PORT` | 6379 | Redis port |
| `REDIS_PASSWORD` | - | Redis password |

### Settings Class

```python
from src.config import get_settings

settings = get_settings()
print(settings.ISL_API_KEYS)
print(settings.CORS_ORIGINS)
```

---

## 7. Security

### Authentication

- API key via `X-API-Key` header
- Keys stored in `ISL_API_KEYS` or `ISL_API_KEY` env var
- Constant-time comparison to prevent timing attacks

### Authorization

- All `/api/v1/*` endpoints require valid API key
- Per-API-key rate limiting
- No role-based access (all keys have equal access)

### CORS

```python
# Production configuration
CORS_ORIGINS = "https://plot.olumi.ai,https://tae.olumi.ai,https://cee.olumi.ai"

# Development (localhost allowed by default)
CORS_ORIGINS = "http://localhost:3000,http://localhost:8080"

# NEVER use wildcards (*) in production
```

### Input Validation

| Limit | Value | Purpose |
|-------|-------|---------|
| Max nodes | 50 | Prevent DoS |
| Max edges | 200 | Prevent DoS |
| Max string length | 10,000 | Prevent memory exhaustion |
| Max request size | 10MB | Prevent DoS |

### Security Headers

Automatically added:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Strict-Transport-Security` (in production)

### Logging

- **PII redaction:** Email, phone, SSN patterns redacted
- **JSON format:** Structured logging for aggregation
- **Correlation:** Request ID in all log entries

---

## 8. Performance

### Benchmarks

| Endpoint | P50 | P95 | P99 | Throughput |
|----------|-----|-----|-----|------------|
| Validation | 8ms | 13ms | 18ms | 120 req/s |
| Counterfactual | 125ms | 245ms | 380ms | 8 req/s |
| Sensitivity | 95ms | 180ms | 290ms | 11 req/s |
| Explanations | 65ms | 120ms | 185ms | 15 req/s |
| Discovery | 450ms | 850ms | 1.2s | 2 req/s |

### Caching

- **Backend:** Redis
- **TTL:** 5 minutes default
- **Key format:** `isl:{endpoint}:{hash(request)}`
- **Hit rate:** ~78% after warmup

### Scaling

- **Horizontal:** Stateless design, scale replicas
- **Redis:** Shared cache across replicas
- **Auto-scaling:** CPU > 70% or Memory > 80%

---

## 9. Deployment

### Docker

```bash
# Build
docker build -t isl:latest .

# Run
docker run -p 8000:8000 \
  -e ISL_API_KEYS=your_key \
  -e CORS_ORIGINS=https://your-domain.com \
  isl:latest
```

### Docker Compose

```bash
# Full stack (app + redis + monitoring)
docker-compose up -d

# Just the app
docker-compose up -d isl-api
```

### Kubernetes

See `k8s/` directory for manifests.

### Health Checks

```bash
# Liveness
GET /health
# → {"status": "healthy"}

# Readiness
GET /ready
# → {"status": "ready", "checks": {...}}

# Metrics (Prometheus format)
GET /metrics
```

### Production Checklist

- [ ] `ISL_API_KEYS` set with secure keys
- [ ] `CORS_ORIGINS` configured for production domains
- [ ] `ENVIRONMENT=production`
- [ ] Redis configured and accessible
- [ ] Prometheus scraping `/metrics`
- [ ] Health checks configured in orchestrator
- [ ] TLS termination at load balancer

---

## Appendix: Quick Reference

### Common cURL Commands

```bash
# Health check
curl https://isl.example.com/health

# Validate model
curl -X POST https://isl.example.com/api/v1/validation/assumptions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{"dag": {"nodes": ["A","B"], "edges": [["A","B"]]}, "treatment": "A", "outcome": "B"}'

# Sensitivity analysis
curl -X POST https://isl.example.com/api/v1/sensitivity/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{"model": {...}, "data": [...]}'
```

### Environment Template

```bash
# .env
ENVIRONMENT=production
ISL_API_KEYS=key1,key2
CORS_ORIGINS=https://app.example.com
REDIS_HOST=redis.example.com
REDIS_PORT=6379
LOG_LEVEL=INFO
REQUEST_TIMEOUT_SECONDS=60
RATE_LIMIT_REQUESTS_PER_MINUTE=100
```

---

**End of Technical Specification**
