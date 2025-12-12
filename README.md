# Inference Service Layer (ISL)

**Causal inference API for Olumi's decision platform**

---

## Quick Start

```bash
# 1. Clone & run
git clone https://github.com/Talchain/Inference-Service-Layer.git
cd Inference-Service-Layer
docker-compose up -d

# 2. Verify
curl http://localhost:8000/health

# 3. Test endpoint
curl -X POST http://localhost:8000/api/v1/validation/assumptions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{"dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]}, "treatment": "A", "outcome": "B"}'
```

### Local vs. Production Configuration

| Setting | Local Development | Production |
|---------|-------------------|------------|
| `ISL_AUTH_DISABLED` | `true` | **Never set** (or `false`) |
| `ISL_API_KEYS` | Not required | **Required** - comma-separated keys |
| `CORS_ORIGINS` | `http://localhost:3000` | Your app domain(s), no wildcards |
| `REDIS_HOST` | `localhost` (optional) | **Required** - production Redis |
| `SENTRY_ENABLED` | `false` | `true` (recommended) |
| `ENVIRONMENT` | `development` | `production` |

**Minimum local setup:**
```bash
# .env file for local development
ISL_AUTH_DISABLED=true
```

**Minimum production setup:**
```bash
# Environment variables for production
ISL_API_KEYS=key1,key2,key3
CORS_ORIGINS=https://app.example.com
REDIS_HOST=redis.example.com
ENVIRONMENT=production
SENTRY_ENABLED=true
SENTRY_DSN=https://...@sentry.io
```

> ⚠️ **Startup will fail** if neither `ISL_API_KEYS` nor `ISL_AUTH_DISABLED=true` is set.

---

## What ISL Does

| Capability | Endpoint | Description |
|------------|----------|-------------|
| **Multi-Criteria Analysis** | `/api/v1/analysis/*` | Dominance, Pareto, risk adjustment, thresholds |
| **Continuous Optimization** | `/api/v1/analysis/optimise` | Grid search with constraints, confidence intervals, sensitivity |
| **Y₀ Identifiability** | `/api/v1/analysis/identifiability` | Causal effect identifiability analysis with hard rule |
| **Decision Robustness** | `/api/v1/analysis/robustness` | Unified sensitivity, robustness bounds, VoI, Pareto |
| **Outcome Logging** | `/api/v1/outcomes/*` | Log decisions and outcomes for calibration |
| **Aggregation** | `/api/v1/aggregation/*` | Multi-criteria scoring (sum/product/lexicographic) |
| **Validation** | `/api/v1/validation/*` | Validate causal DAGs, constraints, coherence |
| **Feasibility Checking** | `/api/v1/validation/feasibility` | Check options against business constraints |
| **Coherence Analysis** | `/api/v1/validation/coherence` | Analyze ranking stability and close races |
| **Correlation Validation** | `/api/v1/validation/correlations` | Validate factor correlations (PSD check) |
| **Utility Functions** | `/api/v1/utility/validate` | Validate multi-goal utility specifications |
| **Counterfactuals** | `/api/v1/counterfactual/*` | "What-if" analysis |
| **Sensitivity** | `/api/v1/sensitivity/*` | Test assumption robustness |
| **Explanations** | `/api/v1/explanations/*` | Multi-level explanations |
| **Discovery** | `/api/v1/discovery/*` | Extract factors from text |

---

## API Testing & Development

### Postman Collection

Import the complete API collection for interactive testing and development:

```bash
# Import into Postman
1. Open Postman → Import
2. Select "docs/postman_collection.json"
3. Choose environment:
   - Local: http://localhost:8000
   - Staging: https://isl-staging.onrender.com
   - Production: https://isl-production.onrender.com
```

**Included Endpoints:**
- ✅ Dominance Detection - Identify dominated options
- ✅ Pareto Frontier - Find non-dominated options
- ✅ Risk Adjustment - Certainty equivalents with risk profiles
- ✅ Threshold Identification - Parameter sensitivity analysis
- ✅ Multi-Criteria Aggregation - Weighted sum/product/lexicographic
- ✅ Continuous Optimization - Grid search with constraints and sensitivity
- ✅ Y₀ Identifiability - Causal effect identifiability with hard rule enforcement
- ✅ Feasibility Checking - Validate options against constraints
- ✅ Coherence Analysis - Ranking stability and close race detection
- ✅ Utility Validation - Multi-goal utility function specs
- ✅ Correlation Validation - Factor correlation groups with PSD check

**Features:**
- Pre-configured example requests for all endpoints
- Environment variables for easy switching between deployments
- Sample responses with realistic data
- Auto-generated request IDs

**Collection file:** `docs/postman_collection.json`

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[Getting Started](docs/GETTING_STARTED.md)** | New developer setup (5 min) |
| **[Technical Spec](docs/TECHNICAL_SPECIFICATION.md)** | Architecture & API reference |
| **[API Examples](docs/api/EXAMPLES.md)** | Working code examples |
| **[Integration](docs/integration/GUIDE.md)** | PLoT/TAE/CEE integration |
| **[Deployment](docs/operations/DEPLOYMENT.md)** | Docker/K8s deployment |
| **[Security](docs/security/GUIDE.md)** | Auth & rate limiting |

---

## Configuration

```bash
# Required (production)
ISL_API_KEYS=key1,key2           # API keys (comma-separated)
CORS_ORIGINS=https://your-app.com # Allowed origins

# Optional
ISL_API_KEY=single_key           # Legacy single key (still supported)
REQUEST_TIMEOUT_SECONDS=60       # Request timeout
REDIS_HOST=localhost             # Redis for caching

# Error Tracking (optional)
SENTRY_ENABLED=true              # Enable Sentry
SENTRY_DSN=https://...@sentry.io # Sentry DSN
```

### Authentication

API authentication is **required by default**. The service will fail to start without valid configuration.

| Configuration | Behavior |
|---------------|----------|
| `ISL_API_KEYS=key1,key2` | Auth enabled with specified keys |
| `ISL_AUTH_DISABLED=true` | Auth disabled (local dev only) |
| Neither set | **Startup fails** with RuntimeError |

**Public endpoints** (no auth required): `/health`, `/metrics`, `/docs`, `/openapi.json`, `/redoc`

```bash
# Production - keys required
ISL_API_KEYS=prod_key_1,prod_key_2

# Local development - explicitly disable auth
ISL_AUTH_DISABLED=true
```

> ⚠️ **Security:** Never set `ISL_AUTH_DISABLED=true` in production.

### Production Configuration Validation

ISL enforces fail-closed security in production. The `validate_production_config()` method (in `src/config/__init__.py:199-235`) checks:

| Check | Requirement | Failure Mode |
|-------|-------------|--------------|
| API Keys | `ISL_API_KEYS` or `ISL_API_KEY` must be set | Startup fails |
| Auth Disabled | `ISL_AUTH_DISABLED` must be false | Startup fails |
| CORS Origins | No wildcards (`*`) allowed | Startup fails |
| CORS Localhost | No localhost origins in production | Startup fails |
| Redis Host | Must be configured (not localhost) | Startup fails |

**CI Safeguards:** The `.github/workflows/config-validation.yml` workflow validates deployment configurations:
- Blocks `ISL_AUTH_DISABLED=true` in non-dev configs
- Warns about missing `SENTRY_ENABLED=true`
- Detects hardcoded secrets in manifests
- Validates required environment variables are documented

---

## Project Structure

```
src/
├── api/          # HTTP endpoints
├── services/     # Business logic
├── models/       # Pydantic schemas
├── middleware/   # Auth, rate limiting
└── config/       # Configuration

docs/
├── GETTING_STARTED.md
├── TECHNICAL_SPECIFICATION.md
├── api/EXAMPLES.md
├── integration/GUIDE.md
├── operations/DEPLOYMENT.md
└── security/GUIDE.md
```

### Key Source Files

| Feature | Implementation | Contract/Schema |
|---------|---------------|-----------------|
| **Error Responses** | `src/middleware/error_handler.py` | `src/models/responses.py` |
| **Request ID Middleware** | `src/api/main.py:45-80` | - |
| **Authentication** | `src/middleware/auth.py` | `src/config/__init__.py` |
| **Rate Limiting** | `src/middleware/rate_limiter.py` | - |
| **Sentry Integration** | `src/utils/tracing.py` | `src/config/__init__.py:57-76` |
| **Configuration** | `src/config/__init__.py` | `Settings` class |
| **Production Validation** | `src/config/__init__.py:199-235` | `validate_production_config()` |
| **Decision Robustness** | `src/services/decision_robustness_analyzer.py` | `src/models/decision_robustness.py` |
| **Identifiability** | `src/services/identifiability_analyzer.py` | `src/models/responses.py` |
| **Outcome Logging** | `src/services/outcome_logger.py` | `src/models/decision_robustness.py` |

---

## Testing

### Running Tests

```bash
# All tests
pytest -o addopts=""

# Unit tests only (fast, no external deps)
pytest tests/unit/ -v

# Integration tests (requires app context)
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_decision_robustness_analyzer.py -v

# With coverage report
pytest --cov=src --cov-report=html
```

### Test Categories

| Category | Path | Count | Description |
|----------|------|-------|-------------|
| Unit | `tests/unit/` | 120+ | Service logic, models, validators |
| Integration | `tests/integration/` | 95+ | API endpoints, error handling |
| Smoke | `tests/smoke/` | 10+ | Health checks, production readiness |

### Smoke Testing Endpoints

Quick verification that critical endpoints respond:
```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Authenticated endpoint
curl -X POST http://localhost:8000/api/v1/analysis/dominance \
  -H "X-API-Key: $ISL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"options": [{"option_id": "A", "option_label": "A", "scores": {"x": 0.5}}], "criteria": ["x"]}'
```

**Total Coverage:** 215+ tests

---

## Security

### Key Controls

| Control | Implementation | Configuration |
|---------|----------------|---------------|
| **Authentication** | API key validation via `X-API-Key` header | `ISL_API_KEYS` env var |
| **Rate Limiting** | Token bucket per IP (100 req/min default) | `RATE_LIMIT_*` env vars |
| **Request Size** | 10MB max payload | `MAX_REQUEST_SIZE_MB` |
| **Timeouts** | 60s default, endpoint-specific overrides | `REQUEST_TIMEOUT_SECONDS` |
| **CORS** | Explicit origin allowlist | `CORS_ORIGINS` |
| **Audit Logging** | Structured JSON logs with request IDs | Automatic |

### API Key Management

```bash
# Multiple keys for rotation
ISL_API_KEYS=active_key,old_key_for_rotation

# Key rotation process:
# 1. Add new key: ISL_API_KEYS=new_key,old_key
# 2. Update clients to use new_key
# 3. Remove old key: ISL_API_KEYS=new_key
```

### Rate Limiting

Default limits (configurable via environment):
- **100 requests/minute** per IP
- **1000 requests/hour** per IP
- Response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`

```bash
# Custom rate limits
RATE_LIMIT_REQUESTS_PER_MINUTE=200
RATE_LIMIT_REQUESTS_PER_HOUR=2000
```

For detailed security documentation, see **[docs/security/GUIDE.md](docs/security/GUIDE.md)**.

---

## Observability

### Request Tracing

Every request is assigned a unique request ID for debugging and correlation.

#### Request ID Format
```
req_{uuid16}
Example: req_a1b2c3d4e5f6g7h8
```

#### Headers

| Header | Direction | Purpose |
|--------|-----------|---------|
| `X-Request-Id` | Request/Response | Primary request identifier |
| `X-Trace-Id` | Request/Response | Legacy (mapped to X-Request-Id, deprecated) |
| `X-User-Id` | Request | Optional user context |
| `X-RateLimit-Limit` | Response | Rate limit ceiling |
| `X-RateLimit-Remaining` | Response | Remaining requests in window |

#### Providing a Request ID

Clients can optionally provide their own request ID:
```bash
curl -X POST https://isl.example.com/api/v1/causal/validate \
  -H "X-Request-Id: req_mycustomid12345" \
  -H "X-API-Key: $ISL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dag": {...}}'
```

If not provided, ISL generates one automatically.

#### Correlation with Sentry

When Sentry is enabled, the request ID is attached to all error events:
```
request_id: req_a1b2c3d4e5f6g7h8
```

This allows correlating user-reported errors with server-side logs.

### Logging

ISL uses structured JSON logging. Each log entry includes:
```json
{
  "timestamp": "2024-12-11T10:30:00Z",
  "level": "INFO",
  "request_id": "req_a1b2c3d4e5f6g7h8",
  "message": "Request completed",
  "duration_ms": 45,
  "status_code": 200
}
```

### Metrics

Prometheus metrics are exposed at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total requests by method, endpoint, status |
| `http_request_duration_seconds` | Histogram | Request duration by method, endpoint |
| `isl_rate_limit_hits_total` | Counter | Rate limit rejections |
| `isl_rate_limit_checks_total` | Counter | All rate limit checks |

For detailed observability documentation, see **[docs/observability.md](docs/observability.md)**.

---

## Deployment

| Environment | URL |
|-------------|-----|
| Staging | https://isl-staging.onrender.com |
| Health | https://isl-staging.onrender.com/health |
| Docs | https://isl-staging.onrender.com/docs |

---

**Version:** 2.1.0 | **Status:** Production Ready
