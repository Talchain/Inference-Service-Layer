# Platform Infrastructure Audit Response

**Workstream:** Inference Service Layer (ISL)
**Date:** 27 November 2025
**Completed by:** Platform Engineering Team

---

## Executive Summary

The Inference Service Layer demonstrates **excellent DevOps maturity** with comprehensive observability, robust testing infrastructure, and production-ready deployment pipelines. The service is currently deployed on Render with automated CI/CD via GitHub Actions, featuring distributed tracing, Prometheus metrics, and extensive monitoring capabilities.

**Overall Infrastructure Maturity:** 9/10

---

## 1. Staging Environment

| Question | Answer |
|----------|--------|
| **Staging URL** | `https://isl-staging.onrender.com` |
| **Hosting platform** | Render (Docker-based deployment) |
| **Auto-deploy from branch?** | Yes - `main` branch auto-deploys to staging via GitHub Actions |
| **Connected to other staging services?** | • Redis 7 (caching layer)<br>• Prometheus (metrics collection)<br>• Grafana (visualization)<br>• Intended integration: PLoT UI (pending) |

### Deployment Architecture

- **CI/CD Pipeline**: GitHub Actions (`.github/workflows/deploy-production.yml`)
- **Pipeline Stages**: Test → Security Scan → Lint → Deploy Staging → Deploy Production
- **Health Checks**: Automated post-deployment validation (10 retries)
- **Smoke Tests**: Automated test suite runs after each deployment
- **Notifications**: Slack notifications for deployment status
- **Rollback**: Manual via GitHub Actions or Render dashboard

### Production URL

- **Production URL**: `https://isl.olumi.com`
- **Deploy Trigger**: Version tags (`v*`) or manual workflow dispatch
- **Additional Steps**: GitHub Release creation, Grafana annotations, deployment records

---

## 2. Observability

| Question | Answer |
|----------|--------|
| **Do you log `X-Request-Id` / correlation IDs?** | **Yes** - Supports both `X-Request-Id` (preferred) and `X-Trace-Id` (legacy) |
| **Do you expose `/metrics` (Prometheus)?** | **Yes** - Full Prometheus endpoint at `/metrics` |
| **Do you use Sentry or equivalent?** | **Yes** - Sentry SDK integrated, enable via `SENTRY_ENABLED=true` |
| **Do you have runbooks documented?** | **Yes** - See [docs/operations/](docs/operations/) |

### Distributed Tracing

**Implementation:** `src/utils/tracing.py`

- **Request ID Format**: `req_{uuid16}` (e.g., `req_a1b2c3d4e5f67890`)
- **Header Support** (priority order):
  - `X-Request-Id`: Primary correlation ID (platform standard)
  - `X-Trace-Id`: Legacy alias (deprecated, for backward compatibility)
  - `X-User-Id`: User context propagation
- **Context Propagation**: ContextVar-based trace storage across async operations
- **Response Headers**: Both `X-Request-Id` and `X-Trace-Id` returned for compatibility
- **Middleware**: `TracingMiddleware` automatically manages trace context

> **Note:** The legacy `trace_{uuid}` format is deprecated. New code uses `req_{uuid16}` format.

**Test Coverage:** Integration tests validate trace ID propagation (`tests/integration/test_production_excellence.py:140-167`)

### Structured Logging

**Implementation:** `src/utils/logging_config.py`

- **Format**: JSON-formatted logs for machine readability
- **Fields**: timestamp, level, logger, module, function, line, request_id, endpoint, duration_ms, user_hash, status_code
- **Privacy**: GDPR-compliant with PII hashing (`src/utils/secure_logging.py`)
- **Security**: User IDs hashed (SHA-256), no sensitive data in logs
- **Exception Tracking**: Full stack traces preserved in structured format

### Prometheus Metrics

**Endpoint:** `/metrics` (port 8000)
**Configuration:** `monitoring/prometheus/prometheus.yml` (15-second scrape interval)

**Metrics Exposed:**

1. **HTTP Metrics**
   - `isl_http_requests_total`: Total requests by method, endpoint, status
   - `isl_http_request_duration_seconds`: Latency histogram (P50, P95, P99)
   - `isl_http_errors_total`: Error counts by type
   - `isl_active_requests`: Current active requests gauge

2. **Business Metrics**
   - `isl_causal_validations_total`: Causal validation counts
   - `isl_counterfactual_analyses_total`: Counterfactual requests
   - `isl_preference_queries_generated_total`: Preference learning queries
   - `isl_belief_updates_total`: Bayesian belief updates
   - `isl_teaching_examples_generated_total`: Teaching examples

3. **Cache Metrics**
   - `isl_redis_operations_total`: Redis operations by type and status
   - `isl_cache_hits_total`: Cache hit counter
   - `isl_cache_misses_total`: Cache miss counter

4. **Service Health**
   - `isl_service_up`: Service availability (1=up, 0=down)
   - `isl_redis_connected`: Redis connection status

5. **Computation Metrics**
   - `isl_activa_computation_duration_seconds`: ActiVA timing
   - `isl_bayesian_update_duration_seconds`: Bayesian update timing
   - `isl_counterfactual_computation_duration_seconds`: Counterfactual timing

### Alerting

**Configuration:** `monitoring/prometheus/alerts.yml`

**Critical Alerts:**
- `ISLServiceDown`: Service down >1 minute
- `ISLVeryHighLatency`: P95 latency >5s for 2 minutes

**Warning Alerts:**
- `ISLRedisDisconnected`: Redis disconnected >2 minutes
- `ISLHighErrorRate`: Error rate >5% for 5 minutes
- `ISLHighLatency`: P95 latency >2s for 5 minutes
- `ISLHighActiveRequests`: >100 active requests for 5 minutes

**Info Alerts:**
- `ISLLowCacheHitRate`: Cache hit rate <20% for 10 minutes
- `ISLNoRequestsReceived`: Zero requests for 10 minutes

### Grafana Dashboards

**Dashboard:** `monitoring/grafana/dashboards/isl-overview.json`
**URL:** `https://grafana.olumi.com/d/isl-overview` (production)

**Panels Include:**
- Service health status
- Request rate and latency (P50, P95, P99)
- Error rate trends
- Cache hit rate
- Redis connection status
- Active requests
- Business metric trends

### Runbooks

**Location:** `docs/operations/`

1. **PILOT_MONITORING_RUNBOOK.md** - Daily monitoring procedures (5-minute checklist)
2. **REDIS_TROUBLESHOOTING.md** - Redis diagnostic procedures
3. **STAGING_DEPLOYMENT_CHECKLIST.md** - Deployment procedures
4. **MONITORING_SETUP.md** - Infrastructure setup guide
5. **OBSERVABILITY_GUIDE.md** - Best practices

**Additional Runbooks:** `docs/runbooks/`
- **HIGH_ERROR_RATE.md** - Error rate incident response
- **HIGH_LLM_COSTS.md** - Cost management procedures

---

## 3. Developer Tooling

| Question | Answer |
|----------|--------|
| **Can your service run locally via Docker?** | **Yes** - `docker-compose.dev.yml` provides full local stack |
| **Do you have a Postman/Insomnia collection?** | **No** - OpenAPI spec available, but no exported collection |
| **Do you have integration tests against other services?** | **Yes** - Redis integration tests (`tests/integration/test_redis_health.py`, `test_redis_failover.py`) |
| **Any blockers for local full-stack development?** | **Minor** - Requires `.env` configuration and Poetry setup. PLoT UI integration needs coordination. |

### Local Development Setup

**Docker Compose Configurations:**

1. **docker-compose.dev.yml** (Recommended for local development)
   - ISL API service with hot reload
   - Redis 7-alpine with persistence
   - Redis Commander (optional web GUI)
   - Network: `isl-network`
   - Volume mounts for live code editing

2. **docker-compose.yml** (Production-like testing)
   - Single service configuration
   - Health checks enabled
   - Similar to production environment

3. **monitoring/docker-compose.monitoring.yml** (Observability stack)
   - Prometheus 2.48.0 for metrics
   - Grafana 10.2.2 for visualization
   - Node Exporter for system metrics
   - Pre-configured dashboards

**Quick Start:**
```bash
# Clone and start
git clone https://github.com/Talchain/Inference-Service-Layer.git
cd Inference-Service-Layer
docker-compose -f docker-compose.dev.yml up -d

# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

**Poetry-based Development:**
```bash
# Install dependencies
poetry install

# Copy environment configuration
cp .env.example .env

# Run with hot reload
poetry run python -m src.api.main

# Run tests
poetry run pytest
```

### API Documentation

**Interactive Docs (when running):**
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

**Static Documentation:**
- **OpenAPI Spec**: `docs/openapi.json` (149KB, OpenAPI 3.0)
- **API Examples**: `docs/API_EXAMPLES.md` (36KB, comprehensive examples)
- **Quick Reference**: `docs/API_QUICK_REFERENCE.md`

**TypeScript Client SDK:**
- **Location**: `clients/typescript/`
- **Types**: Auto-generated from OpenAPI spec
- **Usage**: Import and use with type safety

### Testing Infrastructure

**Test Organization:** `tests/` (64 test files, 140+ tests)

1. **Unit Tests** (`tests/unit/`, 29 files)
   - Core logic validation
   - Determinism verification
   - Security validators
   - Caching logic

2. **Integration Tests** (`tests/integration/`, 19 files)
   - **Redis integration**: `test_redis_health.py`, `test_redis_failover.py`
   - **API endpoints**: All major endpoints covered
   - **Concurrency**: `test_concurrency.py`
   - **Security**: `test_security.py` (19 security tests)
   - **Production readiness**: `test_production_excellence.py`

3. **Smoke Tests** (`tests/smoke/`, 2 files)
   - Production health validation
   - Quick sanity checks

4. **Load Tests** (`tests/load/`)
   - Locust load testing scenarios
   - Performance benchmarking

**Test Execution:**
```bash
# All tests with coverage
poetry run pytest --cov=src --cov-report=html

# Integration tests only
poetry run pytest tests/integration/ -v

# Redis-specific tests
poetry run pytest tests/integration/test_redis_health.py -v
```

**Coverage:**
- **Target**: 80%+
- **CI Minimum**: 50% (enforced in GitHub Actions)
- **Current**: ~90% (per documentation)

### Pre-commit Hooks

**Configuration:** `.pre-commit-config.yaml`

**Checks:**
- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **ruff**: Fast linting
- **mypy**: Type checking
- **bandit**: Security scanning

**Usage:**
```bash
# Install hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

---

## 4. Known Gaps

| Question | Answer |
|----------|--------|
| **What's missing for production-readiness?** | • **APM Integration**: Sentry referenced but not configured<br>• **API Collections**: No Postman/Insomnia workspace<br>• **Kubernetes Support**: No K8s manifests (if planning to scale beyond Render)<br>• **Dependency Scanning**: No Dependabot/Renovate automation<br>• **Contract Testing**: No Pact or similar for API contracts |
| **What would make debugging easier?** | • **External Tracing Backend**: Jaeger/Zipkin for distributed traces<br>• **Centralized Logging**: ELK stack or Datadog for log aggregation<br>• **Request Replay**: Ability to replay production requests in staging<br>• **API Collection**: Postman workspace for manual testing<br>• **Correlation with PLoT**: Unified trace IDs across ISL ↔ PLoT boundary |

### Production-Readiness Gaps (Priority Order)

#### High Priority

1. **APM/Error Tracking** ⚠️
   - **Gap**: Sentry mentioned in code but not configured with credentials
   - **Impact**: Limited visibility into production errors and stack traces
   - **Recommendation**: Configure Sentry DSN in environment variables
   - **Effort**: 1-2 hours

2. **API Collection** ⚠️
   - **Gap**: No Postman/Insomnia workspace for API testing
   - **Impact**: Manual testing requires writing curl commands or reading OpenAPI spec
   - **Recommendation**: Generate Postman collection from `docs/openapi.json`
   - **Effort**: 2-3 hours

3. **Correlation ID Standardization** ✅ RESOLVED
   - **Status**: Now supports both `X-Request-Id` (primary) and `X-Trace-Id` (legacy)
   - **Format**: `req_{uuid16}` (e.g., `req_a1b2c3d4e5f67890`)
   - **Implementation**: `src/utils/tracing.py` - TracingMiddleware

4. **Dependency Automation** ⚠️
   - **Gap**: No Dependabot or Renovate for automated dependency updates
   - **Impact**: Manual effort to track security vulnerabilities and updates
   - **Recommendation**: Enable GitHub Dependabot in repository settings
   - **Effort**: 30 minutes

#### Medium Priority

5. **Distributed Tracing Backend** 📊
   - **Gap**: Trace IDs generated but no centralized trace storage (Jaeger/Zipkin)
   - **Impact**: Cannot visualize request flows across services
   - **Recommendation**: Deploy Jaeger or integrate with Datadog APM
   - **Effort**: 4-6 hours

6. **Centralized Logging** 📊
   - **Gap**: JSON logs to stdout/stderr, no centralized aggregation
   - **Impact**: Difficult to search logs across multiple instances
   - **Recommendation**: Integrate with Render's log aggregation or deploy ELK stack
   - **Effort**: 4-6 hours

7. **Kubernetes Support** 🚀
   - **Gap**: No K8s manifests or Helm charts
   - **Impact**: Limited to Render platform, harder to scale horizontally
   - **Recommendation**: Create K8s manifests if planning multi-cloud or high scale
   - **Effort**: 8-12 hours (includes testing)

8. **Contract Testing** 🧪
   - **Gap**: No consumer-driven contract tests
   - **Impact**: Breaking API changes not caught until integration testing
   - **Recommendation**: Implement Pact or similar for PLoT ↔ ISL contracts
   - **Effort**: 6-8 hours

#### Low Priority

9. **Infrastructure as Code** 🏗️
   - **Gap**: No Terraform/Pulumi for infrastructure provisioning
   - **Impact**: Manual setup of Redis, Prometheus, Grafana
   - **Recommendation**: Codify infrastructure for reproducibility
   - **Effort**: 12-16 hours

10. **Multi-Cloud Deployment** ☁️
    - **Gap**: Only configured for Render deployment
    - **Impact**: Vendor lock-in, no disaster recovery across clouds
    - **Recommendation**: Add AWS/GCP/Azure deployment options
    - **Effort**: 16-24 hours per platform

### Debugging Enhancement Recommendations

#### Immediate Wins (< 1 week)

1. **Postman Collection** 🎯
   - **Action**: Generate from OpenAPI spec, add to repo
   - **Benefit**: Quick manual testing, easier client onboarding
   - **Owner**: Platform Coordination

2. **Sentry Integration** 🎯
   - **Action**: Add `SENTRY_DSN` to environment variables
   - **Benefit**: Real-time error notifications with stack traces
   - **Owner**: ISL Team

3. **Request ID Alignment** ✅ COMPLETE
   - **Status**: ISL now supports both `X-Request-Id` (primary) and `X-Trace-Id` (legacy)
   - **Format**: `req_{uuid16}` format aligned with platform standard
   - **Owner**: Platform Coordination + ISL Team

4. **Runbook Updates** 🎯
   - **Action**: Add cross-service debugging procedures to runbooks
   - **Benefit**: Faster incident response for ISL ↔ PLoT issues
   - **Owner**: ISL Team

#### Short-term Improvements (1-4 weeks)

5. **Distributed Tracing**
   - **Action**: Deploy Jaeger or integrate with Datadog
   - **Benefit**: Visualize request flows, identify bottlenecks
   - **Dependencies**: Requires platform-wide tracing strategy

6. **Log Aggregation**
   - **Action**: Integrate with centralized logging service
   - **Benefit**: Search/filter logs across all ISL instances
   - **Dependencies**: Platform logging infrastructure decision

7. **Request Replay Tool**
   - **Action**: Build tool to capture production requests and replay in staging
   - **Benefit**: Reproduce production issues in safe environment
   - **Dependencies**: Privacy/security review for request data

#### Long-term Enhancements (1-3 months)

8. **End-to-End Tracing**
   - **Action**: Unified trace IDs across PLoT → ISL → Redis → External APIs
   - **Benefit**: Complete visibility into user journeys
   - **Dependencies**: Requires PLoT team coordination

9. **Chaos Engineering**
   - **Action**: Implement chaos testing scenarios (network failures, latency injection)
   - **Benefit**: Build confidence in failure resilience
   - **Dependencies**: Dedicated test environment

10. **Performance Profiling**
    - **Action**: Continuous profiling (py-spy, memory profiling)
    - **Benefit**: Identify performance regressions before production
    - **Dependencies**: Profiling infrastructure setup

---

## 5. Cross-Service Integration Status

### Current Integrations

| Service | Status | Connection Type | Health Monitoring |
|---------|--------|-----------------|-------------------|
| **Redis 7** | ✅ Implemented | Direct connection | `test_redis_health.py` |
| **Prometheus** | ✅ Implemented | Metrics scraping | Built-in |
| **Grafana** | ✅ Implemented | Prometheus datasource | Built-in |

### Planned Integrations

| Service | Status | Blocker | Correlation ID Support |
|---------|--------|---------|------------------------|
| **PLoT UI** | 🚧 Pending | Coordination needed | To be implemented |
| **External LLM APIs** | 💡 Referenced in code | Not yet required | N/A |

### Integration Test Coverage

**Existing Tests:**
- ✅ Redis connection health (`tests/integration/test_redis_health.py`)
- ✅ Redis failover behavior (`tests/integration/test_redis_failover.py`)
- ✅ Concurrent request handling (`tests/integration/test_concurrency.py`)
- ✅ End-to-end API workflows (`tests/integration/test_phase3_e2e.py`)

**Missing Tests:**
- ❌ PLoT ↔ ISL integration tests (waiting for PLoT staging)
- ❌ External API mock tests (not yet integrated)

---

## 6. Platform Coordination Recommendations

### For `@olumi/contracts` Package

**ISL Can Provide:**

1. **TypeScript Types**
   - Location: `clients/typescript/`
   - Contains: Request/response types for all ISL endpoints
   - Suggestion: Move to `@olumi/contracts/isl`

2. **OpenAPI Specification**
   - Location: `docs/openapi.json`
   - Format: OpenAPI 3.0
   - Suggestion: Use as source of truth for contract generation

3. **Pydantic Models**
   - Location: `src/models/`
   - Contains: Request/response schemas with validation
   - Suggestion: Generate TypeScript from these models

### For Local docker-compose

**ISL Provides:**
- ✅ Service definition with health checks
- ✅ Redis dependency configuration
- ✅ Network configuration
- ✅ Environment variable template

**Needs from Platform:**
- 🔄 PLoT service definition to add to `docker-compose.dev.yml`
- 🔄 Shared network name (currently `isl-network`)
- 🔄 Environment variable naming conventions

### For Cross-Service CI Pipeline

**ISL Can Integrate:**
- ✅ Existing GitHub Actions workflow
- ✅ Health check endpoints (`/health`, `/health/redis`)
- ✅ Smoke test suite (`tests/smoke/`)

**Needs from Platform:**
- 🔄 Contract testing framework decision (Pact, Prism, etc.)
- 🔄 Shared test fixtures repository
- 🔄 Integration test trigger strategy

### For Correlation ID Propagation

**Current Implementation:**
- ✅ `X-Request-Id` header extraction (primary)
- ✅ `X-Trace-Id` header extraction (legacy fallback)
- ✅ `X-User-Id` header extraction
- ✅ Trace context propagation via ContextVar
- ✅ Trace ID injection into logs
- ✅ Both `X-Request-Id` and `X-Trace-Id` returned in response headers

**Platform Alignment Status:** ✅ COMPLETE

**Current Standard:**
```
X-Request-Id: req_{uuid16}      # Primary correlation ID (platform-wide) - PREFERRED
X-Trace-Id: req_{uuid16}        # Legacy alias (deprecated, for backward compatibility)
X-User-Id: user_{uuid}          # User context (optional)
```

**Implementation:** `src/utils/tracing.py` - TracingMiddleware accepts both headers (priority: X-Request-Id > X-Trace-Id) and returns both in responses for compatibility.

---

## 7. Service Metadata

### Technical Specifications

| Property | Value |
|----------|-------|
| **Language** | Python 3.11.9 |
| **Framework** | FastAPI 0.104+ with Uvicorn |
| **Package Manager** | Poetry 1.7.1 |
| **Lines of Code** | ~24,000 (Python) |
| **Test Files** | 64 files, 140+ tests |
| **Coverage** | ~90% |
| **API Endpoints** | 15+ routers |
| **Dependencies** | ~40 packages (including dev) |

### Service Characteristics

| Characteristic | Value | Notes |
|----------------|-------|-------|
| **Deterministic** | ✅ Yes | Same inputs → same outputs |
| **Stateless** | ✅ Yes | No session state (uses Redis for caching only) |
| **Horizontally Scalable** | ⚠️ Partial | Limited by single-worker requirement for determinism |
| **Cache-Dependent** | ⚠️ Optional | Redis improves performance but has in-memory fallback |
| **CPU-Bound** | ✅ Yes | Causal inference is computationally intensive |
| **Response Time** | 200ms - 5s | Varies by operation complexity |
| **Memory Usage** | 256MB - 2GB | Depends on model size |

### Resource Requirements

**Development:**
- CPU: 2 cores minimum
- Memory: 4GB minimum
- Disk: 2GB minimum

**Production (per instance):**
- CPU: 2-4 cores recommended
- Memory: 4-8GB recommended
- Disk: 5GB (for logs and cache)

**Redis:**
- Memory: 256MB - 2GB (depending on cache size)
- Persistence: Optional (RDB snapshots)

---

## 8. Documentation Index

### Operational Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| **Main README** | `README.md` | Quick start, architecture overview |
| **Deployment Guide** | `DEPLOYMENT_GUIDE.md` | Advanced features deployment |
| **Monitoring Runbook** | `docs/operations/PILOT_MONITORING_RUNBOOK.md` | Daily monitoring procedures |
| **Redis Troubleshooting** | `docs/operations/REDIS_TROUBLESHOOTING.md` | Redis diagnostic procedures |
| **Staging Checklist** | `docs/operations/STAGING_DEPLOYMENT_CHECKLIST.md` | Deployment procedures |
| **Observability Guide** | `docs/operations/OBSERVABILITY_GUIDE.md` | Best practices |

### API Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| **API Examples** | `docs/API_EXAMPLES.md` | Comprehensive usage examples |
| **Quick Reference** | `docs/API_QUICK_REFERENCE.md` | Endpoint summary |
| **OpenAPI Spec** | `docs/openapi.json` | Machine-readable API contract |
| **Swagger UI** | `http://localhost:8000/docs` | Interactive API explorer |
| **ReDoc** | `http://localhost:8000/redoc` | Alternative API docs |

### Integration Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| **Integration Guide** | `docs/integration/PLOT_INTEGRATION_GUIDE.md` | PLoT UI integration |
| **Integration Examples** | `docs/integration/INTEGRATION_EXAMPLES.md` | Code samples |
| **Cross-Reference Schema** | `docs/integration/CROSS_REFERENCE_SCHEMA.md` | Assumption traceability |

### Technical Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| **Technical Specs** | `docs/TECHNICAL_SPECIFICATIONS.md` | Complete specifications |
| **Architecture** | `docs/PHASE1_ARCHITECTURE.md` | System architecture |
| **Security Audit** | `docs/SECURITY_AUDIT.md` | Security review |
| **Code Quality Report** | `docs/CODE_QUALITY_REPORT.md` | Code metrics |

---

## 9. Action Items for Platform Coordination

### Immediate (This Week)

- [x] **Review correlation ID standard**: ✅ ISL now supports both `X-Request-Id` (primary) and `X-Trace-Id` (legacy)
- [ ] **Generate Postman collection**: Use OpenAPI spec to create shared collection
- [ ] **Document service dependencies**: Map out ISL → PLoT → Backend interactions
- [ ] **Share docker-compose service definition**: Add ISL to platform docker-compose

### Short-term (Next 2 Weeks)

- [ ] **Configure Sentry**: Add DSN to environment variables, test error reporting
- [ ] **Enable Dependabot**: Automate dependency updates for ISL
- [ ] **Create integration test strategy**: Define contract testing approach
- [ ] **Document trace propagation**: Write guide for cross-service tracing

### Medium-term (Next Month)

- [ ] **Deploy distributed tracing**: Set up Jaeger or Datadog APM
- [ ] **Centralize logging**: Integrate with platform logging service
- [ ] **Create @olumi/contracts package**: Extract ISL types to shared package
- [ ] **Build integration CI pipeline**: Test ISL ↔ PLoT integration automatically

---

## 10. Contact & Support

### ISL Team

- **Repository**: https://github.com/Talchain/Inference-Service-Layer
- **Staging Health**: https://isl-staging.onrender.com/health
- **Production Health**: https://isl.olumi.com/health
- **Grafana Dashboard**: https://grafana.olumi.com/d/isl-overview (production)

### Questions?

For questions about this audit or ISL infrastructure:
1. Open an issue in the ISL repository
2. Contact the platform coordination channel
3. Reference this document: `PLATFORM_INFRASTRUCTURE_AUDIT.md`

---

## Appendix A: Quick Reference Commands

### Health Checks

```bash
# Staging health
curl https://isl-staging.onrender.com/health

# Production health
curl https://isl.olumi.com/health

# Redis health
curl https://isl-staging.onrender.com/health/redis

# Metrics
curl https://isl-staging.onrender.com/metrics
```

### Local Development

```bash
# Start local stack
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f isl-api

# Stop services
docker-compose -f docker-compose.dev.yml down

# Run tests
poetry run pytest --cov=src

# Run integration tests
poetry run pytest tests/integration/ -v
```

### Monitoring

```bash
# Check Prometheus alerts
curl http://localhost:9090/api/v1/alerts

# Query metrics
curl http://localhost:9090/api/v1/query?query=isl_http_requests_total

# View Grafana dashboards
open http://localhost:3000
```

---

**Document Version:** 1.0
**Last Updated:** 27 November 2025
**Next Review:** 27 December 2025
