# Security Remediation Plan

## Overview

This document outlines the prioritized plan for addressing security review feedback for the Inference Service Layer. The recommendations are consolidated from two independent security reviews and organized by priority level.

**Review Date:** 2025-11-25
**Target Completion:** Phased implementation over multiple sprints

---

## Executive Summary

| Priority | Category | Issues | Effort |
|----------|----------|--------|--------|
| **P0 - Critical** | Authentication & Authorization | 3 issues | High |
| **P1 - High** | Rate Limiting & DoS Protection | 4 issues | Medium |
| **P1 - High** | Configuration & Secrets | 4 issues | Medium |
| **P2 - Medium** | Redis & Infrastructure | 3 issues | Medium |
| **P2 - Medium** | Dependencies & Supply Chain | 3 issues | Low |
| **P3 - Low** | Observability & Compliance | 4 issues | Medium |
| **P3 - Low** | Testing & QA | 3 issues | Medium |

---

## P0 - Critical Priority (Immediate Action Required)

### 1. Authentication & Authorization Layer

**Current State:** No authentication framework exists. All endpoints are publicly accessible.

**Risk:** Complete API exposure to unauthorized access, data exfiltration, abuse.

#### 1.1 Add JWT/OAuth2 Authentication Middleware

**File:** `src/middleware/authentication.py` (new)

```python
# Implementation approach:
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt

# Required dependencies:
# - python-jose[cryptography]
# - passlib[bcrypt]
```

**Tasks:**
- [ ] Add `python-jose[cryptography]` and `passlib[bcrypt]` to dependencies
- [ ] Create `AuthenticationMiddleware` with JWT validation
- [ ] Implement token verification with configurable issuers
- [ ] Add `get_current_user` dependency for protected routes
- [ ] Support both Bearer tokens and API keys

#### 1.2 Add Role-Based Access Control (RBAC)

**File:** `src/middleware/authorization.py` (new)

**Tasks:**
- [ ] Define permission scopes per endpoint category:
  - `causal:read` - Read causal models
  - `causal:write` - Create/modify causal models
  - `batch:execute` - Execute batch operations
  - `admin:metrics` - Access system metrics
- [ ] Create `require_permissions()` decorator
- [ ] Add per-route scope requirements

#### 1.3 Service-to-Service Authentication

**Tasks:**
- [ ] Implement mTLS support for internal service communication
- [ ] Add signed header verification for service mesh deployments
- [ ] Document authentication flow for microservice architectures

#### 1.4 Harden CORS Configuration

**File:** `src/api/main.py`

**Current Issue:** Hardcoded origins, `*` wildcard in development mode.

**Tasks:**
- [ ] Move CORS origins to environment configuration
- [ ] Remove wildcard `*` support even in development
- [ ] Add explicit origin validation
- [ ] Restrict `allow_credentials` when using broad origins

**Configuration Changes:**
```python
# Add to src/config/__init__.py
CORS_ORIGINS: List[str] = Field(
    default_factory=list,
    description="Allowed CORS origins (comma-separated in env)",
)
CORS_ALLOW_CREDENTIALS: bool = Field(
    default=False,
    description="Allow credentials in CORS requests",
)
```

---

## P1 - High Priority (Address Within Current Sprint)

### 2. Distributed Rate Limiting

**Current State:** In-memory rate limiter using Python dict, IP-based only.

**Risk:** Bypassed by distributed attacks, doesn't work across replicas.

#### 2.1 Redis-Backed Rate Limiting

**File:** `src/middleware/rate_limiting.py` (modify)

**Tasks:**
- [ ] Add `slowapi` or `fastapi-limiter` dependency
- [ ] Implement Redis-backed token bucket algorithm
- [ ] Add sliding window rate limiting support
- [ ] Create fallback to in-memory when Redis unavailable

**Implementation Approach:**
```python
# Option 1: slowapi (recommended)
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379",
)

# Option 2: Custom Redis implementation
class RedisRateLimiter:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        # Implement sliding window counter in Redis
        pass
```

#### 2.2 Proxy-Aware IP Detection

**Tasks:**
- [ ] Parse `X-Forwarded-For` header correctly
- [ ] Configure trusted proxy list
- [ ] Implement IP validation and sanitization

**Configuration:**
```python
# Add to src/config/__init__.py
TRUSTED_PROXIES: List[str] = Field(
    default_factory=list,
    description="List of trusted proxy IPs",
)
RATE_LIMIT_HEADER: str = Field(
    default="X-Forwarded-For",
    description="Header to extract client IP from",
)
```

#### 2.3 Per-Tenant Rate Limiting

**Tasks:**
- [ ] Extract tenant/user ID from JWT claims
- [ ] Implement tiered rate limits (free/paid/enterprise)
- [ ] Add burst allowances for premium tiers
- [ ] Create rate limit configuration per endpoint

#### 2.4 Rate Limit Monitoring

**Tasks:**
- [ ] Add Prometheus metrics for rate limit hits
- [ ] Create alerting rules for sustained rate limiting
- [ ] Log rate limit events for security analysis

---

### 3. Configuration & Secrets Management

**Current State:** Pydantic settings with .env defaults, no secrets management.

**Risk:** Secrets in environment variables, misconfiguration in production.

#### 3.1 Required Configuration Validation

**File:** `src/config/__init__.py` (modify)

**Tasks:**
- [ ] Add `@validator` for critical settings
- [ ] Fail fast on missing required settings in production
- [ ] Add environment detection (dev/staging/prod)

**Implementation:**
```python
from pydantic import validator

class Settings(BaseSettings):
    ENVIRONMENT: str = Field(default="development")

    # Security settings (required in production)
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    JWT_ISSUER: Optional[str] = None

    @validator("JWT_SECRET_KEY", always=True)
    def validate_jwt_secret(cls, v, values):
        if values.get("ENVIRONMENT") == "production" and not v:
            raise ValueError("JWT_SECRET_KEY required in production")
        return v
```

#### 3.2 Secrets Manager Integration

**File:** `src/config/secrets.py` (new)

**Tasks:**
- [ ] Add AWS Secrets Manager client (optional)
- [ ] Add HashiCorp Vault client (optional)
- [ ] Implement secret caching with TTL
- [ ] Add fallback to environment variables

**Interface:**
```python
class SecretsProvider(Protocol):
    async def get_secret(self, key: str) -> str: ...

class EnvironmentSecretsProvider:
    """Fallback provider using environment variables."""

class AWSSecretsProvider:
    """AWS Secrets Manager provider."""

class VaultSecretsProvider:
    """HashiCorp Vault provider."""
```

#### 3.3 TLS Configuration

**Tasks:**
- [ ] Add TLS certificate path configuration
- [ ] Document TLS termination options (app-level vs proxy)
- [ ] Add HTTPS-only mode for production

**Configuration:**
```python
# Add to src/config/__init__.py
TLS_ENABLED: bool = Field(default=False)
TLS_CERT_PATH: Optional[str] = None
TLS_KEY_PATH: Optional[str] = None
FORCE_HTTPS: bool = Field(default=False)
```

#### 3.4 Environment-Specific Configuration

**Tasks:**
- [ ] Create `config/development.py`, `config/production.py`
- [ ] Add configuration override hierarchy
- [ ] Document configuration precedence

---

## P2 - Medium Priority (Address Within Next Sprint)

### 4. Redis Security & Reliability

**Current State:** Basic Redis client without TLS, auth, or proper error handling.

**File:** `src/infrastructure/redis_client.py` (modify)

#### 4.1 Enable Redis TLS/SSL

**Tasks:**
- [ ] Add TLS configuration parameters
- [ ] Support certificate-based authentication
- [ ] Document Redis TLS setup requirements

**Implementation:**
```python
def get_redis_client() -> Optional[redis.Redis]:
    ssl_context = None
    if os.getenv("REDIS_TLS_ENABLED", "false").lower() == "true":
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

    return redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD"),
        ssl=ssl_context is not None,
        ssl_context=ssl_context,
        # ... other params
    )
```

#### 4.2 Connection Pool Configuration

**Tasks:**
- [ ] Configure connection pool size
- [ ] Add health check interval
- [ ] Implement connection recycling

**Configuration:**
```python
REDIS_MAX_CONNECTIONS: int = 50
REDIS_HEALTH_CHECK_INTERVAL: int = 30
REDIS_SOCKET_KEEPALIVE: bool = True
```

#### 4.3 Retry & Circuit Breaker

**Tasks:**
- [ ] Add exponential backoff retry logic
- [ ] Implement circuit breaker pattern
- [ ] Surface Redis health in `/health` endpoint

#### 4.4 Health Monitoring

**Tasks:**
- [ ] Add Redis connection status to health check
- [ ] Create Prometheus metrics for Redis operations
- [ ] Add alerting for Redis connectivity issues

---

### 5. Dependency & Supply Chain Security

**Current State:** Caret versioning, no lockfile enforcement, no SCA.

#### 5.1 Pin Dependencies to Exact Versions

**File:** `pyproject.toml` (modify)

**Tasks:**
- [ ] Change caret (`^`) to exact (`==`) pinning for production
- [ ] Add hash verification for critical dependencies
- [ ] Document upgrade procedure

**Before:**
```toml
fastapi = "^0.104.0"
```

**After:**
```toml
fastapi = "==0.104.1"
```

#### 5.2 Enable Automated Vulnerability Scanning

**Tasks:**
- [ ] Add GitHub Dependabot configuration
- [ ] Configure Renovate Bot as alternative
- [ ] Add `safety` or `pip-audit` to CI pipeline

**File:** `.github/dependabot.yml` (new)
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "security"
```

#### 5.3 CI Security Gates

**File:** `.github/workflows/security.yml` (new)

**Tasks:**
- [ ] Add `safety check` step to CI
- [ ] Add `bandit` for Python security linting
- [ ] Add `trivy` for container scanning
- [ ] Block merges on high/critical vulnerabilities

---

### 6. Input Validation & Threat Modeling

**Current State:** Good field-level validation, but missing comprehensive threat model.

#### 6.1 Request Size Limits

**Tasks:**
- [ ] Add global request body size limit
- [ ] Add per-endpoint payload limits
- [ ] Configure at middleware level

**Implementation:**
```python
from starlette.middleware.base import BaseHTTPMiddleware

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            raise HTTPException(413, "Request too large")
        return await call_next(request)
```

#### 6.2 Async Timeout Protection

**Tasks:**
- [ ] Add request-level timeouts
- [ ] Implement computation timeouts for heavy operations
- [ ] Add cancellation support for long-running tasks

#### 6.3 Threat Model Documentation

**File:** `docs/THREAT_MODEL.md` (new)

**Tasks:**
- [ ] Document threat actors and attack vectors
- [ ] Map threats to each endpoint category
- [ ] Define security controls per threat
- [ ] Create incident response procedures

---

## P3 - Low Priority (Backlog for Future Sprints)

### 7. Observability & Compliance

#### 7.1 Structured Logging Enhancements

**Tasks:**
- [ ] Add correlation IDs to all log entries
- [ ] Implement PII redaction rules
- [ ] Add audit logging for security events
- [ ] Create log rotation and retention policies

#### 7.2 Security Metrics

**Tasks:**
- [ ] Add authentication failure metrics
- [ ] Add rate limit violation metrics
- [ ] Add request anomaly detection
- [ ] Create security-focused dashboard

#### 7.3 SLO/SLI Implementation

**Tasks:**
- [ ] Define availability SLO (e.g., 99.9%)
- [ ] Define latency SLI (p50, p95, p99)
- [ ] Create error budget tracking
- [ ] Implement SLO alerting

#### 7.4 Compliance Controls

**Tasks:**
- [ ] Document data handling procedures
- [ ] Add data classification tags
- [ ] Implement data retention policies
- [ ] Create audit report generation

---

### 8. Testing & QA Maturity

#### 8.1 Security Test Coverage

**Tasks:**
- [ ] Add authentication bypass tests
- [ ] Add rate limit evasion tests
- [ ] Add injection attack tests
- [ ] Add fuzzing for input validation

#### 8.2 Concurrency Testing

**Tasks:**
- [ ] Add race condition tests
- [ ] Add parallel request stress tests
- [ ] Test rate limiter under load
- [ ] Test Redis failover scenarios

#### 8.3 Contract Testing

**Tasks:**
- [ ] Add OpenAPI schema validation tests
- [ ] Add response contract tests
- [ ] Add backwards compatibility tests

---

## Implementation Phases

### Phase 1: Critical Security (Week 1-2)
- P0.1: Authentication middleware
- P0.2: Basic RBAC implementation
- P0.4: CORS hardening

### Phase 2: Production Hardening (Week 3-4)
- P1.1: Redis-backed rate limiting
- P1.2: Proxy-aware IP detection
- P1.3: Configuration validation
- P2.1: Redis TLS enablement

### Phase 3: Supply Chain & Monitoring (Week 5-6)
- P2.5: Dependency pinning
- P2.5: SCA in CI pipeline
- P3.1: Enhanced logging
- P3.2: Security metrics

### Phase 4: Long-term Improvements (Ongoing)
- P1.4: Secrets manager integration
- P3.3: SLO implementation
- P3.4: Compliance controls
- P3.5: Security testing suite

---

## Configuration Summary

### Required Environment Variables (Production)

```bash
# Authentication (P0 - Required)
JWT_SECRET_KEY=<strong-random-key>
JWT_ALGORITHM=HS256
JWT_ISSUER=https://your-auth-server.com
JWT_AUDIENCE=inference-service

# CORS (P0 - Required)
CORS_ORIGINS=https://app.example.com,https://admin.example.com
CORS_ALLOW_CREDENTIALS=false

# Rate Limiting (P1 - Required)
REDIS_HOST=redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=<redis-password>
REDIS_TLS_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
TRUSTED_PROXIES=10.0.0.0/8,172.16.0.0/12

# Environment (P1 - Required)
ENVIRONMENT=production
LOG_LEVEL=INFO

# TLS (P1 - Recommended)
TLS_ENABLED=true
TLS_CERT_PATH=/etc/ssl/certs/server.crt
TLS_KEY_PATH=/etc/ssl/private/server.key
FORCE_HTTPS=true
```

---

## Dependencies to Add

```toml
# pyproject.toml additions

# Authentication (P0)
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
passlib = {extras = ["bcrypt"], version = "^1.7.4"}

# Rate Limiting (P1)
slowapi = "^0.1.9"

# Security Scanning (P2)
safety = "^2.3.0"
bandit = "^1.7.0"
```

---

## Success Criteria

| Priority | Success Metric |
|----------|----------------|
| P0 | All endpoints require valid authentication |
| P0 | CORS rejects unknown origins |
| P1 | Rate limiting works across replicas |
| P1 | Application fails to start with missing required config |
| P2 | Redis connections use TLS in production |
| P2 | CI fails on high/critical vulnerabilities |
| P3 | All security events are logged with correlation IDs |
| P3 | Security test coverage > 80% |

---

## References

- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Redis Security](https://redis.io/docs/management/security/)
- [Python Security Best Practices](https://docs.python.org/3/library/secrets.html)
