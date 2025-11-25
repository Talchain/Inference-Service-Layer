# Security Remediation Plan

## Overview

This document outlines the prioritized plan for addressing security review feedback for the Inference Service Layer. The recommendations are consolidated from two independent security reviews and organized by priority level.

**Review Date:** 2025-11-25
**Last Updated:** 2025-11-25
**Target Completion:** Phased implementation over multiple sprints

---

## Architectural Decision: API Keys Only

**Decision:** The ISL will use API key authentication only for the initial implementation.

**Rationale:**
- ISL is primarily a service-to-service API (PLoT → ISL, CEE → ISL)
- API keys are simpler and appropriate for service-to-service communication
- JWT/OAuth2 can be added later if user-facing endpoints are required

**Implementation:**
- `X-API-Key` header validation via middleware
- Support for multiple API keys (comma-separated in `ISL_API_KEYS` env var)
- Public endpoints exempt: `/health`, `/ready`, `/metrics`, `/docs`, `/redoc`, `/openapi.json`

---

## Executive Summary

| Priority | Category | Issues | Effort | Status |
|----------|----------|--------|--------|--------|
| **P0 - Critical** | Authentication & CORS | 4 items | Medium | 🔄 In Progress |
| **P1 - High** | Rate Limiting & DoS Protection | 4 items | Medium | 🔄 In Progress |
| **P1 - High** | Configuration & Secrets | 4 items | Medium | Pending |
| **P2 - Medium** | Redis & Infrastructure | 4 items | Medium | Pending |
| **P2 - Medium** | Dependencies & Supply Chain | 3 items | Low | ✅ Partial |
| **P3 - Low** | Observability & Compliance | 4 items | Medium | Pending |
| **P3 - Low** | Testing & QA | 3 items | Medium | Pending |

---

## P0 - Critical Priority (Immediate Action Required)

### 1. API Key Authentication

**File:** `src/middleware/auth.py` (new)

**Implementation:**
```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header against configured API keys."""

    PUBLIC_PATHS = {"/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in self.valid_keys:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
```

**Tasks:**
- [x] Create API key authentication middleware
- [x] Support comma-separated API keys (`ISL_API_KEYS`)
- [x] Exempt public endpoints
- [x] Add to main.py middleware stack
- [ ] Add authentication tests

### 2. CORS Hardening

**File:** `src/api/main.py`

**Current Issue:** Hardcoded origins, `*` wildcard in development mode.

**Tasks:**
- [ ] Move CORS origins to Settings class
- [ ] Remove wildcard `*` support in all modes
- [ ] Add explicit origin validation
- [ ] Restrict `allow_credentials` when using broad origins

**Configuration Changes:**
```python
# Add to src/config/__init__.py
CORS_ORIGINS: List[str] = Field(
    default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"],
    description="Allowed CORS origins (comma-separated in env: CORS_ORIGINS)",
)
CORS_ALLOW_CREDENTIALS: bool = Field(
    default=False,
    description="Allow credentials in CORS requests",
)
```

---

## P1 - High Priority (Address Within Current Sprint)

### 3. Distributed Rate Limiting

**Current State:** In-memory rate limiter with proxy-aware IP detection.

**Risk:** Doesn't work across replicas in multi-instance deployments.

#### 3.1 Redis-Backed Rate Limiting

**File:** `src/middleware/rate_limiting.py` (modify)

**Tasks:**
- [ ] Implement Redis-backed sliding window rate limiter
- [ ] Add fallback to in-memory when Redis unavailable
- [ ] Add per-API-key rate limiting (in addition to IP-based)
- [ ] Add rate limit Prometheus metrics

**Implementation Approach:**
```python
class RedisRateLimiter:
    """Redis-backed rate limiter with sliding window."""

    def __init__(self, redis_client: Optional[Redis], fallback: RateLimiter):
        self.redis = redis_client
        self.fallback = fallback

    async def check_rate_limit(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        if not self.redis:
            return self.fallback.check_rate_limit(key)

        # Use Redis MULTI/EXEC for atomic sliding window
        pipe = self.redis.pipeline()
        now = time.time()
        window_start = now - window

        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window)

        results = pipe.execute()
        count = results[2]

        if count > limit:
            return False, window
        return True, 0
```

### 4. Configuration & Secrets Management

**Current State:** Pydantic settings with .env defaults, no validation.

#### 4.1 Required Configuration Validation

**File:** `src/config/__init__.py` (modify)

**Tasks:**
- [ ] Add `ENVIRONMENT` setting (development/staging/production)
- [ ] Add validators for critical settings
- [ ] Fail fast on missing required settings in production
- [ ] Add API key configuration to Settings

**Implementation:**
```python
from pydantic import field_validator
from typing import Optional, List

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = Field(default="development")

    # Authentication
    ISL_API_KEYS: Optional[str] = Field(
        default=None,
        description="Comma-separated list of valid API keys"
    )

    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Comma-separated list of allowed CORS origins"
    )

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=100)
    TRUSTED_PROXIES: str = Field(default="")

    @field_validator("ISL_API_KEYS")
    @classmethod
    def validate_api_keys(cls, v, info):
        if info.data.get("ENVIRONMENT") == "production" and not v:
            raise ValueError("ISL_API_KEYS required in production")
        return v
```

---

## P2 - Medium Priority (Address Within Next Sprint)

### 5. Redis Security & Reliability

**Current State:** Basic Redis client without TLS, auth, or proper error handling.

**File:** `src/infrastructure/redis_client.py` (modify)

**Tasks:**
- [ ] Add TLS/SSL support
- [ ] Configure connection pool size
- [ ] Add retry logic with exponential backoff
- [ ] Add Redis health to `/health` endpoint
- [ ] Add Prometheus metrics for Redis operations

**Implementation:**
```python
import ssl
from typing import Optional
import redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff

def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client with TLS, auth, and retry support."""
    ssl_context = None
    if os.getenv("REDIS_TLS_ENABLED", "false").lower() == "true":
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

    retry = Retry(ExponentialBackoff(), 3)

    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD"),
        db=int(os.getenv("REDIS_DB", 0)),
        ssl=ssl_context is not None,
        ssl_context=ssl_context,
        socket_connect_timeout=2,
        socket_timeout=5,
        retry=retry,
        retry_on_timeout=True,
        health_check_interval=30,
        max_connections=50,
        decode_responses=True,
    )
```

### 6. Dependency & Supply Chain Security

**Current State:** Caret versioning, Dependabot configured.

**Tasks:**
- [x] Add GitHub Dependabot configuration
- [ ] Add security CI workflow with `safety` and `bandit`
- [ ] Pin critical dependencies to exact versions

### 7. Input Validation & Request Limits

**Tasks:**
- [ ] Add global request body size limit middleware
- [ ] Add async timeout protection for heavy computations

---

## P3 - Low Priority (Backlog for Future Sprints)

### 8. Observability & Compliance

**Tasks:**
- [ ] Add correlation IDs to all log entries
- [ ] Add security audit logging
- [ ] Add rate limit violation metrics
- [ ] Create security-focused Grafana dashboard

### 9. Testing & QA Maturity

**Tasks:**
- [ ] Add API key authentication tests
- [ ] Add rate limit evasion tests
- [ ] Add concurrency/race condition tests
- [ ] Add security regression tests

---

## Implementation Phases

### Phase 1: Critical Security (Current)
- ✅ API key authentication middleware
- 🔄 CORS hardening
- 🔄 Configuration validation

### Phase 2: Production Hardening (Next)
- Redis-backed rate limiting
- Redis TLS enablement
- Request size limits

### Phase 3: Supply Chain & Monitoring
- Security CI workflow
- Dependency pinning
- Enhanced logging

### Phase 4: Long-term Improvements
- Security testing suite
- Compliance controls
- SLO implementation

---

## Configuration Summary

### Required Environment Variables (Production)

```bash
# Authentication (P0 - Required)
ISL_API_KEYS=key1,key2,key3

# CORS (P0 - Required)
CORS_ORIGINS=https://app.example.com,https://admin.example.com

# Environment (P1 - Required)
ENVIRONMENT=production
LOG_LEVEL=INFO

# Rate Limiting (P1 - Required)
REDIS_HOST=redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=<redis-password>
RATE_LIMIT_REQUESTS_PER_MINUTE=100
TRUSTED_PROXIES=10.0.0.0/8,172.16.0.0/12

# Redis Security (P2 - Recommended)
REDIS_TLS_ENABLED=true
```

---

## Success Criteria

| Priority | Success Metric |
|----------|----------------|
| P0 | All protected endpoints require valid API key |
| P0 | CORS rejects unknown origins |
| P1 | Rate limiting works across replicas |
| P1 | Application fails to start with missing required config in production |
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
