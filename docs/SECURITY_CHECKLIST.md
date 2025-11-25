# Security Remediation Checklist

Quick reference checklist for tracking security remediation progress.

## P0 - Critical (Must Complete Before Production)

### Authentication & Authorization
- [ ] Add JWT authentication middleware (`src/middleware/authentication.py`)
- [ ] Add `python-jose[cryptography]` dependency
- [ ] Add `passlib[bcrypt]` dependency
- [ ] Implement `get_current_user()` dependency
- [ ] Add role-based access control (`src/middleware/authorization.py`)
- [ ] Define permission scopes per endpoint
- [ ] Add service-to-service auth (mTLS/signed headers)

### CORS Hardening
- [ ] Move CORS origins to environment configuration
- [ ] Remove wildcard `*` support in all modes
- [ ] Add `CORS_ORIGINS` to Settings class
- [ ] Test CORS with production origin list

## P1 - High Priority (Complete Within Sprint)

### Distributed Rate Limiting
- [ ] Add `slowapi` dependency
- [ ] Implement Redis-backed rate limiter
- [ ] Configure proxy-aware IP detection
- [ ] Add `TRUSTED_PROXIES` configuration
- [ ] Implement per-tenant rate limits
- [ ] Add rate limit Prometheus metrics
- [ ] Test rate limiting across replicas

### Configuration & Secrets
- [ ] Add required setting validators
- [ ] Add `ENVIRONMENT` setting (dev/staging/prod)
- [ ] Add `JWT_SECRET_KEY` validator (required in prod)
- [ ] Implement secrets provider interface
- [ ] Add TLS configuration options
- [ ] Document all required environment variables
- [ ] Test fail-fast on missing required config

## P2 - Medium Priority (Next Sprint)

### Redis Security
- [ ] Add TLS/SSL support to Redis client
- [ ] Add `REDIS_TLS_ENABLED` configuration
- [ ] Configure connection pool size
- [ ] Add retry logic with exponential backoff
- [ ] Add Redis health to `/health` endpoint
- [ ] Add Redis operation metrics

### Dependencies & Supply Chain
- [ ] Change caret to exact version pinning
- [ ] Enable Dependabot (`.github/dependabot.yml`)
- [ ] Add `safety` to CI pipeline
- [ ] Add `bandit` to CI pipeline
- [ ] Configure vulnerability gate in CI
- [ ] Document dependency upgrade procedure

### Input Validation
- [ ] Add global request size limit middleware
- [ ] Add async timeout protection
- [ ] Document threat model (`docs/THREAT_MODEL.md`)

## P3 - Low Priority (Backlog)

### Observability
- [ ] Add correlation IDs to all logs
- [ ] Implement PII redaction rules
- [ ] Add security audit logging
- [ ] Create security metrics dashboard
- [ ] Define SLOs and SLIs
- [ ] Implement error budget tracking

### Testing
- [ ] Add authentication bypass tests
- [ ] Add rate limit evasion tests
- [ ] Add injection attack tests
- [ ] Add race condition tests
- [ ] Add contract tests for all routers
- [ ] Achieve 80%+ security test coverage

---

## Quick Configuration Reference

### Required Production Environment Variables

```bash
# P0 - Critical
JWT_SECRET_KEY=<32+ character random string>
CORS_ORIGINS=https://app.example.com

# P1 - High
ENVIRONMENT=production
REDIS_HOST=redis.example.com
REDIS_PASSWORD=<password>
TRUSTED_PROXIES=10.0.0.0/8

# P2 - Medium
REDIS_TLS_ENABLED=true
```

### Dependencies to Add

```toml
# P0 - Authentication
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
passlib = {extras = ["bcrypt"], version = "^1.7.4"}

# P1 - Rate Limiting
slowapi = "^0.1.9"

# P2 - Security Scanning (dev only)
safety = "^2.3.0"
bandit = "^1.7.0"
```

---

## Progress Tracking

| Phase | Items | Completed | % |
|-------|-------|-----------|---|
| P0 - Critical | 11 | 0 | 0% |
| P1 - High | 14 | 0 | 0% |
| P2 - Medium | 12 | 0 | 0% |
| P3 - Low | 12 | 0 | 0% |
| **Total** | **49** | **0** | **0%** |

Last Updated: 2025-11-25
