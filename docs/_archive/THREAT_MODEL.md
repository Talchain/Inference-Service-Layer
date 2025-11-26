# Threat Model - Inference Service Layer (ISL)

**Version:** 1.0
**Last Updated:** 2025-11-26
**Classification:** Internal

## 1. System Overview

### 1.1 Purpose
The Inference Service Layer (ISL) is a deterministic scientific computation core for causal inference and decision enhancement. It provides APIs for:
- Causal graph operations (discovery, validation, inference)
- Counterfactual analysis
- Batch processing
- Bayesian teaching
- Robustness analysis
- Contrastive explanations

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Clients                                  │
│              (Web Apps, Mobile Apps, Internal Services)          │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Load Balancer / API Gateway                  │
│                    (TLS Termination, Rate Limiting)              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Service Layer                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Middleware Stack                                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │ │
│  │  │ Auth     │ │ Rate     │ │ Request  │ │ Circuit      │   │ │
│  │  │ (API Key)│ │ Limiting │ │ Limits   │ │ Breaker      │   │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  API Routers                                               │ │
│  │  /causal  /batch  /teaching  /validation  /analysis        │ │
│  │  /explain  /robustness  /team                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Core Services                                             │ │
│  │  CausalValidator, CounterfactualEngine, BayesianTeacher    │ │
│  │  RobustnessAnalyzer, ContrastiveExplainer                  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      External Dependencies                       │
│  ┌──────────┐ ┌──────────────┐ ┌────────────────────────────┐   │
│  │  Redis   │ │  LLM APIs    │ │  Prometheus/Grafana        │   │
│  │  (Cache) │ │  (OpenAI,    │ │  (Monitoring)              │   │
│  │          │ │   Anthropic) │ │                            │   │
│  └──────────┘ └──────────────┘ └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Trust Boundaries

| Boundary | Description |
|----------|-------------|
| TB-1 | Internet ↔ Load Balancer |
| TB-2 | Load Balancer ↔ ISL Application |
| TB-3 | ISL Application ↔ Redis |
| TB-4 | ISL Application ↔ LLM APIs |
| TB-5 | ISL Application ↔ Monitoring |

## 2. Assets

### 2.1 Data Assets

| Asset | Sensitivity | Description |
|-------|-------------|-------------|
| API Keys | HIGH | Authentication credentials for ISL access |
| Causal Graphs | MEDIUM | User-provided causal structures |
| Evidence Data | MEDIUM | Observational/interventional data |
| LLM API Keys | HIGH | Credentials for OpenAI/Anthropic |
| Redis Data | LOW-MEDIUM | Cached computations, rate limit state |
| Logs | LOW-MEDIUM | Request/response logs, error traces |

### 2.2 System Assets

| Asset | Criticality | Description |
|-------|-------------|-------------|
| ISL Application | HIGH | Core computation service |
| Redis Instance | MEDIUM | Caching and rate limiting backend |
| Configuration | HIGH | Environment variables, settings |
| Prometheus Metrics | LOW | Operational metrics |

## 3. Threat Actors

### 3.1 External Attackers
- **Script Kiddies:** Automated scanning, known exploits
- **Opportunistic Attackers:** Looking for misconfigurations
- **Targeted Attackers:** Attempting to steal data or disrupt service

### 3.2 Malicious Insiders
- **Compromised Accounts:** Stolen API keys
- **Rogue Developers:** With access to source code

### 3.3 Accidental Threats
- **Misconfigured Clients:** Excessive requests, malformed data
- **Developer Errors:** Accidental exposure of secrets

## 4. Threats and Mitigations (STRIDE)

### 4.1 Spoofing (Identity)

| ID | Threat | Impact | Mitigation | Status |
|----|--------|--------|------------|--------|
| S-1 | API key theft/guessing | HIGH | Strong API keys, rate limiting | ✅ Implemented |
| S-2 | IP spoofing for rate limit bypass | MEDIUM | Trusted proxy list, multi-factor identification | ✅ Implemented |
| S-3 | Replay attacks | LOW | Request timestamps, nonces | ⏳ Backlog |

### 4.2 Tampering (Data Integrity)

| ID | Threat | Impact | Mitigation | Status |
|----|--------|--------|------------|--------|
| T-1 | Request modification in transit | HIGH | TLS everywhere | ✅ Infrastructure |
| T-2 | Cache poisoning via Redis | MEDIUM | Redis auth, TLS | ✅ Implemented |
| T-3 | Log injection | LOW | Structured logging, sanitization | ⏳ In Progress |

### 4.3 Repudiation (Non-repudiation)

| ID | Threat | Impact | Mitigation | Status |
|----|--------|--------|------------|--------|
| R-1 | Denial of API usage | LOW | Correlation IDs, audit logs | ⏳ In Progress |
| R-2 | Log tampering | MEDIUM | Immutable log storage | ⏳ Backlog |

### 4.4 Information Disclosure

| ID | Threat | Impact | Mitigation | Status |
|----|--------|--------|------------|--------|
| I-1 | API key exposure in logs | HIGH | PII redaction | ⏳ In Progress |
| I-2 | Error messages leaking internals | MEDIUM | Generic error messages | ✅ Implemented |
| I-3 | Prometheus metrics exposure | LOW | Auth on /metrics endpoint | ⏳ Backlog |
| I-4 | Redis data exposure | MEDIUM | Redis TLS + auth | ✅ Implemented |

### 4.5 Denial of Service

| ID | Threat | Impact | Mitigation | Status |
|----|--------|--------|------------|--------|
| D-1 | Request flooding | HIGH | Rate limiting | ✅ Implemented |
| D-2 | Large payload attacks | MEDIUM | Request size limits | ✅ Implemented |
| D-3 | Slow loris attacks | MEDIUM | Request timeouts | ✅ Implemented |
| D-4 | Memory exhaustion | MEDIUM | Circuit breaker | ✅ Implemented |
| D-5 | Computational DoS (complex graphs) | MEDIUM | Complexity limits | ⏳ Backlog |

### 4.6 Elevation of Privilege

| ID | Threat | Impact | Mitigation | Status |
|----|--------|--------|------------|--------|
| E-1 | Injection attacks (code, command) | HIGH | Input validation, Pydantic | ✅ Implemented |
| E-2 | Dependency vulnerabilities | HIGH | Pinned versions, Dependabot | ✅ Implemented |
| E-3 | Container escape | HIGH | Minimal base image, non-root | ⏳ Infrastructure |

## 5. Attack Vectors

### 5.1 API Abuse

**Vector:** Unauthenticated or under-authenticated API access
**Entry Point:** Public API endpoints
**Mitigations:**
- ✅ API key authentication required for protected endpoints
- ✅ Rate limiting per API key and IP
- ✅ Request size limits
- ⏳ Request complexity limits (backlog)

### 5.2 Injection Attacks

**Vector:** Malicious input in causal graph definitions or evidence data
**Entry Point:** POST /api/v1/causal/*, POST /api/v1/batch/*
**Mitigations:**
- ✅ Pydantic validation on all inputs
- ✅ Strict type checking
- ⏳ Additional injection tests (in progress)

### 5.3 Credential Theft

**Vector:** API keys exposed in logs, errors, or transit
**Entry Point:** Logging system, error responses
**Mitigations:**
- ⏳ PII redaction in logs (in progress)
- ✅ Generic error messages
- ✅ TLS for Redis connections

### 5.4 Supply Chain

**Vector:** Compromised dependencies
**Entry Point:** Python packages
**Mitigations:**
- ✅ Pinned critical dependencies
- ✅ Dependabot for updates
- ✅ Security CI workflow (bandit, safety, semgrep)

## 6. Security Controls Summary

### 6.1 Authentication & Authorization
| Control | Implementation | Location |
|---------|----------------|----------|
| API Key Auth | X-API-Key header validation | `src/middleware/auth.py` |
| Public Endpoints | /health, /ready, /metrics, /docs exempted | `src/middleware/auth.py` |

### 6.2 Rate Limiting & DoS Protection
| Control | Implementation | Location |
|---------|----------------|----------|
| Rate Limiting | Redis-backed sliding window | `src/middleware/rate_limiting.py` |
| Request Size Limit | 10MB default | `src/middleware/request_limits.py` |
| Request Timeout | 60s default | `src/middleware/request_limits.py` |
| Circuit Breaker | Memory-based (85% threshold) | `src/middleware/circuit_breaker.py` |

### 6.3 Input Validation
| Control | Implementation | Location |
|---------|----------------|----------|
| Schema Validation | Pydantic models | `src/models/*.py` |
| CORS | Explicit origins only | `src/api/main.py` |
| Content-Type | FastAPI automatic validation | Framework |

### 6.4 Secrets Management
| Control | Implementation | Location |
|---------|----------------|----------|
| API Keys | Environment variable (comma-separated) | `ISL_API_KEYS` |
| Redis Password | Environment variable | `REDIS_PASSWORD` |
| LLM API Keys | Environment variables | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |

### 6.5 Monitoring & Observability
| Control | Implementation | Location |
|---------|----------------|----------|
| Request Logging | Structured JSON logs | `src/api/main.py` |
| Metrics | Prometheus metrics | `src/api/metrics.py` |
| Tracing | X-Trace-Id header | `src/utils/tracing.py` |

## 7. Residual Risks

### 7.1 Accepted Risks

| Risk | Justification | Owner |
|------|---------------|-------|
| /metrics endpoint public | Required for Prometheus scraping | Platform Team |
| In-memory fallback for rate limiting | Better availability vs. strict enforcement | Security Team |

### 7.2 Risks Requiring Further Action

| Risk | Priority | Action Required |
|------|----------|-----------------|
| Computational DoS | P2 | Add graph complexity limits |
| Log tampering | P3 | Implement immutable logging |
| Replay attacks | P3 | Add request nonces |

## 8. Compliance Considerations

### 8.1 Data Protection
- No PII processed by default
- Logs should redact any user-identifiable information
- Causal graphs may contain business-sensitive logic

### 8.2 Audit Requirements
- All API calls logged with correlation IDs
- Authentication failures logged separately
- Configuration changes tracked

## 9. Review Schedule

| Review Type | Frequency | Next Review |
|-------------|-----------|-------------|
| Threat Model | Quarterly | 2026-Q1 |
| Penetration Test | Annually | TBD |
| Dependency Audit | Monthly (Dependabot) | Automated |
| Security CI | Per commit | Automated |

---

## Appendix A: Environment Variables

### Required in Production

| Variable | Purpose | Example |
|----------|---------|---------|
| `ENVIRONMENT` | Environment indicator | `production` |
| `ISL_API_KEYS` | Valid API keys | `key1,key2,key3` |
| `CORS_ORIGINS` | Allowed CORS origins | `https://app.example.com` |

### Security-Related Optional

| Variable | Purpose | Default |
|----------|---------|---------|
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | Rate limit threshold | `100` |
| `TRUSTED_PROXIES` | Trusted proxy CIDRs | `` |
| `REDIS_PASSWORD` | Redis authentication | `` |
| `REDIS_TLS_ENABLED` | Enable Redis TLS | `false` |
| `MAX_REQUEST_SIZE_MB` | Max request body size | `10` |
| `REQUEST_TIMEOUT_SECONDS` | Request timeout | `60` |

## Appendix B: Security Testing Checklist

- [x] API key authentication tests
- [x] Rate limiting tests
- [x] Request size limit tests
- [x] CORS configuration tests
- [x] Production config validation tests
- [ ] Injection attack tests
- [ ] Race condition tests
- [ ] Fuzz testing
