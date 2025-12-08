# ISL Security Guide

Security configuration and best practices for ISL.

---

## Authentication

### API Key Authentication

All protected endpoints require the `X-API-Key` header:

```bash
curl -H "X-API-Key: your_key" https://isl.example.com/api/v1/...
```

### Environment Variables

ISL supports two environment variables for API keys:

```bash
# Preferred: Multiple keys (comma-separated)
ISL_API_KEYS=key1,key2,key3

# Legacy: Single key (still supported for backward compatibility)
ISL_API_KEY=single_key
```

**Precedence:** `ISL_API_KEYS` takes priority if both are set.

### Public Endpoints (No Auth Required)

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc documentation
- `GET /openapi.json` - OpenAPI spec

### Generating API Keys

```bash
# Generate a secure API key
openssl rand -hex 32
# → 7k9mP2nX8vQ4rL6wF3jH5tY1cB0zS...

# Recommended format
isl_prod_<random_hex>
```

### Key Rotation

1. Add new key to `ISL_API_KEYS`
2. Update clients to use new key
3. Remove old key after transition period

---

## CORS Configuration

### Setting Allowed Origins

```bash
# Production (explicit domains only)
CORS_ORIGINS=https://plot.olumi.ai,https://tae.olumi.ai,https://cee.olumi.ai

# Development
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Security Rules

- **No wildcards in production** - `*` is rejected
- **HTTPS required** - HTTP origins rejected in production
- **Explicit whitelist** - Only listed origins allowed

### CORS Headers Returned

```http
Access-Control-Allow-Origin: https://your-domain.com
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type, X-API-Key
Access-Control-Max-Age: 600
```

---

## Rate Limiting

### Configuration

```bash
RATE_LIMIT_REQUESTS_PER_MINUTE=100  # Default
```

### How It Works

1. **Identification:** By API key (if present) or client IP
2. **Window:** Sliding 1-minute window
3. **Scope:** Per-client, not global

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1700000000
```

### When Rate Limited (429)

```json
{
  "code": "RATE_LIMITED",
  "message": "Rate limit exceeded",
  "retry_after": 42
}
```

---

## Request Limits

### Size Limits

```bash
MAX_REQUEST_SIZE_MB=10  # Default
```

Requests exceeding this return `413 Request Too Large`.

### Timeout Limits

```bash
REQUEST_TIMEOUT_SECONDS=60  # Default
```

Requests exceeding this return `504 Request Timeout`.

### Input Validation Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| Max DAG nodes | 50 | Prevent DoS |
| Max DAG edges | 200 | Prevent DoS |
| Max string length | 10,000 chars | Memory protection |
| Max list items | 1,000 | Memory protection |

---

## Logging & Privacy

### PII Redaction

ISL automatically redacts sensitive patterns:

- Email addresses: `user@example.com` → `[EMAIL_REDACTED]`
- Phone numbers: `555-123-4567` → `[PHONE_REDACTED]`
- SSN patterns: `123-45-6789` → `[SSN_REDACTED]`
- API keys in logs: `key123...` → `[API_KEY_REDACTED]`

### Log Format

```json
{
  "timestamp": "2025-11-26T10:00:00Z",
  "level": "INFO",
  "message": "Request processed",
  "request_id": "req_abc123",
  "client_ip": "[IP_REDACTED]",
  "path": "/api/v1/validation/assumptions"
}
```

### Correlation IDs

Every request gets a unique ID:
- Header: `X-Trace-Id`
- Logs: `request_id` field
- Responses: `metadata.request_id`

---

## Security Headers

Automatically added to all responses:

```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

---

## Network Security

### TLS Requirements

- **Minimum:** TLS 1.2
- **Recommended:** TLS 1.3
- **Certificates:** Valid, not self-signed (production)

### Proxy Configuration

When behind a load balancer, set:

```bash
TRUSTED_PROXIES=10.0.0.0/8,172.16.0.0/12
```

This ensures:
- Correct client IP from `X-Forwarded-For`
- Proper rate limiting per real client

---

## Production Checklist

### Required Settings

- [ ] `ISL_API_KEYS` set with secure, unique keys
- [ ] `CORS_ORIGINS` configured (no wildcards)
- [ ] `ENVIRONMENT=production`
- [ ] TLS termination configured at load balancer
- [ ] `TRUSTED_PROXIES` set if behind proxy

### Recommended Settings

- [ ] Redis password set (`REDIS_PASSWORD`)
- [ ] Prometheus metrics secured (internal network only)
- [ ] Log aggregation configured
- [ ] Alerting on auth failures

### Security Monitoring

Monitor these metrics:

```promql
# Authentication failures
sum(rate(isl_auth_failures_total[5m])) > 10

# Rate limit hits
sum(rate(isl_rate_limit_hits_total[5m])) > 100

# Error rate
sum(rate(isl_request_errors_total[5m])) / sum(rate(isl_requests_total[5m])) > 0.05
```

---

## Incident Response

### Compromised API Key

1. **Immediately:** Remove key from `ISL_API_KEYS`
2. **Audit:** Check logs for unauthorized access
3. **Notify:** Alert affected integrations
4. **Replace:** Issue new key to legitimate users

### DDoS/High Traffic

1. **Rate limits:** Auto-enforced per client
2. **Circuit breaker:** Memory > 85% rejects requests
3. **Scale:** Add replicas if legitimate traffic

### Data Breach

ISL is stateless - no user data persisted. However:

1. Check Redis cache for any leaked data
2. Review logs for data exfiltration
3. Rotate all API keys as precaution

---

## Compliance

### GDPR

- No PII stored permanently
- PII redacted from logs
- Request/response data not persisted

### OWASP Top 10

| Risk | Mitigation |
|------|------------|
| Injection | Pydantic validation, no raw SQL |
| Broken Auth | API key validation, constant-time comparison |
| Sensitive Data | TLS, PII redaction, no storage |
| XXE | JSON only, no XML parsing |
| Broken Access Control | All endpoints auth-protected |
| Security Misconfiguration | Environment-based config, no defaults |
| XSS | JSON responses only, proper headers |
| Insecure Deserialization | Pydantic strict mode |
| Vulnerable Components | Dependabot, regular updates |
| Insufficient Logging | Structured logging, correlation IDs |

---

## Contact

Security issues: Open a confidential GitHub issue or contact the security team.
