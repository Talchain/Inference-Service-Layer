# Multi-Replica Rate Limiting Test Plan

## Overview

This document outlines the test plan for validating distributed rate limiting across multiple ISL replicas using Redis as the shared state backend.

## Prerequisites

- Kubernetes cluster or Docker Compose environment
- Redis instance accessible by all replicas
- Load testing tool (k6, locust, or hey)
- Prometheus/Grafana for monitoring

## Test Environment Setup

### Docker Compose (Development)

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --requirepass ${REDIS_PASSWORD}

  isl-replica-1:
    build: .
    environment:
      - ENVIRONMENT=staging
      - ISL_API_KEYS=${ISL_API_KEYS}
      - REDIS_HOST=redis
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - RATE_LIMIT_REQUESTS_PER_MINUTE=100
    ports:
      - "8001:8000"
    depends_on:
      - redis

  isl-replica-2:
    build: .
    environment:
      - ENVIRONMENT=staging
      - ISL_API_KEYS=${ISL_API_KEYS}
      - REDIS_HOST=redis
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - RATE_LIMIT_REQUESTS_PER_MINUTE=100
    ports:
      - "8002:8000"
    depends_on:
      - redis

  isl-replica-3:
    build: .
    environment:
      - ENVIRONMENT=staging
      - ISL_API_KEYS=${ISL_API_KEYS}
      - REDIS_HOST=redis
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - RATE_LIMIT_REQUESTS_PER_MINUTE=100
    ports:
      - "8003:8000"
    depends_on:
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - isl-replica-1
      - isl-replica-2
      - isl-replica-3
```

### Nginx Load Balancer Config

```nginx
upstream isl_backend {
    least_conn;
    server isl-replica-1:8000;
    server isl-replica-2:8000;
    server isl-replica-3:8000;
}

server {
    listen 80;

    location / {
        proxy_pass http://isl_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Test Scenarios

### Test 1: Basic Distributed Rate Limiting

**Objective:** Verify that rate limits are enforced globally across all replicas.

**Setup:**
- 3 replicas with RATE_LIMIT_REQUESTS_PER_MINUTE=100
- Round-robin load balancing

**Test Steps:**
1. Send 100 requests rapidly through load balancer
2. Verify all 100 requests succeed (distributed across replicas)
3. Send 10 more requests
4. Verify all 10 are rate limited (429 response)

**Expected Results:**
- First 100 requests: 200 OK
- Requests 101-110: 429 Too Many Requests
- Rate limit headers present on all responses

**k6 Test Script:**
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10,
  iterations: 120,
};

export default function () {
  const res = http.get('http://localhost:8000/api/v1/causal/validate', {
    headers: {
      'X-API-Key': __ENV.API_KEY,
      'Content-Type': 'application/json',
    },
  });

  check(res, {
    'status is 200 or 429': (r) => r.status === 200 || r.status === 429,
    'has rate limit headers': (r) => r.headers['X-Ratelimit-Limit'] !== undefined,
  });
}
```

### Test 2: Per-API-Key Rate Limiting

**Objective:** Verify that different API keys have independent rate limits.

**Test Steps:**
1. Send 50 requests with API_KEY_1
2. Send 50 requests with API_KEY_2
3. Send 60 more requests with API_KEY_1
4. Verify API_KEY_1 gets rate limited at request 101
5. Verify API_KEY_2 can still make 50 more requests

**Expected Results:**
- API_KEY_1: 100 allowed, then rate limited
- API_KEY_2: 100 allowed independently

### Test 3: Fallback to In-Memory on Redis Failure

**Objective:** Verify graceful degradation when Redis is unavailable.

**Test Steps:**
1. Start all replicas with Redis running
2. Stop Redis container
3. Send requests to each replica
4. Verify rate limiting still works (per-replica, not global)
5. Restart Redis
6. Verify global rate limiting resumes

**Expected Results:**
- No 5xx errors when Redis fails
- Each replica enforces its own limit (less strict)
- Logs show fallback activation
- Recovery is automatic

### Test 4: Rate Limit Window Boundary

**Objective:** Verify rate limits reset correctly at window boundaries.

**Test Steps:**
1. Send 100 requests (hitting limit)
2. Wait 60 seconds
3. Send 100 more requests
4. Verify all succeed

**Expected Results:**
- Window resets after 60 seconds
- No carryover of old requests

### Test 5: High Concurrency Stress Test

**Objective:** Verify rate limiting accuracy under high load.

**Test Steps:**
1. Send 1000 concurrent requests from single client
2. Count successful vs rate-limited responses

**Expected Results:**
- Approximately 100 ± 5 successful (some variance due to timing)
- Remaining requests rate limited
- No race conditions causing >110 successes

**k6 Stress Test:**
```javascript
export const options = {
  scenarios: {
    stress: {
      executor: 'constant-vus',
      vus: 100,
      duration: '10s',
    },
  },
};
```

### Test 6: Multi-Client Isolation

**Objective:** Verify different clients have independent limits.

**Test Steps:**
1. Client A (IP 1) sends 100 requests
2. Client B (IP 2) sends 100 requests simultaneously
3. Verify both can make 100 requests

**Expected Results:**
- Each client gets full rate limit allocation
- No cross-contamination between client limits

## Metrics to Monitor

During all tests, monitor:

1. **Prometheus Metrics:**
   - `isl_rate_limit_hits_total` - Should match rejected requests
   - `isl_rate_limit_checks_total{result="allowed"}` - Should match successful requests
   - `isl_rate_limit_checks_total{result="blocked"}` - Should match 429 responses

2. **Redis Metrics:**
   - Memory usage
   - Connected clients
   - Operations per second
   - Key count in `isl:ratelimit:*` namespace

3. **Application Logs:**
   - `rate_limit_exceeded` events with correlation IDs
   - Redis fallback warnings if applicable

## Pass/Fail Criteria

| Test | Pass Criteria |
|------|---------------|
| Basic Distributed | ≤105 requests allowed (5% tolerance) |
| Per-API-Key | Independent limits per key |
| Redis Failover | No errors, graceful degradation |
| Window Boundary | Clean reset at 60s |
| Stress Test | ≤110 requests allowed (10% tolerance) |
| Multi-Client | Full allocation per client |

## Automation

Add to CI/CD pipeline:

```yaml
# .github/workflows/rate-limit-test.yml
name: Rate Limit Integration Test

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  rate-limit-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start test environment
        run: docker-compose -f docker-compose.test.yml up -d

      - name: Wait for services
        run: sleep 30

      - name: Run k6 tests
        uses: grafana/k6-action@v0.3.1
        with:
          filename: tests/load/rate_limit_test.js

      - name: Collect logs
        if: always()
        run: docker-compose logs > test-logs.txt

      - name: Teardown
        if: always()
        run: docker-compose -f docker-compose.test.yml down -v
```

## Manual Testing Checklist

- [ ] Deploy 3+ replicas with shared Redis
- [ ] Verify Redis connectivity from all replicas
- [ ] Run Test 1: Basic Distributed Rate Limiting
- [ ] Run Test 2: Per-API-Key Rate Limiting
- [ ] Run Test 3: Redis Failover
- [ ] Run Test 4: Window Boundary
- [ ] Run Test 5: Stress Test
- [ ] Run Test 6: Multi-Client Isolation
- [ ] Review Prometheus metrics
- [ ] Review security audit logs
- [ ] Document any deviations

## Appendix: Redis Commands for Debugging

```bash
# View all rate limit keys
redis-cli KEYS "isl:ratelimit:*"

# Check request count for specific identifier
redis-cli ZCARD "isl:ratelimit:ip:192.168.1.1"

# View all entries with scores (timestamps)
redis-cli ZRANGE "isl:ratelimit:ip:192.168.1.1" 0 -1 WITHSCORES

# Clear all rate limit data (testing only!)
redis-cli KEYS "isl:ratelimit:*" | xargs redis-cli DEL
```
