# ISL Pilot Monitoring Runbook

## Overview

This runbook defines operational procedures for monitoring ISL during pilot launch. Designed for operations teams with limited ISL-specific knowledge.

---

## Daily Monitoring Checklist (5 minutes)

**Every morning (09:00 UTC):**

1. **Check Grafana Dashboard** → [ISL Overview Dashboard URL]
   - [ ] All panels showing data (no "No Data" errors)
   - [ ] P95 latency < 2.5s (target: <2.0s, alert: >2.5s)
   - [ ] Error rate < 2% (target: <1%, alert: >5%)
   - [ ] Redis memory < 80% (target: <70%, alert: >80%)

2. **Review Prometheus Alerts** → [Alerts URL]
   - [ ] No firing alerts in past 24h
   - [ ] If alerts fired: check runbook section below

3. **Spot-Check Health Endpoint**
   ```bash
   curl https://isl-staging.olumi.ai/health
   # Expected: {"status": "healthy", "redis": {"connected": true}}
   ```

4. **Check Error Logs (if error rate > 1%)**
   ```bash
   # View last 100 errors
   kubectl logs -l app=isl --tail=100 | grep ERROR
   ```

**Weekly Review (Friday 16:00 UTC):**

5. **Performance Trends**
   - [ ] P95 latency trend (increasing/stable/decreasing?)
   - [ ] Cache hit rate trend (target: >30%, ideal: >50%)
   - [ ] Error pattern analysis (same errors recurring?)

6. **Capacity Planning**
   - [ ] Redis memory trend (growth rate)
   - [ ] Request volume trend (growing/stable?)
   - [ ] Any endpoints consistently slow?

---

## Alert Response Procedures

### **CRITICAL: ISL Service Down**

**Alert:** `isl_service_up == 0` for > 1 minute

**Symptoms:**
- Health endpoint returns 5xx or unreachable
- All ISL requests failing
- PLoT showing "analysis unavailable"

**Immediate Actions (< 5 minutes):**

1. **Check service status:**
   ```bash
   kubectl get pods -l app=isl
   # Look for: CrashLoopBackOff, ImagePullBackOff, Error
   ```

2. **Check recent logs:**
   ```bash
   kubectl logs -l app=isl --tail=50
   # Look for: startup errors, dependency failures, OOM kills
   ```

3. **Common causes & fixes:**

   **Cause: OOM Kill (Out of Memory)**
   ```bash
   # Check if pod was OOM killed
   kubectl describe pod <pod-name> | grep -A 5 "Last State"
   # Fix: Increase memory limit (contact DevOps)
   ```

   **Cause: Redis Connection Failed**
   ```bash
   # Check Redis health
   redis-cli -h $REDIS_HOST ping
   # Expected: PONG
   # If fails: Check Redis service status, contact DevOps
   ```

   **Cause: Deployment Issue**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/isl
   # Then investigate new version issue
   ```

4. **Escalation:**
   - If not resolved in 15 minutes → Page on-call developer
   - Provide: pod logs, error messages, recent changes
   - Slack: #isl-incidents channel

---

### **HIGH: Error Rate > 5%**

**Alert:** `rate(isl_requests_total{status="error"}[5m]) > 0.05` for > 5 minutes

**Symptoms:**
- PLoT showing intermittent failures
- Users seeing "analysis unavailable" messages
- Specific endpoints failing consistently

**Investigation Steps:**

1. **Identify error types:**
   ```bash
   # Get error code distribution
   curl https://isl-staging.olumi.ai/metrics | grep isl_errors_total
   # Shows: isl_errors_total{code="INVALID_INPUT"} 42
   ```

2. **Common error patterns:**

   **INVALID_INPUT (40%)** → User model issues (expected, not service fault)
   ```bash
   # Check if error rate corresponds to bad user inputs
   # Action: Monitor but don't escalate unless >60%
   ```

   **TIMEOUT (>20%)** → Performance degradation
   ```bash
   # Check P95 latency
   # If P95 > 3s: Performance issue, continue investigation
   # Check Redis memory (might be evicting too aggressively)
   ```

   **INTERNAL_ERROR (>5%)** → Service bug
   ```bash
   # Get stack traces
   kubectl logs -l app=isl --tail=200 | grep -A 20 "INTERNAL_ERROR"
   # Action: Create incident, escalate to developers immediately
   ```

3. **Temporary mitigation:**
   ```bash
   # If specific endpoint failing, consider circuit breaker
   # Contact PLoT team to reduce traffic to failing endpoint
   ```

---

### **MEDIUM: High P95 Latency**

**Alert:** `histogram_quantile(0.95, rate(isl_request_duration_seconds_bucket[15m])) > 2.5` for > 15 minutes

**Symptoms:**
- Requests taking longer than usual
- Users reporting slow analysis
- Timeout errors increasing

**Investigation Steps:**

1. **Check which endpoints are slow:**
   ```bash
   curl https://isl-staging.olumi.ai/metrics | grep duration_seconds_bucket | grep /api/v1
   # Identify: Which endpoint has highest P95?
   ```

2. **Common causes:**

   **Redis cache misses:**
   ```bash
   # Check cache hit rate
   curl https://isl-staging.olumi.ai/metrics | grep cache_hit_rate
   # If < 20%: Users making unique requests (expected early in pilot)
   # If < 20% sustained: Investigate cache TTLs or key patterns
   ```

   **Large model complexity:**
   ```bash
   # Check request payload sizes
   # If consistently >50KB: Users building complex models
   # Action: Expected behaviour, monitor for sustained growth
   ```

   **Resource contention:**
   ```bash
   # Check CPU/memory usage
   kubectl top pods -l app=isl
   # If CPU > 80%: Consider scaling (contact DevOps)
   ```

3. **Escalation threshold:**
   - P95 > 5s for > 30 minutes → Escalate to developers
   - P95 > 10s at any time → Immediate escalation

---

### **MEDIUM: Redis Memory High**

**Alert:** `redis_memory_used_bytes / redis_memory_max_bytes > 0.80` for > 10 minutes

**Symptoms:**
- Cache evictions increasing
- Cache hit rate decreasing
- Potential performance degradation

**Investigation Steps:**

1. **Check current Redis state:**
   ```bash
   redis-cli -h $REDIS_HOST INFO memory
   # Look for:
   # - used_memory_human: Current usage
   # - maxmemory_human: Configured max
   # - evicted_keys: Total evictions
   ```

2. **Check eviction rate:**
   ```bash
   redis-cli -h $REDIS_HOST INFO stats | grep evicted_keys
   # If increasing rapidly (>10/minute sustained): Investigate
   ```

3. **Investigate key patterns:**
   ```bash
   # Check key count by prefix
   redis-cli -h $REDIS_HOST --scan --pattern "isl:*" | cut -d: -f2 | sort | uniq -c
   # Shows distribution: beliefs, ident, result, etc.
   ```

4. **Common causes:**

   **Expected growth (early pilot):**
   - Action: Monitor, no intervention needed unless >90%

   **TTL misconfiguration:**
   ```bash
   # Check if keys have TTL
   redis-cli -h $REDIS_HOST --scan --pattern "isl:*" | xargs -n 1 redis-cli TTL | sort | uniq -c
   # If many showing "-1" (no TTL): BUG, escalate to developers
   ```

   **Large cached values:**
   ```bash
   # Check value sizes
   redis-cli -h $REDIS_HOST DEBUG OBJECT isl:result:some_hash
   # If individual values >1MB: Investigate model complexity
   ```

5. **Temporary mitigation:**
   ```bash
   # Reduce TTLs temporarily (requires config change)
   # Contact developers for guidance before changing
   ```

---

## Metrics Reference

### **Key Metrics to Watch**

**Latency (P95):**
- Target: < 2.0s
- Warning: > 2.5s
- Critical: > 5.0s

**Error Rate:**
- Target: < 1%
- Warning: > 2%
- Critical: > 5%

**Cache Hit Rate:**
- Target: > 30%
- Warning: < 20%
- Critical: < 10% (sustained)

**Redis Memory:**
- Target: < 70%
- Warning: > 80%
- Critical: > 90%

### **Grafana Dashboard Panels**

1. **Request Rate** - Requests per second by endpoint
2. **P50/P95/P99 Latency** - Response time distribution
3. **Error Rate** - Percentage of failed requests
4. **Error Types** - Distribution by error code
5. **Cache Performance** - Hit rate and miss rate
6. **Redis Memory** - Memory usage over time
7. **Redis Operations** - Ops/sec and command distribution
8. **Active Users** - Unique users making requests

### **Prometheus Queries**

**Request rate:**
```promql
rate(isl_requests_total[5m])
```

**P95 latency:**
```promql
histogram_quantile(0.95, rate(isl_request_duration_seconds_bucket[5m]))
```

**Error rate:**
```promql
rate(isl_requests_total{status="error"}[5m]) / rate(isl_requests_total[5m])
```

**Cache hit rate:**
```promql
rate(isl_cache_hits_total[5m]) / rate(isl_cache_requests_total[5m])
```

---

## Escalation Contacts

**Level 1: Operations Team**
- Slack: #isl-operations
- Response: 24/7 during pilot
- Handles: Monitoring, basic troubleshooting, service restarts

**Level 2: On-Call Developer**
- Slack: @isl-oncall
- PagerDuty: [link]
- Response: Within 15 minutes
- Handles: Service bugs, performance issues, data corruption

**Level 3: Product/Architecture**
- Contact: Paul (@paul) or technical lead
- Response: Business hours
- Handles: Design decisions, scope changes, major incidents

---

## Known Issues & Workarounds

### **Issue: Intermittent Redis Timeouts**

**Symptoms:** Occasional `REDIS_TIMEOUT` errors (~0.1-0.5% of requests)

**Cause:** Redis hosted on shared infrastructure with occasional network blips

**Workaround:** Automatic retry with exponential backoff (already implemented)

**Action:** Monitor, escalate only if >1% sustained

---

### **Issue: Large Model Timeouts**

**Symptoms:** Models with >30 nodes occasionally timeout (>30s)

**Cause:** Monte Carlo simulation (10k samples) computationally expensive

**Workaround:** Users can simplify models or retry

**Action:** Expected behaviour, monitor P99 latency for concerning trends

---

## Deployment Procedures

### **Staging Deployment**

**Pre-deployment checklist:**
- [ ] All tests passing in CI
- [ ] Performance benchmarks validated
- [ ] Breaking changes documented (should be none in v1)
- [ ] Rollback plan confirmed

**Deployment steps:**
```bash
# 1. Deploy to staging
kubectl apply -f k8s/staging/deployment.yaml

# 2. Wait for rollout
kubectl rollout status deployment/isl -n staging

# 3. Health check
curl https://isl-staging.olumi.ai/health
# Expected: {"status": "healthy"}

# 4. Smoke test
./scripts/smoke_test.sh staging
# Expected: All tests pass

# 5. Monitor for 15 minutes
# Watch Grafana dashboard for anomalies

# 6. If issues: Rollback
kubectl rollout undo deployment/isl -n staging
```

**Post-deployment validation:**
- [ ] Health endpoint returns healthy
- [ ] All metrics showing data in Grafana
- [ ] No new error patterns in logs
- [ ] P95 latency within expected range

---

### **Production Deployment**

**Pre-deployment requirements:**
- [ ] Staging validated for 48+ hours
- [ ] No critical issues in staging
- [ ] Change approval from product team
- [ ] Operations team notified 24h in advance

**Deployment window:** Tuesday/Wednesday 10:00-12:00 UTC (avoid Mondays/Fridays)

**Deployment steps:** Same as staging but with production config

**Rollback criteria:**
- Error rate > 10% for > 5 minutes
- P95 latency > 10s for > 5 minutes
- Any CRITICAL alert fires

---

## Performance Baselines (Updated Weekly)

**Week 0 (Pilot Launch):**
- Request volume: ~10-50 requests/day
- P95 latency: 0.5-1.5s (mostly cached)
- Error rate: 2-5% (user input validation)
- Cache hit rate: 15-25% (cold cache)

**Expected Growth:**
- Week 2: 100-200 requests/day
- Week 4: 500-1000 requests/day
- Cache hit rate should improve to 40-60%

---

## Appendix: Useful Commands

**View recent logs:**
```bash
kubectl logs -l app=isl --tail=100 -f
```

**Check Redis connection:**
```bash
redis-cli -h $REDIS_HOST ping
```

**Manual health check:**
```bash
curl -H "X-Request-Id: manual-check-$(date +%s)" \
  https://isl-staging.olumi.ai/health
```

**Get metrics:**
```bash
curl https://isl-staging.olumi.ai/metrics
```

**Describe pod (for debugging):**
```bash
kubectl describe pod <pod-name>
```

**Get pod resource usage:**
```bash
kubectl top pod -l app=isl
```

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| v1.0 | 2025-11-20 | Initial pilot runbook | ISL Team |

---

**For questions or updates to this runbook, contact: #isl-operations**
