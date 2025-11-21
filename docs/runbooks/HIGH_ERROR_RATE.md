# Runbook: ISL High Error Rate

## Alert
**Alert Name:** `ISLHighErrorRate`
**Severity:** Critical
**Threshold:** Error rate > 5% for 5 minutes
**Dashboard:** https://grafana.olumi.com/d/isl-overview

## Symptoms
- Prometheus alert firing in #isl-alerts
- Error rate spike in Grafana dashboard
- User reports of 500/502/503 errors
- Increased 5xx status codes in logs

## Impact
- **Availability:** Users unable to validate causal models, generate counterfactuals
- **Downstream:** PLoT, CEE, UI teams blocked
- **Pilot:** User experience severely degraded
- **SLO:** 99.5% availability target at risk

## Diagnosis

### Step 1: Identify Affected Endpoints
```bash
# Open Grafana ISL Overview dashboard
# Look at "Error Rate by Endpoint" panel
# Identify which specific endpoints have errors

# Or query Prometheus directly
kubectl port-forward -n monitoring svc/prometheus-server 9090:80

# Query: Error rate by endpoint
sum(rate(isl_requests_total{status=~"5.."}[5m])) by (endpoint)
/ sum(rate(isl_requests_total[5m])) by (endpoint)
```

### Step 2: Check Recent Logs
```bash
# Get recent error logs from ISL pods
kubectl logs -n production \
  deployment/isl \
  --since=10m \
  --tail=100 \
  | grep "ERROR"

# Check for specific error patterns
kubectl logs -n production deployment/isl --since=10m \
  | grep -E "(500|Exception|Failed|Traceback)"

# Follow logs in real-time
kubectl logs -n production deployment/isl -f \
  | grep "ERROR"
```

### Step 3: Check Dependencies
```bash
# Check Redis connectivity
redis-cli -h redis.olumi.com ping
# Expected: PONG

# Check Redis memory
redis-cli -h redis.olumi.com INFO memory

# Check database connectivity
psql -h db.olumi.com -U isl_user -d isl_db -c "SELECT 1"
# Expected: 1 row

# Check LLM API status
curl -I https://api.openai.com/v1/models
# Expected: 200 OK

curl -I https://api.anthropic.com/v1/messages
# Expected: 200 OK
```

### Step 4: Check Resource Usage
```bash
# Check pod resource usage
kubectl top pods -n production -l app=isl

# Check for OOMKilled or CrashLoopBackOff
kubectl get pods -n production -l app=isl

# Check pod events
kubectl describe pods -n production -l app=isl
```

## Common Causes & Solutions

### Cause 1: Redis Connection Lost
**Symptoms:**
- Cache-related errors in logs
- Increased latency
- LLM fallback rate spike

**Fix:**
```bash
# Restart ISL pods to reset Redis connection pool
kubectl rollout restart deployment/isl -n production

# Monitor during restart
watch kubectl get pods -n production -l app=isl

# Or scale down/up for immediate effect
kubectl scale deployment/isl --replicas=0 -n production
sleep 10
kubectl scale deployment/isl --replicas=3 -n production
```

**Validation:**
- Error rate drops below 1%
- Cache hit rate returns to >30%
- Latency returns to normal (<5s P95)
- Check: `kubectl logs -n production deployment/isl | grep "Redis connection established"`

### Cause 2: LLM API Rate Limiting
**Symptoms:**
- Errors on `/deliberation`, `/preferences` endpoints
- Logs show "Rate limit exceeded" from OpenAI/Anthropic
- LLM fallback rate >20%

**Fix:**
```bash
# Check current LLM request rate
curl -s http://isl.olumi.com/metrics | grep llm_requests_total

# Enable aggressive fallback to rule-based
kubectl set env deployment/isl \
  LLM_FALLBACK_TO_RULES=true \
  MAX_COST_PER_SESSION=0.50 \
  -n production

# Or temporarily disable LLM (emergency only)
kubectl set env deployment/isl \
  LLM_ENABLED=false \
  -n production
```

**Validation:**
- `isl_llm_fallback_to_rules_total` metric increases
- Error rate decreases
- Deliberation still works (degraded quality)
- Check: `kubectl logs | grep "LLM fallback enabled"`

**Rollback when resolved:**
```bash
kubectl set env deployment/isl \
  LLM_FALLBACK_TO_RULES=false \
  LLM_ENABLED=true \
  -n production
```

### Cause 3: Database Connection Pool Exhausted
**Symptoms:**
- Timeout errors in logs
- "Too many connections" errors
- Slow response times across all endpoints

**Check:**
```bash
# Check active connections
psql -h db.olumi.com -U isl_user -d isl_db \
  -c "SELECT count(*) FROM pg_stat_activity WHERE datname='isl_db'"

# Check max connections
psql -h db.olumi.com -U isl_user -d isl_db \
  -c "SHOW max_connections"
```

**Fix:**
```bash
# If near limit, restart ISL pods to release connections
kubectl rollout restart deployment/isl -n production

# If persistent, increase connection pool size
kubectl set env deployment/isl \
  DATABASE_POOL_SIZE=20 \
  -n production

# Or increase database max_connections (requires DB restart)
# Contact DBA team
```

**Validation:**
- Connection count drops
- Error rate normalizes
- No more timeout errors
- Check: `psql ... -c "SELECT count(*) ..."`  shows lower count

### Cause 4: Bad Deployment
**Symptoms:**
- Errors started immediately after deployment
- New code introduced bugs
- Previously working endpoints now failing

**Check:**
```bash
# Check recent deployments
kubectl rollout history deployment/isl -n production

# Check current image/version
kubectl get deployment isl -n production -o yaml | grep image:
```

**Fix:**
```bash
# Rollback to previous version
kubectl rollout undo deployment/isl -n production

# Or rollback to specific revision
kubectl rollout undo deployment/isl -n production --to-revision=<N>

# Monitor rollback
kubectl rollout status deployment/isl -n production
```

**Validation:**
- Error rate drops immediately
- All endpoints return to healthy status
- Logs show previous version running
- Check: `kubectl get pods -n production -l app=isl -o yaml | grep image:`

### Cause 5: External Service Outage
**Symptoms:**
- Specific endpoints failing (e.g., only counterfactuals)
- Third-party API errors in logs
- Multiple services affected simultaneously

**Check:**
```bash
# Check status pages
curl https://status.openai.com
curl https://status.anthropic.com
curl https://status.render.com

# Check recent incidents
# Visit status pages in browser
```

**Fix:**
```bash
# If LLM providers down: enable fallback
kubectl set env deployment/isl \
  LLM_FALLBACK_TO_RULES=true \
  -n production

# If other services down: contact vendor support
# Monitor their status pages
# Communicate to users via #isl-alerts
```

**Validation:**
- Fallback metrics increase
- Core functionality maintained
- Users notified of degraded service

## Escalation

**If error rate remains >5% after 15 minutes:**

### Level 1: Page On-Call Engineer
```bash
# Via PagerDuty
# Subject: URGENT - ISL High Error Rate
# Body: Error rate X% for Y minutes. Tried: [list troubleshooting steps]. Need assistance.
```

### Level 2: Notify Stakeholders
```slack
# Post in #isl-incidents
@here ISL experiencing high error rate (X%).
Affecting: [list endpoints]
Impact: [describe user impact]
Actions taken: [list steps]
ETA: Investigating
```

### Level 3: Update Status Page
```bash
# Update status.olumi.com
# Status: Degraded Performance
# Services Affected: ISL API
# Message: "We're investigating elevated error rates. Some requests may fail. We're working to resolve this quickly."
```

### Level 4: Notify Pilot Users
```bash
# Email to pilot users
# Subject: ISL Service Degradation
# Body: Brief description, expected resolution time, workarounds if available
```

### Level 5: Engage Vendor Support
- **OpenAI:** support@openai.com (if LLM issue)
- **Anthropic:** support@anthropic.com (if LLM issue)
- **Render:** support@render.com (if infrastructure)
- **Redis Cloud:** support@redis.com (if cache)

## Prevention

### Proactive Measures
1. **Monitor error rate trends** - Review weekly in team sync
2. **Load test before major releases** - Catch issues in staging
3. **Set up canary deployments** - 10% → 50% → 100% rollout
4. **Improve retry logic** - Exponential backoff with jitter
5. **Add circuit breakers** - Fail fast for external dependencies
6. **Increase test coverage** - Target: >60% coverage
7. **Add integration tests** - Catch breaking changes early

### Alerting Improvements
- **Adjust thresholds** based on baseline (currently 5%)
- **Add warning alert** at 2% (early detection)
- **Create runbook links** in alert messages
- **Test alert channels** monthly

### Code Quality
- **Code review focus** on error handling
- **Require unit tests** for new endpoints
- **Integration tests** for critical paths
- **Load testing** in CI/CD pipeline

## Post-Incident

### Immediate (within 1 hour of resolution)
- [ ] Confirm error rate < 1%
- [ ] Verify all endpoints healthy
- [ ] Check SLO compliance
- [ ] Update #isl-incidents with resolution

### Short-term (within 24 hours)
- [ ] Create post-mortem: `docs/post-mortems/YYYY-MM-DD-high-error-rate.md`
- [ ] Update this runbook with learnings
- [ ] Create tickets for prevention measures
- [ ] Share learnings in team sync

### Long-term (within 1 week)
- [ ] Implement prevention measures
- [ ] Add monitoring for root cause
- [ ] Update alerting if needed
- [ ] Schedule retro with team

## Post-Mortem Template

```markdown
# Post-Mortem: ISL High Error Rate - YYYY-MM-DD

## Summary
- **Date:** YYYY-MM-DD HH:MM UTC
- **Duration:** X hours Y minutes
- **Severity:** Critical
- **Impact:** X% of requests failed

## Timeline
- HH:MM - Alert fired
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix applied
- HH:MM - Service recovered

## Root Cause
[Detailed description of what went wrong]

## Resolution
[What fixed it]

## Impact
- Users affected: X
- Requests failed: Y
- Revenue impact: $Z (if applicable)
- SLO breach: Yes/No

## Action Items
1. [ ] [Prevention measure 1] - Owner: [Name] - Due: [Date]
2. [ ] [Prevention measure 2] - Owner: [Name] - Due: [Date]

## Lessons Learned
- What went well:
- What didn't go well:
- What we'll do differently:
```

## Related Runbooks
- [HIGH_LLM_COSTS.md](HIGH_LLM_COSTS.md)
- [SERVICE_DOWN.md](SERVICE_DOWN.md)
- [HIGH_LATENCY.md](HIGH_LATENCY.md)

## References
- **Grafana Dashboard:** https://grafana.olumi.com/d/isl-overview
- **Prometheus:** https://prometheus.olumi.com
- **Logs:** `kubectl logs -n production deployment/isl`
- **Metrics:** `http://isl.olumi.com/metrics`
