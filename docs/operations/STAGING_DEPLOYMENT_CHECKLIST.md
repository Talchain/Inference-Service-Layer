# ISL Staging Deployment Checklist

## Pre-Deployment

### **Code Quality**
- [ ] All 119/119 tests passing in CI
- [ ] No critical security vulnerabilities in dependencies
- [ ] Code review approved (if applicable)
- [ ] Performance benchmarks validated (all endpoints meet targets)

### **Configuration**
- [ ] `.env.staging` file prepared with correct values
- [ ] Redis staging instance provisioned and accessible
- [ ] Secrets created in Kubernetes (Redis URL, API keys)
- [ ] TLS certificates valid

### **Documentation**
- [ ] CHANGELOG updated with version changes
- [ ] Breaking changes documented (should be none for v1)
- [ ] Runbook reviewed and updated

### **Stakeholder Communication**
- [ ] Operations team notified 24h in advance
- [ ] PLoT team aware of deployment window
- [ ] UI team aware of any new response fields

---

## Deployment Window

**Recommended:** Tuesday or Wednesday, 10:00-12:00 UTC (avoid Mondays/Fridays)

**Estimated Duration:** 30 minutes

**Rollback Time:** < 5 minutes

---

## Deployment Steps

### **Step 1: Pre-Flight Checks (5 min)**

```bash
# 1. Verify CI passed
# Check GitHub Actions or CI dashboard

# 2. Verify Docker image built
docker pull ghcr.io/talchain/isl:staging-latest

# 3. Verify Kubernetes cluster access
kubectl config current-context
# Expected: staging-cluster

# 4. Verify Redis staging accessible
redis-cli -h staging-redis.olumi.ai ping
# Expected: PONG
```

---

### **Step 2: Deploy to Staging (10 min)**

```bash
# 1. Apply Kubernetes manifests
kubectl apply -f k8s/staging/namespace.yaml
kubectl apply -f k8s/staging/secrets.yaml
kubectl apply -f k8s/staging/configmap.yaml
kubectl apply -f k8s/staging/deployment.yaml
kubectl apply -f k8s/staging/service.yaml
kubectl apply -f k8s/staging/ingress.yaml

# 2. Watch rollout
kubectl rollout status deployment/isl -n isl-staging --timeout=5m

# Expected output:
# deployment "isl" successfully rolled out

# 3. Verify pods running
kubectl get pods -n isl-staging -l app=isl

# Expected:
# NAME                   READY   STATUS    RESTARTS   AGE
# isl-6d4f7c8b9d-abc12   1/1     Running   0          2m
# isl-6d4f7c8b9d-def34   1/1     Running   0          2m
```

---

### **Step 3: Health Checks (5 min)**

```bash
# 1. Basic health check
curl https://isl-staging.olumi.ai/health

# Expected:
# {
#   "status": "healthy",
#   "version": "1.0.0",
#   "redis": {"connected": true},
#   "config_fingerprint": "a1b2c3d4e5f6"
# }

# 2. Metrics endpoint
curl https://isl-staging.olumi.ai/metrics | grep isl_service_up

# Expected:
# isl_service_up 1.0

# 3. Redis connectivity
kubectl exec -n isl-staging deployment/isl -- python -c "
import redis
r = redis.from_url('$REDIS_URL')
print(r.ping())
"

# Expected: True
```

---

### **Step 4: Smoke Tests (10 min)**

Run automated smoke test suite:

```bash
# 1. Run smoke tests
cd scripts/
./smoke_test.sh staging

# Tests include:
# - Causal validation (simple DAG)
# - Counterfactual analysis (basic scenario)
# - Preference elicitation (2-question flow)
# - Error handling (invalid input)
# - Cache behaviour (repeat request)

# Expected: All tests pass

# 2. Manual spot check (optional)
curl -X POST https://isl-staging.olumi.ai/api/v1/causal/validate \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: deploy-test-$(date +%s)" \
  -d '{
    "dag": {
      "nodes": ["A", "B", "C"],
      "edges": [["A", "B"], ["B", "C"]]
    },
    "treatment": "A",
    "outcome": "C"
  }'

# Expected: 200 OK with identifiable=true
```

---

### **Step 5: Monitoring Verification (5 min)**

```bash
# 1. Check Prometheus scraping
curl http://prometheus-staging.olumi.ai/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="isl")'

# Expected: "health": "up"

# 2. Check Grafana dashboard
# Navigate to: https://grafana-staging.olumi.ai/d/isl-overview
# Verify: All panels showing data

# 3. Check alert rules loaded
curl http://prometheus-staging.olumi.ai/api/v1/rules | jq '.data.groups[] | select(.name=="isl")'

# Expected: 10 alert rules present
```

---

## Post-Deployment

### **Immediate (0-15 min after deploy)**

- [ ] Health endpoint returns healthy
- [ ] No error spikes in logs
- [ ] P95 latency within expected range (<2s)
- [ ] No firing alerts
- [ ] Grafana dashboard populating

### **Short-term (15 min - 1 hour)**

- [ ] Smoke tests completed successfully
- [ ] Cache behaviour validated (hit rate >0%)
- [ ] No unexpected error patterns
- [ ] Redis memory stable

### **Medium-term (1-24 hours)**

- [ ] No degraded performance
- [ ] Error rate < 2%
- [ ] Cache hit rate improving (>20% by 24h)
- [ ] No OOM kills or restarts

---

## Rollback Procedure

**Trigger:** Any critical issue not resolved in 15 minutes

### **Quick Rollback (< 5 min)**

```bash
# 1. Rollback deployment
kubectl rollout undo deployment/isl -n isl-staging

# 2. Verify rollback complete
kubectl rollout status deployment/isl -n isl-staging

# 3. Health check
curl https://isl-staging.olumi.ai/health

# 4. Notify stakeholders
# Slack: #isl-incidents
# Message: "Staging deployment rolled back due to [reason]"
```

### **Rollback Triggers**

- **Service Down:** Health endpoint unreachable for >2 minutes
- **High Error Rate:** >10% errors for >5 minutes
- **Performance Degradation:** P95 latency >10s for >5 minutes
- **Critical Alert:** Any CRITICAL alert fires
- **Dependency Failure:** Redis connection lost and not recovering

---

## Validation Sign-Off

**Before proceeding to production, verify:**

- [ ] Staging stable for 48+ hours
- [ ] No critical issues reported
- [ ] Performance within expected ranges
- [ ] Cache behaviour validated
- [ ] PLoT team smoke tested integration
- [ ] Operations team comfortable with monitoring

**Sign-Off:**
- Operations Team: _________________ Date: _______
- Development Team: _________________ Date: _______
- Product Owner: _________________ Date: _______

---

## Common Issues & Solutions

### **Issue: Pods CrashLoopBackOff**

**Symptoms:** Pods restarting repeatedly

**Investigation:**
```bash
kubectl logs -n isl-staging deployment/isl --previous
```

**Common Causes:**
- Invalid Redis URL → Check secrets
- Missing environment variable → Check configmap
- OOM kill → Check resource limits
- Startup timeout → Increase readiness probe initial delay

---

### **Issue: Redis Connection Timeout**

**Symptoms:** Health check shows `redis.connected: false`

**Investigation:**
```bash
# Check Redis service
kubectl get svc -n redis | grep staging

# Check network policy
kubectl get networkpolicy -n isl-staging

# Test Redis directly
redis-cli -h staging-redis.olumi.ai -a $REDIS_PASSWORD ping
```

**Common Causes:**
- Network policy blocking ISL → Redis traffic
- Invalid Redis credentials
- Redis not running
- DNS resolution failure

---

### **Issue: 404 on All Endpoints**

**Symptoms:** Ingress returns 404

**Investigation:**
```bash
# Check service
kubectl get svc -n isl-staging isl

# Check ingress
kubectl get ingress -n isl-staging isl

# Check service selector matches pods
kubectl get pods -n isl-staging -l app=isl --show-labels
```

**Common Causes:**
- Ingress path mismatch
- Service selector doesn't match pod labels
- Service not exposed correctly

---

## Appendix: Environment Variables

**Required:**
```bash
# Redis Configuration
REDIS_URL=redis://:password@staging-redis.olumi.ai:6379/0
REDIS_TLS_ENABLED=true
REDIS_MAX_CONNECTIONS=50

# Service Configuration
ENVIRONMENT=staging
LOG_LEVEL=INFO
PORT=8000

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_URL=https://grafana-staging.olumi.ai

# Feature Flags (if applicable)
FEATURE_ADVANCED_VALIDATION=true
FEATURE_PREFERENCE_LEARNING=true
```

**Optional:**
```bash
# Performance Tuning
MONTE_CARLO_SAMPLES=10000
CACHE_TTL_BELIEFS=86400
CACHE_TTL_IDENT=21600
CACHE_TTL_RESULTS=7200

# Debugging
DEBUG_MODE=false
VERBOSE_LOGGING=false
```

---

**For deployment support, contact: #isl-deployments**
