# ISL Redis Strategy & Operations Guide

## Overview

ISL uses Redis as a caching layer for computationally expensive operations. This document defines key patterns, TTL standards, and operational procedures.

---

## Architecture

**Redis Instance:** Dedicated ISL Redis (not shared with Assistants/PLoT)

**Configuration:**
- **Eviction Policy:** `volatile-lru` (evict keys with TTL when memory full)
- **Max Memory:** 2GB (staging), 4GB (production)
- **Persistence:** RDB snapshots every 15min (backup only, not for recovery)
- **TLS:** Required for staging/production

**Key Principle:** ALL keys MUST have TTL. No infinite keys allowed.

---

## Key Patterns & TTL Standards

### **Pattern 1: User Preferences/Beliefs**

**Key Pattern:** `isl:beliefs:{user_id}`

**Purpose:** Store learned user preferences from ActiVA algorithm

**TTL:** 24 hours

**Rationale:** Preferences stable within session but may change day-to-day

**Value Structure:**
```json
{
  "user_id": "user_123",
  "preferences": {
    "risk_tolerance": 0.65,
    "time_horizon": "short",
    "confidence": 0.82
  },
  "last_updated": "2025-11-20T10:30:00Z",
  "question_count": 7
}
```

**Size:** ~500 bytes per user

**Expected Keys:** ~50-100 during pilot (one per active user)

---

### **Pattern 2: Causal Identification Results**

**Key Pattern:** `isl:ident:{dag_hash}`

**Purpose:** Cache Y₀ identification algorithm results

**TTL:** 6 hours

**Rationale:** Graph structure changes infrequently within session

**Value Structure:**
```json
{
  "dag_hash": "abc123...",
  "identifiable": true,
  "adjustment_set": ["A", "B"],
  "backdoor_paths": [],
  "computed_at": "2025-11-20T10:30:00Z"
}
```

**Size:** ~1KB per unique DAG

**Expected Keys:** ~20-30 during pilot (users exploring similar models)

---

### **Pattern 3: Counterfactual Results**

**Key Pattern:** `isl:result:{request_hash}`

**Purpose:** Cache Monte Carlo simulation results

**TTL:** 2 hours

**Rationale:** Scenario-specific, short-lived exploration

**Value Structure:**
```json
{
  "request_hash": "def456...",
  "outcome_range": [45.2, 68.7],
  "confidence_intervals": {...},
  "samples": 10000,
  "computed_at": "2025-11-20T10:30:00Z"
}
```

**Size:** ~5-10KB per scenario (includes sample statistics)

**Expected Keys:** ~100-200 during pilot (many unique scenarios)

---

### **Pattern 4: Sensitivity Analysis**

**Key Pattern:** `isl:sensitivity:{model_hash}`

**Purpose:** Cache sensitivity analysis results

**TTL:** 4 hours

**Rationale:** Model structure moderately stable

**Value Structure:**
```json
{
  "model_hash": "ghi789...",
  "top_drivers": [
    {"variable": "price", "contribution": 0.58},
    {"variable": "elasticity", "contribution": 0.23}
  ],
  "computed_at": "2025-11-20T10:30:00Z"
}
```

**Size:** ~2KB per model

**Expected Keys:** ~30-50 during pilot

---

## Hash Generation

All cache keys use deterministic hashing:

```python
def compute_cache_key(prefix: str, data: Dict) -> str:
    """
    Compute deterministic cache key.

    Ensures: Same data → same key (enables caching)
    """
    # Normalize data
    normalized = json.dumps(data, sort_keys=True, separators=(',', ':'))

    # SHA256 hash (first 16 chars)
    hash_value = hashlib.sha256(normalized.encode()).hexdigest()[:16]

    return f"{prefix}:{hash_value}"
```

---

## TTL Enforcement

**Code-Level Enforcement:**

```python
def set_with_ttl(key: str, value: Any, ttl_seconds: int):
    """Set value with mandatory TTL."""
    assert ttl_seconds > 0, "TTL must be positive"
    redis_client.setex(key, ttl_seconds, json.dumps(value))
```

**Runtime Validation:**

```python
# Periodic check: Any keys without TTL?
keys_without_ttl = []
for key in redis_client.scan_iter("isl:*"):
    ttl = redis_client.ttl(key)
    if ttl == -1:  # No expiry
        keys_without_ttl.append(key)

if keys_without_ttl:
    logger.error(f"Keys without TTL: {keys_without_ttl}")
    # Alert operations team
```

---

## Memory Management

### **Memory Calculation**

**Expected Memory Usage (Pilot):**

- User beliefs: 50 users × 500 bytes = 25KB
- Identification: 30 DAGs × 1KB = 30KB
- Results: 200 scenarios × 7.5KB = 1.5MB
- Sensitivity: 50 models × 2KB = 100KB
- **Total:** ~1.7MB (0.08% of 2GB allocation)

**Growth Projection:**

- 10x users (500): ~17MB
- 100x users (5,000): ~170MB
- Still well under 2GB limit

### **Eviction Behaviour**

When memory reaches `maxmemory`:

1. Redis identifies keys with TTL (all of them, by design)
2. Evicts least-recently-used (LRU) keys first
3. ISL automatically recomputes on cache miss (graceful degradation)

**Eviction Metrics:**

```promql
rate(redis_evicted_keys_total[5m])
```

**Alert Threshold:** >10 evictions/minute sustained (indicates memory pressure)

---

## Operational Procedures

### **Provisioning New Redis Instance**

**Configuration Checklist:**

```bash
# 1. Create Redis instance (DevOps task)
# Provider: AWS ElastiCache / Google Memorystore / Redis Cloud

# 2. Set configuration
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy volatile-lru
redis-cli CONFIG SET save "900 1 300 10 60 10000"  # RDB snapshots

# 3. Verify configuration
redis-cli CONFIG GET maxmemory
redis-cli CONFIG GET maxmemory-policy

# 4. Set connection credentials in ISL secrets
kubectl create secret generic isl-redis \
  --from-literal=url="redis://:password@host:6379/0" \
  --from-literal=tls=true

# 5. Enable Prometheus exporter
# Follow Redis exporter documentation
```

---

### **Routine Maintenance**

**Daily (Automated):**
- Monitor memory usage
- Check eviction rate
- Validate TTL compliance

**Weekly (Manual):**
- Review key distribution:
  ```bash
  redis-cli --scan --pattern "isl:*" | cut -d: -f2 | sort | uniq -c
  ```
- Check for TTL anomalies:
  ```bash
  redis-cli --scan --pattern "isl:*" | xargs -n 1 redis-cli TTL | sort -n | uniq -c
  ```

**Monthly (Manual):**
- Review memory growth trends
- Adjust TTLs if needed (requires code change + deployment)
- Capacity planning (project 3-6 months ahead)

---

### **Troubleshooting Common Issues**

#### **Issue: High Eviction Rate**

**Symptoms:** >10 evictions/minute, cache hit rate dropping

**Investigation:**
```bash
# Check memory pressure
redis-cli INFO memory | grep used_memory

# Check key distribution
redis-cli --scan --pattern "isl:*" | cut -d: -f2 | sort | uniq -c

# Check largest keys
redis-cli --bigkeys
```

**Solutions:**
- If memory <90%: Expected behaviour during cache warmup
- If memory >90%: Consider increasing `maxmemory` (DevOps task)
- If specific key type dominating: Reduce TTL for that type (requires code change)

---

#### **Issue: Connection Timeouts**

**Symptoms:** `REDIS_TIMEOUT` errors in ISL logs

**Investigation:**
```bash
# Check Redis latency
redis-cli --latency-history

# Check connection pool
redis-cli INFO clients | grep connected_clients
```

**Solutions:**
- If latency >100ms: Network/infrastructure issue (contact DevOps)
- If connected_clients near max: Increase connection pool size (config change)
- If intermittent: Expected on shared infrastructure, monitor for sustained issues

---

#### **Issue: Keys Without TTL**

**Symptoms:** Alert fires for keys with TTL=-1

**Investigation:**
```bash
# Find keys without TTL
redis-cli --scan --pattern "isl:*" | xargs -n 1 redis-cli TTL | grep -n "^-1$"
```

**Solutions:**
- This is a BUG, should never happen
- Manually set TTL: `redis-cli EXPIRE key 3600`
- Escalate to developers immediately
- Delete key if unknown: `redis-cli DEL key`

---

## Performance Tuning

### **Connection Pooling**

**Current Configuration:**
```python
REDIS_CONFIG = {
    "max_connections": 50,
    "socket_timeout": 5.0,
    "socket_connect_timeout": 2.0,
    "retry_on_timeout": True,
    "health_check_interval": 30
}
```

**Tuning Guidelines:**
- `max_connections`: 2x expected concurrent requests (pilot: 10-20 → 50 is safe)
- `socket_timeout`: Should be << P95 target (5s gives buffer for 2s target)
- `health_check_interval`: Balance between overhead and fast failure detection

---

### **Cache Hit Rate Optimization**

**Target:** >30% pilot, >50% production

**Strategies:**

1. **Increase TTLs** (if stale data acceptable)
   - Current: beliefs=24h, ident=6h, result=2h, sensitivity=4h
   - Consider: beliefs=48h if user preferences stable

2. **Normalize inputs** (PLoT team responsibility)
   - Ensure deterministic key generation
   - Sort lists, normalize floats, etc.

3. **Pre-warm cache** (optional, post-pilot)
   - Identify common scenarios
   - Pre-compute on deployment

---

## Monitoring & Alerts

### **Key Metrics**

**Memory:**
```promql
redis_memory_used_bytes / redis_memory_max_bytes
```

**Eviction Rate:**
```promql
rate(redis_evicted_keys_total[5m])
```

**Operation Latency:**
```promql
histogram_quantile(0.95, rate(redis_command_duration_seconds_bucket[5m]))
```

**Connection Pool:**
```promql
redis_connected_clients / redis_config_maxclients
```

### **Alert Definitions**

**Memory High:**
- Threshold: >80% for >10 minutes
- Action: Review key distribution, consider scaling

**Eviction Rate High:**
- Threshold: >10/minute for >15 minutes
- Action: Investigate memory pressure

**Connection Pool Exhausted:**
- Threshold: >90% for >5 minutes
- Action: Increase max_connections

**TTL Violation:**
- Threshold: Any key with TTL=-1
- Action: Escalate to developers immediately

---

## Disaster Recovery

### **Backup Strategy**

**RDB Snapshots:** Every 15 minutes (configured above)

**Retention:** 7 days

**Purpose:** Diagnostic only (not for production recovery)

**Rationale:** Redis is a cache, not source of truth. If Redis lost:
1. ISL continues with in-memory fallback
2. Cache repopulates from fresh computations
3. Performance temporarily degraded (higher latency)
4. No data loss (all data recomputable)

---

### **Recovery Scenarios**

**Scenario 1: Redis Instance Fails**

**Impact:** ISL continues operating, performance temporarily degraded

**Action:**
1. ISL automatic fallback to in-memory caching
2. Provision new Redis instance (DevOps)
3. Update ISL config with new Redis URL
4. Restart ISL pods
5. Cache repopulates automatically

**Downtime:** None (graceful degradation)

---

**Scenario 2: Data Corruption**

**Impact:** Incorrect cached results

**Action:**
1. Flush entire cache: `redis-cli FLUSHDB`
2. ISL recomputes all results fresh
3. Investigate corruption cause (escalate to developers)

**Downtime:** None (higher latency during repopulation)

---

**Scenario 3: Network Partition**

**Impact:** ISL can't reach Redis

**Action:**
1. ISL automatic fallback (same as Scenario 1)
2. Fix network issue (DevOps)
3. ISL reconnects automatically

**Downtime:** None (graceful degradation)

---

## Security

### **Access Control**

**Principle:** Only ISL service should access its Redis instance

**Implementation:**
- Network policy: Only ISL pods can reach Redis
- Authentication: Password-protected (stored in Kubernetes secrets)
- TLS: Required for staging/production

---

### **Data Privacy**

**No PII in Redis:**
- User IDs are hashed
- Model content not cached (only computation results)
- All keys use hashes, not readable identifiers

**Logging:**
- Redis commands not logged (potential PII exposure)
- Only cache hit/miss metrics exported

---

## Capacity Planning

### **Current Allocation**

- Staging: 2GB
- Production: 4GB

### **Growth Projections**

**Conservative (10x pilot):**
- Users: 500
- Memory: ~20MB
- Headroom: 99% unused

**Aggressive (100x pilot):**
- Users: 5,000
- Memory: ~200MB
- Headroom: 95% unused

**Recommendation:** Current allocation sufficient for 12+ months

---

## Appendix: Redis Commands Cheatsheet

**Check memory:**
```bash
redis-cli INFO memory
```

**Check key count:**
```bash
redis-cli DBSIZE
```

**Check specific key:**
```bash
redis-cli GET "isl:beliefs:user_123"
redis-cli TTL "isl:beliefs:user_123"
```

**Find keys by pattern:**
```bash
redis-cli --scan --pattern "isl:ident:*"
```

**Check largest keys:**
```bash
redis-cli --bigkeys
```

**Flush cache (DANGEROUS):**
```bash
redis-cli FLUSHDB
```

**Monitor commands in real-time:**
```bash
redis-cli MONITOR
```

---

**For questions or updates to this guide, contact: #isl-operations**
