# Redis Troubleshooting Runbook

## Quick Diagnostic Commands

**Check if Redis is running:**
```bash
redis-cli ping
# Expected: PONG
```

**Check memory usage:**
```bash
redis-cli INFO memory | grep used_memory_human
```

**Check key count:**
```bash
redis-cli DBSIZE
```

**Check eviction stats:**
```bash
redis-cli INFO stats | grep evicted_keys
```

---

## Common Issues

### **Issue: Connection Refused**

**Symptoms:**
- ISL health check shows `redis.connected: false`
- Errors: `Connection refused` or `Connection timeout`

**Diagnostic:**
```bash
# Check if Redis is running
redis-cli ping

# Check network connectivity
telnet <redis-host> 6379

# Check Redis logs
kubectl logs -l app=redis
```

**Common Causes:**
1. **Redis not running** → Restart Redis service
2. **Network policy blocking** → Update Kubernetes NetworkPolicy
3. **Wrong host/port** → Verify REDIS_URL environment variable
4. **Authentication failing** → Verify password in secrets

**Solutions:**

**If Redis not running:**
```bash
# Kubernetes
kubectl get pods -l app=redis
kubectl logs -l app=redis

# Restart if needed
kubectl rollout restart deployment/redis
```

**If network policy:**
```bash
# Check network policy
kubectl get networkpolicy -n isl-staging

# Allow ISL → Redis traffic
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-isl-to-redis
spec:
  podSelector:
    matchLabels:
      app: redis
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: isl
EOF
```

**If authentication:**
```bash
# Verify secret
kubectl get secret isl-redis -o jsonpath='{.data.password}' | base64 -d

# Test connection with password
redis-cli -h <redis-host> -a <password> ping
```

---

### **Issue: High Memory Usage**

**Symptoms:**
- Redis memory >80%
- Eviction rate increasing
- Cache hit rate decreasing

**Investigation:**
```bash
# Check current memory
redis-cli INFO memory | grep -E "used_memory|maxmemory"

# Check eviction rate
redis-cli INFO stats | grep evicted_keys

# Check key distribution
redis-cli --scan --pattern "isl:*" | cut -d: -f2 | sort | uniq -c
```

**Common Causes:**

**1. Expected growth (early pilot):**
- **Action:** Monitor, no intervention unless >90%

**2. TTL misconfiguration:**
```bash
# Find keys without TTL
redis-cli --scan --pattern "isl:*" | xargs -n 1 redis-cli TTL | grep "^-1$"

# Count keys without TTL
redis-cli --scan --pattern "isl:*" | while read key; do
  ttl=$(redis-cli TTL "$key")
  if [ "$ttl" == "-1" ]; then
    echo "$key"
  fi
done | wc -l
```

**Solution:**
```bash
# If bug found (keys without TTL), escalate to developers
# Temporary fix: Set TTL manually
redis-cli EXPIRE <key> 3600  # 1 hour

# If many keys, bulk fix:
redis-cli --scan --pattern "isl:*" | while read key; do
  ttl=$(redis-cli TTL "$key")
  if [ "$ttl" == "-1" ]; then
    redis-cli EXPIRE "$key" 3600
    echo "Set TTL on $key"
  fi
done
```

**3. Large cached values:**
```bash
# Check for large keys
redis-cli --bigkeys

# Inspect specific key
redis-cli DEBUG OBJECT isl:result:some_hash
# Look for: serializedlength (size in bytes)
```

**Solution:**
- If individual values >1MB: Investigate model complexity
- May need to reduce TTL for large-result keys

**4. Need more capacity:**
```bash
# If sustained >90% memory usage
# Contact DevOps to increase maxmemory
# Typical scaling: 2GB → 4GB → 8GB
```

---

### **Issue: Slow Performance**

**Symptoms:**
- P95 latency >100ms for Redis operations
- Timeout errors in ISL
- `REDIS_TIMEOUT` errors in logs

**Investigation:**
```bash
# Check latency
redis-cli --latency-history

# Check slow log
redis-cli SLOWLOG GET 10

# Check if using disk swap
redis-cli INFO memory | grep mem_allocator

# Check connected clients
redis-cli INFO clients | grep connected_clients
```

**Common Causes:**

**1. Network latency:**
```bash
# Test network latency
time redis-cli -h <redis-host> ping
# Should be <5ms for same datacenter

# Check for packet loss
ping -c 100 <redis-host>
```

**Solution:** Contact DevOps (infrastructure issue)

**2. Memory fragmentation:**
```bash
# Check fragmentation ratio
redis-cli INFO memory | grep mem_fragmentation_ratio
# Should be close to 1.0
# If >1.5: High fragmentation
```

**Solution:**
```bash
# Option 1: Active defragmentation (Redis 4.0+)
redis-cli CONFIG SET activedefrag yes

# Option 2: Restart Redis (careful!)
# Only if pilot not running or during maintenance window
kubectl rollout restart deployment/redis
```

**3. Slow commands:**
```bash
# Check slow log
redis-cli SLOWLOG GET 10

# Look for: KEYS commands (should use SCAN instead)
# Look for: Large value operations
```

**Solution:**
- KEYS command: Should never happen (ISL uses SCAN)
- If found: Bug, escalate to developers

**4. Connection pool exhausted:**
```bash
# Check current connections
redis-cli INFO clients | grep connected_clients

# Check max clients
redis-cli CONFIG GET maxclients
```

**Solution:**
```bash
# If connected_clients near maxclients
# Increase ISL connection pool (config change):
# REDIS_MAX_CONNECTIONS=100 (from 50)

# Or increase Redis maxclients
redis-cli CONFIG SET maxclients 1000
```

---

### **Issue: Keys Without TTL**

**Symptoms:**
- Alert fires for TTL violations
- Memory growing without bounds
- Keys with `TTL=-1` found

**Investigation:**
```bash
# Find offending keys
redis-cli --scan --pattern "isl:*" | while read key; do
  ttl=$(redis-cli TTL "$key")
  if [ "$ttl" == "-1" ]; then
    echo "$key has no TTL"
  fi
done
```

**This is a BUG - keys without TTL should never exist.**

**Immediate Action:**
```bash
# Set TTL manually (temporary fix)
redis-cli EXPIRE <key> 3600  # 1 hour

# For multiple keys
redis-cli --scan --pattern "isl:*" | while read key; do
  ttl=$(redis-cli TTL "$key")
  if [ "$ttl" == "-1" ]; then
    redis-cli EXPIRE "$key" 3600
  fi
done
```

**Follow-up:**
1. **Escalate to developers immediately** (this is a bug)
2. Check ISL logs for error patterns
3. If key is unknown: Delete it
   ```bash
   redis-cli DEL <key>
   ```

---

### **Issue: Cache Hit Rate Low**

**Symptoms:**
- Cache hit rate <20% sustained
- Higher than expected latency
- Users making many unique requests

**Investigation:**
```bash
# Check key count
redis-cli DBSIZE

# Check key ages (TTLs)
redis-cli --scan --pattern "isl:*" | xargs -n 1 redis-cli TTL | sort -n | uniq -c

# Check if keys being evicted
redis-cli INFO stats | grep evicted_keys
```

**Common Causes:**

**1. Cold cache (early pilot):**
- **Action:** Expected, monitor for improvement over time

**2. Non-deterministic request generation:**
```bash
# Check key distribution
redis-cli --scan --pattern "isl:*" | cut -d: -f3- | sort | uniq -c | sort -rn | head -20

# If many unique keys: PLoT generating non-deterministic requests
```

**Solution:**
- Contact PLoT team to verify request normalization
- Check: Sorting lists, normalizing floats, consistent JSON serialization

**3. TTLs too short:**
```bash
# Check average TTL
redis-cli --scan --pattern "isl:*" | xargs -n 1 redis-cli TTL | awk '{sum+=$1; count++} END {print "Average TTL: " sum/count " seconds"}'
```

**Solution:**
- If average TTL <1 hour: Consider increasing TTLs
- Requires code change + deployment

**4. Memory pressure causing evictions:**
```bash
# Check eviction rate
redis-cli INFO stats | grep evicted_keys
```

**Solution:** See "High Memory Usage" section

---

### **Issue: Redis Crashes or OOM**

**Symptoms:**
- Redis pod restarting
- ISL showing intermittent Redis connection failures
- OOM kill events in logs

**Investigation:**
```bash
# Check pod status
kubectl get pods -l app=redis

# Check for OOM kills
kubectl describe pod <redis-pod> | grep -A 5 "Last State"

# Check resource limits
kubectl describe pod <redis-pod> | grep -A 5 "Limits"

# Check actual memory usage
kubectl top pod <redis-pod>
```

**Common Causes:**

**1. Memory limit too low:**
```bash
# Check configured maxmemory vs pod limit
redis-cli CONFIG GET maxmemory
kubectl describe pod <redis-pod> | grep memory
```

**Solution:**
```bash
# Increase pod memory limit (DevOps task)
# Ensure maxmemory < pod limit (leave ~20% headroom)

# Example: Pod limit 2GB → maxmemory 1.6GB
```

**2. Memory leak (unlikely):**
```bash
# Monitor memory growth over time
watch -n 60 'redis-cli INFO memory | grep used_memory_human'
```

**Solution:** Escalate to developers with data

**3. Too many keys:**
```bash
# Check key count
redis-cli DBSIZE

# Check memory per key
used_memory=$(redis-cli INFO memory | grep used_memory: | cut -d: -f2 | tr -d '\r')
key_count=$(redis-cli DBSIZE)
echo "Average bytes per key: $(($used_memory / $key_count))"
```

**Solution:**
- If avg >100KB per key: Investigate
- May need to reduce TTLs or split cache

---

### **Issue: Connection Timeouts**

**Symptoms:**
- `REDIS_TIMEOUT` errors in ISL logs
- Intermittent connection failures
- Requests timing out

**Investigation:**
```bash
# Check Redis latency
redis-cli --latency-history

# Check connection pool
redis-cli INFO clients | grep connected_clients

# Check for long-running commands
redis-cli CLIENT LIST | grep age | awk '{if ($10 > 10) print $0}'
```

**Common Causes:**

**1. Network blips (intermittent):**
```bash
# Check error rate
# If <1% of requests: Expected on shared infrastructure
# If >1%: Infrastructure issue
```

**Solution:** Monitor, escalate if sustained >1%

**2. Slow commands blocking:**
```bash
# Check for slow commands
redis-cli SLOWLOG GET 10
```

**Solution:**
- KEYS command: Bug (should use SCAN)
- Large values: May need to reduce cache value size

**3. Connection pool exhausted:**
```bash
# Check ISL connection pool size
# Default: 50 connections

# Check actual connections
redis-cli INFO clients | grep connected_clients
```

**Solution:**
```bash
# Increase connection pool (ISL config)
# REDIS_MAX_CONNECTIONS=100
```

**4. Redis overloaded:**
```bash
# Check CPU usage
kubectl top pod <redis-pod>

# Check operations per second
redis-cli INFO stats | grep instantaneous_ops_per_sec
```

**Solution:**
- If ops >50k/sec: Consider scaling
- Contact DevOps

---

## Maintenance Procedures

### **Flush Cache (DANGEROUS)**

**When:** Only if data corruption suspected

**Impact:**
- All cached data lost
- ISL will recompute everything
- Temporary performance degradation (2-4x latency)
- No permanent data loss (all recomputable)

**Procedure:**
```bash
# 1. Confirm with team first!
# Slack: #isl-operations

# 2. Notify stakeholders
# Message: "Flushing ISL Redis cache due to [reason]"

# 3. Flush database
redis-cli FLUSHDB

# 4. Verify empty
redis-cli DBSIZE
# Expected: 0

# 5. Monitor ISL performance
# Watch Grafana for 15 minutes
# Latency will be higher (cache warming)

# 6. Confirm cache repopulating
redis-cli DBSIZE
# Should grow gradually
```

---

### **Check for Memory Leaks**

**Procedure:**
```bash
# Monitor memory growth over 1 hour
watch -n 60 'echo "$(date): $(redis-cli INFO memory | grep used_memory_human)"'

# Expected: Stable or slowly growing
# Concern: Rapid growth without corresponding request increase

# Also monitor key count
watch -n 60 'echo "$(date): $(redis-cli DBSIZE) keys"'

# If memory leak suspected:
# 1. Check key count growth
# 2. Check key distribution
redis-cli --scan --pattern "isl:*" | cut -d: -f2 | sort | uniq -c

# 3. Escalate to developers with data
```

---

### **Backup Redis Data**

**Note:** Redis is a cache, not source of truth. Backup is for diagnostics only.

**Procedure:**
```bash
# Trigger RDB snapshot
redis-cli BGSAVE

# Check when last save completed
redis-cli LASTSAVE

# Find snapshot file
# Usually: /data/dump.rdb (inside Redis container)

# Copy snapshot out (if needed for diagnosis)
kubectl cp <redis-pod>:/data/dump.rdb ./redis-backup-$(date +%Y%m%d).rdb
```

---

### **Restart Redis (Careful!)**

**When:** Only during maintenance window or if Redis unresponsive

**Impact:**
- Brief downtime (~10-30 seconds)
- All cache data lost
- ISL continues with in-memory fallback
- Cache repopulates automatically

**Procedure:**
```bash
# 1. Notify stakeholders
# Slack: #isl-operations
# Message: "Restarting Redis for maintenance"

# 2. Check if pilot is running
# If active users: Wait for maintenance window

# 3. Restart Redis
kubectl rollout restart deployment/redis

# 4. Wait for ready
kubectl rollout status deployment/redis

# 5. Verify health
redis-cli ping
# Expected: PONG

# 6. Monitor ISL
# Watch Grafana for 15 minutes
# Latency will be higher (cache warming)

# 7. Confirm no errors
kubectl logs -l app=isl --tail=100 | grep ERROR
```

---

## Emergency Procedures

### **Redis Completely Unavailable**

**Immediate Actions:**
1. ISL automatically falls back to in-memory caching (no action needed)
2. Monitor ISL metrics for performance impact
3. Check Redis logs: `kubectl logs -l app=redis --tail=100`
4. Contact DevOps to restore Redis

**Impact:**
- No downtime (ISL continues)
- Reduced cache efficiency (higher latency)
- No data loss

**Recovery:**
```bash
# Once Redis restored:
1. ISL reconnects automatically
2. Cache repopulates from requests
3. Performance returns to normal within 1 hour
```

---

### **Data Corruption Suspected**

**Symptoms:**
- Incorrect cached results
- Users reporting inconsistent behaviour
- ISL logs showing unexpected values

**Immediate Actions:**
1. **Flush cache:** `redis-cli FLUSHDB`
2. **Notify team:** Slack #isl-incidents
3. **ISL will recompute fresh results**

**Investigation:**
```bash
# Before flushing (if possible):
1. Save Redis snapshot for analysis
   kubectl cp <redis-pod>:/data/dump.rdb ./redis-corrupt-$(date +%Y%m%d).rdb

2. Export problematic keys
   redis-cli --scan --pattern "isl:*" > redis-keys-$(date +%Y%m%d).txt

3. Escalate to developers with:
   - Snapshot file
   - Key list
   - ISL logs showing incorrect behaviour
   - User report details
```

---

## Performance Tuning

### **Optimize Connection Pool**

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
```bash
# Monitor connection usage
redis-cli INFO clients | grep connected_clients

# If consistently >80% of max:
# Increase max_connections (ISL config)
# REDIS_MAX_CONNECTIONS=100

# If timeout errors frequent:
# Increase socket_timeout
# But investigate root cause first
```

---

### **Optimize TTLs**

**Current TTLs:**
- Beliefs: 24 hours
- Identification: 6 hours
- Results: 2 hours
- Sensitivity: 4 hours

**Tuning:**
```bash
# Check cache hit rate by key type
redis-cli --scan --pattern "isl:beliefs:*" | wc -l
redis-cli --scan --pattern "isl:ident:*" | wc -l
redis-cli --scan --pattern "isl:result:*" | wc -l

# If hit rate low for specific type:
# Consider increasing TTL (requires code change)
```

---

## Monitoring Checklist

**Daily:**
- [ ] Memory usage < 80%
- [ ] Eviction rate < 10/minute
- [ ] Connected clients < maxclients * 0.8
- [ ] P95 latency < 10ms

**Weekly:**
- [ ] Review key distribution
- [ ] Check for TTL violations
- [ ] Review slow log
- [ ] Capacity planning

**Monthly:**
- [ ] Memory growth trend
- [ ] Cache hit rate trend
- [ ] Connection pool utilization
- [ ] Incident review

---

**For Redis emergencies, contact: #isl-redis or page on-call**
