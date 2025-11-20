# Redis Deployment Guide for ISL

## Overview

Redis serves as the user storage layer for the Inference Service Layer (ISL), providing:

- **User belief persistence**: Store and retrieve user belief models
- **Query history**: Track user interaction history
- **Caching**: Fast access to frequently used data
- **TTL management**: Automatic expiration of stale data

**Graceful Degradation**: If Redis is unavailable, ISL automatically falls back to in-memory storage (session-only).

---

## Quick Start (Docker)

### 1. Deploy Redis

```bash
cd deployment/redis
docker-compose -f docker-compose.redis.yml up -d
```

This starts Redis on `localhost:6379` with:
- Persistent storage (RDB + AOF)
- Health checks
- Automatic restarts
- Resource limits

### 2. Verify Redis is Running

```bash
# Check container status
docker ps | grep isl-redis

# Test Redis connection
docker exec isl-redis redis-cli ping
# Expected output: PONG

# Check Redis info
docker exec isl-redis redis-cli info server
```

### 3. Configure ISL to Use Redis

Set environment variables:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
```

Or in `.env` file:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
# REDIS_PASSWORD=your-secure-password  # If using password
```

### 4. Test ISL Redis Integration

```bash
# Start ISL
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Make request that stores beliefs
curl -X POST http://localhost:8000/api/v1/preferences/elicit \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_001",
    "context": {"domain": "pricing", "variables": ["revenue", "churn"]},
    "num_queries": 1
  }'

# Check Redis for stored data
docker exec isl-redis redis-cli KEYS "user:*"
```

---

## Production Configuration

### Security Hardening

#### 1. Set Redis Password

Edit `deployment/redis/redis.conf`:

```conf
# Uncomment and set strong password
requirepass YourVeryStrongPasswordHere123!
```

Update ISL configuration:
```bash
export REDIS_PASSWORD=YourVeryStrongPasswordHere123!
```

#### 2. Restrict Network Access

**Option A: Firewall Rules**
```bash
# Allow only ISL server to access Redis
sudo ufw allow from <ISL_SERVER_IP> to any port 6379
sudo ufw deny 6379
```

**Option B: Docker Network Isolation**
```yaml
# In docker-compose: Don't expose port publicly
services:
  redis:
    # Remove: ports: - "6379:6379"
    expose:
      - "6379"  # Only accessible within Docker network
```

#### 3. Disable Dangerous Commands

Already configured in `redis.conf`:
```conf
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""
```

#### 4. Enable TLS/SSL (Recommended for Production)

Generate certificates:
```bash
# Generate self-signed cert (for testing)
openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout redis.key -out redis.crt -days 365

# Generate CA for production (use proper CA)
# Follow your organization's PKI procedures
```

Update `redis.conf`:
```conf
port 0  # Disable non-TLS
tls-port 6379
tls-cert-file /etc/redis/certs/redis.crt
tls-key-file /etc/redis/certs/redis.key
tls-ca-cert-file /etc/redis/certs/ca.crt
```

Update docker-compose:
```yaml
volumes:
  - ./certs:/etc/redis/certs:ro
```

### Persistence Configuration

ISL uses both RDB and AOF for maximum durability:

**RDB (Snapshots)**:
- Fast point-in-time backups
- Smaller file size
- Risk of data loss between snapshots

**AOF (Append-Only File)**:
- Log every write operation
- Better durability (fsync every second)
- Larger file size

**Configuration** (already in `redis.conf`):
```conf
# RDB
save 900 1      # Snapshot every 15 min if 1 change
save 300 10     # Snapshot every 5 min if 10 changes
save 60 10000   # Snapshot every 1 min if 10k changes

# AOF
appendonly yes
appendfsync everysec  # Fsync every second
aof-use-rdb-preamble yes  # Hybrid format
```

### Memory Management

**Default Configuration**:
```conf
maxmemory 512mb
maxmemory-policy allkeys-lru
```

**Tuning Recommendations**:

| Scenario | Max Memory | Policy |
|----------|------------|--------|
| **Pilot (< 100 users)** | 512 MB | allkeys-lru |
| **Production (< 1000 users)** | 2 GB | allkeys-lru |
| **Production (< 10000 users)** | 8 GB | allkeys-lru |

Update `redis.conf`:
```conf
maxmemory 2gb  # Adjust based on needs
```

Update docker-compose resource limits:
```yaml
deploy:
  resources:
    limits:
      memory: 2.5G  # 25% overhead for Redis operations
```

**Monitor Memory Usage**:
```bash
# Check current memory usage
docker exec isl-redis redis-cli INFO memory

# Watch memory in real-time
watch -n 1 'docker exec isl-redis redis-cli INFO memory | grep used_memory_human'
```

---

## High Availability (Optional)

For production environments requiring HA:

### Redis Sentinel (Automatic Failover)

**Architecture**: 1 primary + 2 replicas + 3 sentinels

```bash
# See deployment/redis/redis-sentinel/ for HA setup
docker-compose -f docker-compose.redis-ha.yml up -d
```

**Benefits**:
- Automatic failover
- No data loss (with proper replication)
- 99.9% uptime

### Redis Cluster (Horizontal Scaling)

For > 10,000 concurrent users or > 10 GB data:

```bash
# See deployment/redis/redis-cluster/ for cluster setup
docker-compose -f docker-compose.redis-cluster.yml up -d
```

**Benefits**:
- Horizontal scaling
- Sharding across nodes
- Higher throughput

---

## Monitoring

### Key Metrics to Monitor

| Metric | Command | Alert Threshold |
|--------|---------|-----------------|
| **Memory Usage** | `INFO memory` | > 80% of maxmemory |
| **Connections** | `INFO clients` | > 80% of maxclients |
| **Ops/sec** | `INFO stats` | - |
| **Hit Rate** | `INFO stats` | < 80% |
| **Evicted Keys** | `INFO stats` | > 0 consistently |
| **Persistence Status** | `INFO persistence` | last_save_time |

### Real-Time Monitoring

```bash
# Monitor all operations in real-time
docker exec isl-redis redis-cli MONITOR

# Watch specific stats
docker exec isl-redis redis-cli --stat

# Slow query log
docker exec isl-redis redis-cli SLOWLOG GET 10
```

### Prometheus Integration

Redis metrics are exposed via ISL's `/metrics` endpoint:

```promql
# Cache hit rate
isl_cache_hits_total / (isl_cache_hits_total + isl_cache_misses_total)

# Redis operations
rate(isl_redis_operations_total[5m])

# Redis connection status
isl_redis_connected
```

---

## Backup and Recovery

### Manual Backup

```bash
# Trigger immediate RDB snapshot
docker exec isl-redis redis-cli BGSAVE

# Copy RDB file
docker cp isl-redis:/data/dump.rdb ./backup/dump-$(date +%Y%m%d).rdb

# Copy AOF file
docker cp isl-redis:/data/appendonly.aof ./backup/appendonly-$(date +%Y%m%d).aof
```

### Automated Backup Script

```bash
#!/bin/bash
# backup-redis.sh

BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d-%H%M%S)

mkdir -p $BACKUP_DIR

# Trigger snapshot
docker exec isl-redis redis-cli BGSAVE

# Wait for snapshot to complete
sleep 5

# Copy files
docker cp isl-redis:/data/dump.rdb $BACKUP_DIR/dump-$DATE.rdb
docker cp isl-redis:/data/appendonly.aof $BACKUP_DIR/appendonly-$DATE.aof

# Compress
gzip $BACKUP_DIR/dump-$DATE.rdb
gzip $BACKUP_DIR/appendonly-$DATE.aof

# Keep last 30 days
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

Schedule with cron:
```cron
0 2 * * * /path/to/backup-redis.sh
```

### Recovery

```bash
# Stop Redis
docker-compose -f docker-compose.redis.yml down

# Restore backup
gunzip backup/dump-20250120.rdb.gz
docker cp backup/dump-20250120.rdb isl-redis:/data/dump.rdb

# Start Redis
docker-compose -f docker-compose.redis.yml up -d

# Verify data
docker exec isl-redis redis-cli KEYS "user:*"
```

---

## Troubleshooting

### Redis Won't Start

**Check logs**:
```bash
docker logs isl-redis
```

**Common issues**:
1. **Port conflict**: Another service using port 6379
   ```bash
   sudo lsof -i :6379
   ```
2. **Permission issues**: Data directory not writable
   ```bash
   docker exec isl-redis ls -la /data
   ```
3. **Configuration error**: Syntax error in redis.conf
   ```bash
   docker exec isl-redis redis-server --test-config /usr/local/etc/redis/redis.conf
   ```

### ISL Can't Connect to Redis

**Check network connectivity**:
```bash
# From host
telnet localhost 6379

# From ISL container
docker exec isl-api ping redis
```

**Check ISL logs**:
```bash
docker logs isl-api | grep redis
```

**Verify credentials**:
```bash
# If using password
docker exec isl-redis redis-cli -a your-password ping
```

### High Memory Usage

**Check memory distribution**:
```bash
docker exec isl-redis redis-cli --bigkeys
```

**Check TTLs**:
```bash
# Count keys without TTL
docker exec isl-redis redis-cli KEYS "*" | \
  xargs -I {} docker exec isl-redis redis-cli TTL {} | \
  grep -c "\-1"
```

**Flush specific keys** (if needed):
```bash
# Delete old user data
docker exec isl-redis redis-cli --scan --pattern "user:beliefs:*" | \
  xargs docker exec isl-redis redis-cli DEL
```

### Performance Degradation

**Check slow queries**:
```bash
docker exec isl-redis redis-cli SLOWLOG GET 100
```

**Check latency**:
```bash
docker exec isl-redis redis-cli --latency
```

**Check persistence impact**:
```bash
# If RDB snapshots cause lag, adjust thresholds
# Edit redis.conf:
save 900 1
save 300 100  # Reduced from 10
```

---

## Scaling Recommendations

| User Count | Memory | CPU | Persistence | Replication |
|------------|--------|-----|-------------|-------------|
| **< 100** | 512 MB | 1 core | RDB + AOF | Single instance |
| **100-1000** | 2 GB | 2 cores | RDB + AOF | Primary + Replica |
| **1000-10000** | 8 GB | 4 cores | RDB + AOF | Sentinel (1P + 2R) |
| **> 10000** | 16+ GB | 8+ cores | RDB only | Redis Cluster |

---

## ISL Integration Details

### Key Structure

ISL uses the following Redis key patterns:

```
user:beliefs:{user_id}      → UserBeliefModel (JSON)
user:queries:{user_id}      → Sorted set of queries (by timestamp)
user:responses:{user_id}    → List of user responses
```

### TTL Strategy

| Key Type | TTL | Extension |
|----------|-----|-----------|
| `user:beliefs:*` | 24 hours | Extended on activity |
| `user:queries:*` | 7 days | Fixed |
| `user:responses:*` | 30 days | Fixed |

### Fallback Behavior

If Redis is unavailable:
1. ISL logs warning: `redis_connection_failed`
2. Switches to in-memory storage
3. Sets `isl_redis_connected=0` metric
4. Continues serving requests (session-only storage)
5. Reconnects automatically when Redis recovers

---

## Security Checklist

- [ ] Set strong Redis password (`requirepass`)
- [ ] Disable dangerous commands (FLUSHDB, FLUSHALL, CONFIG)
- [ ] Enable protected mode (`protected-mode yes`)
- [ ] Restrict network access (firewall/Docker network)
- [ ] Use TLS/SSL for encryption (production)
- [ ] Configure maxmemory limits
- [ ] Enable persistence (RDB + AOF)
- [ ] Set up regular backups
- [ ] Monitor memory and connections
- [ ] Review Redis logs regularly
- [ ] Keep Redis updated to latest stable version

---

## References

- [Redis Documentation](https://redis.io/documentation)
- [Redis Security](https://redis.io/topics/security)
- [Redis Persistence](https://redis.io/topics/persistence)
- [Redis Sentinel](https://redis.io/topics/sentinel)
- [Redis Best Practices](https://redis.io/topics/best-practices)

---

**Redis Deployment Complete!** 🎉

ISL now has production-ready Redis storage with persistence, security, and monitoring capabilities.
