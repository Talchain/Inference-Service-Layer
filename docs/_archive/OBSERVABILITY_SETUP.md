# ISL Observability Suite - Quick Start Guide

Complete monitoring and performance testing setup for the Inference Service Layer.

## Overview

This observability suite provides:
- **Performance Profiling** - CPU, memory, latency analysis
- **Load Testing** - Locust-based load generation
- **Monitoring** - Prometheus metrics collection
- **Visualization** - Grafana dashboards
- **Alerting** - Prometheus alert rules

## Quick Start

### 1. Start Monitoring Stack

```bash
# From ISL root directory
docker-compose -f docker-compose.monitoring.yml up -d
```

This starts:
- **Prometheus** (http://localhost:9090) - Metrics collection
- **Grafana** (http://localhost:3000) - Visualization
- **Alertmanager** (http://localhost:9093) - Alert handling
- **Node Exporter** (http://localhost:9100) - System metrics

### 2. Access Grafana Dashboard

```bash
# Open browser
open http://localhost:3000

# Login
Username: admin
Password: admin

# Navigate to:
Dashboards → ISL Performance & Observability
```

### 3. Run Performance Profiling

```bash
# Ensure ISL is running
python -m uvicorn src.api.main:app --reload

# Profile all endpoints
python scripts/performance/profile_endpoints.py

# View results
cat scripts/performance/performance_report.json
```

### 4. Run Load Tests

**Option A: Web UI** (recommended for interactive testing)
```bash
# Start Locust
locust -f tests/load/locustfile.py --host http://localhost:8000

# Open browser
open http://localhost:8089

# Configure test:
# - Number of users: 50
# - Spawn rate: 5
# - Run time: 10m (or leave blank for manual stop)

# Click "Start Swarming"
```

**Option B: Headless** (for automated testing)
```bash
# Ramp test
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 15m --headless

# Sustained load
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 30m --headless

# Spike test
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --users 200 --spawn-rate 50 --run-time 5m --headless
```

## Architecture

### Metrics Flow

```
ISL API (/metrics endpoint)
    ↓
Prometheus (scrape every 15s)
    ↓
Grafana (visualize)
    ↓
Alertmanager (on threshold breach)
```

### Dashboard Panels

**Request Overview**:
- Total RPS
- P50/P95/P99 latency
- Error rate

**Endpoint Health**:
- Latency heatmap
- Success rate by endpoint
- Top 5 slowest endpoints

**Error Recovery**:
- Circuit breaker states
- Fallback trigger rates
- Service health status

**Cache Performance**:
- Hit rate
- Cache size
- Eviction rate

**Resource Usage**:
- Memory consumption
- Active requests
- CPU usage (via Node Exporter)

## Alert Rules

### Performance Alerts

**HighLatency** (Warning)
- Trigger: P95 >3s for 5min
- Action: Investigate slow endpoints

**CriticalLatency** (Critical)
- Trigger: P95 >5s for 2min
- Action: Immediate investigation

### Error Alerts

**HighErrorRate** (Warning)
- Trigger: Error rate >5% for 5min
- Action: Check error logs

**CriticalErrorRate** (Critical)
- Trigger: Error rate >10% for 2min
- Action: Immediate response

### Circuit Breaker Alerts

**CircuitBreakerOpen** (Warning)
- Trigger: Circuit state=OPEN for 2min
- Action: Check service health

**CircuitBreakerStuckOpen** (Critical)
- Trigger: Circuit state=OPEN for 10min
- Action: Manual intervention

### Service Health Alerts

**ServiceDegraded** (Warning)
- Trigger: Health status=DEGRADED for 5min
- Action: Monitor closely

**ServiceFailing** (Critical)
- Trigger: Health status=FAILING for 2min
- Action: Immediate response

## Configuration

### Prometheus

Edit `prometheus/prometheus.yml` to:
- Change scrape interval (default: 15s)
- Add additional targets
- Configure retention period

```yaml
global:
  scrape_interval: 15s  # Change here
```

### Grafana

Dashboard location: `grafana/dashboards/isl-performance.json`

To modify:
1. Edit dashboard in Grafana UI
2. Export JSON
3. Replace `grafana/dashboards/isl-performance.json`
4. Restart Grafana

### Alerts

Edit `prometheus/alerts.yml` to:
- Adjust thresholds
- Add new rules
- Change severity levels

Example:
```yaml
- alert: HighLatency
  expr: histogram_quantile(0.95, ...) > 3  # Change threshold
  for: 5m  # Change duration
```

### Alertmanager

Edit `prometheus/alertmanager.yml` to:
- Configure notification channels
- Add Slack/PagerDuty/Email
- Set routing rules

Example Slack integration:
```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK'
        channel: '#alerts'
```

## Maintenance

### Start/Stop Monitoring

```bash
# Start
docker-compose -f docker-compose.monitoring.yml up -d

# Stop
docker-compose -f docker-compose.monitoring.yml down

# Restart
docker-compose -f docker-compose.monitoring.yml restart

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f

# Remove volumes (reset data)
docker-compose -f docker-compose.monitoring.yml down -v
```

### Backup

```bash
# Backup Grafana dashboards
cp -r grafana/dashboards /backup/

# Backup Prometheus data
docker run --rm -v isl_prometheus-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/prometheus-$(date +%Y%m%d).tar.gz /data
```

### Restore

```bash
# Restore dashboards
cp /backup/dashboards/* grafana/dashboards/

# Restart Grafana
docker-compose -f docker-compose.monitoring.yml restart grafana
```

## Troubleshooting

### Prometheus Not Scraping

```bash
# Check Prometheus targets
open http://localhost:9090/targets

# If ISL target is "DOWN":
# 1. Verify ISL is running
# 2. Check ISL exposes /metrics
curl http://localhost:8000/metrics

# 3. Check Docker network connectivity
docker network inspect monitoring
```

### Grafana Dashboard Empty

```bash
# Check datasource
# Grafana → Configuration → Data Sources → Prometheus

# Test connection
# Should show "Data source is working"

# If not working:
# 1. Check Prometheus URL (http://prometheus:9090)
# 2. Check Docker network
# 3. Restart Grafana
docker-compose -f docker-compose.monitoring.yml restart grafana
```

### Load Test Failing

```bash
# Check ISL is running
curl http://localhost:8000/health

# Check Locust syntax
python -m py_compile tests/load/locustfile.py

# Run with verbose logging
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --loglevel DEBUG
```

## Performance Targets

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| P50 Latency | <500ms | >500ms | >1000ms |
| P95 Latency | <1000ms | >1500ms | >3000ms |
| RPS | >40 | <30 | <20 |
| Error Rate | <1% | >2% | >5% |
| Cache Hit Rate | >70% | <50% | <30% |

## Next Steps

1. **Baseline Performance** - Run initial profiling to establish baseline
2. **Load Test** - Identify current capacity limits
3. **Set Alerts** - Configure notification channels in Alertmanager
4. **Iterate** - Optimize based on findings

## Resources

- **Performance Runbook**: `docs/operations/PERFORMANCE_RUNBOOK.md`
- **Error Recovery Docs**: `docs/ERROR_RECOVERY.md`
- **Prometheus Docs**: https://prometheus.io/docs/
- **Grafana Docs**: https://grafana.com/docs/
- **Locust Docs**: https://docs.locust.io/

---

**Questions?** See `docs/operations/PERFORMANCE_RUNBOOK.md` for detailed troubleshooting.
