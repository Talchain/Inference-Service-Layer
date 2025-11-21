# ISL Monitoring Setup Guide

## Overview

ISL uses Prometheus for metrics collection and Grafana for visualization. This guide will help you set up complete observability for production operations.

## Architecture

```
┌─────────────┐     metrics      ┌────────────┐     queries     ┌─────────┐
│  ISL Service│────────────────>│ Prometheus │<──────────────│ Grafana │
│  :8000     │     /metrics     │   :9090    │                 │  :3000  │
└─────────────┘                  └────────────┘                 └─────────┘
                                       │
                                       │ alerts
                                       ▼
                                ┌──────────────┐
                                │ AlertManager │───> Slack/PagerDuty
                                │    :9093     │
                                └──────────────┘
```

## Quick Start

### 1. Deploy Prometheus

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus with ISL scrape config
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace \
  --values monitoring/prometheus-values.yml \
  --version 25.8.0
```

**Create `monitoring/prometheus-values.yml`:**

```yaml
server:
  global:
    scrape_interval: 30s
    scrape_timeout: 10s
    evaluation_interval: 30s

  persistentVolume:
    enabled: true
    size: 50Gi

  retention: "30d"

serverFiles:
  prometheus.yml:
    scrape_configs:
      - job_name: 'isl'
        static_configs:
          - targets: ['isl-service:8000']
        metrics_path: '/metrics'
        scrape_interval: 30s
        scrape_timeout: 10s

      - job_name: 'redis'
        static_configs:
          - targets: ['redis-exporter:9121']

      - job_name: 'postgresql'
        static_configs:
          - targets: ['postgres-exporter:9187']

    rule_files:
      - '/etc/prometheus/rules/*.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

extraConfigmapMounts:
  - name: isl-alert-rules
    mountPath: /etc/prometheus/rules
    configMap: isl-alert-rules
    readOnly: true
```

### 2. Configure ISL Scraping

The ISL service exposes Prometheus metrics at `/metrics`:

```bash
# Verify metrics endpoint
curl http://isl-service:8000/metrics

# Expected output:
# TYPE isl_requests_total counter
# isl_requests_total{endpoint="/api/v1/causal/validate",status="200"} 1234
# ...
```

### 3. Deploy Grafana

```bash
# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --values monitoring/grafana-values.yml \
  --version 7.0.0

# Get admin password
kubectl get secret --namespace monitoring grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode
echo
```

**Create `monitoring/grafana-values.yml`:**

```yaml
persistence:
  enabled: true
  size: 10Gi

datasources:
  datasources.yaml:
    apiVersion: 1
    datasources:
      - name: Prometheus
        type: prometheus
        url: http://prometheus-server
        access: proxy
        isDefault: true
        editable: false

dashboardProviders:
  dashboardproviders.yaml:
    apiVersion: 1
    providers:
      - name: 'isl-dashboards'
        orgId: 1
        folder: 'ISL'
        type: file
        disableDeletion: false
        updateIntervalSeconds: 10
        allowUiUpdates: true
        options:
          path: /var/lib/grafana/dashboards/isl

dashboardsConfigMaps:
  isl-dashboards: "isl-dashboards"

grafana.ini:
  server:
    root_url: https://grafana.olumi.com
  security:
    admin_user: admin
  smtp:
    enabled: true
    host: smtp.gmail.com:587
    user: alerts@olumi.com
    from_address: alerts@olumi.com
```

### 4. Import Dashboards

```bash
# Create ConfigMap with ISL dashboards
kubectl create configmap isl-dashboards \
  --from-file=dashboards/grafana/ \
  --namespace monitoring

# Label for Grafana auto-discovery
kubectl label configmap isl-dashboards \
  grafana_dashboard=1 \
  --namespace monitoring

# Verify dashboards loaded
kubectl logs -n monitoring deployment/grafana \
  | grep "Provisioning dashboards"
```

### 5. Configure Alerts

```bash
# Create ConfigMap for alert rules
kubectl create configmap isl-alert-rules \
  --from-file=monitoring/alerts/isl-alerts.yml \
  --namespace monitoring

# Apply to Prometheus
kubectl rollout restart statefulset/prometheus-server -n monitoring

# Verify rules loaded
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
# Open http://localhost:9090/rules
```

### 6. Setup AlertManager

```bash
# Install AlertManager
helm install alertmanager prometheus-community/alertmanager \
  --namespace monitoring \
  --values monitoring/alertmanager-values.yml
```

**Create `monitoring/alertmanager-values.yml`:**

```yaml
config:
  global:
    slack_api_url: '<SLACK_WEBHOOK_URL>'

  route:
    group_by: ['alertname', 'severity', 'component']
    group_wait: 10s
    group_interval: 10s
    repeat_interval: 12h
    receiver: 'isl-team'

    routes:
      - match:
          severity: critical
          page: "true"
        receiver: 'pagerduty'
        continue: true

      - match:
          severity: critical
        receiver: 'slack-critical'
        continue: true

      - match:
          severity: warning
        receiver: 'slack-warnings'

      - match:
          cost: "true"
        receiver: 'slack-costs'

  receivers:
    - name: 'isl-team'
      email_configs:
        - to: 'isl-team@olumi.com'
          send_resolved: true

    - name: 'pagerduty'
      pagerduty_configs:
        - service_key: '<PAGERDUTY_SERVICE_KEY>'
          description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

    - name: 'slack-critical'
      slack_configs:
        - channel: '#isl-alerts'
          title: ':rotating_light: {{ .GroupLabels.alertname }}'
          text: |-
            {{ range .Alerts }}
            *Summary:* {{ .Annotations.summary }}
            *Impact:* {{ .Annotations.impact }}
            *Runbook:* {{ .Annotations.runbook }}
            {{ end }}
          color: 'danger'
          send_resolved: true

    - name: 'slack-warnings'
      slack_configs:
        - channel: '#isl-alerts'
          title: ':warning: {{ .GroupLabels.alertname }}'
          text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
          color: 'warning'

    - name: 'slack-costs'
      slack_configs:
        - channel: '#isl-costs'
          title: ':moneybag: {{ .GroupLabels.alertname }}'
          text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
          color: '#FFD700'
```

## Dashboards

### ISL Overview
- **URL:** `/d/isl-overview`
- **Purpose:** High-level health and performance
- **Refresh:** 30s
- **Key Metrics:**
  - Request rate per endpoint (req/min)
  - P95 latency (seconds)
  - Error rate (%)
  - LLM costs ($/hour)
  - Cache hit rate (%)
  - Active deliberation sessions
  - FACET robustness distribution
  - ActiVA convergence rate

**Alerts:**
- High P95 Latency (>5s for 5min)
- High LLM Costs (>$10/hour)

### LLM Cost Analysis
- **URL:** `/d/isl-llm-costs`
- **Purpose:** Detailed cost breakdown and optimization
- **Refresh:** 1m
- **Key Metrics:**
  - Cost by model (pie chart, 24h)
  - Cost per session distribution (p50, p95, p99)
  - Token usage rate (by type)
  - Budget exceeded events
  - Fallback to rules rate
  - Cost trend (7 days)
  - Top 10 expensive sessions
  - Cost efficiency ($/deliberation)

**Alerts:**
- Frequent Budget Exceeded (>10 sessions/hour)

### Team Deliberation
- **URL:** `/d/isl-deliberation`
- **Purpose:** Habermas Machine effectiveness
- **Refresh:** 1m
- **Key Metrics:**
  - Deliberation sessions by status
  - Rounds per session distribution
  - Agreement level over time
  - Convergence rate (24h)
  - Average session duration
  - Values extracted per position
  - Concerns identified per position
  - Session activity timeline
  - LLM usage in deliberation
  - Top unresolved disagreements

## Alert Channels

All alerts route through AlertManager to appropriate channels:

| Severity | Channel | Response Time | Escalation |
|----------|---------|---------------|------------|
| Critical + page:true | PagerDuty | Immediate | On-call engineer |
| Critical | Slack #isl-alerts | 15 minutes | Team lead |
| Warning | Slack #isl-alerts | 1 hour | Team review |
| Info | Slack #isl-alerts | Best effort | Weekly review |
| Cost alerts | Slack #isl-costs | 1 hour | Finance + eng |

## Health Checks

ISL exposes health endpoints:

```bash
# Liveness - service is running
curl http://isl:8000/health
# {"status": "healthy", "uptime": 3600}

# Readiness - service can accept requests
curl http://isl:8000/health/ready
# {"status": "ready", "dependencies": {"redis": "ok", "db": "ok"}}

# Metrics - Prometheus metrics
curl http://isl:8000/metrics
# # HELP isl_requests_total Total requests
# # TYPE isl_requests_total counter
# isl_requests_total{endpoint="/api/v1/causal/validate",status="200"} 1234
```

**Configure Kubernetes probes:**

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

## SLOs (Service Level Objectives)

### Availability
- **Target:** 99.5% uptime
- **Measurement:** `(total_requests - 5xx_errors) / total_requests`
- **Alert:** Error rate >5% for 5 minutes
- **Monthly Downtime Budget:** 3.6 hours

### Latency
- **Target:** P95 < 5s for all endpoints
- **Measurement:** `histogram_quantile(0.95, isl_request_duration_seconds_bucket)`
- **Alert:** P95 >5s for 5 minutes
- **Exception:** FACET (P95 <10s acceptable)

### Error Rate
- **Target:** < 1% error rate
- **Measurement:** `5xx_errors / total_requests`
- **Alert:** Error rate >5% for 5 minutes

### Cost Efficiency
- **Target:** < $0.10 per deliberation round
- **Measurement:** `isl_llm_cost_dollars / isl_habermas_deliberations_total`
- **Alert:** Cost >$10/hour for 10 minutes

### Cache Performance
- **Target:** >50% cache hit rate
- **Measurement:** `isl_llm_cache_hits_total / isl_llm_requests_total`
- **Alert:** Cache hit rate <30% for 30 minutes

## Troubleshooting

### No Metrics Appearing

**Symptoms:** Grafana shows "No data" or empty graphs

**Diagnosis:**
```bash
# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
# Open http://localhost:9090/targets
# ISL target should show State: UP

# Check ISL metrics endpoint
kubectl port-forward svc/isl-service 8000:8000
curl http://localhost:8000/metrics
# Should return metrics in Prometheus format

# Check Prometheus logs
kubectl logs -n monitoring statefulset/prometheus-server | tail -50
```

**Solutions:**
1. Verify ISL service is running: `kubectl get pods -l app=isl`
2. Check firewall rules allow Prometheus → ISL:8000
3. Verify service discovery: `kubectl get svc isl-service`
4. Restart Prometheus: `kubectl rollout restart statefulset/prometheus-server -n monitoring`

### High Memory Usage

**Symptoms:** Prometheus pod OOMKilled, slow queries

**Diagnosis:**
```bash
# Check Prometheus memory
kubectl top pod -n monitoring -l app=prometheus

# Check retention settings
kubectl get cm -n monitoring prometheus-server -o yaml | grep retention
```

**Solutions:**
1. Reduce retention period (default: 30d → 15d)
2. Increase memory limit in Helm values
3. Review scrape intervals (30s may be too frequent)
4. Enable remote write to long-term storage

### Dashboard Not Loading

**Symptoms:** Grafana dashboard shows errors or won't load

**Diagnosis:**
```bash
# Check Grafana logs
kubectl logs -n monitoring deployment/grafana | grep ERROR

# Verify ConfigMap
kubectl get cm -n monitoring isl-dashboards

# Check dashboard JSON syntax
kubectl get cm -n monitoring isl-dashboards -o yaml
```

**Solutions:**
1. Verify data source configured: Grafana → Configuration → Data Sources
2. Check dashboard JSON syntax (use jsonlint)
3. Ensure Prometheus is reachable from Grafana
4. Reimport dashboard: delete ConfigMap and recreate

### Alerts Not Firing

**Symptoms:** Alert conditions met but no notifications

**Diagnosis:**
```bash
# Check alert rules loaded
kubectl port-forward -n monitoring svc/prometheus-server 9090:80
# Open http://localhost:9090/rules

# Check AlertManager
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
# Open http://localhost:9093/#/alerts

# Check AlertManager logs
kubectl logs -n monitoring deployment/alertmanager
```

**Solutions:**
1. Verify alert rules syntax: `promtool check rules monitoring/alerts/isl-alerts.yml`
2. Check AlertManager config: route, receivers correct
3. Verify Slack webhook URL or PagerDuty key
4. Test notification manually: `amtool alert add ...`

## Maintenance

### Daily Tasks
- [ ] Review ISL Overview dashboard for anomalies
- [ ] Check LLM Cost Analysis for unexpected spikes
- [ ] Review active alerts in Slack #isl-alerts
- [ ] Verify SLO compliance (error rate, latency)

### Weekly Tasks
- [ ] Review P95 latency trends across all endpoints
- [ ] Analyze LLM cost trends and optimization opportunities
- [ ] Review deliberation convergence rates
- [ ] Check cache hit rates and Redis performance
- [ ] Review and acknowledge all Info-level alerts

### Monthly Tasks
- [ ] Audit alert rules - adjust thresholds based on baselines
- [ ] Review SLO performance and error budgets
- [ ] Analyze long-term cost trends
- [ ] Update runbooks based on incidents
- [ ] Capacity planning review (request volume growth)

## Capacity Planning

Monitor these metrics for growth trends:

| Metric | Current Baseline | Growth Alert | Action |
|--------|------------------|--------------|---------|
| Request rate | 10 req/min | >100 req/min | Scale horizontally |
| LLM cost | $50/day | >$200/day | Review optimization |
| Deliberation sessions | 20/day | >100/day | Prepare for scale |
| Cache size | 1GB | >5GB | Increase Redis memory |
| Database connections | 10 avg | >80 (of 100 max) | Increase pool size |

**Capacity Planning Query Examples:**

```promql
# Request rate growth (week-over-week)
sum(rate(isl_requests_total[7d])) / sum(rate(isl_requests_total[7d] offset 7d))

# Cost trend (30-day projection)
sum(increase(isl_llm_cost_dollars[7d])) * 4.3

# Active sessions trend
avg_over_time(isl_habermas_deliberations_total{status="active"}[7d])
```

## References

- **Prometheus Documentation:** https://prometheus.io/docs/
- **Grafana Documentation:** https://grafana.com/docs/
- **ISL Metrics Reference:** `docs/METRICS_REFERENCE.md`
- **Runbooks:** `docs/runbooks/`
- **Alert Rules:** `monitoring/alerts/isl-alerts.yml`
- **Dashboard Definitions:** `dashboards/grafana/`
