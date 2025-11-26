# ISL Deployment Guide

How to deploy ISL to staging and production environments.

---

## Deployment Options

| Option | Best For |
|--------|----------|
| Docker Compose | Local development, simple staging |
| Kubernetes | Production, auto-scaling |
| Render | Quick deployment, managed infrastructure |

---

## Quick Deploy (Docker Compose)

### 1. Clone and Configure

```bash
git clone https://github.com/Talchain/Inference-Service-Layer.git
cd Inference-Service-Layer

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

### 2. Start Services

```bash
# Full stack (app + Redis + monitoring)
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

### 3. Access Services

| Service | URL |
|---------|-----|
| ISL API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

---

## Environment Configuration

### Required Variables

```bash
# Authentication (at least one required in production)
ISL_API_KEYS=key1,key2,key3

# CORS (required in production)
CORS_ORIGINS=https://your-app.com,https://another-app.com
```

### Optional Variables

```bash
# Server
ENVIRONMENT=production          # development, staging, production
HOST=0.0.0.0
PORT=8000
WORKERS=4                       # Gunicorn workers

# Logging
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR

# Limits
REQUEST_TIMEOUT_SECONDS=60
MAX_REQUEST_SIZE_MB=10
RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=                 # Optional
REDIS_DB=0

# Performance
DEFAULT_CONFIDENCE_LEVEL=0.95
MAX_MONTE_CARLO_ITERATIONS=10000
```

---

## Docker Deployment

### Build Image

```bash
docker build -t isl:latest .
```

### Run Container

```bash
docker run -d \
  --name isl \
  -p 8000:8000 \
  -e ISL_API_KEYS=your_api_key \
  -e CORS_ORIGINS=https://your-domain.com \
  -e REDIS_HOST=redis \
  isl:latest
```

### Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: '3.8'

services:
  isl-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ISL_API_KEYS=${ISL_API_KEYS}
      - CORS_ORIGINS=${CORS_ORIGINS}
      - REDIS_HOST=redis
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isl-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: isl-api
  template:
    metadata:
      labels:
        app: isl-api
    spec:
      containers:
      - name: isl-api
        image: your-registry/isl:latest
        ports:
        - containerPort: 8000
        env:
        - name: ISL_API_KEYS
          valueFrom:
            secretKeyRef:
              name: isl-secrets
              key: api-keys
        - name: CORS_ORIGINS
          valueFrom:
            configMapKeyRef:
              name: isl-config
              key: cors-origins
        - name: REDIS_HOST
          value: redis-service
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 2000m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: isl-api-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: isl-api
```

### Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: isl-secrets
type: Opaque
stringData:
  api-keys: "key1,key2,key3"
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: isl-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: isl-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Render Deployment

### 1. Connect Repository

1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect your GitHub repository

### 2. Configure Service

```
Name: isl-api
Environment: Docker
Instance Type: Standard (or higher)
```

### 3. Set Environment Variables

In Render dashboard:

```
ISL_API_KEYS=your_secure_key
CORS_ORIGINS=https://your-app.com
REDIS_HOST=your-redis-host
```

### 4. Deploy

Render auto-deploys on push to main branch.

---

## Health Checks

### Endpoints

```bash
# Liveness (is the service running?)
GET /health
→ {"status": "healthy", "version": "2.1.0"}

# Readiness (is the service ready to accept traffic?)
GET /ready
→ {"status": "ready", "checks": {"redis": "ok"}}

# Metrics (Prometheus format)
GET /metrics
```

### Load Balancer Configuration

```yaml
# Example: AWS ALB
health_check:
  path: /health
  interval: 30
  timeout: 5
  healthy_threshold: 2
  unhealthy_threshold: 3
```

---

## Monitoring

### Prometheus

Scrape `/metrics` endpoint:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'isl'
    static_configs:
      - targets: ['isl-api:8000']
    metrics_path: /metrics
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `isl_requests_total` | Total requests by endpoint |
| `isl_request_duration_seconds` | Request latency histogram |
| `isl_errors_total` | Errors by type |
| `isl_cache_hits_total` | Redis cache hits |
| `isl_rate_limit_hits_total` | Rate limit violations |

### Grafana Dashboard

Import the dashboard from `monitoring/grafana/dashboards/`.

---

## Production Checklist

### Before Deployment

- [ ] API keys generated and secured
- [ ] CORS origins configured
- [ ] Redis accessible and configured
- [ ] TLS certificates ready
- [ ] Health check endpoints verified

### After Deployment

- [ ] Health check returns 200
- [ ] API endpoints respond correctly
- [ ] Metrics being scraped
- [ ] Logs flowing to aggregator
- [ ] Alerts configured

### Security

- [ ] No wildcard CORS
- [ ] API keys not in code/logs
- [ ] TLS 1.2+ only
- [ ] Rate limiting active

---

## Rollback

### Docker

```bash
# List previous images
docker images isl

# Run previous version
docker stop isl
docker run -d --name isl isl:previous-tag
```

### Kubernetes

```bash
# Rollback to previous revision
kubectl rollout undo deployment/isl-api

# Rollback to specific revision
kubectl rollout undo deployment/isl-api --to-revision=2
```

### Render

Use Render dashboard to deploy a previous commit.

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs isl

# Common issues:
# - Missing environment variables
# - Redis connection failed
# - Port already in use
```

### High Memory Usage

```bash
# Check memory
docker stats isl

# Solutions:
# - Increase memory limit
# - Reduce WORKERS count
# - Check for memory leaks in logs
```

### Slow Responses

```bash
# Check metrics
curl http://localhost:8000/metrics | grep duration

# Solutions:
# - Scale up replicas
# - Check Redis latency
# - Review slow endpoints in logs
```

---

## Support

- **Issues:** GitHub Issues
- **Logs:** Check container/pod logs
- **Metrics:** Prometheus/Grafana dashboards
