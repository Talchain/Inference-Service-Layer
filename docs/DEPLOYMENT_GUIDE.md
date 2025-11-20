# ISL Deployment Guide - Production Pilot

## Overview

This guide covers deploying the Inference Service Layer (ISL) from development to production for the Phase 1D pilot.

**Deployment Tiers**:
1. **Local Development**: Single developer testing
2. **Staging/QA**: Team testing and CEE integration
3. **Production Pilot**: Limited user production deployment

---

## Table of Contents

- [Quick Start (Local)](#quick-start-local)
- [System Requirements](#system-requirements)
- [Deployment Architecture](#deployment-architecture)
- [Step-by-Step Deployment](#step-by-step-deployment)
- [Production Configuration](#production-configuration)
- [Monitoring](#monitoring)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Pilot Checklist](#pilot-checklist)

---

## Quick Start (Local)

### 1. Prerequisites

```bash
# Install Python 3.11+
python --version  # Should be 3.11+

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install Docker
docker --version  # Should be 20.10+
```

### 2. Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd Inference-Service-Layer

# Install dependencies
poetry install

# Create .env file
cp .env.example .env
```

### 3. Start Services

```bash
# Terminal 1: Start Redis
cd deployment/redis
docker-compose -f docker-compose.redis.yml up -d

# Terminal 2: Start ISL
cd ../..
poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Start Monitoring (optional)
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### 4. Verify

```bash
# Check health
curl http://localhost:8000/health

# Check API docs
open http://localhost:8000/docs

# Check metrics (if monitoring enabled)
curl http://localhost:8000/metrics

# Check Grafana (if monitoring enabled)
open http://localhost:3000  # admin/admin
```

---

## System Requirements

### Minimum (Development/Staging)

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04+), macOS, Windows WSL2 |
| **CPU** | 2 cores |
| **RAM** | 4 GB |
| **Storage** | 20 GB SSD |
| **Docker** | 20.10+ |
| **Python** | 3.11+ |

### Recommended (Production Pilot)

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 22.04 LTS) |
| **CPU** | 4 cores (x86_64) |
| **RAM** | 8 GB |
| **Storage** | 50 GB SSD |
| **Docker** | 24.0+ |
| **Python** | 3.11+ |
| **Redis** | 7.2+ (2 GB RAM) |

### Production (Future Scaling)

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 22.04 LTS) |
| **CPU** | 8+ cores |
| **RAM** | 16+ GB |
| **Storage** | 100+ GB SSD |
| **Load Balancer** | nginx/HAProxy |
| **Redis** | 7.2+ (8 GB RAM, HA setup) |

---

## Deployment Architecture

### Development

```
┌──────────────┐
│  Developer   │
│   (Laptop)   │
├──────────────┤
│     ISL      │ ← Poetry/Uvicorn
│    Redis     │ ← Docker
│  Prometheus  │ ← Docker
│   Grafana    │ ← Docker
└──────────────┘
```

### Production Pilot

```
                    ┌─────────────┐
                    │     CEE     │
                    │  (Frontend) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Load Balance │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐       ┌────▼────┐      ┌────▼────┐
    │  ISL-1  │       │  ISL-2  │      │  ISL-N  │
    └────┬────┘       └────┬────┘      └────┬────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                    ┌──────▼──────┐
                    │    Redis    │
                    │   (Primary  │
                    │  + Replica) │
                    └─────────────┘

    ┌──────────────────────────────────┐
    │        Monitoring Stack          │
    ├──────────────┬───────────────────┤
    │  Prometheus  │     Grafana       │
    └──────────────┴───────────────────┘
```

---

## Step-by-Step Deployment

### Phase 1: Pre-Deployment Setup

#### 1. Server Provisioning

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essentials
sudo apt install -y \
    build-essential \
    curl \
    git \
    python3.11 \
    python3.11-venv \
    python3-pip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### 2. Create Application User

```bash
# Create isl user
sudo useradd -m -s /bin/bash isl
sudo usermod -aG docker isl

# Switch to isl user
sudo su - isl
```

#### 3. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Phase 2: Application Deployment

#### 1. Clone and Configure

```bash
# Clone repository
cd ~
git clone <repository-url>
cd Inference-Service-Layer

# Checkout appropriate branch
git checkout main  # or specific release tag

# Install dependencies
poetry install --no-dev

# Create production .env
cp .env.example .env
```

#### 2. Configure Environment Variables

Edit `.env`:

```env
# Application
ENV=production
LOG_LEVEL=INFO
API_V1_PREFIX=/api/v1

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4  # 2x CPU cores

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your-secure-password-here

# Security (generate with: openssl rand -hex 32)
SECRET_KEY=your-secret-key-here

# Feature Flags
ENABLE_MONITORING=true
ENABLE_CACHE=true
```

#### 3. Deploy Redis

```bash
cd deployment/redis

# Edit redis.conf - set password
sed -i 's/# requirepass/requirepass/' redis.conf
sed -i 's/your-secure-password-here/YOUR_ACTUAL_PASSWORD/' redis.conf

# Start Redis
docker-compose -f docker-compose.redis.yml up -d

# Verify Redis is running
docker ps | grep isl-redis
docker exec isl-redis redis-cli -a YOUR_ACTUAL_PASSWORD ping
```

#### 4. Deploy ISL Application

**Option A: Using systemd (Recommended)**

Create `/etc/systemd/system/isl.service`:

```ini
[Unit]
Description=Inference Service Layer
After=network.target docker.service

[Service]
Type=simple
User=isl
WorkingDirectory=/home/isl/Inference-Service-Layer
Environment="PATH=/home/isl/.local/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/isl/Inference-Service-Layer/.env
ExecStart=/home/isl/.local/bin/poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable isl
sudo systemctl start isl
sudo systemctl status isl

# View logs
sudo journalctl -u isl -f
```

**Option B: Using Docker** (Alternative)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-root

# Copy application
COPY src ./src

# Expose port
EXPOSE 8000

# Run application
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:

```bash
docker build -t isl:latest .
docker run -d --name isl \
    -p 8000:8000 \
    --env-file .env \
    --restart unless-stopped \
    isl:latest
```

### Phase 3: Monitoring Deployment

#### 1. Deploy Prometheus + Grafana

```bash
cd monitoring

# Edit prometheus.yml - update target
sed -i 's/host.docker.internal:8000/YOUR_ISL_HOST:8000/' prometheus/prometheus.yml

# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Verify
docker ps | grep -E "(prometheus|grafana)"
```

#### 2. Configure Grafana

1. Open `http://YOUR_SERVER:3000`
2. Login: admin/admin (change password immediately)
3. Dashboard should auto-load via provisioning
4. Verify Prometheus datasource is connected

#### 3. Test Metrics Collection

```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
open http://localhost:9090/targets

# Verify target is UP
```

### Phase 4: Verification

#### 1. Health Checks

```bash
# API health
curl http://localhost:8000/health

# Expected: {"status": "healthy", "version": "..."}

# Detailed health (if implemented)
curl http://localhost:8000/api/v1/health/detailed
```

#### 2. Functional Tests

```bash
# Run integration tests against deployed instance
poetry run pytest tests/integration/ --host http://localhost:8000

# Or run CEE integration tests
poetry run pytest tests/integration/test_cee_integration.py --host http://localhost:8000
```

#### 3. Performance Benchmarks

```bash
# Run performance benchmark
poetry run python benchmarks/performance_benchmark.py \
    --host http://localhost:8000 \
    --duration 60 \
    --concurrency 10

# Verify all targets met
# - P95 latency < 2.0s (causal/counterfactual)
# - P95 latency < 1.5s (preference/teaching)
```

### Phase 5: Load Balancer Setup (Optional for HA)

#### nginx Configuration

Create `/etc/nginx/sites-available/isl`:

```nginx
upstream isl_backend {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    # Add more servers for HA:
    # server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    # server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name isl.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name isl.example.com;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/isl.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/isl.example.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Proxy configuration
    location / {
        proxy_pass http://isl_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Metrics endpoint (restrict access)
    location /metrics {
        allow 127.0.0.1;
        deny all;
        proxy_pass http://isl_backend;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/isl /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Production Configuration

### Environment Variables

**Required**:
- `ENV=production`
- `REDIS_PASSWORD`: Strong password
- `SECRET_KEY`: Cryptographically secure key

**Optional but Recommended**:
- `LOG_LEVEL=INFO` (DEBUG for troubleshooting only)
- `WORKERS`: 2x CPU cores
- `RELOAD=false` (disable auto-reload in production)

### Redis Configuration

See [Redis Deployment Guide](./REDIS_DEPLOYMENT.md) for:
- Password protection
- Persistence configuration
- Memory limits
- Backup strategy

### Security Hardening

1. **Firewall Rules**:
   ```bash
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw allow 22/tcp
   sudo ufw deny 8000/tcp  # Block direct ISL access
   sudo ufw deny 6379/tcp  # Block direct Redis access
   sudo ufw enable
   ```

2. **SSL/TLS Certificates**:
   ```bash
   # Install certbot
   sudo apt install certbot python3-certbot-nginx

   # Obtain certificate
   sudo certbot --nginx -d isl.example.com
   ```

3. **Security Updates**:
   ```bash
   # Enable automatic security updates
   sudo apt install unattended-upgrades
   sudo dpkg-reconfigure --priority=low unattended-upgrades
   ```

---

## Monitoring

### Accessing Monitoring Tools

- **Prometheus**: http://YOUR_SERVER:9090
- **Grafana**: http://YOUR_SERVER:3000
- **Metrics Endpoint**: http://YOUR_SERVER:8000/metrics

### Key Metrics to Monitor

| Metric | Alert Threshold | Action |
|--------|----------------|--------|
| **Service Up** | == 0 | Immediate restart |
| **P95 Latency** | > 2.0s | Investigate performance |
| **Error Rate** | > 5% | Check logs |
| **Redis Connected** | == 0 | Restart Redis |
| **Active Requests** | > 100 | Scale up |
| **Memory Usage** | > 80% | Add resources |

### Alerting

See [Monitoring Setup Guide](./MONITORING_SETUP.md) for:
- Alert rules configuration
- Notification channels (email, Slack, PagerDuty)
- Escalation policies

---

## Security

### Authentication (Future)

Phase 1D pilot uses network-level security. Future phases will add:
- API key authentication
- OAuth2/JWT tokens
- Rate limiting

### Data Privacy

- User IDs are hashed in logs
- Beliefs stored in Redis with TTL
- No PII logged to Prometheus
- Metrics aggregated, not per-user

### Compliance

- GDPR: User data TTL (24h beliefs, 7d queries)
- SOC 2: Audit logs, access controls
- HIPAA: N/A for pilot (no health data)

---

## Troubleshooting

### ISL Won't Start

```bash
# Check systemd status
sudo systemctl status isl

# Check logs
sudo journalctl -u isl -n 100

# Common issues:
# 1. Redis not running
docker ps | grep redis

# 2. Port already in use
sudo lsof -i :8000

# 3. Permission issues
ls -la /home/isl/Inference-Service-Layer
```

### High Latency

```bash
# Check system resources
top
df -h

# Check Redis memory
docker exec isl-redis redis-cli INFO memory

# Check slow queries
docker exec isl-redis redis-cli SLOWLOG GET 10

# Run performance benchmark
poetry run python benchmarks/performance_benchmark.py
```

### Memory Leaks

```bash
# Monitor memory over time
watch -n 5 'docker stats isl --no-stream'

# Check for memory growth
ps aux | grep uvicorn

# Restart service if needed
sudo systemctl restart isl
```

---

## Pilot Checklist

See [PILOT_READINESS.md](./PILOT_READINESS.md) for complete checklist.

**Quick Check**:
- [ ] ISL deployed and running
- [ ] Redis deployed with persistence
- [ ] Monitoring stack operational
- [ ] Health checks passing
- [ ] Integration tests passing
- [ ] Performance benchmarks passing
- [ ] Security hardening complete
- [ ] Backup strategy implemented
- [ ] Documentation complete
- [ ] CEE integration validated

---

## Rollback Procedure

If issues occur in production:

```bash
# 1. Stop current version
sudo systemctl stop isl

# 2. Checkout previous stable version
cd /home/isl/Inference-Service-Layer
git fetch --all
git checkout <previous-stable-tag>

# 3. Install dependencies (if changed)
poetry install --no-dev

# 4. Restart
sudo systemctl start isl
sudo systemctl status isl

# 5. Verify
curl http://localhost:8000/health

# 6. Check logs
sudo journalctl -u isl -f
```

---

## Scaling Guide

### Vertical Scaling (Single Instance)

Increase resources:
- CPU: 4 → 8 cores
- RAM: 8 → 16 GB
- Redis: 2 → 8 GB

Update `.env`:
```env
WORKERS=16  # 2x new CPU count
```

### Horizontal Scaling (Multiple Instances)

1. Deploy multiple ISL instances on different ports
2. Configure load balancer (nginx)
3. Use Redis sentinel for HA
4. Share Redis across instances

---

## Support

For deployment issues:
1. Check logs: `sudo journalctl -u isl -n 1000`
2. Review monitoring: http://YOUR_SERVER:3000
3. Run diagnostics: `poetry run python scripts/diagnose.py`
4. Contact: <support@example.com>

---

**Deployment Complete!** 🎉

Your ISL instance is now running in production, monitored, and ready for the pilot phase.
