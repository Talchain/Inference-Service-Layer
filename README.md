# Inference Service Layer (ISL)

**Causal inference API for Olumi's decision platform**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-215+-brightgreen.svg)](#testing)

---

## What is ISL?

ISL provides causal inference capabilities via REST API:

- **Causal Validation** - Validate DAGs and identify adjustment sets
- **Counterfactual Analysis** - "What-if" predictions with uncertainty
- **Sensitivity Analysis** - Test assumption robustness
- **Progressive Explanations** - Multi-level causal explanations
- **Causal Discovery** - Extract factors from text

**Consumers:** PLoT Engine, TAE, CEE, UI

---

## Quick Start

### 1. Run with Docker

```bash
git clone https://github.com/Talchain/Inference-Service-Layer.git
cd Inference-Service-Layer
docker-compose up -d
curl http://localhost:8000/health
```

### 2. Local Development

```bash
# Install dependencies
pip install -e .

# Set environment
cp .env.example .env

# Run server
uvicorn src.api.main:app --reload
```

### 3. Test an Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/validation/assumptions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "dag": {"nodes": ["A", "B", "C"], "edges": [["A", "C"], ["B", "C"]]},
    "treatment": "A",
    "outcome": "C"
  }'
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[Getting Started](docs/GETTING_STARTED.md)** | New developer onboarding |
| **[Technical Specification](docs/TECHNICAL_SPECIFICATION.md)** | Complete architecture & API reference |
| **[API Examples](docs/api/EXAMPLES.md)** | Code examples for all endpoints |
| **[Integration Guide](docs/integration/GUIDE.md)** | PLoT/TAE/CEE integration |
| **[Deployment Guide](docs/operations/DEPLOYMENT.md)** | Production deployment |
| **[Security Guide](docs/security/GUIDE.md)** | Authentication, CORS, rate limiting |

---

## API Overview

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/api/v1/validation/assumptions` | POST | Validate causal assumptions |
| `/api/v1/counterfactual/generate` | POST | Generate counterfactuals |
| `/api/v1/sensitivity/analyze` | POST | Sensitivity analysis |
| `/api/v1/explanations/progressive` | POST | Progressive explanations |
| `/api/v1/discovery/extract-factors` | POST | Extract causal factors |

### Authentication

All `/api/v1/*` endpoints require an API key:

```bash
curl -H "X-API-Key: your_api_key" https://isl-staging.onrender.com/api/v1/...
```

**Environment Variables:**
- `ISL_API_KEYS` - Comma-separated API keys (preferred)
- `ISL_API_KEY` - Single API key (legacy, still supported)

### Interactive Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Project Structure

```
src/
├── api/                # FastAPI endpoints
├── services/           # Business logic
├── models/             # Pydantic schemas
├── middleware/         # Auth, rate limiting, timeout
├── config/             # Configuration
└── utils/              # Utilities

tests/
├── unit/               # Unit tests
├── integration/        # Integration tests
└── load/               # Load tests

docs/                   # Documentation
```

---

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ISL_API_KEYS` | - | API keys (comma-separated) |
| `CORS_ORIGINS` | localhost | Allowed CORS origins |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | 100 | Rate limit per client |
| `REQUEST_TIMEOUT_SECONDS` | 60 | Max request duration |
| `REDIS_HOST` | localhost | Redis host for caching |

See [`.env.example`](.env.example) for all options.

---

## Testing

```bash
# Run all tests
pytest -o addopts=""

# Run specific suite
pytest tests/unit/ -v
pytest tests/integration/ -v

# With coverage
pytest --cov=src --cov-report=html
```

**Test Coverage:** 215+ tests, 90%+ coverage

---

## Deployment

### Staging

```
URL: https://isl-staging.onrender.com
Health: https://isl-staging.onrender.com/health
```

### Production Checklist

1. Set `ISL_API_KEYS` (required)
2. Configure `CORS_ORIGINS` for your domains
3. Set up Redis for caching
4. Configure monitoring (Prometheus/Grafana)

See [Deployment Guide](docs/operations/DEPLOYMENT.md) for details.

---

## Contributing

1. Create feature branch from `main`
2. Make changes with tests
3. Run `pytest` and ensure all pass
4. Submit PR with clear description

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Talchain/Inference-Service-Layer/issues)
- **Docs**: [/docs](docs/)

---

**Version:** 2.1.0 | **Status:** Production Ready | **Updated:** 2025-11-26
