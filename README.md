# Inference Service Layer (ISL)

**Causal inference API for Olumi's decision platform**

---

## Quick Start

```bash
# 1. Clone & run
git clone https://github.com/Talchain/Inference-Service-Layer.git
cd Inference-Service-Layer
docker-compose up -d

# 2. Verify
curl http://localhost:8000/health

# 3. Test endpoint
curl -X POST http://localhost:8000/api/v1/validation/assumptions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{"dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]}, "treatment": "A", "outcome": "B"}'
```

---

## What ISL Does

| Capability | Endpoint | Description |
|------------|----------|-------------|
| **Validation** | `/api/v1/validation/*` | Validate causal DAGs |
| **Counterfactuals** | `/api/v1/counterfactual/*` | "What-if" analysis |
| **Sensitivity** | `/api/v1/sensitivity/*` | Test assumption robustness |
| **Explanations** | `/api/v1/explanations/*` | Multi-level explanations |
| **Discovery** | `/api/v1/discovery/*` | Extract factors from text |

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[Getting Started](docs/GETTING_STARTED.md)** | New developer setup (5 min) |
| **[Technical Spec](docs/TECHNICAL_SPECIFICATION.md)** | Architecture & API reference |
| **[API Examples](docs/api/EXAMPLES.md)** | Working code examples |
| **[Integration](docs/integration/GUIDE.md)** | PLoT/TAE/CEE integration |
| **[Deployment](docs/operations/DEPLOYMENT.md)** | Docker/K8s deployment |
| **[Security](docs/security/GUIDE.md)** | Auth & rate limiting |

---

## Configuration

```bash
# Required (production)
ISL_API_KEYS=key1,key2           # API keys (comma-separated)
CORS_ORIGINS=https://your-app.com # Allowed origins

# Optional
ISL_API_KEY=single_key           # Legacy single key (still supported)
REQUEST_TIMEOUT_SECONDS=60       # Request timeout
REDIS_HOST=localhost             # Redis for caching
```

---

## Project Structure

```
src/
├── api/          # HTTP endpoints
├── services/     # Business logic
├── models/       # Pydantic schemas
├── middleware/   # Auth, rate limiting
└── config/       # Configuration

docs/
├── GETTING_STARTED.md
├── TECHNICAL_SPECIFICATION.md
├── api/EXAMPLES.md
├── integration/GUIDE.md
├── operations/DEPLOYMENT.md
└── security/GUIDE.md
```

---

## Testing

```bash
pytest -o addopts=""           # All tests
pytest tests/unit/ -v          # Unit tests only
```

**Coverage:** 215+ tests, 90%+

---

## Deployment

| Environment | URL |
|-------------|-----|
| Staging | https://isl-staging.onrender.com |
| Health | https://isl-staging.onrender.com/health |
| Docs | https://isl-staging.onrender.com/docs |

---

**Version:** 2.1.0 | **Status:** Production Ready
