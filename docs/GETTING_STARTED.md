# Getting Started with ISL

**Time to first API call: ~5 minutes**

---

## Prerequisites

- Python 3.11+
- Docker (optional, recommended)
- Redis (optional, for caching)

---

## Setup

### Option A: Docker (Recommended)

```bash
# Clone and start
git clone https://github.com/Talchain/Inference-Service-Layer.git
cd Inference-Service-Layer
docker-compose up -d

# Verify
curl http://localhost:8000/health
# → {"status": "healthy", "version": "2.1.0"}
```

### Option B: Local Python

```bash
# Clone
git clone https://github.com/Talchain/Inference-Service-Layer.git
cd Inference-Service-Layer

# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env as needed

# Run
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Your First API Call

### 1. Health Check (No Auth)

```bash
curl http://localhost:8000/health
```

### 2. Validate a Causal Model (Requires API Key)

```bash
# Set your API key
export ISL_API_KEY="your_api_key_here"

# Validate a simple DAG
curl -X POST http://localhost:8000/api/v1/validation/assumptions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "dag": {
      "nodes": ["Marketing", "Price", "Sales"],
      "edges": [["Marketing", "Sales"], ["Price", "Sales"]]
    },
    "treatment": "Marketing",
    "outcome": "Sales"
  }'
```

**Expected Response:**
```json
{
  "schema": "validation.v1",
  "is_valid": true,
  "assumptions": [...],
  "metadata": {
    "isl_version": "2.1.0",
    "request_id": "req_abc123"
  }
}
```

### 3. Interactive Documentation

Open http://localhost:8000/docs in your browser for Swagger UI with all endpoints.

---

## Key Concepts

### What ISL Does

ISL answers causal questions:

| Question | ISL Endpoint | Example |
|----------|--------------|---------|
| "Can I estimate causal effect of X on Y?" | `/validation/assumptions` | Is Marketing → Sales identifiable? |
| "What if I change X?" | `/counterfactual/generate` | What if Marketing = $1M? |
| "How sensitive is my conclusion?" | `/sensitivity/analyze` | Is effect robust to confounding? |
| "Explain this to me" | `/explanations/progressive` | Simple → Technical explanation |

### DAG Structure

All causal operations use Directed Acyclic Graphs (DAGs):

```json
{
  "dag": {
    "nodes": ["A", "B", "C"],    // Variables
    "edges": [["A", "B"], ["B", "C"]]  // Causal relationships (A→B, B→C)
  },
  "treatment": "A",  // Variable being manipulated
  "outcome": "C"     // Variable being measured
}
```

### Authentication

Protected endpoints require `X-API-Key` header:

```bash
curl -H "X-API-Key: your_key" http://localhost:8000/api/v1/...
```

**Public endpoints (no auth):** `/health`, `/ready`, `/metrics`, `/docs`, `/redoc`

---

## Project Layout

```
src/
├── api/           # HTTP endpoints (start here for API changes)
│   ├── main.py    # FastAPI app setup, middleware
│   ├── causal.py  # /api/v1/causal/* endpoints
│   ├── validation.py
│   └── ...
│
├── services/      # Business logic (core algorithms)
│   ├── causal_validator.py
│   ├── sensitivity_analyzer.py
│   └── ...
│
├── models/        # Request/Response schemas
│   ├── requests.py
│   ├── responses.py
│   └── ...
│
├── middleware/    # Cross-cutting concerns
│   ├── auth.py           # API key validation
│   ├── rate_limiting.py  # Rate limits
│   └── request_limits.py # Timeout, size limits
│
└── config/        # Configuration
    └── __init__.py  # Settings class
```

---

## Common Tasks

### Adding a New Endpoint

1. **Define models** in `src/models/`:
   ```python
   class MyRequest(BaseModel):
       field: str
   ```

2. **Add endpoint** in `src/api/`:
   ```python
   @router.post("/my-endpoint")
   async def my_endpoint(request: MyRequest):
       return {"result": "..."}
   ```

3. **Write tests** in `tests/unit/`:
   ```python
   def test_my_endpoint():
       response = client.post("/my-endpoint", json={...})
       assert response.status_code == 200
   ```

### Running Tests

```bash
# All tests
pytest -o addopts=""

# Specific file
pytest tests/unit/test_auth_middleware.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

### Debugging

```bash
# Run with debug logging
LOG_LEVEL=DEBUG uvicorn src.api.main:app --reload

# Check logs (JSON format)
# {"timestamp": "...", "level": "INFO", "message": "..."}
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ISL_API_KEYS` | Prod | - | Comma-separated API keys |
| `ISL_API_KEY` | - | - | Single API key (legacy) |
| `CORS_ORIGINS` | Prod | localhost | Allowed CORS origins |
| `REDIS_HOST` | - | localhost | Redis server |
| `LOG_LEVEL` | - | INFO | DEBUG, INFO, WARNING, ERROR |
| `REQUEST_TIMEOUT_SECONDS` | - | 60 | Max request duration |

See `.env.example` for complete list.

---

## Next Steps

1. **Explore the API**: http://localhost:8000/docs
2. **Read the spec**: [Technical Specification](TECHNICAL_SPECIFICATION.md)
3. **See examples**: [API Examples](api/EXAMPLES.md)
4. **Integrate**: [Integration Guide](integration/GUIDE.md)

---

## Troubleshooting

### "401 Unauthorized"
- Missing or invalid `X-API-Key` header
- Check `ISL_API_KEYS` or `ISL_API_KEY` env var is set

### "429 Too Many Requests"
- Rate limit exceeded (100 req/min default)
- Wait and retry, or increase `RATE_LIMIT_REQUESTS_PER_MINUTE`

### "504 Gateway Timeout"
- Request took > 60 seconds
- Simplify input or increase `REQUEST_TIMEOUT_SECONDS`

### Tests failing
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Check pytest is using correct config
pytest -o addopts="" tests/unit/ -v
```

---

**Need help?** Open an issue or check [docs/](.)
