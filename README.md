# Inference Service Layer

**Deterministic scientific computation core for Olumi's decision enhancement platform**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

The Inference Service Layer provides robust causal inference, counterfactual analysis, team alignment, and sensitivity analysis capabilities via a REST API. Built with deterministic computation at its core, it ensures identical inputs always produce identical outputs.

### Key Features

- **Causal Validation**: Identify adjustment sets and validate causal models using Y₀
- **Counterfactual Analysis**: Generate "what-if" predictions with uncertainty quantification
- **Team Alignment**: Find common ground across stakeholder perspectives
- **Sensitivity Analysis**: Test assumption robustness and identify critical factors
- **Deterministic**: Reproducible results for identical inputs
- **Well-Documented**: Comprehensive API documentation with examples

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/Talchain/Inference-Service-Layer.git
cd Inference-Service-Layer

# Start the service
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Local Development

```bash
# Install dependencies with Poetry
poetry install

# Copy environment template
cp .env.example .env

# Run the service
poetry run python -m src.api.main

# Or with uvicorn directly
poetry run uvicorn src.api.main:app --reload
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service status and version.

### Causal Validation

```bash
POST /api/v1/causal/validate
```

Validates whether a causal model (DAG) supports causal identification.

**Example Request:**
```json
{
  "dag": {
    "nodes": ["Price", "Brand", "Revenue"],
    "edges": [["Price", "Revenue"], ["Brand", "Price"], ["Brand", "Revenue"]]
  },
  "treatment": "Price",
  "outcome": "Revenue"
}
```

**Example Response:**
```json
{
  "status": "identifiable",
  "adjustment_sets": [["Brand"]],
  "minimal_set": ["Brand"],
  "confidence": "high",
  "explanation": {
    "summary": "Effect is identifiable by controlling for Brand",
    "reasoning": "Brand influences both Price and Revenue, creating confounding...",
    "technical_basis": "Backdoor criterion satisfied with adjustment set {Brand}",
    "assumptions": ["No unmeasured confounding", "Correct causal structure"]
  }
}
```

### Counterfactual Analysis

```bash
POST /api/v1/causal/counterfactual
```

Analyzes what would happen under a counterfactual intervention.

**Example Request:**
```json
{
  "model": {
    "variables": ["Price", "Brand", "Revenue"],
    "equations": {
      "Brand": "baseline_brand + 0.3 * Price",
      "Revenue": "10000 + 500 * Price - 200 * Brand"
    },
    "distributions": {
      "baseline_brand": {"type": "normal", "parameters": {"mean": 50, "std": 5}}
    }
  },
  "intervention": {"Price": 15},
  "outcome": "Revenue"
}
```

### Team Alignment

```bash
POST /api/v1/team/align
```

Finds common ground across team perspectives and recommends aligned options.

### Sensitivity Analysis

```bash
POST /api/v1/analysis/sensitivity
```

Tests how robust conclusions are to changes in assumptions.

## Interactive API Documentation

Once the service is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Documentation

### Operations Documentation

For operations teams managing ISL in production:

- **[Pilot Monitoring Runbook](docs/operations/PILOT_MONITORING_RUNBOOK.md)** - Daily monitoring procedures, alert response, escalation paths
- **[Redis Strategy](docs/operations/REDIS_STRATEGY.md)** - Cache architecture, key patterns, TTL standards, operational procedures
- **[Redis Troubleshooting](docs/operations/REDIS_TROUBLESHOOTING.md)** - Common issues, diagnostic commands, emergency procedures
- **[Staging Deployment Checklist](docs/operations/STAGING_DEPLOYMENT_CHECKLIST.md)** - Step-by-step deployment procedures, rollback plans

### Integration Documentation

For UI teams integrating with ISL:

- **[Cross-Reference Schema](docs/integration/CROSS_REFERENCE_SCHEMA.md)** - Assumption traceability, stable IDs, navigation flows, TypeScript implementation

### Developer Documentation

For developers optimizing and extending ISL:

- **[Optimization Roadmap](docs/development/OPTIMIZATION_ROADMAP.md)** - 4-phase performance optimization strategy with targets and ROI analysis
- **[Performance Profiling Script](scripts/profile_performance.py)** - Executable profiling tool for identifying bottlenecks
- **[Redis Performance Validation](scripts/validate_redis_performance.py)** - Redis performance testing and validation

## Project Structure

```
inference-service-layer/
├── src/
│   ├── api/                 # FastAPI routes
│   │   ├── main.py          # App setup
│   │   ├── causal.py        # Causal endpoints
│   │   ├── team.py          # Team alignment
│   │   └── analysis.py      # Sensitivity analysis
│   ├── services/            # Core business logic
│   │   ├── causal_validator.py
│   │   ├── counterfactual_engine.py
│   │   ├── team_aligner.py
│   │   └── sensitivity_analyzer.py
│   ├── models/              # Pydantic schemas
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── shared.py
│   ├── utils/               # Utilities
│   │   ├── determinism.py
│   │   ├── graph_parser.py
│   │   └── validation.py
│   └── config.py            # Configuration
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   │   ├── test_fingerprinting.py    # Version fingerprinting
│   │   ├── test_redis_failover.py    # Redis failover scenarios
│   │   ├── test_concurrency.py       # Concurrent request handling
│   │   └── test_redis_health.py      # Redis health checks
│   └── fixtures/
├── docs/                    # Documentation
│   ├── operations/          # Operations runbooks
│   │   ├── PILOT_MONITORING_RUNBOOK.md
│   │   ├── REDIS_STRATEGY.md
│   │   ├── REDIS_TROUBLESHOOTING.md
│   │   └── STAGING_DEPLOYMENT_CHECKLIST.md
│   ├── integration/         # UI integration guides
│   │   └── CROSS_REFERENCE_SCHEMA.md
│   └── development/         # Developer documentation
│       └── OPTIMIZATION_ROADMAP.md
├── scripts/                 # Operational scripts
│   ├── profile_performance.py        # Performance profiling
│   └── validate_redis_performance.py # Redis validation
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
# Run all tests with coverage
poetry run pytest

# Run specific test file
poetry run pytest tests/unit/test_determinism.py

# Run integration tests
poetry run pytest tests/integration/ -v

# Run Redis health checks (requires Redis running)
poetry run pytest tests/integration/test_redis_health.py -v -s

# Run with verbose output
poetry run pytest -v

# Generate coverage report
poetry run pytest --cov=src --cov-report=html
```

### Integration Test Suites

**Fingerprinting & Determinism** (`test_fingerprinting.py`):
- Version fingerprinting metadata on all endpoints
- Deterministic responses for identical inputs
- Request ID uniqueness and propagation

**Redis Failover** (`test_redis_failover.py`):
- Graceful degradation when Redis unavailable
- Error propagation with request IDs
- Service availability under failure conditions

**Concurrency** (`test_concurrency.py`):
- Concurrent request handling without interference
- Performance stability under sustained load
- Cache contention behaviour

**Redis Health** (`test_redis_health.py`):
- Redis connectivity and configuration
- TTL enforcement (no infinite keys)
- Eviction policy and memory limits

## Configuration

Configuration is managed through environment variables. See `.env.example` for all available options.

### Key Settings

- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `MAX_MONTE_CARLO_ITERATIONS`: Number of Monte Carlo samples (default: 10000)
- `DEFAULT_CONFIDENCE_LEVEL`: Confidence level for intervals (default: 0.95)
- `ENABLE_DETERMINISTIC_MODE`: Ensure deterministic computations (default: true)

## Determinism

All computations are deterministic by design:

- Identical inputs → Identical outputs
- Random number generators are seeded from request data
- Results include deterministic hashes for verification

Example:
```python
from src.utils.determinism import make_deterministic, canonical_hash

# Seed all RNGs from request data
seed = make_deterministic(request_data)

# Verify determinism
hash1 = canonical_hash(result1.dict())
hash2 = canonical_hash(result2.dict())
assert hash1 == hash2  # Same input = same output
```

## Development

### Code Style

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff src/ tests/

# Type checking
poetry run mypy src/
```

### Adding New Features

1. Create models in `src/models/`
2. Implement service in `src/services/`
3. Add endpoint in `src/api/`
4. Write tests in `tests/`
5. Update documentation

## Architecture

### Core Dependencies

- **FastAPI**: Modern web framework for APIs
- **Y₀**: Causal identification library
- **NetworkX**: Graph operations
- **NumPy/SciPy**: Scientific computing
- **Pydantic**: Data validation

### Design Principles

1. **Determinism First**: All computations must be reproducible
2. **Clear Explanations**: Every response includes human-readable explanations
3. **Comprehensive Error Handling**: Structured errors with actionable suggestions
4. **Type Safety**: Strict typing with Pydantic models
5. **Testability**: High test coverage (80%+ target)

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t inference-service:latest .

# Run container
docker run -p 8000:8000 inference-service:latest
```

### Production Considerations

- Use `WORKERS > 1` for production (but note: determinism requires single-threaded processing)
- Set `RELOAD=false` in production
- Configure appropriate `LOG_LEVEL`
- Set up health check monitoring on `/health`
- Use reverse proxy (nginx/traefik) for SSL/TLS

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`poetry run pytest`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues:

- Open an issue on GitHub
- Contact the development team
- See full documentation in `/docs`

## Roadmap

### Phase 0: Core Functionality (Completed ✅)
- ✅ Causal validation with Y₀
- ✅ Counterfactual analysis
- ✅ Team alignment
- ✅ Sensitivity analysis
- ✅ Comprehensive testing (119 tests)
- ✅ Docker support

### Phase 1: Production Readiness (Completed ✅)
- ✅ Version fingerprinting for determinism
- ✅ Request ID propagation for distributed tracing
- ✅ Redis caching strategy and integration
- ✅ Configuration fingerprinting
- ✅ API documentation and integration guides

### Phase 2: Operational Excellence & Pilot Readiness (Completed ✅)
- ✅ Operations runbooks and monitoring procedures
- ✅ Redis operational strategy and troubleshooting guides
- ✅ Staging deployment checklists and rollback procedures
- ✅ Enhanced integration testing (21 tests)
- ✅ Performance profiling tools
- ✅ 4-phase optimization roadmap
- ✅ Cross-reference schema for UI integration

### Phase 3 (Future)
- ActiVA integration for value alignment
- Bayesian Teaching for explanations
- Advanced FACET features
- Phase 1-2 optimization implementation (targeting 40-70% latency reduction)
- Additional distribution types
- GraphQL API option

## Acknowledgments

Built for Olumi's decision enhancement platform, leveraging:

- [Y₀](https://github.com/y0-causal-inference/y0) for causal identification
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [NetworkX](https://networkx.org/) for graph operations

---

**Version**: 1.0.0
**Status**: Phase 2 - Operational Excellence Complete, Pilot Ready
**Last Updated**: 2025-01-20
