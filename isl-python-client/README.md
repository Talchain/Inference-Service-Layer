# ISL Python Client

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Type-safe Python client for the Olumi Inference Service Layer (ISL).

## Features

✅ **Type-Safe**: Full Pydantic models for all requests and responses
✅ **Async & Sync**: Both async and synchronous interfaces
✅ **Resilient**: Built-in retry logic with exponential backoff
✅ **Comprehensive**: All ISL endpoints covered
✅ **Well-Tested**: 50+ unit and integration tests
✅ **Developer-Friendly**: Excellent IDE autocomplete and type hints

## Installation

```bash
pip install isl-client
```

## Quick Start

### Async Usage (Recommended)

```python
from isl_client import ISLClient

async def main():
    async with ISLClient(base_url="https://isl.olumi.ai", api_key="your_key") as client:
        # Validate causal identifiability
        result = await client.causal.validate(
            dag={"nodes": ["Price", "Revenue"], "edges": [["Price", "Revenue"]]},
            treatment="Price",
            outcome="Revenue"
        )
        print(f"Status: {result.status}")  # "identifiable"

# Run async function
import asyncio
asyncio.run(main())
```

### Synchronous Usage

```python
from isl_client import ISLClientSync

with ISLClientSync(base_url="https://isl.olumi.ai") as client:
    result = client.causal.validate(
        dag={"nodes": ["Price", "Revenue"], "edges": [["Price", "Revenue"]]},
        treatment="Price",
        outcome="Revenue"
    )
    print(f"Status: {result.status}")
```

## Core Features

### 1. Causal Validation

```python
# Check if causal effect is identifiable
result = await client.causal.validate(dag, treatment="X", outcome="Y")

if result.status == "identifiable":
    print(f"Method: {result.method}")
    print(f"Adjustment sets: {result.adjustment_sets}")
else:
    # Get suggestions for making it identifiable
    for suggestion in result.suggestions:
        print(f"[{suggestion.priority}] {suggestion.description}")
```

### 2. Counterfactual Prediction

```python
# Predict what would happen with intervention
result = await client.causal.counterfactual(
    model=scm,
    intervention={"Price": 45.0},
    seed=42
)
print(f"Predicted revenue: ${result.prediction.prediction['Revenue']:.2f}")
```

### 3. Conformal Intervals

```python
# Get finite-sample valid prediction intervals
result = await client.causal.counterfactual_conformal(
    model=scm,
    intervention={"Price": 45.0},
    calibration_data=historical_data,
    confidence=0.95
)

print(f"95% interval: [{result.conformal_interval.lower}, {result.conformal_interval.upper}]")
print(f"Coverage guaranteed: {result.coverage_guarantee.guaranteed}")
```

### 4. Batch Scenario Analysis

```python
# Analyze multiple scenarios
result = await client.causal.batch_counterfactuals(
    model=scm,
    scenarios=[
        {"id": "base", "intervention": {"Price": 40}},
        {"id": "aggressive", "intervention": {"Price": 50}},
    ],
    analyze_interactions=True
)

for scenario in result.scenarios:
    print(f"{scenario.scenario_id}: {scenario.prediction}")

if result.interactions.has_synergy:
    print(f"Synergy detected: {result.interactions.summary}")
```

### 5. Causal Discovery

```python
# Discover DAG from data
result = await client.discovery.from_data(
    data=[
        {"Price": 40, "Quality": 7, "Revenue": 1200},
        {"Price": 45, "Quality": 8, "Revenue": 1350},
        # ... more samples
    ],
    variable_names=["Price", "Quality", "Revenue"],
    algorithm="notears",
    prior_knowledge={
        "forbidden_edges": [["Revenue", "Price"]],  # No reverse causation
    }
)

print(f"Discovered {len(result.dag.edges)} edges")
print(f"Edges: {result.dag.edges}")
```

### 6. Contrastive Explanations

```python
# Explain why two scenarios differ
result = await client.explain.contrastive(
    model=scm,
    scenario_a={"id": "success", "state": {"Price": 40, "Quality": 8}},
    scenario_b={"id": "failure", "state": {"Price": 60, "Quality": 6}},
    top_k=3
)

print(result.summary)
for explanation in result.explanations:
    print(f"Key difference: {explanation.counterfactual_path}")
```

### 7. Sequential Optimization

```python
# Find optimal intervention sequence
result = await client.optimize.sequential(
    model=scm,
    objective="Revenue",
    constraints={"budget": 1000, "max_price_change": 10},
    horizon=5,
    initial_state={"Price": 40, "Inventory": 100}
)

print(f"Total utility: {result.total_utility}")
for step in result.optimal_sequence:
    print(f"Step {step.step}: {step.intervention} -> ${step.predicted_outcome}")
```

## Error Handling

The client provides specific exceptions for different error types:

```python
from isl_client import (
    ValidationError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailable,
    TimeoutError,
)

try:
    result = await client.causal.validate(dag, "X", "Y")
except ValidationError as e:
    print(f"Invalid request: {e.details}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except ServiceUnavailable as e:
    print(f"Service down: {e}")
except AuthenticationError:
    print("Check your API key")
```

## Configuration

```python
client = ISLClient(
    base_url="https://isl.olumi.ai",
    api_key="your_api_key",           # Optional
    timeout=30.0,                     # Request timeout (seconds)
    max_retries=3,                    # Maximum retry attempts
    retry_backoff_factor=2.0,         # Exponential backoff multiplier
)
```

## Health Checks

```python
# Basic health check
health = await client.health()
print(health["status"])  # "healthy"

# Get detailed service health (if endpoint available)
# This requires ISL to expose /health/services endpoint
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=isl_client --cov-report=html
```

### Type Checking

```bash
mypy isl_client
```

### Code Formatting

```bash
black isl_client
ruff check isl_client
```

## API Coverage

| Category | Endpoints | Status |
|----------|-----------|--------|
| Validation | `validate`, `validate_with_strategies` | ✅ |
| Counterfactual | `counterfactual`, `counterfactual_conformal` | ✅ |
| Batch | `batch_counterfactuals` | ✅ |
| Transport | `transport` | ✅ |
| Discovery | `from_data`, `from_knowledge`, `hybrid` | ✅ |
| Explanation | `contrastive`, `batch_contrastive` | ✅ |
| Optimization | `sequential`, `multi_objective` | ✅ |

## Examples

See the [`examples/`](examples/) directory for complete examples:

- `basic_validation.py` - Causal validation workflow
- `conformal_prediction.py` - Conformal intervals
- `batch_scenarios.py` - Batch scenario analysis
- `discovery.py` - Causal discovery from data
- `sequential_optimization.py` - Multi-step optimization

## Requirements

- Python 3.9+
- httpx >= 0.25.0
- pydantic >= 2.0.0

## License

MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [GitHub README](https://github.com/Talchain/isl-python-client)
- **Issues**: [GitHub Issues](https://github.com/Talchain/isl-python-client/issues)
- **ISL API Docs**: https://isl.olumi.ai/docs

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Run type checking (`mypy`) and formatting (`black`)
6. Submit a pull request

## Changelog

### 0.1.0 (2025-11-23)

- Initial release
- Full ISL API coverage
- Async and sync interfaces
- Comprehensive error handling
- 50+ tests
- Type-safe Pydantic models
