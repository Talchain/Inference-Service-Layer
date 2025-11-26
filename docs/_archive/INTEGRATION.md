# Integration Guide

Guide for integrating the Inference Service Layer with the Causal-Enhanced Environment (CEE).

## Overview

The Inference Service Layer provides a REST API for causal inference and decision analysis. This guide shows how to integrate it with your application.

## Quick Integration

### 1. Start the Service

```bash
# Using Docker (recommended)
docker-compose up -d

# Or locally
poetry run python -m src.api.main
```

### 2. Verify Connection

```python
import requests

response = requests.get('http://localhost:8000/health')
print(response.json())
# {'status': 'healthy', 'version': '0.1.0', ...}
```

### 3. Make Your First Request

```python
# Validate a causal model
response = requests.post(
    'http://localhost:8000/api/v1/causal/validate',
    json={
        'dag': {
            'nodes': ['Treatment', 'Outcome', 'Confounder'],
            'edges': [
                ['Treatment', 'Outcome'],
                ['Confounder', 'Treatment'],
                ['Confounder', 'Outcome']
            ]
        },
        'treatment': 'Treatment',
        'outcome': 'Outcome'
    }
)

result = response.json()
print(f"Status: {result['status']}")
print(f"Adjustment set: {result['minimal_set']}")
```

## Python Client Example

### Basic Client

```python
import requests
from typing import Dict, List, Any


class InferenceClient:
    """Client for Inference Service Layer."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def validate_causal_model(
        self,
        nodes: List[str],
        edges: List[tuple],
        treatment: str,
        outcome: str
    ) -> Dict[str, Any]:
        """Validate a causal model."""
        response = self.session.post(
            f"{self.base_url}/api/v1/causal/validate",
            json={
                'dag': {'nodes': nodes, 'edges': edges},
                'treatment': treatment,
                'outcome': outcome
            }
        )
        response.raise_for_status()
        return response.json()

    def analyze_counterfactual(
        self,
        model: Dict[str, Any],
        intervention: Dict[str, float],
        outcome: str
    ) -> Dict[str, Any]:
        """Perform counterfactual analysis."""
        response = self.session.post(
            f"{self.base_url}/api/v1/causal/counterfactual",
            json={
                'model': model,
                'intervention': intervention,
                'outcome': outcome
            }
        )
        response.raise_for_status()
        return response.json()

    def find_team_alignment(
        self,
        perspectives: List[Dict],
        options: List[Dict]
    ) -> Dict[str, Any]:
        """Find team alignment."""
        response = self.session.post(
            f"{self.base_url}/api/v1/team/align",
            json={'perspectives': perspectives, 'options': options}
        )
        response.raise_for_status()
        return response.json()


# Usage
client = InferenceClient()

result = client.validate_causal_model(
    nodes=['X', 'Y', 'Z'],
    edges=[('Z', 'X'), ('X', 'Y'), ('Z', 'Y')],
    treatment='X',
    outcome='Y'
)
```

## JavaScript/TypeScript Client Example

```typescript
interface DAGStructure {
  nodes: string[];
  edges: [string, string][];
}

interface CausalValidationRequest {
  dag: DAGStructure;
  treatment: string;
  outcome: string;
}

class InferenceClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async validateCausalModel(
    request: CausalValidationRequest
  ): Promise<any> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/causal/validate`,
      {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(request)
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  async analyzeCounterfactual(
    model: any,
    intervention: Record<string, number>,
    outcome: string
  ): Promise<any> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/causal/counterfactual`,
      {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({model, intervention, outcome})
      }
    );

    return await response.json();
  }
}

// Usage
const client = new InferenceClient();

const result = await client.validateCausalModel({
  dag: {
    nodes: ['X', 'Y', 'Z'],
    edges: [['Z', 'X'], ['X', 'Y'], ['Z', 'Y']]
  },
  treatment: 'X',
  outcome: 'Y'
});
```

## Error Handling

### Handling Validation Errors

```python
try:
    result = client.validate_causal_model(...)
except requests.HTTPError as e:
    if e.response.status_code == 400:
        error = e.response.json()
        print(f"Validation error: {error['message']}")
        print(f"Suggested action: {error['suggested_action']}")
    elif e.response.status_code == 500:
        error = e.response.json()
        print(f"Server error: {error['trace_id']}")
        if error['retryable']:
            # Retry logic
            pass
```

### Handling Computation Errors

```python
def safe_counterfactual_analysis(client, model, intervention, outcome):
    """Wrapper with error handling and retry logic."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            return client.analyze_counterfactual(model, intervention, outcome)
        except requests.HTTPError as e:
            error = e.response.json()

            if not error.get('retryable', False):
                # Don't retry non-retryable errors
                raise

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
```

## Best Practices

### 1. Connection Pooling

```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class InferenceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
```

### 2. Response Caching

```python
from functools import lru_cache
import hashlib
import json

class InferenceClient:
    @lru_cache(maxsize=128)
    def validate_causal_model_cached(
        self,
        nodes_hash: str,
        edges_hash: str,
        treatment: str,
        outcome: str
    ):
        # Reconstruct from hashes and call API
        # Only cache for identical inputs
        pass
```

### 3. Async Operations

```python
import aiohttp
import asyncio

class AsyncInferenceClient:
    async def validate_causal_model(self, dag, treatment, outcome):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/causal/validate",
                json={'dag': dag, 'treatment': treatment, 'outcome': outcome}
            ) as response:
                return await response.json()

# Usage
async def main():
    client = AsyncInferenceClient()
    result = await client.validate_causal_model(...)
```

## Integration Patterns

### Pattern 1: Decision Workflow

```python
def decision_workflow(decision_context):
    """Full decision analysis workflow."""
    client = InferenceClient()

    # 1. Validate causal model
    validation = client.validate_causal_model(
        nodes=decision_context['variables'],
        edges=decision_context['relationships'],
        treatment=decision_context['action'],
        outcome=decision_context['goal']
    )

    if validation['status'] != 'identifiable':
        return {'error': 'Causal model cannot be validated'}

    # 2. Analyze counterfactuals
    counterfactual = client.analyze_counterfactual(
        model=decision_context['model'],
        intervention=decision_context['proposed_action'],
        outcome=decision_context['goal']
    )

    # 3. Find team alignment
    alignment = client.find_team_alignment(
        perspectives=decision_context['stakeholders'],
        options=decision_context['options']
    )

    # 4. Combine results
    return {
        'causal_valid': True,
        'adjustment_set': validation['minimal_set'],
        'predicted_outcome': counterfactual['prediction'],
        'uncertainty': counterfactual['uncertainty']['overall'],
        'team_recommendation': alignment['recommendation']['top_option'],
        'explanations': {
            'causal': validation['explanation'],
            'prediction': counterfactual['explanation'],
            'alignment': alignment['explanation']
        }
    }
```

### Pattern 2: Batch Processing

```python
async def batch_validate_models(models: List[Dict]):
    """Validate multiple models in parallel."""
    client = AsyncInferenceClient()

    tasks = [
        client.validate_causal_model(
            m['dag'], m['treatment'], m['outcome']
        )
        for m in models
    ]

    results = await asyncio.gather(*tasks)
    return results
```

## Monitoring Integration

### Health Check Integration

```python
# In your service's health check
def check_inference_service():
    try:
        response = requests.get(
            'http://localhost:8000/health',
            timeout=5
        )
        return response.status_code == 200
    except:
        return False
```

### Metrics Collection

```python
import time

class InferenceClient:
    def __init__(self, base_url, metrics_client=None):
        self.base_url = base_url
        self.metrics = metrics_client

    def validate_causal_model(self, *args, **kwargs):
        start = time.time()
        try:
            result = self._validate(*args, **kwargs)
            duration = time.time() - start

            if self.metrics:
                self.metrics.timing('inference.validate.duration', duration)
                self.metrics.incr('inference.validate.success')

            return result
        except Exception as e:
            if self.metrics:
                self.metrics.incr('inference.validate.error')
            raise
```

## Deployment Configuration

### Development

```python
INFERENCE_SERVICE_URL = "http://localhost:8000"
INFERENCE_TIMEOUT = 30
```

### Production

```python
INFERENCE_SERVICE_URL = "https://inference.yourdomain.com"
INFERENCE_TIMEOUT = 60
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY")  # Future
```

## Testing Your Integration

```python
import pytest
from unittest.mock import Mock, patch

def test_integration_with_mock():
    """Test integration with mocked service."""
    mock_response = {
        'status': 'identifiable',
        'minimal_set': ['Confounder'],
        'confidence': 'high'
    }

    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.status_code = 200

        client = InferenceClient()
        result = client.validate_causal_model(
            nodes=['A', 'B', 'C'],
            edges=[('A', 'B')],
            treatment='A',
            outcome='B'
        )

        assert result['status'] == 'identifiable'
```

## Troubleshooting

### Connection Issues

```python
# Check if service is running
try:
    response = requests.get('http://localhost:8000/health', timeout=5)
    print(f"Service status: {response.json()['status']}")
except requests.ConnectionError:
    print("ERROR: Cannot connect to Inference Service")
    print("Make sure service is running: docker-compose up")
```

### Slow Response Times

- Check `MAX_MONTE_CARLO_ITERATIONS` setting (lower = faster)
- Use caching for repeated requests
- Consider batch processing for multiple requests

### Validation Errors

- Ensure DAG has no cycles
- Check that treatment/outcome nodes exist in graph
- Verify structural equations are valid Python expressions

## Support

For integration questions:
- See full API documentation: `/docs/API.md`
- Check interactive docs: `http://localhost:8000/docs`
- Open GitHub issue for bugs
- Contact development team for support

## Next Steps

1. Review full API documentation
2. Implement client wrapper for your language
3. Add error handling and retry logic
4. Set up monitoring and metrics
5. Write integration tests
6. Deploy to production

---

Last updated: 2025-01-19
