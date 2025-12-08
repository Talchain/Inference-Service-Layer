# Causal Discovery

## Overview

Automatically **discovers DAG structures** from observational data or domain knowledge, helping formalize causal relationships when structure is uncertain.

## Key Value Proposition

**Traditional Approach:** Manually specify complete DAG structure
**Causal Discovery:** "Here's the data, suggest plausible causal structures"

## Core Features

- **Data-Driven Discovery**: Learn structure from correlations in observational data
- **Knowledge-Guided Discovery**: Generate candidate DAGs from domain descriptions
- **Prior Knowledge Integration**: Enforce required/forbidden edges
- **Confidence Scoring**: Quantify confidence in discovered structures
- **Multiple Candidates**: Return top-k plausible structures

## API Endpoints

### Discovery from Data

```
POST /api/v1/causal/discover/from-data
```

**Request Example**:
```json
{
  "data": [
    [40, 7.5, 30000],
    [45, 8.0, 32500],
    [50, 8.5, 35000]
  ],
  "variable_names": ["Price", "Quality", "Revenue"],
  "prior_knowledge": {
    "required_edges": [["Price", "Revenue"]],
    "forbidden_edges": [["Revenue", "Price"]]
  },
  "threshold": 0.3,
  "seed": 42
}
```

**Response Example**:
```json
{
  "discovered_dags": [
    {
      "nodes": ["Price", "Quality", "Revenue"],
      "edges": [
        ["Price", "Revenue"],
        ["Quality", "Revenue"],
        ["Quality", "Price"]
      ],
      "confidence": 0.85,
      "method": "correlation"
    }
  ],
  "explanation": {
    "summary": "Discovered DAG structure from data with 85% confidence",
    "reasoning": "Found 3 edges using correlation threshold 0.3",
    "technical_basis": "Correlation-based structure learning",
    "assumptions": ["Linear relationships", "No hidden confounders", "Stationarity"]
  }
}
```

### Discovery from Knowledge

```
POST /api/v1/causal/discover/from-knowledge
```

**Request Example**:
```json
{
  "domain_description": "Price affects revenue. Quality affects both price and revenue. Marketing influences revenue.",
  "variable_names": ["Price", "Quality", "Marketing", "Revenue"],
  "prior_knowledge": {
    "required_edges": [["Price", "Revenue"]]
  },
  "top_k": 3
}
```

**Response Example**:
```json
{
  "discovered_dags": [
    {
      "nodes": ["Price", "Quality", "Marketing", "Revenue"],
      "edges": [
        ["Price", "Revenue"],
        ["Quality", "Price"],
        ["Quality", "Revenue"],
        ["Marketing", "Revenue"]
      ],
      "confidence": 0.75,
      "method": "knowledge"
    }
  ],
  "explanation": {
    "summary": "Generated 3 candidate DAG structures from domain knowledge",
    "reasoning": "Used heuristic patterns to match domain description",
    "technical_basis": "Knowledge-guided structure generation",
    "assumptions": ["Domain description is accurate"]
  }
}
```

## Discovery Methods

### Correlation-Based (from Data)

**Algorithm**: Correlation threshold + acyclicity enforcement

**Pros**:
- Fast and simple
- Works with small samples
- Distribution-free

**Cons**:
- Cannot distinguish correlation from causation
- May miss weak effects
- Assumes no hidden confounders

**Best For**: Initial exploration, small datasets (n < 1000)

### Knowledge-Guided (from Domain Description)

**Algorithm**: Heuristic pattern matching (chain, fork, collider)

**Pros**:
- Works without data
- Incorporates domain expertise
- Multiple candidate structures

**Cons**:
- Simplified heuristics
- Cannot validate against data
- Dependent on description quality

**Best For**: When you have domain knowledge but need help formalizing structure

## Parameters

### From Data

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `data` | List[List[float]] | Yes | - | Data matrix (rows = samples, columns = variables) |
| `variable_names` | List[str] | Yes | - | Variable names |
| `prior_knowledge` | Dict | No | None | Required/forbidden edges |
| `threshold` | float | No | 0.3 | Correlation threshold (0-1) |
| `seed` | int | No | None | Random seed |

**Requirements**:
- Minimum 10 observations
- 2-50 variables
- Data should be roughly stationary

### From Knowledge

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `domain_description` | string | Yes | - | Natural language description |
| `variable_names` | List[str] | Yes | - | Variable names |
| `prior_knowledge` | Dict | No | None | Required/forbidden edges |
| `top_k` | int | No | 3 | Number of candidate DAGs |

## Use Cases

### 1. Initial Causal Model Development

**Scenario**: Building first causal model for new domain

**Action**: Use discovery to generate initial structure candidates

**Example**: "Discover structure from 6 months of sales data"

### 2. Validating Domain Knowledge

**Scenario**: Have informal domain knowledge, need formal DAG

**Action**: Use knowledge-guided discovery + validate against data

**Example**: "Generate DAG from product manager's description"

### 3. Hypothesis Generation

**Scenario**: Exploring potential causal relationships

**Action**: Discover from data, examine top-k candidates

**Example**: "What causal structures explain user engagement patterns?"

### 4. Data Collection Planning

**Scenario**: Need to identify which variables to measure

**Action**: Discover from partial data + knowledge

**Example**: "Which additional metrics should we track?"

## Confidence Interpretation

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| 0.8 - 1.0 | High confidence | Use for analysis |
| 0.6 - 0.8 | Moderate confidence | Validate with expert knowledge |
| 0.4 - 0.6 | Low confidence | Use as hypothesis, not fact |
| < 0.4 | Very uncertain | Collect more data or refine knowledge |

## Prior Knowledge Format

```json
{
  "required_edges": [
    ["Variable1", "Variable2"],
    ["Variable2", "Variable3"]
  ],
  "forbidden_edges": [
    ["Variable3", "Variable1"]
  ]
}
```

## Examples

### Example 1: Basic Data Discovery

```python
import requests
import numpy as np

# Generate correlated data
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = 2 * x + np.random.randn(n) * 0.1
z = 3 * y + np.random.randn(n) * 0.1
data = np.column_stack([x, y, z]).tolist()

response = requests.post(
    "http://localhost:8000/api/v1/causal/discover/from-data",
    json={
        "data": data,
        "variable_names": ["X", "Y", "Z"],
        "threshold": 0.5,
        "seed": 42,
    }
)

result = response.json()
dag = result['discovered_dags'][0]
print(f"Edges: {dag['edges']}")
print(f"Confidence: {dag['confidence']}")
```

### Example 2: Knowledge-Guided with Constraints

```python
response = requests.post(
    "http://localhost:8000/api/v1/causal/discover/from-knowledge",
    json={
        "domain_description": "Price drives revenue, quality affects both",
        "variable_names": ["Price", "Quality", "Revenue"],
        "prior_knowledge": {
            "required_edges": [["Price", "Revenue"]],
            "forbidden_edges": [["Revenue", "Price"]]
        },
        "top_k": 3,
    }
)

for i, dag in enumerate(response.json()['discovered_dags']):
    print(f"DAG {i+1} (confidence={dag['confidence']}):")
    print(f"  Edges: {dag['edges']}")
```

## Limitations

1. **Correlation ≠ Causation**: Correlation-based discovery cannot prove causality
2. **Hidden Confounders**: Cannot detect unmeasured confounders
3. **Equivalence Classes**: Multiple DAGs may explain same data
4. **Sample Size**: Small samples yield low confidence
5. **Non-Stationarity**: Assumes relationships are stable over time

## Related Features

- [Causal Validation](./causal-validation.md)
- [Enhanced Validation Strategies](./enhanced-validation-strategies.md)
