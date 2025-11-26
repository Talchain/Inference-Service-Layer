# Enhanced Y₀ Validation Strategies

## Overview

Provides **complete adjustment strategies** for non-identifiable DAGs, going beyond simple validation to give actionable recommendations on how to achieve causal identifiability.

## Key Value Proposition

**Standard Validation:** "Effect is not identifiable"
**Enhanced Strategies:** "Add variable X and control for it to achieve identifiability via Pearl's backdoor criterion"

## Core Features

- **Complete Adjustment Strategies**: Specifies exactly which nodes and edges to add
- **Path Analysis**: Identifies backdoor paths, frontdoor paths, and critical nodes
- **Multiple Strategy Types**: Backdoor, frontdoor, and instrumental variable approaches
- **Theoretical Justification**: Each strategy includes theoretical basis
- **Ranked Recommendations**: Strategies ranked by expected success

## API Endpoint

```
POST /api/v1/causal/validate/strategies
```

### Request Example

```json
{
  "dag": {
    "nodes": ["Price", "Competitors", "Revenue"],
    "edges": [
      ["Competitors", "Price"],
      ["Competitors", "Revenue"],
      ["Price", "Revenue"]
    ]
  },
  "treatment": "Price",
  "outcome": "Revenue"
}
```

### Response Example

```json
{
  "strategies": [
    {
      "strategy_type": "backdoor",
      "nodes_to_add": [],
      "edges_to_add": [],
      "explanation": "Control for existing variable Competitors to block backdoor paths",
      "theoretical_basis": "Pearl's backdoor criterion",
      "expected_identifiability": 0.9
    }
  ],
  "path_analysis": {
    "backdoor_paths": [
      ["Price", "Competitors", "Revenue"]
    ],
    "frontdoor_paths": [
      ["Price", "Revenue"]
    ],
    "blocked_paths": [],
    "critical_nodes": ["Competitors"]
  },
  "explanation": {
    "summary": "Found 1 adjustment strategy",
    "reasoning": "Best strategy: Control for existing variable Competitors",
    "technical_basis": "Path analysis and adjustment set identification",
    "assumptions": ["DAG structure is correct"]
  }
}
```

## Strategy Types

### Backdoor Adjustment

**When**: Confounding paths exist between treatment and outcome

**Action**: Control for variables on backdoor paths

**Example**: "Add and measure Competitors variable, then control for it"

### Frontdoor Adjustment

**When**: Cannot block backdoor paths but have complete mediators

**Action**: Use frontdoor criterion via mediator set

**Example**: "Use frontdoor criterion via mediators: CustomerSentiment"

### Instrumental Variables

**When**: Have variables affecting treatment but not outcome directly

**Action**: Use instrumental variable identification

**Example**: "Use RandomizedPrice as instrumental variable"

## Path Analysis

### Backdoor Paths

Paths from treatment to outcome that go through confounders.

**Example**: Price ← Competitors → Revenue

### Frontdoor Paths

Direct causal paths from treatment to outcome.

**Example**: Price → Revenue

### Critical Nodes

Nodes that block multiple paths if controlled.

**Interpretation**: High-leverage intervention points

## Use Cases

### 1. Making Non-Identifiable Effects Identifiable

**Scenario**: DAG validation shows "cannot_identify"

**Action**: Use enhanced strategies to determine what data to collect

**Example**: "Measure competitor activity to enable identification"

### 2. Data Collection Planning

**Scenario**: Planning observational study

**Action**: Identify critical variables to measure

**Example**: "Must measure Brand to block confounding"

### 3. Experimental Design

**Scenario**: Cannot randomize treatment

**Action**: Identify instrumental variables or mediators

**Example**: "Use price randomization in subset of markets as instrument"

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dag` | DAGStructure | Yes | Causal graph structure |
| `treatment` | string | Yes | Treatment variable |
| `outcome` | string | Yes | Outcome variable |

## Response Fields

### AdjustmentStrategy

- `strategy_type`: backdoor, frontdoor, or instrumental
- `nodes_to_add`: Variables to add/measure
- `edges_to_add`: Edges to add to DAG
- `explanation`: Plain English description
- `theoretical_basis`: Theoretical justification
- `expected_identifiability`: Confidence (0-1)

### PathAnalysis

- `backdoor_paths`: Confounding paths
- `frontdoor_paths`: Direct causal paths
- `blocked_paths`: Already blocked paths
- `critical_nodes`: High-leverage nodes

## Examples

### Example 1: Simple Confounding

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/causal/validate/strategies",
    json={
        "dag": {
            "nodes": ["Treatment", "Confounder", "Outcome"],
            "edges": [
                ["Confounder", "Treatment"],
                ["Confounder", "Outcome"],
                ["Treatment", "Outcome"],
            ],
        },
        "treatment": "Treatment",
        "outcome": "Outcome",
    }
)

result = response.json()
print(f"Number of strategies: {len(result['strategies'])}")
print(f"Best strategy: {result['strategies'][0]['explanation']}")
```

### Example 2: Path Analysis

```python
result = response.json()
path_analysis = result['path_analysis']

print(f"Backdoor paths: {path_analysis['backdoor_paths']}")
print(f"Critical nodes: {path_analysis['critical_nodes']}")
```

## Related Features

- [Standard Causal Validation](./causal-validation.md)
- [Y₀ Transportability](./transportability.md)
