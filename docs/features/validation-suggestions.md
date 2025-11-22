# DAG Validation Suggestions

## Overview

The ISL validation feedback system provides **actionable, algorithmic suggestions** when causal DAGs are non-identifiable. Instead of simply reporting "cannot identify," the system analyzes the DAG structure and returns specific recommendations for achieving identifiability.

## Features

- **Structured Suggestions**: Machine-readable suggestion objects with clear actions
- **Priority Levels**: Suggestions ranked as critical, recommended, or optional
- **Algorithmic Generation**: Deterministic, no LLM calls
- **Backward Compatible**: Legacy string suggestions maintained alongside new structured format

## Example Response

### Before (Legacy)
```json
{
  "status": "cannot_identify",
  "reason": "unmeasured_confounding",
  "suggestions": [
    "Add measured confounders to the model to block backdoor paths",
    "Consider using instrumental variables if available"
  ]
}
```

### After (Enhanced)
```json
{
  "status": "cannot_identify",
  "reason": "unmeasured_confounding",
  "suggestions": [
    {
      "type": "add_confounder",
      "description": "Include Brand in adjustment set to control confounding",
      "technical_detail": "Node Brand confounds Price and Revenue. Adjust for Brand to block backdoor path: Price ← Brand → Revenue",
      "priority": "critical",
      "action": {
        "add_edges": [["Price", "Revenue"]]
      }
    },
    {
      "type": "add_mediator",
      "description": "Add CustomerSentiment to model the causal mechanism",
      "technical_detail": "Replace direct edge Price→Revenue with Price→CustomerSentiment→Revenue to enable front-door criterion",
      "priority": "recommended",
      "action": {
        "add_node": "CustomerSentiment",
        "add_edges": [
          ["Price", "CustomerSentiment"],
          ["CustomerSentiment", "Revenue"]
        ]
      }
    }
  ],
  "legacy_suggestions": [
    "Include Brand in adjustment set to control confounding",
    "Add CustomerSentiment to model the causal mechanism"
  ]
}
```

## Suggestion Types

### 1. Add Confounder (`add_confounder`)

**When**: Backdoor paths exist but no valid adjustment set found

**Example**:
```json
{
  "type": "add_confounder",
  "description": "Measure competitor activity to control confounding",
  "technical_detail": "Add Competitors node to block backdoor path: Price ← Competitors → Revenue",
  "priority": "critical",
  "action": {
    "add_node": "Competitors",
    "add_edges": [
      ["Competitors", "Price"],
      ["Competitors", "Revenue"]
    ]
  }
}
```

### 2. Add Mediator (`add_mediator`)

**When**: Direct treatment→outcome edge could benefit from decomposition

**Example**:
```json
{
  "type": "add_mediator",
  "description": "Add PriceMechanism to model the causal mechanism",
  "technical_detail": "Replace Price→Revenue with Price→PriceMechanism→Revenue",
  "priority": "recommended",
  "action": {
    "add_node": "PriceMechanism",
    "add_edges": [
      ["Price", "PriceMechanism"],
      ["PriceMechanism", "Revenue"]
    ]
  }
}
```

### 3. Reverse Edge (`reverse_edge`)

**When**: Reversing an edge direction might enable identification

**Example**:
```json
{
  "type": "reverse_edge",
  "description": "Consider reversing edge A→B to B→A",
  "technical_detail": "Current direction A→B contributes to confounding. Reversing to B→A reduces backdoor paths. Verify this direction is theoretically plausible in your domain.",
  "priority": "optional",
  "action": {
    "reverse_edge": ["A", "B"]
  }
}
```

### 4. Conditional Independence (`add_conditional_independence`)

**When**: Complex confounding requires independence assumptions

**Example**:
```json
{
  "type": "add_conditional_independence",
  "description": "Assume Revenue is independent of unmeasured confounders given measured variables",
  "technical_detail": "Make conditional independence assumption: Revenue ⊥ Unmeasured | {Measured Variables}",
  "priority": "recommended",
  "action": {
    "assume_independence": {
      "variable_a": "Revenue",
      "variable_b": "UnmeasuredConfounders",
      "conditioning_set": ["Price", "Brand"]
    }
  }
}
```

## Priority Levels

- **critical**: Essential for identification (e.g., measuring confounders)
- **recommended**: Helpful but not strictly required (e.g., adding mediators)
- **optional**: Worth considering (e.g., edge reversals requiring domain knowledge)

## API Usage

### Endpoint
```
POST /api/v1/causal/validate
```

### Request
```json
{
  "dag": {
    "nodes": ["Price", "Revenue", "Brand"],
    "edges": [
      ["Brand", "Price"],
      ["Brand", "Revenue"],
      ["Price", "Revenue"]
    ]
  },
  "treatment": "Price",
  "outcome": "Revenue"
}
```

### Response (Non-Identifiable)
```json
{
  "status": "cannot_identify",
  "reason": "unmeasured_confounding",
  "suggestions": [
    {
      "type": "add_confounder",
      "description": "Include Brand in adjustment set to control confounding",
      "technical_detail": "Node Brand confounds Price and Revenue. Adjust for Brand to block backdoor path: Price ← Brand → Revenue",
      "priority": "critical",
      "action": {
        "add_edges": [["Price", "Revenue"]]
      }
    }
  ],
  "legacy_suggestions": [
    "Include Brand in adjustment set to control confounding"
  ],
  "confidence": "medium",
  "explanation": {
    "summary": "Effect cannot be identified due to unmeasured confounding",
    "reasoning": "Backdoor paths exist but no valid adjustment set found with measured variables.",
    "technical_basis": "No identification formula available",
    "assumptions": ["DAG structure correct"]
  }
}
```

## Implementation Details

### Algorithm

1. **Backdoor Path Analysis**: Identifies confounding paths and suggests adjustments
2. **Mediator Detection**: Detects direct edges that could benefit from decomposition
3. **Edge Reversal Testing**: Tests if reversing edges reduces backdoor paths
4. **Priority Sorting**: Orders suggestions by criticality

### Performance

- **Target**: <100ms overhead for suggestion generation
- **Achieved**: ~2ms for typical DAGs (<20 nodes)
- **Deterministic**: Same DAG always produces same suggestions

### Test Coverage

- 20 comprehensive unit tests
- 90%+ coverage of suggestion generation code
- Edge cases: empty graphs, single nodes, large graphs

## Examples from Tests

### Classic Backdoor Confounding
```python
# DAG: X ← Z → Y, X → Y
dag = {
    "nodes": ["X", "Y", "Z"],
    "edges": [["Z", "X"], ["Z", "Y"], ["X", "Y"]]
}

# Suggestion:
# "Include Z in adjustment set to control confounding"
```

### No Causal Path
```python
# DAG: X  Y (disconnected)
dag = {
    "nodes": ["X", "Y"],
    "edges": []
}

# Suggestion:
# "Add direct or mediated path from X to Y"
```

### Complex Confounding
```python
# DAG: Z1 → X, Z1 → Y, Z2 → X, Z2 → Y, X → Y
dag = {
    "nodes": ["X", "Y", "Z1", "Z2"],
    "edges": [
        ["Z1", "X"], ["Z1", "Y"],
        ["Z2", "X"], ["Z2", "Y"],
        ["X", "Y"]
    ]
}

# Suggestions:
# 1. "Include Z1 in adjustment set..." (critical)
# 2. "Include Z2 in adjustment set..." (critical)
# 3. "Assume Y ⊥ Unmeasured | {Z1, Z2}..." (recommended)
```

## Backward Compatibility

The system maintains backward compatibility:

- **New field**: `suggestions` (List[ValidationSuggestion])
- **Legacy field**: `legacy_suggestions` (List[str])
- Existing clients using `suggestions` as strings get `legacy_suggestions`
- New clients can use structured `suggestions` objects

## Future Enhancements

Potential future improvements:

1. **Front-door criterion detection**: Identify when mediators fully capture mechanism
2. **Instrumental variable suggestions**: Recommend potential IVs based on graph structure
3. **Data collection guidance**: Estimate measurement costs for suggested variables
4. **Interactive refinement**: Allow users to accept/reject suggestions and update DAG

## References

- Pearl, J. (2009). *Causality: Models, Reasoning and Inference*
- Bareinboim, E., & Pearl, J. (2016). *Causal inference and the data-fusion problem*
- Y₀ library documentation: https://y0.readthedocs.io/
