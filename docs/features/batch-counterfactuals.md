# Batch Counterfactual Analysis with Interaction Detection

## Overview

The batch counterfactual system enables efficient analysis of multiple intervention scenarios in a single request, with automatic detection of synergistic and antagonistic interactions between variables.

## Key Features

- **Batch Processing**: Analyze 2-20 scenarios simultaneously
- **Interaction Detection**: Automatically identify synergistic/antagonistic effects
- **Deterministic**: Shared exogenous samples ensure reproducibility
- **Efficient**: Reuses counterfactual engine infrastructure
- **Comprehensive**: Full uncertainty and robustness analysis per scenario

---

## API Endpoint

```
POST /api/v1/causal/counterfactual/batch
```

### Request Example

```json
{
  "model": {
    "variables": ["Price", "Quality", "Marketing", "Revenue"],
    "equations": {
      "Revenue": "10000 + 500*Price + 200*Quality + 0.5*Marketing + 100*Price*Quality"
    },
    "distributions": {
      "noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
    }
  },
  "scenarios": [
    {
      "id": "baseline",
      "intervention": {"Price": 40},
      "label": "Current pricing"
    },
    {
      "id": "price_only",
      "intervention": {"Price": 50},
      "label": "10% price increase"
    },
    {
      "id": "quality_only",
      "intervention": {"Quality": 9},
      "label": "Quality improvement"
    },
    {
      "id": "combined",
      "intervention": {"Price": 50, "Quality": 9},
      "label": "Premium strategy"
    }
  ],
  "outcome": "Revenue",
  "analyze_interactions": true,
  "samples": 1000,
  "seed": 42
}
```

### Response Example

```json
{
  "scenarios": [
    {
      "scenario_id": "baseline",
      "intervention": {"Price": 40},
      "label": "Current pricing",
      "prediction": {
        "point_estimate": 30000,
        "confidence_interval": {"lower": 28000, "upper": 32000}
      },
      "uncertainty": {"overall": "medium"},
      "robustness": {"score": "robust"}
    },
    {
      "scenario_id": "combined",
      "intervention": {"Price": 50, "Quality": 9},
      "label": "Premium strategy",
      "prediction": {
        "point_estimate": 45000,
        "confidence_interval": {"lower": 42000, "upper": 48000}
      },
      "uncertainty": {"overall": "medium"},
      "robustness": {"score": "moderate"}
    }
  ],
  "interactions": {
    "pairwise": [
      {
        "variables": ["Price", "Quality"],
        "type": "synergistic",
        "effect_size": 5000,
        "significance": 0.85,
        "explanation": "Price and Quality interact synergistically: combined effect (45k) exceeds sum of individual effects (40k) by 5k"
      }
    ],
    "summary": "Strong synergistic interaction between Price and Quality (£5k additional gain)"
  },
  "comparison": {
    "best_outcome": "combined",
    "most_robust": "baseline",
    "marginal_gains": {
      "price_only": 5000,
      "quality_only": 3000,
      "combined": 15000
    },
    "ranking": ["combined", "price_only", "quality_only", "baseline"]
  },
  "explanation": {
    "summary": "Best scenario 'combined' yields £15k gain vs baseline with moderate robustness",
    "reasoning": "Analyzed 4 scenarios. Detected synergistic interaction between Price and Quality.",
    "technical_basis": "Batch counterfactual analysis with shared exogenous samples; interaction detection via additive decomposition",
    "assumptions": ["Structural equations correct", "Intervention effects stable", "No external confounding"]
  }
}
```

---

## Interaction Detection

### Algorithm

The system detects interactions by comparing combined effects to the sum of individual effects:

1. **Additive Assumption**: Effect(A+B) = Effect(A) + Effect(B)
2. **Test**: Actual effect vs expected additive effect
3. **Classify**:
   - **Synergistic**: Actual > Expected (variables amplify each other)
   - **Antagonistic**: Actual < Expected (variables interfere)
   - **Additive**: Actual ≈ Expected (±5% threshold)

### Example: Synergistic Interaction

```
Price only:    Revenue = 35k  (gain: +5k)
Quality only:  Revenue = 33k  (gain: +3k)
Expected:      Revenue = 38k  (5k + 3k)
Actual (both): Revenue = 45k  (gain: +15k)
Interaction:   +7k synergistic effect
```

### Example: Antagonistic Interaction

```
Price up:      Revenue = 35k
Marketing up:  Revenue = 33k
Expected:      Revenue = 38k
Actual (both): Revenue = 34k
Interaction:   -4k antagonistic effect (diminishing returns)
```

---

## Use Cases

### 1. Strategy Comparison
Compare multiple intervention strategies:
- Conservative vs aggressive pricing
- Quality-focused vs cost-focused
- Regional variations

### 2. Interaction Discovery
Identify when variables work together:
- Price + Quality synergy
- Marketing + Product antagonism
- Channel + Message combinations

### 3. Marginal Analysis
Understand incremental gains:
- What's the uplift from each additional intervention?
- Where do diminishing returns start?
- Which combinations are cost-effective?

---

## Performance

- **Complexity**: O(n_scenarios * simulation_cost)
- **Typical Time**: ~5-10s for 5 scenarios with 1000 samples each
- **Scaling**: Linear with number of scenarios
- **Optimization**: Shared exogenous samples reduce variance

---

## Testing

### Unit Tests (30+ tests)
- Batch processing with multiple scenarios
- Interaction detection (synergistic, antagonistic, additive)
- Scenario comparison and ranking
- Marginal gains computation
- Determinism verification
- Edge cases

### Integration Tests (10+ tests)
- API endpoint validation
- Deterministic behavior
- Complex multi-variable models
- Custom sample counts
- Label preservation

---

## Comparison with Single Counterfactual

| Feature | Single Counterfactual | Batch Counterfactual |
|---------|----------------------|---------------------|
| **Scenarios** | 1 per request | 2-20 per request |
| **Interactions** | Not detected | Automatic detection |
| **Comparison** | Manual | Built-in ranking |
| **Efficiency** | 1x | ~N scenarios in N time |
| **Use Case** | Single "what if" | Strategy comparison |

---

## Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | StructuralModel | Yes | - | Causal model |
| `scenarios` | List[ScenarioSpec] | Yes | - | 2-20 scenarios |
| `outcome` | string | Yes | - | Outcome variable |
| `analyze_interactions` | boolean | No | true | Enable interaction detection |
| `robustness_radius` | float | No | 0.1 | Perturbation radius |
| `samples` | int | No | 1000 | Monte Carlo samples |
| `seed` | int | No | None | Random seed |

---

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `scenarios` | List[ScenarioResult] | Results for each scenario |
| `interactions` | InteractionAnalysis | Detected interactions (if enabled) |
| `comparison` | ScenarioComparison | Ranking and marginal gains |
| `explanation` | ExplanationMetadata | Plain English summary |

---

## Best Practices

1. **Baseline First**: Include current state as first scenario for marginal gains
2. **Test Pairs**: To detect interactions, include single-variable and combined scenarios
3. **Limit Scenarios**: Keep to 5-10 scenarios for reasonable response times
4. **Use Labels**: Provide descriptive labels for easier interpretation
5. **Set Seed**: Use seed for reproducible results

---

## Examples

### Example 1: Pricing Strategy

```json
{
  "scenarios": [
    {"id": "baseline", "intervention": {"Price": 40}},
    {"id": "conservative", "intervention": {"Price": 42}},
    {"id": "moderate", "intervention": {"Price": 45}},
    {"id": "aggressive", "intervention": {"Price": 50}}
  ]
}
```

Result: Compare revenue uplift across pricing strategies

### Example 2: Interaction Detection

```json
{
  "scenarios": [
    {"id": "price_only", "intervention": {"Price": 50}},
    {"id": "marketing_only", "intervention": {"Marketing": 100000}},
    {"id": "both", "intervention": {"Price": 50, "Marketing": 100000}}
  ]
}
```

Result: Detect if price and marketing synergize or interfere

---

## Integration with Other Features

- **Contrastive Explanations**: Use batch to test interventions found by contrastive analysis
- **FACET Robustness**: Each scenario includes full robustness analysis
- **Sensitivity Analysis**: Compare robustness across scenarios

---

## Implementation Notes

- Reuses `CounterfactualEngine` for individual scenarios
- Shared exogenous samples ensure determinism
- Interaction detection via additive decomposition
- Simple heuristics for significance (effect_size / expected > threshold)

---

## Future Enhancements

- Higher-order interactions (3-way, 4-way)
- Pareto frontier analysis
- Automatic scenario generation
- Sensitivity of interactions to assumptions
