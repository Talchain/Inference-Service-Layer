# Sequential Experiment Optimization

## Overview

Recommends the **next best experiment** using Thompson sampling, balancing exploration (learning about parameters) with exploitation (optimizing outcomes) for efficient sequential experimentation.

## Key Value Proposition

**Traditional A/B Testing:** Test random variations or intuition-based choices
**Sequential Optimization:** "Test Price=55 next - high information gain (0.75) in underexplored region"

## Core Features

- **Thompson Sampling**: Bayesian algorithm for optimal exploration/exploitation
- **Information Gain Estimation**: Quantifies how much you'll learn
- **Belief State Tracking**: Maintains uncertainty about model parameters
- **Cost-Benefit Analysis**: Balances experiment value vs cost
- **Rationale Generation**: Explains why this experiment is recommended

## API Endpoint

```
POST /api/v1/causal/experiment/recommend
```

### Request Example

```json
{
  "beliefs": [
    {
      "parameter_name": "effect_price",
      "distribution_type": "normal",
      "parameters": {"mean": 500, "std": 50}
    },
    {
      "parameter_name": "effect_quality",
      "distribution_type": "uniform",
      "parameters": {"low": 100, "high": 300}
    }
  ],
  "objective": {
    "target_variable": "Revenue",
    "goal": "maximize"
  },
  "constraints": {
    "budget": 100000,
    "time_horizon": 10,
    "feasible_interventions": {
      "Price": [30, 100],
      "Quality": [5, 10]
    }
  },
  "history": [
    {
      "intervention": {"Price": 45},
      "outcome": {"Revenue": 32000},
      "cost": 5000
    },
    {
      "intervention": {"Price": 50},
      "outcome": {"Revenue": 35000},
      "cost": 5000
    }
  ],
  "seed": 42
}
```

### Response Example

```json
{
  "recommendation": {
    "intervention": {"Price": 55, "Quality": 8.5},
    "expected_outcome": {"Revenue": 38000},
    "expected_information_gain": 0.75,
    "cost_estimate": 10000,
    "rationale": "Explore: Test Price=55, Quality=8.5 to learn more (high information gain: 0.75)",
    "exploration_vs_exploitation": 0.8
  },
  "explanation": {
    "summary": "Recommend explore strategy: high information gain in underexplored region",
    "reasoning": "Information gain: 0.75, Cost: 10000",
    "technical_basis": "Thompson sampling with 100 posterior samples",
    "assumptions": ["Parameter beliefs accurate", "Cost estimates reasonable"]
  }
}
```

## How Thompson Sampling Works

### Algorithm

1. **Sample Parameters**: Draw from current belief distributions
2. **Evaluate Candidates**: Score each feasible intervention
3. **Select Best**: Choose intervention with highest expected value
4. **Repeat**: Average over many samples (typically 100)

### Why It Works

- **Bayesian**: Naturally incorporates uncertainty
- **Efficient**: Provably optimal in many settings
- **Adaptive**: Automatically balances exploration/exploitation
- **Simple**: Easy to implement and explain

## Belief Distributions

### Normal Distribution

**Use when**: Symmetric uncertainty around mean

```json
{
  "parameter_name": "effect_price",
  "distribution_type": "normal",
  "parameters": {"mean": 500, "std": 50}
}
```

**Interpretation**: Effect is likely ~500, with 95% probability in [400, 600]

### Uniform Distribution

**Use when**: Only know plausible range

```json
{
  "parameter_name": "effect_quality",
  "distribution_type": "uniform",
  "parameters": {"low": 100, "high": 300}
}
```

**Interpretation**: Effect equally likely anywhere in [100, 300]

## Optimization Objectives

### Maximize

```json
{
  "target_variable": "Revenue",
  "goal": "maximize"
}
```

**Use when**: Bigger is better (revenue, conversion, engagement)

### Minimize

```json
{
  "target_variable": "Cost",
  "goal": "minimize"
}
```

**Use when**: Smaller is better (churn, latency, cost)

### Target

```json
{
  "target_variable": "Quality",
  "goal": "target",
  "target_value": 8.5
}
```

**Use when**: Have specific target value

## Exploration vs Exploitation

### Exploration Score Interpretation

| Score | Type | When | Why |
|-------|------|------|-----|
| 0.8 - 1.0 | Pure Exploration | No/little history | Learn about parameters |
| 0.5 - 0.8 | Balanced | Early-mid experiments | Learn while optimizing |
| 0.2 - 0.5 | Leaning Exploit | Mid-late experiments | Focus on optimization |
| 0.0 - 0.2 | Pure Exploitation | Late experiments | Optimize known best |

### Information Gain Interpretation

| Gain | Interpretation | Action |
|------|----------------|--------|
| 0.8 - 1.0 | Very high | Unexplored region, high learning potential |
| 0.5 - 0.8 | High | Moderate learning potential |
| 0.2 - 0.5 | Medium | Some learning, mostly refining estimates |
| 0.0 - 0.2 | Low | Well-explored, little new information |

## Use Cases

### 1. A/B Testing Campaigns

**Scenario**: Sequential marketing experiments

**Action**: Recommend next test to maximize learning + revenue

**Example**: "Test price $55 - high info gain in underexplored range"

### 2. Clinical Trial Design

**Scenario**: Dose-finding study

**Action**: Balance learning optimal dose with patient outcomes

**Example**: "Test 5mg dose - balances safety learning with efficacy"

### 3. Product Optimization

**Scenario**: Iteratively improving product features

**Action**: Guide feature experiments for efficient optimization

**Example**: "Test feature combination X+Y - unexplored but promising"

### 4. Scientific Inquiry

**Scenario**: Budget-constrained research

**Action**: Maximize knowledge gain per experiment

**Example**: "Next experiment should test edge of current knowledge"

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `beliefs` | List[BeliefDistribution] | Yes | Current parameter beliefs |
| `objective` | OptimizationObjective | Yes | What to optimize |
| `constraints` | ExperimentConstraints | Yes | Budget, time, feasibility |
| `history` | List[ExperimentHistory] | No | Previous experiments |
| `seed` | int | No | Random seed for reproducibility |

## Examples

### Example 1: Simple Recommendation

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/causal/experiment/recommend",
    json={
        "beliefs": [
            {
                "parameter_name": "effect_price",
                "distribution_type": "normal",
                "parameters": {"mean": 500, "std": 50}
            }
        ],
        "objective": {
            "target_variable": "Revenue",
            "goal": "maximize"
        },
        "constraints": {
            "budget": 100000,
            "time_horizon": 10,
            "feasible_interventions": {
                "Price": [30, 100]
            }
        },
        "seed": 42
    }
)

result = response.json()
rec = result['recommendation']
print(f"Recommended intervention: {rec['intervention']}")
print(f"Expected outcome: {rec['expected_outcome']}")
print(f"Information gain: {rec['expected_information_gain']:.2f}")
print(f"Rationale: {rec['rationale']}")
```

### Example 2: Sequential Experiments

```python
history = []

for i in range(5):
    # Get recommendation
    response = requests.post(url, json={
        "beliefs": beliefs,
        "objective": objective,
        "constraints": constraints,
        "history": history,
        "seed": 42 + i,
    })

    rec = response.json()['recommendation']

    # Run experiment (simulated)
    intervention = rec['intervention']
    outcome = run_experiment(intervention)  # Your function

    # Update history
    history.append({
        "intervention": intervention,
        "outcome": outcome,
        "cost": rec['cost_estimate'],
    })

    print(f"Experiment {i+1}: {intervention} -> {outcome}")
```

## Belief Updating

After each experiment, update your beliefs (outside this API):

```python
# After experiment
observed_outcome = 35000
predicted_outcome = 33000
residual = observed_outcome - predicted_outcome

# Update belief mean (simple example)
new_mean = old_mean + 0.1 * residual
new_std = old_std * 0.95  # Reduce uncertainty

# Next recommendation uses updated beliefs
```

## Limitations

1. **Simplified Model**: Assumes linear relationship for evaluation
2. **No Belief Updating**: API doesn't update beliefs (you must do this)
3. **Cost Estimation**: Uses simple magnitude-based cost model
4. **Myopic**: Optimizes next step, not full sequence
5. **No Constraints**: Doesn't handle complex feasibility constraints

## Best Practices

1. **Start Broad**: Use wide distributions initially
2. **Update Beliefs**: After each experiment, refine your beliefs
3. **Trust the Algorithm**: Thompson sampling is provably efficient
4. **Balance Horizon**: Short horizon → exploit, long horizon → explore
5. **Validate Costs**: Ensure cost estimates are realistic

## Related Features

- [Counterfactual Analysis](./counterfactuals.md)
- [Batch Counterfactuals](./batch-counterfactuals.md)
- [Sensitivity Analysis](./sensitivity-analysis.md)
