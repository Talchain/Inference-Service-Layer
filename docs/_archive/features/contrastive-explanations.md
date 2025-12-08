# Contrastive Explanations with Minimal Interventions

## Overview

The ISL contrastive explanation system transforms "what if X=50" predictions into **actionable recommendations**: "Change X from 40 to 45 to achieve target." It finds **minimal sufficient interventions** to achieve desired outcomes, combining binary search optimization with FACET robustness verification.

## Features

- **Minimal Single-Variable Interventions**: Binary search to find smallest change to each feasible variable
- **Multi-Variable Combinations**: Grid search for optimal combinations (if allowed by constraints)
- **FACET Robustness Scoring**: Each intervention evaluated for robustness to perturbations
- **Cost & Feasibility Estimates**: Automatic estimation of implementation difficulty
- **Flexible Ranking**: Rank by change magnitude, cost, feasibility, or robustness
- **Deterministic Results**: Same seed produces identical recommendations

## Example Response

### Current State
```json
{
  "current": {
    "Price": 40,
    "Revenue": 40000
  },
  "target": {
    "Revenue": "50000-55000"
  }
}
```

### Response
```json
{
  "minimal_interventions": [
    {
      "rank": 1,
      "changes": {
        "Price": {
          "variable": "Price",
          "from_value": 40,
          "to_value": 45,
          "delta": 5,
          "relative_change": 12.5,
          "unit": "£"
        }
      },
      "expected_outcome": {
        "Revenue": 51000
      },
      "confidence_interval": {
        "Revenue": {
          "lower": 48000,
          "upper": 54000,
          "confidence_level": 0.95
        }
      },
      "feasibility": 0.95,
      "cost_estimate": "low",
      "robustness": "robust",
      "robustness_score": 0.85
    },
    {
      "rank": 2,
      "changes": {
        "Marketing": {
          "variable": "Marketing",
          "from_value": 30000,
          "to_value": 40000,
          "delta": 10000,
          "relative_change": 33.3
        }
      },
      "expected_outcome": {
        "Revenue": 52000
      },
      "confidence_interval": {
        "Revenue": {
          "lower": 47000,
          "upper": 57000,
          "confidence_level": 0.95
        }
      },
      "feasibility": 0.85,
      "cost_estimate": "high",
      "robustness": "moderate",
      "robustness_score": 0.65
    }
  ],
  "comparison": {
    "best_by_cost": 1,
    "best_by_robustness": 1,
    "best_by_feasibility": 1,
    "synergies": "Combining interventions yields diminishing returns",
    "tradeoffs": "Price increase is cheaper and more robust; Marketing spend is costlier but has wider impact"
  },
  "explanation": {
    "summary": "Increase Price from £40 to £45 to achieve target revenue",
    "reasoning": "Price has strong causal effect on Revenue. A 12.5% increase yields expected £11k revenue gain with high robustness.",
    "technical_basis": "Binary search for minimal change with FACET robustness verification",
    "assumptions": [
      "Structural equations capture true causal relationships",
      "Variable values can be changed as specified",
      "No external confounding factors"
    ]
  }
}
```

## Use Cases

### 1. Revenue Optimization
**Goal**: Increase revenue from £40k to £50k

**Before (Counterfactual)**:
- "If Price=50, Revenue=£52k"
- "If Marketing=£50k, Revenue=£55k"

**After (Contrastive)**:
- "Change Price from £40 to £45 (£5 increase) achieves target"
- "Alternative: Increase Marketing from £30k to £40k (£10k spend)"
- "Recommended: Price increase (lower cost, higher robustness)"

### 2. Quality Target
**Goal**: Achieve customer satisfaction score of 8.5-9.0

**Intervention**:
- "Improve ResponseTime from 2h to 1.5h AND increase SupportStaff from 10 to 12"
- Cost: Medium, Robustness: High

### 3. Multi-Objective Optimization
**Goal**: Increase revenue while maintaining quality

**Constraints**:
- Price: feasible (can change)
- Quality: fixed (cannot change)
- Marketing: feasible (can change)
- Target: Revenue ≥ £50k

**Result**:
- "Increase Marketing from £30k to £45k (given Quality=7.5 is fixed)"

## Algorithm

### 1. Binary Search for Minimal Change

For each feasible variable:

```python
def find_minimal_change(variable, current_state, target):
    """
    Binary search to find minimal change achieving target.

    1. Determine search direction (increase vs decrease)
    2. Binary search over variable range
    3. For each value, simulate outcome
    4. Find smallest change satisfying target
    """
    low, high = bounds[variable]
    best_value = None

    while high - low > precision:
        mid = (low + high) / 2
        intervention = {variable: mid}
        outcome = simulate(intervention)

        if target_min <= outcome <= target_max:
            best_value = mid
            # Try smaller change
            high = mid
        else:
            # Need larger change
            low = mid

    return best_value
```

**Complexity**: O(log(range/precision) * simulation_cost)

### 2. Multi-Variable Grid Search

For combinations of 2+ variables:

```python
def find_minimal_combination(var1, var2, target):
    """
    Grid search for minimal two-variable intervention.

    1. Create grid of variable values
    2. Test each combination
    3. Find minimal total change satisfying target
    """
    best_combo = None
    best_distance = float('inf')

    for v1 in grid_points(var1):
        for v2 in grid_points(var2):
            intervention = {var1: v1, var2: v2}
            outcome = simulate(intervention)

            if target_min <= outcome <= target_max:
                distance = euclidean_distance(intervention, current_state)
                if distance < best_distance:
                    best_distance = distance
                    best_combo = intervention

    return best_combo
```

**Complexity**: O(grid_size^num_vars * simulation_cost)

### 3. FACET Robustness Verification

For each candidate intervention:

```python
def evaluate_robustness(intervention):
    """
    Evaluate intervention robustness via FACET.

    1. Generate perturbation region around intervention (±10%)
    2. Sample 100+ points from region
    3. Verify all samples achieve target
    4. Compute robustness score
    """
    region = create_perturbation_region(intervention, radius=0.1)
    samples = region.sample(n=100)

    successful = 0
    for sample in samples:
        outcome = simulate(sample)
        if target_min <= outcome <= target_max:
            successful += 1

    robustness_score = successful / len(samples)
    return robustness_score
```

### 4. Ranking by Criterion

```python
def rank_interventions(interventions, criterion):
    """
    Rank interventions by optimization criterion.

    - change_magnitude: Smallest total change first
    - cost: Low cost before high cost
    - feasibility: Highest feasibility first
    - robustness: Highest robustness score first
    """
    if criterion == "change_magnitude":
        return sorted(interventions, key=lambda i: total_change(i))
    elif criterion == "cost":
        cost_order = {"low": 0, "medium": 1, "high": 2}
        return sorted(interventions, key=lambda i: cost_order[i.cost])
    elif criterion == "feasibility":
        return sorted(interventions, key=lambda i: -i.feasibility)
    elif criterion == "robustness":
        return sorted(interventions, key=lambda i: -i.robustness_score)
```

## API Usage

### Endpoint
```
POST /api/v1/explain/contrastive
```

### Request Schema

```json
{
  "model": {
    "variables": ["Price", "Quality", "Marketing", "Brand", "Revenue"],
    "equations": {
      "Brand": "50 + 0.3 * Quality - 0.1 * Price",
      "Revenue": "10000 + 800*Price + 200*Quality + 0.5*Marketing + 300*Brand"
    },
    "distributions": {
      "noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
    }
  },
  "current_state": {
    "Price": 40,
    "Quality": 7.5,
    "Marketing": 30000
  },
  "observed_outcome": {
    "Revenue": 40000
  },
  "target_outcome": {
    "Revenue": [50000, 55000]
  },
  "constraints": {
    "feasible": ["Price", "Marketing"],
    "fixed": ["Quality"],
    "max_changes": 2,
    "minimize": "cost",
    "variable_bounds": {
      "Price": [30, 100],
      "Marketing": [10000, 100000]
    }
  },
  "seed": 42
}
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | StructuralModel | Yes | Causal model with equations and distributions |
| `current_state` | Dict[str, float] | Yes | Current values for all variables |
| `observed_outcome` | Dict[str, float] | Yes | Current observed outcome values |
| `target_outcome` | Dict[str, Tuple[float, float]] | Yes | Target outcome ranges (min, max) |
| `constraints` | InterventionConstraints | Yes | Constraints on interventions |
| `seed` | int | No | Random seed for determinism |

### Constraints Schema

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `feasible` | List[str] | Yes | Variables that can be changed |
| `fixed` | List[str] | No | Variables that cannot be changed |
| `max_changes` | int | No | Maximum variables to change (default: 1) |
| `minimize` | str | No | Optimization criterion: "change_magnitude" (default), "cost", "feasibility" |
| `variable_bounds` | Dict[str, Tuple[float, float]] | No | Optional bounds for each variable |

### Response Schema

```typescript
{
  minimal_interventions: [
    {
      rank: number,                    // 1 = best
      changes: {
        [variable: string]: {
          variable: string,
          from_value: number,
          to_value: number,
          delta: number,
          relative_change: number,
          unit?: string
        }
      },
      expected_outcome: {
        [outcome: string]: number
      },
      confidence_interval: {
        [outcome: string]: {
          lower: number,
          upper: number,
          confidence_level: number
        }
      },
      feasibility: number,             // 0-1
      cost_estimate: "low" | "medium" | "high",
      robustness: "robust" | "moderate" | "fragile",
      robustness_score: number         // 0-1
    }
  ],
  comparison: {
    best_by_cost: number,
    best_by_robustness: number,
    best_by_feasibility: number,
    synergies: string,
    tradeoffs: string
  },
  explanation: {
    summary: string,
    reasoning: string,
    technical_basis: string,
    assumptions: string[]
  }
}
```

## Cost & Feasibility Estimation

### Cost Estimation

Heuristic based on number of changes and magnitude:

```python
def estimate_cost(intervention, current_state):
    """
    Estimate implementation cost.

    Low:    1 variable, <20% change
    Medium: 1-2 variables, 20-50% change
    High:   2+ variables, >50% change
    """
    num_changes = len(intervention)
    avg_relative_change = mean([
        abs((new_val - old_val) / old_val)
        for var, new_val in intervention.items()
    ])

    if num_changes == 1 and avg_relative_change < 0.2:
        return "low"
    elif num_changes <= 2 and avg_relative_change < 0.5:
        return "medium"
    else:
        return "high"
```

### Feasibility Estimation

Based on change magnitude and bounds:

```python
def compute_feasibility(intervention, current_state, constraints):
    """
    Compute feasibility score (0-1).

    - Check variables are in feasible set
    - Check values within bounds
    - Penalize large changes (>100% change reduces feasibility)
    """
    # Check feasibility
    for var in intervention:
        if var not in constraints.feasible:
            return 0.0

    # Check bounds
    for var, val in intervention.items():
        if var in constraints.variable_bounds:
            min_val, max_val = constraints.variable_bounds[var]
            if not (min_val <= val <= max_val):
                return 0.0

    # Compute based on change magnitude
    total_change = sum([
        abs((new_val - current_state[var]) / current_state[var])
        for var, new_val in intervention.items()
    ])
    avg_change = total_change / len(intervention)

    # Smaller changes = more feasible
    feasibility = max(0.0, min(1.0, 1.0 - (avg_change / 2)))
    return feasibility
```

## Performance

### Complexity

- **Single-variable search**: O(log(range) * simulation_cost)
  - Binary search: ~20 iterations
  - Simulation: ~2-5k Monte Carlo samples
  - Total: <3s per variable

- **Multi-variable search**: O(grid_size^num_vars * simulation_cost)
  - Grid: 10x10 = 100 points
  - Total: <5s for 2 variables

- **Robustness verification**: O(samples * simulation_cost)
  - Samples: 100 per intervention
  - Total: ~1s per intervention

### Total Latency

- **1 variable, 5 candidates**: ~3s
- **2 variables, 10 candidates**: ~8s
- **Target**: <10s for typical requests

### Optimization Strategies

1. **Adaptive sampling** in counterfactual engine (2-5x speedup)
2. **Parallel evaluation** of candidates (planned)
3. **Caching** of structural model computations
4. **Early termination** if excellent solution found

## Test Coverage

### Unit Tests (30+ tests)

```python
# tests/unit/test_contrastive_explainer.py

def test_single_variable_minimal_intervention():
    """Test finding minimal change to single variable."""

def test_multiple_feasible_variables():
    """Test finding interventions across multiple feasible variables."""

def test_multi_variable_combination():
    """Test finding minimal multi-variable combination."""

def test_respects_fixed_constraints():
    """Test that fixed variables are not changed."""

def test_no_solution_returns_empty():
    """Test that unachievable targets return empty list."""

def test_deterministic_with_seed():
    """Test that same seed produces identical results."""

def test_rank_by_change_magnitude():
    """Test ranking by change magnitude."""

def test_rank_by_cost():
    """Test ranking by cost."""

def test_rank_by_feasibility():
    """Test ranking by feasibility."""

def test_robustness_evaluated():
    """Test that robustness is evaluated for interventions."""

def test_feasibility_within_bounds():
    """Test feasibility computation for intervention within bounds."""

def test_cost_estimate_small_change():
    """Test cost estimate for small change."""
```

### Integration Tests (10+ tests)

```python
# tests/integration/test_contrastive_endpoints.py

@pytest.mark.asyncio
async def test_contrastive_explanation_basic(client):
    """Test basic contrastive explanation request."""

@pytest.mark.asyncio
async def test_contrastive_explanation_deterministic(client):
    """Test that contrastive explanation is deterministic with seed."""

@pytest.mark.asyncio
async def test_contrastive_explanation_complex_model(client):
    """Test contrastive explanation with complex multi-variable model."""
```

## Edge Cases Handled

1. **No solution exists**: Returns empty list with explanation
2. **Very tight target range**: Handles precision constraints
3. **Empty feasible variables**: Validation error (caught by Pydantic)
4. **Circular dependencies**: Detected by topological sort
5. **Unbounded variables**: Uses default bounds (±100% from current)
6. **Multi-modal outcome distributions**: Monte Carlo handles via sampling

## Comparison with Related Features

| Feature | Purpose | Output |
|---------|---------|--------|
| **Counterfactual Analysis** | "What if X=50?" | Predicted outcome with uncertainty |
| **Contrastive Explanations** | "How to achieve Y=target?" | Minimal interventions to reach target |
| **Sensitivity Analysis** | "Which assumptions matter?" | Assumption importance ranking |
| **FACET Robustness** | "Is intervention robust?" | Robustness score and regions |

## Future Enhancements

Potential improvements:

1. **Pareto frontier**: Show tradeoff curves (cost vs robustness vs feasibility)
2. **Conditional interventions**: "If Price>50, then..."
3. **Time-varying interventions**: Multi-stage plans
4. **Resource constraints**: Budget limits, capacity constraints
5. **Risk-aware recommendations**: Incorporate downside risk
6. **Interactive refinement**: Allow user to adjust constraints iteratively

## References

- **Wachter et al. (2018)**: Counterfactual Explanations without Opening the Black Box
- **Karimi et al. (2020)**: Model-Agnostic Counterfactual Explanations for Consequential Decisions
- **Miller, T. (2019)**: Explanation in Artificial Intelligence: Insights from the Social Sciences
- **Pearl, J. (2009)**: Causality: Models, Reasoning and Inference

## Implementation Notes

### Service Architecture

```
ContrastiveExplainer
├── find_minimal_interventions()
│   ├── _find_minimal_single_variable_change()
│   │   └── Binary search with counterfactual simulation
│   ├── _find_minimal_multi_variable_combinations()
│   │   └── Grid search for 2+ variables
│   ├── _evaluate_intervention()
│   │   ├── _simulate_intervention() (via CounterfactualEngine)
│   │   ├── _analyze_intervention_robustness() (via RobustnessAnalyzer)
│   │   ├── _compute_feasibility()
│   │   └── _estimate_cost()
│   ├── _rank_interventions()
│   ├── _generate_comparison()
│   └── _generate_explanation()
```

### Dependencies

- `CounterfactualEngine`: For outcome simulation
- `RobustnessAnalyzer`: For FACET robustness verification
- `ExplanationGenerator`: For plain English explanations

### Determinism

All computations are deterministic given a seed:

```python
seed = make_deterministic(request.model_dump())
if request.seed is not None:
    seed = request.seed
np.random.seed(seed)
```

This ensures:
- Identical requests → identical responses
- Reproducible recommendations
- Testable behavior
