# ISL Edge Case Handling

This document describes how ISL handles edge cases and unusual inputs across all endpoints.

## Table of Contents

- [Utility Validation](#utility-validation)
- [Multi-Criteria Aggregation](#multi-criteria-aggregation)
- [Pareto Frontier](#pareto-frontier)
- [Risk Adjustment](#risk-adjustment)
- [Threshold Detection](#threshold-detection)

---

## Utility Validation

### Edge Case: Weights Don't Sum to 1.0

**Input:**
```json
{
  "weights": {
    "outcome_a": 0.6,
    "outcome_b": 0.6
  }
}
```

**Behavior:**
- Returns `valid: false`
- Includes error with code `WEIGHTS_NOT_NORMALIZED`
- Provides `normalized_utility` with corrected weights: `{"outcome_a": 0.5, "outcome_b": 0.5}`

**Rationale:** Utility weights must sum to 1.0 to represent a valid probability distribution.

---

### Edge Case: Empty Weights

**Input:**
```json
{
  "weights": {}
}
```

**Behavior:**
- Returns `valid: false`
- Includes error with code `NO_WEIGHTS`
- No suggestions provided

**Rationale:** At least one weight is required to define a utility function.

---

### Edge Case: All Weights are Zero

**Input:**
```json
{
  "weights": {
    "outcome_a": 0.0,
    "outcome_b": 0.0
  }
}
```

**Behavior:**
- Returns `valid: false`
- Includes error with code `ALL_ZERO_WEIGHTS`
- Cannot normalize (division by zero)

**Rationale:** Zero weights provide no guidance for decision-making.

---

### Edge Case: Node Not in Graph

**Input:**
```json
{
  "weights": {
    "nonexistent_node": 1.0
  }
}
```

**Behavior:**
- Returns `valid: false`
- Includes error with code `NODE_NOT_FOUND`
- Specifies which node is missing

**Rationale:** Cannot assign utility to nodes that don't exist in the graph.

---

### Edge Case: Conflicting Objectives

**Input:**
Graph where action "price_increase" positively affects "revenue" but negatively affects "customer_satisfaction".

**Behavior:**
- Returns `valid: true` (not an error)
- Includes info-level issue with code `CONFLICTING_OBJECTIVES`
- Explains which objectives conflict and why

**Rationale:** Conflicting objectives are common in real decisions. This is informational, not an error.

---

## Multi-Criteria Aggregation

### Edge Case: All Scores Uniform for a Criterion

**Input:**
```json
{
  "criterion_results": [
    {
      "criterion_id": "revenue",
      "option_scores": {
        "A": 100.0,
        "B": 100.0,
        "C": 100.0
      }
    }
  ]
}
```

**Behavior:**
- All options receive score `0.5` for that criterion (neutral)
- Logged as info message
- Does not cause error

**Implementation:**
```python
if max_score == min_score:
    # All options identical - assign neutral score
    for opt_id in option_scores.keys():
        normalized[opt_id][criterion_id] = 0.5
```

**Rationale:** When all options are equal on a criterion, that criterion provides no discriminatory power. Assigning 0.5 (neutral) ensures it doesn't bias the aggregation.

---

### Edge Case: Empty Criterion (No Options)

**Input:**
```json
{
  "criterion_results": [
    {
      "criterion_id": "revenue",
      "option_scores": {}
    }
  ]
}
```

**Behavior:**
- Criterion is skipped entirely
- Warning logged
- Aggregation continues with remaining criteria

**Rationale:** Empty criteria provide no information and should not block the entire aggregation.

---

### Edge Case: Missing Option in Some Criteria

**Input:**
```json
{
  "criterion_results": [
    {
      "criterion_id": "revenue",
      "option_scores": {"A": 100, "B": 80}
    },
    {
      "criterion_id": "cost",
      "option_scores": {"A": 50}
    }
  ]
}
```

**Behavior:**
- Option B receives score `0.0` for cost (worst possible)
- Effectively penalizes incomplete evaluations
- Documented in logs

**Rationale:** Missing scores indicate the option wasn't evaluated on that criterion, which should result in a penalty. Users should ensure complete data.

**Alternative Strategy (Future):** Could impute median score instead of zero via configuration flag.

---

### Edge Case: Single Criterion

**Input:**
```json
{
  "criterion_results": [
    {
      "criterion_id": "revenue",
      "weight": 1.0,
      "option_scores": {"A": 100, "B": 80}
    }
  ]
}
```

**Behavior:**
- Aggregation proceeds normally
- Rankings determined solely by that criterion
- No trade-offs identified (requires ≥2 criteria)

**Rationale:** Valid use case - multi-criteria framework should support single criterion.

---

### Edge Case: Criteria Weights Don't Sum to 1.0

**Input:**
```json
{
  "criterion_results": [
    {"criterion_id": "revenue", "weight": 0.6, ...},
    {"criterion_id": "cost", "weight": 0.6, ...}
  ]
}
```

**Behavior:**
- Weights are automatically normalized before aggregation
- Logged as warning
- Computation proceeds with normalized weights

**Implementation:**
```python
total_weight = sum(c.weight for c in criteria)
normalized_weights = {c.id: c.weight / total_weight for c in criteria}
```

**Rationale:** Pragmatic - normalize rather than reject. Users may not realize weights should sum to 1.0.

---

## Pareto Frontier

### Edge Case: All Options Identical

**Input:**
All options have identical scores across all criteria.

**Behavior:**
- All options are in Pareto frontier (none dominates any other)
- No dominated options
- `frontier_size = total_options`

**Rationale:** Correct behavior - when options are identical, none dominates.

---

### Edge Case: Single Option

**Input:**
```json
{
  "option_scores": {
    "A": {"revenue": 100, "cost": 50}
  }
}
```

**Behavior:**
- Frontier contains single option
- No dominated options
- `frontier_size = 1`

**Rationale:** Valid - single option is trivially Pareto-optimal.

---

### Edge Case: Linear Pareto Frontier

**Input:**
All options lie on the Pareto frontier (no dominated options).

**Behavior:**
- `pareto_frontier` contains all options
- `dominated_options` is empty
- Each point dominates no others

**Rationale:** Correct behavior - indicates strong trade-offs where each option excels on different criteria.

---

### Edge Case: Frontier Size Exceeds max_frontier_size

**Input:**
```json
{
  "option_scores": { /* 30 options, all on frontier */ },
  "max_frontier_size": 20
}
```

**Behavior:**
- Return top 20 by sum of normalized scores
- Remaining 10 moved to `dominated_options` with low `domination_degree`
- Documented in response

**Rationale:** Prevents overwhelming users with too many options. Progressive disclosure.

---

## Risk Adjustment

### Edge Case: Zero Standard Deviation (No Uncertainty)

**Input:**
```json
{
  "option_distributions": [
    {
      "option_id": "A",
      "mean": 100,
      "std_dev": 0.0,
      "p10": 100,
      "p50": 100,
      "p90": 100
    }
  ]
}
```

**Behavior:**
- Certainty equivalent = mean (no adjustment)
- Risk premium = 0
- Rankings unaffected by risk profile

**Rationale:** No uncertainty means no risk to adjust for.

---

### Edge Case: Negative Mean

**Input:**
```json
{
  "option_distributions": [
    {"option_id": "A", "mean": -50, "std_dev": 10, ...}
  ]
}
```

**Behavior:**
- Mean-variance adjustment proceeds normally
- CRRA utility function requires care (log/power of negative)
- CRRA implementation uses `max(samples, 0.001)` to avoid log(0)

**Rationale:** Negative utilities are valid (losses). Mean-variance handles them correctly.

---

### Edge Case: All Options Have Same Mean

**Input:**
All options have `mean = 50` but different `std_dev`.

**Behavior (Risk-Averse):**
- Rankings determined entirely by variance
- Lower-variance options ranked higher
- Risk premiums differ

**Behavior (Risk-Neutral):**
- All options ranked equally (no adjustment)
- Certainty equivalent = mean for all

**Rationale:** Correct behavior - risk-averse prefers certainty when means are equal.

---

## Threshold Detection

### Edge Case: No Ranking Changes

**Input:**
All sweep points have identical rankings.

**Behavior:**
- `thresholds` is empty list
- Sensitivity score = 0.0 for that parameter
- Parameter ranked as least sensitive

**Rationale:** Parameter has no effect on rankings within tested range.

---

### Edge Case: Ranking Changes at Every Step

**Input:**
Rankings change between every consecutive sweep point.

**Behavior:**
- Multiple thresholds detected
- Sensitivity score = 1.0 (maximum)
- Parameter ranked as most sensitive

**Rationale:** Indicates highly influential parameter - small changes cause ranking shifts.

---

### Edge Case: Single Sweep Point

**Input:**
```json
{
  "parameter_sweep_results": [
    {"parameter_value": 0.5, "rankings": ["A", "B"]}
  ]
}
```

**Behavior:**
- Returns error: "At least 2 sweep points required"
- Cannot detect thresholds with single point

**Rationale:** Need at least 2 points to detect a change.

---

### Edge Case: Identical Parameter Values

**Input:**
Two sweep results have same `parameter_value` but different rankings (data error).

**Behavior:**
- Logs warning about duplicate parameter values
- Uses first occurrence
- May affect threshold accuracy

**Rationale:** Data inconsistency - should be caught by caller (PLoT).

---

## General Principles

### 1. Graceful Degradation

ISL attempts to compute meaningful results even with imperfect inputs:
- Normalize weights automatically
- Skip empty criteria
- Handle uniform scores
- Provide partial results when possible

### 2. Informative Errors

When computation cannot proceed:
- Clear error codes
- Specific field references
- Suggested fixes when available

### 3. Logging Strategy

- **INFO**: Normal edge cases (uniform scores, normalization)
- **WARNING**: Potential data issues (missing options, empty criteria)
- **ERROR**: Cannot compute (validation failures)

### 4. Determinism

All edge case handling is deterministic:
- Same inputs → same outputs
- No randomness in edge case logic
- Reproducible results for debugging

---

## Testing Edge Cases

All edge cases are covered by unit tests:

```python
# Example edge case tests
def test_uniform_scores_assigned_neutral():
    """All options get 0.5 when scores are uniform."""
    ...

def test_missing_option_gets_zero_score():
    """Missing options receive worst score (0.0)."""
    ...

def test_zero_std_dev_no_adjustment():
    """Zero variance results in no risk adjustment."""
    ...
```

See `tests/unit/test_edge_cases.py` for complete test suite.
