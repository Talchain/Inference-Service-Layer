# ISL Sequence Diagrams

Visual representations of PLoT↔ISL interactions for all major workflows.

## Table of Contents

- [Phase 1: Utility Validation](#phase-1-utility-validation)
- [Phase 2: Multi-Criteria Aggregation](#phase-2-multi-criteria-aggregation)
- [Phase 2: Pareto Frontier](#phase-2-pareto-frontier)
- [Phase 3: Risk Adjustment](#phase-3-risk-adjustment)
- [Phase 3: Threshold Detection](#phase-3-threshold-detection)

---

## Phase 1: Utility Validation

**Flow:** PLoT validates utility function before running inference.

```
┌──────────┐                           ┌──────────┐
│   PLoT   │                           │   ISL    │
└────┬─────┘                           └────┬─────┘
     │                                      │
     │ User submits decision                │
     │ with utility function                │
     │                                      │
     │ 1. Validate utility                  │
     │ POST /api/v1/utility/validate        │
     ├─────────────────────────────────────>│
     │ {                                    │
     │   "utility_function": {              │
     │     "weights": {                     │ Check weights sum to 1.0
     │       "revenue": 0.6,                │ Check nodes exist in graph
     │       "cost": 0.4                    │ Check for conflicts
     │     }                                │ Generate suggestions
     │   },                                 │
     │   "graph": {...}                     │
     │ }                                    │
     │                                      │
     │ 2. Validation result                 │
     │<─────────────────────────────────────┤
     │ {                                    │
     │   "valid": true,                     │
     │   "issues": [],                      │
     │   "suggestions": []                  │
     │ }                                    │
     │                                      │
     │ 3. Proceed with inference            │
     │ (utility is valid)                   │
     │                                      │
     ▼                                      ▼
```

**Error Case: Invalid Weights**

```
     │ 1. Validate utility                  │
     ├─────────────────────────────────────>│
     │ {                                    │
     │   "weights": {                       │ Weights sum to 1.2
     │     "revenue": 0.7,                  │ (not 1.0)
     │     "cost": 0.5                      │
     │   }                                  │
     │ }                                    │
     │                                      │
     │ 2. Validation FAILED                 │
     │<─────────────────────────────────────┤
     │ {                                    │
     │   "valid": false,                    │
     │   "issues": [{                       │
     │     "code": "WEIGHTS_NOT_NORMALIZED" │
     │   }],                                │
     │   "normalized_utility": {            │
     │     "weights": {                     │
     │       "revenue": 0.583,              │
     │       "cost": 0.417                  │
     │     }                                │
     │   }                                  │
     │ }                                    │
     │                                      │
     │ 3. Show error to user OR             │
     │    Auto-fix with normalized weights  │
     │                                      │
```

---

## Phase 2: Multi-Criteria Aggregation

**Flow:** PLoT runs inference per criterion, then calls ISL to aggregate.

```
┌──────────┐                           ┌──────────┐
│   PLoT   │                           │   ISL    │
└────┬─────┘                           └────┬─────┘
     │                                      │
     │ User requests multi-criteria         │
     │ decision analysis                    │
     │                                      │
     │ Step 1: Run inference for            │
     │         criterion "revenue"          │
     │ ─────────────────────────────        │
     │ run_inference(                       │
     │   utility={revenue: 1.0}             │
     │ )                                    │
     │ → Result: A=100, B=80, C=60          │
     │                                      │
     │ Step 2: Run inference for            │
     │         criterion "cost"             │
     │ ─────────────────────────────        │
     │ run_inference(                       │
     │   utility={cost: 1.0}                │
     │ )                                    │
     │ → Result: A=50, B=30, C=40           │
     │                                      │
     │ Step 3: Run inference for            │
     │         criterion "speed"            │
     │ ─────────────────────────────        │
     │ run_inference(                       │
     │   utility={speed: 1.0}               │
     │ )                                    │
     │ → Result: A=90, B=70, C=85           │
     │                                      │
     │ Step 4: Call ISL for aggregation     │
     │ POST /aggregation/multi-criteria     │
     ├─────────────────────────────────────>│
     │ {                                    │
     │   "criterion_results": [             │
     │     {                                │ Normalize scores 0-1
     │       "criterion_id": "revenue",     │ per criterion
     │       "weight": 0.5,                 │
     │       "direction": "maximize",       │ Flip minimize criteria
     │       "option_scores": {             │
     │         "A": 100, "B": 80, "C": 60   │ Apply weighted_sum:
     │       }                              │ score = Σ(w_i × norm_i)
     │     },                               │
     │     {                                │ Identify trade-offs
     │       "criterion_id": "cost",        │
     │       "weight": 0.3,                 │
     │       "direction": "minimize",       │
     │       "option_scores": {             │
     │         "A": 50, "B": 30, "C": 40    │
     │       }                              │
     │     },                               │
     │     {                                │
     │       "criterion_id": "speed",       │
     │       "weight": 0.2,                 │
     │       "direction": "maximize",       │
     │       "option_scores": {             │
     │         "A": 90, "B": 70, "C": 85    │
     │       }                              │
     │     }                                │
     │   ],                                 │
     │   "aggregation_method": "weighted_sum"│
     │ }                                    │
     │                                      │
     │ Step 5: Receive aggregated rankings  │
     │<─────────────────────────────────────┤
     │ {                                    │
     │   "aggregated_rankings": [           │
     │     {                                │
     │       "option_id": "A",              │
     │       "rank": 1,                     │
     │       "aggregated_score": 0.85,      │
     │       "scores_by_criterion": {       │
     │         "revenue": 1.0,              │
     │         "cost": 0.0,                 │
     │         "speed": 1.0                 │
     │       }                              │
     │     },                               │
     │     {                                │
     │       "option_id": "C",              │
     │       "rank": 2,                     │
     │       "aggregated_score": 0.68       │
     │     },                               │
     │     {                                │
     │       "option_id": "B",              │
     │       "rank": 3,                     │
     │       "aggregated_score": 0.51       │
     │     }                                │
     │   ],                                 │
     │   "trade_offs": [                    │
     │     {                                │
     │       "option_a": "A",               │
     │       "option_b": "B",               │
     │       "a_better_on": ["revenue", "speed"],│
     │       "b_better_on": ["cost"]        │
     │     }                                │
     │   ]                                  │
     │ }                                    │
     │                                      │
     │ Step 6: Enrich with narrative        │
     │         and present to user          │
     │                                      │
     ▼                                      ▼

Key Points:
- PLoT runs inference 3 times (once per criterion)
- Each inference uses single-criterion utility function
- ISL receives pre-computed scores, aggregates them
- ISL is purely computational, no inference
```

---

## Phase 2: Pareto Frontier

**Flow:** PLoT evaluates options on multiple criteria, ISL identifies non-dominated options.

```
┌──────────┐                           ┌──────────┐
│   PLoT   │                           │   ISL    │
└────┬─────┘                           └────┬─────┘
     │                                      │
     │ Step 1-3: Run inference per          │
     │           criterion (as above)       │
     │                                      │
     │ Step 4: Call ISL for Pareto analysis │
     │ POST /api/v1/analysis/pareto         │
     ├─────────────────────────────────────>│
     │ {                                    │
     │   "option_scores": {                 │
     │     "A": {"revenue": 100, "cost": 50},│ Build option matrix
     │     "B": {"revenue": 80, "cost": 30}, │
     │     "C": {"revenue": 60, "cost": 60}  │ Normalize scores
     │   },                                 │
     │   "criteria_directions": {           │ Check dominance:
     │     "revenue": "maximize",           │ A dominates B if:
     │     "cost": "minimize"               │   A >= B on all criteria
     │   },                                 │   A > B on ≥1 criterion
     │   "max_frontier_size": 20            │
     │ }                                    │ Use skyline algorithm
     │                                      │ (O(n log n))
     │                                      │
     │ Step 5: Receive Pareto frontier      │
     │<─────────────────────────────────────┤
     │ {                                    │
     │   "pareto_frontier": [               │
     │     {                                │
     │       "option_id": "A",              │
     │       "scores": {"revenue": 100, "cost": 50},│
     │       "dominates": ["C"]             │
     │     },                               │
     │     {                                │
     │       "option_id": "B",              │
     │       "scores": {"revenue": 80, "cost": 30},│
     │       "dominates": ["C"]             │
     │     }                                │
     │   ],                                 │
     │   "dominated_options": [             │
     │     {                                │
     │       "option_id": "C",              │
     │       "dominated_by": ["A", "B"],    │
     │       "domination_degree": 0.65      │
     │     }                                │
     │   ],                                 │
     │   "frontier_size": 2                 │
     │ }                                    │
     │                                      │
     │ Step 6: Present Pareto-optimal       │
     │         options to user              │
     │                                      │
     ▼                                      ▼

Key Points:
- A and B are on frontier (trade-off: A better revenue, B better cost)
- C is dominated by both A and B
- User chooses from frontier based on preferences
```

---

## Phase 3: Risk Adjustment

**Flow:** PLoT provides uncertainty distributions, ISL adjusts for risk attitude.

```
┌──────────┐                           ┌──────────┐
│   PLoT   │                           │   ISL    │
└────┬─────┘                           └────┬─────┘
     │                                      │
     │ Step 1: Run Monte Carlo simulation   │
     │ ──────────────────────────────────   │
     │ For each option, collect samples:    │
     │ Option A: [45, 52, 48, 51, ...]      │
     │ → p10=42, p50=50, p90=58, std=8      │
     │                                      │
     │ Option B: [20, 65, 35, 80, ...]      │
     │ → p10=15, p50=50, p90=85, std=25     │
     │                                      │
     │ Step 2: Call ISL for risk adjustment │
     │ POST /api/v1/analysis/risk-adjust    │
     ├─────────────────────────────────────>│
     │ {                                    │
     │   "option_distributions": [          │
     │     {                                │ Calculate certainty
     │       "option_id": "A",              │ equivalents:
     │       "mean": 50,                    │
     │       "std_dev": 8,                  │ Risk-averse:
     │       "p10": 42, "p50": 50, "p90": 58│ CE = mean - λ*var
     │     },                               │
     │     {                                │ Risk-neutral:
     │       "option_id": "B",              │ CE = mean
     │       "mean": 50,                    │
     │       "std_dev": 25,                 │ Risk-seeking:
     │       "p10": 15, "p50": 50, "p90": 85│ CE = mean + λ*var
     │     }                                │
     │   ],                                 │ Re-rank by CE
     │   "risk_profile": {                  │
     │     "type": "risk_averse",           │
     │     "preset": "conservative"         │
     │   }                                  │
     │ }                                    │
     │                                      │
     │ Step 3: Receive adjusted rankings    │
     │<─────────────────────────────────────┤
     │ {                                    │
     │   "adjusted_rankings": [             │
     │     {                                │
     │       "option_id": "A",              │
     │       "original_rank": 1,            │
     │       "adjusted_rank": 1,            │
     │       "original_expected": 50,       │
     │       "adjusted_expected": 44.4,     │
     │       "risk_premium": 5.6            │
     │     },                               │
     │     {                                │
     │       "option_id": "B",              │
     │       "original_rank": 1,            │
     │       "adjusted_rank": 2,            │
     │       "original_expected": 50,       │
     │       "adjusted_expected": 25.0,     │
     │       "risk_premium": 25.0           │
     │     }                                │
     │   ],                                 │
     │   "certainty_equivalents": {         │
     │     "A": 44.4,                       │
     │     "B": 25.0                        │
     │   }                                  │
     │ }                                    │
     │                                      │
     │ Step 4: Show rankings with           │
     │         risk-adjusted preferences    │
     │                                      │
     ▼                                      ▼

Key Points:
- Same mean (50), but B has higher variance
- Risk-averse user prefers A (lower variance)
- Risk premium shows how much certainty is worth
```

---

## Phase 3: Threshold Detection

**Flow:** PLoT sweeps parameter range, ISL analyzes where rankings change.

```
┌──────────┐                           ┌──────────┐
│   PLoT   │                           │   ISL    │
└────┬─────┘                           └────┬─────┘
     │                                      │
     │ Step 1: Parameter sweep              │
     │ ─────────────────────────────        │
     │ for price in [10, 15, 20, 25, 30]:  │
     │   run_inference(price)               │
     │                                      │
     │ Results:                             │
     │ price=10 → ranking=[A, B, C]         │
     │ price=15 → ranking=[A, B, C]         │
     │ price=20 → ranking=[B, A, C] ← change│
     │ price=25 → ranking=[B, A, C]         │
     │ price=30 → ranking=[B, C, A] ← change│
     │                                      │
     │ Step 2: Call ISL for threshold       │
     │         detection                    │
     │ POST /api/v1/analysis/thresholds     │
     ├─────────────────────────────────────>│
     │ {                                    │
     │   "parameter_sweep_results": [       │
     │     {                                │ Sort by param value
     │       "parameter_id": "price",       │
     │       "parameter_value": 10,         │ Find ranking changes
     │       "rankings": ["A", "B", "C"]    │
     │     },                               │ Estimate threshold
     │     {                                │ as midpoint
     │       "parameter_id": "price",       │
     │       "parameter_value": 15,         │ Calculate sensitivity
     │       "rankings": ["A", "B", "C"]    │ = changes / steps
     │     },                               │
     │     {                                │
     │       "parameter_id": "price",       │
     │       "parameter_value": 20,         │
     │       "rankings": ["B", "A", "C"]    │
     │     },                               │
     │     ...                              │
     │   ]                                  │
     │ }                                    │
     │                                      │
     │ Step 3: Receive thresholds           │
     │<─────────────────────────────────────┤
     │ {                                    │
     │   "thresholds": [                    │
     │     {                                │
     │       "parameter_id": "price",       │
     │       "threshold_value": 17.5,       │
     │       "ranking_before": ["A","B","C"],│
     │       "ranking_after": ["B","A","C"], │
     │       "affected_options": ["A", "B"], │
     │       "confidence": 0.9              │
     │     },                               │
     │     {                                │
     │       "parameter_id": "price",       │
     │       "threshold_value": 27.5,       │
     │       "ranking_before": ["B","A","C"],│
     │       "ranking_after": ["B","C","A"], │
     │       "affected_options": ["A", "C"], │
     │       "confidence": 0.9              │
     │     }                                │
     │   ],                                 │
     │   "sensitivity_ranking": ["price"]   │
     │ }                                    │
     │                                      │
     │ Step 4: Show critical thresholds     │
     │         to user                      │
     │ "If price > 17.5, choose B instead   │
     │  of A"                               │
     │                                      │
     ▼                                      ▼

Key Points:
- PLoT does expensive parameter sweep
- ISL analyzes sweep results to find critical values
- Thresholds indicate decision-changing parameter values
- Sensitivity ranking shows most influential parameters
```

---

## Architecture Principles

### 1. **Unidirectional Flow**

```
UI → PLoT → ISL
     ↑      ↓
     └──────┘
```

- UI calls PLoT only
- PLoT calls ISL only
- ISL never calls PLoT (no circular dependencies)

### 2. **PLoT Orchestrates, ISL Computes**

- **PLoT**: Runs inference, manages state, aggregates for UI
- **ISL**: Pure computation, stateless, deterministic

### 3. **Pre-computed Data**

ISL receives pre-computed results from PLoT:
- Criterion scores (from PLoT inference)
- Distributions (from PLoT Monte Carlo)
- Sweep results (from PLoT parameter variations)

ISL never triggers inference itself.

---

## Error Handling Flows

### Validation Error Flow

```
PLoT → ISL: Invalid request
ISL → PLoT: 400 Bad Request + error details
PLoT → UI: Show validation errors to user
```

### Computation Error Flow

```
PLoT → ISL: Valid request
ISL → PLoT: 500 Internal Error (computation failed)
PLoT: Retry with fallback OR
PLoT → UI: Show error, suggest retry
```

### Partial Success Flow

```
PLoT: Run 5 criteria evaluations
      3 succeed, 2 fail
PLoT → ISL: Send 3 successful results
ISL → PLoT: Aggregated rankings with warning
PLoT → UI: Show results + warning about missing criteria
```

---

## Performance Considerations

### Parallel Criterion Evaluation

```python
# In PLoT Engine
import asyncio

async def multi_criteria_analysis(criteria):
    # Evaluate all criteria in parallel (not sequential)
    tasks = [
        run_inference(utility={c.node_id: 1.0})
        for c in criteria
    ]

    results = await asyncio.gather(*tasks)

    # Then call ISL once for aggregation
    aggregated = await isl_client.post(
        "/aggregation/multi-criteria",
        json={"criterion_results": results}
    )
```

**Key Point:** PLoT parallelizes inference calls to minimize latency.

---

## Caching Strategy

### PLoT-Side Caching

```
PLoT checks cache before calling itself:
  If inference(graph, utility) in cache:
    Use cached result
  Else:
    Run inference
    Cache result (5 min TTL)

Then call ISL with results (ISL doesn't cache)
```

### ISL-Side Caching (Optional)

```
ISL can cache expensive computations:
  Pareto frontier for large option sets
  Threshold detection for many parameters

Cache key = hash(request_body)
TTL = 5 minutes
```

**Recommendation:** Start without ISL caching, add if needed for performance.
