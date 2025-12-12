# ISL API Examples

Complete working examples for all ISL endpoints.

---

## Setup

```bash
# Set your API key
export ISL_API_KEY="your_api_key_here"
export ISL_URL="http://localhost:8000"  # or https://isl-staging.onrender.com
```

---

## 1. Causal Validation

### Validate Assumptions

Check if a causal effect is identifiable from the DAG.

```bash
curl -X POST "$ISL_URL/api/v1/validation/assumptions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "dag": {
      "nodes": ["Marketing", "Price", "Brand", "Sales"],
      "edges": [
        ["Marketing", "Sales"],
        ["Price", "Sales"],
        ["Brand", "Marketing"],
        ["Brand", "Sales"]
      ]
    },
    "treatment": "Marketing",
    "outcome": "Sales"
  }'
```

**Response:**
```json
{
  "schema": "validation.v1",
  "is_valid": true,
  "identifiable": true,
  "adjustment_sets": [["Brand"], ["Brand", "Price"]],
  "minimal_adjustment_set": ["Brand"],
  "assumptions": [
    {
      "name": "unconfoundedness",
      "status": "satisfied",
      "confidence": 0.92,
      "details": "Backdoor criterion satisfied with adjustment set {Brand}"
    }
  ],
  "metadata": {
    "isl_version": "2.1.0",
    "request_id": "req_abc123"
  }
}
```

### Advanced Validation

Comprehensive model validation with quality scoring.

```bash
curl -X POST "$ISL_URL/api/v1/validation/validate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "dag": {
      "nodes": ["X", "Y", "Z"],
      "edges": [["X", "Z"], ["Y", "Z"]]
    },
    "validation_level": "COMPREHENSIVE"
  }'
```

---

## 2. Counterfactual Analysis

### Generate Counterfactuals

What would happen if we changed a variable?

```bash
curl -X POST "$ISL_URL/api/v1/counterfactual/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "model": {
      "dag": {
        "nodes": ["Marketing", "Sales"],
        "edges": [["Marketing", "Sales"]]
      },
      "treatment": "Marketing",
      "outcome": "Sales"
    },
    "intervention": {
      "Marketing": 50000
    },
    "baseline": {
      "Marketing": 30000,
      "Sales": 100000
    }
  }'
```

**Response:**
```json
{
  "schema": "counterfactual.v1",
  "counterfactual_outcome": {
    "Sales": 133333.33
  },
  "causal_effect": {
    "absolute": 33333.33,
    "relative": 0.333
  },
  "confidence_interval": {
    "lower": 125000,
    "upper": 142000
  },
  "metadata": {...}
}
```

### Goal-Seeking

Find what intervention achieves a target outcome.

```bash
curl -X POST "$ISL_URL/api/v1/counterfactual/goal-seek" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "model": {
      "dag": {
        "nodes": ["Marketing", "Price", "Sales"],
        "edges": [["Marketing", "Sales"], ["Price", "Sales"]]
      },
      "treatment": "Marketing",
      "outcome": "Sales"
    },
    "target_outcome": 150000,
    "constraints": {
      "Marketing": {"min": 0, "max": 100000},
      "Price": {"min": 10, "max": 100}
    }
  }'
```

---

## 3. Sensitivity Analysis

### Full Analysis

Test how robust conclusions are to assumption violations.

```bash
curl -X POST "$ISL_URL/api/v1/sensitivity/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "model": {
      "dag": {
        "nodes": ["Treatment", "Confounder", "Outcome"],
        "edges": [
          ["Treatment", "Outcome"],
          ["Confounder", "Treatment"],
          ["Confounder", "Outcome"]
        ]
      },
      "treatment": "Treatment",
      "outcome": "Outcome"
    },
    "data": [
      {"Treatment": 1, "Confounder": 0.5, "Outcome": 10},
      {"Treatment": 0, "Confounder": 0.3, "Outcome": 5}
    ],
    "assumptions": ["unconfoundedness", "positivity"]
  }'
```

**Response:**
```json
{
  "schema": "sensitivity.v1",
  "metrics": [
    {
      "assumption": "unconfoundedness",
      "elasticity": 0.15,
      "robustness_score": 0.87,
      "critical": false,
      "max_deviation_percent": 12.5,
      "interpretation": "Outcome changes by 15% for each 100% violation"
    }
  ],
  "overall_robustness": 0.85,
  "critical_count": 0,
  "metadata": {...}
}
```

---

## 4. Progressive Explanations

### Multi-Level Explanation

Get explanations at different complexity levels.

```bash
curl -X POST "$ISL_URL/api/v1/explanations/progressive" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "model": {
      "dag": {
        "nodes": ["Treatment", "Outcome"],
        "edges": [["Treatment", "Outcome"]]
      },
      "treatment": "Treatment",
      "outcome": "Outcome"
    },
    "concepts": ["treatment_effect", "confounding"],
    "effect_size": 0.20
  }'
```

**Response:**
```json
{
  "schema": "explanation.v1",
  "explanations": [
    {
      "concept": "treatment_effect",
      "levels": {
        "simple": "When we apply the treatment, the outcome changes by about 20%.",
        "intermediate": "The treatment has a causal effect on the outcome. We estimate a 20% change, meaning for every unit increase in treatment, the outcome increases by 20%.",
        "technical": "Under the assumptions of unconfoundedness and consistency, the Average Treatment Effect (ATE) is estimated at 0.20 with 95% CI [0.15, 0.25]."
      },
      "quality_scores": {
        "simple": {"flesch_reading_ease": 85, "grade_level": 4},
        "technical": {"flesch_reading_ease": 32, "grade_level": 16}
      }
    }
  ],
  "metadata": {...}
}
```

---

## 5. Causal Discovery

### Extract Factors from Text

Extract causal factors from unstructured text.

```bash
curl -X POST "$ISL_URL/api/v1/discovery/extract-factors" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "texts": [
      "Price increases led to demand reduction in Q3",
      "Marketing campaigns significantly boosted brand awareness",
      "Quality improvements resulted in higher customer satisfaction",
      "Competition from new entrants affected market share"
    ],
    "num_factors": 4
  }'
```

**Response:**
```json
{
  "schema": "discovery.v1",
  "factors": [
    {
      "name": "Price Sensitivity",
      "keywords": ["price", "demand", "reduction"],
      "strength": 0.82,
      "prevalence": 0.25,
      "representative_texts": ["Price increases led to demand reduction in Q3"]
    },
    {
      "name": "Marketing Impact",
      "keywords": ["marketing", "brand", "awareness"],
      "strength": 0.78,
      "prevalence": 0.25
    }
  ],
  "suggested_dag": {
    "nodes": ["Price Sensitivity", "Marketing Impact", "Quality", "Competition"],
    "edges": [
      ["Price Sensitivity", "Demand"],
      ["Marketing Impact", "Brand Awareness"]
    ],
    "confidence": 0.72
  },
  "metadata": {...}
}
```

---

## 6. Error Handling

### Handle Validation Errors

```bash
# Invalid DAG (cycle)
curl -X POST "$ISL_URL/api/v1/validation/assumptions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "dag": {
      "nodes": ["A", "B"],
      "edges": [["A", "B"], ["B", "A"]]
    },
    "treatment": "A",
    "outcome": "B"
  }'
```

**Response (400):**
```json
{
  "schema": "error.v1",
  "code": "VALIDATION_ERROR",
  "message": "DAG contains a cycle: A → B → A",
  "details": {
    "cycle": ["A", "B", "A"]
  },
  "retryable": false,
  "suggested_action": "Remove one of the edges to break the cycle"
}
```

### Handle Rate Limiting

```json
{
  "schema": "error.v1",
  "code": "RATE_LIMITED",
  "message": "Rate limit exceeded. 100 requests per minute allowed.",
  "retryable": true,
  "retry_after": 42
}
```

---

## 7. Python Client Examples

### Using the ISL Python Client

```python
from isl_client import ISLClient

client = ISLClient(
    base_url="https://isl-staging.onrender.com",
    api_key="your_api_key"
)

# Validate a model
result = client.validation.check_assumptions(
    dag={"nodes": ["A", "B"], "edges": [["A", "B"]]},
    treatment="A",
    outcome="B"
)
print(result.is_valid)

# Sensitivity analysis
sensitivity = client.sensitivity.analyze(
    model=my_model,
    data=my_data,
    assumptions=["unconfoundedness"]
)
print(f"Robustness: {sensitivity.overall_robustness}")
```

### Using requests library

```python
import requests

ISL_URL = "https://isl-staging.onrender.com"
API_KEY = "your_api_key"

response = requests.post(
    f"{ISL_URL}/api/v1/validation/assumptions",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    },
    json={
        "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
        "treatment": "A",
        "outcome": "B"
    }
)

result = response.json()
print(result["is_valid"])
```

---

## 8. TypeScript Examples

```typescript
const ISL_URL = "https://isl-staging.onrender.com";
const API_KEY = process.env.ISL_API_KEY;

async function validateModel() {
  const response = await fetch(`${ISL_URL}/api/v1/validation/assumptions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
    },
    body: JSON.stringify({
      dag: { nodes: ["A", "B"], edges: [["A", "B"]] },
      treatment: "A",
      outcome: "B",
    }),
  });

  const result = await response.json();
  console.log(result.is_valid);
}
```

---

## Tips

1. **Use request IDs** - Pass `X-Request-Id` header for tracing
2. **Handle timeouts** - Requests timeout after 60s, use simpler inputs if hitting limits
3. **Cache results** - ISL caches for 5 minutes, same request = cached response
4. **Check rate limits** - Monitor `X-RateLimit-Remaining` header

---

## 9. Constraint Feasibility Checking

### Check Option Feasibility

Validate which decision options satisfy business constraints.

```bash
curl -X POST "$ISL_URL/api/v1/validation/feasibility" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "constraints": [
      {
        "constraint_id": "budget",
        "label": "Budget Limit",
        "constraint_type": "budget",
        "attribute": "cost",
        "relation": "le",
        "threshold": 100000,
        "priority": "hard"
      },
      {
        "constraint_id": "min_quality",
        "label": "Minimum Quality",
        "constraint_type": "requirement",
        "attribute": "quality_score",
        "relation": "ge",
        "threshold": 7.0,
        "priority": "medium"
      }
    ],
    "options": [
      {"option_id": "premium", "attributes": {"cost": 80000, "quality_score": 9.5}},
      {"option_id": "standard", "attributes": {"cost": 50000, "quality_score": 7.2}},
      {"option_id": "budget", "attributes": {"cost": 30000, "quality_score": 5.0}},
      {"option_id": "deluxe", "attributes": {"cost": 150000, "quality_score": 9.8}}
    ],
    "include_partial_violations": true
  }'
```

**Response:**
```json
{
  "schema_version": "feasibility.v1",
  "constraint_validation": {
    "all_valid": true,
    "results": [
      {"constraint_id": "budget", "valid": true},
      {"constraint_id": "min_quality", "valid": true}
    ]
  },
  "feasibility": {
    "feasible_options": ["premium", "standard"],
    "infeasible_options": ["budget", "deluxe"],
    "option_results": [
      {"option_id": "premium", "feasible": true, "hard_violations": [], "soft_violations": []},
      {"option_id": "standard", "feasible": true, "hard_violations": [], "soft_violations": []},
      {"option_id": "budget", "feasible": false, "hard_violations": [], "soft_violations": ["min_quality"]},
      {"option_id": "deluxe", "feasible": false, "hard_violations": ["budget"], "soft_violations": []}
    ]
  },
  "warnings": []
}
```

---

## 10. Coherence Analysis

### Analyze Inference Result Coherence

Check if ranked options are stable and sensible.

```bash
curl -X POST "$ISL_URL/api/v1/validation/coherence" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "options": [
      {
        "option_id": "expand_market",
        "label": "Expand to New Market",
        "expected_value": 150000,
        "confidence_interval": {"lower": 120000, "upper": 180000}
      },
      {
        "option_id": "optimize_current",
        "label": "Optimize Current Operations",
        "expected_value": 140000,
        "confidence_interval": {"lower": 130000, "upper": 150000}
      },
      {
        "option_id": "reduce_costs",
        "label": "Cost Reduction Initiative",
        "expected_value": 80000,
        "confidence_interval": {"lower": 70000, "upper": 90000}
      }
    ],
    "perturbation_magnitude": 0.1,
    "num_perturbations": 100,
    "close_race_threshold": 0.15
  }'
```

**Response:**
```json
{
  "schema_version": "coherence.v1",
  "coherence_analysis": {
    "top_option_id": "expand_market",
    "top_option_positive": true,
    "margin_to_second": {
      "absolute": 10000,
      "percentage": 0.071
    },
    "is_close_race": true,
    "ranking_stability": "sensitive",
    "stability_score": 0.78,
    "warnings": [
      {
        "code": "CLOSE_RACE",
        "message": "Top two options are within 15% of each other",
        "severity": "medium"
      }
    ]
  },
  "stability_analysis": {
    "perturbations_run": 100,
    "ranking_changes": 22,
    "most_frequent_alternative": "optimize_current",
    "alternative_frequency": 0.22
  },
  "recommendations": [
    "Consider additional analysis to distinguish between top options",
    "The close race suggests collecting more data may change rankings"
  ]
}
```

---

## 11. Utility Function Validation

### Validate Multi-Goal Utility Specification

Validate utility function configuration for multi-criteria aggregation.

```bash
curl -X POST "$ISL_URL/api/v1/utility/validate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "utility_spec": {
      "goals": [
        {"goal_id": "revenue", "label": "Maximize Revenue", "direction": "maximize", "weight": 0.5},
        {"goal_id": "cost", "label": "Minimize Cost", "direction": "minimize", "weight": 0.3},
        {"goal_id": "satisfaction", "label": "Customer Satisfaction", "direction": "maximize", "weight": 0.2}
      ],
      "aggregation_method": "weighted_sum",
      "risk_tolerance": "risk_neutral"
    }
  }'
```

**Response:**
```json
{
  "schema_version": "utility.v1",
  "valid": true,
  "normalised_weights": {
    "revenue": 0.5,
    "cost": 0.3,
    "satisfaction": 0.2
  },
  "normalised_goals": [
    {"goal_id": "revenue", "label": "Maximize Revenue", "direction": "maximize", "normalised_weight": 0.5, "original_weight": 0.5},
    {"goal_id": "cost", "label": "Minimize Cost", "direction": "minimize", "normalised_weight": 0.3, "original_weight": 0.3},
    {"goal_id": "satisfaction", "label": "Customer Satisfaction", "direction": "maximize", "normalised_weight": 0.2, "original_weight": 0.2}
  ],
  "aggregation_method": "weighted_sum",
  "risk_tolerance": "risk_neutral",
  "default_behaviour_applied": [],
  "warnings": [],
  "errors": []
}
```

### Validate with Default Weights

```bash
curl -X POST "$ISL_URL/api/v1/utility/validate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "utility_spec": {
      "goals": [
        {"goal_id": "profit", "label": "Profit"},
        {"goal_id": "growth", "label": "Growth"},
        {"goal_id": "risk", "label": "Risk", "direction": "minimize"}
      ],
      "aggregation_method": "weighted_sum"
    }
  }'
```

**Response (with default equal weighting):**
```json
{
  "valid": true,
  "normalised_weights": {
    "profit": 0.333,
    "growth": 0.333,
    "risk": 0.333
  },
  "default_behaviour_applied": ["Equal weighting applied (no weights specified)"],
  "warnings": [
    {"code": "DEFAULT_WEIGHTS", "message": "No weights specified - applying equal weights (0.333 each)"}
  ]
}
```

---

## 12. Correlation Group Validation

### Validate Factor Correlations

Validate correlation specifications for Monte Carlo sampling.

```bash
curl -X POST "$ISL_URL/api/v1/validation/correlations" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "correlation_groups": [
      {
        "group_id": "market_conditions",
        "factors": ["demand", "competition"],
        "correlation": 0.7,
        "label": "Market factors tend to move together"
      },
      {
        "group_id": "cost_factors",
        "factors": ["labor_cost", "material_cost", "energy_cost"],
        "correlation": 0.5,
        "label": "Cost components are moderately correlated"
      }
    ],
    "check_positive_definite": true
  }'
```

**Response:**
```json
{
  "schema_version": "correlation.v1",
  "valid": true,
  "validated_groups": [
    {"group_id": "market_conditions", "is_valid": true, "correlation": 0.7, "issues": []},
    {"group_id": "cost_factors", "is_valid": true, "correlation": 0.5, "issues": []}
  ],
  "implied_matrix": {
    "factors": ["demand", "competition", "labor_cost", "material_cost", "energy_cost"],
    "matrix": [
      [1.0, 0.7, 0.0, 0.0, 0.0],
      [0.7, 1.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.5, 0.5],
      [0.0, 0.0, 0.5, 1.0, 0.5],
      [0.0, 0.0, 0.5, 0.5, 1.0]
    ]
  },
  "matrix_analysis": {
    "is_positive_semi_definite": true,
    "min_eigenvalue": 0.27,
    "condition_number": 3.7,
    "suggested_regularization": null
  },
  "warnings": [],
  "errors": []
}
```

### Validate with Direct Matrix Input

```bash
curl -X POST "$ISL_URL/api/v1/validation/correlations" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "correlation_matrix": {
      "factors": ["demand", "price", "competition"],
      "matrix": [
        [1.0, -0.6, 0.7],
        [-0.6, 1.0, -0.3],
        [0.7, -0.3, 1.0]
      ]
    },
    "check_positive_definite": true
  }'
```

---

## 13. Continuous Optimization

### Optimize Decision Variables

Find optimal values for continuous decision variables.

```bash
curl -X POST "$ISL_URL/api/v1/analysis/optimise" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "objective": {
      "variable_id": "profit",
      "direction": "maximize",
      "coefficients": {"price": 1000, "quantity": -5},
      "constant": -50000
    },
    "decision_variables": [
      {"variable_id": "price", "lower_bound": 10, "upper_bound": 100},
      {"variable_id": "quantity", "lower_bound": 0, "upper_bound": 500}
    ],
    "constraints": [
      {
        "constraint_id": "capacity",
        "coefficients": {"quantity": 1},
        "relation": "le",
        "rhs": 400
      },
      {
        "constraint_id": "min_price",
        "coefficients": {"price": 1},
        "relation": "ge",
        "rhs": 20
      }
    ],
    "grid_points": 20,
    "confidence_level": 0.95
  }'
```

**Response:**
```json
{
  "schema_version": "optimise.v1",
  "optimal_point": {
    "variable_values": {
      "price": 100.0,
      "quantity": 0.0
    },
    "objective_value": 50000.0,
    "confidence_interval": {
      "lower": 45100.0,
      "upper": 54900.0,
      "confidence_level": 0.95
    },
    "is_boundary": true,
    "boundary_variables": ["price", "quantity"],
    "feasible": true
  },
  "sensitivity": {
    "range_within_5pct": {
      "price": [95.0, 100.0],
      "quantity": [0.0, 25.0]
    },
    "gradient_at_optimum": {
      "price": 1000.0,
      "quantity": -5.0
    },
    "robustness": "fragile",
    "robustness_score": 0.15,
    "critical_variables": ["price"]
  },
  "grid_metrics": {
    "grid_points_evaluated": 400,
    "feasible_points": 380,
    "computation_time_ms": 125.5,
    "convergence_achieved": true
  },
  "warnings": [
    {
      "code": "BOUNDARY_OPTIMUM",
      "message": "Optimal point is at variable bounds: ['price', 'quantity']. Consider expanding bounds.",
      "affected_variables": ["price", "quantity"]
    }
  ]
}
```

### Optimization with Budget Constraint

```bash
curl -X POST "$ISL_URL/api/v1/analysis/optimise" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "objective": {
      "variable_id": "revenue",
      "direction": "maximize",
      "coefficients": {"marketing": 2.5, "sales_staff": 1.8}
    },
    "decision_variables": [
      {"variable_id": "marketing", "lower_bound": 0, "upper_bound": 100000},
      {"variable_id": "sales_staff", "lower_bound": 0, "upper_bound": 50000}
    ],
    "constraints": [
      {
        "constraint_id": "total_budget",
        "coefficients": {"marketing": 1.0, "sales_staff": 1.0},
        "relation": "le",
        "rhs": 80000
      }
    ],
    "grid_points": 15
  }'
```

**Response:**
```json
{
  "optimal_point": {
    "variable_values": {
      "marketing": 80000.0,
      "sales_staff": 0.0
    },
    "objective_value": 200000.0,
    "is_boundary": true
  },
  "sensitivity": {
    "gradient_at_optimum": {"marketing": 2.5, "sales_staff": 1.8},
    "robustness": "moderate",
    "critical_variables": ["marketing"]
  },
  "warnings": [
    {"code": "CONSTRAINT_ACTIVE", "message": "Constraints active at optimum: ['total_budget']"}
  ]
}
```

---

## 14. Y₀ Identifiability Analysis

### Check Effect Identifiability (GraphV1 Format)

Determine if a causal effect is identifiable from observational data using the Y₀ algorithm.

**HARD RULE:** If the decision→goal effect is non-identifiable, recommendations are marked as "exploratory" (not "actionable").

```bash
curl -X POST "$ISL_URL/api/v1/analysis/identifiability" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "graph": {
      "nodes": [
        {"id": "price", "kind": "decision", "label": "Pricing Strategy"},
        {"id": "market_segment", "kind": "factor", "label": "Market Segment"},
        {"id": "revenue", "kind": "goal", "label": "Revenue Target"}
      ],
      "edges": [
        {"from": "price", "to": "revenue", "weight": 2.0},
        {"from": "market_segment", "to": "price", "weight": 1.5},
        {"from": "market_segment", "to": "revenue", "weight": 1.0}
      ]
    }
  }'
```

**Response (Identifiable - Actionable):**
```json
{
  "schema_version": "identifiability.v1",
  "identifiability": {
    "effect": "price → revenue",
    "identifiable": true,
    "method": "backdoor",
    "adjustment_set": ["market_segment"],
    "confidence": "high",
    "explanation": "The effect of 'price' on 'revenue' is identifiable using the backdoor criterion. Adjust for: market_segment."
  },
  "recommendation_status": "actionable",
  "recommendation_caveat": null,
  "suggestions": null,
  "backdoor_paths": ["price → market_segment → revenue"]
}
```

### Check Non-Identifiable Effect

When an effect is non-identifiable, the hard rule enforces exploratory status:

```bash
curl -X POST "$ISL_URL/api/v1/analysis/identifiability" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "graph": {
      "nodes": [
        {"id": "decision", "kind": "decision", "label": "Decision"},
        {"id": "other", "kind": "outcome", "label": "Other Outcome"},
        {"id": "goal", "kind": "goal", "label": "Goal"}
      ],
      "edges": [
        {"from": "decision", "to": "other", "weight": 1.0}
      ]
    }
  }'
```

**Response (Non-Identifiable - Exploratory):**
```json
{
  "schema_version": "identifiability.v1",
  "identifiability": {
    "effect": "decision → goal",
    "identifiable": false,
    "method": "non_identifiable",
    "adjustment_set": null,
    "confidence": "high",
    "explanation": "No causal path exists from 'decision' to 'goal'. The decision cannot affect the goal in this model."
  },
  "recommendation_status": "exploratory",
  "recommendation_caveat": "No causal connection exists between decision and goal. Any recommendations would be meaningless.",
  "suggestions": [
    {
      "description": "Add direct or mediated causal path from 'decision' to 'goal'",
      "edges_to_add": [["decision", "goal"]],
      "priority": "critical"
    },
    {
      "description": "Verify that 'decision' actually causally affects 'goal' in the real-world domain",
      "priority": "critical"
    }
  ],
  "backdoor_paths": null
}
```

### Simple DAG Format

Alternative endpoint for simple DAG structures:

```bash
curl -X POST "$ISL_URL/api/v1/analysis/identifiability/dag" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ISL_API_KEY" \
  -d '{
    "nodes": ["X", "Y", "Z"],
    "edges": [["X", "Y"], ["Z", "X"], ["Z", "Y"]],
    "treatment": "X",
    "outcome": "Y"
  }'
```

**Response:**
```json
{
  "schema_version": "identifiability.v1",
  "identifiability": {
    "effect": "X → Y",
    "identifiable": true,
    "method": "backdoor",
    "adjustment_set": ["Z"],
    "confidence": "high",
    "explanation": "The effect of 'X' on 'Y' is identifiable using the backdoor criterion. Adjust for: Z."
  },
  "recommendation_status": "actionable",
  "recommendation_caveat": null,
  "suggestions": null,
  "backdoor_paths": ["X → Z → Y"]
}
```

### Identification Methods

The Y₀ algorithm supports multiple identification methods:

| Method | Description | Example |
|--------|-------------|---------|
| `backdoor` | Block all backdoor paths by conditioning | Adjust for confounders |
| `frontdoor` | Use mediating variables when backdoor blocked | Mechanism-based identification |
| `instrumental` | Use variables affecting X but not Y directly | Natural experiments |
| `do_calculus` | General graphical rules | Complex identification |
| `non_identifiable` | No identification method found | Exploratory recommendations |

### Hard Rule Summary

| Identifiability | recommendation_status | recommendation_caveat |
|-----------------|----------------------|----------------------|
| `true` | `actionable` | `null` |
| `false` | `exploratory` | Warning message (required) |
