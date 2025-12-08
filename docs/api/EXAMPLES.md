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
