# API Documentation

Complete API reference for the Inference Service Layer.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required (Phase 0). Future versions will support API keys and OAuth2.

## Common Response Fields

All endpoints include an `explanation` field with:

- `summary`: One-line summary
- `reasoning`: Plain English explanation
- `technical_basis`: Mathematical/technical justification
- `assumptions`: Key assumptions made

## Endpoints

---

## Health Check

### `GET /health`

Returns service health status.

**Response: 200 OK**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-01-19T10:30:00Z"
}
```

---

## Causal Validation

### `POST /api/v1/causal/validate`

Validates whether a causal model (DAG) supports causal identification for a treatment-outcome pair.

**Request Body**

```json
{
  "dag": {
    "nodes": ["Price", "Brand", "Revenue", "CustomerAcquisition"],
    "edges": [
      ["Price", "Revenue"],
      ["Brand", "Price"],
      ["Brand", "Revenue"],
      ["CustomerAcquisition", "Revenue"]
    ]
  },
  "treatment": "Price",
  "outcome": "Revenue",
  "observed": ["Brand"]  // Optional
}
```

**Response: 200 OK (Identifiable)**

```json
{
  "status": "identifiable",
  "adjustment_sets": [["Brand"], ["Brand", "CustomerAcquisition"]],
  "minimal_set": ["Brand"],
  "backdoor_paths": ["Price ← Brand → Revenue"],
  "confidence": "high",
  "explanation": {
    "summary": "Effect is identifiable by controlling for Brand",
    "reasoning": "Brand influences both Price and Revenue, creating confounding. Controlling for Brand blocks the backdoor path and isolates the causal effect.",
    "technical_basis": "Backdoor criterion satisfied with adjustment set {Brand}",
    "assumptions": [
      "No unmeasured confounding",
      "Correct causal structure specified",
      "Causal sufficiency (all relevant variables included)"
    ]
  }
}
```

**Response: 200 OK (Uncertain)**

```json
{
  "status": "uncertain",
  "issues": [
    {
      "type": "missing_connection",
      "description": "CustomerAcquisition may confound the Price-Revenue relationship",
      "affected_nodes": ["CustomerAcquisition", "Price", "Revenue"],
      "suggested_action": "Clarify: Does CustomerAcquisition affect Price? If yes, add edge."
    }
  ],
  "confidence": "low",
  "explanation": {...}
}
```

**Error Responses**

- `400 Bad Request`: Invalid DAG structure, missing nodes, or cycles detected
- `500 Internal Server Error`: Computation error

---

## Counterfactual Analysis

### `POST /api/v1/causal/counterfactual`

Analyzes what would happen under a counterfactual intervention.

**Request Body**

```json
{
  "model": {
    "variables": ["Price", "Brand", "Revenue"],
    "equations": {
      "Brand": "baseline_brand + 0.3 * Price",
      "Revenue": "10000 + 500 * Price - 200 * Brand"
    },
    "distributions": {
      "baseline_brand": {
        "type": "normal",
        "parameters": {"mean": 50, "std": 5}
      }
    }
  },
  "intervention": {"Price": 15},
  "outcome": "Revenue",
  "context": {"baseline_brand": 52}  // Optional
}
```

**Response: 200 OK**

```json
{
  "scenario": {
    "intervention": {"Price": 15},
    "outcome": "Revenue",
    "context": {"baseline_brand": 52}
  },
  "prediction": {
    "point_estimate": 51000,
    "confidence_interval": {
      "lower": 45000,
      "upper": 55000,
      "confidence_level": 0.95
    },
    "sensitivity_range": {
      "optimistic": 62000,
      "pessimistic": 38000,
      "explanation": "Range accounts for uncertainty in competitive response"
    }
  },
  "uncertainty": {
    "overall": "medium",
    "sources": [
      {
        "factor": "Brand Perception Lag",
        "impact": 3000,
        "confidence": "medium",
        "explanation": "Brand changes typically take 2-4 weeks to affect revenue",
        "basis": "Historical data from 3 previous price changes"
      },
      {
        "factor": "Competitive Response",
        "impact": 4000,
        "confidence": "low",
        "explanation": "Competitors may adjust prices in response",
        "basis": "Market structure analysis, no direct historical data"
      }
    ]
  },
  "robustness": {
    "score": "moderate",
    "critical_assumptions": [
      {
        "assumption": "Customer price sensitivity remains constant",
        "impact": 15000,
        "confidence": "medium",
        "recommendation": "Consider A/B testing price sensitivity before full rollout"
      }
    ]
  },
  "explanation": {...}
}
```

**Supported Distribution Types**

- `normal`: Parameters: `mean`, `std`
- `uniform`: Parameters: `min`, `max`
- `beta`: Parameters: `alpha`, `beta`
- `exponential`: Parameters: `scale`

**Equation Syntax**

Equations support:
- Basic arithmetic: `+`, `-`, `*`, `/`
- NumPy functions: `sqrt`, `exp`, `log`, `abs`
- Variable references
- Constants

Example: `"Revenue": "baseline + 500*Price - 0.1*Price*Price + np.sqrt(Market)"`

---

## Team Alignment

### `POST /api/v1/team/align`

Finds common ground across team perspectives and recommends aligned options.

**Request Body**

```json
{
  "perspectives": [
    {
      "role": "Product Manager",
      "priorities": ["User acquisition", "Revenue growth", "Fast time-to-market"],
      "constraints": ["Limited budget", "Q4 deadline"],
      "preferred_options": ["option_a", "option_b"]
    },
    {
      "role": "Designer",
      "priorities": ["User experience", "Brand consistency", "Accessibility"],
      "constraints": ["Design system limitations"],
      "preferred_options": ["option_b", "option_c"]
    },
    {
      "role": "Engineer",
      "priorities": ["Code quality", "Maintainability", "Tech debt reduction"],
      "constraints": ["Team capacity"],
      "preferred_options": ["option_c"]
    }
  ],
  "options": [
    {
      "id": "option_a",
      "name": "Quick MVP launch",
      "attributes": {"speed": "fast", "quality": "medium"}
    },
    {
      "id": "option_b",
      "name": "Polished feature set",
      "attributes": {"speed": "medium", "quality": "high"}
    },
    {
      "id": "option_c",
      "name": "Refactor first",
      "attributes": {"speed": "slow", "quality": "high"}
    }
  ]
}
```

**Response: 200 OK**

```json
{
  "common_ground": {
    "shared_goals": ["Deliver value to users", "Meet Q4 deadline"],
    "shared_constraints": ["Limited budget"],
    "agreement_level": 72
  },
  "aligned_options": [
    {
      "option": "option_b",
      "satisfies_roles": ["Product Manager", "Designer", "Engineer"],
      "satisfaction_score": 85,
      "tradeoffs": [
        {
          "role": "Product Manager",
          "gives": "Fastest time-to-market",
          "gets": "Higher quality leading to better retention"
        }
      ]
    }
  ],
  "conflicts": [
    {
      "between": ["Product Manager", "Engineer"],
      "about": "Speed vs quality trade-off",
      "severity": "moderate",
      "suggestion": "Compromise: Launch polished MVP, schedule refactor for Q1"
    }
  ],
  "recommendation": {
    "top_option": "option_b",
    "rationale": "This option satisfies 85% of stated priorities across all roles...",
    "confidence": "high",
    "next_steps": [
      "Define MVP scope collaboratively",
      "Schedule Q1 tech debt sprint",
      "Establish quality checkpoints"
    ]
  },
  "explanation": {...}
}
```

---

## Sensitivity Analysis

### `POST /api/v1/analysis/sensitivity`

Tests how robust conclusions are to changes in assumptions.

**Request Body**

```json
{
  "model": {
    "variables": ["Price", "Revenue"],
    "equations": {"Revenue": "10000 + sensitivity * Price"},
    "distributions": {
      "sensitivity": {"type": "normal", "parameters": {"mean": 500, "std": 50}}
    }
  },
  "baseline_result": 51000,
  "assumptions": [
    {
      "name": "Customer price sensitivity",
      "current_value": 0.5,
      "type": "parametric",
      "variation_range": {"min": 0.3, "max": 0.8}
    },
    {
      "name": "Competitor response timing",
      "current_value": "90 days",
      "type": "structural"
    }
  ]
}
```

**Response: 200 OK**

```json
{
  "conclusion": {
    "statement": "Revenue increases by £45k-£55k per month",
    "base_case": 51000
  },
  "assumptions": [
    {
      "name": "Customer price sensitivity",
      "current_value": 0.5,
      "importance": "critical",
      "impact": {
        "if_wrong": 15000,
        "percentage": 58
      },
      "confidence": "medium",
      "evidence": "Based on historical data from 3 price changes",
      "recommendation": "CRITICAL: Validate with A/B test before full rollout"
    },
    {
      "name": "Competitor response timing",
      "current_value": "90 days",
      "importance": "moderate",
      "impact": {
        "if_wrong": 8000,
        "percentage": 31
      },
      "confidence": "low",
      "evidence": "Market structure analysis, limited historical data",
      "recommendation": "Consider validating if possible to reduce uncertainty"
    }
  ],
  "robustness": {
    "overall": "moderate",
    "summary": "Conclusion holds unless price sensitivity is substantially different than assumed",
    "breakpoints": [
      {
        "assumption": "Customer price sensitivity",
        "threshold": "If sensitivity > 0.75, revenue impact becomes negative"
      }
    ]
  },
  "explanation": {...}
}
```

**Assumption Types**

- `parametric`: Numerical parameter (e.g., coefficient value)
- `structural`: Structural form (e.g., equation specification)
- `distributional`: Distribution assumption

---

## Error Responses

All error responses follow this structure:

```json
{
  "error_code": "invalid_dag_structure",
  "message": "DAG contains cycles",
  "details": {"cycle": ["A", "B", "C", "A"]},
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "retryable": false,
  "suggested_action": "fix_input"
}
```

**Error Codes**

- `invalid_dag_structure`: DAG validation failed
- `invalid_structural_model`: Model specification invalid
- `validation_error`: Request validation failed
- `y0_library_error`: Y₀ computation error
- `facet_computation_error`: FACET analysis error
- `computation_error`: General computation error

---

## Rate Limiting

Currently no rate limiting (Phase 0). Future versions will implement rate limits.

## Versioning

API version is included in URL path: `/api/v1/...`

Breaking changes will increment the version number.

## Changelog

### Version 0.1.0 (2025-01-19)

- Initial release
- Causal validation endpoint
- Counterfactual analysis endpoint
- Team alignment endpoint
- Sensitivity analysis endpoint
- Health check endpoint

---

## Support

For API questions or issues, please open an issue on GitHub or contact the development team.
