# ISL API Quick Reference

## Base URLs
- **Staging:** `https://isl-staging.onrender.com`
- **Production:** `https://isl.olumi.com` (post-pilot)

## Authentication

All requests require API key in header:
```http
X-API-Key: your_api_key_here
```

**Request API Key:**
- Email: isl-team@olumi.com
- Include: Team name, use case, expected volume

---

## Endpoints

### 1. Causal Validation
**Validates whether a causal effect is identifiable from observational data.**

```http
POST /api/v1/causal/validate
Content-Type: application/json
X-API-Key: your_api_key

{
  "dag": {
    "nodes": ["X", "Y", "Z"],
    "edges": [["X", "Y"], ["Z", "Y"]]
  },
  "treatment": "X",
  "outcome": "Y"
}
```

**Response (200 OK):**
```json
{
  "status": "identifiable",
  "confidence": "high",
  "method": "backdoor",
  "formula": "P(Y|do(X)) = Σ_Z P(Y|X,Z)P(Z)",
  "adjustment_set": ["Z"],
  "required_assumptions": [
    {
      "name": "conditional_ignorability",
      "description": "Y ⊥ X | Z (Y independent of X given Z)",
      "evidence_strength": "strong",
      "testable": true,
      "consequences_if_violated": "Causal estimate will be biased"
    }
  ],
  "explanation": {
    "why_identifiable": "Z blocks all backdoor paths from X to Y",
    "graphical_criterion": "Backdoor criterion satisfied",
    "required_data": ["X", "Y", "Z"]
  },
  "_metadata": {
    "isl_version": "1.0.0",
    "request_id": "req_abc123",
    "config_fingerprint": "a1b2c3"
  }
}
```

**Status Values:**
- `identifiable` - Effect can be calculated from data
- `not_identifiable` - Cannot determine causal effect
- `conditional` - Identifiable if you control for certain variables

**Error Response (400 Bad Request):**
```json
{
  "detail": "Invalid DAG: cycle detected between nodes X and Y",
  "error_code": "INVALID_DAG",
  "request_id": "req_xyz789"
}
```

**Latency:** 100-300ms
**Rate Limit:** 100 req/min

---

### 2. Counterfactual Generation
**Simulates what would happen under hypothetical interventions.**

```http
POST /api/v1/causal/counterfactual
Content-Type: application/json
X-API-Key: your_api_key

{
  "causal_model": {
    "nodes": ["price", "demand", "revenue"],
    "edges": [["price", "demand"], ["demand", "revenue"]],
    "structural_equations": {
      "demand": "1000 - 10 * price",
      "revenue": "price * demand"
    }
  },
  "intervention": {"price": 55},
  "outcome_variables": ["revenue"],
  "baseline": {"price": 50},
  "samples": 1000
}
```

**Response (200 OK):**
```json
{
  "prediction": {
    "point_estimate": 105000,
    "confidence_interval": {
      "confidence_level": 0.95,
      "lower": 95000,
      "upper": 115000
    },
    "sensitivity_range": {
      "optimistic": 120000,
      "pessimistic": 90000,
      "explanation": "Range accounts for model parameter uncertainty"
    }
  },
  "causal_effect": {
    "absolute": 5000,
    "relative": 0.05,
    "interpretation": "Revenue increases by 5% (£5,000) under this intervention"
  },
  "robustness": {
    "score": "moderate",
    "critical_assumptions": [
      {
        "assumption": "Structural equation for revenue correctly specified",
        "confidence": "medium",
        "impact": 6750.0,
        "recommendation": "Validate revenue equation with additional data"
      }
    ]
  },
  "explanation": {
    "summary": "revenue likely increases to 105000 (range: 95000 to 115000)",
    "reasoning": "Under intervention price=55.0, revenue predicted to be 105000...",
    "technical_basis": "FACET region-based counterfactual analysis with Monte Carlo sampling",
    "assumptions": [
      "Structural equations correctly specified",
      "Prior distributions reflect true uncertainty",
      "No unmeasured confounders"
    ]
  },
  "_metadata": {
    "isl_version": "1.0.0",
    "request_id": "req_def456"
  }
}
```

**Parameters:**
- `intervention`: Variables to set (e.g., `{price: 55}`)
- `outcome_variables`: Which outcomes to predict
- `baseline`: Current state (optional, for computing causal effect)
- `samples`: Monte Carlo samples (default: 1000, max: 10000)

**Latency:** 500-3500ms (depends on samples)
**Rate Limit:** 50 req/min

---

### 3. Robustness Analysis
**Verifies whether a recommendation is robust to small changes.**

```http
POST /api/v1/robustness/analyze
Content-Type: application/json
X-API-Key: your_api_key

{
  "causal_model": {
    "nodes": ["price", "demand", "revenue"],
    "edges": [["price", "demand"], ["demand", "revenue"]]
  },
  "intervention_proposal": {"price": 55},
  "target_outcome": {"revenue": [95000, 105000]},
  "perturbation_radius": 0.1,
  "min_samples": 100
}
```

**Response (200 OK - Robust):**
```json
{
  "analysis": {
    "status": "robust",
    "robustness_score": 0.75,
    "is_fragile": false,
    "region_count": 2,
    "robust_regions": [
      {
        "variable_ranges": {
          "price": [52.0, 58.0]
        },
        "outcome_guarantees": {
          "revenue": [96000, 104000]
        },
        "volume": 0.12,
        "sample_count": 150
      }
    ],
    "interpretation": "ROBUST RECOMMENDATION. Multiple intervention strategies achieve target. Operating ranges well-defined.",
    "recommendation": "Proceed with confidence. Operating ranges: price 52-58 reliably achieves revenue 95k-105k",
    "sensitivity": {
      "most_sensitive_variable": "price",
      "least_sensitive_variable": null,
      "margin_to_fragility": 0.25
    }
  },
  "_metadata": {
    "isl_version": "1.0.0",
    "request_id": "req_ghi789"
  }
}
```

**Response (200 OK - Fragile):**
```json
{
  "analysis": {
    "status": "fragile",
    "robustness_score": 0.15,
    "is_fragile": true,
    "region_count": 0,
    "robust_regions": [],
    "interpretation": "FRAGILE RECOMMENDATION. Small changes break target outcome. High implementation risk.",
    "recommendation": "CAUTION: Only narrow intervention range works. Consider alternative strategies.",
    "fragility_warnings": [
      "No robust regions found within ±10% of proposal",
      "Outcome very sensitive to price changes",
      "Recommendation not reliable for real-world implementation"
    ]
  }
}
```

**Parameters:**
- `intervention_proposal`: Proposed intervention to test
- `target_outcome`: Desired outcome range (e.g., revenue 95k-105k)
- `perturbation_radius`: How far to explore (0.1 = ±10%)
- `min_samples`: Minimum samples per region (default: 100)

**Interpretation:**
- `robust` → Wide operating range, proceed with confidence
- `fragile` → Narrow range, high risk, reconsider

**Latency:** 1-5 seconds
**Rate Limit:** 20 req/min

---

### 4. Preference Elicitation
**Learns user values through strategic counterfactual questions (ActiVA).**

```http
POST /api/v1/preferences/elicit
Content-Type: application/json
X-API-Key: your_api_key

{
  "user_id": "user_123",
  "causal_model": {
    "nodes": ["marketing", "price", "revenue", "churn"],
    "edges": [...]
  },
  "context": {
    "revenue": 100000,
    "churn": 0.10,
    "satisfaction": 0.75
  },
  "feature_names": ["revenue", "churn", "satisfaction"],
  "max_queries": 10
}
```

**Response (200 OK - Eliciting):**
```json
{
  "status": "eliciting",
  "preference_id": "pref_abc123",
  "scenario_pair": {
    "scenario_a": {
      "revenue": 110000,
      "churn": 0.12,
      "satisfaction": 0.70
    },
    "scenario_b": {
      "revenue": 105000,
      "churn": 0.08,
      "satisfaction": 0.78
    }
  },
  "question": "Which scenario do you prefer?",
  "query_number": 3,
  "estimated_queries_remaining": 5
}
```

**Submit User Response:**
```http
POST /api/v1/preferences/submit
Content-Type: application/json
X-API-Key: your_api_key

{
  "user_id": "user_123",
  "preference_id": "pref_abc123",
  "choice": "B"
}
```

**Response (200 OK - Converged):**
```json
{
  "status": "converged",
  "preference_id": "pref_abc123",
  "value_function": {
    "weights": {
      "revenue": 0.45,
      "churn": 0.35,
      "satisfaction": 0.20
    },
    "confidence": 0.87,
    "uncertainty": {
      "revenue": 0.08,
      "churn": 0.12,
      "satisfaction": 0.15
    }
  },
  "num_queries": 7,
  "convergence_reason": "posterior_entropy_below_threshold",
  "recommendations": {
    "most_valued": "churn reduction",
    "least_valued": "satisfaction improvement",
    "trade_offs": "User willing to sacrifice 2% satisfaction for 1% churn reduction"
  }
}
```

**Workflow:**
1. Call `/elicit` → Get scenario pair
2. Show to user, collect choice (A or B)
3. Call `/submit` with choice → Get next pair OR converged status
4. Repeat until `status: "converged"`

**Latency:** 200-400ms
**Rate Limit:** 50 req/min

---

### 5. Team Deliberation
**Facilitates democratic deliberation toward shared understanding (Habermas Machine).**

```http
POST /api/v1/deliberation/deliberate
Content-Type: application/json
X-API-Key: your_api_key

{
  "decision_context": "Should we increase price to £55?",
  "positions": [
    {
      "member_id": "pm_001",
      "position_statement": "I support this because revenue would increase and robustness score gives confidence.",
      "timestamp": "2025-11-21T10:00:00Z"
    },
    {
      "member_id": "eng_001",
      "position_statement": "I'm concerned about implementation complexity and technical debt.",
      "timestamp": "2025-11-21T10:01:00Z"
    },
    {
      "member_id": "designer_001",
      "position_statement": "We should ensure user experience doesn't degrade with price change.",
      "timestamp": "2025-11-21T10:02:00Z"
    }
  ],
  "session_id": null
}
```

**Response (200 OK - Round 1):**
```json
{
  "session_id": "delib_abc123",
  "round_number": 1,
  "status": "active",
  "common_ground": {
    "shared_values": [
      {
        "value_name": "revenue_growth",
        "value_description": "Increasing company revenue",
        "agreement_score": 0.9,
        "supporting_members": ["pm_001", "eng_001"],
        "examples": ["Pricing strategy should drive revenue"]
      },
      {
        "value_name": "user_experience",
        "value_description": "Maintaining quality user experience",
        "agreement_score": 0.85,
        "supporting_members": ["designer_001", "eng_001"],
        "examples": ["Users should not experience degradation"]
      }
    ],
    "shared_concerns": [
      {
        "concern_description": "Implementation complexity and technical debt",
        "severity": "medium",
        "affected_members": ["eng_001"],
        "potential_mitigations": ["Phased rollout", "Technical review before implementation"]
      }
    ],
    "agreement_level": 0.75
  },
  "consensus_statement": {
    "statement_id": "cons_xyz789",
    "text": "We agree that revenue growth is important for the business, and we're committed to maintaining excellent user experience. While we support the price increase to £55, we want to ensure implementation is done carefully with a phased rollout to mitigate technical complexity and monitor user impact. We'll proceed with engineering review and gradual deployment.",
    "support_score": 0.82,
    "version": 1,
    "unresolved_disagreements": [],
    "generated_at": "2025-11-21T10:05:00Z"
  },
  "member_positions": [
    {
      "member_id": "pm_001",
      "values_extracted": [
        {
          "value_name": "revenue_growth",
          "weight": 0.8,
          "examples": ["revenue would increase"]
        }
      ],
      "concerns_extracted": [],
      "rationale": "Revenue increase with high confidence (robustness score)"
    },
    {
      "member_id": "eng_001",
      "values_extracted": [],
      "concerns_extracted": [
        {
          "concern_description": "Implementation complexity and technical debt",
          "severity": "medium",
          "conditions": "If rushed implementation"
        }
      ],
      "rationale": "Technical implementation concerns"
    },
    {
      "member_id": "designer_001",
      "values_extracted": [
        {
          "value_name": "user_experience",
          "weight": 0.9,
          "examples": ["user experience doesn't degrade"]
        }
      ],
      "concerns_extracted": [],
      "rationale": "Focus on maintaining UX quality"
    }
  ],
  "next_steps": [
    "Team members review consensus statement",
    "Submit edit suggestions if needed (POST /api/v1/deliberation/suggest-edits)",
    "Finalize consensus or conduct another round"
  ],
  "cost": {
    "llm_cost": 0.078,
    "session_total_cost": 0.078,
    "budget_remaining": 0.922
  }
}
```

**Continue Deliberation (Round 2):**
```http
POST /api/v1/deliberation/suggest-edits
Content-Type: application/json
X-API-Key: your_api_key

{
  "session_id": "delib_abc123",
  "edits": [
    {
      "member_id": "eng_001",
      "edit_text": "Add specific timeline: 'with a 3-week phased rollout'",
      "priority": "high",
      "rationale": "Need concrete timeline for planning"
    }
  ]
}
```

**LLM-Powered Features:**
- Value extraction from natural language positions
- Consensus statement generation (natural, not template)
- Concern identification and severity assessment
- **Fallback:** Rule-based if LLM unavailable/budget exceeded

**Cost:** ~$0.08 per deliberation round
**Latency:** 1-3 seconds
**Rate Limit:** 10 req/min (cost protection)

---

## Error Handling

All endpoints return standard error format:

```json
{
  "detail": "Error message describing what went wrong",
  "error_code": "SPECIFIC_ERROR_CODE",
  "request_id": "req_xyz789",
  "timestamp": "2025-11-21T10:00:00Z"
}
```

### Common Error Codes

| Code | Status | Description | Action |
|------|--------|-------------|--------|
| `INVALID_DAG` | 400 | DAG contains cycles or invalid structure | Fix DAG structure |
| `TREATMENT_NOT_IN_DAG` | 400 | Treatment variable not in nodes | Add variable to DAG |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Wait and retry |
| `INVALID_API_KEY` | 401 | API key missing or invalid | Check key |
| `LLM_BUDGET_EXCEEDED` | 402 | Session exceeded LLM budget | Increase budget or wait |
| `SERVICE_UNAVAILABLE` | 503 | ISL temporarily overloaded | Retry with backoff |
| `INTERNAL_ERROR` | 500 | Unexpected server error | Report to ISL team |

### HTTP Status Codes

- `200 OK` - Success
- `400 Bad Request` - Invalid input (check validation)
- `401 Unauthorized` - Invalid or missing API key
- `402 Payment Required` - LLM budget exceeded
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server-side issue
- `503 Service Unavailable` - Service overloaded

---

## Rate Limits

| Endpoint | Limit | Burst |
|----------|-------|-------|
| Causal Validation | 100/min | 20/sec |
| Counterfactual | 50/min | 10/sec |
| Robustness | 20/min | 5/sec |
| Preferences | 50/min | 10/sec |
| Deliberation | 10/min | 2/sec |

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1637490000
```

**When Rate Limited (429):**
```json
{
  "detail": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 30,
  "limit": 100,
  "window": "1 minute"
}
```

---

## Best Practices

### 1. Always Validate Causally First
```typescript
// ✓ Good: Validate before counterfactual
const validation = await isl.validateCausal({...});
if (validation.status === 'identifiable') {
  const cf = await isl.generateCounterfactual({...});
}

// ✗ Bad: Generate counterfactual without validation
const cf = await isl.generateCounterfactual({...});  // May be invalid!
```

### 2. Cache Deterministic Responses
```typescript
// Causal validation is deterministic for same DAG
const cacheKey = JSON.stringify({ dag, treatment, outcome });
const cached = cache.get(cacheKey);
if (cached) return cached;

const result = await isl.validateCausal({...});
cache.set(cacheKey, result);
```

### 3. Use Robustness for High-Stakes Decisions
```typescript
// For critical decisions, check robustness
const cf = await isl.generateCounterfactual({...});
const robustness = await isl.analyzeRobustness({
  intervention_proposal: intervention,
  target_outcome: { outcome: [cf.prediction.point_estimate * 0.95,
                                cf.prediction.point_estimate * 1.05] }
});

if (robustness.analysis.is_fragile) {
  console.warn('Recommendation not robust - reconsider!');
}
```

### 4. Monitor LLM Costs
```typescript
// Track deliberation costs
const delib = await isl.conductDeliberation({...});
console.log(`Cost: $${delib.cost.llm_cost}, Remaining: $${delib.cost.budget_remaining}`);

if (delib.cost.budget_remaining < 0.20) {
  console.warn('Budget low - consider increasing limit');
}
```

### 5. Handle Errors Gracefully
```typescript
try {
  const result = await isl.validateCausal({...});
} catch (error) {
  if (error.status === 429) {
    // Rate limited - retry after delay
    await sleep(error.retryAfter * 1000);
    return retry();
  } else if (error.status === 503) {
    // Service unavailable - exponential backoff
    await sleep(backoff);
    return retry();
  } else {
    // Other error - log and show user
    console.error(error);
    showError(error.message);
  }
}
```

---

## Support

- **Documentation:** https://docs.olumi.com/isl
- **Slack:** #isl-integration
- **Office Hours:** Wednesdays 2pm GMT
- **Email:** isl-team@olumi.com
- **On-Call:** isl-oncall@olumi.com (critical issues only)
- **Status Page:** https://status.olumi.com

## Changelog

- **v1.0.0** (2025-11-21): Production launch with LLM integration
- **v0.9.0** (2025-11-15): Added Habermas Machine deliberation
- **v0.8.0** (2025-11-10): Added ActiVA preference learning
- **v0.7.0** (2025-11-05): Added FACET robustness analysis
- **v0.6.0** (2025-11-01): Core Y₀ causal validation
