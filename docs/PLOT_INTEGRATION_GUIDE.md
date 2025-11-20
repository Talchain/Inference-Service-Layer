# PLoT Integration Guide

**Inference Service Layer (ISL) Integration for Product-Led Onboarding & Teaching (PLoT)**

**Version**: 1.0
**Last Updated**: 2025-11-20
**Status**: Pilot Ready ✅

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Integration Patterns](#integration-patterns)
4. [API Endpoints](#api-endpoints)
5. [Common Workflows](#common-workflows)
6. [Error Handling](#error-handling)
7. [Performance Considerations](#performance-considerations)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- ISL deployed and accessible (default: `http://localhost:8000`)
- User ID management (PLoT user identifiers)
- Decision context available (domain, variables, constraints)

### Basic Integration Flow

```python
import httpx

# Initialize client
client = httpx.AsyncClient(base_url="http://localhost:8000")

# 1. Check service health
health = await client.get("/health")
print(f"ISL Status: {health.json()['status']}")

# 2. Elicit user preferences
response = await client.post("/api/v1/preferences/elicit", json={
    "user_id": "plot_user_123",
    "context": {
        "domain": "pricing",
        "variables": ["revenue", "churn", "brand_perception"]
    },
    "num_queries": 3
})

queries = response.json()["queries"]

# 3. Present queries to user and collect responses
# ... (PLoT UI integration)

# 4. Update beliefs based on user response
await client.post("/api/v1/preferences/update", json={
    "user_id": "plot_user_123",
    "query_id": queries[0]["id"],
    "response": "A",  # User chose scenario A
    "confidence": 0.8
})
```

**Estimated Integration Time**: 2-4 hours for basic flow

---

## Core Concepts

### 1. User Preferences as Bayesian Beliefs

ISL models user preferences as probability distributions over values:

```json
{
  "value_weights": {
    "revenue": {"type": "normal", "parameters": {"mean": 0.7, "std": 0.2}},
    "churn": {"type": "normal", "parameters": {"mean": 0.3, "std": 0.2}}
  },
  "risk_tolerance": {"type": "beta", "parameters": {"alpha": 2, "beta": 2}},
  "uncertainty_estimates": {
    "revenue_weight": 0.4,
    "churn_weight": 0.6
  }
}
```

**Why this matters**: Captures uncertainty about user preferences, enabling adaptive questioning.

### 2. Counterfactual Queries (ActiVA)

Queries present two scenarios with different trade-offs:

```json
{
  "id": "query_001",
  "question": "Which outcome would you prefer?",
  "scenario_a": {
    "description": "High revenue, moderate churn",
    "outcomes": {"revenue": 100000, "churn": 0.08},
    "trade_offs": ["Higher revenue", "More customer loss"]
  },
  "scenario_b": {
    "description": "Moderate revenue, low churn",
    "outcomes": {"revenue": 80000, "churn": 0.03},
    "trade_offs": ["Lower revenue", "Better retention"]
  },
  "information_gain": 0.32
}
```

**Why this matters**: Each query maximizes learning about user priorities.

### 3. Causal Models for Decision Support

ISL uses Directed Acyclic Graphs (DAGs) to represent causal relationships:

```
Price → Revenue
Brand → Price
Brand → Revenue
Churn → Revenue
```

**Why this matters**: Enables "what-if" analysis grounded in causal reasoning.

---

## Integration Patterns

### Pattern 1: Progressive Preference Elicitation

**Use Case**: Onboarding new user, learn preferences over 3-5 questions

```python
async def onboard_user(user_id: str, domain: str):
    """Progressive preference elicitation during onboarding."""

    # Step 1: Generate initial queries (cold start)
    response = await client.post("/api/v1/preferences/elicit", json={
        "user_id": user_id,
        "context": {
            "domain": domain,
            "variables": ["revenue", "churn", "satisfaction"]
        },
        "num_queries": 3
    })

    queries = response.json()["queries"]
    strategy = response.json()["strategy"]

    # Step 2: Present queries one at a time
    for i, query in enumerate(queries):
        # Show query to user via PLoT UI
        user_choice = await present_query_ui(query)

        # Update beliefs immediately
        await client.post("/api/v1/preferences/update", json={
            "user_id": user_id,
            "query_id": query["id"],
            "response": user_choice["answer"],  # "A" or "B"
            "confidence": user_choice["confidence"]  # 0.0-1.0
        })

        # Check if more queries needed
        if i < len(queries) - 1:
            # Optionally: get refined queries based on new beliefs
            pass

    return {"status": "onboarding_complete", "queries_answered": len(queries)}
```

**Key Points**:
- Start with 3 queries for quick onboarding
- Update beliefs after each answer for adaptive flow
- `strategy.type` indicates confidence level (e.g., "exploration" vs "uncertainty_sampling")

### Pattern 2: Contextual Preference Adaptation

**Use Case**: User enters new decision context, adapt existing beliefs

```python
async def adapt_to_new_context(user_id: str, new_context: dict):
    """Adapt user preferences to new decision context."""

    # Get current beliefs from previous interactions
    current_beliefs = await get_stored_beliefs(user_id)

    # Generate context-specific queries
    response = await client.post("/api/v1/preferences/elicit", json={
        "user_id": user_id,
        "context": new_context,
        "current_beliefs": current_beliefs,  # Warm start
        "num_queries": 2  # Fewer queries since we have prior
    })

    queries = response.json()["queries"]
    estimated_remaining = response.json()["estimated_queries_remaining"]

    # If estimated_remaining <= 2, beliefs are well-calibrated
    if estimated_remaining <= 2:
        return {"status": "beliefs_sufficient", "queries": queries}
    else:
        return {"status": "more_learning_needed", "queries": queries}
```

**Key Points**:
- Provide `current_beliefs` for warm start
- Check `estimated_queries_remaining` to gauge confidence
- Fewer queries needed when transferring from similar context

### Pattern 3: Causal "What-If" Analysis

**Use Case**: User wants to understand impact of decision options

```python
async def analyze_decision_impact(dag: dict, intervention: dict, outcome: str):
    """Analyze causal impact of decision on outcome."""

    # Step 1: Validate causal model
    validation = await client.post("/api/v1/causal/validate", json={
        "dag": dag,
        "treatment": list(intervention.keys())[0],
        "outcome": outcome
    })

    if validation.json()["status"] != "identifiable":
        return {"error": "Cannot identify causal effect", "details": validation.json()}

    # Step 2: Build structural model (simplified example)
    structural_model = {
        "variables": dag["nodes"],
        "equations": {
            "Revenue": "10000 + 500 * Price - 200 * Churn",
            "Churn": "0.05 + 0.001 * Price"
        },
        "distributions": {
            "Price": {"type": "normal", "parameters": {"mean": 50, "std": 5}}
        }
    }

    # Step 3: Run counterfactual analysis
    result = await client.post("/api/v1/causal/counterfactual", json={
        "model": structural_model,
        "intervention": intervention,  # e.g., {"Price": 60}
        "outcome": outcome,
        "num_samples": 1000
    })

    cf_data = result.json()
    return {
        "point_estimate": cf_data["prediction"]["point_estimate"],
        "confidence_interval": cf_data["prediction"]["confidence_interval"],
        "uncertainty": cf_data["uncertainty"]["overall"],
        "robustness": cf_data["robustness"]["level"],
        "explanation": cf_data["explanation"]["summary"]
    }
```

**Key Points**:
- Always validate DAG before counterfactual analysis
- Provide structural equations for your domain
- Use `num_samples=1000` for production (good accuracy/speed trade-off)

### Pattern 4: Teaching with Examples

**Use Case**: Explain causal concepts to user with pedagogical examples

```python
async def teach_causal_concept(user_id: str, concept: str, user_beliefs: dict):
    """Generate teaching examples tailored to user understanding."""

    response = await client.post("/api/v1/teaching/teach", json={
        "user_id": user_id,
        "current_beliefs": user_beliefs,
        "target_concept": concept,  # e.g., "confounding", "trade_offs"
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn"]
        },
        "max_examples": 3
    })

    examples = response.json()["examples"]

    for example in examples:
        # Present example to user via PLoT UI
        await show_teaching_example(
            scenario=example["scenario"],
            explanation=example["explanation"],
            pedagogical_notes=example["pedagogical_focus"]
        )

    return {
        "concept": concept,
        "examples_shown": len(examples),
        "time_estimate": response.json()["estimated_time_minutes"]
    }
```

**Key Points**:
- Target specific concepts: "confounding", "trade_offs", "uncertainty", "causal_paths"
- Examples ranked by pedagogical value
- `estimated_time_minutes` helps with UX pacing

---

## API Endpoints

### Core Endpoints for PLoT Integration

#### 1. Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-11-20T12:00:00Z"
}
```

**Use**: Service health monitoring, integration tests

#### 2. Preference Elicitation

```http
POST /api/v1/preferences/elicit
```

**Request**:
```json
{
  "user_id": "plot_user_123",
  "context": {
    "domain": "pricing",
    "variables": ["revenue", "churn", "satisfaction"],
    "constraints": {
      "industry": "SaaS",
      "current_price": 49
    }
  },
  "current_beliefs": null,
  "num_queries": 3
}
```

**Response**:
```json
{
  "queries": [
    {
      "id": "query_001",
      "question": "Which outcome would you prefer?",
      "scenario_a": {
        "description": "High revenue, moderate churn",
        "outcomes": {"revenue": 100000, "churn": 0.08},
        "trade_offs": ["Higher revenue", "More customer loss"]
      },
      "scenario_b": {
        "description": "Moderate revenue, low churn",
        "outcomes": {"revenue": 80000, "churn": 0.03},
        "trade_offs": ["Lower revenue", "Better retention"]
      },
      "information_gain": 0.32
    }
  ],
  "strategy": {
    "type": "exploration",
    "rationale": "Initial preference learning - exploring value space",
    "focus_areas": ["revenue vs churn trade-off"]
  },
  "estimated_queries_remaining": 4,
  "explanation": {
    "summary": "Generating queries to learn your priorities between revenue and churn",
    "reasoning": "These queries explore different trade-offs to maximize information gain"
  }
}
```

**Performance**: ~28ms P50, ~41ms P95 (see PERFORMANCE_REPORT.md)

#### 3. Preference Update

```http
POST /api/v1/preferences/update
```

**Request**:
```json
{
  "user_id": "plot_user_123",
  "query_id": "query_001",
  "response": "A",
  "confidence": 0.8
}
```

**Response**:
```json
{
  "updated_beliefs": {
    "value_weights": {
      "revenue": {"type": "normal", "parameters": {"mean": 0.75, "std": 0.15}},
      "churn": {"type": "normal", "parameters": {"mean": 0.25, "std": 0.15}}
    },
    "uncertainty_estimates": {
      "revenue_weight": 0.3,
      "churn_weight": 0.35
    }
  },
  "learning_progress": {
    "entropy_reduction": 0.18,
    "queries_answered": 1,
    "estimated_queries_remaining": 3
  },
  "insights": {
    "top_priorities": ["revenue", "churn"],
    "confidence_level": "medium",
    "recommendations_ready": false
  }
}
```

**Performance**: ~50-100ms P50 (Bayesian update computation)

#### 4. Causal Validation

```http
POST /api/v1/causal/validate
```

**Request**:
```json
{
  "dag": {
    "nodes": ["Price", "Revenue", "Brand", "Churn"],
    "edges": [
      ["Price", "Revenue"],
      ["Brand", "Price"],
      ["Brand", "Revenue"],
      ["Churn", "Revenue"]
    ]
  },
  "treatment": "Price",
  "outcome": "Revenue"
}
```

**Response**:
```json
{
  "status": "identifiable",
  "confidence": "high",
  "adjustment_sets": [["Brand"], ["Brand", "Churn"]],
  "explanation": {
    "summary": "Price → Revenue effect is identifiable",
    "reasoning": "By adjusting for Brand, we can isolate the causal effect of Price on Revenue",
    "assumptions": [
      "No unmeasured confounders",
      "Causal Markov assumption holds"
    ]
  }
}
```

**Performance**: ~4.4ms P50 (see PERFORMANCE_REPORT.md)

#### 5. Counterfactual Analysis

```http
POST /api/v1/causal/counterfactual
```

**Request**:
```json
{
  "model": {
    "variables": ["Price", "Revenue", "Churn"],
    "equations": {
      "Revenue": "10000 + 500 * Price - 200 * Churn",
      "Churn": "0.05 + 0.001 * Price"
    },
    "distributions": {
      "Price": {"type": "normal", "parameters": {"mean": 50, "std": 5}}
    }
  },
  "intervention": {"Price": 60},
  "outcome": "Revenue",
  "num_samples": 1000
}
```

**Response**:
```json
{
  "prediction": {
    "point_estimate": 38000,
    "confidence_interval": {"lower": 35000, "upper": 41000, "level": 0.95}
  },
  "uncertainty": {
    "overall": "medium",
    "sources": [
      {"type": "structural", "contribution": 0.6},
      {"type": "parametric", "contribution": 0.3},
      {"type": "distributional", "contribution": 0.1}
    ]
  },
  "robustness": {
    "level": "high",
    "score": 0.85,
    "sensitivity_ranges": [
      {
        "assumption": "Price elasticity",
        "current_value": 500,
        "range": {"min": 400, "max": 600}
      }
    ]
  },
  "explanation": {
    "summary": "Increasing price to $60 would increase revenue by ~$5,000",
    "reasoning": "Higher price directly increases revenue, but also increases churn",
    "key_factors": ["Price elasticity", "Churn sensitivity"]
  }
}
```

**Performance**: ~1.5s P95 (Monte Carlo with 1000 samples)

#### 6. Teaching Examples

```http
POST /api/v1/teaching/teach
```

**Request**:
```json
{
  "user_id": "plot_user_123",
  "current_beliefs": {
    "value_weights": {
      "revenue": {"type": "normal", "parameters": {"mean": 0.7, "std": 0.2}}
    },
    "risk_tolerance": {"type": "beta", "parameters": {"alpha": 2, "beta": 2}},
    "time_horizon": {"type": "normal", "parameters": {"mean": 12, "std": 3}},
    "uncertainty_estimates": {"revenue_weight": 0.3}
  },
  "target_concept": "trade_offs",
  "context": {
    "domain": "pricing",
    "variables": ["revenue", "churn"]
  },
  "max_examples": 3
}
```

**Response**:
```json
{
  "examples": [
    {
      "id": "example_001",
      "concept": "trade_offs",
      "scenario": {
        "description": "Premium pricing strategy",
        "outcomes": {"revenue": 120000, "churn": 0.12},
        "explanation": "Higher prices increase revenue but also increase customer churn"
      },
      "pedagogical_focus": "Illustrates revenue-churn trade-off",
      "difficulty": "beginner",
      "estimated_time": 2
    }
  ],
  "learning_objectives": ["Understand trade-offs", "Recognize causal relationships"],
  "estimated_time_minutes": 5,
  "next_steps": "Practice identifying trade-offs in your domain"
}
```

**Performance**: ~500ms P95 (example generation + ranking)

---

## Common Workflows

### Workflow 1: Full Onboarding Sequence

```
┌─────────────┐
│ User starts │
│  onboarding │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Generate 3 initial  │
│ queries (cold start)│ POST /api/v1/preferences/elicit
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Present query 1     │
│ to user via PLoT UI │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ User chooses        │
│ scenario A or B     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Update beliefs      │ POST /api/v1/preferences/update
│ based on response   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Check: queries      │
│ remaining <= 1?     │
└──────┬──────────────┘
       │
       ├─ No ──► Repeat with next query
       │
       ▼ Yes
┌─────────────────────┐
│ Onboarding complete │
│ Preferences learned │
└─────────────────────┘
```

**Expected Time**: 2-3 minutes (3 queries × 30-60 seconds each)

### Workflow 2: Decision Analysis with Causal Model

```
┌─────────────┐
│ User wants  │
│ to evaluate │
│  a decision │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Build/retrieve DAG  │
│ for decision domain │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Validate causal     │ POST /api/v1/causal/validate
│ model structure     │
└──────┬──────────────┘
       │
       ├─ Not identifiable ──► Show error, suggest improvements
       │
       ▼ Identifiable
┌─────────────────────┐
│ Run counterfactual  │ POST /api/v1/causal/counterfactual
│ for each option     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Present results:    │
│ - Point estimates   │
│ - Confidence ranges │
│ - Key assumptions   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ User makes informed │
│ decision            │
└─────────────────────┘
```

**Expected Time**: 5-10 seconds per option analyzed

---

## Error Handling

### HTTP Status Codes

| Status | Meaning | Action |
|--------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad Request | Check request format, validate inputs |
| 422 | Validation Error | Check Pydantic schema violations |
| 500 | Internal Error | Retry with exponential backoff, check ISL logs |

### Common Errors

#### 1. Empty Context Variables

```json
{
  "error": "Validation error",
  "detail": "context.variables must contain at least one variable"
}
```

**Solution**: Ensure `context.variables` has 1+ items

#### 2. DAG Not Identifiable

```json
{
  "status": "cannot_identify",
  "explanation": {
    "summary": "Cannot identify Price → Revenue effect",
    "reasoning": "Unmeasured confounders present",
    "suggestions": ["Add missing variables", "Check causal assumptions"]
  }
}
```

**Solution**: Add missing confounders to DAG or revise causal model

#### 3. User Beliefs Not Found

```json
{
  "error": "User beliefs not found",
  "detail": "No stored beliefs for user_id=plot_user_123"
}
```

**Solution**: Provide `current_beliefs` explicitly or run preference elicitation first

### Retry Strategy

```python
import asyncio

async def api_call_with_retry(func, max_retries=3, backoff_base=2):
    """Exponential backoff retry for API calls."""
    for attempt in range(max_retries):
        try:
            return await func()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 500 and attempt < max_retries - 1:
                wait_time = backoff_base ** attempt
                await asyncio.sleep(wait_time)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries")
```

---

## Performance Considerations

### Latency Guidelines

Based on PERFORMANCE_REPORT.md (validated with 119/119 passing tests):

| Endpoint | P50 Latency | P95 Latency | Target | Notes |
|----------|-------------|-------------|--------|-------|
| Health | 2.9ms | 3.3ms | N/A | Always fast |
| Causal validation | 4.4ms | ~5.3ms | <2.0s | 400x better than target |
| Preference elicitation | 28ms | 41ms | <1.5s | 50x better than target |
| Preference update | ~50ms | ~100ms | <1.5s | Bayesian update |
| Counterfactual | ~1.0s | ~1.5s | <2.0s | Monte Carlo simulation |
| Teaching | ~300ms | ~500ms | <1.5s | Example generation |

### Optimization Tips

1. **Cache Preference Queries**: For identical contexts, queries are deterministic
   ```python
   cache_key = hash(f"{user_id}_{context}_{num_queries}")
   ```

2. **Reduce Counterfactual Samples**: Use 500 samples for faster response (vs 1000)
   - 500 samples: ~750ms, adequate accuracy
   - 1000 samples: ~1.5s, higher precision

3. **Batch User Updates**: Update multiple users' beliefs in parallel
   ```python
   tasks = [update_beliefs(user) for user in users]
   await asyncio.gather(*tasks)
   ```

4. **Pre-warm DAG Validation**: Validate DAGs once, cache results
   - DAG validation: ~4ms per call
   - Caching saves repeated validation

### Concurrency Limits

**Pilot Phase** (10-25 concurrent users):
- ✅ Validated: System handles 25 concurrent users comfortably
- Expected throughput: 350-875 req/s
- P95 latency: <100ms for most endpoints

**Production Phase** (100+ concurrent users):
- ⚠️ Requires load testing
- Recommend: Multi-instance deployment + Redis cluster
- Target throughput: 3500+ req/s

---

## Testing & Validation

### Integration Test Checklist

```python
# Test 1: Health check
response = await client.get("/health")
assert response.status_code == 200
assert response.json()["status"] == "healthy"

# Test 2: Preference elicitation
response = await client.post("/api/v1/preferences/elicit", json={
    "user_id": "test_user",
    "context": {"domain": "pricing", "variables": ["revenue", "churn"]},
    "num_queries": 3
})
assert response.status_code == 200
assert len(response.json()["queries"]) == 3

# Test 3: Causal validation
response = await client.post("/api/v1/causal/validate", json={
    "dag": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
    "treatment": "X",
    "outcome": "Y"
})
assert response.status_code == 200
assert response.json()["status"] == "identifiable"
```

### End-to-End Test

```python
async def test_full_workflow():
    """Test complete PLoT integration workflow."""
    user_id = "e2e_test_user"

    # 1. Elicit preferences
    queries_response = await client.post("/api/v1/preferences/elicit", json={
        "user_id": user_id,
        "context": {"domain": "pricing", "variables": ["revenue", "churn"]},
        "num_queries": 2
    })
    assert queries_response.status_code == 200
    query_id = queries_response.json()["queries"][0]["id"]

    # 2. Update beliefs
    update_response = await client.post("/api/v1/preferences/update", json={
        "user_id": user_id,
        "query_id": query_id,
        "response": "A",
        "confidence": 0.8
    })
    assert update_response.status_code == 200
    beliefs = update_response.json()["updated_beliefs"]

    # 3. Generate teaching example
    teaching_response = await client.post("/api/v1/teaching/teach", json={
        "user_id": user_id,
        "current_beliefs": beliefs,
        "target_concept": "trade_offs",
        "context": {"domain": "pricing", "variables": ["revenue", "churn"]},
        "max_examples": 1
    })
    assert teaching_response.status_code == 200
    assert len(teaching_response.json()["examples"]) >= 1

    return "✅ End-to-end workflow validated"
```

---

## Troubleshooting

### Issue 1: Slow Response Times

**Symptom**: Endpoints taking >2s to respond

**Diagnosis**:
1. Check counterfactual `num_samples` parameter (reduce from 1000 to 500)
2. Verify Redis connection (belief storage/retrieval)
3. Check network latency between PLoT and ISL

**Solution**:
```python
# Reduce samples for faster response
response = await client.post("/api/v1/causal/counterfactual", json={
    # ...
    "num_samples": 500  # Instead of 1000
})
```

### Issue 2: Queries Not Adapting to User Responses

**Symptom**: Subsequent queries seem random, not personalized

**Diagnosis**: Beliefs not being updated or stored correctly

**Solution**:
1. Verify `/api/v1/preferences/update` returns 200
2. Check `updated_beliefs` in response
3. Provide `current_beliefs` in next elicitation request

```python
# Correct pattern: pass updated beliefs forward
beliefs = update_response.json()["updated_beliefs"]

next_queries = await client.post("/api/v1/preferences/elicit", json={
    "user_id": user_id,
    "context": context,
    "current_beliefs": beliefs,  # ← Important!
    "num_queries": 2
})
```

### Issue 3: DAG Validation Fails

**Symptom**: `status: "cannot_identify"`

**Diagnosis**: Missing confounders or incorrect DAG structure

**Solution**:
1. Review `explanation.suggestions` in response
2. Add missing variables to DAG
3. Verify edge directions match causal flow

```python
# Before (incorrect):
dag = {
    "nodes": ["Price", "Revenue"],
    "edges": [["Revenue", "Price"]]  # Wrong direction!
}

# After (correct):
dag = {
    "nodes": ["Price", "Revenue", "Brand"],  # Added confounder
    "edges": [
        ["Price", "Revenue"],
        ["Brand", "Price"],
        ["Brand", "Revenue"]
    ]
}
```

---

## Next Steps

### For PLoT Integration Team

1. **Week 1**: Basic integration (health check + preference elicitation)
2. **Week 2**: Full workflow (elicitation → update → teaching)
3. **Week 3**: Causal analysis integration (validation → counterfactual)
4. **Week 4**: Pilot testing with 5-10 users

### For CEE Integration

See separate CEE integration documentation for:
- Event stream integration
- Belief persistence patterns
- Multi-user concurrency

### Support

- **API Documentation**: `/docs` (interactive Swagger UI)
- **Performance Report**: `benchmarks/PERFORMANCE_REPORT.md`
- **Test Suite**: 119/119 passing tests validate correctness
- **GitHub Issues**: [ISL Repository Issues](https://github.com/Talchain/Inference-Service-Layer/issues)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Validation Status**: ✅ Pilot Ready (119/119 tests passing)
