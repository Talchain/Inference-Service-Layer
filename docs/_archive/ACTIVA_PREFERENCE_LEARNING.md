# ActiVA Preference Learning - Integration Guide

## Overview

ISL's preference elicitor uses **ActiVA (Active Value Alignment)** to efficiently learn user values through strategic counterfactual comparisons. This guide explains how to integrate ActiVA into your application to understand and adapt to user preferences.

## Table of Contents

- [Key Benefits](#key-benefits)
- [How It Works](#how-it-works)
- [API Workflow](#api-workflow)
- [Integration Examples](#integration-examples)
- [Interpreting Results](#interpreting-results)
- [Best Practices](#best-practices)
- [Metrics & Monitoring](#metrics--monitoring)

---

## Key Benefits

### Efficiency
- **5-10 questions** vs 15-20 with generic questioning
- Each question is strategically chosen to maximize information gain
- Converges to accurate preference model faster

### Informativeness
- Questions are ranked by **expected information gain**
- Uses information theory: `EIG = H(current beliefs) - E[H(posterior beliefs)]`
- Focuses queries on areas of highest uncertainty

### Realism
- Scenarios derived from **causal counterfactuals**
- Realistic trade-offs based on your domain's causal structure
- Questions feel relevant and grounded in reality

### Transparency
- Track confidence and convergence in real-time
- Understand learning progress through `learning_summary`
- Know when enough information has been gathered

---

## How It Works

ActiVA learns user preferences through three core steps:

### 1. Belief Representation

User values are modeled as a **Bayesian belief distribution**:

```
UserBeliefModel:
  - value_weights: P(w | data)  # Distribution over feature importance
  - risk_tolerance: P(r | data)  # Distribution over risk preferences
  - uncertainty_estimates: Map[feature, confidence]  # How certain we are
```

**Example:** "We believe the user values revenue with 70% weight (±20% uncertainty)"

### 2. Strategic Query Generation

ActiVA generates **counterfactual scenario pairs** that maximize information gain:

```
For each candidate pair (A, B):
  1. Compute: current_entropy = H(current beliefs)
  2. Simulate: "What if user prefers A?" → P(w | A chosen)
  3. Simulate: "What if user prefers B?" → P(w | B chosen)
  4. Calculate: EIG = current_entropy - E[posterior_entropy]
  5. Rank by EIG (higher = more informative)
```

**Example:** A query asking about "high revenue, high risk" vs "low revenue, low risk" has high EIG when risk tolerance is uncertain.

### 3. Bayesian Belief Updating

When the user answers, beliefs are updated using **Bayes' rule**:

```
P(weights | user choice) ∝ P(user choice | weights) × P(weights)
                           └─ likelihood ─┘           └─ prior ─┘
```

**Example:** User chooses low-risk option → increase weight on safety features, decrease uncertainty

After 5-10 iterations, confidence converges and the system is ready for personalized recommendations.

---

## API Workflow

### Step 1: Initialize Preference Learning

**Endpoint:** `POST /api/v1/preferences/elicit`

**First Request (No Prior Beliefs):**
```json
{
  "user_id": "user-12345",
  "context": {
    "domain": "pricing",
    "variables": ["revenue", "churn", "brand_perception"],
    "constraints": {
      "industry": "SaaS",
      "current_price": 49
    }
  },
  "num_queries": 3
}
```

**Response:**
```json
{
  "queries": [
    {
      "id": "query_001",
      "question": "Which outcome would you prefer?\n\n**Option A:** Price increases by 10% (conservative approach)\n  - revenue: 52,500\n  - churn: 0.055\n  - brand_perception: -0.5\n\nTrade-offs: Moderate price increase, Limited side effects\n\n**Option B:** Price increases by 10% (aggressive approach)\n  - revenue: 54,000\n  - churn: 0.070\n  - brand_perception: -3.5\n\nTrade-offs: Significant price increase, Notable side effects",
      "scenario_a": {
        "description": "Conservative approach",
        "outcomes": {
          "revenue": 52500,
          "churn": 0.055,
          "brand_perception": -0.5
        },
        "trade_offs": ["Moderate price increase", "Limited side effects"]
      },
      "scenario_b": {
        "description": "Aggressive approach",
        "outcomes": {
          "revenue": 54000,
          "churn": 0.070,
          "brand_perception": -3.5
        },
        "trade_offs": ["Significant price increase", "Notable side effects"]
      },
      "information_gain": 0.42
    }
  ],
  "strategy": {
    "type": "uncertainty_sampling",
    "rationale": "High uncertainty - focusing on areas where your preferences are least clear",
    "focus_areas": ["All preference dimensions"]
  },
  "expected_information_gain": 1.26,
  "estimated_queries_remaining": 5,
  "explanation": {
    "summary": "Generated 3 queries using uncertainty_sampling strategy",
    "reasoning": "High uncertainty - focusing on areas where your preferences are least clear"
  }
}
```

### Step 2: Collect User Response

User chooses their preferred option (typically in your UI):
- **Choice:** "A" or "B" or "indifferent"
- **Confidence:** 0.0 to 1.0 (optional, defaults to 1.0)

### Step 3: Update Beliefs

**Endpoint:** `POST /api/v1/preferences/update`

**Request:**
```json
{
  "user_id": "user-12345",
  "query_id": "query_001",
  "response": "A",
  "confidence": 0.9
}
```

**Response:**
```json
{
  "updated_beliefs": {
    "value_weights": {
      "revenue": {
        "type": "normal",
        "parameters": {"mean": 0.45, "std": 0.25}
      },
      "churn": {
        "type": "normal",
        "parameters": {"mean": 0.55, "std": 0.25}
      },
      "brand_perception": {
        "type": "normal",
        "parameters": {"mean": 0.50, "std": 0.28}
      }
    },
    "risk_tolerance": {
      "type": "beta",
      "parameters": {"alpha": 2, "beta": 2}
    },
    "time_horizon": {
      "type": "normal",
      "parameters": {"mean": 12, "std": 3}
    },
    "uncertainty_estimates": {
      "revenue_weight": 0.51,
      "churn_weight": 0.47,
      "brand_perception_weight": 0.54
    }
  },
  "queries_completed": 1,
  "estimated_queries_remaining": 4,
  "next_queries": [
    {
      "id": "query_002",
      "question": "...",
      "scenario_a": {...},
      "scenario_b": {...},
      "information_gain": 0.38
    }
  ],
  "learning_summary": {
    "top_priorities": ["churn", "revenue", "brand_perception"],
    "confidence": 0.48,
    "insights": [
      "You strongly prioritize churn in your decisions",
      "You value churn over revenue",
      "We're developing a good understanding of your preferences"
    ],
    "ready_for_recommendations": false
  }
}
```

### Step 4: Continue Until Convergence

Repeat Steps 1-3 until either:
- `learning_summary.ready_for_recommendations == true` (confidence threshold met)
- `queries_completed >= max_queries` (typically 10)

When `ready_for_recommendations` is true, you have a reliable model of user preferences and can:
- Personalize recommendations
- Predict user choices
- Explain decisions in terms of their values

---

## Integration Examples

### Example 1: Basic Preference Learning Loop

```python
import httpx

ISL_BASE_URL = "https://isl-staging.onrender.com"
USER_ID = "user-12345"

async def learn_user_preferences():
    """Learn user preferences through ActiVA."""
    async with httpx.AsyncClient(base_url=ISL_BASE_URL) as client:
        converged = False
        queries_completed = 0
        max_queries = 10

        while not converged and queries_completed < max_queries:
            # Step 1: Get next query
            elicit_response = await client.post(
                "/api/v1/preferences/elicit",
                json={
                    "user_id": USER_ID,
                    "context": {
                        "domain": "pricing",
                        "variables": ["revenue", "churn", "brand"]
                    },
                    "num_queries": 1
                }
            )

            query = elicit_response.json()["queries"][0]

            # Step 2: Present to user and get response
            user_choice = present_query_to_user(query)  # Your UI logic

            # Step 3: Update beliefs
            update_response = await client.post(
                "/api/v1/preferences/update",
                json={
                    "user_id": USER_ID,
                    "query_id": query["id"],
                    "response": user_choice,
                    "confidence": 1.0
                }
            )

            result = update_response.json()
            converged = result["learning_summary"]["ready_for_recommendations"]
            queries_completed = result["queries_completed"]

            print(f"Query {queries_completed}: {result['learning_summary']['insights']}")

        print(f"✓ Learning complete! Top priorities: {result['learning_summary']['top_priorities']}")
        return result["updated_beliefs"]
```

### Example 2: Resuming with Stored Beliefs

```python
async def continue_preference_learning(user_id: str, current_beliefs: dict):
    """Continue preference learning with existing beliefs."""
    async with httpx.AsyncClient(base_url=ISL_BASE_URL) as client:
        # Provide current beliefs to get refined queries
        response = await client.post(
            "/api/v1/preferences/elicit",
            json={
                "user_id": user_id,
                "context": {
                    "domain": "pricing",
                    "variables": ["revenue", "churn"]
                },
                "current_beliefs": current_beliefs,  # Pass existing beliefs
                "num_queries": 3
            }
        )

        # Strategy will adapt based on current uncertainty
        strategy = response.json()["strategy"]
        print(f"Strategy: {strategy['type']} - {strategy['rationale']}")

        return response.json()["queries"]
```

### Example 3: Different Domains

```python
# Pricing decisions
pricing_context = {
    "domain": "pricing",
    "variables": ["revenue", "churn", "brand_perception"],
    "constraints": {"industry": "SaaS", "current_price": 49}
}

# Feature prioritization
feature_context = {
    "domain": "feature_prioritization",
    "variables": ["user_satisfaction", "development_cost", "time_to_market"],
    "constraints": {"team_size": 5, "quarter": "Q4"}
}

# Marketing campaigns
marketing_context = {
    "domain": "marketing",
    "variables": ["reach", "engagement", "conversion", "cost"],
    "constraints": {"budget": 50000, "channel": "digital"}
}
```

---

## Interpreting Results

### Understanding Learned Weights

The `value_weights` in the belief model represent **relative importance**:

```json
{
  "revenue": {"type": "normal", "parameters": {"mean": 0.7, "std": 0.15}},
  "churn": {"type": "normal", "parameters": {"mean": 0.3, "std": 0.15}}
}
```

**Interpretation:**
- User values **revenue 2.3x more than churn** (0.7 / 0.3)
- Still some uncertainty (std = 0.15), but converging
- Can predict: User will prefer +£10k revenue over -1% churn

### Confidence Levels

The `confidence` field (0-1) indicates learning progress:

- **< 0.4:** Early stages, keep asking questions
- **0.4-0.6:** Developing understanding, several more queries needed
- **0.6-0.8:** Good understanding, ready for initial recommendations
- **> 0.8:** High confidence, reliable preference model

### Uncertainty Estimates

Track uncertainty for each feature separately:

```json
"uncertainty_estimates": {
  "revenue_weight": 0.2,   // Low - we're confident
  "churn_weight": 0.7,     // High - still learning
  "brand_weight": 0.5      // Medium - partially learned
}
```

Use this to:
- **Focus queries** on high-uncertainty features
- **Explain confidence** to users ("We're still learning about your risk tolerance")
- **Decide when to stop** (all uncertainties < 0.3)

### Information Gain

Queries are ranked by expected information gain (EIG):

- **EIG > 0.5:** Highly informative query, will significantly reduce uncertainty
- **EIG 0.2-0.5:** Moderately informative
- **EIG < 0.2:** Low information value, might skip

---

## Best Practices

### 1. Context Design

**Good context:**
```json
{
  "domain": "pricing",
  "variables": ["revenue", "churn", "brand_perception"],
  "constraints": {
    "industry": "SaaS",
    "current_price": 49,
    "competitors": [39, 59, 79]
  }
}
```

**Why it's good:**
- Specific domain
- Relevant variables
- Realistic constraints
- Enables grounded counterfactuals

**Avoid:**
```json
{
  "domain": "general",
  "variables": ["x", "y", "z"]
}
```
- Too abstract, counterfactuals won't be meaningful

### 2. Number of Queries

**Recommended per request:**
- **Initial elicitation:** 3-5 queries
- **Refinement:** 1-3 queries
- **Total needed:** 5-10 queries to convergence

**Why not more?**
- Cognitive load on users
- Diminishing returns after ~10 queries
- Quality > quantity

### 3. User Experience

**Do:**
- Show progress: "Question 3 of ~7"
- Explain why: "This helps us understand your risk tolerance"
- Allow confidence: "How sure are you? (50%, 75%, 100%)"
- Visualize scenarios: Use charts/graphs for outcomes
- Celebrate convergence: "✓ We understand your priorities!"

**Don't:**
- Ask all questions at once (spread across sessions)
- Use jargon in questions
- Force responses (allow "skip" or "indifferent")
- Hide learning progress

### 4. Storage & Persistence

**Store beliefs between sessions:**
```python
# After each update
beliefs = response["updated_beliefs"]
await redis.set(f"user:{user_id}:beliefs", json.dumps(beliefs), ex=604800)  # 7 days

# On next session
stored_beliefs = await redis.get(f"user:{user_id}:beliefs")
if stored_beliefs:
    current_beliefs = json.loads(stored_beliefs)
```

ISL's `UserStorage` service handles this automatically with Redis, but you can also:
- Store in your own database
- Include in user profile
- Sync across devices

### 5. Handling Edge Cases

**User changes their mind:**
- Reset beliefs: Start fresh elicitation
- Or use low confidence on new responses to gradually shift

**Indecisive users:**
- Accept "indifferent" responses (they still provide signal!)
- Lower your convergence threshold
- Offer default recommendations earlier

**Domain changes:**
- Different contexts = different preferences
- E.g., pricing preferences ≠ feature preferences
- Maintain separate belief models per domain

---

## Metrics & Monitoring

ISL automatically tracks ActiVA performance metrics:

### Query Metrics

```promql
# Total queries generated
isl_activa_queries_generated_total

# Total queries answered by choice
isl_activa_queries_answered_total{choice="A"}
isl_activa_queries_answered_total{choice="B"}
```

### Convergence Metrics

```promql
# Users reaching convergence by query count
isl_activa_convergence_total{num_queries_bucket="1-5"}   # Fast learners
isl_activa_convergence_total{num_queries_bucket="6-10"}  # Typical
isl_activa_convergence_total{num_queries_bucket="11+"}   # Slow learners
```

### Quality Metrics

```promql
# Distribution of information gain
histogram_quantile(0.5, isl_activa_information_gain)  # Median EIG
histogram_quantile(0.95, isl_activa_information_gain) # 95th percentile
```

### Monitoring Dashboards

**Key questions to monitor:**
1. **Are queries informative?** → Check median EIG > 0.3
2. **Do users converge?** → Check convergence rate > 80%
3. **How many queries needed?** → Check median in "6-10" bucket
4. **Any performance issues?** → Check p95 latency < 400ms

**Alert conditions:**
- Median EIG < 0.2 (queries not informative)
- Convergence rate < 50% (users not learning)
- p95 latency > 1000ms (performance degradation)

---

## Troubleshooting

### Low Information Gain

**Problem:** All queries have EIG < 0.2

**Causes:**
- Beliefs already converged (good!)
- Context too simple (only 1-2 variables)
- Scenarios too similar

**Solutions:**
- Check if `ready_for_recommendations == true`
- Add more variables to context
- Increase scenario diversity

### Slow Convergence

**Problem:** Still not converged after 10+ queries

**Causes:**
- User giving inconsistent responses
- Confidence too low
- Context too complex (many variables)

**Solutions:**
- Show user their previous answers for consistency
- Ask user to be more decisive
- Focus on 2-3 key variables first

### Redis Connection Issues

**Problem:** Beliefs not persisting

**Causes:**
- Redis unavailable
- Connection timeout

**Solutions:**
- UserStorage has graceful fallback to in-memory
- Check Redis health: `await redis.ping()`
- Verify Redis connection in logs

---

## Further Reading

- [Technical Specifications](../TECHNICAL_SPECIFICATIONS.md#preference-elicitation-activa) - API details
- [Phase 1 Architecture](../PHASE1_ARCHITECTURE.md#preference-elicitation-activa) - System design
- [API Examples](../API_EXAMPLES.md) - More code samples
- [Monitoring Guide](../operations/OBSERVABILITY_GUIDE.md) - Metrics setup

**Research Paper:**
- ActiVA: Active Value Alignment through Counterfactual Queries (Binns et al., NeurIPS 2025)

---

## Support

For questions or issues with ActiVA integration:
- Check [API documentation](../API.md)
- Review [integration examples](./INTEGRATION_EXAMPLES.md)
- File an issue on the GitHub repository
