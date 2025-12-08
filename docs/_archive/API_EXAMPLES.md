# ISL API Examples

**Concrete code examples for all Inference Service Layer endpoints**

**Version**: 1.0
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Setup & Authentication](#setup--authentication)
2. [Health & Monitoring](#health--monitoring)
3. [Preference Learning](#preference-learning)
4. [Causal Analysis](#causal-analysis)
5. [Teaching & Pedagogy](#teaching--pedagogy)
6. [Team Alignment](#team-alignment)
7. [Advanced Validation](#advanced-validation)
8. [Complete Workflows](#complete-workflows)

---

## Setup & Authentication

### Python with `httpx`

```python
import httpx
import asyncio

# Initialize async client
client = httpx.AsyncClient(
    base_url="http://localhost:8000",
    timeout=30.0
)

async def main():
    # Your API calls here
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

### Python with `requests` (synchronous)

```python
import requests

BASE_URL = "http://localhost:8000"

def call_api(endpoint, method="POST", json=None):
    url = f"{BASE_URL}{endpoint}"
    if method == "POST":
        return requests.post(url, json=json, timeout=30)
    else:
        return requests.get(url, timeout=30)
```

### JavaScript/TypeScript

```typescript
const BASE_URL = "http://localhost:8000";

async function callAPI(endpoint: string, data?: any) {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    method: data ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: data ? JSON.stringify(data) : undefined,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# POST request
curl -X POST http://localhost:8000/api/v1/preferences/elicit \
  -H "Content-Type: application/json" \
  -d @request.json
```

---

## Health & Monitoring

### Example 1: Basic Health Check

```python
async def check_service_health():
    """Check if ISL service is running."""
    response = await client.get("/health")
    data = response.json()

    print(f"Status: {data['status']}")
    print(f"Version: {data['version']}")
    print(f"Timestamp: {data['timestamp']}")

    return data['status'] == 'healthy'

# Expected output:
# Status: healthy
# Version: 0.1.0
# Timestamp: 2025-11-20T12:00:00Z
```

### Example 2: Prometheus Metrics

```python
async def get_metrics():
    """Retrieve Prometheus metrics for monitoring."""
    response = await client.get("/metrics")
    metrics_text = response.text

    # Parse key metrics
    for line in metrics_text.split('\n'):
        if 'http_requests_total' in line and not line.startswith('#'):
            print(line)

# Expected output:
# http_requests_total{endpoint="/api/v1/preferences/elicit",method="POST",status="200"} 45
```

---

## Preference Learning

### Example 1: Initial Preference Elicitation (Cold Start)

```python
async def initial_preference_elicitation():
    """Generate initial queries for a new user."""

    request = {
        "user_id": "alice_2025",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn", "customer_satisfaction"],
            "constraints": {
                "industry": "SaaS",
                "current_price": 49,
                "competitor_range": [29, 99]
            }
        },
        "num_queries": 3
    }

    response = await client.post("/api/v1/preferences/elicit", json=request)
    data = response.json()

    print(f"Generated {len(data['queries'])} queries")
    print(f"Strategy: {data['strategy']['type']}")
    print(f"Estimated queries remaining: {data['estimated_queries_remaining']}")

    for i, query in enumerate(data['queries'], 1):
        print(f"\nQuery {i} (info gain: {query['information_gain']:.3f}):")
        print(f"  Scenario A: {query['scenario_a']['description']}")
        print(f"  - {query['scenario_a']['outcomes']}")
        print(f"  Scenario B: {query['scenario_b']['description']}")
        print(f"  - {query['scenario_b']['outcomes']}")

    return data

# Expected output:
# Generated 3 queries
# Strategy: exploration
# Estimated queries remaining: 4
#
# Query 1 (info gain: 0.320):
#   Scenario A: Prioritize revenue over churn
#   - {'revenue': 0.8, 'churn': 0.2, 'customer_satisfaction': 0.5}
#   Scenario B: Prioritize churn over revenue
#   - {'revenue': 0.2, 'churn': 0.8, 'customer_satisfaction': 0.7}
```

### Example 2: Adaptive Elicitation (Warm Start)

```python
async def adaptive_preference_elicitation():
    """Generate queries based on existing beliefs."""

    # Existing beliefs from previous interactions
    current_beliefs = {
        "value_weights": {
            "revenue": {
                "type": "normal",
                "parameters": {"mean": 0.6, "std": 0.25}
            },
            "churn": {
                "type": "normal",
                "parameters": {"mean": 0.4, "std": 0.25}
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
            "revenue_weight": 0.5,
            "churn_weight": 0.5
        }
    }

    request = {
        "user_id": "bob_2025",
        "context": {
            "domain": "feature_prioritization",
            "variables": ["user_satisfaction", "development_cost", "time_to_market"]
        },
        "current_beliefs": current_beliefs,
        "num_queries": 2  # Fewer queries needed with warm start
    }

    response = await client.post("/api/v1/preferences/elicit", json=request)
    data = response.json()

    print(f"Strategy: {data['strategy']['type']}")
    print(f"Focus areas: {data['strategy']['focus_areas']}")
    print(f"Queries remaining: {data['estimated_queries_remaining']}")

    return data

# Expected output:
# Strategy: uncertainty_sampling
# Focus areas: ['user_satisfaction vs development_cost trade-off']
# Queries remaining: 2
```

### Example 3: Update Beliefs Based on User Response

```python
async def update_user_beliefs():
    """Update beliefs after user answers a query."""

    # User answered query_001 by choosing scenario B with high confidence
    request = {
        "user_id": "alice_2025",
        "query_id": "query_001",
        "response": "B",  # User chose scenario B
        "confidence": 0.85  # 85% confident in their choice
    }

    response = await client.post("/api/v1/preferences/update", json=request)
    data = response.json()

    print("Updated Beliefs:")
    for var, belief in data['updated_beliefs']['value_weights'].items():
        mean = belief['parameters']['mean']
        std = belief['parameters']['std']
        print(f"  {var}: mean={mean:.3f}, std={std:.3f}")

    print(f"\nLearning Progress:")
    print(f"  Entropy reduction: {data['learning_progress']['entropy_reduction']:.3f}")
    print(f"  Queries answered: {data['learning_progress']['queries_answered']}")
    print(f"  Estimated remaining: {data['learning_progress']['estimated_queries_remaining']}")

    print(f"\nInsights:")
    print(f"  Top priorities: {data['insights']['top_priorities']}")
    print(f"  Confidence: {data['insights']['confidence_level']}")
    print(f"  Ready for recommendations: {data['insights']['recommendations_ready']}")

    return data

# Expected output:
# Updated Beliefs:
#   revenue: mean=0.450, std=0.180
#   churn: mean=0.550, std=0.180
#   customer_satisfaction: mean=0.500, std=0.200
#
# Learning Progress:
#   Entropy reduction: 0.182
#   Queries answered: 1
#   Estimated remaining: 3
#
# Insights:
#   Top priorities: ['churn', 'revenue']
#   Confidence: medium
#   Ready for recommendations: False
```

### Example 4: Complete Preference Learning Session

```python
async def complete_preference_session(user_id: str, domain: str):
    """Run a complete preference learning session."""

    print(f"Starting preference learning for {user_id}")

    # Step 1: Generate initial queries
    elicit_response = await client.post("/api/v1/preferences/elicit", json={
        "user_id": user_id,
        "context": {
            "domain": domain,
            "variables": ["revenue", "churn", "satisfaction"]
        },
        "num_queries": 3
    })

    queries = elicit_response.json()["queries"]
    print(f"Generated {len(queries)} queries\n")

    # Step 2: Simulate user answering each query
    for i, query in enumerate(queries, 1):
        print(f"Query {i}/{len(queries)}")
        print(f"Question: {query['question']}\n")

        # Simulate user choice (in real app, this comes from UI)
        user_choice = "A" if i % 2 == 0 else "B"
        user_confidence = 0.7 + (i * 0.1)  # Increasing confidence

        print(f"User chose: Scenario {user_choice} (confidence: {user_confidence:.1f})")

        # Update beliefs
        update_response = await client.post("/api/v1/preferences/update", json={
            "user_id": user_id,
            "query_id": query["id"],
            "response": user_choice,
            "confidence": user_confidence
        })

        learning_progress = update_response.json()["learning_progress"]
        print(f"Entropy reduction: {learning_progress['entropy_reduction']:.3f}")
        print(f"Estimated queries remaining: {learning_progress['estimated_queries_remaining']}\n")

    # Step 3: Final insights
    final_update = update_response.json()
    print("Final Insights:")
    print(f"  Top priorities: {final_update['insights']['top_priorities']}")
    print(f"  Confidence level: {final_update['insights']['confidence_level']}")
    print(f"  Ready for recommendations: {final_update['insights']['recommendations_ready']}")

    return final_update

# Example usage:
# await complete_preference_session("carol_2025", "pricing")
```

---

## Causal Analysis

### Example 1: Validate Simple DAG

```python
async def validate_simple_dag():
    """Validate a simple causal DAG."""

    request = {
        "dag": {
            "nodes": ["Price", "Revenue"],
            "edges": [["Price", "Revenue"]]
        },
        "treatment": "Price",
        "outcome": "Revenue"
    }

    response = await client.post("/api/v1/causal/validate", json=request)
    data = response.json()

    print(f"Status: {data['status']}")
    print(f"Confidence: {data['confidence']}")
    if data['adjustment_sets']:
        print(f"Adjustment sets: {data['adjustment_sets']}")
    print(f"\nExplanation: {data['explanation']['summary']}")

    return data

# Expected output:
# Status: identifiable
# Confidence: high
# Adjustment sets: [[]]
#
# Explanation: Price → Revenue effect is directly identifiable
```

### Example 2: Validate Complex DAG with Confounders

```python
async def validate_complex_dag():
    """Validate a DAG with confounding."""

    request = {
        "dag": {
            "nodes": ["Price", "Brand", "Revenue", "Churn"],
            "edges": [
                ["Price", "Revenue"],
                ["Price", "Churn"],
                ["Brand", "Price"],
                ["Brand", "Revenue"],
                ["Churn", "Revenue"]
            ]
        },
        "treatment": "Price",
        "outcome": "Revenue"
    }

    response = await client.post("/api/v1/causal/validate", json=request)
    data = response.json()

    print(f"Status: {data['status']}")
    print(f"Confidence: {data['confidence']}")

    if data['adjustment_sets']:
        print(f"\nAdjustment Sets:")
        for i, adj_set in enumerate(data['adjustment_sets'], 1):
            print(f"  Option {i}: Adjust for {adj_set if adj_set else 'nothing (backdoor-free)'}")

    print(f"\nExplanation:")
    print(f"  Summary: {data['explanation']['summary']}")
    print(f"  Reasoning: {data['explanation']['reasoning']}")

    print(f"\nAssumptions:")
    for assumption in data['explanation']['assumptions']:
        print(f"  - {assumption}")

    return data

# Expected output:
# Status: identifiable
# Confidence: high
#
# Adjustment Sets:
#   Option 1: Adjust for ['Brand']
#   Option 2: Adjust for ['Brand', 'Churn']
#
# Explanation:
#   Summary: Price → Revenue effect is identifiable by adjusting for confounders
#   Reasoning: Brand confounds the Price-Revenue relationship. By adjusting for Brand, we can isolate the causal effect.
#
# Assumptions:
#   - No unmeasured confounders
#   - Causal Markov assumption holds
#   - No selection bias
```

### Example 3: Basic Counterfactual Analysis

```python
async def basic_counterfactual():
    """Perform simple counterfactual analysis."""

    request = {
        "model": {
            "variables": ["Price", "Revenue"],
            "equations": {
                "Revenue": "10000 + 500 * Price"
            },
            "distributions": {
                "Price": {
                    "type": "normal",
                    "parameters": {"mean": 50, "std": 5}
                }
            }
        },
        "intervention": {"Price": 60},
        "outcome": "Revenue",
        "num_samples": 1000
    }

    response = await client.post("/api/v1/causal/counterfactual", json=request)
    data = response.json()

    pred = data['prediction']
    print(f"Point Estimate: ${pred['point_estimate']:,.2f}")
    print(f"95% Confidence Interval: [${pred['confidence_interval']['lower']:,.2f}, "
          f"${pred['confidence_interval']['upper']:,.2f}]")

    print(f"\nUncertainty: {data['uncertainty']['overall']}")
    print(f"Robustness: {data['robustness']['level']} (score: {data['robustness']['score']:.2f})")

    print(f"\nExplanation: {data['explanation']['summary']}")

    return data

# Expected output:
# Point Estimate: $40,000.00
# 95% Confidence Interval: [$39,000.00, $41,000.00]
#
# Uncertainty: low
# Robustness: high (score: 0.92)
#
# Explanation: Setting Price to $60 would result in Revenue of approximately $40,000
```

### Example 4: Complex Counterfactual with Multiple Variables

```python
async def complex_counterfactual():
    """Counterfactual analysis with dependent variables."""

    request = {
        "model": {
            "variables": ["Price", "Churn", "Revenue"],
            "equations": {
                "Churn": "0.05 + 0.001 * Price",  # Churn depends on Price
                "Revenue": "10000 + 500 * Price - 15000 * Churn"  # Revenue depends on both
            },
            "distributions": {
                "Price": {
                    "type": "normal",
                    "parameters": {"mean": 50, "std": 5}
                }
            }
        },
        "intervention": {"Price": 70},
        "outcome": "Revenue",
        "num_samples": 1000
    }

    response = await client.post("/api/v1/causal/counterfactual", json=request)
    data = response.json()

    # Main prediction
    pred = data['prediction']
    print(f"Predicted Revenue at Price=$70:")
    print(f"  Point estimate: ${pred['point_estimate']:,.2f}")
    print(f"  95% CI: [${pred['confidence_interval']['lower']:,.2f}, "
          f"${pred['confidence_interval']['upper']:,.2f}]")

    # Uncertainty breakdown
    print(f"\nUncertainty Analysis:")
    print(f"  Overall: {data['uncertainty']['overall']}")
    print(f"  Sources:")
    for source in data['uncertainty']['sources']:
        print(f"    - {source['type']}: {source['contribution']:.1%}")

    # Robustness analysis
    print(f"\nRobustness Assessment:")
    print(f"  Level: {data['robustness']['level']}")
    print(f"  Score: {data['robustness']['score']:.2f}")
    print(f"  Sensitivity ranges:")
    for sens in data['robustness']['sensitivity_ranges']:
        print(f"    - {sens['assumption']}: "
              f"[{sens['range']['min']}, {sens['range']['max']}] "
              f"(current: {sens['current_value']})")

    # Explanation
    print(f"\n{data['explanation']['summary']}")
    print(f"Key factors: {', '.join(data['explanation']['key_factors'])}")

    return data

# Expected output:
# Predicted Revenue at Price=$70:
#   Point estimate: $43,250.00
#   95% CI: [$41,500.00, $45,000.00]
#
# Uncertainty Analysis:
#   Overall: medium
#   Sources:
#     - structural: 50.0%
#     - parametric: 35.0%
#     - distributional: 15.0%
#
# Robustness Assessment:
#   Level: high
#   Score: 0.85
#   Sensitivity ranges:
#     - Price elasticity: [400, 600] (current: 500)
#     - Churn sensitivity: [0.0008, 0.0012] (current: 0.001)
#
# Increasing price to $70 would increase revenue to ~$43,250, accounting for increased churn
# Key factors: Price elasticity, Churn sensitivity
```

### Example 5: Compare Multiple Interventions

```python
async def compare_interventions():
    """Compare counterfactual outcomes for multiple interventions."""

    base_model = {
        "variables": ["Price", "Churn", "Revenue"],
        "equations": {
            "Churn": "0.05 + 0.001 * Price",
            "Revenue": "10000 + 500 * Price - 15000 * Churn"
        },
        "distributions": {
            "Price": {
                "type": "normal",
                "parameters": {"mean": 50, "std": 5}
            }
        }
    }

    # Test multiple price points
    price_points = [40, 50, 60, 70, 80]
    results = []

    for price in price_points:
        request = {
            "model": base_model,
            "intervention": {"Price": price},
            "outcome": "Revenue",
            "num_samples": 1000
        }

        response = await client.post("/api/v1/causal/counterfactual", json=request)
        data = response.json()

        results.append({
            "price": price,
            "revenue": data['prediction']['point_estimate'],
            "ci_lower": data['prediction']['confidence_interval']['lower'],
            "ci_upper": data['prediction']['confidence_interval']['upper'],
            "uncertainty": data['uncertainty']['overall']
        })

    # Print comparison
    print("Price Point Comparison:")
    print(f"{'Price':<10} {'Revenue':<12} {'95% CI':<25} {'Uncertainty'}")
    print("-" * 70)
    for r in results:
        print(f"${r['price']:<9} ${r['revenue']:>10,.0f}  "
              f"[${r['ci_lower']:>8,.0f}, ${r['ci_upper']:>8,.0f}]  "
              f"{r['uncertainty']}")

    # Find optimal price
    optimal = max(results, key=lambda x: x['revenue'])
    print(f"\nOptimal price point: ${optimal['price']} (Revenue: ${optimal['revenue']:,.0f})")

    return results

# Expected output:
# Price Point Comparison:
# Price      Revenue      95% CI                    Uncertainty
# ----------------------------------------------------------------------
# $40        $39,400      [$38,200 , $40,600 ]     low
# $50        $42,250      [$40,800 , $43,700 ]     low
# $60        $45,100      [$43,300 , $46,900 ]     medium
# $70        $47,950      [$45,700 , $50,200 ]     medium
# $80        $50,800      [$47,900 , $53,700 ]     high
#
# Optimal price point: $80 (Revenue: $50,800)
```

---

## Teaching & Pedagogy

### Example 1: Generate Teaching Examples for Trade-Offs

```python
async def teach_tradeoffs():
    """Generate teaching examples to explain trade-offs."""

    request = {
        "user_id": "david_2025",
        "current_beliefs": {
            "value_weights": {
                "revenue": {
                    "type": "normal",
                    "parameters": {"mean": 0.7, "std": 0.2}
                },
                "churn": {
                    "type": "normal",
                    "parameters": {"mean": 0.3, "std": 0.2}
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
                "revenue_weight": 0.3,
                "churn_weight": 0.4
            }
        },
        "target_concept": "trade_offs",
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn"]
        },
        "max_examples": 3
    }

    response = await client.post("/api/v1/teaching/teach", json=request)
    data = response.json()

    print(f"Teaching Concept: {request['target_concept']}")
    print(f"Generated {len(data['examples'])} examples")
    print(f"Estimated time: {data['estimated_time_minutes']} minutes\n")

    for i, example in enumerate(data['examples'], 1):
        print(f"Example {i}: {example['id']}")
        print(f"  Difficulty: {example['difficulty']}")
        print(f"  Estimated time: {example['estimated_time']} min")
        print(f"  Focus: {example['pedagogical_focus']}")
        print(f"\n  Scenario: {example['scenario']['description']}")
        print(f"  Outcomes: {example['scenario']['outcomes']}")
        print(f"  Explanation: {example['scenario']['explanation']}\n")

    print(f"Learning Objectives:")
    for obj in data['learning_objectives']:
        print(f"  - {obj}")

    print(f"\nNext Steps: {data['next_steps']}")

    return data

# Expected output:
# Teaching Concept: trade_offs
# Generated 3 examples
# Estimated time: 5 minutes
#
# Example 1: example_001
#   Difficulty: beginner
#   Estimated time: 2 min
#   Focus: Illustrates revenue-churn trade-off
#
#   Scenario: Premium pricing strategy
#   Outcomes: {'revenue': 120000, 'churn': 0.12}
#   Explanation: Higher prices increase revenue but also increase customer churn
#
# Example 2: example_002
#   Difficulty: intermediate
#   Estimated time: 2 min
#   Focus: Shows non-linear trade-off effects
#
#   Scenario: Moderate pricing with retention focus
#   Outcomes: {'revenue': 90000, 'churn': 0.05}
#   Explanation: Balanced approach prioritizes retention over maximum revenue
#
# Learning Objectives:
#   - Understand trade-offs between competing objectives
#   - Recognize causal relationships in decision-making
#
# Next Steps: Practice identifying trade-offs in your specific domain
```

### Example 2: Teach Confounding Concept

```python
async def teach_confounding():
    """Generate examples to explain confounding."""

    request = {
        "user_id": "eve_2025",
        "current_beliefs": {
            "value_weights": {
                "revenue": {"type": "normal", "parameters": {"mean": 0.5, "std": 0.3}}
            },
            "risk_tolerance": {"type": "beta", "parameters": {"alpha": 2, "beta": 2}},
            "time_horizon": {"type": "normal", "parameters": {"mean": 12, "std": 3}},
            "uncertainty_estimates": {"revenue_weight": 0.6}
        },
        "target_concept": "confounding",
        "context": {
            "domain": "marketing",
            "variables": ["ad_spend", "revenue", "seasonality"]
        },
        "max_examples": 2
    }

    response = await client.post("/api/v1/teaching/teach", json=request)
    data = response.json()

    print("Teaching: Confounding in Causal Analysis\n")

    for example in data['examples']:
        print(f"Scenario: {example['scenario']['description']}")
        print(f"Key Insight: {example['scenario']['explanation']}")
        print(f"Pedagogical Focus: {example['pedagogical_focus']}\n")

    return data

# Expected output:
# Teaching: Confounding in Causal Analysis
#
# Scenario: Holiday sales example
# Key Insight: Both ad spend and revenue increase during holidays, creating spurious correlation
# Pedagogical Focus: Illustrates how seasonality confounds the ad spend-revenue relationship
#
# Scenario: Brand strength example
# Key Insight: Strong brands can spend less on ads while maintaining revenue
# Pedagogical Focus: Shows how unmeasured confounders can bias causal estimates
```

---

## Team Alignment

### Example 1: Basic Team Alignment

```python
async def align_team_perspectives():
    """Find common ground across team members."""

    request = {
        "team_perspectives": [
            {
                "role": "Product Manager",
                "priorities": ["Revenue growth", "User acquisition", "Time to market"],
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
                "constraints": ["Team capacity", "Legacy system dependencies"],
                "preferred_options": ["option_c", "option_a"]
            }
        ],
        "decision_options": [
            {
                "id": "option_a",
                "name": "Quick MVP launch",
                "attributes": {
                    "speed": "fast",
                    "quality": "medium",
                    "user_experience": "basic",
                    "maintainability": "low"
                }
            },
            {
                "id": "option_b",
                "name": "Polished feature set",
                "attributes": {
                    "speed": "medium",
                    "quality": "high",
                    "user_experience": "excellent",
                    "maintainability": "medium"
                }
            },
            {
                "id": "option_c",
                "name": "Refactor-first approach",
                "attributes": {
                    "speed": "slow",
                    "quality": "high",
                    "user_experience": "good",
                    "maintainability": "high"
                }
            }
        ]
    }

    response = await client.post("/api/v1/team/align", json=request)
    data = response.json()

    print("Team Alignment Analysis\n")
    print("Common Ground:")
    for item in data['common_ground']:
        print(f"  - {item}")

    print(f"\nConflicts:")
    for conflict in data['conflicts']:
        print(f"  - {conflict['description']}")
        print(f"    Perspectives: {conflict['perspectives']}")
        print(f"    Severity: {conflict['severity']}")

    if 'recommended_option' in data:
        print(f"\nRecommended Option: {data['recommended_option']['id']}")
        print(f"Rationale: {data['recommended_option']['rationale']}")
        print(f"Compromise areas: {data['recommended_option']['compromise_areas']}")

    return data

# Expected output:
# Team Alignment Analysis
#
# Common Ground:
#   - All perspectives value quality
#   - Shared concern about constraints
#   - Agreement on need for user focus
#
# Conflicts:
#   - Speed vs Quality trade-off
#     Perspectives: ['Product Manager', 'Engineer']
#     Severity: medium
#   - MVP vs Polish debate
#     Perspectives: ['Product Manager', 'Designer']
#     Severity: high
#
# Recommended Option: option_b
# Rationale: Balances PM's time pressure with Designer's UX needs and Engineer's quality concerns
# Compromise areas: ['Medium development time', 'High quality', 'Excellent UX']
```

---

## Advanced Validation

### Example 1: Validate Decision Model

```python
async def validate_decision_model():
    """Validate a complete decision model."""

    request = {
        "dag": {
            "nodes": ["Decision", "Outcome", "Context", "Constraint"],
            "edges": [
                ["Context", "Decision"],
                ["Decision", "Outcome"],
                ["Constraint", "Decision"],
                ["Context", "Outcome"]
            ]
        },
        "validation_level": "comprehensive"
    }

    response = await client.post("/api/v1/validation/validate-model", json=request)
    data = response.json()

    print(f"Model Validation: {data['overall_status']}")
    print(f"\nChecks Performed:")
    for check in data['checks']:
        status = "✓" if check['passed'] else "✗"
        print(f"  {status} {check['name']}: {check['message']}")

    if data['issues']:
        print(f"\nIssues Found:")
        for issue in data['issues']:
            print(f"  - {issue['severity']}: {issue['description']}")
            if 'suggestions' in issue:
                for suggestion in issue['suggestions']:
                    print(f"    → {suggestion}")

    return data

# Expected output:
# Model Validation: valid
#
# Checks Performed:
#   ✓ DAG structure: Valid directed acyclic graph
#   ✓ Node coverage: All variables accounted for
#   ✓ Edge consistency: No contradictory relationships
#   ✓ Identifiability: Causal effects are identifiable
#
# Issues Found:
#   (none)
```

---

## Complete Workflows

### Example 1: End-to-End Decision Support Workflow

```python
async def decision_support_workflow(user_id: str):
    """Complete workflow: preference learning → causal analysis → recommendation."""

    print("=" * 60)
    print("DECISION SUPPORT WORKFLOW")
    print("=" * 60)

    # Phase 1: Learn User Preferences
    print("\nPhase 1: Learning User Preferences...")
    elicit_response = await client.post("/api/v1/preferences/elicit", json={
        "user_id": user_id,
        "context": {
            "domain": "pricing",
            "variables": ["revenue", "churn", "satisfaction"]
        },
        "num_queries": 3
    })

    queries = elicit_response.json()["queries"]
    print(f"  Generated {len(queries)} queries")

    # Simulate user responses
    beliefs = None
    for i, query in enumerate(queries):
        response = await client.post("/api/v1/preferences/update", json={
            "user_id": user_id,
            "query_id": query["id"],
            "response": "A" if i % 2 == 0 else "B",
            "confidence": 0.8
        })
        beliefs = response.json()["updated_beliefs"]

    print(f"  ✓ Preferences learned ({len(queries)} queries answered)")

    # Phase 2: Validate Causal Model
    print("\nPhase 2: Validating Causal Model...")
    dag = {
        "nodes": ["Price", "Revenue", "Churn"],
        "edges": [
            ["Price", "Revenue"],
            ["Price", "Churn"],
            ["Churn", "Revenue"]
        ]
    }

    validation = await client.post("/api/v1/causal/validate", json={
        "dag": dag,
        "treatment": "Price",
        "outcome": "Revenue"
    })

    val_data = validation.json()
    print(f"  Status: {val_data['status']}")
    print(f"  ✓ Model validated")

    # Phase 3: Analyze Decision Options
    print("\nPhase 3: Analyzing Decision Options...")
    price_options = [45, 50, 55, 60]
    results = []

    structural_model = {
        "variables": ["Price", "Revenue", "Churn"],
        "equations": {
            "Churn": "0.05 + 0.001 * Price",
            "Revenue": "10000 + 500 * Price - 15000 * Churn"
        },
        "distributions": {
            "Price": {"type": "normal", "parameters": {"mean": 50, "std": 5}}
        }
    }

    for price in price_options:
        cf_response = await client.post("/api/v1/causal/counterfactual", json={
            "model": structural_model,
            "intervention": {"Price": price},
            "outcome": "Revenue",
            "num_samples": 500
        })
        cf_data = cf_response.json()
        results.append({
            "price": price,
            "revenue": cf_data["prediction"]["point_estimate"],
            "uncertainty": cf_data["uncertainty"]["overall"]
        })

    print(f"  ✓ Analyzed {len(price_options)} options")

    # Phase 4: Make Recommendation
    print("\nPhase 4: Generating Recommendation...")
    # Weight by user preferences (revenue priority)
    revenue_weight = beliefs["value_weights"]["revenue"]["parameters"]["mean"]
    best_option = max(results, key=lambda x: x["revenue"] * revenue_weight)

    print(f"\n{'-' * 60}")
    print("RECOMMENDATION")
    print(f"{'-' * 60}")
    print(f"Optimal Price: ${best_option['price']}")
    print(f"Expected Revenue: ${best_option['revenue']:,.0f}")
    print(f"Uncertainty: {best_option['uncertainty']}")
    print(f"Based on your preference for revenue (weight: {revenue_weight:.2f})")
    print(f"{'-' * 60}")

    return best_option

# Example usage:
# result = await decision_support_workflow("frank_2025")
```

### Example 2: Onboarding + Teaching Workflow

```python
async def onboarding_with_teaching(user_id: str):
    """Onboard user with pedagogical examples."""

    print(f"Starting onboarding for {user_id}\n")

    # Step 1: Initial preference elicitation
    print("Step 1: Initial preference questions...")
    elicit = await client.post("/api/v1/preferences/elicit", json={
        "user_id": user_id,
        "context": {
            "domain": "product_decisions",
            "variables": ["user_value", "development_cost", "time_to_market"]
        },
        "num_queries": 2
    })

    queries = elicit.json()["queries"]

    # Simulate answering queries
    for query in queries:
        await client.post("/api/v1/preferences/update", json={
            "user_id": user_id,
            "query_id": query["id"],
            "response": "B",
            "confidence": 0.7
        })

    print(f"  ✓ Answered {len(queries)} questions\n")

    # Step 2: Teach key concepts
    print("Step 2: Learning key concepts...")
    concepts = ["trade_offs", "uncertainty"]

    for concept in concepts:
        teaching = await client.post("/api/v1/teaching/teach", json={
            "user_id": user_id,
            "current_beliefs": elicit.json().get("current_beliefs"),
            "target_concept": concept,
            "context": {
                "domain": "product_decisions",
                "variables": ["user_value", "development_cost"]
            },
            "max_examples": 1
        })

        examples = teaching.json()["examples"]
        print(f"  ✓ Learned about {concept} ({len(examples)} examples)")

    print("\n✅ Onboarding complete!")
    print(f"   - Preferences learned")
    print(f"   - Key concepts taught: {', '.join(concepts)}")

    return {"status": "complete", "concepts_learned": concepts}

# Example usage:
# await onboarding_with_teaching("grace_2025")
```

---

## Error Handling Examples

### Example: Comprehensive Error Handling

```python
async def robust_api_call(endpoint: str, data: dict, max_retries: int = 3):
    """Make API call with comprehensive error handling."""
    import time

    for attempt in range(max_retries):
        try:
            response = await client.post(endpoint, json=data)

            # Success
            if response.status_code == 200:
                return response.json()

            # Client errors (don't retry)
            elif response.status_code in [400, 422]:
                error_detail = response.json().get("detail", "Unknown error")
                print(f"❌ Validation Error: {error_detail}")
                return None

            # Server errors (retry with backoff)
            elif response.status_code == 500:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️ Server error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Server error after {max_retries} attempts")
                    return None

        except httpx.ConnectError:
            print(f"❌ Connection error: Is ISL running?")
            return None

        except httpx.TimeoutException:
            print(f"⚠️ Timeout (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return None

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    return None

# Example usage:
# result = await robust_api_call(
#     "/api/v1/preferences/elicit",
#     {"user_id": "test", "context": {...}, "num_queries": 3}
# )
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Complete API Reference**: See `/docs` (interactive Swagger UI)
