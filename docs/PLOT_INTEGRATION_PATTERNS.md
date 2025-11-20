# PLoT Integration Patterns

**Complete Working Examples for ISL Integration**

**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Pattern 1: Causal Model Validation](#pattern-1-causal-model-validation)
2. [Pattern 2: Counterfactual Analysis](#pattern-2-counterfactual-analysis)
3. [Pattern 3: Preference Elicitation Flow](#pattern-3-preference-elicitation-flow)
4. [Pattern 4: Teaching Examples Generation](#pattern-4-teaching-examples-generation)
5. [Pattern 5: Team Alignment](#pattern-5-team-alignment)
6. [Pattern 6: Sensitivity Analysis](#pattern-6-sensitivity-analysis)
7. [Pattern 7: Async Concurrent Requests](#pattern-7-async-concurrent-requests)
8. [Pattern 8: Advanced Validation with Suggestions](#pattern-8-advanced-validation-with-suggestions)

---

## Prerequisites

All examples assume:

```python
import httpx
import uuid
import asyncio
from typing import Dict, Any, List

# Configuration
ISL_BASE_URL = "http://localhost:8000"
TIMEOUT = 10.0  # seconds

def generate_request_id() -> str:
    """Generate unique request ID for tracing."""
    return f"req_{uuid.uuid4().hex[:12]}"
```

---

## Pattern 1: Causal Model Validation

**Use Case**: Validate that a causal DAG supports identification of treatment → outcome effects before running experiments or building decision models.

### Complete Example

```python
def validate_causal_model(
    dag_nodes: List[str],
    dag_edges: List[List[str]],
    treatment: str,
    outcome: str
) -> Dict[str, Any]:
    """
    Validate causal model for identifiability.

    Args:
        dag_nodes: List of variable names in the DAG
        dag_edges: List of [source, target] edges
        treatment: Treatment variable name
        outcome: Outcome variable name

    Returns:
        Validation results with adjustment sets or issues

    Raises:
        httpx.HTTPError: If request fails
        ValueError: If validation fails
    """
    request_id = generate_request_id()

    payload = {
        "dag": {
            "nodes": dag_nodes,
            "edges": dag_edges
        },
        "treatment": treatment,
        "outcome": outcome
    }

    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{ISL_BASE_URL}/api/v1/causal/validate",
            json=payload,
            headers={"X-Request-Id": request_id}
        )

        response.raise_for_status()
        result = response.json()

        # Check validation status
        if result["status"] == "identifiable":
            print(f"✓ Effect is identifiable")
            print(f"  Adjustment sets: {result['adjustment_sets']}")
            print(f"  Minimal set: {result['minimal_set']}")

        elif result["status"] == "uncertain":
            print(f"⚠ Identification uncertain")
            print(f"  Issues: {result.get('issues', [])}")

        else:  # cannot_identify
            print(f"✗ Cannot identify effect")
            print(f"  Issues: {result.get('issues', [])}")
            raise ValueError(f"Causal effect not identifiable: {result.get('issues')}")

        # Extract metadata for reproducibility
        metadata = result.get("_metadata", {})
        print(f"  Config fingerprint: {metadata.get('config_fingerprint')}")
        print(f"  Request ID: {metadata.get('request_id')}")

        return result


# Example usage: Pricing decision model
if __name__ == "__main__":
    result = validate_causal_model(
        dag_nodes=["Price", "Revenue", "Brand", "Churn"],
        dag_edges=[
            ["Price", "Revenue"],
            ["Brand", "Price"],
            ["Brand", "Revenue"],
            ["Price", "Churn"],
            ["Churn", "Revenue"]
        ],
        treatment="Price",
        outcome="Revenue"
    )

    # Use adjustment set in downstream analysis
    adjustment_set = result["minimal_set"]
    print(f"\nAdjust for: {adjustment_set}")
```

### Expected Output

```
✓ Effect is identifiable
  Adjustment sets: [['Brand'], ['Brand', 'Churn']]
  Minimal set: ['Brand']
  Config fingerprint: a1b2c3d4e5f6
  Request ID: req_abc123def456

Adjust for: ['Brand']
```

---

## Pattern 2: Counterfactual Analysis

**Use Case**: Estimate "what if" scenarios for decision making using structural causal models.

### Complete Example

```python
def run_counterfactual_analysis(
    structural_model: Dict[str, Any],
    outcome: str,
    intervention: Dict[str, float],
    num_samples: int = 1000
) -> Dict[str, Any]:
    """
    Run counterfactual analysis with Monte Carlo sampling.

    Args:
        structural_model: Structural equations and distributions
        outcome: Variable to predict
        intervention: Interventions to apply (variable -> value)
        num_samples: Monte Carlo samples (default: 1000)

    Returns:
        Counterfactual predictions with confidence intervals
    """
    request_id = generate_request_id()

    payload = {
        "model": structural_model,
        "outcome": outcome,
        "intervention": intervention,
        "num_samples": num_samples
    }

    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{ISL_BASE_URL}/api/v1/causal/counterfactual",
            json=payload,
            headers={"X-Request-Id": request_id}
        )

        response.raise_for_status()
        result = response.json()

        # Extract key results
        prediction = result["prediction"]
        uncertainty = result["uncertainty"]
        robustness = result["robustness"]

        print(f"Counterfactual Analysis Results:")
        print(f"  Scenario: {result['scenario']['intervention']}")
        print(f"  Point estimate: {prediction['point_estimate']:.2f}")
        print(f"  95% CI: [{prediction['confidence_interval']['lower']:.2f}, "
              f"{prediction['confidence_interval']['upper']:.2f}]")
        print(f"  Overall uncertainty: {uncertainty['overall']:.1%}")
        print(f"  Robustness score: {robustness['score']:.1%}")

        # Check critical assumptions
        if robustness.get("critical_assumptions"):
            print(f"\n⚠ Critical assumptions:")
            for assumption in robustness["critical_assumptions"]:
                print(f"    - {assumption}")

        return result


# Example usage: Price increase scenario
if __name__ == "__main__":
    model = {
        "variables": ["Price", "Churn", "Revenue"],
        "equations": {
            "Churn": "0.1 + 0.005 * Price",
            "Revenue": "10000 + 500 * Price - 2000 * Churn"
        },
        "distributions": {
            "Price": {
                "type": "normal",
                "parameters": {"mean": 50, "std": 5}
            }
        }
    }

    # Scenario: Increase price to $60
    result = run_counterfactual_analysis(
        structural_model=model,
        outcome="Revenue",
        intervention={"Price": 60},
        num_samples=1000
    )

    # Decision logic
    baseline_revenue = 51000  # Current revenue
    predicted_revenue = result["prediction"]["point_estimate"]

    if predicted_revenue > baseline_revenue:
        decision = "APPROVE price increase"
    else:
        decision = "REJECT price increase"

    print(f"\nDecision: {decision}")
    print(f"Expected revenue change: ${predicted_revenue - baseline_revenue:,.0f}")
```

### Expected Output

```
Counterfactual Analysis Results:
  Scenario: {'Price': 60}
  Point estimate: 51200.50
  95% CI: [49500.00, 53000.00]
  Overall uncertainty: 12.5%
  Robustness score: 85.3%

⚠ Critical assumptions:
    - Linear relationship between Price and Churn
    - No unmeasured confounders

Decision: APPROVE price increase
Expected revenue change: $200
```

---

## Pattern 3: Preference Elicitation Flow

**Use Case**: Learn user value weights through adaptive questioning (multi-step interaction).

### Complete Example

```python
class PreferenceElicitationSession:
    """Manages preference elicitation session for a user."""

    def __init__(self, user_id: str, context: Dict[str, Any]):
        self.user_id = user_id
        self.context = context
        self.session_queries = []
        self.session_responses = []

    def elicit_queries(self, num_queries: int = 3) -> List[Dict[str, Any]]:
        """Generate preference queries for the user."""
        request_id = generate_request_id()

        payload = {
            "user_id": self.user_id,
            "context": self.context,
            "num_queries": num_queries
        }

        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(
                f"{ISL_BASE_URL}/api/v1/preferences/elicit",
                json=payload,
                headers={"X-Request-Id": request_id}
            )

            response.raise_for_status()
            result = response.json()

            queries = result["queries"]
            self.session_queries.extend(queries)

            print(f"Generated {len(queries)} queries:")
            for i, query in enumerate(queries, 1):
                print(f"\n  Query {i} (ID: {query['query_id']}):")
                print(f"    {query['question']}")
                print(f"    A: {query['options']['option_a']}")
                print(f"    B: {query['options']['option_b']}")
                print(f"    Information value: {query['information_value']:.3f}")

            return queries

    def update_beliefs(
        self,
        query_id: str,
        selected_option: str
    ) -> Dict[str, Any]:
        """Update user beliefs based on query response."""
        request_id = generate_request_id()

        payload = {
            "user_id": self.user_id,
            "query_id": query_id,
            "response": {
                "selected_option": selected_option
            }
        }

        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(
                f"{ISL_BASE_URL}/api/v1/preferences/update",
                json=payload,
                headers={"X-Request-Id": request_id}
            )

            response.raise_for_status()
            result = response.json()

            self.session_responses.append({
                "query_id": query_id,
                "selected_option": selected_option
            })

            beliefs = result["updated_beliefs"]
            uncertainty_reduction = result["uncertainty_reduction"]

            print(f"\n✓ Beliefs updated:")
            print(f"  Uncertainty reduced by: {uncertainty_reduction:.1%}")
            print(f"  Value weights: {beliefs['value_weights']}")

            # Check if more queries needed
            remaining_uncertainty = result.get("remaining_uncertainty", 0)
            if remaining_uncertainty > 0.2:
                print(f"  ⚠ Remaining uncertainty: {remaining_uncertainty:.1%}")
                print(f"  → Recommend {result.get('suggested_queries', 2)} more queries")
            else:
                print(f"  ✓ Sufficient certainty achieved!")

            return result

    def get_final_beliefs(self) -> Dict[str, Any]:
        """Retrieve final learned beliefs."""
        # In production, fetch from Redis or include in update response
        return {
            "user_id": self.user_id,
            "queries_answered": len(self.session_responses),
            "belief_state": "converged"
        }


# Example usage: Learn pricing preferences
if __name__ == "__main__":
    # Initialize session
    session = PreferenceElicitationSession(
        user_id="alice_ceo",
        context={
            "domain": "pricing",
            "variables": ["revenue", "churn", "brand_reputation"]
        }
    )

    # Step 1: Generate queries
    queries = session.elicit_queries(num_queries=3)

    # Step 2: User answers queries (simulated)
    user_responses = ["A", "B", "A"]  # From UI interaction

    for query, response in zip(queries, user_responses):
        print(f"\n{'='*60}")
        print(f"User selected: {response}")
        session.update_beliefs(
            query_id=query["query_id"],
            selected_option=response
        )

    # Step 3: Retrieve final beliefs
    final_beliefs = session.get_final_beliefs()
    print(f"\n{'='*60}")
    print(f"Preference elicitation complete!")
    print(f"Total queries answered: {final_beliefs['queries_answered']}")
```

### Expected Output

```
Generated 3 queries:

  Query 1 (ID: query_abc123):
    Would you prefer Option A or Option B?
    A: +£5k revenue, +2% churn
    B: +£3k revenue, +0.5% churn
    Information value: 0.856

  Query 2 (ID: query_def456):
    Would you prefer Option A or Option B?
    A: +£10k revenue, -5% brand reputation
    B: +£8k revenue, +2% brand reputation
    Information value: 0.723

============================================================
User selected: A

✓ Beliefs updated:
  Uncertainty reduced by: 35.2%
  Value weights: {'revenue': 0.72, 'churn': 0.15, 'brand_reputation': 0.13}
  ⚠ Remaining uncertainty: 28.5%
  → Recommend 2 more queries

============================================================
Preference elicitation complete!
Total queries answered: 3
```

---

## Pattern 4: Teaching Examples Generation

**Use Case**: Generate pedagogical examples to help users understand trade-offs in their decision domain.

### Complete Example

```python
def generate_teaching_examples(
    user_id: str,
    target_concept: str,
    current_beliefs: Dict[str, Any],
    context: Dict[str, Any],
    max_examples: int = 3
) -> Dict[str, Any]:
    """
    Generate Bayesian teaching examples.

    Args:
        user_id: Unique user identifier
        target_concept: Concept to teach (e.g., "trade_offs", "diminishing_returns")
        current_beliefs: User's current belief state
        context: Decision domain context
        max_examples: Maximum examples to generate

    Returns:
        Teaching examples with learning objectives
    """
    request_id = generate_request_id()

    payload = {
        "user_id": user_id,
        "target_concept": target_concept,
        "current_beliefs": current_beliefs,
        "context": context,
        "max_examples": max_examples
    }

    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{ISL_BASE_URL}/api/v1/teaching/teach",
            json=payload,
            headers={"X-Request-Id": request_id}
        )

        response.raise_for_status()
        result = response.json()

        examples = result["examples"]
        objectives = result["learning_objectives"]
        expected_time = result["expected_learning_time"]

        print(f"Teaching Module: {target_concept.replace('_', ' ').title()}")
        print(f"Expected learning time: {expected_time} minutes\n")

        print(f"Learning Objectives:")
        for obj in objectives:
            print(f"  • {obj}")

        print(f"\nExamples:")
        for i, example in enumerate(examples, 1):
            print(f"\n  Example {i}:")
            print(f"    Scenario: {example['scenario']}")
            print(f"    Key insight: {example['explanation']}")
            print(f"    Information value: {example['information_value']:.3f}")

        return result


# Example usage: Teach revenue-churn trade-offs
if __name__ == "__main__":
    beliefs = {
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
            "revenue_weight": 0.25
        }
    }

    result = generate_teaching_examples(
        user_id="bob_product_mgr",
        target_concept="trade_offs",
        current_beliefs=beliefs,
        context={
            "domain": "pricing",
            "variables": ["revenue", "churn"]
        },
        max_examples=3
    )

    # Use examples in UI to build user understanding
    print(f"\n→ Present these examples to user in interactive tutorial")
```

### Expected Output

```
Teaching Module: Trade Offs
Expected learning time: 5 minutes

Learning Objectives:
  • Understand that revenue and churn often move in opposite directions
  • Recognize when short-term revenue gains sacrifice long-term sustainability
  • Identify the optimal balance point for your specific context

Examples:

  Example 1:
    Scenario: Aggressive price increase (+20%) yields +£15k revenue but +8% churn
    Key insight: High churn erodes customer lifetime value over 12-month horizon
    Information value: 0.892

  Example 2:
    Scenario: Moderate price increase (+10%) yields +£8k revenue but +3% churn
    Key insight: Balanced approach maintains growth while limiting customer loss
    Information value: 0.745

  Example 3:
    Scenario: Value-add pricing (premium tier) yields +£10k revenue with -1% churn
    Key insight: Differentiation strategy can increase revenue AND reduce churn
    Information value: 0.823

→ Present these examples to user in interactive tutorial
```

---

## Pattern 5: Team Alignment

**Use Case**: Find common ground across multiple stakeholders for consensus decision-making.

### Complete Example

```python
def find_team_alignment(
    perspectives: List[Dict[str, Any]],
    options: List[Dict[str, Any]],
    decision_context: str = ""
) -> Dict[str, Any]:
    """
    Find team alignment across stakeholder perspectives.

    Args:
        perspectives: List of stakeholder perspectives (role, goals, constraints)
        options: Decision options to evaluate
        decision_context: Context for the decision

    Returns:
        Alignment analysis with recommendations
    """
    request_id = generate_request_id()

    payload = {
        "perspectives": perspectives,
        "options": options,
        "context": decision_context
    }

    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{ISL_BASE_URL}/api/v1/team/align",
            json=payload,
            headers={"X-Request-Id": request_id}
        )

        response.raise_for_status()
        result = response.json()

        common_ground = result["common_ground"]
        aligned_options = result["aligned_options"]
        conflicts = result["conflicts"]
        recommendation = result["recommendation"]

        print(f"Team Alignment Analysis")
        print(f"{'='*60}\n")

        print(f"Common Ground (Agreement: {common_ground['agreement_level']}%):")
        print(f"  Shared goals:")
        for goal in common_ground["shared_goals"]:
            print(f"    • {goal}")
        print(f"  Shared constraints:")
        for constraint in common_ground["shared_constraints"]:
            print(f"    • {constraint}")

        print(f"\nOptions Ranked by Alignment:")
        for i, option in enumerate(aligned_options[:3], 1):
            print(f"  {i}. {option['option_name']}")
            print(f"     Satisfaction score: {option['satisfaction_score']}/100")
            print(f"     Stakeholders satisfied: {option['stakeholders_satisfied']}")

        if conflicts:
            print(f"\n⚠ Conflicts Identified ({len(conflicts)}):")
            for conflict in conflicts[:2]:  # Show top 2
                print(f"  • {conflict['description']}")
                print(f"    Severity: {conflict['severity']}")
                if conflict.get("resolution"):
                    print(f"    Suggested resolution: {conflict['resolution']}")

        print(f"\n✓ Top Recommendation:")
        print(f"  Option: {recommendation['top_option']}")
        print(f"  Rationale: {recommendation['rationale']}")

        return result


# Example usage: Product launch decision
if __name__ == "__main__":
    result = find_team_alignment(
        perspectives=[
            {
                "role": "CEO",
                "goals": ["Maximize revenue", "Meet Q4 targets"],
                "constraints": ["Limited budget"],
                "priority_weights": {"revenue": 0.7, "timeline": 0.3}
            },
            {
                "role": "Engineering",
                "goals": ["Build quality product", "Minimize tech debt"],
                "constraints": ["3-month timeline"],
                "priority_weights": {"quality": 0.6, "timeline": 0.4}
            },
            {
                "role": "Customer Success",
                "goals": ["High user satisfaction", "Low churn"],
                "constraints": ["Current support capacity"],
                "priority_weights": {"satisfaction": 0.8, "timeline": 0.2}
            }
        ],
        options=[
            {
                "name": "Option A: Full-feature launch",
                "attributes": {
                    "revenue_potential": "high",
                    "timeline": "6 months",
                    "quality": "high",
                    "risk": "medium"
                }
            },
            {
                "name": "Option B: MVP launch",
                "attributes": {
                    "revenue_potential": "medium",
                    "timeline": "3 months",
                    "quality": "medium",
                    "risk": "low"
                }
            },
            {
                "name": "Option C: Phased rollout",
                "attributes": {
                    "revenue_potential": "medium-high",
                    "timeline": "4 months",
                    "quality": "high",
                    "risk": "low"
                }
            }
        ],
        decision_context="Q4 product launch decision"
    )

    # Use recommendation in decision-making process
    print(f"\n→ Proceed with: {result['recommendation']['top_option']}")
```

### Expected Output

```
Team Alignment Analysis
============================================================

Common Ground (Agreement: 68%):
  Shared goals:
    • Deliver value to users
    • Meet Q4 deadline
    • Maintain quality standards
  Shared constraints:
    • Limited budget
    • 3-6 month timeline window

Options Ranked by Alignment:
  1. Option C: Phased rollout
     Satisfaction score: 82/100
     Stakeholders satisfied: ['CEO', 'Engineering', 'Customer Success']
  2. Option B: MVP launch
     Satisfaction score: 71/100
     Stakeholders satisfied: ['CEO', 'Engineering']
  3. Option A: Full-feature launch
     Satisfaction score: 58/100
     Stakeholders satisfied: ['Engineering', 'Customer Success']

⚠ Conflicts Identified (2):
  • Timeline vs. Quality: CEO prioritizes Q4, Engineering wants 6 months
    Severity: MODERATE
    Suggested resolution: Phased rollout allows Q4 launch with quality focus
  • Revenue vs. Risk: CEO wants high revenue, CS wants low risk
    Severity: LOW
    Suggested resolution: MVP minimizes risk while capturing revenue

✓ Top Recommendation:
  Option: Option C: Phased rollout
  Rationale: Balances revenue goals (CEO), quality requirements (Engineering),
             and user satisfaction (Customer Success). Phased approach reduces
             risk while meeting Q4 deadline.

→ Proceed with: Option C: Phased rollout
```

---

## Pattern 6: Sensitivity Analysis

**Use Case**: Test how robust your conclusions are to changes in assumptions.

### Complete Example

```python
def analyze_sensitivity(
    baseline_result: str,
    assumptions: List[Dict[str, Any]],
    model_description: str = ""
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis on assumptions.

    Args:
        baseline_result: Current conclusion/estimate
        assumptions: List of assumptions to test
        model_description: Description of the model

    Returns:
        Sensitivity analysis with robustness metrics
    """
    request_id = generate_request_id()

    payload = {
        "baseline_result": baseline_result,
        "assumptions": assumptions,
        "model_description": model_description
    }

    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{ISL_BASE_URL}/api/v1/analysis/sensitivity",
            json=payload,
            headers={"X-Request-Id": request_id}
        )

        response.raise_for_status()
        result = response.json()

        conclusion = result["conclusion"]
        analyzed_assumptions = result["assumptions"]
        robustness = result["robustness"]

        print(f"Sensitivity Analysis")
        print(f"{'='*60}\n")

        print(f"Baseline Conclusion: {conclusion['statement']}")
        print(f"Overall Robustness: {robustness['overall']:.1%}\n")

        print(f"Assumption Analysis:")
        for assumption in analyzed_assumptions:
            importance_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }.get(assumption["importance"], "⚪")

            print(f"\n  {importance_emoji} {assumption['name']}")
            print(f"     Importance: {assumption['importance'].upper()}")
            print(f"     Impact if violated: {assumption['impact_if_violated']}")

            if assumption.get("sensitivity_range"):
                range_info = assumption["sensitivity_range"]
                print(f"     Tolerance: ±{range_info.get('tolerance', 'N/A')}")

        # Breakpoints
        if robustness.get("breakpoints"):
            print(f"\n⚠ Critical Breakpoints:")
            for bp in robustness["breakpoints"][:2]:
                print(f"  • {bp['description']}")
                print(f"    Value: {bp['threshold']}")

        # Overall assessment
        if robustness["overall"] >= 0.8:
            print(f"\n✓ Conclusion is ROBUST to assumption changes")
        elif robustness["overall"] >= 0.6:
            print(f"\n⚠ Conclusion has MODERATE robustness")
        else:
            print(f"\n✗ Conclusion is FRAGILE to assumption changes")

        return result


# Example usage: Pricing decision robustness
if __name__ == "__main__":
    result = analyze_sensitivity(
        baseline_result="Price increase to £60 yields +£5k revenue",
        assumptions=[
            {
                "name": "Price elasticity",
                "description": "10% price increase → 5% churn increase",
                "baseline_value": 0.5,
                "plausible_range": {"min": 0.3, "max": 0.8}
            },
            {
                "name": "Customer lifetime value",
                "description": "Average CLV = £10,000",
                "baseline_value": 10000,
                "plausible_range": {"min": 8000, "max": 12000}
            },
            {
                "name": "Market growth rate",
                "description": "5% annual market growth",
                "baseline_value": 0.05,
                "plausible_range": {"min": 0.02, "max": 0.08}
            }
        ],
        model_description="Pricing impact model with churn elasticity"
    )

    # Decision logic
    if result["robustness"]["overall"] >= 0.7:
        print(f"\n→ Proceed with price increase (robust conclusion)")
    else:
        print(f"\n→ Gather more data to reduce uncertainty")
```

### Expected Output

```
Sensitivity Analysis
============================================================

Baseline Conclusion: Price increase to £60 yields +£5k revenue
Overall Robustness: 73.5%

Assumption Analysis:

  🔴 Price elasticity
     Importance: CRITICAL
     Impact if violated: Revenue could decrease if elasticity > 0.65
     Tolerance: ±30%

  🟡 Customer lifetime value
     Importance: MEDIUM
     Impact if violated: ROI calculation changes but direction stable
     Tolerance: ±20%

  🟢 Market growth rate
     Importance: LOW
     Impact if violated: Minimal impact on short-term decision
     Tolerance: ±60%

⚠ Critical Breakpoints:
  • Price elasticity > 0.65: Revenue becomes negative
    Value: 0.65
  • CLV < £7,500: Payback period exceeds acceptable threshold
    Value: 7500

✓ Conclusion is ROBUST to assumption changes

→ Proceed with price increase (robust conclusion)
```

---

## Pattern 7: Async Concurrent Requests

**Use Case**: Make multiple ISL requests concurrently for improved performance.

### Complete Example

```python
import asyncio
import httpx
from typing import List, Dict, Any


async def validate_multiple_models_async(
    models: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Validate multiple causal models concurrently.

    Args:
        models: List of model specifications

    Returns:
        List of validation results
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = []

        for model_spec in models:
            request_id = generate_request_id()

            payload = {
                "dag": model_spec["dag"],
                "treatment": model_spec["treatment"],
                "outcome": model_spec["outcome"]
            }

            task = client.post(
                f"{ISL_BASE_URL}/api/v1/causal/validate",
                json=payload,
                headers={"X-Request-Id": request_id}
            )
            tasks.append(task)

        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"✗ Model {i+1} failed: {response}")
                results.append({"status": "error", "error": str(response)})
            else:
                response.raise_for_status()
                result = response.json()
                print(f"✓ Model {i+1}: {result['status']}")
                results.append(result)

        return results


async def mixed_concurrent_requests() -> Dict[str, Any]:
    """
    Make different types of requests concurrently.

    Returns:
        Combined results from all endpoints
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Prepare different requests
        validation_request = client.post(
            f"{ISL_BASE_URL}/api/v1/causal/validate",
            json={
                "dag": {
                    "nodes": ["X", "Y", "Z"],
                    "edges": [["X", "Y"], ["Z", "Y"]]
                },
                "treatment": "X",
                "outcome": "Y"
            },
            headers={"X-Request-Id": generate_request_id()}
        )

        preference_request = client.post(
            f"{ISL_BASE_URL}/api/v1/preferences/elicit",
            json={
                "user_id": "concurrent_user",
                "context": {
                    "domain": "pricing",
                    "variables": ["revenue", "churn"]
                },
                "num_queries": 2
            },
            headers={"X-Request-Id": generate_request_id()}
        )

        health_request = client.get(
            f"{ISL_BASE_URL}/health",
            headers={"X-Request-Id": generate_request_id()}
        )

        # Execute all concurrently
        validation_resp, preference_resp, health_resp = await asyncio.gather(
            validation_request,
            preference_request,
            health_request
        )

        return {
            "validation": validation_resp.json(),
            "preferences": preference_resp.json(),
            "health": health_resp.json()
        }


# Example usage
if __name__ == "__main__":
    # Test 1: Validate multiple models
    models_to_validate = [
        {
            "dag": {
                "nodes": ["A", "B", "C"],
                "edges": [["A", "B"], ["B", "C"]]
            },
            "treatment": "A",
            "outcome": "C"
        },
        {
            "dag": {
                "nodes": ["X", "Y", "Z"],
                "edges": [["X", "Y"], ["X", "Z"], ["Y", "Z"]]
            },
            "treatment": "X",
            "outcome": "Z"
        },
        {
            "dag": {
                "nodes": ["Price", "Revenue"],
                "edges": [["Price", "Revenue"]]
            },
            "treatment": "Price",
            "outcome": "Revenue"
        }
    ]

    print("Validating multiple models concurrently...")
    results = asyncio.run(validate_multiple_models_async(models_to_validate))
    print(f"\nValidated {len(results)} models")

    # Test 2: Mixed requests
    print(f"\n{'='*60}")
    print("Making mixed concurrent requests...")
    combined = asyncio.run(mixed_concurrent_requests())
    print(f"✓ Received responses from {len(combined)} endpoints")
    print(f"  Validation status: {combined['validation']['status']}")
    print(f"  Preferences: {len(combined['preferences']['queries'])} queries")
    print(f"  Health: {combined['health']['status']}")
```

### Expected Output

```
Validating multiple models concurrently...
✓ Model 1: identifiable
✓ Model 2: identifiable
✓ Model 3: identifiable

Validated 3 models

============================================================
Making mixed concurrent requests...
✓ Received responses from 3 endpoints
  Validation status: identifiable
  Preferences: 2 queries
  Health: healthy
```

---

## Pattern 8: Advanced Validation with Suggestions

**Use Case**: Get comprehensive model validation with actionable improvement suggestions.

### Complete Example

```python
def validate_model_advanced(
    dag: Dict[str, Any],
    structural_model: Dict[str, Any] = None,
    validation_level: str = "standard"
) -> Dict[str, Any]:
    """
    Perform advanced validation with suggestions.

    Args:
        dag: DAG structure with nodes and edges
        structural_model: Optional structural equations
        validation_level: "basic", "standard", or "comprehensive"

    Returns:
        Validation results with quality score and suggestions
    """
    request_id = generate_request_id()

    payload = {
        "dag": dag,
        "validation_level": validation_level
    }

    if structural_model:
        payload["structural_model"] = structural_model

    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{ISL_BASE_URL}/api/v1/validation/validate-model",
            json=payload,
            headers={"X-Request-Id": request_id}
        )

        response.raise_for_status()
        result = response.json()

        quality = result["overall_quality"]
        score = result["quality_score"]
        validation_results = result["validation_results"]
        suggestions = result["suggestions"]
        best_practices = result["best_practices"]

        print(f"Model Validation Report")
        print(f"{'='*60}\n")

        print(f"Overall Quality: {quality.upper()} ({score:.0f}/100)")

        # Show validation results by category
        for category in ["structural", "statistical", "domain"]:
            cat_results = getattr(validation_results, category, None)
            if cat_results:
                checks = cat_results.checks
                passed = sum(1 for c in checks if c.status == "pass")
                print(f"\n{category.title()} Checks: {passed}/{len(checks)} passed")

                # Show failed checks
                failed = [c for c in checks if c.status in ["fail", "warning"]]
                for check in failed[:2]:  # Show top 2
                    status_emoji = "⚠" if check.status == "warning" else "✗"
                    print(f"  {status_emoji} {check.name}: {check.message}")

        # Show suggestions
        if suggestions:
            print(f"\n💡 Suggestions for Improvement:")
            for i, suggestion in enumerate(suggestions[:5], 1):
                print(f"  {i}. {suggestion['description']}")
                print(f"     Priority: {suggestion['priority']}")
                if suggestion.get("impact"):
                    print(f"     Expected impact: {suggestion['impact']}")

        # Best practices
        practices_followed = sum(1 for p in best_practices if p["followed"])
        print(f"\nBest Practices: {practices_followed}/{len(best_practices)} followed")

        return result


# Example usage
if __name__ == "__main__":
    # Complex model with potential issues
    result = validate_model_advanced(
        dag={
            "nodes": ["A", "B", "C", "D", "E", "F"],
            "edges": [
                ["A", "B"],
                ["B", "C"],
                ["C", "D"],
                ["D", "E"],
                ["E", "F"],
                ["A", "F"]  # Long path + shortcut
            ]
        },
        structural_model={
            "equations": {
                "B": "2 * A + noise",
                "C": "B + 0.5 * A",
                "D": "C * 1.5",
                "E": "D + 0.1",
                "F": "E + A"
            },
            "distributions": {
                "A": {"type": "normal", "parameters": {"mean": 0, "std": 1}},
                "noise": {"type": "normal", "parameters": {"mean": 0, "std": 0.1}}
            }
        },
        validation_level="comprehensive"
    )

    # Decision logic
    if result["quality_score"] >= 75:
        print(f"\n✓ Model ready for production use")
    elif result["quality_score"] >= 50:
        print(f"\n⚠ Address suggestions before deploying")
    else:
        print(f"\n✗ Significant issues - redesign recommended")
```

### Expected Output

```
Model Validation Report
============================================================

Overall Quality: GOOD (78/100)

Structural Checks: 8/10 passed
  ⚠ Long causal chains: Path A→B→C→D→E→F has 5 edges (recommend <4)
  ✗ Naming convention: Node 'A' should have descriptive name

Statistical Checks: 5/5 passed

Domain Checks: 3/4 passed
  ⚠ Documentation: Add domain context for better interpretability

💡 Suggestions for Improvement:
  1. Use descriptive variable names instead of A, B, C
     Priority: MEDIUM
     Expected impact: +15% interpretability
  2. Consider simplifying causal chain A→B→C→D→E→F
     Priority: LOW
     Expected impact: +5% model clarity
  3. Add domain context documentation
     Priority: LOW
     Expected impact: Better team understanding

Best Practices: 6/8 followed

⚠ Address suggestions before deploying
```

---

## Common Integration Patterns Summary

| Pattern | Use Case | Complexity | Latency |
|---------|----------|------------|---------|
| **Causal Validation** | Verify DAG before analysis | Low | <50ms |
| **Counterfactual** | "What if" scenario analysis | Medium | <1.5s |
| **Preference Flow** | Multi-step user preference learning | Medium | <100ms per step |
| **Teaching** | Pedagogical example generation | Medium | <500ms |
| **Team Alignment** | Multi-stakeholder consensus | Medium | <400ms |
| **Sensitivity** | Robustness testing | Low | <200ms |
| **Async Concurrent** | Parallel request execution | High | N×latency → max(latency) |
| **Advanced Validation** | Comprehensive model checks | Medium | <200ms |

---

## Integration Best Practices

1. **Always include Request IDs** for tracing
2. **Handle errors gracefully** (see [Error Handling Guide](PLOT_ERROR_HANDLING.md))
3. **Use async for multiple requests** to minimize latency
4. **Cache deterministic results** using config fingerprints
5. **Validate inputs locally** before sending to ISL
6. **Set appropriate timeouts** (5-10s recommended)
7. **Monitor performance** and track P95 latencies

---

## Next Steps

- Review [Error Handling Guide](PLOT_ERROR_HANDLING.md) for robust error management
- Consult [Performance Guide](PLOT_PERFORMANCE_GUIDE.md) for optimization
- Test patterns in your environment

---

**Last Updated**: 2025-11-20
**Document Version**: 1.0.0
**ISL Version**: 1.0.0
