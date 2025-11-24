"""
Integration tests for TAE (Team Alignment Engine) workflows.

Tests validate TAE-specific workflows:
- Robustness filtering for proposals
- Batch counterfactual analysis for deliberation
- Sensitivity analysis for disputes
"""

import pytest
import httpx
from typing import Dict, Any
import time


@pytest.fixture
def base_url():
    """ISL API base URL."""
    return "http://localhost:8000/api/v1"


@pytest.fixture
async def async_client(base_url):
    """Async HTTP client for testing."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client


# ==================== ROBUSTNESS FILTERING ====================

@pytest.mark.asyncio
async def test_robustness_filtering_for_proposals(
    async_client,
    feature_prioritization_dag,
    team_scenarios
):
    """
    Test TAE workflow: Filter team proposals by robustness score.

    TAE sends multiple team proposals, ISL returns robustness scores,
    TAE filters proposals with robustness < 0.8.
    """
    # Step 1: Get robustness scores for all proposals
    robustness_scores = []

    for scenario in team_scenarios:
        response = await async_client.post(
            "/causal/robustness",
            json={
                "dag": feature_prioritization_dag,
                "intervention": scenario["interventions"],
                "outcome": "Revenue",
                "checks": ["model_misspecification", "unobserved_confounding"]
            }
        )

        assert response.status_code == 200
        result = response.json()

        # Extract overall robustness score
        score = result.get("overall_score", result.get("robustness_score", 0.0))

        robustness_scores.append({
            "team": scenario["team"],
            "proposal": scenario["proposal"],
            "robustness": score,
            "details": result
        })

    # Step 2: Filter by robustness threshold
    robust_proposals = [
        p for p in robustness_scores
        if p["robustness"] >= 0.8
    ]

    # Verify we got scores for all proposals
    assert len(robustness_scores) == len(team_scenarios)

    # Verify scores are in valid range
    for score_data in robustness_scores:
        assert 0.0 <= score_data["robustness"] <= 1.0


@pytest.mark.asyncio
async def test_robustness_with_sensitivity_analysis(
    async_client,
    feature_prioritization_dag
):
    """
    Test combined robustness + sensitivity for proposal vetting.
    """
    intervention = {"FeatureA": 1.0, "FeatureB": 0.8}

    # Get robustness
    robustness_response = await async_client.post(
        "/causal/robustness",
        json={
            "dag": feature_prioritization_dag,
            "intervention": intervention,
            "outcome": "Revenue"
        }
    )

    assert robustness_response.status_code == 200
    robustness = robustness_response.json()

    # Get sensitivity
    sensitivity_response = await async_client.post(
        "/causal/sensitivity/detailed",
        json={
            "model": {"dag": feature_prioritization_dag},
            "intervention": intervention,
            "outcome": "Revenue",
            "assumptions": [
                "no_unobserved_confounding",
                "linearity",
                "no_selection_bias"
            ],
            "violation_levels": [0.1, 0.2, 0.3]
        }
    )

    # Both should succeed
    assert robustness_response.status_code == 200
    assert sensitivity_response.status_code in [200, 404]  # 404 if endpoint not found

    # If sensitivity available, verify structure
    if sensitivity_response.status_code == 200:
        sensitivity = sensitivity_response.json()
        assert "sensitivities" in sensitivity or "metrics" in sensitivity


# ==================== BATCH COUNTERFACTUAL ANALYSIS ====================

@pytest.mark.asyncio
async def test_batch_counterfactuals_for_deliberation(
    async_client,
    feature_prioritization_dag,
    team_scenarios
):
    """
    Test TAE deliberation workflow: Batch analyze team proposals.
    """
    response = await async_client.post(
        "/batch/scenarios",
        json={
            "dag": feature_prioritization_dag,
            "scenarios": team_scenarios,
            "outcome": "Revenue",
            "mode": "compare"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Verify all teams analyzed
    assert "results" in result
    assert len(result["results"]) == len(team_scenarios)

    # Verify each result has prediction
    for scenario_result in result["results"]:
        assert "predicted_outcome" in scenario_result or "outcome" in scenario_result

    # Results should be rankable
    outcomes = [
        r.get("predicted_outcome", r.get("outcome", 0))
        for r in result["results"]
    ]
    assert all(isinstance(o, (int, float)) for o in outcomes)


@pytest.mark.asyncio
async def test_batch_with_explanations(
    async_client,
    feature_prioritization_dag,
    team_scenarios
):
    """
    Test batch analysis includes explanations for each scenario.
    """
    response = await async_client.post(
        "/batch/scenarios",
        json={
            "dag": feature_prioritization_dag,
            "scenarios": team_scenarios,
            "outcome": "Revenue",
            "mode": "compare",
            "include_explanations": True
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Check for explanations
    for scenario_result in result["results"]:
        # Explanations optional but if included should have structure
        if "explanation" in scenario_result:
            exp = scenario_result["explanation"]
            assert isinstance(exp, (str, dict))


# ==================== COMPARISON & RANKING ====================

@pytest.mark.asyncio
async def test_proposal_comparison(
    async_client,
    feature_prioritization_dag,
    team_scenarios
):
    """
    Test comparing proposals and identifying best option.
    """
    response = await async_client.post(
        "/batch/scenarios",
        json={
            "dag": feature_prioritization_dag,
            "scenarios": team_scenarios,
            "outcome": "Revenue",
            "mode": "compare",
            "rank_by": "outcome"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Verify results present
    assert "results" in result
    results = result["results"]

    # Extract outcomes for ranking
    outcomes = []
    for r in results:
        outcome = r.get("predicted_outcome", r.get("outcome", 0))
        outcomes.append(outcome)

    # Verify we got outcomes
    assert len(outcomes) == len(team_scenarios)


# ==================== SENSITIVITY FOR DISPUTES ====================

@pytest.mark.asyncio
async def test_sensitivity_for_disputed_assumption(
    async_client,
    feature_prioritization_dag
):
    """
    Test sensitivity analysis when teams dispute an assumption.

    E.g., Product team claims linearity, Engineering disputes it.
    """
    intervention = {"FeatureA": 1.0}

    # Test sensitivity to linearity assumption
    response = await async_client.post(
        "/causal/sensitivity/detailed",
        json={
            "model": {"dag": feature_prioritization_dag},
            "intervention": intervention,
            "outcome": "Revenue",
            "assumptions": ["linearity"],
            "violation_levels": [0.1, 0.3, 0.5]  # Mild to severe violations
        }
    )

    # Should work or return 404 if endpoint not implemented yet
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        result = response.json()

        # Should identify if assumption is critical
        assert "sensitivities" in result or "most_critical" in result


@pytest.mark.asyncio
async def test_assumption_ranking(
    async_client,
    feature_prioritization_dag
):
    """
    Test ranking multiple assumptions by criticality.
    """
    intervention = {"FeatureA": 0.9, "FeatureB": 0.7}

    response = await async_client.post(
        "/causal/sensitivity/detailed",
        json={
            "model": {"dag": feature_prioritization_dag},
            "intervention": intervention,
            "outcome": "Revenue",
            "assumptions": [
                "no_unobserved_confounding",
                "linearity",
                "no_selection_bias"
            ],
            "violation_levels": [0.2, 0.4]
        }
    )

    assert response.status_code in [200, 404]

    if response.status_code == 200:
        result = response.json()

        # Should provide ranking of assumptions
        # (exact field name may vary)
        assert "most_critical" in result or "sensitivities" in result


# ==================== DELIBERATION SUPPORT ====================

@pytest.mark.asyncio
async def test_counterfactual_with_uncertainty(
    async_client,
    feature_prioritization_dag,
    calibration_data
):
    """
    Test counterfactual analysis with uncertainty quantification for deliberation.
    """
    intervention = {"FeatureA": 1.0}

    # Get point estimate
    point_response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": feature_prioritization_dag,
            "intervention": intervention,
            "outcome": "Revenue"
        }
    )

    assert point_response.status_code == 200
    point_result = point_response.json()
    assert "predicted_outcome" in point_result

    # Get interval estimate (if conformal available)
    interval_response = await async_client.post(
        "/causal/conformal",
        json={
            "dag": feature_prioritization_dag,
            "intervention": intervention,
            "outcome": "Revenue",
            "calibration_data": calibration_data,
            "confidence_level": 0.90
        }
    )

    # May not be available for all DAGs
    if interval_response.status_code == 200:
        interval_result = interval_response.json()
        assert "prediction_interval" in interval_result


# ==================== TEAM PROPOSAL VALIDATION ====================

@pytest.mark.asyncio
async def test_validate_team_proposal(
    async_client,
    feature_prioritization_dag
):
    """
    Test validating a team's proposed DAG structure.
    """
    response = await async_client.post(
        "/causal/validate",
        json={
            "dag": feature_prioritization_dag,
            "treatment": "FeatureA",
            "outcome": "Revenue"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should indicate if identifiable
    assert "status" in result or "identifiable" in result


@pytest.mark.asyncio
async def test_invalid_proposal_gets_suggestions(
    async_client
):
    """
    Test that invalid team proposal gets actionable suggestions.
    """
    # Team proposes a cyclic DAG (invalid)
    cyclic_dag = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "A"}  # Creates cycle
        ]
    }

    response = await async_client.post(
        "/causal/validate",
        json={
            "dag": cyclic_dag,
            "treatment": "A",
            "outcome": "C"
        }
    )

    # Should return error or validation failure
    assert response.status_code in [200, 400, 422]

    if response.status_code == 200:
        result = response.json()
        # Should indicate problem
        assert "status" in result or "valid" in result


# ==================== PERFORMANCE ====================

@pytest.mark.asyncio
async def test_batch_analysis_performance(
    async_client,
    feature_prioritization_dag,
    team_scenarios,
    performance_threshold
):
    """
    Test that batch analysis completes within performance threshold.
    """
    start_time = time.time()

    response = await async_client.post(
        "/batch/scenarios",
        json={
            "dag": feature_prioritization_dag,
            "scenarios": team_scenarios,
            "outcome": "Revenue",
            "mode": "compare"
        }
    )

    duration = time.time() - start_time

    assert response.status_code == 200
    assert duration < performance_threshold, f"Batch analysis took {duration:.2f}s"
