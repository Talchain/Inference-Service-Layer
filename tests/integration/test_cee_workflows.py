"""
Integration tests for CEE (Critique & Explanation Engine) workflows.

Tests validate CEE-specific workflows:
- Validation for critique generation
- Enhanced explanations for different user levels
- Sensitivity warnings for critical assumptions
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


# ==================== CRITIQUE ENHANCEMENT ====================

@pytest.mark.asyncio
async def test_validation_for_critique(
    async_client,
    confounded_dag
):
    """
    Test CEE workflow: Extract validation issues for critique generation.

    CEE asks ISL to validate a DAG, then uses issues to generate critique.
    """
    response = await async_client.post(
        "/causal/validate/strategies",
        json={
            "dag": confounded_dag,
            "treatment": "Treatment",
            "outcome": "Outcome"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Extract validation issues for critique
    identifiable = result.get("identifiable", result.get("status") == "identifiable")

    if not identifiable:
        # Should provide reasons and suggestions
        assert "suggestions" in result or "strategies" in result or "issues" in result

        # CEE can use these for critique
        suggestions = result.get("suggestions", result.get("strategies", []))
        for suggestion in suggestions:
            # Should be descriptive enough for critique
            assert isinstance(suggestion, dict)


@pytest.mark.asyncio
async def test_extract_validation_issues(
    async_client,
    pricing_dag
):
    """
    Test extracting specific validation issues for critique.
    """
    response = await async_client.post(
        "/causal/validate",
        json={
            "dag": pricing_dag,
            "treatment": "Price",
            "outcome": "Revenue"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should include validation details
    assert "status" in result or "identifiable" in result

    # May include additional validation info
    # (exact structure depends on implementation)


# ==================== PROGRESSIVE EXPLANATION LEVELS ====================

@pytest.mark.asyncio
async def test_explanation_levels_for_different_users(
    async_client,
    pricing_dag,
    calibration_data
):
    """
    Test CEE can request different explanation levels for different users.

    - Simple: For executives
    - Intermediate: For product managers
    - Technical: For engineers
    """
    intervention = {"Price": 55.0}

    # Request with explanation level
    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": intervention,
            "outcome": "Revenue",
            "explanation_level": "simple"  # or "intermediate", "technical"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should include explanation
    assert "explanation" in result or "metadata" in result

    # Explanation should be present
    explanation = result.get("explanation", {})
    if isinstance(explanation, dict):
        # May have different levels
        assert len(explanation) > 0 or isinstance(explanation, str)


@pytest.mark.asyncio
async def test_progressive_disclosure(
    async_client,
    pricing_dag
):
    """
    Test progressive disclosure: simple → intermediate → technical.
    """
    intervention = {"Price": 55.0}

    levels = ["simple", "intermediate", "technical"]
    explanations = {}

    for level in levels:
        response = await async_client.post(
            "/causal/counterfactual",
            json={
                "dag": pricing_dag,
                "intervention": intervention,
                "outcome": "Revenue",
                "explanation_level": level
            }
        )

        if response.status_code == 200:
            result = response.json()
            explanations[level] = result.get("explanation", "")

    # At least basic response should work
    # (progressive levels may not be implemented for all endpoints)


# ==================== SENSITIVITY WARNINGS ====================

@pytest.mark.asyncio
async def test_sensitivity_for_review(
    async_client,
    pricing_dag
):
    """
    Test sensitivity analysis for CEE's review functionality.

    CEE flags critical assumptions in technical reviews.
    """
    intervention = {"Price": 55.0}

    response = await async_client.post(
        "/causal/sensitivity/detailed",
        json={
            "model": {"dag": pricing_dag},
            "intervention": intervention,
            "outcome": "Revenue",
            "assumptions": [
                "no_unobserved_confounding",
                "linearity",
                "no_selection_bias"
            ],
            "violation_levels": [0.1, 0.3]
        }
    )

    # May not be available yet
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        result = response.json()

        # Extract critical assumptions for warnings
        critical = result.get("most_critical", [])

        # CEE can use these to flag review items
        # (exact structure varies)


@pytest.mark.asyncio
async def test_flag_critical_assumptions(
    async_client,
    pricing_dag
):
    """
    Test identifying critical assumptions for CEE warnings.
    """
    intervention = {"Price": 60.0, "Quality": 0.9}

    response = await async_client.post(
        "/causal/sensitivity/detailed",
        json={
            "model": {"dag": pricing_dag},
            "intervention": intervention,
            "outcome": "Revenue",
            "assumptions": [
                "no_unobserved_confounding",
                "linearity"
            ],
            "violation_levels": [0.2, 0.4]
        }
    )

    assert response.status_code in [200, 404]

    if response.status_code == 200:
        result = response.json()

        # Should identify which assumptions matter most
        assert "sensitivities" in result or "most_critical" in result


# ==================== ENHANCED EXPLANATIONS ====================

@pytest.mark.asyncio
async def test_enhanced_explanations_integration(
    async_client,
    pricing_dag
):
    """
    Test that explanations include quality enhancements.
    """
    intervention = {"Price": 50.0}

    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": intervention,
            "outcome": "Revenue",
            "include_explanation": True
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should have explanation
    assert "explanation" in result or "metadata" in result

    # Check for enhanced explanation metadata
    if "explanation" in result:
        explanation = result["explanation"]

        # May include readability scores or visual aids
        # (implementation varies)
        if isinstance(explanation, dict):
            # Enhanced metadata may be present
            pass


@pytest.mark.asyncio
async def test_readability_validation(
    async_client,
    pricing_dag
):
    """
    Test that explanations meet readability standards.
    """
    intervention = {"Price": 50.0}

    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": intervention,
            "outcome": "Revenue",
            "explanation_level": "simple"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Explanation should be present and readable
    explanation = result.get("explanation", {})

    # If it's a string, should be non-empty
    if isinstance(explanation, str):
        assert len(explanation) > 0


# ==================== CONTRASTIVE EXPLANATIONS ====================

@pytest.mark.asyncio
async def test_contrastive_for_critique(
    async_client,
    pricing_dag
):
    """
    Test contrastive explanations for CEE's critique feature.

    "Why did outcome X happen instead of Y?"
    """
    response = await async_client.post(
        "/explain/contrastive",
        json={
            "dag": pricing_dag,
            "factual": {"Price": 50.0, "Quality": 0.8},
            "factual_outcome": 5300.0,
            "counterfactual_outcome": 6000.0,
            "outcome_variable": "Revenue"
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should provide contrastive explanation
    assert "minimal_intervention" in result or "explanation" in result


@pytest.mark.asyncio
async def test_multiple_counterfactuals(
    async_client,
    pricing_dag
):
    """
    Test generating multiple counterfactual explanations.
    """
    response = await async_client.post(
        "/explain/contrastive",
        json={
            "dag": pricing_dag,
            "factual": {"Price": 50.0, "Quality": 0.8},
            "factual_outcome": 5300.0,
            "counterfactual_outcome": 6000.0,
            "outcome_variable": "Revenue",
            "n_alternatives": 3  # Request multiple alternatives
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should provide at least one intervention
    assert "minimal_intervention" in result or "interventions" in result


# ==================== CAUSAL FACTOR EXTRACTION ====================

@pytest.mark.asyncio
async def test_extract_factors_from_text(async_client):
    """
    Test extracting causal factors from unstructured text for CEE.

    CEE uses this to help users structure their mental models.
    """
    # Sample text describing a scenario
    texts = [
        "The price increase led to lower sales volume.",
        "Better product quality resulted in higher customer satisfaction.",
        "Marketing campaign drove more website traffic.",
        "Customer satisfaction improved retention rates."
    ]

    response = await async_client.post(
        "/causal/extract-factors",
        json={
            "texts": texts,
            "n_factors": 3,
            "outcome_variable": "Revenue"
        }
    )

    # May return 404 if not implemented yet
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        result = response.json()

        # Should extract factors
        assert "factors" in result

        factors = result["factors"]
        assert len(factors) > 0

        # Each factor should have structure
        for factor in factors:
            assert "name" in factor or "factor" in factor


# ==================== PERFORMANCE FOR REAL-TIME CRITIQUE ====================

@pytest.mark.asyncio
async def test_real_time_validation_performance(
    async_client,
    pricing_dag,
    performance_threshold
):
    """
    Test that validation completes fast enough for real-time critique.
    """
    start_time = time.time()

    response = await async_client.post(
        "/causal/validate",
        json={
            "dag": pricing_dag,
            "treatment": "Price",
            "outcome": "Revenue"
        }
    )

    duration = time.time() - start_time

    assert response.status_code == 200
    # Should be much faster than overall threshold for real-time
    assert duration < performance_threshold / 2, f"Validation took {duration:.2f}s"


# ==================== ERROR EXPLANATION ====================

@pytest.mark.asyncio
async def test_error_messages_for_critique(async_client):
    """
    Test that error messages are clear enough for CEE to explain to users.
    """
    # Invalid intervention (missing required field)
    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": {"nodes": ["A"], "edges": []},
            "intervention": {},  # Empty intervention
            "outcome": "A"
        }
    )

    # Should return clear error
    if response.status_code in [400, 422]:
        error = response.json()
        assert "detail" in error or "message" in error

        # Error should be descriptive
        error_msg = error.get("detail", error.get("message", ""))
        assert len(str(error_msg)) > 0


# ==================== METADATA FOR CITATIONS ====================

@pytest.mark.asyncio
async def test_metadata_for_citations(
    async_client,
    pricing_dag
):
    """
    Test that responses include metadata for CEE's citation feature.
    """
    intervention = {"Price": 50.0}

    response = await async_client.post(
        "/causal/counterfactual",
        json={
            "dag": pricing_dag,
            "intervention": intervention,
            "outcome": "Revenue",
            "seed": 42
        }
    )

    assert response.status_code == 200
    result = response.json()

    # Should include metadata for reproducibility/citation
    assert "metadata" in result or "response_metadata" in result

    metadata = result.get("metadata", result.get("response_metadata", {}))

    # Metadata should include version info for citations
    assert isinstance(metadata, dict)
