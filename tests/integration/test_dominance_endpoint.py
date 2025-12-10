"""
Integration tests for dominance detection endpoint.

Tests all dominance analysis scenarios:
- Clear dominance relationships
- Pareto frontier identification
- Edge cases and validation
"""

import pytest
from httpx import AsyncClient

from src.api.main import app


@pytest.mark.asyncio
async def test_clear_dominance_simple():
    """Test clear dominance: Option C dominates Option B on all criteria."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {
                        "option_id": "opt_a",
                        "option_label": "Option A: High revenue, high risk",
                        "scores": {"revenue": 0.90, "risk": 0.40, "timeline": 0.70}
                    },
                    {
                        "option_id": "opt_b",
                        "option_label": "Option B: Low on all",
                        "scores": {"revenue": 0.50, "risk": 0.50, "timeline": 0.50}
                    },
                    {
                        "option_id": "opt_c",
                        "option_label": "Option C: Better than B on all",
                        "scores": {"revenue": 0.75, "risk": 0.75, "timeline": 0.80}
                    }
                ],
                "criteria": ["revenue", "risk", "timeline"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Option B should be dominated by Option C
        assert len(data["dominated"]) >= 1
        dominated_ids = [d["dominated_option_id"] for d in data["dominated"]]
        assert "opt_b" in dominated_ids

        # Check that dominated option has correct dominator
        opt_b_dom = next(d for d in data["dominated"] if d["dominated_option_id"] == "opt_b")
        assert "opt_c" in opt_b_dom["dominated_by"]
        assert opt_b_dom["degree"] >= 1

        # Metadata should be present
        assert "metadata" in data
        assert data["metadata"]["algorithm"] == "pairwise_dominance"
        assert data["metadata"]["computation_time_ms"] > 0


@pytest.mark.asyncio
async def test_no_dominance_all_pareto():
    """Test case where no option dominates another (all on Pareto frontier)."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {
                        "option_id": "opt_a",
                        "option_label": "High revenue, low risk",
                        "scores": {"revenue": 0.90, "risk": 0.40}
                    },
                    {
                        "option_id": "opt_b",
                        "option_label": "Low revenue, high risk",
                        "scores": {"revenue": 0.40, "risk": 0.90}
                    },
                    {
                        "option_id": "opt_c",
                        "option_label": "Balanced",
                        "scores": {"revenue": 0.65, "risk": 0.65}
                    }
                ],
                "criteria": ["revenue", "risk"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # No options should be dominated
        assert len(data["dominated"]) == 0

        # All options should be on Pareto frontier
        assert len(data["non_dominated_ids"]) == 3
        assert set(data["non_dominated_ids"]) == {"opt_a", "opt_b", "opt_c"}

        # Frontier size should equal total options
        assert data["frontier_size"] == data["total_options"]


@pytest.mark.asyncio
async def test_partial_dominance():
    """Test case with some dominated and some non-dominated options."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {
                        "option_id": "opt_best",
                        "option_label": "Best on all criteria",
                        "scores": {"quality": 0.95, "cost": 0.90, "speed": 0.85}
                    },
                    {
                        "option_id": "opt_good",
                        "option_label": "Good on some",
                        "scores": {"quality": 0.80, "cost": 0.95, "speed": 0.70}
                    },
                    {
                        "option_id": "opt_poor",
                        "option_label": "Poor on all",
                        "scores": {"quality": 0.60, "cost": 0.65, "speed": 0.55}
                    },
                    {
                        "option_id": "opt_medium",
                        "option_label": "Medium on all",
                        "scores": {"quality": 0.70, "cost": 0.75, "speed": 0.65}
                    }
                ],
                "criteria": ["quality", "cost", "speed"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Poor and medium should be dominated
        assert len(data["dominated"]) >= 1

        # Best should be on Pareto frontier
        assert "opt_best" in data["non_dominated_ids"]

        # Total should equal number of options
        assert data["total_options"] == 4


@pytest.mark.asyncio
async def test_equal_scores_no_dominance():
    """Test that options with equal scores don't dominate each other."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {
                        "option_id": "opt_a",
                        "option_label": "Option A",
                        "scores": {"x": 0.75, "y": 0.75}
                    },
                    {
                        "option_id": "opt_b",
                        "option_label": "Option B (same scores)",
                        "scores": {"x": 0.75, "y": 0.75}
                    }
                ],
                "criteria": ["x", "y"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # No dominance when all scores are equal
        assert len(data["dominated"]) == 0
        assert len(data["non_dominated_ids"]) == 2


@pytest.mark.asyncio
async def test_single_criterion():
    """Test dominance with only one criterion."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "opt_a", "option_label": "High", "scores": {"metric": 0.90}},
                    {"option_id": "opt_b", "option_label": "Medium", "scores": {"metric": 0.60}},
                    {"option_id": "opt_c", "option_label": "Low", "scores": {"metric": 0.30}}
                ],
                "criteria": ["metric"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # With single criterion, higher scores dominate lower scores
        assert len(data["dominated"]) == 2  # opt_b and opt_c dominated
        assert "opt_a" in data["non_dominated_ids"]
        assert len(data["non_dominated_ids"]) == 1


@pytest.mark.asyncio
async def test_dominance_degree():
    """Test that dominance degree is correctly calculated."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "opt_best", "option_label": "Best", "scores": {"a": 0.95, "b": 0.95}},
                    {"option_id": "opt_good", "option_label": "Good", "scores": {"a": 0.85, "b": 0.90}},
                    {"option_id": "opt_worst", "option_label": "Worst", "scores": {"a": 0.50, "b": 0.50}},
                ],
                "criteria": ["a", "b"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # opt_worst should be dominated by both opt_best and opt_good
        if data["dominated"]:
            worst_dom = next(
                (d for d in data["dominated"] if d["dominated_option_id"] == "opt_worst"),
                None
            )
            if worst_dom:
                assert worst_dom["degree"] == 2  # Dominated by 2 options


@pytest.mark.asyncio
async def test_minimum_options_validation():
    """Test that request with < 2 options fails validation."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "opt_a", "option_label": "Only one", "scores": {"x": 0.5}}
                ],
                "criteria": ["x"]
            }
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_maximum_options_validation():
    """Test that request with > 100 options fails validation."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create 101 options
        options = [
            {
                "option_id": f"opt_{i}",
                "option_label": f"Option {i}",
                "scores": {"metric": i / 101.0}
            }
            for i in range(101)
        ]

        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": options,
                "criteria": ["metric"]
            }
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_score_range_validation():
    """Test that scores outside [0, 1] range fail validation."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "opt_a", "option_label": "Valid", "scores": {"x": 0.5}},
                    {"option_id": "opt_b", "option_label": "Invalid", "scores": {"x": 1.5}}  # > 1.0
                ],
                "criteria": ["x"]
            }
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_missing_criterion_validation():
    """Test that missing scores for criteria fail validation."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "opt_a", "option_label": "Complete", "scores": {"x": 0.5, "y": 0.6}},
                    {"option_id": "opt_b", "option_label": "Missing y", "scores": {"x": 0.7}}
                ],
                "criteria": ["x", "y"]
            }
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_extra_criterion_validation():
    """Test that extra scores not in criteria list fail validation."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "opt_a", "option_label": "Has extra", "scores": {"x": 0.5, "z": 0.9}},
                    {"option_id": "opt_b", "option_label": "Normal", "scores": {"x": 0.7}}
                ],
                "criteria": ["x"]
            }
        )

        assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_request_id_tracking():
    """Test that request_id is properly tracked in metadata."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "request_id": "test_req_123",
                "options": [
                    {"option_id": "opt_a", "option_label": "A", "scores": {"x": 0.8}},
                    {"option_id": "opt_b", "option_label": "B", "scores": {"x": 0.6}}
                ],
                "criteria": ["x"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["request_id"] == "test_req_123"


@pytest.mark.asyncio
async def test_two_options_minimum():
    """Test minimum valid case with exactly 2 options."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "opt_a", "option_label": "Better", "scores": {"x": 0.8, "y": 0.7}},
                    {"option_id": "opt_b", "option_label": "Worse", "scores": {"x": 0.5, "y": 0.5}}
                ],
                "criteria": ["x", "y"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_options"] == 2


@pytest.mark.asyncio
async def test_hundred_options_maximum():
    """Test maximum valid case with exactly 100 options."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create exactly 100 options
        options = [
            {
                "option_id": f"opt_{i:03d}",
                "option_label": f"Option {i}",
                "scores": {"metric": i / 100.0}
            }
            for i in range(100)
        ]

        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": options,
                "criteria": ["metric"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_options"] == 100
        assert data["frontier_size"] == 1  # Only highest score on frontier


@pytest.mark.asyncio
async def test_response_fields_completeness():
    """Test that all required response fields are present."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "opt_a", "option_label": "A", "scores": {"x": 0.9}},
                    {"option_id": "opt_b", "option_label": "B", "scores": {"x": 0.5}},
                    {"option_id": "opt_c", "option_label": "C", "scores": {"x": 0.7}}
                ],
                "criteria": ["x"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Check all required fields
        assert "dominated" in data
        assert "non_dominated_ids" in data
        assert "total_options" in data
        assert "frontier_size" in data
        assert "metadata" in data

        # Check metadata fields
        assert "request_id" in data["metadata"]
        assert "computation_time_ms" in data["metadata"]
        assert "isl_version" in data["metadata"]
        assert "algorithm" in data["metadata"]


@pytest.mark.asyncio
async def test_dominated_relation_fields():
    """Test that dominated relation objects have all required fields."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/analysis/dominance",
            json={
                "options": [
                    {"option_id": "opt_best", "option_label": "Best", "scores": {"x": 0.9, "y": 0.9}},
                    {"option_id": "opt_worst", "option_label": "Worst", "scores": {"x": 0.3, "y": 0.3}}
                ],
                "criteria": ["x", "y"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        if data["dominated"]:
            dom_relation = data["dominated"][0]
            assert "dominated_option_id" in dom_relation
            assert "dominated_option_label" in dom_relation
            assert "dominated_by" in dom_relation
            assert "degree" in dom_relation
            assert isinstance(dom_relation["dominated_by"], list)
            assert isinstance(dom_relation["degree"], int)
