"""
Integration tests for transportability analysis endpoint.

NOTE: Tests converted to async to avoid Starlette TestClient async middleware bug.
Uses httpx.AsyncClient with pytest-asyncio.
"""

import pytest


@pytest.fixture
def simple_domain_spec():
    """Simple domain specification for testing."""
    return {
        "name": "UK",
        "dag": {
            "nodes": ["Price", "Revenue"],
            "edges": [["Price", "Revenue"]],
        },
        "data_summary": {
            "n_samples": 1000,
            "available_variables": ["Price", "Revenue"],
        },
    }


@pytest.fixture
def simple_target_domain():
    """Simple target domain specification."""
    return {
        "name": "Germany",
        "dag": {
            "nodes": ["Price", "Revenue"],
            "edges": [["Price", "Revenue"]],
        },
        "data_summary": {
            "n_samples": 800,
            "available_variables": ["Price", "Revenue"],
        },
    }


@pytest.fixture
def domain_with_confounder():
    """Domain with confounder specification."""
    return {
        "name": "UK",
        "dag": {
            "nodes": ["Price", "Revenue", "Brand"],
            "edges": [
                ["Brand", "Price"],
                ["Brand", "Revenue"],
                ["Price", "Revenue"],
            ],
        },
        "data_summary": {
            "n_samples": 1000,
            "available_variables": ["Price", "Revenue", "Brand"],
        },
    }


@pytest.fixture
def target_with_confounder():
    """Target domain with confounder specification."""
    return {
        "name": "Germany",
        "dag": {
            "nodes": ["Price", "Revenue", "Brand"],
            "edges": [
                ["Brand", "Price"],
                ["Brand", "Revenue"],
                ["Price", "Revenue"],
            ],
        },
        "data_summary": {
            "n_samples": 800,
            "available_variables": ["Price", "Revenue", "Brand"],
        },
    }


class TestDirectTransport:
    """Tests for direct transport scenarios."""

    @pytest.mark.asyncio
    async def test_direct_transport_simple_dag(
        self, client, simple_domain_spec, simple_target_domain
    ):
        """Test direct transport with identical simple DAGs."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["transportable"] is True
        assert data["method"] == "direct"
        assert data["formula"] is not None
        assert "P_target" in data["formula"]
        assert "P_source" in data["formula"]
        assert data["robustness"] in ["robust", "moderate", "fragile"]
        assert len(data["required_assumptions"]) > 0

    @pytest.mark.asyncio
    async def test_direct_transport_has_assumptions(
        self, client, simple_domain_spec, simple_target_domain
    ):
        """Test that direct transport includes required assumptions."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check assumptions structure
        assumptions = data["required_assumptions"]
        assert len(assumptions) > 0

        for assumption in assumptions:
            assert "type" in assumption
            assert "description" in assumption
            assert "critical" in assumption
            assert "testable" in assumption
            assert isinstance(assumption["critical"], bool)
            assert isinstance(assumption["testable"], bool)

    @pytest.mark.asyncio
    async def test_direct_transport_explanation(
        self, client, simple_domain_spec, simple_target_domain
    ):
        """Test that direct transport includes explanation."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "explanation" in data
        assert "summary" in data["explanation"]
        assert "reasoning" in data["explanation"]
        assert "technical_basis" in data["explanation"]
        assert "assumptions" in data["explanation"]
        assert len(data["explanation"]["summary"]) > 0


class TestSelectionDiagramTransport:
    """Tests for selection diagram transport."""

    @pytest.mark.asyncio
    async def test_selection_diagram_with_explicit_variables(
        self, client, simple_domain_spec, simple_target_domain
    ):
        """Test transport with explicitly specified selection variables."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
                "selection_variables": ["MarketSize"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["transportable"] is True
        assert data["method"] == "selection_diagram"
        assert data["formula"] is not None
        assert "MarketSize" in data["formula"]

    @pytest.mark.asyncio
    async def test_selection_diagram_assumptions(
        self, client, domain_with_confounder, target_with_confounder
    ):
        """Test assumptions for selection diagram transport."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": domain_with_confounder,
                "target_domain": target_with_confounder,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        if data["transportable"]:
            assumption_types = [a["type"] for a in data["required_assumptions"]]
            # Should have core assumptions
            assert "same_mechanism" in assumption_types
            assert "common_support" in assumption_types


class TestNonTransportable:
    """Tests for non-transportable scenarios."""

    @pytest.mark.asyncio
    async def test_different_dag_structures(self, client):
        """Test with different DAG structures between domains."""
        source_domain = {
            "name": "UK",
            "dag": {
                "nodes": ["Price", "Revenue"],
                "edges": [["Price", "Revenue"]],
            },
        }

        target_domain = {
            "name": "Germany",
            "dag": {
                "nodes": ["Price", "Revenue", "Regulation"],
                "edges": [
                    ["Price", "Revenue"],
                    ["Regulation", "Revenue"],
                ],
            },
        }

        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": source_domain,
                "target_domain": target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["transportable"] is False
        assert data["reason"] is not None
        assert data["suggestions"] is not None
        assert len(data["suggestions"]) > 0

    @pytest.mark.asyncio
    async def test_non_transportable_has_suggestions(self, client):
        """Test that non-transportable case includes suggestions."""
        source_domain = {
            "name": "UK",
            "dag": {
                "nodes": ["Price", "Revenue"],
                "edges": [["Price", "Revenue"]],
            },
        }

        target_domain = {
            "name": "Germany",
            "dag": {
                "nodes": ["Price", "Revenue", "Tax"],
                "edges": [
                    ["Price", "Revenue"],
                    ["Tax", "Price"],
                ],
            },
        }

        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": source_domain,
                "target_domain": target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        if not data["transportable"]:
            assert data["suggestions"] is not None
            assert len(data["suggestions"]) >= 2
            # Should suggest investigating differences
            suggestions_text = " ".join(data["suggestions"]).lower()
            assert "differ" in suggestions_text or "structural" in suggestions_text


class TestResponseStructure:
    """Tests for response structure and metadata."""

    @pytest.mark.asyncio
    async def test_response_has_metadata(
        self, client, simple_domain_spec, simple_target_domain
    ):
        """Test that response includes metadata."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should have metadata (aliased as _metadata)
        assert "_metadata" in data
        metadata = data["_metadata"]
        assert "request_id" in metadata
        assert "timestamp" in metadata

    @pytest.mark.asyncio
    async def test_confidence_levels(
        self, client, simple_domain_spec, simple_target_domain
    ):
        """Test that confidence is one of the expected values."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["confidence"] in ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_robustness_levels(
        self, client, simple_domain_spec, simple_target_domain
    ):
        """Test that robustness is one of the expected values."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["robustness"] in ["robust", "moderate", "fragile"]


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.asyncio
    async def test_deterministic_results(
        self, client, simple_domain_spec, simple_target_domain
    ):
        """Test that same input produces same output."""
        request_data = {
            "source_domain": simple_domain_spec,
            "target_domain": simple_target_domain,
            "treatment": "Price",
            "outcome": "Revenue",
        }

        response1 = await client.post("/api/v1/causal/transport", json=request_data)
        response2 = await client.post("/api/v1/causal/transport", json=request_data)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        assert data1["transportable"] == data2["transportable"]
        assert data1["method"] == data2["method"]
        assert data1["formula"] == data2["formula"]
        assert data1["robustness"] == data2["robustness"]
        assert data1["confidence"] == data2["confidence"]


class TestRequestValidation:
    """Tests for request validation."""

    @pytest.mark.asyncio
    async def test_missing_treatment(self, client, simple_domain_spec, simple_target_domain):
        """Test error handling with missing treatment."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "outcome": "Revenue",
                # Missing treatment
            },
        )

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_outcome(self, client, simple_domain_spec, simple_target_domain):
        """Test error handling with missing outcome."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "treatment": "Price",
                # Missing outcome
            },
        )

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_source_domain(self, client, simple_target_domain):
        """Test error handling with missing source domain."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "target_domain": simple_target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
                # Missing source_domain
            },
        )

        # Should return validation error
        assert response.status_code == 422


class TestRequestTracing:
    """Tests for request tracing."""

    @pytest.mark.asyncio
    async def test_custom_request_id(
        self, client, simple_domain_spec, simple_target_domain
    ):
        """Test that custom request ID is preserved in metadata."""
        custom_request_id = "test-transport-123"

        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": simple_domain_spec,
                "target_domain": simple_target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
            headers={"X-Request-Id": custom_request_id},
        )

        assert response.status_code == 200
        data = response.json()

        # Check metadata contains custom request ID
        assert "_metadata" in data
        assert data["_metadata"]["request_id"] == custom_request_id


class TestComplexScenarios:
    """Tests for complex transportability scenarios."""

    @pytest.mark.asyncio
    async def test_larger_dag(self, client):
        """Test with a larger DAG."""
        source_domain = {
            "name": "Source",
            "dag": {
                "nodes": ["A", "B", "C", "D", "E"],
                "edges": [
                    ["A", "B"],
                    ["B", "C"],
                    ["C", "D"],
                    ["D", "E"],
                ],
            },
        }

        target_domain = {
            "name": "Target",
            "dag": {
                "nodes": ["A", "B", "C", "D", "E"],
                "edges": [
                    ["A", "B"],
                    ["B", "C"],
                    ["C", "D"],
                    ["D", "E"],
                ],
            },
        }

        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": source_domain,
                "target_domain": target_domain,
                "treatment": "A",
                "outcome": "E",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["transportable"], bool)

    @pytest.mark.asyncio
    async def test_with_data_summaries(self, client):
        """Test transport with detailed data summaries."""
        source_domain = {
            "name": "UK",
            "dag": {
                "nodes": ["Price", "Revenue"],
                "edges": [["Price", "Revenue"]],
            },
            "data_summary": {
                "n_samples": 5000,
                "available_variables": ["Price", "Revenue"],
                "notes": ["High quality data", "Recent collection"],
            },
        }

        target_domain = {
            "name": "Germany",
            "dag": {
                "nodes": ["Price", "Revenue"],
                "edges": [["Price", "Revenue"]],
            },
            "data_summary": {
                "n_samples": 3000,
                "available_variables": ["Price", "Revenue"],
                "notes": ["Older data", "Some missing values"],
            },
        }

        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": source_domain,
                "target_domain": target_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should complete successfully with data summaries
        assert isinstance(data["transportable"], bool)
        assert data["confidence"] in ["high", "medium", "low"]
