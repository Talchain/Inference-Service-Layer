"""
P2 Verification Integration Tests.

Verifies that P2 implementation is correctly wired to the endpoints PLoT calls.
Tests actual HTTP responses from the /api/v1/robustness/analyze/v2 endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


# Valid request payload for V2 endpoint (with dual uncertainty: exists_probability + strength)
VALID_V2_REQUEST = {
    "graph": {
        "nodes": [
            {"id": "price", "kind": "factor", "label": "Price"},
            {"id": "sales", "kind": "goal", "label": "Sales"},
        ],
        "edges": [
            {
                "from": "price",
                "to": "sales",
                "exists_probability": 0.95,  # Required for V2 schema
                "strength": {"mean": -0.8, "std": 0.2},
            }
        ],
    },
    "options": [
        {"id": "opt1", "label": "Raise price", "interventions": {"price": 120}},
        {"id": "opt2", "label": "Lower price", "interventions": {"price": 80}},
    ],
    "goal_node_id": "sales",
    "seed": 42,
    "n_samples": 100,  # Keep small for test speed
}

# Request that triggers 422 via our RequestValidator (graph cycle)
# This bypasses Pydantic validation to test our custom ISLV2Error422 format
INVALID_REQUEST_422 = {
    "graph": {
        "nodes": [
            {"id": "a", "kind": "factor", "label": "A"},
            {"id": "b", "kind": "factor", "label": "B"},
            {"id": "c", "kind": "goal", "label": "C"},
        ],
        "edges": [
            # Create a cycle: a -> b -> c -> a
            {
                "from": "a",
                "to": "b",
                "exists_probability": 0.95,
                "strength": {"mean": 0.5, "std": 0.1},
            },
            {
                "from": "b",
                "to": "c",
                "exists_probability": 0.95,
                "strength": {"mean": 0.5, "std": 0.1},
            },
            {
                "from": "c",
                "to": "a",
                "exists_probability": 0.95,
                "strength": {"mean": 0.5, "std": 0.1},
            },
        ],
    },
    "options": [
        {"id": "opt1", "label": "Option 1", "interventions": {"a": 10}},
        {"id": "opt2", "label": "Option 2", "interventions": {"a": 20}},
    ],
    "goal_node_id": "c",
    "seed": 42,
    "n_samples": 100,
}


class TestP2Task1EndpointExists:
    """Task 1: Verify the endpoint exists and is accessible."""

    def test_endpoint_exists(self, client):
        """Endpoint /api/v1/robustness/analyze/v2 exists."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={"X-ISL-Response-Version": "2"},
        )
        # Should not be 404
        assert response.status_code != 404, "Endpoint not found"
        # Should be 200 or 422 (not 405 Method Not Allowed)
        assert response.status_code in (200, 422), f"Unexpected status: {response.status_code}"


class TestP2Task2V2ResponseFormat:
    """Task 2: Verify V2 response format on actual endpoint."""

    def test_response_has_version_field(self, client):
        """Response contains 'version' field (alias for response_schema_version)."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={
                "X-ISL-Response-Version": "2",
                "X-Request-Id": "verify-v2-format-001",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "version" in data, "Missing 'version' field in response"
        assert data["version"] == "2.0", f"Expected version '2.0', got '{data.get('version')}'"

    def test_response_has_analysis_status(self, client):
        """Response contains 'analysis_status' field."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "analysis_status" in data, "Missing 'analysis_status' field"
        assert data["analysis_status"] in ("computed", "partial", "failed")

    def test_response_has_timestamp_iso8601(self, client):
        """Response contains 'timestamp' in ISO 8601 format ending with Z."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data, "Missing 'timestamp' field"
        assert data["timestamp"].endswith("Z"), f"Timestamp not UTC: {data['timestamp']}"

    def test_response_has_seed_used(self, client):
        """Response contains 'seed_used' field."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "seed_used" in data, "Missing 'seed_used' field"
        assert data["seed_used"] == "42", f"Expected seed_used '42', got '{data.get('seed_used')}'"

    def test_response_no_response_hash(self, client):
        """Response does NOT contain 'response_hash' (PLoT owns this)."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response_hash" not in data, "response_hash should NOT be present (PLoT owns it)"


class TestP2Task3RequestIdTracing:
    """Task 3: Verify request ID tracing headers."""

    def test_request_id_echoed_in_header(self, client):
        """X-Request-Id header in response matches sent header."""
        test_id = "verify-trace-001"
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={
                "X-ISL-Response-Version": "2",
                "X-Request-Id": test_id,
            },
        )
        assert response.status_code == 200
        assert "x-request-id" in response.headers, "Missing X-Request-Id header in response"
        assert response.headers["x-request-id"] == test_id

    def test_request_id_echoed_in_body(self, client):
        """request_id in response body matches sent header."""
        test_id = "verify-trace-002"
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={
                "X-ISL-Response-Version": "2",
                "X-Request-Id": test_id,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data, "Missing 'request_id' field in response body"
        assert data["request_id"] == test_id

    def test_processing_time_header_present(self, client):
        """X-Processing-Time-Ms header is present."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        assert "x-processing-time-ms" in response.headers, "Missing X-Processing-Time-Ms header"
        # Should be a valid integer (as string)
        time_ms = response.headers["x-processing-time-ms"]
        assert time_ms.isdigit(), f"Processing time should be integer string, got: {time_ms}"


class TestP2Task4UnwrappedError422:
    """Task 4: Verify 422 returns unwrapped ISLV2Error422 schema."""

    def test_422_status_code(self, client):
        """Invalid request returns HTTP 422."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=INVALID_REQUEST_422,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_422_has_analysis_status_blocked(self, client):
        """422 response has analysis_status: 'blocked' at top level."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=INVALID_REQUEST_422,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "analysis_status" in data, "Missing 'analysis_status' at top level"
        assert data["analysis_status"] == "blocked", f"Expected 'blocked', got '{data['analysis_status']}'"

    def test_422_has_critiques_at_top_level(self, client):
        """422 response has critiques array at top level."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=INVALID_REQUEST_422,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "critiques" in data, "Missing 'critiques' at top level"
        assert isinstance(data["critiques"], list), "critiques should be a list"
        assert len(data["critiques"]) > 0, "critiques should not be empty for 422"

    def test_422_no_error_wrapper(self, client):
        """422 response is NOT wrapped in 'error' envelope."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=INVALID_REQUEST_422,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "error" not in data, "Response should NOT have 'error' wrapper"

    def test_422_no_success_field(self, client):
        """422 response does NOT have 'success: false' field."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=INVALID_REQUEST_422,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422
        data = response.json()
        assert "success" not in data, "Response should NOT have 'success' field"

    def test_422_has_request_id(self, client):
        """422 response includes request_id."""
        test_id = "verify-422-format-001"
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=INVALID_REQUEST_422,
            headers={
                "X-ISL-Response-Version": "2",
                "X-Request-Id": test_id,
            },
        )
        assert response.status_code == 422
        data = response.json()
        assert "request_id" in data, "Missing 'request_id' in 422 response"
        assert data["request_id"] == test_id


class TestP2Task5ProcessingTimeOnBoth:
    """Task 5: Verify processing time header on both 200 and 422."""

    def test_processing_time_on_200(self, client):
        """200 response includes X-Processing-Time-Ms header."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=VALID_V2_REQUEST,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        assert "x-processing-time-ms" in response.headers

    def test_processing_time_on_422(self, client):
        """422 response includes X-Processing-Time-Ms header."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=INVALID_REQUEST_422,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422
        assert "x-processing-time-ms" in response.headers


class TestSeedTruthfulness:
    """Verify seed_used in response matches actual seed used for analysis.

    This is a critical reproducibility requirement: users must be able to
    reproduce results by using the reported seed.
    """

    def test_explicit_seed_echoed_correctly(self, client):
        """When explicit seed provided, seed_used matches exactly."""
        request_with_explicit_seed = {
            **VALID_V2_REQUEST,
            "seed": 12345,
        }
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request_with_explicit_seed,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["seed_used"] == "12345"

    def test_computed_seed_when_no_explicit_seed(self, client):
        """When no seed provided, seed_used shows computed seed (not hardcoded)."""
        # Remove seed from request
        request_without_seed = {
            k: v for k, v in VALID_V2_REQUEST.items() if k != "seed"
        }

        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request_without_seed,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        data = response.json()

        # seed_used should NOT be "42" (old hardcoded value)
        # It should be a computed hash from the graph
        assert data["seed_used"] != "42", "seed_used should not be hardcoded '42'"
        assert data["seed_used"].isdigit(), "seed_used should be a numeric string"
        # Graph-derived seed should be a reasonable integer
        seed_int = int(data["seed_used"])
        assert 0 <= seed_int < 2**32, "seed should be 32-bit unsigned integer"

    def test_computed_seed_is_deterministic(self, client):
        """Same graph structure should produce same computed seed."""
        request_without_seed = {
            k: v for k, v in VALID_V2_REQUEST.items() if k != "seed"
        }

        # Make two requests with same graph
        response1 = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request_without_seed,
            headers={"X-ISL-Response-Version": "2"},
        )
        response2 = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request_without_seed,
            headers={"X-ISL-Response-Version": "2"},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Same graph → same computed seed
        assert data1["seed_used"] == data2["seed_used"]

    def test_reproducibility_with_same_seed(self, client):
        """Same request + same seed → identical deterministic results."""
        request_with_seed = {
            **VALID_V2_REQUEST,
            "seed": 99999,
            "n_samples": 500,  # More samples for stable comparison
        }

        response1 = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request_with_seed,
            headers={"X-ISL-Response-Version": "2"},
        )
        response2 = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request_with_seed,
            headers={"X-ISL-Response-Version": "2"},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Same seed should produce identical results
        assert data1["seed_used"] == data2["seed_used"] == "99999"

        # Outcome distributions should be identical
        for i, opt1 in enumerate(data1["options"]):
            opt2 = data2["options"][i]
            assert opt1["outcome"]["mean"] == opt2["outcome"]["mean"]
            assert opt1["outcome"]["std"] == opt2["outcome"]["std"]
            assert opt1["outcome"]["p10"] == opt2["outcome"]["p10"]
            assert opt1["outcome"]["p50"] == opt2["outcome"]["p50"]
            assert opt1["outcome"]["p90"] == opt2["outcome"]["p90"]

        # Robustness should be identical
        assert data1["robustness"]["level"] == data2["robustness"]["level"]
        assert data1["robustness"]["confidence"] == data2["robustness"]["confidence"]

    def test_different_graphs_produce_different_computed_seeds(self, client):
        """Different graph structures should produce different computed seeds."""
        request1 = {
            k: v for k, v in VALID_V2_REQUEST.items() if k != "seed"
        }

        # Create different graph with different edge probability
        # (but same structure so options remain valid)
        request2 = {
            k: v for k, v in VALID_V2_REQUEST.items() if k != "seed"
        }
        # Modify the edge probability to change the graph hash
        request2["graph"] = {
            "nodes": [
                {"id": "price", "kind": "factor", "label": "Price"},
                {"id": "sales", "kind": "goal", "label": "Sales"},
            ],
            "edges": [
                {
                    "from": "price",
                    "to": "sales",
                    "exists_probability": 0.50,  # Different from 0.95 in VALID_V2_REQUEST
                    "strength": {"mean": -0.8, "std": 0.2},
                }
            ],
        }

        response1 = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request1,
            headers={"X-ISL-Response-Version": "2"},
        )
        response2 = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request2,
            headers={"X-ISL-Response-Version": "2"},
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Different graphs → different computed seeds
        assert data1["seed_used"] != data2["seed_used"]


# Request payload with goal_threshold for testing probability_of_goal
V2_REQUEST_WITH_GOAL_THRESHOLD = {
    "graph": {
        "nodes": [
            {"id": "input", "kind": "factor", "label": "Input", "observed_state": {"value": 100.0}},
            {"id": "output", "kind": "goal", "label": "Output"},
        ],
        "edges": [
            {
                "from": "input",
                "to": "output",
                "exists_probability": 1.0,
                "strength": {"mean": 2.0, "std": 0.01},  # Output ~ 2 * input
            }
        ],
    },
    "options": [
        {"id": "low", "label": "Low Input", "interventions": {"input": 50}},
        {"id": "high", "label": "High Input", "interventions": {"input": 150}},
    ],
    "goal_node_id": "output",
    "seed": 42,
    "n_samples": 100,
    "goal_threshold": 200.0,  # Between low output (100) and high output (300)
}


class TestGoalThresholdProbabilityEndpoint:
    """Integration tests for goal_threshold and probability_of_goal feature."""

    def test_probability_of_goal_included_when_threshold_provided(self, client):
        """When goal_threshold is provided, probability_of_goal should be in each result."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=V2_REQUEST_WITH_GOAL_THRESHOLD,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        data = response.json()

        # V2 response uses 'options' field for results
        assert "options" in data, f"Expected 'options' in response, got keys: {list(data.keys())}"
        for option in data["options"]:
            assert "probability_of_goal" in option, (
                f"probability_of_goal missing from result for option {option['id']}"
            )
            prob = option["probability_of_goal"]
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0

    def test_probability_of_goal_values_correct_for_options(self, client):
        """Low option should have low probability, high option should have high probability."""
        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=V2_REQUEST_WITH_GOAL_THRESHOLD,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        data = response.json()

        # V2 response uses 'options' field and 'id' (not 'option_id')
        results_by_id = {r["id"]: r for r in data["options"]}

        # Low option: input=50, output~100, threshold=200 → should rarely meet
        low_prob = results_by_id["low"]["probability_of_goal"]
        assert low_prob < 0.1, f"Expected low probability for 'low' option, got {low_prob}"

        # High option: input=150, output~300, threshold=200 → should almost always meet
        high_prob = results_by_id["high"]["probability_of_goal"]
        assert high_prob > 0.99, f"Expected high probability for 'high' option, got {high_prob}"

    def test_probability_of_goal_absent_when_no_threshold(self, client):
        """When goal_threshold is not provided, probability_of_goal should be absent."""
        # Create request without goal_threshold
        request = {
            k: v for k, v in V2_REQUEST_WITH_GOAL_THRESHOLD.items() if k != "goal_threshold"
        }

        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=request,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 200
        data = response.json()

        # V2 response uses 'options' field
        assert "options" in data, f"Expected 'options' in response, got keys: {list(data.keys())}"
        for option in data["options"]:
            assert "probability_of_goal" not in option, (
                f"probability_of_goal should be absent when threshold not provided, "
                f"but found in option {option['id']}"
            )

    def test_invalid_goal_threshold_rejected(self, client):
        """NaN/inf goal_threshold should return 422 validation error."""
        # NaN threshold
        nan_request = dict(V2_REQUEST_WITH_GOAL_THRESHOLD)
        nan_request["goal_threshold"] = float("nan")

        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=nan_request,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422

        # Infinity threshold
        inf_request = dict(V2_REQUEST_WITH_GOAL_THRESHOLD)
        inf_request["goal_threshold"] = float("inf")

        response = client.post(
            "/api/v1/robustness/analyze/v2",
            json=inf_request,
            headers={"X-ISL-Response-Version": "2"},
        )
        assert response.status_code == 422
