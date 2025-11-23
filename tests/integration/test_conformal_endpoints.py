"""
Integration tests for conformal prediction endpoints.

Tests the full API including:
- Request/response handling
- Validation
- Error cases
- Determinism
- Metadata
"""

import pytest


@pytest.fixture
def simple_conformal_request():
    """Simple conformal prediction request."""
    return {
        "model": {
            "variables": ["Price", "Revenue"],
            "equations": {"Revenue": "10000 + 500*Price"},
            "distributions": {
                "noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
            },
        },
        "intervention": {"Price": 50},
        "calibration_data": [
            {"inputs": {"Price": 40}, "outcome": {"Revenue": 30000}},
            {"inputs": {"Price": 42}, "outcome": {"Revenue": 31000}},
            {"inputs": {"Price": 45}, "outcome": {"Revenue": 32500}},
            {"inputs": {"Price": 48}, "outcome": {"Revenue": 34000}},
            {"inputs": {"Price": 50}, "outcome": {"Revenue": 35000}},
            {"inputs": {"Price": 52}, "outcome": {"Revenue": 36000}},
            {"inputs": {"Price": 55}, "outcome": {"Revenue": 37500}},
            {"inputs": {"Price": 58}, "outcome": {"Revenue": 39000}},
            {"inputs": {"Price": 60}, "outcome": {"Revenue": 40000}},
            {"inputs": {"Price": 62}, "outcome": {"Revenue": 41000}},
            {"inputs": {"Price": 65}, "outcome": {"Revenue": 42500}},
            {"inputs": {"Price": 68}, "outcome": {"Revenue": 44000}},
            {"inputs": {"Price": 70}, "outcome": {"Revenue": 45000}},
            {"inputs": {"Price": 72}, "outcome": {"Revenue": 46000}},
            {"inputs": {"Price": 75}, "outcome": {"Revenue": 47500}},
            {"inputs": {"Price": 78}, "outcome": {"Revenue": 49000}},
            {"inputs": {"Price": 80}, "outcome": {"Revenue": 50000}},
            {"inputs": {"Price": 82}, "outcome": {"Revenue": 51000}},
            {"inputs": {"Price": 85}, "outcome": {"Revenue": 52500}},
            {"inputs": {"Price": 88}, "outcome": {"Revenue": 54000}},
        ],
        "confidence_level": 0.95,
        "method": "split",
        "samples": 1000,
        "seed": 42,
    }


class TestConformalEndpointBasic:
    """Basic tests for conformal prediction endpoint."""

    @pytest.mark.asyncio
    async def test_conformal_endpoint_success(self, client, simple_conformal_request):
        """Test successful conformal prediction request."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        assert "prediction_interval" in data
        assert "coverage_guarantee" in data
        assert "calibration_quality" in data
        assert "comparison_to_standard" in data
        assert "explanation" in data

    @pytest.mark.asyncio
    async def test_conformal_endpoint_has_prediction_interval(self, client, simple_conformal_request):
        """Test that response includes prediction interval."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        interval = data["prediction_interval"]
        assert "lower_bound" in interval
        assert "upper_bound" in interval
        assert "point_estimate" in interval
        assert "interval_width" in interval

        # Check that Revenue is the outcome
        assert "Revenue" in interval["lower_bound"]
        assert "Revenue" in interval["upper_bound"]

    @pytest.mark.asyncio
    async def test_conformal_endpoint_has_coverage_guarantee(self, client, simple_conformal_request):
        """Test that response includes coverage guarantee."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        coverage = data["coverage_guarantee"]
        assert "nominal_coverage" in coverage
        assert "guaranteed_coverage" in coverage
        assert "finite_sample_valid" in coverage
        assert "assumptions" in coverage

        assert coverage["nominal_coverage"] == 0.95
        assert coverage["finite_sample_valid"] is True


class TestConformalCalibrationMetrics:
    """Tests for calibration quality metrics."""

    @pytest.mark.asyncio
    async def test_calibration_metrics_present(self, client, simple_conformal_request):
        """Test that calibration metrics are present."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        calibration = data["calibration_quality"]
        assert "calibration_size" in calibration
        assert "residual_statistics" in calibration
        assert "interval_adaptivity" in calibration

    @pytest.mark.asyncio
    async def test_residual_statistics_complete(self, client, simple_conformal_request):
        """Test that residual statistics are complete."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        stats = data["calibration_quality"]["residual_statistics"]
        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert "iqr" in stats

    @pytest.mark.asyncio
    async def test_calibration_size_reasonable(self, client, simple_conformal_request):
        """Test that calibration size is reasonable after split."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        calib_size = data["calibration_quality"]["calibration_size"]
        original_size = len(simple_conformal_request["calibration_data"])

        # After 50/50 split, should be roughly half
        assert calib_size >= original_size * 0.4
        assert calib_size <= original_size * 0.6


class TestConformalComparison:
    """Tests for Monte Carlo comparison."""

    @pytest.mark.asyncio
    async def test_comparison_to_monte_carlo(self, client, simple_conformal_request):
        """Test that comparison to Monte Carlo is present."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        comparison = data["comparison_to_standard"]
        assert "monte_carlo_interval" in comparison
        assert "conformal_interval" in comparison
        assert "width_ratio" in comparison
        assert "interpretation" in comparison

    @pytest.mark.asyncio
    async def test_width_ratio_present(self, client, simple_conformal_request):
        """Test that width ratio is computed."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        width_ratio = data["comparison_to_standard"]["width_ratio"]
        assert "Revenue" in width_ratio
        assert isinstance(width_ratio["Revenue"], (int, float))
        assert width_ratio["Revenue"] > 0


class TestConformalValidation:
    """Tests for request validation."""

    @pytest.mark.asyncio
    async def test_missing_calibration_data(self, client):
        """Test error when calibration data is missing."""
        request = {
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price"},
            },
            "intervention": {"Price": 50},
            "calibration_data": None,
            "confidence_level": 0.95,
        }

        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request,
        )

        assert response.status_code in [400, 422, 500]

    @pytest.mark.asyncio
    async def test_insufficient_calibration_data(self, client):
        """Test error when calibration data is insufficient."""
        request = {
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {"Revenue": "10000 + 500*Price"},
            },
            "intervention": {"Price": 50},
            "calibration_data": [
                {"inputs": {"Price": 40}, "outcome": {"Revenue": 30000}},
                {"inputs": {"Price": 50}, "outcome": {"Revenue": 35000}},
            ],
            "confidence_level": 0.95,
        }

        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request,
        )

        assert response.status_code in [400, 500]
        if response.status_code == 400:
            error_detail = response.json()
            assert "at least 10" in error_detail.get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_invalid_confidence_level(self, client, simple_conformal_request):
        """Test error when confidence level is invalid."""
        request = simple_conformal_request.copy()
        request["confidence_level"] = 1.5  # Invalid (> 1)

        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request,
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_invalid_method(self, client, simple_conformal_request):
        """Test error when method is invalid."""
        request = simple_conformal_request.copy()
        request["method"] = "invalid_method"

        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request,
        )

        assert response.status_code == 422  # Validation error


class TestConformalDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.asyncio
    async def test_deterministic_with_seed(self, client, simple_conformal_request):
        """Test that results are deterministic with seed."""
        response1 = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        response2 = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Intervals should be identical with same seed
        assert data1["prediction_interval"]["lower_bound"] == data2["prediction_interval"]["lower_bound"]
        assert data1["prediction_interval"]["upper_bound"] == data2["prediction_interval"]["upper_bound"]
        assert data1["coverage_guarantee"]["guaranteed_coverage"] == data2["coverage_guarantee"]["guaranteed_coverage"]

    @pytest.mark.asyncio
    async def test_different_seeds_can_differ(self, client, simple_conformal_request):
        """Test that different seeds can give different results."""
        request1 = simple_conformal_request.copy()
        request1["seed"] = 42

        request2 = simple_conformal_request.copy()
        request2["seed"] = 99

        response1 = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request1,
        )

        response2 = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request2,
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Both should complete successfully (results might differ)
        data1 = response1.json()
        data2 = response2.json()

        assert "prediction_interval" in data1
        assert "prediction_interval" in data2


class TestConformalMetadata:
    """Tests for response metadata."""

    @pytest.mark.asyncio
    async def test_metadata_present(self, client, simple_conformal_request):
        """Test that metadata is present in response."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        assert "_metadata" in data
        assert "request_id" in data["_metadata"]
        assert "timestamp" in data["_metadata"]

    @pytest.mark.asyncio
    async def test_custom_request_id(self, client, simple_conformal_request):
        """Test that custom request ID is preserved."""
        custom_request_id = "test-conformal-123"

        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
            headers={"X-Request-Id": custom_request_id},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["_metadata"]["request_id"] == custom_request_id


class TestConformalExplanation:
    """Tests for explanation quality."""

    @pytest.mark.asyncio
    async def test_explanation_present(self, client, simple_conformal_request):
        """Test that explanation is present."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        explanation = data["explanation"]
        assert "summary" in explanation
        assert "reasoning" in explanation
        assert "technical_basis" in explanation
        assert "assumptions" in explanation

    @pytest.mark.asyncio
    async def test_explanation_mentions_coverage(self, client, simple_conformal_request):
        """Test that explanation mentions coverage guarantee."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        summary = data["explanation"]["summary"]
        # Should mention coverage percentage
        assert any(str(x) in summary for x in ["94", "95", "coverage", "guaranteed"])

    @pytest.mark.asyncio
    async def test_explanation_mentions_method(self, client, simple_conformal_request):
        """Test that explanation mentions conformal method."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        reasoning = data["explanation"]["reasoning"]
        # Should mention the method
        assert "split" in reasoning.lower() or "conformal" in reasoning.lower()


class TestConformalConfidenceLevels:
    """Tests for different confidence levels."""

    @pytest.mark.asyncio
    async def test_90_percent_confidence(self, client, simple_conformal_request):
        """Test 90% confidence level."""
        request = simple_conformal_request.copy()
        request["confidence_level"] = 0.90

        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["coverage_guarantee"]["nominal_coverage"] == 0.90

    @pytest.mark.asyncio
    async def test_99_percent_confidence(self, client, simple_conformal_request):
        """Test 99% confidence level."""
        request = simple_conformal_request.copy()
        request["confidence_level"] = 0.99

        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["coverage_guarantee"]["nominal_coverage"] == 0.99

    @pytest.mark.asyncio
    async def test_higher_confidence_wider_intervals(self, client, simple_conformal_request):
        """Test that higher confidence gives wider intervals."""
        request_90 = simple_conformal_request.copy()
        request_90["confidence_level"] = 0.90
        request_90["seed"] = 42

        request_99 = simple_conformal_request.copy()
        request_99["confidence_level"] = 0.99
        request_99["seed"] = 42

        response_90 = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request_90,
        )

        response_99 = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=request_99,
        )

        assert response_90.status_code == 200
        assert response_99.status_code == 200

        data_90 = response_90.json()
        data_99 = response_99.json()

        width_90 = data_90["prediction_interval"]["interval_width"]["Revenue"]
        width_99 = data_99["prediction_interval"]["interval_width"]["Revenue"]

        # 99% interval should be wider than or equal to 90% interval
        assert width_99 >= width_90


class TestConformalResponseStructure:
    """Tests for response structure compliance."""

    @pytest.mark.asyncio
    async def test_interval_width_matches_bounds(self, client, simple_conformal_request):
        """Test that interval width equals upper - lower."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        interval = data["prediction_interval"]
        for var in interval["lower_bound"]:
            lower = interval["lower_bound"][var]
            upper = interval["upper_bound"][var]
            width = interval["interval_width"][var]

            # Width should equal upper - lower
            assert abs(width - (upper - lower)) < 1e-6

    @pytest.mark.asyncio
    async def test_guaranteed_coverage_below_nominal(self, client, simple_conformal_request):
        """Test that guaranteed coverage is below or equal to nominal."""
        response = await client.post(
            "/api/v1/causal/counterfactual/conformal",
            json=simple_conformal_request,
        )

        assert response.status_code == 200
        data = response.json()

        coverage = data["coverage_guarantee"]
        # Guaranteed should be slightly below nominal (finite-sample correction)
        assert coverage["guaranteed_coverage"] <= coverage["nominal_coverage"] + 1e-6
