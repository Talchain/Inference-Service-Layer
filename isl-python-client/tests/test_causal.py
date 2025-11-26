"""Tests for causal inference API."""

import pytest
from unittest.mock import MagicMock, patch

from isl_client import ISLClient
from isl_client.models import ValidationResponse, ConformalResponse


@pytest.mark.asyncio
async def test_validate_identifiable():
    """Test successful validation (identifiable)."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "identifiable",
            "method": "backdoor",
            "adjustment_sets": [["Z"]],
            "suggestions": None,
            "explanation": {
                "summary": "Effect is identifiable via backdoor adjustment",
                "assumptions": ["No unmeasured confounding"],
                "caveats": [],
                "confidence": 0.95,
            },
        }

        with patch.object(client, "post", return_value=mock_response):
            result = await client.causal.validate(
                dag={"nodes": ["X", "Y", "Z"], "edges": [["X", "Y"], ["Z", "X"], ["Z", "Y"]]},
                treatment="X",
                outcome="Y",
            )

            assert isinstance(result, ValidationResponse)
            assert result.status == "identifiable"
            assert result.method == "backdoor"
            assert result.adjustment_sets == [["Z"]]


@pytest.mark.asyncio
async def test_validate_not_identifiable():
    """Test validation returns suggestions when not identifiable."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "cannot_identify",
            "method": None,
            "adjustment_sets": None,
            "suggestions": [
                {
                    "type": "backdoor",
                    "description": "Collect data on confounder U",
                    "technical_detail": "U confounds X and Y",
                    "priority": "critical",
                    "action": {
                        "action_type": "collect_data",
                        "details": {"variables": ["U"]},
                    },
                }
            ],
            "explanation": {
                "summary": "Effect not identifiable due to unmeasured confounding",
                "assumptions": [],
                "caveats": ["Requires additional data"],
                "confidence": 0.3,
            },
        }

        with patch.object(client, "post", return_value=mock_response):
            result = await client.causal.validate(
                dag={"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
                treatment="X",
                outcome="Y",
            )

            assert result.status == "cannot_identify"
            assert result.suggestions is not None
            assert len(result.suggestions) == 1
            assert result.suggestions[0].priority == "critical"


@pytest.mark.asyncio
async def test_counterfactual():
    """Test counterfactual prediction."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "intervention": {"Price": 45.0},
            "prediction": {
                "prediction": {"Revenue": 1250.0},
                "uncertainty": {"Revenue": 50.0},
                "explanation": "Increasing price to $45 increases revenue to $1250",
            },
            "model_assumptions": ["Linear relationships", "No spillover effects"],
            "explanation": {
                "summary": "Counterfactual prediction complete",
                "assumptions": ["Structural equations correct"],
                "caveats": [],
                "confidence": 0.85,
            },
        }

        with patch.object(client, "post", return_value=mock_response):
            result = await client.causal.counterfactual(
                model={"equations": {}},
                intervention={"Price": 45.0},
                seed=42,
            )

            assert result.intervention == {"Price": 45.0}
            assert result.prediction.prediction["Revenue"] == 1250.0


@pytest.mark.asyncio
async def test_conformal_prediction():
    """Test conformal prediction with intervals."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "conformal_interval": {
                "lower": 1150.0,
                "upper": 1350.0,
                "width": 200.0,
                "point_estimate": 1250.0,
            },
            "coverage_guarantee": {
                "guaranteed": True,
                "theoretical_coverage": 0.95,
                "finite_sample_valid": True,
                "assumptions": ["Exchangeability"],
            },
            "comparison_to_monte_carlo": {
                "monte_carlo_width": 250.0,
                "conformal_width": 200.0,
                "width_ratio": 0.8,
                "relative_efficiency": 1.25,
            },
            "explanation": {
                "summary": "95% conformal interval computed",
                "assumptions": ["Exchangeability"],
                "caveats": [],
                "confidence": 0.95,
            },
        }

        with patch.object(client, "post", return_value=mock_response):
            result = await client.causal.counterfactual_conformal(
                model={"equations": {}},
                intervention={"Price": 45.0},
                calibration_data=[{"Price": 40, "Revenue": 1200}],
                confidence=0.95,
            )

            assert isinstance(result, ConformalResponse)
            assert result.conformal_interval.lower == 1150.0
            assert result.conformal_interval.upper == 1350.0
            assert result.coverage_guarantee.guaranteed is True


@pytest.mark.asyncio
async def test_batch_counterfactuals():
    """Test batch scenario analysis."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "scenarios": [
                {
                    "scenario_id": "base",
                    "intervention": {"Price": 40},
                    "prediction": {"Revenue": 1200},
                    "uncertainty": None,
                    "explanation": "Base scenario",
                },
                {
                    "scenario_id": "high",
                    "intervention": {"Price": 50},
                    "prediction": {"Revenue": 1300},
                    "uncertainty": None,
                    "explanation": "High price scenario",
                },
            ],
            "interactions": {
                "has_synergy": False,
                "synergy_score": None,
                "summary": "No significant interactions detected",
                "details": {},
            },
            "optimal_scenario": "high",
            "explanation": {
                "summary": "Batch analysis complete",
                "assumptions": [],
                "caveats": [],
                "confidence": 0.85,
            },
        }

        with patch.object(client, "post", return_value=mock_response):
            result = await client.causal.batch_counterfactuals(
                model={"equations": {}},
                scenarios=[
                    {"id": "base", "intervention": {"Price": 40}},
                    {"id": "high", "intervention": {"Price": 50}},
                ],
            )

            assert len(result.scenarios) == 2
            assert result.optimal_scenario == "high"
            assert result.interactions.has_synergy is False


@pytest.mark.asyncio
async def test_transport():
    """Test transportability analysis."""
    async with ISLClient("http://localhost:8000") as client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "transportable": True,
            "validity_conditions": ["No selection bias", "Same causal mechanisms"],
            "adaptation_required": False,
            "suggestions": [],
            "explanation": {
                "summary": "Effect transports to target domain",
                "assumptions": ["Causal mechanisms invariant"],
                "caveats": [],
                "confidence": 0.9,
            },
        }

        with patch.object(client, "post", return_value=mock_response):
            result = await client.causal.transport(
                source_domain={"dag": {}, "population": "A"},
                target_domain={"dag": {}, "population": "B"},
                treatment="X",
                outcome="Y",
            )

            assert result.transportable is True
            assert not result.adaptation_required
