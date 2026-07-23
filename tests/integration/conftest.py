"""
Shared fixtures for integration tests.

Provides:
- Mock workstream clients
- Standard test models
- Historical calibration data
- Common DAG structures
"""

import pytest
import pytest_asyncio
from typing import Dict, List, Any
from datetime import datetime
import json

from fastapi.testclient import TestClient

from src.api.main import app


# ==================== STANDARD TEST MODELS ====================

@pytest.fixture
def pricing_dag():
    """Standard pricing model DAG for testing."""
    return {
        "nodes": ["Price", "Quality", "Marketing", "Revenue", "Cost"],
        "edges": [
            {"from": "Price", "to": "Revenue"},
            {"from": "Quality", "to": "Revenue"},
            {"from": "Marketing", "to": "Revenue"},
            {"from": "Price", "to": "Cost"},
            {"from": "Quality", "to": "Cost"}
        ]
    }


@pytest.fixture
def feature_prioritization_dag():
    """Feature prioritization model for testing."""
    return {
        "nodes": ["FeatureA", "FeatureB", "UserSatisfaction", "Retention", "Revenue"],
        "edges": [
            {"from": "FeatureA", "to": "UserSatisfaction"},
            {"from": "FeatureB", "to": "UserSatisfaction"},
            {"from": "UserSatisfaction", "to": "Retention"},
            {"from": "Retention", "to": "Revenue"}
        ]
    }


@pytest.fixture
def confounded_dag():
    """Non-identifiable DAG with unobserved confounder."""
    return {
        "nodes": ["Treatment", "Outcome", "Confounder"],
        "edges": [
            {"from": "Treatment", "to": "Outcome"},
            {"from": "Confounder", "to": "Treatment"},
            {"from": "Confounder", "to": "Outcome"}
        ],
        "latent": ["Confounder"]
    }


@pytest.fixture
def identifiable_backdoor_dag():
    """Identifiable DAG with backdoor path."""
    return {
        "nodes": ["Treatment", "Outcome", "Confounder"],
        "edges": [
            {"from": "Treatment", "to": "Outcome"},
            {"from": "Confounder", "to": "Treatment"},
            {"from": "Confounder", "to": "Outcome"}
        ]
    }


# ==================== CALIBRATION DATA ====================

@pytest.fixture
def calibration_data():
    """Historical calibration data for conformal prediction."""
    return {
        "features": [
            {"Price": 50, "Quality": 0.8, "Marketing": 1000},
            {"Price": 60, "Quality": 0.7, "Marketing": 1200},
            {"Price": 45, "Quality": 0.9, "Marketing": 900},
            {"Price": 55, "Quality": 0.75, "Marketing": 1100},
            {"Price": 50, "Quality": 0.85, "Marketing": 950},
        ],
        "outcomes": [5200, 5100, 5400, 5150, 5300]
    }


@pytest.fixture
def large_calibration_data():
    """Larger calibration dataset for robust testing."""
    import numpy as np
    np.random.seed(42)

    n_samples = 100
    features = []
    outcomes = []

    for _ in range(n_samples):
        price = np.random.uniform(40, 70)
        quality = np.random.uniform(0.5, 1.0)
        marketing = np.random.uniform(800, 1500)

        # Simulated outcome with noise
        outcome = (100 * price + 5000 * quality + 2 * marketing +
                  np.random.normal(0, 100))

        features.append({
            "Price": price,
            "Quality": quality,
            "Marketing": marketing
        })
        outcomes.append(outcome)

    return {"features": features, "outcomes": outcomes}


# ==================== SCENARIO DATA ====================

@pytest.fixture
def batch_scenarios():
    """Batch of scenarios for comparison testing."""
    return [
        {
            "name": "Baseline",
            "interventions": {"Price": 50.0}
        },
        {
            "name": "Price Increase",
            "interventions": {"Price": 60.0}
        },
        {
            "name": "Quality Improvement",
            "interventions": {"Quality": 0.9}
        },
        {
            "name": "Combined Strategy",
            "interventions": {"Price": 55.0, "Quality": 0.85}
        }
    ]


@pytest.fixture
def team_scenarios():
    """Team proposal scenarios for TAE testing."""
    return [
        {
            "team": "Product",
            "proposal": "Increase feature velocity",
            "interventions": {"FeatureA": 1.0, "FeatureB": 1.0}
        },
        {
            "team": "Engineering",
            "proposal": "Focus on quality",
            "interventions": {"FeatureA": 0.8, "FeatureB": 0.5}
        },
        {
            "team": "Marketing",
            "proposal": "User satisfaction first",
            "interventions": {"FeatureA": 0.6, "FeatureB": 0.9}
        }
    ]


# ==================== MOCK WORKSTREAM DATA ====================

@pytest.fixture
def mock_plot_context():
    """Mock PLoT context for testing."""
    return {
        "user_id": "user_123",
        "session_id": "session_456",
        "workflow_type": "standard_analysis",
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_tae_context():
    """Mock TAE context for testing."""
    return {
        "deliberation_id": "delib_789",
        "round": 3,
        "teams": ["Product", "Engineering", "Marketing"],
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_cee_context():
    """Mock CEE context for testing."""
    return {
        "critique_id": "critique_012",
        "document_type": "technical_proposal",
        "author": "engineer_456",
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== TRANSPORT DATA ====================

@pytest.fixture
def source_market_data():
    """Source market data for transportability testing."""
    return {
        "market": "US",
        "features": ["Price", "Quality", "Marketing"],
        "distributions": {
            "Price": {"mean": 50, "std": 10},
            "Quality": {"mean": 0.8, "std": 0.1},
            "Marketing": {"mean": 1000, "std": 200}
        }
    }


@pytest.fixture
def target_market_data():
    """Target market data for transportability testing."""
    return {
        "market": "EU",
        "features": ["Price", "Quality", "Marketing"],
        "distributions": {
            "Price": {"mean": 60, "std": 12},
            "Quality": {"mean": 0.85, "std": 0.08},
            "Marketing": {"mean": 1200, "std": 250}
        }
    }


# ==================== HELPER FUNCTIONS ====================

@pytest.fixture
def performance_threshold():
    """Performance threshold for workflow tests."""
    return 5.0  # seconds


@pytest.fixture
def make_request_id():
    """Factory for generating unique request IDs."""
    counter = 0

    def _make_id(prefix: str = "req") -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}_{counter}_{datetime.utcnow().timestamp()}"

    return _make_id


# ==================== S2 ROBUSTNESS-V2 ENDPOINT FIXTURES ====================
# Shared by the two S2 factor-sensitivity suites — test_confidence_provenance.py
# (disclosure marker) and test_provenance_echo.py (value-origin echo) — which both
# drive /api/v1/robustness/analyze/v2 with the same 2-factor graph where
# 'marketing' is the uncertain factor that drives factor_sensitivity. Named
# distinctly (v2_client, not `client`) so they do NOT override the async httpx
# `client` fixture in tests/conftest.py that the other integration suites use.


@pytest.fixture
def v2_client():
    """Sync FastAPI TestClient for the robustness v2 endpoint (S2 suites)."""
    return TestClient(app)


@pytest.fixture
def two_factor_request():
    """Return a builder for the shared 2-factor robustness request.

    'price' is the decision variable (intervened on by both options); 'marketing'
    is a free uncertain factor that drives factor_sensitivity. Pass
    ``marketing_observed_state`` to control provenance metadata, or None to omit
    observed_state entirely (the "value defaulted to 0.0" case). Set
    ``include_marketing_uncertainty=False`` to drop the marketing PU. With the
    defaults it reproduces the disclosure-marker suite's request exactly.
    """

    def _build(marketing_observed_state=None, include_marketing_uncertainty=True):
        marketing_node = {"id": "marketing", "kind": "factor", "label": "Marketing"}
        if marketing_observed_state is not None:
            marketing_node["observed_state"] = marketing_observed_state

        request = {
            "graph": {
                "nodes": [
                    {"id": "price", "kind": "factor", "label": "Price"},
                    marketing_node,
                    {"id": "revenue", "kind": "goal", "label": "Revenue"},
                ],
                "edges": [
                    {"from": "price", "to": "revenue", "strength": {"mean": 0.6, "std": 0.15}},
                    {"from": "marketing", "to": "revenue", "strength": {"mean": 0.5, "std": 0.15}},
                ],
            },
            "options": [
                {"id": "opt1", "label": "Raise price", "interventions": {"price": 120}},
                {"id": "opt2", "label": "Lower price", "interventions": {"price": 80}},
            ],
            "goal_node_id": "revenue",
            "seed": 42,
            "n_samples": 200,
        }
        if include_marketing_uncertainty:
            request["parameter_uncertainties"] = [
                {"node_id": "marketing", "distribution": "normal", "std": 5.0}
            ]
        return request

    return _build


# ==================== A3 COUNTERFACTUAL ROUTE CLIENT ====================
# Shared by the counterfactual-honesty (F11 log redaction) suite and the
# intervention/context input-validation suite. Mounts the LIVE counterfactual
# route AND the (prod-dark) conformal route on the full causal router, both with
# the production exception handlers, so the ValueError->422 mapping (D-12(cf)) and
# the conformal 400 envelope are exercised at the router level. Named distinctly
# (cf_client, not `client`) so it does not shadow the async httpx `client` fixture
# in tests/conftest.py.


@pytest_asyncio.fixture
async def cf_client():
    from fastapi import FastAPI, HTTPException
    from httpx import ASGITransport, AsyncClient

    from src.api import main as isl_main
    from src.api.causal import counterfactual_router, router as causal_full_router

    test_app = FastAPI()
    test_app.include_router(counterfactual_router, prefix="/api/v1/causal")
    # conformal lives on the full causal router (dark in prod); mount it here so
    # the F11 site-3 redaction is exercised through the real handler.
    test_app.include_router(causal_full_router, prefix="/api/v1/causal")
    test_app.add_exception_handler(HTTPException, isl_main.http_exception_handler)
    test_app.add_exception_handler(Exception, isl_main.global_exception_handler)

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        yield ac


# ==================== SEQUENTIAL-DECISION PAYLOAD ====================
# One canonical sequential decision-tree request, consumed by the phase4 endpoint
# suite (test_phase4_endpoints) and the config_details honesty suite. Function
# scope returns a fresh dict per test, so callers may mutate it in place.


@pytest.fixture
def sequential_analysis_request():
    """Sample sequential analysis request."""
    return {
        "graph": {
            "nodes": [
                {"id": "invest", "type": "decision", "label": "Investment Decision"},
                {"id": "market", "type": "chance", "label": "Market Outcome"},
                {"id": "success", "type": "terminal", "label": "Success", "payoff": 100000},
                {"id": "failure", "type": "terminal", "label": "Failure", "payoff": -20000},
                {"id": "no_invest", "type": "terminal", "label": "No Investment", "payoff": 0},
            ],
            "edges": [
                {"from": "invest", "to": "market", "action": "invest", "immediate_payoff": -10000},
                {"from": "invest", "to": "no_invest", "action": "wait"},
                {"from": "market", "to": "success", "outcome": "favorable", "probability": 0.6},
                {"from": "market", "to": "failure", "outcome": "unfavorable", "probability": 0.4},
            ],
            "stage_assignments": {
                "invest": 0,
                "market": 1,
                "success": 2,
                "failure": 2,
                "no_invest": 1,
            },
        },
        "stages": [
            {"stage_index": 0, "stage_label": "Investment", "decision_nodes": ["invest"]},
            {
                "stage_index": 1,
                "stage_label": "Market",
                "decision_nodes": [],
                "resolution_nodes": ["market"],
            },
            {"stage_index": 2, "stage_label": "Terminal", "decision_nodes": []},
        ],
        "discount_factor": 0.95,
        "risk_tolerance": "neutral",
    }


# ==================== A3 F11 REDACTION POSITIVE-CONTROL ====================
# One shared redaction sentinel + the trap-#13 positive-control helper, consumed
# by the counterfactual log-redaction suite (test_a3_cf_honesty) and the
# intervention/context value-validation suite (test_a3_intervention_key_validation).
# A value that cannot arise from a hash / seed / percentile, so its appearance in a
# log or error message is unambiguous leakage of a client-private input.
REDACTION_SENTINEL = 987654.321


@pytest.fixture
def redaction_sentinel() -> float:
    """The shared F11 redaction sentinel value."""
    return REDACTION_SENTINEL


@pytest.fixture
def assert_harness_can_see_value():
    """Trap #13 positive control: prove the capture harness is NOT blind to the
    sentinel by emitting it through the same logger and confirming it is seen, so a
    following absence assertion is not vacuous. Returns a callable
    ``(caplog_records, logger_name)``."""
    import logging

    def _assert(caplog_records, logger_name):
        log = logging.getLogger(logger_name)
        log.info("a3_f11_probe_control", extra={"intervention": {"Probe": REDACTION_SENTINEL}})
        seen = [r for r in caplog_records if getattr(r, "msg", None) == "a3_f11_probe_control"]
        assert seen, "positive control failed: caplog did not capture the probe log at all"
        assert any(str(REDACTION_SENTINEL) in repr(r.__dict__) for r in seen), (
            "positive control failed: the harness cannot SEE the raw value even when "
            "it is deliberately logged — an absence assertion here would be vacuous"
        )

    return _assert
