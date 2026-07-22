"""
Shared fixtures for integration tests.

Provides:
- Mock workstream clients
- Standard test models
- Historical calibration data
- Common DAG structures
"""

import pytest
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
