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
