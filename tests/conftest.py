"""
Pytest configuration and fixtures.

Provides reusable test fixtures for all test modules.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """
    FastAPI test client.

    Returns:
        TestClient: FastAPI test client for making requests
    """
    return TestClient(app)


@pytest.fixture
def sample_dag():
    """
    Simple causal DAG for testing.

    Structure: Z → X → Y, Z → Y (Z confounds X-Y relationship)

    Returns:
        dict: DAG structure
    """
    return {
        "nodes": ["X", "Y", "Z"],
        "edges": [["Z", "X"], ["X", "Y"], ["Z", "Y"]],
    }


@pytest.fixture
def pricing_dag():
    """
    Realistic pricing decision DAG.

    Returns:
        dict: Pricing DAG structure
    """
    return {
        "nodes": ["Price", "Brand", "Revenue", "CustomerAcquisition"],
        "edges": [
            ["Price", "Revenue"],
            ["Brand", "Price"],
            ["Brand", "Revenue"],
            ["CustomerAcquisition", "Revenue"],
        ],
    }


@pytest.fixture
def sample_structural_model():
    """
    Simple structural model for testing.

    Returns:
        dict: Structural model
    """
    return {
        "variables": ["X", "Y", "Z"],
        "equations": {"Y": "10 + 2*X + 3*Z", "Z": "5 + 0.5*X"},
        "distributions": {
            "X": {"type": "normal", "parameters": {"mean": 0, "std": 1}}
        },
    }


@pytest.fixture
def pricing_structural_model():
    """
    Realistic pricing structural model.

    Returns:
        dict: Pricing structural model
    """
    return {
        "variables": ["Price", "Brand", "Revenue"],
        "equations": {
            "Brand": "baseline_brand + 0.3 * Price",
            "Revenue": "10000 + 500 * Price - 200 * Brand",
        },
        "distributions": {
            "baseline_brand": {"type": "normal", "parameters": {"mean": 50, "std": 5}}
        },
    }


@pytest.fixture
def team_perspectives():
    """
    Sample team perspectives for testing.

    Returns:
        list: Team perspectives
    """
    return [
        {
            "role": "Product Manager",
            "priorities": ["User acquisition", "Revenue growth", "Fast time-to-market"],
            "constraints": ["Limited budget", "Q4 deadline"],
            "preferred_options": ["option_a", "option_b"],
        },
        {
            "role": "Designer",
            "priorities": ["User experience", "Brand consistency", "Accessibility"],
            "constraints": ["Design system limitations"],
            "preferred_options": ["option_b", "option_c"],
        },
        {
            "role": "Engineer",
            "priorities": ["Code quality", "Maintainability", "Tech debt reduction"],
            "constraints": ["Team capacity", "Technical limitations"],
            "preferred_options": ["option_c"],
        },
    ]


@pytest.fixture
def decision_options():
    """
    Sample decision options for testing.

    Returns:
        list: Decision options
    """
    return [
        {
            "id": "option_a",
            "name": "Quick MVP launch",
            "attributes": {
                "speed": "fast",
                "quality": "medium",
                "acquisition_potential": "high",
            },
        },
        {
            "id": "option_b",
            "name": "Polished feature set",
            "attributes": {
                "speed": "medium",
                "quality": "high",
                "acquisition_potential": "medium",
            },
        },
        {
            "id": "option_c",
            "name": "Architectural refactor first",
            "attributes": {
                "speed": "slow",
                "quality": "high",
                "acquisition_potential": "low",
            },
        },
    ]


@pytest.fixture
def sample_assumptions():
    """
    Sample assumptions for sensitivity analysis.

    Returns:
        list: Assumptions
    """
    return [
        {
            "name": "Customer price sensitivity",
            "current_value": 0.5,
            "type": "parametric",
            "variation_range": {"min": 0.3, "max": 0.8},
        },
        {
            "name": "Competitor response timing",
            "current_value": "90 days",
            "type": "structural",
        },
        {
            "name": "Seasonal demand variation",
            "current_value": {"Q4_multiplier": 1.08},
            "type": "distributional",
        },
    ]
