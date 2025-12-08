"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def mock_dag():
    """Sample DAG structure for testing."""
    return {
        "nodes": ["X", "Y", "Z"],
        "edges": [["X", "Y"], ["Z", "X"], ["Z", "Y"]],
    }


@pytest.fixture
def mock_model():
    """Sample structural causal model for testing."""
    return {
        "equations": {
            "Revenue": {
                "formula": "2.5 * Price + noise",
                "noise_dist": "normal(0, 50)",
            }
        },
        "variables": ["Price", "Revenue"],
    }


@pytest.fixture
def mock_calibration_data():
    """Sample calibration data for testing."""
    return [
        {"Price": 40, "Revenue": 1200},
        {"Price": 42, "Revenue": 1250},
        {"Price": 38, "Revenue": 1150},
        {"Price": 45, "Revenue": 1350},
        {"Price": 41, "Revenue": 1210},
        {"Price": 43, "Revenue": 1280},
        {"Price": 39, "Revenue": 1170},
        {"Price": 44, "Revenue": 1320},
        {"Price": 40, "Revenue": 1190},
        {"Price": 42, "Revenue": 1260},
    ]
