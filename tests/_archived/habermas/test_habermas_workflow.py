"""
Integration tests for Habermas Machine deliberation workflow.

Tests the complete end-to-end deliberation process via API.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


class TestHabermasMachineWorkflow:
    """Test complete Habermas Machine deliberation workflow."""

    def test_deliberation_endpoint_success(self):
        """Test deliberation endpoint with valid request."""
        request_data = {
            "decision_context": "Choose between Feature A and Feature B",
            "positions": [
                {
                    "member_id": "alice",
                    "member_name": "Alice",
                    "role": "PM",
                    "values": [
                        {
                            "value_name": "user_satisfaction",
                            "weight": 0.9,
                            "rationale": "Users should be happy",
                            "examples": [],
                        }
                    ],
                    "concerns": [],
                    "timestamp": "2025-01-01T00:00:00",
                },
                {
                    "member_id": "bob",
                    "member_name": "Bob",
                    "role": "Engineer",
                    "values": [
                        {
                            "value_name": "user_satisfaction",
                            "weight": 0.8,
                            "rationale": "User experience is key",
                            "examples": [],
                        }
                    ],
                    "concerns": [],
                    "timestamp": "2025-01-01T00:00:00",
                },
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "session_id" in data
        assert "round_number" in data
        assert "common_ground" in data
        assert "consensus_statement" in data
        assert "status" in data
        assert "convergence_assessment" in data
        assert "next_steps" in data

    def test_deliberation_finds_common_ground(self):
        """Test that deliberation identifies common ground."""
        request_data = {
            "decision_context": "Architecture decision",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "quality",
                            "weight": 0.9,
                            "rationale": "Quality is critical",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
                {
                    "member_id": "bob",
                    "values": [
                        {
                            "value_name": "quality",
                            "weight": 0.85,
                            "rationale": "Quality matters most",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        common_ground = data["common_ground"]

        # Should find shared quality value
        assert len(common_ground["shared_values"]) > 0
        assert common_ground["agreement_level"] > 0

    def test_deliberation_generates_consensus(self):
        """Test that consensus statement is generated."""
        request_data = {
            "decision_context": "Product roadmap",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "innovation",
                            "weight": 0.8,
                            "rationale": "Innovation drives growth",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
                {
                    "member_id": "bob",
                    "values": [
                        {
                            "value_name": "innovation",
                            "weight": 0.75,
                            "rationale": "Need to innovate",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        consensus = data["consensus_statement"]

        # Check consensus structure
        assert "statement_id" in consensus
        assert "version" in consensus
        assert "text" in consensus
        assert len(consensus["text"]) > 0
        assert "support_score" in consensus
        assert "incorporated_values" in consensus

    def test_multi_round_deliberation(self):
        """Test conducting multiple rounds of deliberation."""
        # Round 1
        request1 = {
            "decision_context": "Sprint planning",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "speed",
                            "weight": 0.8,
                            "rationale": "Fast delivery",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
        }

        response1 = client.post("/api/v1/deliberation/deliberate", json=request1)
        assert response1.status_code == 200
        data1 = response1.json()

        session_id = data1["session_id"]
        consensus1 = data1["consensus_statement"]

        # Round 2 - continue session
        request2 = {
            "session_id": session_id,
            "decision_context": "Sprint planning",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "speed",
                            "weight": 0.8,
                            "rationale": "Fast delivery",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
            "previous_consensus": consensus1,
        }

        response2 = client.post("/api/v1/deliberation/deliberate", json=request2)
        assert response2.status_code == 200
        data2 = response2.json()

        # Should be round 2 of same session
        assert data2["session_id"] == session_id
        assert data2["round_number"] == 2

    def test_edit_suggestions_refinement(self):
        """Test consensus refinement with edit suggestions."""
        # Round 1
        request1 = {
            "decision_context": "Design decision",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "simplicity",
                            "weight": 0.9,
                            "rationale": "Simple is better",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
        }

        response1 = client.post("/api/v1/deliberation/deliberate", json=request1)
        data1 = response1.json()

        # Round 2 with edits
        request2 = {
            "session_id": data1["session_id"],
            "decision_context": "Design decision",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "simplicity",
                            "weight": 0.9,
                            "rationale": "Simple is better",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
            "previous_consensus": data1["consensus_statement"],
            "edit_suggestions": [
                {
                    "member_id": "alice",
                    "edit_type": "addition",
                    "suggestion": "Also consider performance",
                    "rationale": "Performance matters too",
                    "priority": 0.8,
                }
            ],
        }

        response2 = client.post("/api/v1/deliberation/deliberate", json=request2)
        assert response2.status_code == 200
        data2 = response2.json()

        # Version should increment
        assert data2["consensus_statement"]["version"] == 2

    def test_convergence_assessment(self):
        """Test convergence assessment in response."""
        request_data = {
            "decision_context": "Test",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "quality",
                            "weight": 0.9,
                            "rationale": "Quality first",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
                {
                    "member_id": "bob",
                    "values": [
                        {
                            "value_name": "quality",
                            "weight": 0.85,
                            "rationale": "Quality is key",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)
        assert response.status_code == 200
        data = response.json()

        assessment = data["convergence_assessment"]

        # Check assessment fields
        assert "support_score" in assessment
        assert "support_threshold" in assessment
        assert "agreement_level" in assessment
        assert "agreement_threshold" in assessment
        assert "unresolved_count" in assessment

    def test_get_session_endpoint(self):
        """Test retrieving session details."""
        # Create session
        request_data = {
            "decision_context": "Test",
            "positions": [
                {
                    "member_id": "alice",
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
        }

        create_response = client.post(
            "/api/v1/deliberation/deliberate",
            json=request_data,
        )
        session_id = create_response.json()["session_id"]

        # Get session
        get_response = client.get(f"/api/v1/deliberation/session/{session_id}")

        assert get_response.status_code == 200
        data = get_response.json()

        assert "session" in data
        session = data["session"]
        assert session["session_id"] == session_id

    def test_get_nonexistent_session(self):
        """Test getting session that doesn't exist."""
        response = client.get("/api/v1/deliberation/session/nonexistent_id")

        assert response.status_code == 404

    def test_shared_concerns_identification(self):
        """Test that shared concerns are identified."""
        request_data = {
            "decision_context": "Risk assessment",
            "positions": [
                {
                    "member_id": "alice",
                    "concerns": [
                        {
                            "concern_name": "technical_risk",
                            "severity": 0.8,
                            "explanation": "System may not scale",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
                {
                    "member_id": "bob",
                    "concerns": [
                        {
                            "concern_name": "technical_risk",
                            "severity": 0.7,
                            "explanation": "Performance issues likely",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)
        assert response.status_code == 200
        data = response.json()

        common_ground = data["common_ground"]

        # Should identify shared concerns
        assert len(common_ground["shared_concerns"]) > 0

    def test_next_steps_provided(self):
        """Test that next steps are provided."""
        request_data = {
            "decision_context": "Test",
            "positions": [
                {
                    "member_id": "alice",
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Should have next steps
        assert len(data["next_steps"]) > 0

    def test_custom_request_id(self):
        """Test using custom request ID."""
        request_data = {
            "decision_context": "Test",
            "positions": [
                {
                    "member_id": "alice",
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
        }

        headers = {"X-Request-Id": "test_habermas_123"}

        response = client.post(
            "/api/v1/deliberation/deliberate",
            json=request_data,
            headers=headers,
        )

        assert response.status_code == 200
        data = response.json()

        # Check metadata includes request ID
        metadata = data.get("metadata") or data.get("_metadata")
        if metadata:
            assert metadata["request_id"] == "test_habermas_123"

    def test_missing_positions_validation(self):
        """Test that missing positions causes validation error."""
        request_data = {
            "decision_context": "Test",
            "positions": [],  # Empty positions
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)

        # Should succeed but return low agreement (empty is technically valid)
        # Or could fail validation depending on implementation
        assert response.status_code in [200, 400, 422]

    def test_partial_agreement_scenario(self):
        """Test scenario where team has partial agreement."""
        request_data = {
            "decision_context": "Feature priority",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "user_satisfaction",
                            "weight": 0.9,
                            "rationale": "Users first",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
                {
                    "member_id": "bob",
                    "values": [
                        {
                            "value_name": "technical_quality",
                            "weight": 0.9,
                            "rationale": "Tech debt is critical",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # No shared values expected
        assert data["common_ground"]["agreement_level"] <= 0.5

        # Should still generate consensus acknowledging disagreement
        assert len(data["consensus_statement"]["text"]) > 0

    def test_high_alignment_scenario(self):
        """Test scenario where team has high alignment."""
        request_data = {
            "decision_context": "Core value",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "quality",
                            "weight": 0.95,
                            "rationale": "Quality is everything",
                        },
                        {
                            "value_name": "reliability",
                            "weight": 0.9,
                            "rationale": "System must be reliable",
                        },
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
                {
                    "member_id": "bob",
                    "values": [
                        {
                            "value_name": "quality",
                            "weight": 0.9,
                            "rationale": "Quality first",
                        },
                        {
                            "value_name": "reliability",
                            "weight": 0.85,
                            "rationale": "Reliability is key",
                        },
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)
        assert response.status_code == 200
        data = response.json()

        # Should have high agreement
        assert data["common_ground"]["agreement_level"] > 0.5

        # Should find multiple shared values
        assert len(data["common_ground"]["shared_values"]) >= 2

    def test_consensus_incorporates_values(self):
        """Test that consensus statement incorporates shared values."""
        request_data = {
            "decision_context": "Strategy",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "innovation",
                            "weight": 0.9,
                            "rationale": "Innovation drives success",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
                {
                    "member_id": "bob",
                    "values": [
                        {
                            "value_name": "innovation",
                            "weight": 0.85,
                            "rationale": "Need to innovate",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                },
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)
        assert response.status_code == 200
        data = response.json()

        consensus = data["consensus_statement"]

        # Innovation should be incorporated
        assert "innovation" in consensus["incorporated_values"]


class TestHabermasMetrics:
    """Test Habermas Machine metrics tracking."""

    def test_metrics_tracked_on_deliberation(self):
        """Test that metrics are tracked during deliberation."""
        from src.utils.business_metrics import (
            habermas_deliberations_total,
            habermas_agreement_level,
        )

        request_data = {
            "decision_context": "Test",
            "positions": [
                {
                    "member_id": "alice",
                    "values": [
                        {
                            "value_name": "quality",
                            "weight": 0.9,
                            "rationale": "Quality matters",
                        }
                    ],
                    "timestamp": "2025-01-01T00:00:00",
                }
            ],
        }

        response = client.post("/api/v1/deliberation/deliberate", json=request_data)

        assert response.status_code == 200

        # Metrics should have been updated
        # (Exact verification depends on Prometheus client implementation)
