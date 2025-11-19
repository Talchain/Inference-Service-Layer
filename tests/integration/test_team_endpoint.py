"""
Integration tests for team alignment endpoint.
"""


def test_team_alignment_basic(client, team_perspectives, decision_options):
    """Test basic team alignment."""
    response = client.post(
        "/api/v1/team/align",
        json={"perspectives": team_perspectives, "options": decision_options},
    )

    assert response.status_code == 200
    data = response.json()

    assert "common_ground" in data
    assert "aligned_options" in data
    assert "conflicts" in data
    assert "recommendation" in data
    assert "explanation" in data


def test_team_alignment_common_ground(client, team_perspectives, decision_options):
    """Test that common ground is identified."""
    response = client.post(
        "/api/v1/team/align",
        json={"perspectives": team_perspectives, "options": decision_options},
    )

    data = response.json()
    common_ground = data["common_ground"]

    assert "shared_goals" in common_ground
    assert "shared_constraints" in common_ground
    assert "agreement_level" in common_ground
    assert isinstance(common_ground["agreement_level"], (int, float))
    assert 0 <= common_ground["agreement_level"] <= 100


def test_team_alignment_ranked_options(client, team_perspectives, decision_options):
    """Test that options are ranked by satisfaction."""
    response = client.post(
        "/api/v1/team/align",
        json={"perspectives": team_perspectives, "options": decision_options},
    )

    data = response.json()
    aligned_options = data["aligned_options"]

    assert len(aligned_options) == len(decision_options)

    # Options should be sorted by satisfaction score (descending)
    for i in range(len(aligned_options) - 1):
        assert (
            aligned_options[i]["satisfaction_score"]
            >= aligned_options[i + 1]["satisfaction_score"]
        )


def test_team_alignment_recommendation(client, team_perspectives, decision_options):
    """Test that recommendation is provided."""
    response = client.post(
        "/api/v1/team/align",
        json={"perspectives": team_perspectives, "options": decision_options},
    )

    data = response.json()
    recommendation = data["recommendation"]

    assert "top_option" in recommendation
    assert "rationale" in recommendation
    assert "confidence" in recommendation
    assert "next_steps" in recommendation
    assert len(recommendation["next_steps"]) > 0
