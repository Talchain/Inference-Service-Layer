"""Discriminating objective checks through request parse, worker wire and both APIs."""

from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.api import robustness as api
from src.models.robustness_v2 import RobustnessRequestV2, RobustnessResponseV2
from src.models.robustness_v2 import ObjectiveRanking
from src.models.response_v2 import ObjectiveRankingV2
from src.services import robustness_analyzer_v2 as science


@pytest.fixture
def client(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    assert Path(science.__file__).resolve() == root / "src/services/robustness_analyzer_v2.py"
    assert Path(api.__file__).resolve() == root / "src/api/robustness.py"

    async def run(_app, request, request_id, _key, _cost):
        result = science.RobustnessAnalyzerV2().analyze(
            request.model_copy(update={"request_id": request_id})
        )
        # Exercise the same dump/validate boundary as the process worker.
        return RobustnessResponseV2.model_validate_json(result.model_dump_json()), None

    monkeypatch.setattr(api, "_admit_and_run", run)
    return TestClient(app)


def payload(**overrides):
    data = {
        "graph": {
            "nodes": [
                {
                    "id": "driver",
                    "kind": "factor",
                    "label": "Driver",
                    "observed_state": {"value": 0.5, "baseline": 0.5},
                },
                {
                    "id": "goal",
                    "kind": "outcome",
                    "label": "Goal",
                    "observed_state": {"value": 0.5, "baseline": 0.5},
                },
            ],
            "edges": [
                {
                    "from": "driver",
                    "to": "goal",
                    "exists_probability": 1,
                    "strength": {"mean": 1, "std": 0.01},
                }
            ],
        },
        "options": [
            {"id": name, "label": name, "interventions": {"driver": value}}
            for name, value in (("low", 0.1), ("middle", 0.5), ("high", 0.9))
        ],
        "goal_node_id": "goal",
        "n_samples": 400,
        "seed": 42,
        "analysis_types": ["comparison"],
        "goal_direction": "maximise",
    }
    data.update(overrides)
    return data


def post(client, data, version=2):
    response = client.post(f"/api/v1/robustness/analyze/v2?response_version={version}", json=data)
    assert response.status_code == 200, response.text
    return response.json()


def options(response, version):
    return {
        row["id" if version == 2 else "option_id"]: row
        for row in response["options" if version == 2 else "results"]
    }


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "direction,frame,leader",
    [
        ("maximise", None, "high"),
        ("minimise", None, "low"),
        ("target", "delta", "middle"),
        ("target", "level", "middle"),
    ],
)
def test_objective_changes_producer_order_and_same_run_share(
    client, version, direction, frame, leader
):
    data = payload(goal_direction=direction)
    if frame:
        data.update(goal_threshold=0.5, goal_threshold_frame=frame)
    result = post(client, data, version)
    ranking = result["objective_ranking"]
    assert ranking["status"] == "computed" and ranking["attested"] is True
    assert ranking["direction"] == direction
    assert ranking["ranked_options"][0] == {"option_id": leader, "rank": 1, "win_probability": 1.0}
    by_id = options(result, version)
    assert all(
        row["win_probability"] == by_id[row["option_id"]]["win_probability"]
        for row in ranking["ranked_options"]
    )
    if direction != "maximise":
        assert "robustness" not in result
        assert all(
            "downside" not in row and "pre_noise_expected_regret" not in row
            for row in by_id.values()
        )


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("missing", [True, False])
def test_withheld_never_leaks_recommendation_or_zero_share(client, version, missing):
    data = payload(
        include_path_decomposition=True,
        include_e_values=True,
        include_factor_flips=True,
        include_voi=True,
    )
    if missing:
        del data["goal_direction"]
    else:
        data.update(goal_direction="target", goal_threshold=0.6, goal_threshold_frame="level")
        # A goal pinned by any option cannot be converted to the requested frame.
        data["options"][0]["interventions"] = {"goal": 0.3}
    for ordered in (data["options"], list(reversed(data["options"]))):
        data["options"] = ordered
        result = post(client, data, version)
        ranking = result["objective_ranking"]
        assert ranking["status"] == "withheld" and ranking["ranked_options"] == []
        if missing:
            assert "direction" not in ranking and ranking["attested"] is False
        assert all("win_probability" not in row for row in options(result, version).values())
        for field in (
            "recommended_option_id",
            "recommendation_confidence",
            "robustness",
            "path_decomposition",
            "conditional_winners",
            "p_win_sensitivity",
            "factor_evppi",
            "factor_evpc",
            "edge_e_values",
            "factor_flip_values",
            "decision_evpi",
        ):
            assert field not in result
        if version == 2:
            assert result["robustness_status"] == "unavailable"


@pytest.mark.parametrize("version", [1, 2])
def test_equal_shares_are_stably_ordered_ties_without_a_crown(client, version):
    data = payload(include_path_decomposition=True)
    # The effective intervention is identical; an unrelated factor distinguishes
    # valid options without changing any goal outcome.
    data["graph"]["nodes"].append(
        {
            "id": "unrelated",
            "kind": "factor",
            "label": "Unrelated",
            "observed_state": {"value": 0.5},
        }
    )
    for index, option in enumerate(data["options"]):
        option["interventions"] = {"driver": 0.5, "unrelated": 0.1 + index * 0.2}
    left = post(client, data, version)
    data["options"].reverse()
    right = post(client, data, version)
    assert left["objective_ranking"] == right["objective_ranking"]
    rows = left["objective_ranking"]["ranked_options"]
    assert [row["option_id"] for row in rows] == ["high", "low", "middle"]
    assert {row["rank"] for row in rows} == {1}
    assert all(row["win_probability"] == pytest.approx(1 / 3) for row in rows)
    for result in (left, right):
        assert "recommended_option_id" not in result
        assert "path_decomposition" not in result
        assert "robustness" not in result


def test_unrelated_metadata_does_not_move_science(client):
    data = payload()
    before = post(client, data)
    changed = deepcopy(data)
    changed["graph"]["nodes"][0]["label"] = "A display change"
    changed["graph"]["nodes"][0]["observed_state"]["source"] = "user_input"
    after = post(client, changed)
    assert after["objective_ranking"] == before["objective_ranking"]
    assert [row["outcome"] for row in after["options"]] == [
        row["outcome"] for row in before["options"]
    ]


def test_no_informative_draws_withholds_instead_of_zero_tie(client, monkeypatch):
    monkeypatch.setattr(
        science.RobustnessAnalyzerV2, "_winners_for_draw", staticmethod(lambda *args: [])
    )
    result = post(client, payload())
    assert result["objective_ranking"]["withheld_reason"] == "no_informative_draws"
    assert result["objective_ranking"]["ranked_options"] == []
    assert all("win_probability" not in row for row in result["options"])


@pytest.mark.parametrize(
    "changes",
    [
        {"goal_direction": "reduce"},
        {"goal_direction": "target"},
        {"goal_direction": "target", "goal_threshold": 0.5},
    ],
)
def test_unsupported_or_incomplete_objective_is_rejected(client, changes):
    response = client.post(
        "/api/v1/robustness/analyze/v2?response_version=2", json=payload(**changes)
    )
    assert response.status_code == 422


@pytest.mark.parametrize("version", [1, 2])
def test_max_only_enrichments_have_a_computed_control_and_nonmax_suppression(client, version):
    data = payload(
        include_voi=True,
        include_e_values=True,
        include_factor_flips=True,
        include_path_decomposition=True,
        parameter_uncertainties=[
            {"node_id": "background", "distribution": "uniform", "range_min": 0.1, "range_max": 0.9}
        ],
        control_candidates=[{"factor_id": "background", "values": [0.1, 0.9]}],
    )
    data["graph"]["nodes"].append(
        {
            "id": "background",
            "kind": "factor",
            "label": "Background",
            "observed_state": {"value": 0.5},
        }
    )
    data["graph"]["edges"].append(
        {
            "from": "background",
            "to": "goal",
            "exists_probability": 1,
            "strength": {"mean": 0.1, "std": 0.01},
        }
    )
    maximum = post(client, data, version)
    assert maximum.get("robustness")
    assert maximum.get("factor_evppi") and maximum.get("factor_evpc")
    assert maximum.get("factor_flip_values")
    assert all(
        ("downside" if version == 2 else "pre_noise_expected_regret") in row
        for row in options(maximum, version).values()
    )
    data["goal_direction"] = "minimise"
    minimum = post(client, data, version)
    assert minimum["objective_ranking"]["ranked_options"][0]["option_id"] == "low"
    for field in (
        "robustness",
        "factor_evppi",
        "factor_evpc",
        "factor_flip_values",
        "edge_e_values",
        "decision_evpi",
    ):
        assert field not in minimum
    warning = next(
        row
        for row in minimum["inference_warnings"]
        if row["code"] == "OBJECTIVE_METRICS_UNAVAILABLE"
    )
    assert {"factor_evppi", "factor_evpc", "factor_flip_values", "edge_e_values"} <= set(
        warning["detail"]["suppressed_fields"]
    )


@pytest.mark.parametrize("model", [ObjectiveRanking, ObjectiveRankingV2])
@pytest.mark.parametrize(
    "rows",
    [
        [{"option_id": "a", "rank": 1, "win_probability": 0}],
        [{"option_id": "a", "rank": 2, "win_probability": 1}],
        [
            {"option_id": "a", "rank": 1, "win_probability": 0.4},
            {"option_id": "b", "rank": 2, "win_probability": 0.6},
        ],
        [
            {"option_id": "a", "rank": 1, "win_probability": 0.5},
            {"option_id": "b", "rank": 2, "win_probability": 0.5},
        ],
        [
            {"option_id": "a", "rank": 1, "win_probability": 0.5},
            {"option_id": "a", "rank": 1, "win_probability": 0.5},
        ],
        [
            {"option_id": "a", "rank": 1, "win_probability": 0.7},
            {"option_id": "b", "rank": 2, "win_probability": 0.6},
        ],
    ],
)
def test_python_contract_rejects_invalid_order_rank_identity_or_population(model, rows):
    with pytest.raises(ValidationError):
        model(direction="maximise", attested=True, status="computed", ranked_options=rows)


def test_confounding_without_an_objective_cannot_invent_a_robust_recommendation():
    from src.services.confounding_sensitivity import analyze_confounding_sensitivity

    request = RobustnessRequestV2(**payload())
    result = analyze_confounding_sensitivity(
        request.graph,
        [option.model_dump() for option in request.options],
        "goal",
        [("driver", "goal")],
        n_samples=100,
    )
    assert result is None
