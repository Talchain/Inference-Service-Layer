"""P45 round 3: accumulation change origins await a Science ruling."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from tests.unit.test_accumulation_identity import (
    CARRIER,
    GOAL,
    PRICE,
    accumulation_request,
)
from tests.unit.test_r3_identity_evaluation import v2_body


REASON = "accumulation_change_frame_unsupported"


def change_request(frame: str) -> Dict[str, Any]:
    """The reviewer's exact lower-price target/constraint reproducer."""
    request = accumulation_request()
    request["options"][1]["interventions"][PRICE] = 39.0 / 200.0
    request.update(goal_direction="target", goal_threshold=0.0, goal_threshold_frame=frame)
    request["goal_constraints"] = [
        {
            "constraint_id": "mrr-change",
            "node_id": GOAL,
            "operator": ">=",
            "value": 0.0,
            "value_frame": frame,
        }
    ]
    if frame == "change_rel":
        next(node for node in request["graph"]["nodes"] if node["id"] == GOAL)["raw_range"] = {
            "min": 0.0,
            "max": 100000.0,
        }
    return request


@pytest.mark.parametrize("frame", ["change_abs", "change_rel"])
def test_accumulation_change_goal_and_constraint_are_withheld(frame: str) -> None:
    body = v2_body(change_request(frame))
    for option in body["options"]:
        assert option.get("probability_of_goal") is None
        constraint = option.get("constraint_analysis")
        assert constraint is None or not constraint.get("constraints")
        assert constraint is None or constraint.get("joint_probability") is None
        assert option.get("win_probability") is None
    assert body["objective_ranking"]["status"] == "withheld"
    refusals = [
        warning
        for warning in body["inference_warnings"]
        if warning["detail"].get("reason") == REASON
    ]
    assert {warning["code"] for warning in refusals} == {
        "GOAL_THRESHOLD_NOT_CONVERTIBLE",
        "CONSTRAINT_NOT_CONVERTIBLE",
    }
    assert all(warning["severity"] == "warning" for warning in refusals)
    assert {warning["field"] for warning in refusals} == {
        "goal_threshold_frame",
        "goal_constraints[0].value_frame",
    }


def test_accumulation_level_goal_still_evaluates() -> None:
    request = change_request("change_abs")
    request.update(goal_threshold=0.18, goal_threshold_frame="level")
    request["goal_constraints"][0].update(value=0.18, value_frame="level")
    body = v2_body(request)
    keep, lower = body["options"]
    assert (keep["probability_of_goal"], lower["probability_of_goal"]) == (1.0, 0.0)
    assert (keep["win_probability"], lower["win_probability"]) == (1.0, 0.0)
    assert (
        keep["constraint_analysis"]["constraints"][0]["prob_satisfied"],
        lower["constraint_analysis"]["constraints"][0]["prob_satisfied"],
    ) == (1.0, 0.0)
    assert not any(warning["detail"].get("reason") == REASON for warning in body["inference_warnings"])


@pytest.mark.parametrize("frame", ["change_abs", "change_rel"])
def test_product_change_frame_remains_exactly_unchanged(frame: str) -> None:
    request = change_request(frame)
    next(node for node in request["graph"]["nodes"] if node["id"] == CARRIER).pop("nonlinear_identity")
    request["graph"]["edges"] = [edge for edge in request["graph"]["edges"] if edge["to"] != CARRIER]
    body = v2_body(request)
    keep, lower = body["options"]
    # Round-2 product control: no transcendental arithmetic and no tolerance.
    assert (keep["probability_of_goal"], lower["probability_of_goal"]) == (1.0, 0.0)
    assert (keep["win_probability"], lower["win_probability"]) == (1.0, 0.0)
    assert (
        keep["constraint_analysis"]["constraints"][0]["prob_satisfied"],
        lower["constraint_analysis"]["constraints"][0]["prob_satisfied"],
    ) == (1.0, 0.0)
    assert (
        keep["constraint_analysis"]["joint_probability"],
        lower["constraint_analysis"]["joint_probability"],
    ) == (1.0, 0.0)
    assert not any(warning["detail"].get("reason") == REASON for warning in body["inference_warnings"])


def test_accumulation_carrier_and_ordinary_descendant_limits_are_scoped() -> None:
    request = accumulation_request()
    request["graph"]["nodes"].append(
        {
            "id": "cashflow",
            "kind": "factor",
            "label": "Cashflow",
            "observed_state": {"value": 0.5, "baseline": 0.5, "source": "brief_extraction"},
            "execution_frame": {"frame": 1.0, "carrier": "cap"},
        }
    )
    request["graph"]["edges"].append(
        {
            "from": CARRIER,
            "to": "cashflow",
            "exists_probability": 1.0,
            "strength": {"mean": 0.5, "std": 0.01},
        }
    )
    request["goal_constraints"] = [
        {
            "constraint_id": "carrier-change",
            "node_id": CARRIER,
            "operator": ">=",
            "value": 0.0,
            "value_frame": "change_abs",
        },
        {
            "constraint_id": "cashflow-change",
            "node_id": "cashflow",
            "operator": ">=",
            "value": 0.0,
            "value_frame": "change_abs",
        },
        {
            "constraint_id": "price-level",
            "node_id": PRICE,
            "operator": ">=",
            "value": 0.0,
            "value_frame": "level",
        },
    ]
    body = v2_body(request)
    for option in body["options"]:
        assert option["probability_of_goal"] == 1.0
        assert [row["constraint_id"] for row in option["constraint_analysis"]["constraints"]] == [
            "price-level"
        ]
        assert option["constraint_analysis"]["constraints"][0]["prob_satisfied"] == 1.0
        assert option["constraint_analysis"].get("joint_probability") is None
    refusals = [
        warning
        for warning in body["inference_warnings"]
        if warning["detail"].get("reason") == REASON
    ]
    assert {warning["detail"]["constraint_id"] for warning in refusals} == {
        "carrier-change",
        "cashflow-change",
    }
    assert all(warning["code"] == "CONSTRAINT_NOT_CONVERTIBLE" for warning in refusals)
