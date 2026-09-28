from __future__ import annotations

import copy
import json
import re
from fractions import Fraction

import pytest

from mutant_checks import run as run_mutants
from oracle import truth
from r3b_adapter import PricingAdapter, reproduce_stored_thresholds
from regions import Refusal, classify_coordinate, sha256, validate_request, validate_result
from synthetic import measurements, request
from visual import render_html


def test_frozen_reference_crossovers_and_island():
    for case, coords in {
        "F1": [(0, 1), (1, 0), (.5, .5)],
        "F2": [(.2, .1), (.1, .2), (.5, .5)],
        "F3": [(1, 1), (.4, .4), (0, 1)],
        "F4": [(1, 1)],
        "F5": [(-1, -1), (-1, 1), (0, 0)],
        "F6": [(-1, 0), (0, 0), (1, 0)],
        "F8": [(0, 0), (1, 1)],
        "F9": [(.8, .8), (.2, .2)],
        "F11": [(0, 0), (.01, .01), (.5, .5)],
        "F13": [(0, 0), (1, 0)],
    }.items():
        q = request(case, 11)
        validate_request(q, {"x": "unit", "y": "unit"})
        for x, y in coords:
            raw = measurements(case, x, y)
            result = classify_coordinate(q, x, y, raw, status="OUTSIDE_DECLARED_DOMAIN" if raw is None else "EVALUATED")
            expected = truth(case, x, y)
            assert result["state"] == expected["state"]
            if raw is not None:
                assert result["named_preference"] == expected["named_preference"]
                assert {oid: r["feasibility"] for oid, r in result["options"].items()} == expected["feasibility"]


def test_goal_and_constraint_do_not_compensate():
    q = request("F3", 11)
    p = classify_coordinate(q, 1, 1, measurements("F3", 1, 1))
    assert p["options"]["A"]["goal"] == "ATTAINED"
    assert p["options"]["A"]["feasibility"] == "INFEASIBLE"
    assert p["overall_preference"] == "NO_FEASIBLE_OPTION"
    p4 = classify_coordinate(request("F4", 11), 1, 1, measurements("F4", 1, 1))
    assert p4["overall_preference"] == "INCOMPLETE_COMPARISON"


def test_objective_absent_still_reports_feasibility():
    q = request("F3", 11, objective=False)
    p = classify_coordinate(q, .4, .4, measurements("F3", .4, .4))
    assert p["named_preference"] == "OBJECTIVE_UNSPECIFIED"
    assert all(o["feasibility"] == "FEASIBLE" for o in p["options"].values())


def test_equivalence_groups_require_every_pair():
    q = request("F1", 11)
    q["option_ids"] = ["A", "B", "C"]
    q["option_labels"]["C"] = "Synthetic option C"
    q["comparison_set"] = ["A", "B", "C"]
    q["objective"]["delta"] = .1
    rows = {oid: {"status": "COMPUTED", "objective_value": value, "goal_value": None, "constraints": {}, "tier": "SYNTHETIC", "reason": None} for oid, value in {"A": Fraction(0), "B": Fraction(9, 100), "C": Fraction(18, 100)}.items()}
    p = classify_coordinate(q, 0, 0, rows)
    assert p["pairwise"]["A|B"]["state"] == "PRACTICALLY_EQUIVALENT"
    assert p["pairwise"]["B|C"]["state"] == "PRACTICALLY_EQUIVALENT"
    assert p["named_preference"] == "NUMERICALLY_UNRESOLVED"


def test_schema_rejects_duplicate_ids_and_unsupported_operation():
    q = request("F1", 11)
    q["axes"][1]["id"] = "x"
    with pytest.raises(Refusal, match="DUPLICATE_ID"):
        validate_request(q, {"x": "unit", "y": "unit"})
    q = request("F1", 11)
    q["axes"][0]["operation"] = "CONDITION_ON_OBSERVATION"
    with pytest.raises(Refusal, match="UNSUPPORTED_SEMANTICS"):
        validate_request(q, {"x": "unit", "y": "unit"})


def test_semantic_identity_includes_fixed_and_criterion():
    original = request("F13", 11)
    changed = request("F13", 11, fixed_c=.2)
    assert sha256(original) != sha256(changed)
    changed2 = copy.deepcopy(original)
    changed2["objective"]["delta"] = .1
    assert len({sha256(original), sha256(changed), sha256(changed2)}) == 3


def test_synthetic_result_is_byte_identical_across_runs(tmp_path):
    from study import run_synthetic
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    run_synthetic("F1", 11, first)
    run_synthetic("F1", 11, second)
    assert (first / "synthetic-F1-11.json").read_bytes() == (second / "synthetic-F1-11.json").read_bytes()


def test_known_bad_controls():
    evidence = run_mutants()
    assert all(evidence["controls"].values())


def test_r3b_frozen_replay_and_competitive_response():
    assert reproduce_stored_thresholds()["status"] == "MATCH"
    a = PricingAdapter("X_net_reading")
    q = a.request(11)
    validate_request(q, a.binding_units)
    a.validate_request_identity(q)
    a.assert_composition(0, -31250)
    p = classify_coordinate(q, 0, -31250, a.measurements(0, -31250))
    option = p["options"]["59_with_feature_release"]
    assert option["goal"] == "MISSED"
    assert option["feasibility"] == "FEASIBLE"
    assert option["constraints"][0]["state"] == "SATISFIED"
    assert p["overall_preference"] == "INCOMPLETE_COMPARISON"
    assert p["named_preference"].startswith("POINT_ESTIMATE_PREFERRED:")


def test_saved_results_and_offline_view():
    from pathlib import Path
    output = Path(__file__).resolve().parent.parent / "output"
    result = json.loads((output / "r3b-X_net_reading-41.json").read_text())
    validate_result(result)
    assert result["manifest"]["source_verified"]
    assert result["metrics"]["false_feasible"] is None
    html = render_html([result, result])
    assert 'type="application/json"' in html
    assert re.search(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://', html, re.I) is None
    assert "Matched point" in html and "Threshold plus question" in html and "Two-dimensional region" in html
