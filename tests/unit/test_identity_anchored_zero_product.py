"""Capacity starts at zero; hiring changes the literal product, without a fitted 0/0.

Synthetic capacity shape: additional capacity = specialists x capacity per specialist.
Stated zero and its provenance survive. This is an exact definition, not a slope guess.
"""

import copy

import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
    identity_partials,
    resolve_identity_plans,
)
from tests.unit.test_r3_identity_evaluation import blocked_422, identity_critique, v2_body


def capacity_request():
    def node(id, value, frame, kind="factor"):
        return {
            "id": id, "label": id, "kind": kind,
            "observed_state": {"value": value, "raw_value": value * frame,
                               "source": "brief_extraction"},
            "execution_frame": {"frame": frame, "carrier": "cap"},
        }

    total = node("capacity", 0, 100, "outcome")
    total["nonlinear_identity"] = {
        "operation": "product", "factor_ids": ["specialists", "capacity_each"],
        "stated_in_brief": True,
    }
    return {
        "graph": {
            "nodes": [node("specialists", 0, 10), node("capacity_each", .4, 30), total],
            # Deliberately >1: these edges are definitions, never sampled belief slopes.
            "edges": [{"from": p, "to": "capacity", "exists_probability": 1,
                       "strength": {"mean": 2.2, "std": .1}}
                      for p in ["specialists", "capacity_each"]],
        },
        "options": [
            {"id": "stay", "label": "Stay", "interventions": {"specialists": 0}},
            {"id": "hire", "label": "Hire two", "interventions": {"specialists": .2}},
        ],
        "goal_node_id": "capacity", "goal_threshold": .1,
        "goal_threshold_frame": "level", "seed": 4242, "n_samples": 100,
    }


def graph(d):
    return RobustnessRequestV2.model_validate(d).graph


def test_stated_zero_is_kept_and_the_hiring_effect_is_24_in_natural_units():
    d = capacity_request()
    before = copy.deepcopy(d)
    g = graph(d)
    evaluator = SCMEvaluatorV2(g)
    strengths = {(e.from_, e.to): e.strength.mean for e in g.edges}
    assert evaluator.identity_plans["capacity"].evaluated
    assert evaluator.identity_plans["capacity"].target_level == 0
    assert evaluator.evaluate(strengths, {}, "capacity") == 0
    assert evaluator.evaluate(strengths, {"specialists": .2}, "capacity") * 100 == pytest.approx(24)
    assert d == before
    assert g.nodes[-1].observed_state.source == "brief_extraction"


def test_current_partials_preserve_the_zero_gate_and_the_real_hiring_effect():
    partials = identity_partials(graph(capacity_request()), {"specialists": 0, "capacity_each": .4})
    assert partials[("capacity_each", "capacity")] == 0
    assert partials[("specialists", "capacity")] == pytest.approx(12 * 10 / 100)


def test_rescaling_the_specialist_frame_does_not_change_the_natural_unit_effect():
    d = capacity_request()
    d["graph"]["nodes"][0]["execution_frame"]["frame"] *= 2
    g = graph(d)
    evaluator = SCMEvaluatorV2(g)
    strengths = {(e.from_, e.to): e.strength.mean for e in g.edges}
    assert evaluator.evaluate(strengths, {"specialists": .1}, "capacity") * 100 == pytest.approx(24)


def test_two_zero_parts_keep_the_interaction_when_both_change():
    d = capacity_request()
    d["graph"]["nodes"][1]["observed_state"].update(value=0, raw_value=0)
    g = graph(d)
    evaluator = SCMEvaluatorV2(g)
    strengths = {(e.from_, e.to): e.strength.mean for e in g.edges}
    assert evaluator.evaluate(strengths, {"specialists": .2, "capacity_each": .4}, "capacity") * 100 == pytest.approx(24)


def test_api_computes_goal_chances_and_says_the_stated_identity_was_evaluated():
    body = v2_body(capacity_request())
    entry = body["identity_evaluations"][0]
    assert entry["evaluated"] and entry["level_source"] == "stated_level"
    assert entry["stated_in_brief"] is True
    # The ordinary analyzer carries the same goal probability used by the API adapter.
    response = RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(capacity_request()))
    assert {r.option_id: r.probability_of_goal for r in response.results} == {"stay": 0, "hire": 1}


@pytest.mark.parametrize("which", ["nonzero_total", "nonzero_parts", "addend", "belief"])
def test_unsupported_zero_shapes_still_block_instead_of_guessing(which):
    d = capacity_request()
    nodes = {n["id"]: n for n in d["graph"]["nodes"]}
    if which == "nonzero_total":
        nodes["capacity"]["observed_state"].update(value=.1, raw_value=10)
    elif which == "nonzero_parts":
        nodes["specialists"]["observed_state"].update(value=.1, raw_value=1)
    else:
        d["graph"]["nodes"].append({
            "id": "extra", "label": "Extra", "kind": "factor",
            "observed_state": {"value": 0},
            "execution_frame": {"frame": 100, "carrier": "cap"},
        })
        d["graph"]["edges"].append({"from": "extra", "to": "capacity",
                                     "strength": {"mean": .2, "std": .1}})
        if which == "addend":
            nodes["capacity"]["nonlinear_identity"]["addends"] = ["extra"]
    assert resolve_identity_plans(graph(d))["capacity"].withheld_reason == "identity_zero_level"
    assert identity_critique(blocked_422(d))["identity"]["withheld_reason"] == "identity_zero_level"


def test_definitional_coefficients_above_one_do_not_disable_e_values(monkeypatch):
    d = capacity_request()
    d["include_e_values"] = True
    analyzer = RobustnessAnalyzerV2()
    calls = []

    def search(*args, **kwargs):
        calls.append(True)
        return []

    monkeypatch.setattr(analyzer, "_compute_edge_e_values", search)
    response = analyzer.analyze(RobustnessRequestV2.model_validate(d))
    assert calls == [True]
    assert not any(w.code == "E_VALUES_UNAVAILABLE" for w in response.inference_warnings)
