"""Accumulation refusals survive noise and condition statistics on informative draws."""

import copy

import numpy as np

import src.services.robustness_analyzer_v2 as rav2
from tests.unit.test_accumulation_identity import (
    CARRIER, GOAL, RATE, accumulation_request, assert_finite_floats,
)
from tests.unit.test_r3_identity_evaluation import v2_body


def cashflow_request():
    request = accumulation_request()
    request["graph"]["nodes"].append(
        {"id": "cashflow", "kind": "outcome", "label": "Cashflow", "epsilon_std": 0.01}
    )
    request["graph"]["edges"].append(
        {"from": CARRIER, "to": "cashflow", "exists_probability": 1.0,
         "strength": {"mean": 0.5, "std": 0.01}}
    )
    request["goal_constraints"] = [
        {"node_id": "cashflow", "operator": ">=", "value": 0.5, "value_frame": "delta"}
    ]
    return request


def test_refused_accumulation_survives_noisy_downstream_constraint(monkeypatch):
    request = cashflow_request()
    request["options"].append(
        {"id": "invalid", "label": "Invalid churn", "interventions": {RATE: 1.2}}
    )
    captured = {}
    original = rav2.RobustnessAnalyzerV2._run_monte_carlo

    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured["constraints"] = copy.deepcopy(result[5])
        return result

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_run_monte_carlo", capture)
    body = v2_body(request)
    assert all(np.isnan(captured["constraints"]["invalid"]["cashflow"]))
    option = next(option for option in body["options"] if option["id"] == "invalid")
    assert option["status"] == "failed"
    assert option.get("constraint_analysis") is None, option
    assert option.get("probability_of_goal") is None
    assert option.get("win_probability") is None
    assert option.get("probability_of_goal_drivers") is None
    assert all(option["outcome"].get(key) is None for key in ("mean", "std", "p10", "p50", "p90"))
    assert option.get("downside") is None
    assert_finite_floats(body)


def test_partial_accumulation_uses_same_seeded_informative_draws(monkeypatch):
    request = cashflow_request()
    node = next(node for node in request["graph"]["nodes"] if node["id"] == RATE)
    node["observed_state"].update(value=0.95, baseline=0.95)
    request["parameter_uncertainties"] = [{"node_id": RATE, "distribution": "normal", "std": 0.15}]
    request["goal_threshold"] = 0.012
    request["goal_threshold_frame"] = "delta"
    request["goal_constraints"][0]["value"] = 0.015
    captured = {}
    original = rav2.RobustnessAnalyzerV2._run_monte_carlo

    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured["result"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_run_monte_carlo", capture)
    body = v2_body(request)
    outcomes, wins, _, _, _, constraints, _, references = captured["result"]
    parsed = rav2.RobustnessRequestV2.model_validate(request)
    plan = rav2.resolve_identity_plans(parsed.graph, rav2.factor_centres(parsed))[GOAL]
    held = rav2._anchor_of(plan, parsed.graph)
    anchor = held.level / held.frame
    for option in body["options"]:
        samples = np.asarray(outcomes[option["id"]])
        finite = np.isfinite(samples)
        assert 0 < finite.sum() < samples.size
        expected_goal = int(np.count_nonzero(samples[finite] >= request["goal_threshold"])) / int(finite.sum())
        assert 0.0 < expected_goal < 1.0
        assert option["probability_of_goal"] == expected_goal
        assert option["probability_of_goal_precision"]["n_informative"] == int(finite.sum())
        values = np.asarray(constraints[option["id"]]["cashflow"])
        informative = np.isfinite(values)
        assert np.array_equal(informative, finite)
        expected_constraint = int(np.count_nonzero(values[informative] >= 0.015)) / int(informative.sum())
        assert 0.0 < expected_constraint < 1.0
        assert option["constraint_analysis"]["constraints"][0]["prob_satisfied"] == expected_constraint
        assert option["constraint_analysis"]["joint_probability"] == expected_constraint
        assert option["win_probability"] == wins[option["id"]] / int(finite.sum())
        reported = anchor + (samples - np.asarray(references[GOAL]))
        reported = reported[np.isfinite(reported)]
        assert option["outcome"]["mean"] == float(np.mean(reported))
        assert option["outcome"]["p50"] == float(np.percentile(reported, 50))
    assert_finite_floats(body)


def test_no_refused_accumulation_figures_match_round_two(monkeypatch):
    request = cashflow_request()
    captured = {}
    original = rav2.RobustnessAnalyzerV2._run_monte_carlo

    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured["result"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_run_monte_carlo", capture)
    body = v2_body(request)
    outcomes, wins, _, _, _, constraints, _, references = captured["result"]
    parsed = rav2.RobustnessRequestV2.model_validate(request)
    plan = rav2.resolve_identity_plans(parsed.graph, rav2.factor_centres(parsed))[GOAL]
    held = rav2._anchor_of(plan, parsed.graph)
    anchor = held.level / held.frame
    # Round-2 formulae used the complete raw population. Equality must hold exactly;
    # use the same seeded draws rather than pinning platform-specific libm results.
    for option in body["options"]:
        samples = np.asarray(outcomes[option["id"]])
        assert np.isfinite(samples).all()
        reported = anchor + (samples - np.asarray(references[GOAL]))
        assert option["outcome"]["mean"] == float(np.mean(reported))
        assert option["outcome"]["std"] == float(np.std(reported))
        for field, percentile in (("p10", 10), ("p50", 50), ("p90", 90)):
            assert option["outcome"][field] == float(np.percentile(reported, percentile))
        assert option["win_probability"] == wins[option["id"]] / 100
        assert option["probability_of_goal"] == 1.0
        values = np.asarray(constraints[option["id"]]["cashflow"])
        assert np.isfinite(values).all()
        expected = int(np.count_nonzero(values >= 0.5)) / 100
        assert option["constraint_analysis"]["constraints"][0]["prob_satisfied"] == expected
        assert option["constraint_analysis"]["joint_probability"] == expected
