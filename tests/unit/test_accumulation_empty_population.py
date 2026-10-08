"""Empty and partial accumulation populations preserve honest omissions and extrema."""

import copy

import numpy as np

import src.services.robustness_analyzer_v2 as rav2
from tests.unit.test_accumulation_identity import PRICE, RATE, accumulation_request, assert_finite_floats
from tests.unit.test_accumulation_refused_draws import cashflow_request
from tests.unit.test_r3_identity_evaluation import v2_body


def test_all_accumulation_options_refused_withhold_every_figure():
    request = cashflow_request()
    for option in request["options"]:
        option["interventions"][RATE] = 1.2

    body = v2_body(request)

    assert body["objective_ranking"]["status"] == "withheld"
    assert body.get("robustness") is None
    assert body["robustness_status"] == "unavailable"
    assert body.get("recommended_option_id") is None
    assert body.get("recommendation_confidence") is None
    assert body.get("conditional_winners") is None
    assert any(
        warning["code"] == "OBJECTIVE_RANKING_WITHHELD"
        and warning["detail"]["reason"] == "no_informative_accumulation_draws"
        for warning in body["inference_warnings"]
    )
    for option in body["options"]:
        assert option["status"] == "failed"
        assert option["outcome"]["n_valid_samples"] == 0
        for field in (
            "win_probability", "probability_of_goal", "probability_of_goal_precision",
            "probability_of_goal_drivers", "constraint_analysis", "downside",
        ):
            assert option.get(field) is None, (option["id"], field, option.get(field))
        for field in ("mean", "std", "p10", "p50", "p90"):
            assert option["outcome"].get(field) is None, (option["id"], field)
    assert_finite_floats(body)


def test_weak_alternative_excludes_that_options_refused_draws():
    analyzer = rav2.RobustnessAnalyzerV2()
    rows = analyzer._compute_alternative_winners(
        {"x->y": ("x", "y")},
        [{("x", "y"): 0.5}] * 4,
        ["keep", "raise", "keep", None],
        "keep",
        informative_win_masks={
            "keep": [True, True, True, False],
            "raise": [False, True, True, False],
        },
    )

    assert len(rows) == 1
    assert rows[0].alternative_winner_id == "raise"
    assert rows[0].switch_probability == 1 / 2


def test_bootstrap_stability_uses_informative_runs_only(monkeypatch):
    request_dict = accumulation_request()
    request_dict["parameter_uncertainties"] = [
        {"node_id": RATE, "distribution": "normal", "std": 0.15},
        {"node_id": PRICE, "distribution": "normal", "std": 0.01},
    ]
    request = rav2.RobustnessRequestV2.model_validate(request_dict)
    monkeypatch.setattr(
        rav2.RobustnessAnalyzerV2,
        "_run_bootstrap_iterations",
        lambda self, *args, **kwargs: {
            RATE: [1.0, np.nan, 3.0], PRICE: [2.0, np.nan, 4.0],
        },
    )

    result = rav2.RobustnessAnalyzerV2()._compute_bootstrap_stability(
        request,
        1.0,
        request.options[0],
        rav2.SCMEvaluatorV2(request.graph),
        rav2.SeededRNG(7),
        {RATE: 1.0, PRICE: 2.0},
        {RATE: 2, PRICE: 1},
        n_bootstrap_override=3,
    )

    assert result[RATE]["elasticity_std"] == round(float(np.std([1.0, 3.0], ddof=1)), 8)
    assert result[PRICE]["elasticity_std"] == round(float(np.std([2.0, 4.0], ddof=1)), 8)
    for node_id in (RATE, PRICE):
        assert result[node_id]["attribution_stability"] is not None
        assert result[node_id]["rank_flip_rate"] == 0.0


def test_partial_accumulation_extrema_condition_on_informative_draws(monkeypatch):
    request = cashflow_request()
    rate = next(node for node in request["graph"]["nodes"] if node["id"] == RATE)
    rate["observed_state"].update(value=0.95, baseline=0.95)
    request["parameter_uncertainties"] = [
        {"node_id": RATE, "distribution": "normal", "std": 0.15}
    ]
    request["goal_threshold"] = 0.0
    request["goal_threshold_frame"] = "delta"
    request["goal_constraints"][0]["value"] = 0.0
    captured = {}
    original = rav2.RobustnessAnalyzerV2._run_monte_carlo

    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured["result"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_run_monte_carlo", capture)
    body = v2_body(request)
    outcomes, wins, _, _, _, constraints, _, _ = captured["result"]

    for option in body["options"]:
        samples = np.asarray(outcomes[option["id"]])
        informative = np.isfinite(samples)
        assert 0 < int(informative.sum()) < samples.size
        values = np.asarray(constraints[option["id"]]["cashflow"])
        assert np.array_equal(np.isfinite(values), informative)
        expected_goal = int(np.count_nonzero(samples[informative] >= 0.0)) / int(informative.sum())
        expected_constraint = int(np.count_nonzero(values[informative] >= 0.0)) / int(informative.sum())
        assert expected_goal == expected_constraint == 1.0
        assert option["probability_of_goal"] == expected_goal
        assert option["probability_of_goal_precision"]["n_informative"] == int(informative.sum())
        assert option["constraint_analysis"]["constraints"][0]["prob_satisfied"] == expected_constraint
        assert option["constraint_analysis"]["joint_probability"] == expected_constraint
        assert option["win_probability"] == wins[option["id"]] / int(informative.sum())
    assert_finite_floats(body)


def test_level_target_excludes_refused_status_quo_comparisons(monkeypatch):
    request = cashflow_request()
    rate = next(node for node in request["graph"]["nodes"] if node["id"] == RATE)
    rate["observed_state"].update(value=0.95, baseline=0.95)
    request["parameter_uncertainties"] = [
        {"node_id": RATE, "distribution": "normal", "std": 0.15}
    ]
    for option in request["options"]:
        option["interventions"][RATE] = 0.5
    request["goal_direction"] = "target"
    request["goal_threshold"] = 0.018
    request["goal_threshold_frame"] = "level"
    captured = {}
    original = rav2.RobustnessAnalyzerV2._run_monte_carlo

    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured["result"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_run_monte_carlo", capture)
    body = v2_body(request)
    outcomes, wins, winners, _, _, _, _, _ = captured["result"]
    ranking_informative = np.asarray([winner is not None for winner in winners])
    assert 0 < int(ranking_informative.sum()) < ranking_informative.size

    for option in body["options"]:
        samples = np.asarray(outcomes[option["id"]])
        assert np.isfinite(samples).all()
        expected = wins[option["id"]] / int(ranking_informative.sum())
        assert option["win_probability"] == expected
    assert body["robustness"]["confidence"] == max(
        option["win_probability"] for option in body["options"]
    )
    assert_finite_floats(body)
