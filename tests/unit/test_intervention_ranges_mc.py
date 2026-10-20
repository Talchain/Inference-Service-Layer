"""L3: only a declared outcome law propagates; endpoints do not invent one."""

import copy

import numpy as np
import pytest

from fastapi.testclient import TestClient

from src.api.main import app
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


def payload(frame=1000, subject="bookings"):
    x, y, goal = (f"{subject}_{name}" for name in ("variable", "exact", "total"))
    nodes = [
        {
            "id": n, "kind": "goal" if n == goal else "factor", "label": n,
            "observed_state": {"value": value / frame},
            "execution_frame": {"frame": frame, "carrier": "cap"},
            "intercept": 0, "epsilon_std": 0,
        }
        for n, value in ((x, 40), (y, 7), (goal, 47))
    ]
    nodes[-1]["nonlinear_identity"] = {
        "operation": "sum", "factor_ids": [x, y], "stated_in_brief": True,
    }
    return {
        "graph": {
            "nodes": nodes,
            "edges": [
                {"from_": n, "to": goal, "exists_probability": 1,
                 "strength": {"mean": 1, "std": 0.01}}
                for n in (x, y)
            ],
        },
        "options": [
            {"id": "ranged", "label": "Uncertain effect", "interventions": {x: 40 / frame, y: 7 / frame},
             "intervention_ranges": {x: {
                 "low": 20, "high": 80, "meaning": "likely_range",
                 "source": "Our added output is likely 20–80 per week, relative to today's output.",
                 "normalisation": {"raw_at_zero": 0, "raw_at_one": frame},
                 "law": {
                     "family": "uniform", "role": "uncertain_outcome",
                     "origin": "olumi_illustrative",
                     "basis": "Untested illustrative uniform law within the supplied bounds; no tails.",
                     "quantity": subject, "reference": "increment relative to today",
                     "period": "one week", "conditions": ["same operating conditions"],
                     "independent_of_model": True,
                 },
             }}},
            {"id": "point", "label": "Exact setting", "interventions": {x: 50 / frame, y: 7 / frame}},
        ],
        "goal_node_id": goal, "goal_threshold": 57 / frame,
        "goal_threshold_frame": "delta", "n_samples": 2000, "seed": 42,
        "include_voi": True,
    }


def post(body):
    response = TestClient(app).post(
        "/api/v1/robustness/analyze/v2", json=body,
        headers={"X-ISL-Response-Version": "2"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    return data, {option["id"]: option for option in data["options"]}


def captured_draws(body):
    request = RobustnessRequestV2.model_validate(body)
    request._capture_draws = True
    return RobustnessAnalyzerV2().analyze(request)._mc_draws["option_outcomes"]


@pytest.mark.parametrize("frame,subject", [(1000, "bookings"), (100, "hours")])
@pytest.mark.parametrize("with_limit", [False, True])
def test_real_route_declared_law_spread_exact_sum_and_exact_other_option(frame, subject, with_limit):
    body = payload(frame, subject)
    if with_limit:
        body["goal_constraints"] = [{
            "constraint_id": "output", "node_id": f"{subject}_total", "operator": ">=",
            "value": 57 / frame, "value_frame": "delta",
        }]
    data, options = post(body)
    ranged, point = options["ranged"], options["point"]
    assert ranged["outcome"]["std"] == pytest.approx(60 / np.sqrt(12) / frame, rel=0.06)
    assert ranged["probability_of_goal"] == pytest.approx(0.5, abs=0.04)
    assert point["outcome"]["std"] == pytest.approx(0, abs=1e-14)
    assert point["outcome"]["p10"] == point["outcome"]["p90"] == 57 / frame
    assert point["probability_of_goal"] == 1
    assert data["auto_noise_applied"] is False
    if with_limit:
        for option in options.values():
            assert option["constraint_analysis"]["constraints"][0]["prob_satisfied"] == option["probability_of_goal"]
    original = body["options"][0]["intervention_ranges"][f"{subject}_variable"]
    assert ranged["sampled_intervention_ranges"] == [{
        "node_id": f"{subject}_variable", "meaning": "likely_range", "family": "uniform",
        "coverage": 1, "low": 20, "high": 80,
        "source": original["source"], "law": original["law"],
    }]
    assert "sampled_intervention_ranges" not in point
    # Marginal populations do not license a joint comparison, for ANY target.
    assert data["objective_ranking"]["status"] == "withheld"
    for option in options.values():
        assert "win_probability" not in option
        assert "downside" not in option
        assert "status_reason" not in option
    for key in ("decision_evpi", "factor_evppi", "factor_evpc", "conditional_winners",
                "p_win_sensitivity", "robustness", "path_decomposition"):
        assert data.get(key) is None
    draws = captured_draws(body)
    raw = np.asarray(draws["ranged"]) * frame - 7
    assert raw.min() >= 20 and raw.max() <= 80
    np.testing.assert_allclose(np.quantile(raw, [0.1, 0.5, 0.9]), [26, 50, 74], atol=2)
    np.testing.assert_array_equal(draws["point"], np.full(body["n_samples"], 57 / frame))


@pytest.mark.parametrize("missing", ["law", "independence", "multiple", "source", "family", "min_max"])
def test_no_law_or_undeclared_dependence_retains_points_withholds_only_unsupported_chance(missing):
    body = payload()
    ranges = body["options"][0]["intervention_ranges"]
    if missing == "law":
        ranges["bookings_variable"].pop("law")
    elif missing == "min_max":
        ranges["bookings_variable"].pop("law")
        ranges["bookings_variable"]["meaning"] = "min_max"
    elif missing == "family":
        ranges["bookings_variable"]["law"]["family"] = "lognormal"
    elif missing == "source":
        ranges["bookings_variable"].pop("source")
    elif missing == "independence":
        ranges["bookings_variable"]["law"]["independent_of_model"] = False
    else:
        ranges["bookings_exact"] = copy.deepcopy(ranges["bookings_variable"])
        ranges["bookings_exact"].update(low=5, high=9)
    data, options = post(body)
    assert options["ranged"]["outcome"]["std"] == pytest.approx(0, abs=1e-14)
    assert options["ranged"]["outcome"]["p10"] == options["ranged"]["outcome"]["p90"] == 0.047
    for key in ("probability_of_goal", "probability_of_goal_precision", "probability_of_goal_drivers",
                "sampled_intervention_ranges"):
        assert key not in options["ranged"]
    assert options["point"]["probability_of_goal"] == 1
    assert any(w["detail"].get("option_id") == "ranged"
               for w in data["inference_warnings"])


def test_controllable_setting_is_not_an_uncertain_outcome():
    body = payload()
    body["options"][0]["intervention_ranges"]["bookings_variable"]["law"]["role"] = "setting"
    _, options = post(body)
    assert options["ranged"]["outcome"]["std"] == pytest.approx(0, abs=1e-14)
    assert "sampled_intervention_ranges" not in options["ranged"]


def test_unrelated_range_does_not_suppress_a_goals_supported_probability():
    body = payload()
    body["graph"]["nodes"].append({"id": "unrelated", "kind": "factor", "label": "unrelated"})
    option = body["options"][0]
    option["interventions"]["unrelated"] = 0.04
    option["intervention_ranges"] = {"unrelated": option["intervention_ranges"]["bookings_variable"]}
    option["intervention_ranges"]["unrelated"].pop("law")
    _, options = post(body)
    assert options["ranged"]["probability_of_goal"] == 0
    assert "sampled_intervention_ranges" not in options["ranged"]


def test_absent_ranges_preserve_exact_controls_and_legacy_ranking():
    body = payload()
    body["options"][0].pop("intervention_ranges")
    data, options = post(body)
    assert data["objective_ranking"]["status"] == "computed"
    assert options["ranged"]["outcome"]["std"] == pytest.approx(0, abs=1e-14)
    assert options["ranged"]["probability_of_goal"] == 0
    assert options["point"]["win_probability"] == 1


def test_other_options_law_does_not_change_draws_or_borrow_source():
    body = payload()
    before = captured_draws(body)
    other = body["options"][1]
    other["intervention_ranges"] = copy.deepcopy(body["options"][0]["intervention_ranges"])
    stated = other["intervention_ranges"]["bookings_variable"]
    stated.update(low=30, high=90, source="A different speaker's range is 30–90.")
    after = captured_draws(body)
    assert before["ranged"] == after["ranged"]
    _, options = post(body)
    assert options["point"]["sampled_intervention_ranges"][0]["source"] == stated["source"]
    assert options["ranged"]["sampled_intervention_ranges"][0]["low"] == 20
    body["options"].reverse()
    assert captured_draws(body) == after


def test_v1_route_and_worker_payload_preserve_law_and_withheld_comparison():
    body = payload()
    response = TestClient(app).post(
        "/api/v1/robustness/analyze/v2", json=body,
        headers={"X-ISL-Response-Version": "1"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["objective_ranking"]["status"] == "withheld"
    assert all(row.get("win_probability") is None for row in data["results"])
    # This is the process-pool's actual dump/validate boundary, not a private attr.
    from src.models.robustness_v2 import RobustnessResponseV2

    result = RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(body))
    restored = RobustnessResponseV2.model_validate_json(result.model_dump_json())
    ranged = next(row for row in restored.results if row.option_id == "ranged")
    original = body["options"][0]["intervention_ranges"]["bookings_variable"]
    assert ranged.sampled_intervention_ranges[0].law.model_dump() == original["law"]
    assert ranged.sampled_intervention_ranges[0].source == original["source"]
    assert all(row.win_probability is None for row in restored.results)
