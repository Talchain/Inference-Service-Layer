"""Round-two regressions: deliberate probes and carrier-scoped epsilon cuts."""

import copy

import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import RobustnessRequestV2
from tests.unit.test_accumulation_identity import CARRIER, GOAL, INFLOW, RATE, STOCK
from tests.unit.test_accumulation_rate_spread import b1_request, sampled_response, two_distinct_carriers


def test_deliberate_inflow_probe_finds_reviewers_central_crossing():
    raw = b1_request(n_samples=100)
    raw.update(include_factor_flips=True, goal_direction="maximise")
    raw["options"] = [
        {"id": "A", "label": "A", "interventions": {STOCK: 0.25}},
        {"id": "B", "label": "B", "interventions": {STOCK: 0.20, INFLOW: 0.30}},
    ]
    raw["parameter_uncertainties"] = [
        {"node_id": INFLOW, "distribution": "normal", "std": 0.05}
    ]
    response = sampled_response(raw)
    flip = next(row for row in response.factor_flip_values if row["factor_id"] == INFLOW)
    assert flip["flip_reason"] == "found"
    assert flip["flip_value"] == pytest.approx(0.266006, rel=1e-6)
    assert flip["direction"] == "increase"
    assert flip["alternative_winner_id"] == "A"
    assert flip["stability"]["n_seeds"] > 0
    assert flip["stability"]["n_seeds_flipped"] == flip["stability"]["n_seeds"]
    assert flip["stability"]["seed_flip_values"] == pytest.approx(
        [0.266006] * flip["stability"]["n_seeds"], rel=1e-6
    )

    request = RobustnessRequestV2.model_validate(raw)
    evaluator = rav2.SCMEvaluatorV2(request.graph, factor_centres=rav2.factor_centres(request))
    edges = {(e.from_, e.to): e.strength.mean for e in request.graph.edges}
    analyzer = rav2.RobustnessAnalyzerV2()
    d = 0.97 ** 12
    g = (1.0 - d) / 0.03
    crossing = 30.0 - 50.0 * d / g
    assert crossing == pytest.approx(26.60056, rel=1e-6)
    # Reproduce the reviewer's table through the real deliberate-probe consumer.
    for incoming, expected_a, expected_b, winner in (
        (20.0, 18500.72, 21801.38, "B"),
        (30.0, 23501.29, 21801.38, "A"),
    ):
        goals = analyzer._option_goals(request, evaluator, edges, {INFLOW: incoming / 100.0})
        assert goals["A"] * 100000.0 == pytest.approx(49.0 * (250.0 * d + incoming * g), rel=1e-12)
        assert goals["B"] * 100000.0 == pytest.approx(49.0 * (200.0 * d + 30.0 * g), rel=1e-12)
        assert goals["A"] * 100000.0 == pytest.approx(expected_a, abs=0.01)
        assert goals["B"] * 100000.0 == pytest.approx(expected_b, abs=0.01)
        assert analyzer._argmax_option(goals) == winner
    rate_row = next(row for row in response.factor_flip_values if row["factor_id"] == RATE)
    assert rate_row["flip_reason"] == "nonlinear_response"
    assert rate_row["flip_value"] is None
    assert rate_row["direction"] is None
    assert rate_row["alternative_winner_id"] is None
    assert rate_row.get("stability") is None
    baseline = 49.0 * (250.0 * d + 25.0 * g) / 100000.0
    expected_elasticity = (49.0 * 25.0 * g / 100000.0) / baseline
    sensitivities = analyzer._compute_factor_sensitivity(
        request, {"A": [baseline], "B": [49.0 * (200.0 * d + 30.0 * g) / 100000.0]},
        rav2.SeededRNG(101), evaluator,
    )
    sensitivity = next(row for row in sensitivities if row.node_id == INFLOW)
    assert sensitivity.elasticity == pytest.approx(expected_elasticity, rel=1e-12)
    bootstrap = analyzer._run_bootstrap_iterations(
        request, baseline, request.options[0], evaluator, 101, 2
    )
    assert len(bootstrap[INFLOW]) == 2
    assert bootstrap[INFLOW] == pytest.approx([expected_elasticity] * 2, rel=1e-12)


def stock_blocked_epsilon_request():
    raw = b1_request(n_samples=100)
    raw["graph"]["nodes"].append({
        "id": "upstream", "kind": "factor", "label": "upstream",
        "observed_state": {"value": 0.5, "baseline": 0.5}, "epsilon_std": 0.1,
    })
    raw["graph"]["edges"].extend([
        {"from": "upstream", "to": STOCK, "exists_probability": 1.0,
         "strength": {"mean": 0.1, "std": 0.01}},
        {"from": STOCK, "to": RATE, "exists_probability": 1.0,
         "strength": {"mean": 0.02, "std": 0.01}},
    ])
    return raw


def test_deliberate_operand_levels_override_draws_and_options_override_probes():
    request = RobustnessRequestV2.model_validate(b1_request(n_samples=100))
    evaluator = rav2.SCMEvaluatorV2(request.graph, factor_centres=rav2.factor_centres(request))
    edges = {(edge.from_, edge.to): edge.strength.mean for edge in request.graph.edges}
    draws = {STOCK: 0.8, RATE: 0.8, INFLOW: 0.8}
    probes = {STOCK: 0.2, RATE: 0.04, INFLOW: 0.3}

    def monthly(stock, churn, incoming):
        for _ in range(12):
            stock = stock * (1.0 - churn) + incoming
        return stock / 1000.0

    assert evaluator.evaluate(edges, {}, CARRIER, factor_values=draws) == pytest.approx(
        monthly(250.0, 0.03, 25.0), rel=1e-12
    )
    assert evaluator.evaluate(
        edges, {}, CARRIER, factor_values=draws, factor_level_overrides=probes
    ) == pytest.approx(monthly(200.0, 0.04, 30.0), rel=1e-12)
    multi = evaluator.evaluate_multi(
        edges, {INFLOW: 0.4}, [CARRIER], factor_values=draws, factor_level_overrides=probes
    )
    assert multi[CARRIER] == pytest.approx(monthly(200.0, 0.04, 40.0), rel=1e-12)


def test_epsilon_blocked_by_exact_stock_preserves_goal_probabilities_and_distributions():
    raw = stock_blocked_epsilon_request()
    quiet = copy.deepcopy(raw)
    quiet["graph"]["nodes"][-1]["epsilon_std"] = 0.0
    noisy_response, quiet_response = sampled_response(raw), sampled_response(quiet)
    noisy = {row.option_id: row for row in noisy_response.results}
    control = {row.option_id: row for row in quiet_response.results}
    assert {key: row.probability_of_goal for key, row in noisy.items()} == {
        key: row.probability_of_goal for key, row in control.items()
    } == {"keep": 0.76, "raise": 1.0}
    for key in noisy:
        assert noisy[key].outcome_distribution.model_dump() == control[key].outcome_distribution.model_dump()
    assert not any(warning.detail.get("reason") == "epsilon_breaks_status_quo_reference"
                   for warning in noisy_response.inference_warnings)
    assert rav2.RobustnessAnalyzerV2._noisy_influencers(
        RobustnessRequestV2.model_validate(raw), GOAL
    ) == []
    frames = {frame.node_id: frame for frame in noisy_response.node_levels}
    assert frames[CARRIER].frame == frames[GOAL].frame == "anchored_level"


def test_epsilon_bypass_around_exact_stock_still_withholds_goal_probability():
    raw = stock_blocked_epsilon_request()
    raw["graph"]["edges"].append({
        "from": "upstream", "to": GOAL, "exists_probability": 1.0,
        "strength": {"mean": 0.01, "std": 0.002},
    })
    response = sampled_response(raw)
    assert {row.option_id for row in response.results} == {"keep", "raise"}
    assert all(row.probability_of_goal is None for row in response.results)
    refusals = [warning for warning in response.inference_warnings
                if warning.detail.get("reason") == "epsilon_breaks_status_quo_reference"]
    assert refusals
    assert refusals[0].detail["noisy_node_ids"] == ["upstream"]


def test_epsilon_cuts_are_scoped_to_the_carrier_reading_the_operand():
    raw = two_distinct_carriers("stock_b")
    # B's exact stock does not suppress its raw epsilon when A reads it as an
    # upstream driver of A's churn. Cutting stock nodes globally would be unsafe.
    next(node for node in raw["graph"]["nodes"] if node["id"] == "stock_b")["epsilon_std"] = 0.1
    request = RobustnessRequestV2.model_validate(raw)
    assert rav2.RobustnessAnalyzerV2._noisy_influencers(request, "carrier_b") == []
    assert rav2.RobustnessAnalyzerV2._noisy_influencers(request, CARRIER) == ["stock_b"]
