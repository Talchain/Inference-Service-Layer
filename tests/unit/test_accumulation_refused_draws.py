"""An accumulation refusal invalidates its whole option or status-quo identity."""

import copy
import inspect

import numpy as np
import pytest

import src.services.robustness_analyzer_v2 as rav2
from tests.unit.test_accumulation_identity import (
    CARRIER, GOAL, INFLOW, PRICE, RATE, accumulation_request, assert_finite_floats,
)
from tests.unit.test_r3_identity_evaluation import blocked_422, v2_body


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


def assert_failed_option(option):
    assert option["status"] == "failed", option
    assert option["outcome"]["n_valid_samples"] == 0
    for field in (
        "win_probability", "probability_of_goal", "probability_of_goal_precision",
        "probability_of_goal_drivers", "constraint_analysis", "downside",
    ):
        assert option.get(field) is None, (field, option.get(field))
    for field in ("mean", "std", "p10", "p50", "p90"):
        assert option["outcome"].get(field) is None, field


def test_one_refused_option_withholds_all_figures_and_preserves_other_options(monkeypatch):
    """Reviewer row: a framed 120% churn affects only the failed option."""
    request = cashflow_request()
    control = v2_body(copy.deepcopy(request))
    request["options"].append(
        {"id": "invalid", "label": "120% monthly churn", "interventions": {RATE: 1.2}}
    )
    captured = {}
    original = rav2.RobustnessAnalyzerV2._run_monte_carlo

    def capture(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        captured["result"] = copy.deepcopy(result)
        return result

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_run_monte_carlo", capture)
    body = v2_body(request)
    options = {option["id"]: option for option in body["options"]}
    assert_failed_option(options["invalid"])
    # Compare every returned field of each remaining option, including its whole
    # outcome band, win/goal chances, precision, drivers, constraint/joint chance
    # and downside. Both requests have the same seed, healthy options and order.
    assert [options[option["id"]] for option in control["options"]] == control["options"]
    # The original MC result is the population entering shared consumers: no
    # failed-option draw or win can enter sensitivity, VOI, flips or calibration.
    outcomes, wins, winners, _, _, constraints, _, _ = captured["result"]
    assert "invalid" not in outcomes
    assert "invalid" not in wins
    assert "invalid" not in constraints
    assert "invalid" not in winners
    assert_finite_floats(captured["result"])
    assert_finite_floats(body)


def test_churn_uncertainty_crossing_one_fails_the_whole_option(monkeypatch):
    request = accumulation_request(rate=90.0)
    request["graph"]["nodes"].append(
        {"id": "churn_pressure", "kind": "factor", "label": "Churn pressure",
         "observed_state": {"value": 0.1, "baseline": 0.1, "source": "brief_extraction"}}
    )
    request["graph"]["edges"].append(
        {"from": "churn_pressure", "to": RATE, "exists_probability": 1.0,
         "strength": {"mean": 0.5, "std": 0.01}}
    )
    request["parameter_uncertainties"] = [
        {"node_id": "churn_pressure", "distribution": "uniform", "range_min": 0.0, "range_max": 0.2}
    ]
    # At status quo the framed nonroot churn stays at its held 90% level.
    # The third option changes its uncertain parent's level, making this
    # option's churn .9 + sampled_edge * (.25 - sampled_pressure) straddle 1.
    for option in request["options"]:
        option["interventions"][RATE] = 0.9
    request["options"].append(
        {"id": "uncertain", "label": "Uncertain monthly churn",
         "interventions": {PRICE: 0.295, "churn_pressure": 0.25}}
    )
    sampled_rates = []
    original = rav2.SCMEvaluatorV2._identity_value

    def capture(self, plan, edge_strengths, values, status_quo):
        if plan.operation == "accumulation" and values.get("churn_pressure") == 0.25:
            assert status_quo is not None
            sampled_rates.append(plan.levels[RATE] + values[RATE] - status_quo[RATE])
        return original(self, plan, edge_strengths, values, status_quo)

    monkeypatch.setattr(rav2.SCMEvaluatorV2, "_identity_value", capture)
    body = v2_body(request)
    assert sampled_rates and min(sampled_rates) < 1.0 <= max(sampled_rates)
    options = {option["id"]: option for option in body["options"]}
    assert_failed_option(options["uncertain"])
    assert all(options[option_id]["status"] != "failed" for option_id in ("keep", "raise"))
    assert_finite_floats(body)


def test_failed_option_reaches_no_shared_analysis_consumer(monkeypatch):
    request = cashflow_request()
    request["options"].append(
        {"id": "invalid", "label": "Refused monthly churn", "interventions": {RATE: 1.2}}
    )
    request["parameter_uncertainties"] = [
        {"node_id": INFLOW, "distribution": "uniform", "range_min": 0.15, "range_max": 0.25}
    ]
    request["analysis_types"] = ["comparison", "sensitivity", "robustness"]
    request["include_voi"] = True
    request["include_e_values"] = True
    request["include_factor_flips"] = True
    methods = (
        "_compute_sensitivity", "_compute_factor_sensitivity", "_compute_robustness",
        "_compute_conditional_winners", "_compute_evpi", "_compute_factor_evppi",
        "_compute_edge_e_values", "_compute_factor_flip_values",
    )
    observed = set()

    def guard(method):
        original = getattr(rav2.RobustnessAnalyzerV2, method)
        signature = inspect.signature(original)

        def guarded(self, *args, **kwargs):
            observed.add(method)
            arguments = signature.bind(self, *args, **kwargs).arguments
            for argument in arguments.values():
                if isinstance(argument, rav2.RobustnessRequestV2):
                    assert "invalid" not in {option.id for option in argument.options}, method
                if isinstance(argument, dict):
                    assert "invalid" not in argument, method
                assert_finite_floats(argument)
            return original(self, *args, **kwargs)

        return guarded

    for method in methods:
        monkeypatch.setattr(rav2.RobustnessAnalyzerV2, method, guard(method))
    body = v2_body(request)
    assert observed == set(methods)
    assert_failed_option(next(option for option in body["options"] if option["id"] == "invalid"))
    assert_finite_floats(body)


def test_refused_status_quo_withholds_identity_and_names_churn():
    request = accumulation_request(rate=120.0)
    rate = next(node for node in request["graph"]["nodes"] if node["id"] == RATE)
    rate["label"] = "Monthly customer churn"
    # Even when every option corrects churn, the refused status quo cannot be
    # replaced by an approximation or a subset of reference draws.
    for option in request["options"]:
        option["interventions"][RATE] = 0.03
    body = blocked_422(request)
    assert body["analysis_status"] == "blocked"
    critique = next(
        row for row in body["critiques"]
        if row["code"] == "IDENTITY_NOT_EVALUATED" and row["identity"]["node_id"] == CARRIER
    )
    assert critique["identity"]["withheld_reason"] == "accumulation_draw_refused"
    assert "churn" in critique["message"].lower()
    assert "rate" in critique["message"].lower()
    assert "withheld rather than approximated" in critique["message"]
    assert_finite_floats(body)


def test_one_refused_status_quo_draw_withholds_the_whole_identity():
    request = accumulation_request()
    request["parameter_uncertainties"] = [
        {"node_id": RATE, "distribution": "uniform", "range_min": 0.02, "range_max": 1.2}
    ]
    # The central churn is valid, so this exercises refusal discovered in the
    # draw sweep rather than only the static identity validation.
    for option in request["options"]:
        option["interventions"][RATE] = 0.04
    body = blocked_422(request)
    critique = next(
        row for row in body["critiques"]
        if row["code"] == "IDENTITY_NOT_EVALUATED" and row["identity"]["node_id"] == CARRIER
    )
    assert body["analysis_status"] == "blocked"
    assert critique["identity"]["withheld_reason"] == "accumulation_draw_refused"
    assert "churn" in critique["message"].lower() or RATE in critique["message"]
    assert "withheld rather than approximated" in critique["message"]
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
    # Round-2 formulae used the complete raw population. Equality must hold
    # exactly on the same seeded draws, avoiding platform-specific libm goldens.
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
    assert_finite_floats(body)


def test_all_accumulation_options_refused_withhold_every_figure():
    request = cashflow_request()
    for option in request["options"]:
        option["interventions"][RATE] = 1.2
    body = v2_body(request)
    assert body["objective_ranking"]["status"] == "withheld"
    assert body.get("robustness") is None
    assert body["robustness_status"] == "skipped"
    assert body.get("recommended_option_id") is None
    assert body.get("recommendation_confidence") is None
    assert body.get("conditional_winners") is None
    for option in body["options"]:
        assert_failed_option(option)
    assert_finite_floats(body)


def test_failed_option_survives_the_worker_boundary_without_nan():
    import json
    from src.services.robustness_worker import decode_analysis_response, run_robustness_v2

    request = accumulation_request()
    request["options"].append({
        "id": "invalid", "label": "Refused monthly churn", "interventions": {RATE: 1.2},
    })
    parsed = rav2.RobustnessRequestV2.model_validate(request)
    response = decode_analysis_response(run_robustness_v2(parsed.model_dump_json()))
    failed = next(row for row in response.results if row.option_id == "invalid")
    assert failed.win_probability is None
    assert failed.outcome_distribution.samples == []
    assert all(getattr(failed.outcome_distribution, key) is None
               for key in ("mean", "std", "median", "ci_lower", "ci_upper"))
    assert failed.constraint_analysis is None and failed.probability_of_goal_drivers is None
    # The real worker codec supports old NaNs; this class introduces none.
    json.dumps(response.model_dump(), allow_nan=False)
    assert_finite_floats(response.model_dump())


def test_sampled_status_quo_blocker_survives_exception_pickling():
    import pickle

    parsed = rav2.RobustnessRequestV2.model_validate(accumulation_request())
    evaluator = rav2.SCMEvaluatorV2(parsed.graph)
    means = {(edge.from_, edge.to): edge.strength.mean for edge in parsed.graph.edges}
    with pytest.raises(rav2.IdentityNotEvaluatedError) as raised:
        evaluator.evaluate(means, {PRICE: 0.245}, GOAL, factor_values={RATE: 1.2})
    refusal = pickle.loads(pickle.dumps(raised.value))
    assert str(refusal) == str(raised.value)
    assert refusal.critiques[0].code == "IDENTITY_NOT_EVALUATED"
    assert refusal.critiques[0].identity.withheld_reason == "accumulation_draw_refused"
    assert "churn/rate" in str(refusal)


@pytest.mark.parametrize("method,field", [
    ("_compute_sensitivity", "robustness.edge_sensitivity"),
    ("_compute_factor_sensitivity", "factor_sensitivity"),
    ("_compute_bootstrap_stability", "factor_sensitivity"),
    ("_compute_evpi", "p_win_sensitivity"),
])
def test_refused_optional_parameter_probe_omits_its_result(monkeypatch, method, field):
    request = cashflow_request()
    request["parameter_uncertainties"] = [
        {"node_id": INFLOW, "distribution": "uniform", "range_min": 0.15, "range_max": 0.25}
    ]
    request["analysis_types"] = ["comparison", "sensitivity", "robustness"]
    request["include_voi"] = True

    def refuse(*args, **kwargs):
        raise rav2.AccumulationDrawRefusedError("monthly churn/rate probe c=1.2")

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, method, refuse)
    body = v2_body(request)
    assert all(option["status"] == "computed" for option in body["options"])
    warning = next(w for w in body["inference_warnings"] if w["field"] == field)
    assert warning["detail"]["reason"] == "accumulation_draw_refused"
    if field.startswith("robustness."):
        assert body["robustness"].get("edge_sensitivity") is None
    else:
        assert body.get(field) is None
    assert_finite_floats(body)


def test_refused_marginal_switch_probe_preserves_mc_robustness(monkeypatch):
    request = cashflow_request()
    request["analysis_types"] = ["comparison", "sensitivity", "robustness"]

    def sensitivity(*args, **kwargs):
        return [rav2.SensitivityResult(
            edge_from=CARRIER, edge_to="cashflow", sensitivity_type="magnitude",
            elasticity=0.5, importance_rank=1, interpretation="Cashflow sensitivity",
        )]

    def refuse(*args, **kwargs):
        raise rav2.AccumulationDrawRefusedError("monthly churn/rate marginal probe c=1.2")

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_compute_sensitivity", sensitivity)
    control = v2_body(copy.deepcopy(request))
    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_compute_marginal_switch_probability", refuse)
    body = v2_body(request)
    assert body["options"] == control["options"]
    assert body["robustness"]["recommendation_stability"] == control["robustness"]["recommendation_stability"]
    assert body["robustness"]["confidence"] == control["robustness"]["confidence"]
    rows = body["robustness"]["fragile_edges"]
    assert rows and all(row.get("marginal_switch_probability") is None for row in rows)
    for actual, reference in zip(rows, control["robustness"]["fragile_edges"]):
        assert actual.get("switch_probability") == reference.get("switch_probability")
    warning = next(w for w in body["inference_warnings"]
                   if w["field"] == "robustness.fragile_edges[].marginal_switch_probability")
    assert warning["detail"]["reason"] == "accumulation_draw_refused"
    assert_finite_floats(body)


def test_refused_status_quo_centre_during_stated_product_setup_is_blocked():
    request = accumulation_request()
    request["parameter_uncertainties"] = [
        {"node_id": RATE, "distribution": "uniform", "range_min": 1.1, "range_max": 1.2}
    ]
    for node_id, level in (("a", 0.1), ("b", 0.2), ("stated_product", 0.02)):
        request["graph"]["nodes"].append({
            "id": node_id, "kind": "factor", "label": node_id,
            "observed_state": {"value": level, "baseline": level, "source": "brief_extraction"},
            "execution_frame": {"frame": 1.0, "carrier": "cap"},
        })
    product = request["graph"]["nodes"][-1]
    product["nonlinear_identity"] = {
        "operation": "product", "factor_ids": ["a", "b"], "stated_in_brief": True,
    }
    for node_id in ("a", "b"):
        request["graph"]["edges"].append({
            "from": node_id, "to": "stated_product", "exists_probability": 1.0,
            "strength": {"mean": 0.5, "std": 0.01},
        })
    body = blocked_422(request)
    critique = next(row for row in body["critiques"] if row["code"] == "IDENTITY_NOT_EVALUATED")
    assert critique["identity"]["node_id"] == CARRIER
    assert critique["identity"]["withheld_reason"] == "accumulation_draw_refused"
    assert "churn/rate" in critique["message"]
    assert_finite_floats(body)
