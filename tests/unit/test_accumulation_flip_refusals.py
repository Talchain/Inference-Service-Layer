"""A refused accumulation probe withholds its existing optional diagnostic block."""

import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import DecisionFlipRequestV2
from src.services import decision_flip as df
from tests.unit.test_accumulation_identity import (
    CARRIER, GOAL, PRICE, RATE, accumulation_request, assert_finite_floats,
)
from tests.unit.test_decision_flip import L1, d1
from tests.unit.test_factor_flip_values import _control_graph
from tests.unit.test_r3_identity_evaluation import v2_body


def test_factor_flip_perturbation_refusal_reuses_unavailable_warning(monkeypatch):
    request = accumulation_request()
    request["include_factor_flips"] = True
    refused_rates = []
    original = rav2.SCMEvaluatorV2.evaluate

    def evaluate(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except (rav2.AccumulationDrawRefusedError, rav2.IdentityNotEvaluatedError):
            refused_rates.append(kwargs.get("factor_values", {}).get(RATE))
            raise

    monkeypatch.setattr(rav2.SCMEvaluatorV2, "evaluate", evaluate)
    body = v2_body(request)
    assert refused_rates and 1.0 in refused_rates  # c=1 at the factor screen's real endpoint
    assert all(option["status"] == "computed" for option in body["options"])
    warning = next(w for w in body["inference_warnings"] if w["code"] == "FACTOR_FLIPS_UNAVAILABLE")
    assert warning["detail"]["reason"] == "accumulation_draw_refused"
    assert body.get("factor_flip_values") is None
    assert_finite_floats(body)


def test_e_value_refusal_reuses_unavailable_warning_without_failed_options(monkeypatch):
    request = accumulation_request()
    request["include_e_values"] = True

    def refuse(*args, **kwargs):
        raise rav2.AccumulationDrawRefusedError("monthly_churn perturbation c=1.2")

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_compute_edge_e_values", refuse)
    body = v2_body(request)
    assert all(option["status"] == "computed" for option in body["options"])
    warning = next(w for w in body["inference_warnings"] if w["code"] == "E_VALUES_UNAVAILABLE")
    assert warning["detail"]["reason"] == "accumulation_draw_refused"
    assert body["robustness"].get("edge_e_values") is None
    assert_finite_floats(body)


def test_stability_refusal_withholds_whole_band_not_one_background(monkeypatch):
    request = accumulation_request()
    next(n for n in request["graph"]["nodes"] if n["id"] == GOAL).pop("nonlinear_identity")
    request["include_e_values"] = True
    calls = []

    def flip(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise rav2.AccumulationDrawRefusedError("monthly_churn background c=1.2")
        return 0.25

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_flip_mean_under_background", flip)
    body = v2_body(request)
    assert len(calls) == 2
    rows = body["robustness"]["edge_e_values"]
    assert rows and all(row.get("stability") is None for row in rows)
    warning = next(w for w in body["inference_warnings"] if w["code"] == "STABILITY_BANDS_UNAVAILABLE")
    assert warning["detail"]["reason"] == "accumulation_draw_refused"
    assert_finite_floats(body)


def test_decision_flip_refused_probe_is_absent_with_finite_json(monkeypatch):
    original = rav2.RobustnessAnalyzerV2.analyze
    refused = []

    def analyze(self, request):
        moved = next(e for e in request.graph.edges if (e.from_, e.to) == L1)
        if moved.strength.mean == 0.0:
            refused.append(True)
            raise rav2.AccumulationDrawRefusedError("monthly_churn probe c=1.2")
        return original(self, request)

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "analyze", analyze)
    block = df.compute_decision_flip_block(DecisionFlipRequestV2.model_validate({
        "request": d1(n=100), "replicates": 2,
        "links": [{"from_id": L1[0], "to_id": L1[1]}],
    }))
    assert refused
    (link,) = block.links
    assert (link.status, link.reason) == ("absent", "accumulation_draw_refused")
    assert link.threshold is None and link.replicate_thresholds is None
    assert_finite_floats(block.model_dump(mode="json"))


def test_decision_flip_failed_option_never_enters_replicate_request(monkeypatch):
    request = accumulation_request()
    request["options"].append({
        "id": "refused", "label": "Refused churn", "interventions": {RATE: 1.2},
    })
    original = rav2.RobustnessAnalyzerV2.analyze
    captured_option_ids = []

    def analyze(self, query):
        if getattr(query, "_capture_draws", False):
            captured_option_ids.append([option.id for option in query.options])
        return original(self, query)

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "analyze", analyze)
    block = df.compute_decision_flip_block(DecisionFlipRequestV2.model_validate({
        "request": request, "replicates": 2,
        "links": [{"from_id": PRICE, "to_id": GOAL}],
    }))
    assert captured_option_ids == [["keep", "raise"], ["keep", "raise"]]
    assert block.leader_option_id == "raise"
    assert_finite_floats(block.model_dump(mode="json"))


@pytest.mark.parametrize("refusal", ["option", "status_quo"])
def test_decision_flip_new_replicate_refusal_is_absent(monkeypatch, refusal):
    request = accumulation_request()
    original = rav2.RobustnessAnalyzerV2.analyze
    refused = []

    def analyze(self, query):
        if not getattr(query, "_capture_draws", False):
            return original(self, query)
        refused.append(True)
        if refusal == "status_quo":
            raise rav2.IdentityNotEvaluatedError(
                "monthly_churn accumulation refused",
                [rav2.accumulation_refusal_critique(query.graph, CARRIER)],
            )
        response = original(self, query)
        response._mc_draws["accumulation_refused_option_ids"] = ["keep"]
        response._mc_draws["option_outcomes"].pop("keep")
        return response

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "analyze", analyze)
    block = df.compute_decision_flip_block(DecisionFlipRequestV2.model_validate({
        "request": request, "replicates": 2,
        "links": [{"from_id": PRICE, "to_id": GOAL}],
    }))
    assert refused == [True]
    (link,) = block.links
    assert (link.status, link.reason) == ("absent", "accumulation_draw_refused")
    assert link.threshold is None and link.replicate_thresholds is None
    assert_finite_floats(block.model_dump(mode="json"))


def test_legacy_flip_control_remains_exact():
    request = _control_graph()
    rows = rav2.RobustnessAnalyzerV2()._compute_factor_flip_values(
        request, rav2.SCMEvaluatorV2(request.graph), 4242
    )
    lever = next(row for row in rows if row["factor_id"] == "fac_lever")
    assert lever["flip_value"] == 0.8
    assert lever["baseline_winner_id"] == "opt_a"
    assert lever["alternative_winner_id"] == "opt_c"
    assert lever["direction"] == "increase"
