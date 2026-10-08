"""P45 accumulation contract 8 Oct: closed-form stock carrier and horizon anchor."""

from __future__ import annotations

import copy
import math

from typing import Any, Dict

import pytest

from pydantic import ValidationError

import src.services.robustness_analyzer_v2 as rav2

from src.models.robustness_v2 import NonlinearIdentityV2, RobustnessRequestV2
from src.services import decision_flip
from tests.unit.test_r3_identity_evaluation import (
    AT_59_SUBS_HELD,
    query_gbp,
    tally_effects,
    tally_request,
    v2_body,
    wire,
)


def test_product_and_sum_head_float_parity():
    """Values recorded by this node at staging HEAD 82842bb8, before any source edits."""
    result = (query_gbp(wire(), AT_59_SUBS_HELD), tally_effects(tally_request(None)))
    assert result == (
        15102.040816326546,
        {"features": 20000.0, "advertising": 20000.0, "carry_on": 0.0},
    )


STOCK, RATE, INFLOW, CARRIER, PRICE, GOAL = (
    "stock_today",
    "monthly_churn",
    "monthly_inflow",
    "stock_at_horizon",
    "price",
    "mrr",
)


def accumulation_request(*, horizon: int = 12, rate: float = 3.0) -> Dict[str, Any]:
    """Current stock is deliberately stated on the carrier; its horizon value is different."""
    frames = {
        STOCK: 1000.0,
        RATE: 100.0,
        INFLOW: 100.0,
        CARRIER: 1000.0,
        PRICE: 200.0,
        GOAL: 100000.0,
    }
    levels = {STOCK: 250.0, RATE: rate, INFLOW: 20.0, CARRIER: 250.0, PRICE: 49.0, GOAL: 12250.0}
    nodes = [
        {
            "id": node_id,
            "kind": "outcome" if node_id == GOAL else "factor",
            "label": node_id,
            "observed_state": {
                "value": levels[node_id] / frames[node_id],
                "baseline": levels[node_id] / frames[node_id],
                "source": "brief_extraction",
            },
            "execution_frame": {"frame": frames[node_id], "carrier": "cap"},
        }
        for node_id in frames
    ]
    by_id = {n["id"]: n for n in nodes}
    by_id[CARRIER]["nonlinear_identity"] = {
        "operation": "accumulation",
        "factor_ids": [STOCK, RATE, INFLOW],
        "horizon_months": horizon,
        "rate_scale": 0.01,
        "stated_in_brief": False,
    }
    by_id[GOAL]["nonlinear_identity"] = {
        "operation": "product",
        "factor_ids": [PRICE, CARRIER],
        "stated_in_brief": True,
    }
    return {
        "request_id": "p45-accumulation",
        "graph": {
            "nodes": nodes,
            "edges": [
                {
                    "from": source,
                    "to": target,
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.5, "std": 0.01},
                }
                for source, target in [
                    (STOCK, CARRIER),
                    (RATE, CARRIER),
                    (INFLOW, CARRIER),
                    (PRICE, GOAL),
                    (CARRIER, GOAL),
                ]
            ],
        },
        "options": [
            {"id": "keep", "label": "Keep price", "interventions": {PRICE: 49.0 / 200.0}},
            {"id": "raise", "label": "Raise price", "interventions": {PRICE: 59.0 / 200.0}},
        ],
        "goal_node_id": GOAL,
        "goal_threshold": 0.18,
        "goal_threshold_frame": "level",
        "n_samples": 100,
        "seed": 7,
        "include_e_values": False,
        "include_voi": False,
        "include_factor_flips": False,
        "include_path_decomposition": False,
        "analysis_types": ["comparison"],
    }


def stepped(stock: float, churn: float, inflow: float, horizon: int) -> float:
    """Independent time-stepped oracle, never calling production accumulation arithmetic."""
    for _ in range(horizon):
        stock = stock * (1.0 - churn) + inflow
    return stock


def term(stock: float, rate: float, inflow: float, horizon: int = 12, scale: float = 1.0) -> float:
    return rav2._identity_term(
        "accumulation", [stock, rate, inflow], horizon_months=horizon, rate_scale=scale
    )


def evaluator(d: Dict[str, Any]):
    graph = RobustnessRequestV2.model_validate(d).graph
    edges = {(e.from_, e.to): e.strength.mean for e in graph.edges}
    roots = {
        n.id: n.observed_state.value for n in graph.nodes if n.id in (STOCK, RATE, INFLOW, PRICE)
    }
    return rav2.SCMEvaluatorV2(graph), edges, roots


def assert_finite_floats(value: Any) -> None:
    if isinstance(value, float):
        assert math.isfinite(value), value
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite_floats(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_finite_floats(child)


def test_closed_form_matches_time_stepped_oracle():
    assert term(250.0, 0.03, 20.0) == pytest.approx(377.6, abs=0.05)
    for stock, churn, inflow, horizon in (
        (250.0, 0.03, 20.0, 12),
        (17.0, 0.5, 4.0, 120),
        (1234.0, 0.001, 9.5, 37),
        (0.0, 0.9, 3.0, 1),
    ):
        assert term(stock, churn, inflow, horizon) == pytest.approx(
            stepped(stock, churn, inflow, horizon), rel=1e-9
        )
    assert term(250.0, 3.0, 20.0, scale=0.01) == term(250.0, 0.03, 20.0)


def test_zero_rate_limit_is_continuous():
    expected = 250.0 + 20.0 * 12
    assert term(250.0, 0.0, 20.0) == expected
    assert term(250.0, 1e-12, 20.0) == pytest.approx(expected, abs=1e-9)
    at_small_rate = term(250.0, 1e-6, 20.0)
    assert abs(at_small_rate - expected) < 0.01
    assert at_small_rate == pytest.approx(stepped(250.0, 1e-6, 20.0, 12), rel=1e-9)


def test_out_of_range_rate_draw_is_uninformative_and_wire_is_finite():
    for rate in (1.0, 1.2, -0.01):
        with pytest.raises(ValueError, match="accumulation"):
            term(250.0, rate, 20.0)
        d = accumulation_request()
        d["options"].append(
            {"id": "invalid", "label": "Invalid rate draw", "interventions": {RATE: rate}}
        )
        body = v2_body(d)
        invalid = next(o for o in body["options"] if o["id"] == "invalid")
        assert invalid["status"] == "failed"
        assert invalid["win_probability"] == 0.0
        assert_finite_floats(body)


def test_baseline_is_status_quos_own_horizon_stock():
    d = accumulation_request()
    request = RobustnessRequestV2.model_validate(d)
    goal_plan, warning = rav2.RobustnessAnalyzerV2._resolve_goal_threshold_in_sample_frame(request)
    expected = 49.0 * stepped(250.0, 0.03, 20.0, 12) / 100000.0
    assert goal_plan is not None
    assert goal_plan.goal_baseline == pytest.approx(expected, rel=1e-12)
    assert goal_plan.goal_baseline != 49.0 * 250.0 / 100000.0
    body = v2_body(d)
    keep = next(o for o in body["options"] if o["id"] == "keep")
    assert keep["outcome"]["mean"] == pytest.approx(expected, rel=1e-12)
    assert keep["probability_of_goal"] == 1.0


def test_partials_2037_match_central_finite_difference():
    graph = RobustnessRequestV2.model_validate(accumulation_request()).graph
    partials = rav2.identity_partials(graph)
    h = 1e-5
    d_churn = (stepped(250.0, 0.03 + h, 20.0, 12) - stepped(250.0, 0.03 - h, 20.0, 12)) / (2.0 * h)
    d_inflow = (stepped(250.0, 0.03, 20.0 + h, 12) - stepped(250.0, 0.03, 20.0 - h, 12)) / (2.0 * h)
    assert partials[(RATE, CARRIER)] == pytest.approx(d_churn * 100.0 * 0.01 / 1000.0, rel=1e-5)
    assert partials[(INFLOW, CARRIER)] == pytest.approx(d_inflow * 100.0 / 1000.0, rel=1e-5)
    assert partials[(STOCK, CARRIER)] == pytest.approx(0.97**12, rel=1e-12)
    assert partials[(RATE, CARRIER)] != 1.0


def test_identity_term_1721_never_falls_through_to_sum():
    expected = stepped(250.0, 0.03, 20.0, 12)
    assert term(250.0, 0.03, 20.0) == pytest.approx(expected, rel=1e-12)
    assert term(250.0, 0.03, 20.0) != math.fsum([250.0, 0.03, 20.0])
    with pytest.raises(ValueError):
        rav2._identity_term("next_operation", [2.0, 3.0])


def test_reconciliation_1811_skips_five_percent_check():
    d = accumulation_request()
    carrier = next(n for n in d["graph"]["nodes"] if n["id"] == CARRIER)
    carrier["observed_state"].update(value=0.0, baseline=0.0)
    goal = next(n for n in d["graph"]["nodes"] if n["id"] == GOAL)
    goal.pop("nonlinear_identity")  # isolate the accumulation's check
    plan = rav2.resolve_identity_plans(RobustnessRequestV2.model_validate(d).graph)[CARRIER]
    assert plan.evaluated and plan.withheld_reason is None
    assert plan.mismatch_share is None
    ev, edges, roots = evaluator(d)
    assert ev.evaluate(edges, {}, CARRIER, factor_values=roots) * 1000.0 == pytest.approx(
        stepped(250.0, 0.03, 20.0, 12), rel=1e-12
    )


def test_identity_value_2643_evaluates_carrier_in_user_units():
    ev, edges, roots = evaluator(accumulation_request())
    expected = stepped(250.0, 0.03, 20.0, 12)
    value = ev.evaluate(edges, {}, CARRIER, factor_values=roots) * 1000.0
    assert value == pytest.approx(expected, rel=1e-12)
    assert value != math.fsum([250.0, 3.0, 20.0])
    moved = ev.evaluate(edges, {INFLOW: 0.3}, CARRIER, factor_values=roots) * 1000.0
    assert moved == pytest.approx(stepped(250.0, 0.03, 30.0, 12), rel=1e-12)


def test_decision_flip_146_explicitly_refuses_accumulation_affine_path(monkeypatch):
    d = accumulation_request()
    next(n for n in d["graph"]["nodes"] if n["id"] == GOAL).pop("nonlinear_identity")
    request = RobustnessRequestV2.model_validate(d)
    assert decision_flip.downstream_nonlinearity(request, STOCK, CARRIER) == f"identity:{CARRIER}"

    class UnknownPlan:
        evaluated, operation = True, "future_operation"

    monkeypatch.setattr(
        rav2, "_resolve_structural_identity_plans", lambda graph: {CARRIER: UnknownPlan()}
    )
    with pytest.raises(ValueError):
        decision_flip.downstream_nonlinearity(request, STOCK, CARRIER)


def test_strict_accumulation_carrier_refusals():
    original = accumulation_request()
    RobustnessRequestV2.model_validate(original)  # accepted carrier is prerequisite to the refusals
    cases = [
        {"factor_ids": [STOCK, RATE]},
        {"factor_ids": [STOCK, RATE, INFLOW, PRICE]},
        {"factor_ids": [STOCK, RATE, RATE]},
        {"factor_ids": [STOCK, RATE, PRICE]},
        {"addends": [PRICE]},
        {"addends": None},
        {"unknown": 12},
        {"horizon_months": 0},
        {"horizon_months": 121},
        {"horizon_months": 1.2},
        {"horizon_months": 12.0},
        {"horizon_months": True},
        {"rate_scale": 0},
        {"rate_scale": 1.5},
    ]
    for patch in cases:
        d = copy.deepcopy(original)
        carrier = next(n for n in d["graph"]["nodes"] if n["id"] == CARRIER)
        carrier["nonlinear_identity"].update(patch)
        with pytest.raises(ValidationError):
            RobustnessRequestV2.model_validate(d)
    for missing in ("horizon_months", "rate_scale"):
        d = copy.deepcopy(original)
        next(n for n in d["graph"]["nodes"] if n["id"] == CARRIER)["nonlinear_identity"].pop(
            missing
        )
        with pytest.raises(ValidationError):
            RobustnessRequestV2.model_validate(d)
    for operation in ("product", "sum"):
        for forbidden in ({"horizon_months": 12}, {"horizon_months": None}, {"rate_scale": None}):
            d = copy.deepcopy(original)
            goal = next(n for n in d["graph"]["nodes"] if n["id"] == GOAL)
            goal["nonlinear_identity"].update(operation=operation, **forbidden)
            with pytest.raises(ValidationError):
                RobustnessRequestV2.model_validate(d)
    schema = NonlinearIdentityV2.model_json_schema()
    assert schema["additionalProperties"] is False
    conditional = schema["allOf"][0]
    assert conditional["if"]["properties"]["operation"]["const"] == "accumulation"
    assert conditional["then"]["required"] == ["horizon_months", "rate_scale"]
    assert conditional["then"]["properties"]["factor_ids"] == {
        "minItems": 3,
        "maxItems": 3,
        "uniqueItems": True,
    }
    assert conditional["then"]["properties"]["horizon_months"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 120,
    }
    assert conditional["then"]["properties"]["rate_scale"] == {
        "type": "number",
        "exclusiveMinimum": 0,
        "maximum": 1,
    }
    assert conditional["then"]["not"] == {"required": ["addends"]}
    assert conditional["else"]["not"]["anyOf"] == [
        {"required": ["horizon_months"]},
        {"required": ["rate_scale"]},
    ]


def test_carrier_contract_survives_worker_json_round_trip():
    for d in (accumulation_request(), wire(), tally_request(None)):
        parsed = RobustnessRequestV2.model_validate(d)
        reparsed = RobustnessRequestV2.model_validate_json(parsed.model_dump_json())
        for node in reparsed.graph.nodes:
            identity = node.nonlinear_identity
            if identity is None:
                continue
            serialized = identity.model_dump()
            if identity.operation == "accumulation":
                assert serialized["factor_ids"] == [STOCK, RATE, INFLOW]
                assert serialized["horizon_months"] == 12 and serialized["rate_scale"] == 0.01
                assert "addends" not in serialized
            else:
                assert "horizon_months" not in serialized and "rate_scale" not in serialized


def test_accumulation_carrier_cannot_be_the_goal():
    d = accumulation_request()
    d["goal_node_id"] = CARRIER
    with pytest.raises(ValidationError, match="accumulation"):
        RobustnessRequestV2.model_validate(d)


def test_non_finite_normalised_stock_withholds_identity_without_float_leaks():
    d = accumulation_request()
    next(n for n in d["graph"]["nodes"] if n["id"] == CARRIER)["execution_frame"]["frame"] = 1e-320
    request = RobustnessRequestV2.model_validate(d)
    plans = rav2.resolve_identity_plans(request.graph)
    assert plans[CARRIER].withheld_reason == "identity_non_finite"
    evaluations = [e.model_dump() for e in rav2.identity_evaluations(request.graph)]
    assert_finite_floats(evaluations)
    critiques = rav2.identity_blocking_critiques(request)
    assert critiques
    assert any(c.identity.withheld_reason == "identity_non_finite" for c in critiques)
    assert_finite_floats([c.model_dump() for c in critiques])


def test_identity_evaluations_discloses_accumulation_horizon_only():
    body = v2_body(accumulation_request())
    carrier = next(e for e in body["identity_evaluations"] if e["node_id"] == CARRIER)
    assert carrier["operation"] == "accumulation" and carrier["evaluated"] is True
    assert carrier["horizon_months"] == 12
    product = next(e for e in body["identity_evaluations"] if e["node_id"] == GOAL)
    assert product.get("horizon_months") is None
    sum_request = tally_request(None)
    sum_request["options"][-1]["interventions"] = {"features_spend": 0.0}
    sum_entry = v2_body(sum_request)["identity_evaluations"][0]
    assert sum_entry["operation"] == "sum" and sum_entry.get("horizon_months") is None


def test_horizon_goal_level_is_not_clamped_to_todays_stock_domain(monkeypatch):
    d = accumulation_request()
    original = rav2.RobustnessAnalyzerV2._compute_option_results

    def with_today_domain(self, *args, **kwargs):
        kwargs["level_domains"] = {GOAL: (0.0, 0.13)}
        return original(self, *args, **kwargs)

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_compute_option_results", with_today_domain)
    d["goal_threshold"] = 0.18
    body = v2_body(d)
    keep = next(o for o in body["options"] if o["id"] == "keep")
    assert keep["probability_of_goal"] == 1.0
    assert keep["outcome"]["mean"] > 0.18
