"""R-P2-1: exact S0 cuts upstream influence at its accumulation carrier."""

from typing import Any, Dict, Optional, Tuple

import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import RobustnessRequestV2
from tests.unit.test_accumulation_identity import CARRIER, GOAL, INFLOW, PRICE, RATE, STOCK
from tests.unit.test_accumulation_rate_spread import (
    b1_request, sampled_response, stable_wire, two_distinct_carriers,
)
from tests.unit.test_r3_identity_evaluation import v2_body, wire


UPSTREAM = "upstream"
SIGMAS = [None, (0.136, 0.136)]


def stock_cut_request(sigma: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    """The reviewer's request, without an additional route around exact stock."""
    raw = b1_request(sigma=sigma, n_samples=100)
    raw["analysis_types"] = ["comparison", "sensitivity"]
    raw["graph"]["nodes"].append({
        "id": UPSTREAM, "kind": "factor", "label": UPSTREAM,
        "observed_state": {"value": 0.5, "baseline": 0.5},
    })
    raw["graph"]["edges"].append({
        "from": UPSTREAM, "to": STOCK, "exists_probability": 1.0,
        "strength": {"mean": 0.1, "std": 0.01},
    })
    raw["parameter_uncertainties"] = [
        {"node_id": UPSTREAM, "distribution": "normal", "std": 0.1}
    ]
    return raw


def parts(raw):
    request = RobustnessRequestV2.model_validate(raw)
    evaluator = rav2.SCMEvaluatorV2(request.graph, factor_centres=rav2.factor_centres(request))
    edges = {(edge.from_, edge.to): edge.strength.mean for edge in request.graph.edges}
    return request, evaluator, edges


def probe(request, evaluator, edges, node_id, value):
    return evaluator.evaluate(
        edges, request.options[0].interventions, request.goal_node_id,
        factor_values={node_id: value}, factor_level_overrides={node_id: value},
    )


def raw_influence(request, node_id, goal_node_id=None):
    raw = {}
    _, truncated = rav2.RobustnessAnalyzerV2()._compute_structural_influence(
        request.graph, [node_id], goal_node_id or request.goal_node_id,
        factor_centres=rav2.factor_centres(request), raw_out=raw,
    )
    assert truncated == []
    return raw[node_id]


def disable_stock_cuts(monkeypatch):
    monkeypatch.setattr(rav2, "accumulation_stock_cuts", lambda graph, factor_centres=None: set())
    monkeypatch.setattr(
        rav2, "accumulation_stock_cut_quantities",
        lambda graph, goal_node_id, factor_centres=None: set(),
    )


@pytest.mark.parametrize("sigma", SIGMAS)
def test_reviewer_upstream_is_zero_in_both_served_influence_cohorts(sigma):
    body = v2_body(stock_cut_request(sigma))
    structural = next(row for row in body["structural_influence"] if row["node_id"] == UPSTREAM)
    sensitivity = next(row for row in body["factor_sensitivity"] if row["node_id"] == UPSTREAM)
    assert structural["influence_score"] == 0.0
    assert sensitivity["influence_score"] == sensitivity["elasticity"] == 0.0
    # ISL retains a zero-influence factor row, and diagnoses its disconnected
    # effective path rather than calling it a driver with influence one.
    assert sensitivity["zero_reason"] == "disconnected"
    assert any(row["influence_score"] > structural["influence_score"]
               for row in body["structural_influence"])
    assert structural["influence_rank"] > 1
    for option in body["options"]:
        drivers = option["probability_of_goal_drivers"]
        assert drivers["drivers"] == []
        assert drivers["n_candidates"] == 0
    blocked_edge_rows = [
        row for row in body["robustness"]["edge_sensitivity"]
        if (row["from_id"], row["to_id"]) == (UPSTREAM, STOCK)
    ]
    assert blocked_edge_rows
    assert all(row["elasticity"] == 0.0 for row in blocked_edge_rows)


@pytest.mark.parametrize("sigma", SIGMAS)
def test_reviewer_upstream_probes_remain_exactly_equal(sigma):
    request, evaluator, edges = parts(stock_cut_request(sigma))
    outcomes = [probe(request, evaluator, edges, UPSTREAM, value) for value in (0.0, 0.5, 1.0)]
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("sigma", SIGMAS)
def test_direct_upstream_route_keeps_exactly_its_own_influence(sigma):
    raw = stock_cut_request(sigma)
    raw["graph"]["edges"].append({
        "from": UPSTREAM, "to": GOAL, "exists_probability": 1.0,
        "strength": {"mean": 0.01, "std": 0.002},
    })
    request, evaluator, edges = parts(raw)
    direct = request.graph.edges[-1]
    expected = direct.strength.mean * direct.exists_probability
    assert raw_influence(request, UPSTREAM) == pytest.approx(expected, rel=1e-12)
    assert probe(request, evaluator, edges, UPSTREAM, 1.0) != probe(
        request, evaluator, edges, UPSTREAM, 0.0
    )
    body = v2_body(raw)
    structural = next(row for row in body["structural_influence"] if row["node_id"] == UPSTREAM)
    sensitivity = next(row for row in body["factor_sensitivity"] if row["node_id"] == UPSTREAM)
    assert structural["influence_score"] > 0.0
    assert sensitivity["influence_score"] > 0.0
    assert sensitivity["elasticity"] != 0.0
    assert all(option["probability_of_goal_drivers"]["n_candidates"] > 0
               for option in body["options"])
    edge_rows = body["robustness"]["edge_sensitivity"]
    blocked_rows = [row for row in edge_rows if (row["from_id"], row["to_id"]) == (UPSTREAM, STOCK)]
    direct_rows = [row for row in edge_rows if (row["from_id"], row["to_id"]) == (UPSTREAM, GOAL)]
    assert blocked_rows and all(row["elasticity"] == 0.0 for row in blocked_rows)
    assert direct_rows and any(row["elasticity"] != 0.0 for row in direct_rows)


def test_edge_stock_cut_keeps_live_edge_rows_and_rng_stream_exactly_unchanged(monkeypatch):
    raw = stock_cut_request((0.136, 0.136))
    raw["graph"]["edges"].append({
        "from": UPSTREAM, "to": GOAL, "exists_probability": 1.0,
        "strength": {"mean": 0.2, "std": 0.002},
    })
    request, evaluator, edges = parts(raw)
    baseline_value = evaluator.evaluate(edges, request.options[0].interventions, GOAL)
    baseline = {option.id: [baseline_value] for option in request.options}
    enabled_rng = rav2.SeededRNG(101)
    enabled_rows = rav2.RobustnessAnalyzerV2()._compute_sensitivity(
        request, baseline, rav2.DualUncertaintySampler(request.graph.edges, enabled_rng),
        enabled_rng, evaluator,
    )
    disable_stock_cuts(monkeypatch)
    _, control_evaluator, _ = parts(raw)
    control_rng = rav2.SeededRNG(101)
    control_rows = rav2.RobustnessAnalyzerV2()._compute_sensitivity(
        request, baseline, rav2.DualUncertaintySampler(request.graph.edges, control_rng),
        control_rng, control_evaluator,
    )
    blocked = [row for row in enabled_rows if (row.edge_from, row.edge_to) == (UPSTREAM, STOCK)]
    original_blocked = [
        row for row in control_rows if (row.edge_from, row.edge_to) == (UPSTREAM, STOCK)
    ]
    assert len(blocked) == len(original_blocked) == 2
    assert all(row.elasticity == 0.0 for row in blocked)
    # Independent draws of the live edge in on/off backgrounds previously
    # gave the frozen edge a spurious contrast. Its stream still gets consumed.
    assert any(row.elasticity != 0.0 for row in original_blocked)
    live = lambda rows: {
        (row.edge_from, row.edge_to, row.sensitivity_type): row.model_dump()
        for row in rows if (row.edge_from, row.edge_to) != (UPSTREAM, STOCK)
    }
    assert live(enabled_rows) == live(control_rows)
    assert enabled_rng.random() == control_rng.random()


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("bypass_target", [GOAL, INFLOW])
def test_stock_cut_preserves_raw_stock_bypass_and_another_operand_route(sigma, bypass_target):
    raw = stock_cut_request(sigma)
    source = STOCK if bypass_target == GOAL else UPSTREAM
    raw["graph"]["edges"].append({
        "from": source, "to": bypass_target, "exists_probability": 1.0,
        "strength": {"mean": 0.02, "std": 0.002},
    })
    request, evaluator, edges = parts(raw)
    if bypass_target == GOAL:
        high = probe(request, evaluator, edges, UPSTREAM, 0.6)
        low = probe(request, evaluator, edges, UPSTREAM, 0.4)
        expected = abs((high - low) / (0.6 - 0.4))
    else:
        # Inflow is non-root here; its stated centre anchors the one-at-a-time
        # upstream probe. The raw structural route still carries these partials.
        partials = rav2.identity_partials(request.graph, rav2.factor_centres(request))
        expected = abs(
            edges[(UPSTREAM, INFLOW)] * partials[(INFLOW, CARRIER)] * partials[(CARRIER, GOAL)]
        )
    assert expected > 0.0
    assert raw_influence(request, UPSTREAM) == pytest.approx(expected, rel=1e-12)
    cut_quantities = rav2.accumulation_stock_cut_quantities(
        request.graph, request.goal_node_id, rav2.factor_centres(request)
    )
    assert UPSTREAM not in cut_quantities
    # The stock edge itself is still productive when stock has a raw bypass.
    if bypass_target == GOAL:
        assert f"{UPSTREAM}->{STOCK}" not in cut_quantities


@pytest.mark.parametrize("sigma", SIGMAS)
def test_deliberate_stock_options_move_and_factor_probes_are_exactly_unchanged(sigma, monkeypatch):
    raw = stock_cut_request(sigma)
    raw["parameter_uncertainties"] = [
        {"node_id": STOCK, "distribution": "normal", "std": 0.1}
    ]
    request, evaluator, edges = parts(raw)
    probes = [probe(request, evaluator, edges, STOCK, value) for value in (0.2, 0.3)]
    low = evaluator.evaluate(edges, {PRICE: 49.0 / 200.0, STOCK: 0.2}, GOAL)
    high = evaluator.evaluate(edges, {PRICE: 49.0 / 200.0, STOCK: 0.3}, GOAL)
    assert high > low
    if sigma is not None:
        assert probes == [low, high]
    baseline = probe(request, evaluator, edges, STOCK, 0.25)
    sensitivities = rav2.RobustnessAnalyzerV2()._compute_factor_sensitivity(
        request, {option.id: [baseline] for option in request.options},
        rav2.SeededRNG(101), evaluator,
    )
    if sigma is not None:
        assert next(row for row in sensitivities if row.node_id == STOCK).elasticity > 0.0
    disable_stock_cuts(monkeypatch)
    _, control_evaluator, control_edges = parts(raw)
    assert [probe(request, control_evaluator, control_edges, STOCK, value)
            for value in (0.2, 0.3)] == probes
    control_sensitivities = rav2.RobustnessAnalyzerV2()._compute_factor_sensitivity(
        request, {option.id: [baseline] for option in request.options},
        rav2.SeededRNG(101), control_evaluator,
    )
    assert [row.model_dump() for row in control_sensitivities] == [
        row.model_dump() for row in sensitivities
    ]


@pytest.mark.parametrize("sigma", SIGMAS)
def test_deliberate_stock_factor_flip_probes_are_exactly_unchanged(sigma, monkeypatch):
    # Factor flips deliberately target roots. Stock is a root in this control,
    # while the reviewer request above separately checks its upstream cut.
    raw = b1_request(sigma=sigma, n_samples=100)
    raw.update(include_factor_flips=True, goal_direction="maximise")
    raw["options"] = [
        {"id": "A", "label": "A", "interventions": {PRICE: 59.0 / 200.0, INFLOW: 0.2}},
        {"id": "B", "label": "B", "interventions": {PRICE: 49.0 / 200.0, INFLOW: 0.3}},
    ]
    raw["parameter_uncertainties"] = [
        {"node_id": STOCK, "distribution": "normal", "std": 0.1}
    ]
    response = sampled_response(raw)
    if sigma is not None:
        flip = next(row for row in response.factor_flip_values if row["factor_id"] == STOCK)
        assert flip["flip_reason"] == "found"
        assert flip["flip_value"] is not None
        request, evaluator, edges = parts(raw)
        analyzer = rav2.RobustnessAnalyzerV2()
        crossing = flip["flip_value"]
        below = analyzer._option_goals(request, evaluator, edges, {STOCK: crossing - 0.01})
        above = analyzer._option_goals(request, evaluator, edges, {STOCK: crossing + 0.01})
        assert analyzer._argmax_option(below) != analyzer._argmax_option(above)
    disable_stock_cuts(monkeypatch)
    assert sampled_response(raw).factor_flip_values == response.factor_flip_values


@pytest.mark.parametrize("sigma", SIGMAS)
def test_path_decomposition_cuts_upstream_but_preserves_stock_as_entry(sigma):
    raw = stock_cut_request(sigma)
    raw["options"] = [
        {"id": "upstream_probe", "label": "Upstream", "interventions": {UPSTREAM: 0.6}},
        {"id": "stock_probe", "label": "Stock", "interventions": {STOCK: 0.3}},
    ]
    request, evaluator, _ = parts(raw)
    analyzer = rav2.RobustnessAnalyzerV2()
    upstream = analyzer._compute_path_decomposition(request, "upstream_probe", evaluator.graph)
    stock = analyzer._compute_path_decomposition(request, "stock_probe", evaluator.graph)
    assert upstream is not None and stock is not None
    assert upstream.entry_nodes == [UPSTREAM]
    assert upstream.paths == [] and upstream.path_count == 0
    assert stock.entry_nodes == [STOCK]
    assert stock.path_count > 0 and stock.paths


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("bridge_operation", ["sum", "product"])
def test_cut_only_upstream_is_neither_anchored_blind_nor_zero_gated(sigma, bridge_operation):
    raw = stock_cut_request(sigma)
    upstream = next(node for node in raw["graph"]["nodes"] if node["id"] == UPSTREAM)
    upstream["execution_frame"] = {"frame": 1.0, "carrier": "cap"}
    bridge = {
        "id": "bridge", "kind": "factor", "label": "Bridge",
        "execution_frame": {"frame": 1.0, "carrier": "cap"},
        "nonlinear_identity": {
            "operation": bridge_operation, "factor_ids": [UPSTREAM], "stated_in_brief": False,
        },
    }
    if bridge_operation == "sum":
        bridge["observed_state"] = {"value": 0.5, "baseline": 0.5}
    else:
        # An unstated product with a zero operand is evaluated and would gate
        # upstream if the frozen stock boundary were still treated as a path.
        bridge["nonlinear_identity"]["factor_ids"].append("zero")
        raw["graph"]["nodes"].append({
            "id": "zero", "kind": "factor", "label": "Zero",
            "observed_state": {"value": 0.0, "baseline": 0.0},
            "execution_frame": {"frame": 1.0, "carrier": "cap"},
        })
    raw["graph"]["nodes"].append(bridge)
    raw["graph"]["edges"][-1]["from"] = "bridge"
    raw["graph"]["edges"].append({
        "from": UPSTREAM, "to": "bridge", "exists_probability": 1.0,
        "strength": {"mean": 0.5, "std": 0.01},
    })
    if bridge_operation == "product":
        raw["graph"]["edges"].append({
            "from": "zero", "to": "bridge", "exists_probability": 1.0,
            "strength": {"mean": 0.5, "std": 0.01},
        })
    request, _, _ = parts(raw)
    centres = rav2.factor_centres(request)
    assert rav2.resolve_identity_plans(request.graph, centres)["bridge"].evaluated
    assert raw_influence(request, UPSTREAM) == 0.0
    assert rav2.anchored_blind_factor_ids(request.graph, [UPSTREAM], GOAL, centres) == {}
    assert rav2.zero_gated_factor_ids(request.graph, [UPSTREAM], GOAL, centres) == {}


def test_another_carriers_stock_remains_a_raw_rate_driver_for_this_carrier():
    request, _, edges = parts(two_distinct_carriers("stock_b"))
    assert raw_influence(request, PRICE, "carrier_b") == 0.0
    # PRICE -> B stock -> A churn is a raw path even though B reads the same
    # stock exactly. A global stock-node cut would erase this real route.
    partial = rav2.identity_partials(request.graph, rav2.factor_centres(request))[(RATE, CARRIER)]
    expected = abs(edges[(PRICE, "stock_b")] * edges[("stock_b", RATE)] * partial)
    assert expected > 0.0
    assert raw_influence(request, PRICE, CARRIER) == pytest.approx(expected, rel=1e-12)
    assert PRICE not in rav2.accumulation_stock_cut_quantities(
        request.graph, CARRIER, rav2.factor_centres(request)
    )


def test_same_carrier_rate_route_cannot_reenter_through_its_frozen_stock():
    raw = stock_cut_request((0.136, 0.136))
    raw["graph"]["edges"].append({
        "from": STOCK, "to": RATE, "exists_probability": 1.0,
        "strength": {"mean": 0.02, "std": 0.01},
    })
    request, evaluator, edges = parts(raw)
    values = [probe(request, evaluator, edges, UPSTREAM, value) for value in (0.0, 0.5, 1.0)]
    assert values[0] == values[1] == values[2]
    assert raw_influence(request, UPSTREAM) == 0.0
    cut_quantities = rav2.accumulation_stock_cut_quantities(
        request.graph, request.goal_node_id, rav2.factor_centres(request)
    )
    assert UPSTREAM in cut_quantities
    assert f"{UPSTREAM}->{STOCK}" in cut_quantities


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("operation", ["sum", "product"])
def test_cut_route_does_not_supply_anchor_or_gate_provenance(sigma, operation):
    raw = stock_cut_request(sigma)
    upstream = next(node for node in raw["graph"]["nodes"] if node["id"] == UPSTREAM)
    upstream["execution_frame"] = {"frame": 1.0, "carrier": "cap"}
    stock = next(node for node in raw["graph"]["nodes"] if node["id"] == STOCK)
    for node_id, operand_id, frame, centre in (
        ("p", STOCK, stock["execution_frame"]["frame"], stock["observed_state"]["value"]),
        ("q", UPSTREAM, 1.0, upstream["observed_state"]["value"]),
    ):
        participant_ids = [operand_id]
        node = {
            "id": node_id, "kind": "factor", "label": node_id,
            "execution_frame": {"frame": frame, "carrier": "cap"},
            "nonlinear_identity": {
                "operation": operation, "factor_ids": participant_ids, "stated_in_brief": True,
            },
        }
        if operation == "sum":
            node["observed_state"] = {"value": centre, "baseline": centre}
        else:
            zero_id = f"zero_{node_id}"
            participant_ids.append(zero_id)
            raw["graph"]["nodes"].append({
                "id": zero_id, "kind": "factor", "label": zero_id,
                "observed_state": {"value": 0.0, "baseline": 0.0},
                "execution_frame": {"frame": 1.0, "carrier": "cap"},
            })
        raw["graph"]["nodes"].append(node)
        for participant_id in participant_ids:
            raw["graph"]["edges"].append({
                "from": participant_id, "to": node_id, "exists_probability": 1.0,
                "strength": {"mean": 1.0, "std": 0.01},
            })
    raw["graph"]["edges"].extend([
        {"from": "p", "to": RATE, "exists_probability": 1.0,
         "strength": {"mean": 0.01, "std": 0.01}},
        {"from": "q", "to": GOAL, "exists_probability": 1.0,
         "strength": {"mean": 0.01, "std": 0.01}},
    ])
    request, _, _ = parts(raw)
    centres = rav2.factor_centres(request)
    plans = rav2.resolve_identity_plans(request.graph, centres)
    assert plans["p"].evaluated and plans["q"].evaluated
    helper = rav2.anchored_blind_factor_ids if operation == "sum" else rav2.zero_gated_factor_ids
    expected = "q" if operation == "sum" else "zero_q"
    # The route f -> stock -> P -> own rate -> carrier is cut. Only the
    # separately retained f -> Q -> goal route can explain this diagnosis.
    assert helper(request.graph, [UPSTREAM], GOAL, centres) == {UPSTREAM: [expected]}


def test_product_identity_whole_response_is_exactly_unchanged_by_stock_cut(monkeypatch):
    raw = wire()
    before = stable_wire(monkeypatch, raw)
    disable_stock_cuts(monkeypatch)
    assert stable_wire(monkeypatch, raw) == before
