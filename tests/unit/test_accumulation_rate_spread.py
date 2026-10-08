"""Science §(ab): accumulation rate spread replaces operand spread additively."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import math

import numpy as np
import pytest

from tests.unit.test_accumulation_identity import accumulation_request
from tests.unit.test_accumulation_identity import CARRIER, GOAL, INFLOW, PRICE, RATE, STOCK
from tests.unit.test_r3_identity_evaluation import v2_body, wire
from src.models.robustness_v2 import RobustnessRequestV2
from src.utils.rng import SeededRNG
import src.services.robustness_analyzer_v2 as rav2


def stable_wire(monkeypatch: pytest.MonkeyPatch, request: Dict[str, Any]) -> bytes:
    """Freeze transport clocks so the ENTIRE serialized response can be pinned."""
    import src.utils.response_builder as rb

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 8, 16, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(rb, "datetime", FixedDatetime)
    monkeypatch.setattr(rb.ResponseBuilder, "get_processing_time_ms", lambda self: 0)
    # Build provenance remains present, fixed to HEAD's local default, rather
    # than depending on the CI runner's deployment environment.
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    return json.dumps(v2_body(request), sort_keys=True, separators=(",", ":")).encode()


HEAD_WIRE_HASHES = {
    # Recorded at 0c057b3b37bbc7701dfc9e375a1aefaa13585771 BEFORE source changes.
    "accumulation": "f6c2b3c088ad21b93b79e3a5bbe55e589fccade87da5ebd8d2a5c331a54f6b59",
    "product": "1699d2bc7749b2a091f8d91cea195e40229b15a5835861a4714d754009474d45",
}


@pytest.mark.parametrize("fixture", ["accumulation", "product"])
def test_absent_field_response_is_byte_identical_to_head(monkeypatch, fixture):
    request = accumulation_request() if fixture == "accumulation" else wire()
    assert hashlib.sha256(stable_wire(monkeypatch, request)).hexdigest() == HEAD_WIRE_HASHES[fixture]


def b1_request(
    inflow: float = 25.0,
    sigma: Optional[Tuple[float, float]] = (0.136, 0.136),
    *, n_samples: int = 10000,
) -> Dict[str, Any]:
    """B1-ACC: stated £49, S₀=250, churn=3%, candidate inflow=20 or25; month12 £20k."""
    request = accumulation_request()
    request.pop("seed")  # The request default is the deterministic graph-derived seed.
    request.update(n_samples=n_samples, goal_threshold=20000.0 / 100000.0)
    nodes = {node["id"]: node for node in request["graph"]["nodes"]}
    nodes[RATE]["observed_state"].update(raw_value=3.0, cap=100.0)
    nodes[INFLOW]["observed_state"].update(value=inflow / 100.0, baseline=inflow / 100.0)
    if sigma is not None:
        nodes[CARRIER]["nonlinear_identity"]["rate_sigma_log"] = list(sigma)
    return request


def sampled_response(request: Dict[str, Any]):
    return rav2.RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(request))


def keep_samples(request: Dict[str, Any]) -> np.ndarray:
    result = next(result for result in sampled_response(request).results if result.option_id == "keep")
    return np.array(result.outcome_distribution.samples)


@pytest.mark.parametrize("inflow,sigma", [(20.0, 0.136), (25.0, 0.136), (20.0, 0.246), (25.0, 0.246)])
def test_b1_keep_known_answer_and_both_inflows(inflow, sigma, capsys):
    request = b1_request(inflow, (sigma, sigma))
    response = sampled_response(request)
    result = next(result for result in response.results if result.option_id == "keep")
    assert result.probability_of_goal is not None
    # The independent oracle follows twelve MONTHS, rather than the implementation's
    # closed form. Its dedicated seed matches the declared stream, not any shared RNG.
    seed, _ = rav2.compute_effective_seed(RobustnessRequestV2.model_validate(request))
    rng = np.random.Generator(np.random.PCG64(seed + 1 + 10007))
    z = rng.normal(size=(request["n_samples"], 2))
    churn = 0.03 * np.exp(sigma * z[:, 0])
    incoming = inflow * np.exp(sigma * z[:, 1])
    stock = np.full(request["n_samples"], 250.0)
    for _ in range(12):
        stock = stock * (1.0 - churn) + incoming
    refusal_rate = float(np.mean((churn >= 1.0) | (churn < 0.0)))
    expected = float(np.mean(stock * 49.0 >= 20000.0))
    with capsys.disabled():
        print("B1_MEASUREMENT", {"inflow": inflow, "sigma": sigma, "seed": seed,
                                 "n_samples": request["n_samples"],
                                 "keep": result.probability_of_goal, "refusal_rate": refusal_rate})
    assert refusal_rate == 0.0
    assert result.probability_of_goal == expected
    if inflow == 25.0 and sigma == 0.136:
        assert result.probability_of_goal == pytest.approx(0.71, abs=0.02)


def test_zero_sigma_is_keep_100_percent_and_byte_identical_to_absent(monkeypatch):
    zero = b1_request(sigma=(0.0, 0.0), n_samples=1000)
    absent = b1_request(sigma=None, n_samples=1000)
    zero_bytes = stable_wire(monkeypatch, zero)
    absent_bytes = stable_wire(monkeypatch, absent)

    def differences(left, right, path=""):
        if isinstance(left, dict) and isinstance(right, dict):
            return [entry for key in left.keys() | right.keys()
                    for entry in differences(left.get(key), right.get(key), path + "." + key)]
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            return [entry for i, pair in enumerate(zip(left, right))
                    for entry in differences(*pair, path + f"[{i}]")]
        return [] if left == right else [(path, left, right)]

    assert zero_bytes == absent_bytes, differences(json.loads(zero_bytes), json.loads(absent_bytes))
    assert next(option for option in json.loads(zero_bytes)["options"] if option["id"] == "keep")[
        "probability_of_goal"
    ] == 1.0


def test_operand_parameter_std_is_ignored_when_rate_sigma_present():
    zero_std = b1_request(n_samples=1000)
    zero_std["parameter_uncertainties"] = [
        # The existing wire rejects Normal(std=0); point_mass is its exact,
        # zero-std representation. Spread still comes ONLY from rate_sigma_log.
        {"node_id": node_id, "distribution": "point_mass", "std": 0.0}
        for node_id in (STOCK, RATE, INFLOW)
    ]
    synthesised_std = copy.deepcopy(zero_std)
    for uncertainty in synthesised_std["parameter_uncertainties"]:
        uncertainty["distribution"] = "normal"
        uncertainty["std"] = 0.08
    assert np.array_equal(keep_samples(zero_std), keep_samples(synthesised_std))


@pytest.mark.parametrize("operation", ["product", "sum"])
def test_rate_sigma_is_strictly_refused_on_other_operations(operation):
    from fastapi.testclient import TestClient
    from src.api.main import app

    request = accumulation_request()
    identity = next(node for node in request["graph"]["nodes"] if node["id"] == GOAL)["nonlinear_identity"]
    identity.update(operation=operation, rate_sigma_log=[0.136, 0.136])
    response = TestClient(app).post("/api/v1/robustness/analyze/v2", json=request)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("value", [None, [], [0.136], [0.136, 0.136, 0.136], [-0.1, 0.136],
                                  [float("nan"), 0.136], [0.136, float("inf")]])
def test_rate_sigma_wrong_length_negative_or_nonfinite_is_422(value):
    from fastapi.testclient import TestClient
    from src.api.main import app

    request = b1_request(n_samples=100)
    identity = next(node for node in request["graph"]["nodes"] if node["id"] == CARRIER)["nonlinear_identity"]
    identity["rate_sigma_log"] = value
    response = TestClient(app).post("/api/v1/robustness/analyze/v2", content=json.dumps(request),
                                    headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text


def test_spread_worker_round_trip_retains_present_and_omits_absent():
    for sigma in (None, (0.0, 0.0), (0.136, 0.246)):
        parsed = RobustnessRequestV2.model_validate(b1_request(sigma=sigma, n_samples=100))
        dumped = json.loads(parsed.model_dump_json())
        identity = next(node for node in dumped["graph"]["nodes"] if node["id"] == CARRIER)[
            "nonlinear_identity"
        ]
        if sigma is None:
            assert "rate_sigma_log" not in identity
        else:
            assert identity["rate_sigma_log"] == list(sigma)
        reparsed = RobustnessRequestV2.model_validate_json(parsed.model_dump_json())
        assert reparsed.graph == parsed.graph
        carrier = next(node for node in reparsed.graph.nodes if node.id == CARRIER)
        assert carrier.nonlinear_identity is not None
        assert carrier.nonlinear_identity.rate_sigma_log == sigma
        product = next(node for node in reparsed.graph.nodes if node.id == GOAL)
        assert product.nonlinear_identity is not None
        assert "rate_sigma_log" not in product.nonlinear_identity.model_dump()


def test_dedicated_stream_preserves_unrelated_sampled_figures():
    absent = b1_request(sigma=None, n_samples=400)
    absent["graph"]["nodes"].extend([
        {"id": "unrelated_factor", "kind": "factor", "label": "Unrelated factor",
         "observed_state": {"value": 0.4, "baseline": 0.4, "source": "brief_extraction"}},
        {"id": "unrelated_outcome", "kind": "outcome", "label": "Unrelated outcome", "epsilon_std": 0.01},
    ])
    absent["graph"]["edges"].append({
        "from": "unrelated_factor", "to": "unrelated_outcome", "exists_probability": 0.8,
        "strength": {"mean": 0.5, "std": 0.08},
    })
    absent["parameter_uncertainties"] = [
        {"node_id": "unrelated_factor", "distribution": "normal", "std": 0.1}
    ]
    present = copy.deepcopy(absent)
    next(node for node in present["graph"]["nodes"] if node["id"] == CARRIER)[
        "nonlinear_identity"
    ]["rate_sigma_log"] = [0.136, 0.136]

    def figures(raw):
        request = RobustnessRequestV2.model_validate(raw)
        seed, _ = rav2.compute_effective_seed(request)
        edge_sampler = rav2.DualUncertaintySampler(request.graph.edges, SeededRNG(seed))
        factor_sampler = rav2.FactorSampler(request.graph.nodes, request.parameter_uncertainties, SeededRNG(seed + 1))
        evaluator = rav2.SCMEvaluatorV2(request.graph, epsilon_rng=SeededRNG(seed + 3),
                                      factor_centres=rav2.factor_centres(request))
        rows = []
        for _ in range(400):
            edges = edge_sampler.sample_edge_configuration()
            factors = factor_sampler.sample_factor_values()
            figure = evaluator.evaluate(edges, {}, "unrelated_outcome", factor_values=factors)
            rows.append((edges, factors["unrelated_factor"], figure))
        return rows

    control = figures(absent)
    assert len({row[2] for row in control}) > 1  # Positive control: this figure IS sampled.
    assert control == figures(present)


def test_sampled_rate_log_moments_and_independence(monkeypatch):
    request = RobustnessRequestV2.model_validate(b1_request())
    seed, _ = rav2.compute_effective_seed(request)
    sampler = rav2.FactorSampler(request.graph.nodes, request.parameter_uncertainties, SeededRNG(seed + 1))
    evaluator = rav2.SCMEvaluatorV2(request.graph, factor_centres=rav2.factor_centres(request))
    edges = {(edge.from_, edge.to): edge.strength.mean for edge in request.graph.edges}
    original_term = rav2._identity_term
    evaluated_operands = []

    def capture(operation, operands, **kwargs):
        if operation == "accumulation":
            evaluated_operands.append(tuple(operands))
        return original_term(operation, operands, **kwargs)

    monkeypatch.setattr(rav2, "_identity_term", capture)
    log_draws = []
    for _ in range(10000):
        factors = sampler.sample_factor_values()
        evaluator.evaluate(edges, {}, CARRIER, factor_values=factors)
        stock, churn, inflow = evaluated_operands[-1]
        assert stock == 250.0
        log_draws.append([math.log(churn / 3.0), math.log(inflow / 25.0)])
    draw_array = np.array(log_draws)
    assert np.abs(np.mean(draw_array, axis=0)).max() < 0.006
    assert np.std(draw_array, axis=0) == pytest.approx([0.136, 0.136], abs=0.004)
    assert abs(np.corrcoef(draw_array.T)[0, 1]) < 0.04


def test_option_centres_keep_price_edge_changes_replace_own_noise_and_hold_stock_exact():
    """A non-root churn keeps the sampled price edge; its own std/epsilon is replaced."""
    raw = b1_request(n_samples=100)
    raw["graph"]["edges"].append({
        "from": PRICE, "to": RATE, "exists_probability": 1.0,
        "strength": {"mean": 0.5, "std": 0.08},
    })
    nodes = {node["id"]: node for node in raw["graph"]["nodes"]}
    for node_id in (STOCK, RATE, INFLOW):
        nodes[node_id]["epsilon_std"] = 0.2
    raw["parameter_uncertainties"] = [
        {"node_id": node_id, "distribution": "normal", "std": 0.08}
        for node_id in (STOCK, RATE, INFLOW)
    ]
    request = RobustnessRequestV2.model_validate(raw)
    evaluator = rav2.SCMEvaluatorV2(request.graph, epsilon_rng=SeededRNG(303),
                                  factor_centres=rav2.factor_centres(request))
    edges = {(edge.from_, edge.to): edge.strength.mean for edge in request.graph.edges}
    # The paired draw deliberately gives each operand a wildly wrong exogenous
    # sample. The accumulation must instead read the noiseless causal centre.
    factors = {STOCK: 0.8, RATE: 0.8, INFLOW: 0.8,
               rav2.accumulation_rate_z_key(CARRIER, 0): 0.7,
               rav2.accumulation_rate_z_key(CARRIER, 1): -0.4}

    def monthly(central_churn, central_inflow):
        churn = central_churn * math.exp(0.136 * 0.7)
        inflow = central_inflow * math.exp(0.136 * -0.4)
        stock = 250.0
        for _ in range(12):
            stock = stock * (1.0 - churn) + inflow
        return stock / 1000.0

    keep = evaluator.evaluate(edges, {PRICE: 49.0 / 200.0}, CARRIER, factor_values=factors)
    assert keep == pytest.approx(monthly(0.03, 25.0), rel=1e-12)
    raised = evaluator.evaluate(edges, {PRICE: 59.0 / 200.0}, CARRIER, factor_values=factors)
    assert raised == pytest.approx(monthly(0.03 + 0.5 * (0.295 - 0.245), 25.0), rel=1e-12)
    assert raised != keep
    # Direct do(churn=4%) takes precedence over the price edge while the same
    # draw's two z values remain common to Keep, Raise and direct-rate options.
    direct = evaluator.evaluate(edges, {PRICE: 0.295, RATE: 0.04}, CARRIER, factor_values=factors)
    assert direct == pytest.approx(monthly(0.04, 25.0), rel=1e-12)
    second_edge = dict(edges)
    second_edge[(PRICE, RATE)] = 0.7
    assert evaluator.evaluate(second_edge, {PRICE: 0.295}, CARRIER, factor_values=factors) == pytest.approx(
        monthly(0.03 + 0.7 * (0.295 - 0.245), 25.0), rel=1e-12
    )


def test_drawn_rate_at_or_above_one_or_negative_fails_entire_option():
    from tests.unit.test_accumulation_refused_draws import assert_failed_option

    raw = b1_request(sigma=(0.246, 0.246), n_samples=1000)
    raw["options"].extend([
        {"id": "near_one", "label": "90% monthly churn", "interventions": {RATE: 0.9}},
        {"id": "negative", "label": "Negative monthly churn", "interventions": {RATE: -0.01}},
    ])
    body = v2_body(raw)
    options = {option["id"]: option for option in body["options"]}
    assert_failed_option(options["near_one"])
    assert_failed_option(options["negative"])
    assert options["keep"]["status"] == "computed"
    assert options["raise"]["status"] == "computed"


def test_drawn_status_quo_rate_at_one_refuses_identity():
    from tests.unit.test_r3_identity_evaluation import blocked_422

    raw = b1_request(sigma=(0.246, 0.246), n_samples=1000)
    rate = next(node for node in raw["graph"]["nodes"] if node["id"] == RATE)
    rate["observed_state"].update(value=0.9, baseline=0.9, raw_value=90.0)
    body = blocked_422(raw)
    critique = next(row for row in body["critiques"] if row["code"] == "IDENTITY_NOT_EVALUATED")
    assert critique["identity"]["node_id"] == CARRIER
    assert critique["identity"]["withheld_reason"] == "accumulation_draw_refused"


def test_level_limits_read_the_same_spread_draw_as_goal_probability():
    raw = b1_request(n_samples=1000)
    raw["goal_constraints"] = [
        {"constraint_id": "mrr", "node_id": GOAL, "operator": ">=", "value": 0.2,
         "value_frame": "level"},
        {"constraint_id": "stock", "node_id": CARRIER, "operator": ">=",
         "value": 20000.0 / 49.0 / 1000.0, "value_frame": "level"},
    ]
    keep = next(option for option in v2_body(raw)["options"] if option["id"] == "keep")
    assert 0.69 < keep["probability_of_goal"] < 0.73
    assert keep["constraint_analysis"]["constraints"][0]["prob_satisfied"] == keep["probability_of_goal"]
    assert keep["constraint_analysis"]["constraints"][1]["prob_satisfied"] == keep["probability_of_goal"]
    assert keep["constraint_analysis"]["joint_probability"] == keep["probability_of_goal"]


def test_target_objective_uses_spread_draws_without_status_quo_cancellation():
    raw = b1_request(n_samples=1000)
    raw["goal_direction"] = "target"
    response = sampled_response(raw)
    seed, _ = rav2.compute_effective_seed(RobustnessRequestV2.model_validate(raw))
    rng = np.random.Generator(np.random.PCG64(seed + 10008))
    z = rng.normal(size=(1000, 2))
    churn = 0.03 * np.exp(0.136 * z[:, 0])
    inflow = 25.0 * np.exp(0.136 * z[:, 1])
    stock = np.full(1000, 250.0)
    for _ in range(12):
        stock = stock * (1.0 - churn) + inflow
    expected_keep = float(np.mean(np.abs(stock * 49.0 - 20000.0) < np.abs(stock * 59.0 - 20000.0)))
    keep = next(result for result in response.results if result.option_id == "keep")
    assert 0.0 < expected_keep < 1.0
    assert keep.win_probability == expected_keep


def test_main_monte_carlo_tie_changes_do_not_shift_existing_streams(monkeypatch):
    """Without spread S₁=225 for both options; spread breaks those ties without shifting MC."""
    absent = b1_request(sigma=None, n_samples=1000)
    nodes = {node["id"]: node for node in absent["graph"]["nodes"]}
    nodes[CARRIER]["nonlinear_identity"]["horizon_months"] = 1
    nodes[RATE]["observed_state"].update(value=0.0, baseline=0.0, raw_value=0.0)
    absent["options"] = [
        {"id": "a", "label": "A", "interventions": {STOCK: 0.2}},
        {"id": "b", "label": "B", "interventions": {STOCK: 0.15, INFLOW: 0.75}},
    ]
    absent["graph"]["nodes"].extend([
        {"id": "unrelated_factor", "kind": "factor", "label": "Unrelated factor",
         "observed_state": {"value": 0.4, "baseline": 0.4}},
        {"id": "unrelated_outcome", "kind": "outcome", "label": "Unrelated outcome"},
    ])
    absent["graph"]["edges"].append({
        "from": "unrelated_factor", "to": "unrelated_outcome", "exists_probability": 0.8,
        "strength": {"mean": 0.5, "std": 0.08},
    })
    absent["parameter_uncertainties"] = [
        {"node_id": "unrelated_factor", "distribution": "normal", "std": 0.1},
    ]
    present = copy.deepcopy(absent)
    next(node for node in present["graph"]["nodes"] if node["id"] == CARRIER)[
        "nonlinear_identity"
    ]["rate_sigma_log"] = [0.0, 0.136]
    captured = []
    original = rav2.RobustnessAnalyzerV2._run_monte_carlo

    def capture(self, request, *args, **kwargs):
        assert request._capture_draws is False
        result = original(self, request, *args, **kwargs)
        captured.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_run_monte_carlo", capture)
    sampled_response(absent)
    sampled_response(present)
    control, spread = captured
    assert control[4] == 1000  # Positive control: legacy draws really consumed tie picks.
    assert spread[4] == 0      # The added spread really changes that winner geometry.
    assert control[3] == spread[3]  # Every sampled edge, including the unrelated edge.
    assert [row["unrelated_factor"] for row in control[6]] == [
        row["unrelated_factor"] for row in spread[6]
    ]


def test_operand_epsilon_is_ignored_on_accumulation_only_paths():
    raw = b1_request(n_samples=1000)
    for node in raw["graph"]["nodes"]:
        if node["id"] in (STOCK, RATE, INFLOW):
            node["epsilon_std"] = 0.2
    keep = next(option for option in v2_body(raw)["options"] if option["id"] == "keep")
    assert 0.69 < keep["probability_of_goal"] < 0.73


def test_operand_epsilon_on_an_alternate_belief_path_still_refuses_goal_level():
    raw = b1_request(n_samples=1000)
    next(node for node in raw["graph"]["nodes"] if node["id"] == RATE)["epsilon_std"] = 0.2
    raw["graph"]["edges"].append({
        "from": RATE, "to": GOAL, "exists_probability": 1.0,
        "strength": {"mean": 0.01, "std": 0.002},
    })
    body = v2_body(raw)
    assert all(option.get("probability_of_goal") is None for option in body["options"])
    assert any(warning["code"] == "GOAL_THRESHOLD_NOT_CONVERTIBLE"
               for warning in body["inference_warnings"])


def test_invalid_ignored_prior_cannot_refuse_legacy_tie_clock_setup():
    raw = b1_request(n_samples=100)
    raw["parameter_uncertainties"] = [
        {"node_id": RATE, "distribution": "uniform", "range_min": 1.1, "range_max": 1.2}
    ]
    # An ordinary stated product makes evaluator setup propagate the whole graph.
    # The absent-field shadow cannot evaluate that prior, but rate_sigma_log
    # replaces it with the stated 3% centre, so the actual request is valid.
    for node_id, value in (("other_a", 0.5), ("other_b", 0.5), ("other_product", 0.25)):
        raw["graph"]["nodes"].append({
            "id": node_id, "kind": "factor", "label": node_id,
            "observed_state": {"value": value, "baseline": value, "source": "brief_extraction"},
            "execution_frame": {"frame": 1.0, "carrier": "cap"},
        })
    raw["graph"]["nodes"][-1]["nonlinear_identity"] = {
        "operation": "product", "factor_ids": ["other_a", "other_b"], "stated_in_brief": True,
    }
    raw["graph"]["edges"].extend({
        "from": node_id, "to": "other_product", "exists_probability": 1.0,
        "strength": {"mean": 0.5, "std": 0.01},
    } for node_id in ("other_a", "other_b"))
    response = sampled_response(raw)
    assert len(response.results) == 2
    assert all(result.probability_of_goal is not None for result in response.results)


def two_distinct_carriers(upstream_operand: str) -> Dict[str, Any]:
    """Only raw factors connect: neither accumulation consumes another identity node."""
    raw = b1_request(n_samples=100)
    for node_id, level, frame in (
        ("stock_b", 100.0, 1000.0),
        ("rate_b", 2.0, 100.0),
        ("inflow_b", 5.0, 100.0),
        ("carrier_b", 100.0, 1000.0),
    ):
        raw["graph"]["nodes"].append({
            "id": node_id, "kind": "factor", "label": node_id,
            "observed_state": {"value": level / frame, "baseline": level / frame,
                               "source": "brief_extraction"},
            "execution_frame": {"frame": frame, "carrier": "cap"},
        })
    raw["graph"]["nodes"][-1]["nonlinear_identity"] = {
        "operation": "accumulation", "factor_ids": ["stock_b", "rate_b", "inflow_b"],
        "horizon_months": 12, "rate_scale": 0.01, "rate_sigma_log": [0.136, 0.246],
        "stated_in_brief": False,
    }
    raw["graph"]["edges"].extend({
        "from": source, "to": "carrier_b", "exists_probability": 1.0,
        "strength": {"mean": 0.5, "std": 0.08},
    } for source in ("stock_b", "rate_b", "inflow_b"))
    raw["graph"]["edges"].append({
        "from": upstream_operand, "to": RATE, "exists_probability": 1.0,
        "strength": {"mean": 0.4 if upstream_operand == "stock_b" else 0.5, "std": 0.08},
    })
    if upstream_operand == "stock_b":
        raw["graph"]["edges"].append({
            "from": PRICE, "to": "stock_b", "exists_probability": 1.0,
            "strength": {"mean": 0.5, "std": 0.08},
        })
    raw["parameter_uncertainties"] = [
        {"node_id": node_id, "distribution": "normal", "std": 0.08}
        for node_id in (STOCK, RATE, INFLOW, "stock_b", "rate_b", "inflow_b")
    ]
    return raw


def carrier_monthly_oracle(stock, churn, inflow):
    for _ in range(12):
        stock = stock * (1.0 - churn) + inflow
    return stock / 1000.0


def two_carrier_draw():
    return {
        STOCK: 0.8, RATE: 0.8, INFLOW: 0.8,
        "stock_b": 0.8, "rate_b": 0.8, "inflow_b": 0.09,
        rav2.accumulation_rate_z_key(CARRIER, 0): 0.7,
        rav2.accumulation_rate_z_key(CARRIER, 1): -0.4,
        rav2.accumulation_rate_z_key("carrier_b", 0): 0.2,
        rav2.accumulation_rate_z_key("carrier_b", 1): -0.3,
    }


def test_distinct_carrier_stock_mask_does_not_erase_upstream_price_change():
    """B stock is exact inside B; price→B stock→A churn remains causal inside A."""
    request = RobustnessRequestV2.model_validate(two_distinct_carriers("stock_b"))
    evaluator = rav2.SCMEvaluatorV2(request.graph, factor_centres=rav2.factor_centres(request))
    edges = {(edge.from_, edge.to): edge.strength.mean for edge in request.graph.edges}
    factors = two_carrier_draw()
    keep = evaluator.evaluate_multi(edges, {PRICE: 0.245}, [CARRIER, "carrier_b"], factor_values=factors)
    raised = evaluator.evaluate_multi(edges, {PRICE: 0.295}, [CARRIER, "carrier_b"], factor_values=factors)
    assert keep[CARRIER] == pytest.approx(carrier_monthly_oracle(
        250.0, 0.03 * math.exp(0.136 * 0.7), 25.0 * math.exp(0.136 * -0.4)
    ), rel=1e-12)
    # A's held churn receives BOTH sampled causal edge strengths. B's stock
    # mask must not erase their effect merely because B also has an identity.
    central_churn = 0.03 + (0.295 - 0.245) * edges[(PRICE, "stock_b")] * edges[("stock_b", RATE)]
    assert raised[CARRIER] == pytest.approx(carrier_monthly_oracle(
        250.0, central_churn * math.exp(0.136 * 0.7), 25.0 * math.exp(0.136 * -0.4)
    ), rel=1e-12)
    assert raised[CARRIER] != keep[CARRIER]
    expected_b = carrier_monthly_oracle(
        100.0, 0.02 * math.exp(0.136 * 0.2), 5.0 * math.exp(0.246 * -0.3)
    )
    assert keep["carrier_b"] == pytest.approx(expected_b, rel=1e-12)
    assert raised["carrier_b"] == keep["carrier_b"]  # B's S₀ alone remains exact.


def test_distinct_carrier_rate_mask_preserves_upstream_sampled_reference():
    """B replaces its own inflow spread; A retains that factor's paired causal background."""
    request = RobustnessRequestV2.model_validate(two_distinct_carriers("inflow_b"))
    evaluator = rav2.SCMEvaluatorV2(request.graph, factor_centres=rav2.factor_centres(request))
    edges = {(edge.from_, edge.to): edge.strength.mean for edge in request.graph.edges}
    factors = two_carrier_draw()
    intervention = {"inflow_b": 0.1}
    values = evaluator.evaluate_multi(edges, intervention, [CARRIER, "carrier_b"], factor_values=factors)
    central_churn = 0.03 + (intervention["inflow_b"] - factors["inflow_b"]) * edges[("inflow_b", RATE)]
    assert central_churn == pytest.approx(0.035)
    assert values[CARRIER] == pytest.approx(carrier_monthly_oracle(
        250.0, central_churn * math.exp(0.136 * 0.7), 25.0 * math.exp(0.136 * -0.4)
    ), rel=1e-12)
    expected_b = carrier_monthly_oracle(
        100.0, 0.02 * math.exp(0.136 * 0.2), 10.0 * math.exp(0.246 * -0.3)
    )
    assert values["carrier_b"] == pytest.approx(expected_b, rel=1e-12)
    other_background = dict(factors, inflow_b=0.06)
    second = evaluator.evaluate_multi(edges, intervention, [CARRIER, "carrier_b"], factor_values=other_background)
    assert second[CARRIER] == pytest.approx(carrier_monthly_oracle(
        250.0, (0.03 + (0.1 - 0.06) * 0.5) * math.exp(0.136 * 0.7),
        25.0 * math.exp(0.136 * -0.4)
    ), rel=1e-12)
    assert second[CARRIER] != values[CARRIER]
    assert second["carrier_b"] == values["carrier_b"]
