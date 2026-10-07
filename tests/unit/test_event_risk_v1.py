"""event_risk.v1: a risk is an EVENT that may happen within a horizon (Science 393023 pilot).

Spec: ``science-richness-P0-20261007.md`` ruling (a) + §4 PILOT. Design:
``inflight/lane-event-risk-DESIGN.md`` (DL 0fd71f, lane EVENT-RISK, 7 Oct 2026).

Known-answer fixture (Science §4): "Key supplier fails within 12 months", p in [0.05, 0.15],
severity -£8k/month on gross profit if it happens. The "Dual-source" option lowers the chance
by 70% at £1k/month. Normalised units: 1.0 = £20k/month gross profit. Gross profit is £16k/month
today (level 0.80); the goal is at least £12k/month, a LEVEL (0.60, frame "level"; Science
C-FRAME).

Analytic answers, stated before the build (verified by Science 393023, ruling Q1-Q8):

- Status quo: P(fails) = E[p] = 0.10; P(goal met) = 0.900; mean = 0.80 - 0.40 x 0.10 = 0.760.
- Dual-source: P(fails) = 0.10 x 0.3 = 0.03; P(goal met) = 0.970; mean = 0.75 - 0.40 x 0.03 = 0.738.
- Downside p05: Status quo 0.40 (the failure level); Dual-source 0.75 (3% < 5%).
- cvar_10 is bound to the REALISED failure share s of the run (Science Q8):
  (min(s, 0.1) x fail_level + max(0, 0.1 - s) x safe_level) / 0.1.
- Expected regret (the EVPI basis) at the p-conditional expectation (Science C-EVPI): Dual-source
  minus status quo = -0.05 + 0.28 p < 0 for every p in [0.05, 0.15], so the status quo is the
  informed choice on every draw: regret(status quo) = 0, regret(Dual-source) = 0.05 - 0.028 = 0.022.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

os.environ.setdefault("ISL_AUTH_DISABLED", "true")

from src.models.robustness_v2 import RobustnessRequestV2  # noqa: E402
from src.services import event_risk as event_risk_module  # noqa: E402
from src.services.event_risk import (  # noqa: E402
    OCCURRENCE_STREAM_OFFSET,
    resolve_event_risk_graph,
    resolve_event_risk_plans,
)
from src.services.robustness_analyzer_v2 import (  # noqa: E402
    DualUncertaintySampler,
    FactorSampler,
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
    definitional_strengths,
    factor_centres,
)
from src.utils.rng import SeededRNG  # noqa: E402

N = 10_000
SEED = 7

# Analytic answers (module docstring). MC tolerance: 4 standard errors at n = 10,000.
P_GOAL = {"status_quo": 0.900, "dual_source": 0.970}
MEAN = {"status_quo": 0.760, "dual_source": 0.738}


def _se(p: float, n: int = N) -> float:
    return math.sqrt(p * (1.0 - p) / n)


def _edge(src: str, dst: str, mean: float, exists: float = 1.0, std: float = 0.002) -> Dict[str, Any]:
    return {"from": src, "to": dst, "exists_probability": exists, "strength": {"mean": mean, "std": std}}


EVENT_RISK = {
    "version": 1,
    "occurrence": {"p_low": 0.05, "p_high": 0.15, "basis": "reference"},
    "horizon": {"months": 12},
    "mitigations": [{"factor_id": "dual_sourcing", "occurrence_reduction": 0.7}],
}


def supplier_request(*, event: bool = True, **overrides: Any) -> Dict[str, Any]:
    """The Science §4 fixture. ``event=False`` is the LEGACY twin: the same graph with the risk
    node unversioned (today's linear node)."""
    risk: Dict[str, Any] = {"id": "supplier_fails", "kind": "risk", "label": "Key supplier fails"}
    if event:
        risk["event_risk"] = copy.deepcopy(EVENT_RISK)
    body: Dict[str, Any] = {
        "graph": {
            "nodes": [
                {
                    "id": "gross_profit",
                    "kind": "goal",
                    "label": "Gross profit",
                    "observed_state": {"value": 0.80, "baseline": 0.80, "source": "user"},
                },
                risk,
                {
                    "id": "dual_sourcing",
                    "kind": "factor",
                    "label": "Dual sourcing in place",
                    "observed_state": {"value": 0.0},
                },
                {
                    "id": "dual_source_cost",
                    "kind": "factor",
                    "label": "Dual-source cost",
                    "observed_state": {"value": 0.0},
                },
            ],
            "edges": [
                # The producer's strength on the mitigation link is ignored by contract.
                _edge("dual_sourcing", "supplier_fails", -0.5, exists=0.8, std=0.1),
                _edge("supplier_fails", "gross_profit", -0.40),
                _edge("dual_source_cost", "gross_profit", -0.05),
            ],
        },
        "options": [
            {
                "id": "status_quo",
                "label": "Status quo",
                "interventions": {"dual_sourcing": 0.0, "dual_source_cost": 0.0},
            },
            {
                "id": "dual_source",
                "label": "Dual-source",
                "interventions": {"dual_sourcing": 1.0, "dual_source_cost": 1.0},
            },
        ],
        "goal_node_id": "gross_profit",
        "n_samples": N,
        "seed": SEED,
        "goal_threshold": 0.60,
        "goal_threshold_frame": "level",
        "request_id": "event-risk-v1-fixture",
    }
    body.update(overrides)
    return body


def _analyze(body: Dict[str, Any]):
    return RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**body))


def _result(response, option_id: str):
    matches = [r for r in response.results if r.option_id == option_id]
    assert len(matches) == 1, f"expected exactly one result for {option_id}"
    return matches[0]


@pytest.fixture(scope="module")
def event_response():
    return _analyze(supplier_request())


@pytest.fixture(scope="module")
def legacy_response():
    return _analyze(supplier_request(event=False))


# =============================================================================
# K: the known answer
# =============================================================================


class TestKnownAnswer:
    @pytest.mark.parametrize("option_id", ["status_quo", "dual_source"])
    def test_k1_chance_of_meeting_the_goal(self, event_response, option_id):
        """P(goal) = 1 - P(the supplier fails under this option), within 4 standard errors."""
        got = _result(event_response, option_id).probability_of_goal
        want = P_GOAL[option_id]
        assert got is not None
        assert abs(got - want) <= 4 * _se(want), (option_id, got, want)

    @pytest.mark.parametrize("option_id", ["status_quo", "dual_source"])
    def test_k2_mean_gross_profit(self, event_response, option_id):
        """The mean moves by severity x P(fails): 0.760 (status quo) vs 0.738 (Dual-source)."""
        got = _result(event_response, option_id).outcome_distribution.mean
        assert abs(got - MEAN[option_id]) <= 0.005, (option_id, got)

    def test_k3_mitigation_lowers_the_chance_by_the_stated_fraction(self, event_response):
        """Occurrence share under Dual-source / status quo = 1 - m = 0.3 (within MC error)."""
        fails_sq = 1.0 - _result(event_response, "status_quo").probability_of_goal
        fails_ds = 1.0 - _result(event_response, "dual_source").probability_of_goal
        assert abs(fails_ds / fails_sq - 0.3) <= 0.03, (fails_sq, fails_ds)


# =============================================================================
# T: twins, common random numbers, central agreement, evidence value, echo
# =============================================================================


class TestLegacyTwin:
    def test_t1_legacy_twin_runs_the_risk_as_today(self, legacy_response):
        """Unversioned, the risk is today's LINEAR node: it never happens, so it never lowers the
        goal (the D-09 behaviour, pinned as such, and the reason event_risk is opt-in). The
        producer's -0.5 / 0.8 link is then an ordinary belief: Dual-source = 0.75 + 0.8 x 0.5 x
        0.40 = 0.91."""
        for option_id, level in (("status_quo", 0.80), ("dual_source", 0.91)):
            result = _result(legacy_response, option_id)
            assert result.probability_of_goal == 1.0
            assert abs(result.outcome_distribution.mean - level) <= 0.005

    def test_t1_legacy_response_carries_no_echo(self, legacy_response):
        assert legacy_response.metadata.event_risks_applied is None
        dumped = legacy_response.model_dump(by_alias=True, exclude_none=True)
        assert "event_risks_applied" not in dumped["_metadata"]

    def test_t1_legacy_graph_is_the_same_object(self):
        request = RobustnessRequestV2(**supplier_request(event=False))
        assert resolve_event_risk_graph(request.graph) is request.graph
        assert resolve_event_risk_plans(request.graph.nodes) == {}

    def test_t1_legacy_factor_sampler_builds_no_occurrence_stream(self):
        request = RobustnessRequestV2(**supplier_request(event=False))
        sampler = FactorSampler(request.graph.nodes, request.parameter_uncertainties, SeededRNG(8))
        assert sampler._occurrence_rng is None
        assert sampler.sample_factor_values() == {}


def _draws(body: Dict[str, Any], n: int = 2_000):
    """Per-draw outcomes for every option on the analyzer's own samplers."""
    request = RobustnessRequestV2(**body)
    graph = resolve_event_risk_graph(request.graph)
    request = request.model_copy(update={"graph": graph})
    centres = factor_centres(request)
    edges = DualUncertaintySampler(graph.edges, SeededRNG(SEED), definitional_strengths(graph, centres))
    factors = FactorSampler(graph.nodes, request.parameter_uncertainties, SeededRNG(SEED + 1))
    evaluator = SCMEvaluatorV2(graph, factor_centres=centres)
    rows: List[Dict[str, Any]] = []
    for _ in range(n):
        edge_config = edges.sample_edge_configuration()
        factor_values = factors.sample_factor_values()
        rows.append(
            {
                "edges": edge_config,
                "factors": factor_values,
                "outcomes": {
                    o.id: evaluator.evaluate(
                        edge_strengths=edge_config,
                        interventions=o.interventions,
                        goal_node=request.goal_node_id,
                        factor_values=factor_values,
                    )
                    for o in request.options
                },
            }
        )
    return rows


class TestCommonRandomNumbers:
    def test_t2_mitigation_only_ever_removes_an_occurrence(self):
        """One occurrence draw per risk per draw, shared by every option: on no draw does
        Dual-source suffer the failure while the status quo does not (control: the status quo
        does fail on some draws)."""
        rows = _draws(supplier_request())
        # Sample frame: the change from today's 0.80; a failure is -0.40 (threshold 0.60 = -0.20).
        failed_sq = [r["outcomes"]["status_quo"] < -0.2 for r in rows]
        failed_ds = [r["outcomes"]["dual_source"] < -0.2 for r in rows]
        assert sum(failed_sq) > 100  # control: the probe sees failures
        assert sum(1 for sq, ds in zip(failed_sq, failed_ds) if ds and not sq) == 0
        # and the mitigation does remove some
        assert sum(1 for sq, ds in zip(failed_sq, failed_ds) if sq and not ds) > 50

    def test_t2_occurrence_is_its_own_stream(self):
        """Adding the event moves no other draw: edge draws and factor draws equal the legacy twin's,
        draw for draw (occurrence is separate from existence and effect)."""
        # An uncertain factor makes the factor stream draw, so a mutant that draws occurrence from
        # it would move that factor's values (a factor-free fixture would pass vacuously).
        event_rows = _draws(_with_uncertain_demand(supplier_request()), n=300)
        # Twin: the legacy graph minus the mitigation link, which the event graph holds as a
        # definition (it draws nothing), so both edge streams draw the same links in order.
        twin = _with_uncertain_demand(supplier_request(event=False))
        twin["graph"]["edges"] = [e for e in twin["graph"]["edges"] if e["from"] != "dual_sourcing"]
        legacy_rows = _draws(twin, n=300)
        for event_row, legacy_row in zip(event_rows, legacy_rows):
            shared = {k: v for k, v in event_row["edges"].items() if k != ("dual_sourcing", "supplier_fails")}
            legacy_shared = {
                k: v for k, v in legacy_row["edges"].items() if k != ("dual_sourcing", "supplier_fails")
            }
            assert shared == legacy_shared
            assert {
                k: v for k, v in event_row["factors"].items() if not k.startswith("supplier_fails")
            } == legacy_row["factors"]
            assert ("dual_sourcing", "supplier_fails") not in legacy_row["edges"]
            assert "demand" in legacy_row["factors"]  # control: the factor stream did draw

    def test_t2_stream_is_dedicated_and_ordered_p_then_u(self):
        """The occurrence state is drawn from SeededRNG(factor seed + OCCURRENCE_STREAM_OFFSET):
        p ~ U(p_low, p_high) first, then u (realised occurrence). z = u x p_mid / p."""
        request = RobustnessRequestV2(**supplier_request())
        sampler = FactorSampler(request.graph.nodes, None, SeededRNG(SEED + 1))
        values = sampler.sample_factor_values()
        replay = SeededRNG(SEED + 1 + OCCURRENCE_STREAM_OFFSET)
        p = replay.uniform(0.05, 0.15)
        u = replay.random()
        assert values["supplier_fails"] == u * 0.10 / p
        assert values["supplier_fails@p"] == p  # recorded beside z (Science Q6, slice-3 p-EVPPI)


class TestCentralAgreement:
    @pytest.mark.parametrize("option_id", ["status_quo", "dual_source"])
    def test_t3_deterministic_evaluation_is_the_expected_occurrence(self, option_id):
        """With no draw, the risk takes its EXPECTED occurrence p_mid x (1 - m x), so a central
        evaluation equals the analytic mean exactly (in the sample frame: the change from today's
        0.80, i.e. -0.040 and -0.062). MEAN-based phases only (Science Q4)."""
        request = RobustnessRequestV2(**supplier_request())
        graph = resolve_event_risk_graph(request.graph)
        evaluator = SCMEvaluatorV2(graph)
        option = next(o for o in request.options if o.id == option_id)
        central = {(e.from_, e.to): e.strength.mean * e.exists_probability for e in graph.edges}
        got = evaluator.evaluate(central, option.interventions, "gross_profit")
        assert got == pytest.approx(MEAN[option_id] - 0.80, abs=1e-12)

    def test_t3_mitigation_link_is_its_exact_linear_coefficient(self):
        """The preventer -> risk link is rewritten to -p_mid x m = -0.07 with existence 1.0, so
        coefficient-only phases agree with the evaluator; the producer's -0.5 / 0.8 is ignored."""
        request = RobustnessRequestV2(**supplier_request())
        graph = resolve_event_risk_graph(request.graph)
        link = next(e for e in graph.edges if (e.from_, e.to) == ("dual_sourcing", "supplier_fails"))
        assert link.strength.mean == pytest.approx(-0.07, abs=1e-12)
        assert link.exists_probability == 1.0
        assert definitional_strengths(graph, factor_centres(request)) == {
            ("dual_sourcing", "supplier_fails"): pytest.approx(-0.07, abs=1e-12)
        }

    def test_t3_path_decomposition_reads_the_rewritten_coefficient(self):
        """Dual-source's path through the risk carries -0.07 x -0.40 = +0.028 (prevented loss)."""
        response = _analyze(supplier_request(include_path_decomposition=True, n_samples=1000))
        decomposition = response.path_decomposition
        assert decomposition is not None
        dumped = decomposition.model_dump()
        text = repr(dumped)
        assert "supplier_fails" in text
        coefficients = _path_coefficients(dumped, ["dual_sourcing", "supplier_fails", "gross_profit"])
        assert coefficients, f"no dual_sourcing -> supplier_fails -> gross_profit path in {text[:600]}"
        assert all(c == pytest.approx(0.028, abs=1e-9) for c in coefficients), coefficients


def _path_coefficients(obj: Any, path: List[str]) -> List[float]:
    """Every coefficient attached to a path equal to ``path`` anywhere in the decomposition."""
    found: List[float] = []
    if isinstance(obj, dict):
        nodes = obj.get("path") or obj.get("node_ids") or obj.get("nodes")
        if nodes == path:
            for key in ("path_effect", "coefficient", "path_coefficient"):
                if isinstance(obj.get(key), (int, float)):
                    found.append(float(obj[key]))
        for value in obj.values():
            found.extend(_path_coefficients(value, path))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_path_coefficients(value, path))
    return found


class TestEvidenceValue:
    def test_t4a_no_value_of_information_figure_names_the_risk(self, event_response):
        """No EVPI/EVPPI row is fabricated for occurrence (no supported contract in v1)."""
        dumped = event_response.model_dump(by_alias=True, exclude_none=True)
        for key in ("factor_evppi", "evpi", "factor_sensitivity"):
            assert "supplier_fails" not in repr(dumped.get(key)), key

    def test_t4d_regret_credits_no_clairvoyance_about_occurrence(self, event_response):
        """Science C-EVPI: the information arms read the risk at its p-conditional expectation.
        Then Dual-source - status quo = -0.05 + 0.28 p < 0 on every draw: the status quo is the
        informed choice everywhere, so the whole-decision EVPI (min regret) is 0 and Dual-source's
        regret is 0.05 - 0.28 x 0.10 = 0.022. Seeing whether the supplier fails would credit about
        0.07 x 0.35 = 0.0245 of fake value to the status quo's regret."""
        sq = _result(event_response, "status_quo").pre_noise_expected_regret
        ds = _result(event_response, "dual_source").pre_noise_expected_regret
        assert sq == 0.0, f"status-quo regret {sq}: clairvoyance about occurrence was credited"
        assert ds == pytest.approx(0.022, abs=0.001), ds

    def test_t4d_information_arm_equals_the_p_conditional_recomputation(self):
        """Independent recomputation on the same draws: replace occurrence by p x (1 - m x)."""
        rows = _draws(supplier_request(), n=2_000)
        regret_sq = []
        for row in rows:
            p = row["factors"]["supplier_fails@p"]
            e = row["edges"]
            sq = e[("supplier_fails", "gross_profit")] * p
            ds = e[("supplier_fails", "gross_profit")] * p * 0.3 + e[("dual_source_cost", "gross_profit")]
            regret_sq.append(max(sq, ds) - sq)
        assert max(regret_sq) == 0.0

    def test_t4b_severity_stays_an_ordinary_link(self, event_response):
        """The risk -> child severity link is NOT definitional: edge-level reasoning keeps it."""
        request = RobustnessRequestV2(**supplier_request())
        graph = resolve_event_risk_graph(request.graph)
        assert ("supplier_fails", "gross_profit") not in definitional_strengths(graph, factor_centres(request))
        assert any(
            (row.edge_from, row.edge_to) == ("supplier_fails", "gross_profit")
            for row in event_response.sensitivity
        )


class TestEcho:
    def test_t5_echo_names_the_applied_event_risks(self, event_response):
        echo = event_response.metadata.event_risks_applied
        assert echo is not None and [e.node_id for e in echo] == ["supplier_fails"]

    def test_q5_echo_discloses_the_midpoint_is_used(self, event_response):
        """Science Q5: occurrence marginalises to the midpoint, so the range's width changes no v1
        figure. The echo says so instead of implying the range was propagated."""
        (echo,) = event_response.metadata.event_risks_applied
        assert echo.occurrence_used == pytest.approx(0.10, abs=1e-12)
        assert (echo.p_low, echo.p_high) == (0.05, 0.15)
        assert echo.range_width_propagated is False

    def test_q5_the_width_changes_no_figure(self, event_response):
        """The same midpoint with zero width gives the same goal chance (within MC error of the
        two runs); only a twin at p_low / p_high shows the width."""
        body = supplier_request()
        body["graph"]["nodes"][1]["event_risk"]["occurrence"].update(p_low=0.10, p_high=0.10)
        point = _analyze(body)
        for option_id in ("status_quo", "dual_source"):
            wide = _result(event_response, option_id).probability_of_goal
            narrow = _result(point, option_id).probability_of_goal
            assert abs(wide - narrow) <= 4 * math.sqrt(2) * _se(P_GOAL[option_id]), option_id

    def test_t5_event_risk_is_not_a_defaulted_root(self, event_response):
        """An event risk's value is its occurrence draw, never a silent default 0."""
        assert event_response.metadata.n_defaulted_root_nodes is None


# =============================================================================
# T6: the downside the user sees, through the V2 route (where `downside` is built)
# =============================================================================

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}


@pytest.fixture(scope="module")
def route_body():
    from fastapi.testclient import TestClient

    from src.api.main import app

    response = TestClient(app).post(ENDPOINT, json=supplier_request(), headers=V2_HEADERS)
    assert response.status_code == 200, response.text[:2000]
    return response.json()


def _route_option(body: Dict[str, Any], option_id: str) -> Dict[str, Any]:
    rows = [r for r in body.get("options", []) if r.get("id") == option_id]
    assert len(rows) == 1, f"no single route result for {option_id}: keys {list(body)}"
    return rows[0]


# Each option's outcome level if the supplier does / does not fail (the conditional severity and
# the cost are drawn with std 0.002, so a level is exact to about 0.002).
FAIL_LEVEL = {"status_quo": 0.40, "dual_source": 0.35}
SAFE_LEVEL = {"status_quo": 0.80, "dual_source": 0.75}


class TestDownsideWitness:
    """The downside the user sees moves with the mitigation (PL #87 6037274514)."""

    def test_t6_downside_p05_moves_from_the_failure_level(self, route_body):
        sq = _route_option(route_body, "status_quo")["downside"]["p05"]
        ds = _route_option(route_body, "dual_source")["downside"]["p05"]
        assert sq == pytest.approx(FAIL_LEVEL["status_quo"], abs=0.01), sq
        assert ds == pytest.approx(SAFE_LEVEL["dual_source"], abs=0.01), ds

    @pytest.mark.parametrize("option_id", ["status_quo", "dual_source"])
    def test_q8_cvar_10_is_bound_to_the_realised_failure_share(self, route_body, option_id):
        """Science Q8: bind by identity, not a window. s = the realised failure share on this
        seed = 1 - the option's goal chance (every failure, and only a failure, misses 0.60)."""
        row = _route_option(route_body, option_id)
        s = 1.0 - row["probability_of_goal"]
        want = (min(s, 0.1) * FAIL_LEVEL[option_id] + max(0.0, 0.1 - s) * SAFE_LEVEL[option_id]) / 0.1
        assert row["downside"]["cvar_10"] == pytest.approx(want, abs=0.003), (s, row["downside"])

    def test_t6_wire_echo(self, route_body):
        (echo,) = route_body["event_risks_applied"]
        assert echo["node_id"] == "supplier_fails"
        assert echo["occurrence_used"] == pytest.approx(0.10, abs=1e-12)


# =============================================================================
# V: the contract refuses what v1 cannot evaluate (422), never approximates it
# =============================================================================


def _refused(body: Dict[str, Any], code: str) -> None:
    with pytest.raises(ValidationError) as caught:
        RobustnessRequestV2(**body)
    assert code in str(caught.value), str(caught.value)[:800]


class TestContract:
    def test_v1_only_on_a_risk_node(self):
        body = supplier_request()
        body["graph"]["nodes"][2]["event_risk"] = copy.deepcopy(EVENT_RISK)
        body["graph"]["nodes"][2]["event_risk"]["mitigations"] = None
        _refused(body, "EVENT_RISK_WRONG_KIND")

    def test_v2_a_driver_parent_is_refused(self):
        body = supplier_request()
        body["graph"]["edges"].append(_edge("dual_source_cost", "supplier_fails", 0.1))
        _refused(body, "EVENT_RISK_DRIVER_NOT_SUPPORTED")

    def test_v3_a_mitigation_must_be_a_parent(self):
        body = supplier_request()
        body["graph"]["edges"] = [e for e in body["graph"]["edges"] if e["from"] != "dual_sourcing"]
        _refused(body, "EVENT_RISK_UNKNOWN_MITIGATION")

    def test_v4_an_option_cannot_set_the_event(self):
        body = supplier_request()
        body["options"][1]["interventions"]["supplier_fails"] = 0.0
        _refused(body, "EVENT_RISK_ROLE_REFUSED")

    def test_v5_no_parameter_uncertainty_on_the_event(self):
        body = supplier_request(
            parameter_uncertainties=[{"node_id": "supplier_fails", "distribution": "normal", "std": 0.1}]
        )
        _refused(body, "EVENT_RISK_ROLE_REFUSED")

    def test_v6_ordered_bounds_and_strict_keys(self):
        body = supplier_request()
        body["graph"]["nodes"][1]["event_risk"]["occurrence"]["p_low"] = 0.2
        _refused(body, "p_low must not exceed p_high")
        body = supplier_request()
        body["graph"]["nodes"][1]["event_risk"]["likelihood"] = 0.1
        _refused(body, "likelihood")

    def test_v7_control_the_valid_fixture_parses(self):
        request = RobustnessRequestV2(**supplier_request())
        assert request.graph.nodes[1].event_risk is not None


def test_module_has_one_owner():
    """The semantics live in one module; the analyzer imports, never re-implements, them."""
    assert event_risk_module.occurrence_value.__module__ == "src.services.event_risk"


# =============================================================================
# Science Q1 / Q2 / Q4 conditions, each with a row
# =============================================================================


class TestScienceConditions:
    def test_q1_a_preventer_has_no_other_children(self):
        body = supplier_request()
        body["graph"]["edges"].append(_edge("dual_sourcing", "gross_profit", 0.01))
        _refused(body, "EVENT_RISK_PREVENTER_HAS_OTHER_CHILDREN")

    def test_q1_an_edge_without_a_mitigation_entry_is_refused(self):
        body = supplier_request()
        body["graph"]["nodes"][1]["event_risk"]["mitigations"] = None
        _refused(body, "EVENT_RISK_DRIVER_NOT_SUPPORTED")

    def test_q2_two_preventers_multiply(self):
        """One option setting two preventers: P(fails) = p_mid (1 - 0.7)(1 - 0.5) = 0.015."""
        body = supplier_request()
        body["graph"]["nodes"].append(
            {"id": "buffer_stock", "kind": "factor", "label": "Buffer stock", "observed_state": {"value": 0.0}}
        )
        body["graph"]["edges"].append(_edge("buffer_stock", "supplier_fails", -0.1))
        body["graph"]["nodes"][1]["event_risk"]["mitigations"].append(
            {"factor_id": "buffer_stock", "occurrence_reduction": 0.5}
        )
        body["options"].append(
            {
                "id": "both",
                "label": "Dual-source and buffer stock",
                "interventions": {"dual_sourcing": 1.0, "dual_source_cost": 1.0, "buffer_stock": 1.0},
            }
        )
        for option in body["options"][:2]:
            option["interventions"]["buffer_stock"] = 0.0
        response = _analyze(body)
        fails = 1.0 - _result(response, "both").probability_of_goal
        assert abs(fails - 0.015) <= 4 * _se(0.015), fails

    def test_q4_threshold_figures_come_from_the_draws(self, event_response):
        """At the EXPECTED occurrence the status quo reads -0.04 against a threshold of -0.20 (in
        the sample frame), i.e. "met for certain"; the truth is 90%. The goal chance and its
        counts come from the Monte Carlo draws only."""
        result = _result(event_response, "status_quo")
        assert result.probability_of_goal < 0.95
        precision = result.probability_of_goal_precision
        assert precision is not None
        assert precision.n_met / precision.n_informative == pytest.approx(result.probability_of_goal, abs=1e-12)


# =============================================================================
# Codex buddy r1 findings + Science Q9 conditions
# =============================================================================


def _with_uncertain_demand(body: Dict[str, Any]) -> Dict[str, Any]:
    """Add a non-lever uncertain factor so the EVPI / EVPPI phases run."""
    body["graph"]["nodes"].append(
        {"id": "demand", "kind": "factor", "label": "Demand", "observed_state": {"value": 0.5}}
    )
    body["graph"]["edges"].append(_edge("demand", "gross_profit", 0.1, std=0.05))
    body["parameter_uncertainties"] = [{"node_id": "demand", "distribution": "normal", "std": 0.1}]
    body["include_voi"] = True
    return body


class TestBuddyRoundOne:
    def test_r1_occurrence_state_is_never_a_goal_chance_driver(self):
        """z and p are internal draw state: no driver row may name them (a z row described a
        uniform draw, P(goal | low z) = 0.70). Control: the uncertain demand factor IS a driver
        candidate and the drivers block exists."""
        response = _analyze(_with_uncertain_demand(supplier_request()))
        for result in response.results:
            drivers = result.probability_of_goal_drivers
            assert drivers is not None
            ids = {row.quantity_id for row in drivers.drivers}
            assert not ids & {"supplier_fails", "supplier_fails@p"}, ids
            # Control: the drivers block does see sampled quantities (the severity link is one).
            assert "supplier_fails->gross_profit" in ids, ids

    def test_r1_a_preventer_must_be_a_root_switch(self):
        body = supplier_request()
        body["graph"]["nodes"].append({"id": "audit", "kind": "factor", "label": "Supplier audit"})
        body["graph"]["edges"].append(_edge("audit", "dual_sourcing", 0.5))
        _refused(body, "EVENT_RISK_PREVENTER_NOT_ROOT")

    def test_r1_a_preventer_carries_no_parameter_uncertainty(self):
        body = supplier_request(
            parameter_uncertainties=[{"node_id": "dual_sourcing", "distribution": "normal", "std": 0.1}]
        )
        _refused(body, "EVENT_RISK_ROLE_REFUSED")

    def test_r1_fixed_policy_evpi_arms_read_realised_occurrence(self, monkeypatch):
        """The per-factor EVPI arms hold the policy fixed and count goal attainment / wins, so
        they must read the REALISED event (thresholding a conditional mean gave 0.50, not 0.90)."""
        seen: List[str] = []
        original = RobustnessAnalyzerV2._compute_evpi

        def spy(self, request, sampler, factor_sampler, evaluator, *args, **kwargs):
            seen.append(evaluator._occurrence_mode)
            return original(self, request, sampler, factor_sampler, evaluator, *args, **kwargs)

        monkeypatch.setattr(RobustnessAnalyzerV2, "_compute_evpi", spy)
        _analyze(_with_uncertain_demand(supplier_request(n_samples=1000)))
        assert seen == ["realised"]

    def test_r1_server_seed_is_the_seed_of_the_graph_as_sent(self):
        """Without a client seed, the streams use the seed of the graph AS SENT (the one the
        route reports), not of the rewritten mitigation link."""
        from src.services.robustness_analyzer_v2 import compute_effective_seed

        body = supplier_request()
        del body["seed"]
        request = RobustnessRequestV2(**body)
        response = RobustnessAnalyzerV2().analyze(request)
        assert response.metadata.seed_used == compute_effective_seed(request)[0]


class TestReferenceAtToday:
    def test_q9a_the_reference_never_draws_the_event(self):
        """Science Q9: the status-quo reference holds the event at today's level 0 for every
        option, so no reference draw ever carries the failure (-0.40). Control: the status-quo
        OPTION does carry it on about 10% of draws."""
        request = RobustnessRequestV2(**supplier_request(n_samples=2000))
        request._capture_draws = True
        response = RobustnessAnalyzerV2().analyze(request)
        draws = response._mc_draws
        assert draws is not None
        reference = draws["status_quo"]
        assert reference and max(abs(v) for v in reference) < 0.01
        failures = sum(1 for v in draws["option_outcomes"]["status_quo"] if v < -0.2)
        assert 120 <= failures <= 280, failures
