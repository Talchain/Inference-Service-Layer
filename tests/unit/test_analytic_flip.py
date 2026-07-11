"""
Track S Phase 1 SPIKE — analytic flip-threshold derivation (RED-first).

Why: the flip thresholds ISL reports (``edge_e_values[].flip_mean``, and the
per-seed values inside the #71 stability bands) are found by 20-step MC-style
bisection on the expectation function — ~22 evaluator sweeps per edge per
direction. But the search runs on the *deterministic* ``SCMEvaluatorV2``
(``analyze()`` sets ``evaluator._epsilon_rng = None`` before
``_compute_edge_e_values``), whose structural equations are linear-additive:

    node_value = base + intercept + sum(parent_value * edge_strength)

Varying ONE edge's ``strength.mean`` m (effective strength ``m * ep``) while
every other edge is held fixed therefore makes each option's goal value an
EXACTLY affine function of m — so the winner-flip point is a closed-form
crossing of two lines, no bisection needed.

Contract under test (spike module ``src/services/analytic_flip.py`` — NOT
wired into any production path):

- ``analytic_edge_e_values(request, evaluator)`` mirrors
  ``RobustnessAnalyzerV2._compute_edge_e_values`` entry-for-entry (same keys,
  rounding, e_value >= 1.0 clamp, no-flip sentinel with ``e_value == inf``,
  bidirected-edge skip, increase-direction-first), with ``flip_mean`` derived
  in closed form.
- ``analytic_flip_mean_under_background(request, evaluator, edge,
  background)`` mirrors ``RobustnessAnalyzerV2._flip_mean_under_background``
  (the #71 stability-band inner search), returning None where the sampled
  background admits no flip.
- HONESTY BOUNDARY — ``AnalyticInvalidityError`` is raised (never a silently
  wrong number) when:
  (a) the evaluator can apply per-node epsilon noise (``_epsilon_rng`` set AND
      some node has ``epsilon_std > 0``): the [0,1] clamp makes outcomes
      piecewise-affine and stochastic — not analytically solvable this way;
  (b) the three-point affinity tripwire detects a non-affine outcome function
      (future non-linear structural equations must fail loud, not agree-ish).
- Import inertness: no production module under src/ imports the spike.

Agreement tolerance: bisection resolution is ``(hi - lo) / 2**20`` of a
bracket no wider than 2.0 => <= ~2e-6; wire values are rounded to 6 dp. The
tests use 1e-5 absolute on flip_mean.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.analytic_flip import (
    AnalyticInvalidityError,
    analytic_edge_e_values,
    analytic_flip_mean_under_background,
)
from src.services.robustness_analyzer_v2 import (
    DualUncertaintySampler,
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
)
from src.utils.rng import SeededRNG

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

FLIP_MEAN_TOL = 1e-5  # > bisection resolution (~2e-6) + 6-dp wire rounding
E_VALUE_TOL = 1e-4  # e_value is rounded to 4 dp on the wire


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def _price_demand_request() -> RobustnessRequestV2:
    """The sample_variants[0]-shaped price/demand/revenue graph (hand-derivable)."""
    return RobustnessRequestV2(
        request_id="analytic-flip-hand-001",
        graph={
            "nodes": [
                {"id": "price", "kind": "factor", "label": "Price"},
                {"id": "demand", "kind": "chance", "label": "Demand"},
                {"id": "revenue", "kind": "outcome", "label": "Revenue"},
            ],
            "edges": [
                {
                    "from": "price",
                    "to": "demand",
                    "exists_probability": 0.95,
                    "strength": {"mean": -0.4, "std": 0.1},
                },
                {
                    "from": "demand",
                    "to": "revenue",
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.8, "std": 0.05},
                },
                {
                    "from": "price",
                    "to": "revenue",
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.5, "std": 0.1},
                },
            ],
        },
        options=[
            {"id": "low_price", "label": "Low price", "interventions": {"price": 0.3}},
            {"id": "high_price", "label": "High price", "interventions": {"price": 0.7}},
        ],
        goal_node_id="revenue",
        seed=42,
        n_samples=200,
        include_e_values=True,
    )


def _synthetic_12n_17e_request() -> RobustnessRequestV2:
    """Deterministic 12-node / 17-edge layered DAG, 3 options.

    Copied from benchmarks/flip_stability_budget.py::synthetic_request so the
    test is self-contained (benchmarks/ is not an importable package).
    """
    factors = [f"f{i}" for i in range(4)]
    mids = [f"m{i}" for i in range(4)]
    aggs = ["a0", "a1"]
    nodes = (
        [{"id": f, "kind": "factor", "label": f} for f in factors]
        + [{"id": m, "kind": "chance", "label": m} for m in mids]
        + [{"id": a, "kind": "chance", "label": a} for a in aggs]
        + [
            {"id": "risk", "kind": "risk", "label": "risk"},
            {"id": "goal", "kind": "outcome", "label": "goal"},
        ]
    )
    edges: List[Dict[str, Any]] = []

    def edge(src: str, dst: str, mean: float, std: float, ep: float = 1.0) -> None:
        edges.append(
            {
                "from": src,
                "to": dst,
                "exists_probability": ep,
                "strength": {"mean": mean, "std": std},
            }
        )

    edge("f0", "m0", 0.5, 0.12, 0.95)
    edge("f0", "m1", -0.3, 0.1)
    edge("f1", "m1", 0.4, 0.15, 0.9)
    edge("f1", "m2", 0.35, 0.08)
    edge("f2", "m2", -0.45, 0.12, 0.85)
    edge("f2", "m3", 0.25, 0.1)
    edge("f3", "m3", 0.6, 0.14, 0.95)
    edge("f3", "m0", -0.2, 0.09)
    edge("m0", "a0", 0.55, 0.1)
    edge("m1", "a0", 0.45, 0.12, 0.9)
    edge("m2", "a1", 0.5, 0.11)
    edge("m3", "a1", -0.35, 0.13, 0.9)
    edge("a0", "risk", -0.3, 0.1)
    edge("a0", "goal", 0.6, 0.1)
    edge("a1", "goal", 0.5, 0.12, 0.95)
    edge("a1", "risk", 0.2, 0.08)
    edge("risk", "goal", -0.4, 0.1)

    options = [
        {"id": "opt_a", "label": "A", "interventions": {"f0": 0.8, "f1": 0.2}},
        {"id": "opt_b", "label": "B", "interventions": {"f0": 0.3, "f2": 0.7}},
        {"id": "opt_c", "label": "C", "interventions": {"f1": 0.6, "f3": 0.4}},
    ]
    return RobustnessRequestV2(
        request_id="analytic-flip-synthetic-001",
        graph={"nodes": nodes, "edges": edges},
        options=options,
        goal_node_id="goal",
        seed=42,
        n_samples=200,
        include_e_values=True,
    )


def _deterministic_evaluator(request: RobustnessRequestV2) -> SCMEvaluatorV2:
    """The evaluator state _compute_edge_e_values actually runs on.

    analyze() constructs the evaluator (epsilon RNG only when some node has
    epsilon_std > 0) and then sets ``evaluator._epsilon_rng = None`` before
    every structural analysis including the flip search.
    """
    return SCMEvaluatorV2(request.graph, epsilon_rng=None)


def _mc_edge_e_values(request: RobustnessRequestV2) -> List[Dict[str, Any]]:
    analyzer = RobustnessAnalyzerV2()
    result = analyzer._compute_edge_e_values(request, _deterministic_evaluator(request))
    assert result is not None, "MC e-value sweep unexpectedly exceeded budget"
    return result


def _assert_entries_agree(
    analytic: List[Dict[str, Any]], mc: List[Dict[str, Any]], context: str
) -> None:
    assert len(analytic) == len(mc), context
    for an, ref in zip(analytic, mc):
        label = f"{context}: {ref['edge_id']}"
        assert an["edge_id"] == ref["edge_id"], label
        assert an["from_id"] == ref["from_id"], label
        assert an["to_id"] == ref["to_id"], label
        assert an["current_mean"] == ref["current_mean"], label
        assert an["flip_direction"] == ref["flip_direction"], label
        an_inf = an["e_value"] == float("inf")
        ref_inf = ref["e_value"] == float("inf")
        assert an_inf == ref_inf, f"{label}: flip/no-flip disagreement"
        assert (
            abs(an["flip_mean"] - ref["flip_mean"]) <= FLIP_MEAN_TOL
        ), f"{label}: flip_mean analytic={an['flip_mean']} mc={ref['flip_mean']}"
        if not ref_inf:
            assert (
                abs(an["e_value"] - ref["e_value"]) <= E_VALUE_TOL
            ), f"{label}: e_value analytic={an['e_value']} mc={ref['e_value']}"


# ---------------------------------------------------------------------------
# Closed-form exactness (hand-derived crossings)
# ---------------------------------------------------------------------------


class TestClosedFormExactness:
    """The analytic values must equal the hand-derived line crossings.

    With effective strengths eff = m * ep and do(price=p):
        demand  = p * eff(price->demand)
        revenue = demand * eff(demand->revenue) + p * eff(price->revenue)
    Baseline winner (m at request values) is high_price (p=0.7).
    """

    def test_price_demand_edge_matches_hand_crossing(self) -> None:
        # revenue_i(m) = 0.8*0.95*p_i*m + 0.5*p_i; crossing of the two options:
        # 0.304*m + 0.2 = 0  =>  m* = -0.2/0.304  (decrease direction)
        request = _price_demand_request()
        entries = analytic_edge_e_values(request, _deterministic_evaluator(request))
        entry = next(e for e in entries if e["edge_id"] == "price->demand")
        assert entry["flip_direction"] == "decrease"
        # flip_mean carries the wire contract's 6-dp rounding, so compare
        # against the rounded closed-form value exactly.
        assert entry["flip_mean"] == pytest.approx(round(-0.2 / 0.304, 6), abs=1e-9)
        assert entry["e_value"] == pytest.approx((0.2 / 0.304) / 0.4, abs=1e-3)

    def test_price_revenue_edge_matches_hand_crossing_and_e_value_clamp(self) -> None:
        # revenue_i(m) = p_i*(-0.38*0.8) + p_i*m = p_i*(m - 0.304)
        # crossing at m* = 0.304 < current 0.5 => decrease direction;
        # |0.304/0.5| = 0.608 clamps to the e_value >= 1.0 floor.
        request = _price_demand_request()
        entries = analytic_edge_e_values(request, _deterministic_evaluator(request))
        entry = next(e for e in entries if e["edge_id"] == "price->revenue")
        assert entry["flip_direction"] == "decrease"
        assert entry["flip_mean"] == pytest.approx(0.304, abs=1e-9)
        assert entry["e_value"] == 1.0

    def test_demand_revenue_edge_cannot_flip(self) -> None:
        # revenue_i(m) = p_i*(-0.38)*m + 0.5*p_i; margin = 0.4*(-0.38m + 0.5)
        # zero at m = 1.3157... outside [-1, 1] => no flip either direction.
        request = _price_demand_request()
        entries = analytic_edge_e_values(request, _deterministic_evaluator(request))
        entry = next(e for e in entries if e["edge_id"] == "demand->revenue")
        assert entry["e_value"] == float("inf")
        assert entry["flip_mean"] == entry["current_mean"] == 0.8
        assert entry["flip_direction"] == "increase"  # MC's no-flip sentinel value


# ---------------------------------------------------------------------------
# Agreement with the MC bisection (the shipped implementation)
# ---------------------------------------------------------------------------


class TestAgreementWithMCBisection:
    def test_hand_graph_agreement(self) -> None:
        request = _price_demand_request()
        analytic = analytic_edge_e_values(request, _deterministic_evaluator(request))
        _assert_entries_agree(analytic, _mc_edge_e_values(request), "price-demand graph")

    def test_synthetic_12n_17e_agreement(self) -> None:
        request = _synthetic_12n_17e_request()
        analytic = analytic_edge_e_values(request, _deterministic_evaluator(request))
        mc = _mc_edge_e_values(request)
        assert len(mc) == 17
        _assert_entries_agree(analytic, mc, "synthetic 12n/17e graph")


# ---------------------------------------------------------------------------
# Agreement on the stability-band inner search (#71 backgrounds)
# ---------------------------------------------------------------------------


class TestBackgroundAgreement:
    def test_sampled_background_flip_means_match_bisection(self) -> None:
        """Per sampled background: same None/flip verdict, values within tol."""
        request = _synthetic_12n_17e_request()
        evaluator = _deterministic_evaluator(request)
        analyzer = RobustnessAnalyzerV2()

        checked = 0
        none_agreements = 0
        for child_seed in (11, 42, 20260711):
            sampler = DualUncertaintySampler(request.graph.edges, SeededRNG(child_seed))
            background = sampler.sample_edge_configuration()
            for edge in request.graph.edges:
                mc_value = analyzer._flip_mean_under_background(
                    request, evaluator, edge, background
                )
                an_value = analytic_flip_mean_under_background(request, evaluator, edge, background)
                label = f"seed={child_seed} edge={edge.from_}->{edge.to}"
                if mc_value is None:
                    assert an_value is None, f"{label}: MC found no flip, analytic did"
                    none_agreements += 1
                else:
                    assert an_value is not None, f"{label}: MC flipped, analytic did not"
                    assert (
                        abs(an_value - mc_value) <= FLIP_MEAN_TOL
                    ), f"{label}: analytic={an_value} mc={mc_value}"
                checked += 1
        assert checked == 3 * 17


# ---------------------------------------------------------------------------
# Honesty boundary — invalid structures must raise, never mis-answer
# ---------------------------------------------------------------------------


class TestHonestyBoundary:
    def test_epsilon_noise_evaluator_rejected(self) -> None:
        """epsilon RNG + epsilon_std > 0 => stochastic clamped outcomes: refuse."""
        request = _price_demand_request()
        noisy_graph = request.graph.model_copy(deep=True)
        noisy_graph.nodes[2].epsilon_std = 0.2  # revenue node
        noisy_request = request.model_copy(update={"graph": noisy_graph})
        evaluator = SCMEvaluatorV2(noisy_graph, epsilon_rng=SeededRNG(7))
        with pytest.raises(AnalyticInvalidityError):
            analytic_edge_e_values(noisy_request, evaluator)

    def test_epsilon_rng_without_epsilon_std_is_still_affine(self) -> None:
        """epsilon RNG present but every epsilon_std == 0 => noise can never
        fire; the evaluation is exactly affine and must still be accepted."""
        request = _price_demand_request()
        evaluator = SCMEvaluatorV2(request.graph, epsilon_rng=SeededRNG(7))
        analytic = analytic_edge_e_values(request, evaluator)
        _assert_entries_agree(analytic, _mc_edge_e_values(request), "epsilon-rng-no-std")

    def test_nonaffine_evaluator_trips_affinity_tripwire(self) -> None:
        """A non-linear structural equation must raise, not return a number."""
        request = _price_demand_request()

        class QuadraticEvaluator(SCMEvaluatorV2):
            def evaluate(
                self,
                edge_strengths: Dict[Tuple[str, str], float],
                interventions: Dict[str, float],
                goal_node: str,
                base_values: Optional[Dict[str, float]] = None,
                factor_values: Optional[Dict[str, float]] = None,
            ) -> float:
                linear = super().evaluate(
                    edge_strengths, interventions, goal_node, base_values, factor_values
                )
                return linear + 0.3 * edge_strengths.get(("price", "demand"), 0.0) ** 2

        evaluator = QuadraticEvaluator(request.graph, epsilon_rng=None)
        with pytest.raises(AnalyticInvalidityError):
            analytic_edge_e_values(request, evaluator)


# ---------------------------------------------------------------------------
# Import inertness — the spike must not touch any default path
# ---------------------------------------------------------------------------


class TestImportInertness:
    def test_no_production_module_imports_the_spike(self) -> None:
        importers = []
        for path in SRC_ROOT.rglob("*.py"):
            if path.name == "analytic_flip.py":
                continue
            if re.search(r"analytic_flip", path.read_text()):
                importers.append(str(path.relative_to(REPO_ROOT)))
        assert importers == [], f"spike module imported by production code: {importers}"
