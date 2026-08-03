"""ROADMAP 2.356 — the compute-admission ORACLE: advertised cost must UPPER-BOUND
the evaluator work a request actually performs.

WHY THIS FILE EXISTS, AND WHY IT IS DIFFERENT FROM EVERY OTHER ADMISSION TEST.

`test_admission_calibration.py` proves the formula is SELF-CONSISTENT (its terms
agree with the advertisement, its versions are pinned, its parameters are
sufficient). `TestPhasePricingInventory` proves every phase has ANSWERED the
pricing question. Neither one ever RUNS an analysis, so neither can notice that
an answer is WRONG — that a phase's real evaluate() count exceeds what the
formula charges for it. That is precisely how ROADMAP 2.356 happened twice:

  * `_run_monte_carlo` is attributed `priced:base_mc` (S*O*W). Since ROADMAP
    2.286 it ALSO runs a complete status-quo SCM evaluation per draw for a
    level-framed goal (`sq_evaluator.evaluate(...)`), which base_mc does not
    price. The attribution was true when written and became false underneath it.
  * `_compute_alternative_winners` is attributed `bounded:` with an HONEST
    "KNOWN-UNDERCHARGE" confession — but a confession in a registry is not a
    bound, and the phase's evaluate() count is genuinely unbounded in the fragile
    edge count.

So this file closes the loop the other two leave open: it INSTRUMENTS the real
`SCMEvaluatorV2` and counts every invocation an end-to-end `analyze()` performs,
then asserts

    compute_weighted_cost(request).total  >=  counted_invocations * W

which is the admission gate's own stated convention (BASE_COST_COEF's comment:
"1 unit per sample x option x (nodes+edges) evaluate()", i.e. one evaluate()
costs W units).

⚠ HONEST SCOPE — read before trusting a green here.

 1. This is an UPPER-BOUND oracle over the WHOLE request, not a per-term
    equality. Terms charged flat and generously (bands at 200*E*O, the
    influence walk pool at 400k) create headroom that can MASK an undercharge
    elsewhere. The boundary shapes below are therefore built to switch those
    phases OFF, so the inequality is tight and an undercharge has nowhere to
    hide. A green on a shape with every phase enabled proves much less; that is
    why the sharp shapes are the ones that gate.
 2. It counts evaluate() invocations, so it is blind to work that is not an SCM
    evaluation (matrix solves inside EVPPI, path enumeration). Those phases are
    priced by their own non-evaluate terms and are out of this oracle's scope.
 3. `evaluate_multi` is counted as ONE invocation: it is a single topological
    pass that reads several target nodes out of the same computed state.
"""
from typing import Dict, List, Tuple

import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
    compute_weighted_cost,
)

from tests.unit.test_admission_calibration import _graph, _request_dict


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------
class _EvaluateCounter:
    """Counts every SCMEvaluatorV2.evaluate/evaluate_multi call made anywhere.

    Patched onto the CLASS, so it sees evaluators the analyzer constructs
    internally — including the ones a caller never gets a reference to, which is
    exactly where both 2.356 residuals live (`sq_evaluator` is constructed inside
    `_run_monte_carlo` and never escapes it).
    """

    def __init__(self) -> None:
        self.evaluate = 0
        self.evaluate_multi = 0

    @property
    def total(self) -> int:
        return self.evaluate + self.evaluate_multi


@pytest.fixture
def count_evaluates(monkeypatch):
    """Yield a counter wired into SCMEvaluatorV2 for the duration of one test."""
    counter = _EvaluateCounter()
    real_evaluate = SCMEvaluatorV2.evaluate
    real_evaluate_multi = SCMEvaluatorV2.evaluate_multi

    def counting_evaluate(self, *args, **kwargs):
        counter.evaluate += 1
        return real_evaluate(self, *args, **kwargs)

    def counting_evaluate_multi(self, *args, **kwargs):
        counter.evaluate_multi += 1
        return real_evaluate_multi(self, *args, **kwargs)

    monkeypatch.setattr(SCMEvaluatorV2, "evaluate", counting_evaluate)
    monkeypatch.setattr(SCMEvaluatorV2, "evaluate_multi", counting_evaluate_multi)
    return counter


def _run(body: dict, counter: _EvaluateCounter) -> Tuple[int, int, int]:
    """Analyse `body` under the counter. Returns (advertised, actual, W)."""
    request = RobustnessRequestV2(**body)
    advertised = compute_weighted_cost(request).total
    RobustnessAnalyzerV2().analyze(request)
    W = len(request.graph.nodes) + len(request.graph.edges)
    return advertised, counter.total * W, W


# ---------------------------------------------------------------------------
# Boundary shapes
# ---------------------------------------------------------------------------
def _level_framed_body(n_nodes: int, n_edges: int, n_samples: int, n_options: int) -> dict:
    """A LEVEL-framed goal threshold with EVERY optional phase off.

    Sharpness: with sensitivity, VOI, e-values, flips and paths all absent, the
    advertised cost is EXACTLY base_mc = S*O*W and there is ZERO headroom, so the
    per-draw status-quo evaluation has nothing to hide behind.

    The goal must carry an attested observed_state.BASELINE for the level frame
    to resolve — `value` alone is not enough, and the resolver refuses without
    it, leaving `needs_status_quo_reference` False so the shape silently stops
    testing anything (a trap-13 vacuity: the assertion would pass by exercising
    the un-instrumented path). This is not a hypothetical: the first cut of this
    fixture set `value` only, and the positive control below is what caught it.
    """
    body = _request_dict(n_nodes, n_edges, n_samples, n_options, sensitivity=False)
    body["analysis_types"] = ["comparison"]
    goal_id = f"n{n_nodes - 1}"
    for node in body["graph"]["nodes"]:
        if node["id"] == goal_id:
            node["observed_state"] = {"value": 0.7, "baseline": 0.7, "std": 0.05}
    body["goal_threshold"] = 0.9
    body["goal_threshold_frame"] = "level"
    return body


def _fragile_edge_body(n_nodes: int, n_edges: int, n_samples: int, n_options: int) -> dict:
    """A sensitivity request whose edges are fragile, so the marginal-switch
    sweep runs on as many edges as the graph has.

    Sharpness: the sensitivity term is charged at 4*E*min(CAP, S//DIV)*W, which
    is close to 1:1 with the sub-sweep's real evaluate() count, so it contributes
    almost no headroom. Whatever the alternative-winner sweep spends is therefore
    visible as excess.
    """
    body = _request_dict(n_nodes, n_edges, n_samples, n_options, sensitivity=True)
    # Wide, uncertain edges → high elasticity → the fragile set fills up.
    for edge in body["graph"]["edges"]:
        edge["strength"] = {"mean": 0.8, "std": 0.3}
        edge["exists_probability"] = 0.6
    return body


class TestEvaluatorCallCountOracle:
    """The advertised price must never be less than the work performed."""

    def test_level_frame_shape_actually_runs_the_status_quo_evaluator(self, count_evaluates):
        """POSITIVE CONTROL (trap 13) — before asserting the level-framed shape is
        UNDER-priced, prove the shape does the extra work at all.

        A delta-framed twin of the same graph is the comparator: the ONLY
        difference is the frame, so any gap between their evaluate() counts is
        the status-quo series and nothing else. Without this control, a resolver
        that quietly refused the level threshold would make the undercharge test
        below pass by testing nothing.
        """
        level = _level_framed_body(12, 20, 400, 3)
        RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**level))
        level_calls = count_evaluates.total

        count_evaluates.evaluate = 0
        count_evaluates.evaluate_multi = 0
        delta = dict(level)
        delta["goal_threshold_frame"] = "delta"
        RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**delta))
        delta_calls = count_evaluates.total

        assert level_calls - delta_calls == level["n_samples"], (
            f"the level-framed shape did not run one extra evaluation per draw "
            f"(level={level_calls}, delta={delta_calls}, S={level['n_samples']}). "
            f"Either the resolver refused the threshold — in which case this "
            f"file's status-quo assertions are VACUOUS and the shape must be "
            f"repaired, not the assertion relaxed — or the phase moved."
        )

    def test_status_quo_evaluation_is_within_the_advertised_price(self, count_evaluates):
        """RED at pristine: base_mc charges S*O*W; the request performs
        (S*O + S) evaluations, so it is under-priced by exactly S*W."""
        body = _level_framed_body(12, 20, 400, 3)
        advertised, actual, W = _run(body, count_evaluates)
        assert advertised >= actual, (
            f"UNDER-PRICED by {actual - advertised} units "
            f"({(actual / advertised - 1) * 100:.1f}%): advertised={advertised}, "
            f"actual={actual} ({count_evaluates.total} evaluate() calls x W={W}). "
            f"The per-draw status-quo reference (ROADMAP 2.286) is not in the "
            f"cost formula."
        )

    def test_alternative_winner_sweep_is_within_the_advertised_price(self, count_evaluates):
        """RED at pristine: the marginal-switch sweep spends (K+1)*O evaluations
        per fragile edge and the fragile set is not count-capped."""
        body = _fragile_edge_body(14, 26, 1000, 4)
        advertised, actual, W = _run(body, count_evaluates)
        assert advertised >= actual, (
            f"UNDER-PRICED by {actual - advertised} units "
            f"({(actual / advertised - 1) * 100:.1f}%): advertised={advertised}, "
            f"actual={actual} ({count_evaluates.total} evaluate() calls x W={W}). "
            f"_compute_alternative_winners runs an unpriced, uncapped "
            f"marginal-switch sweep per fragile edge."
        )

    @pytest.mark.parametrize(
        "label,body",
        [
            ("representative_base", _request_dict(12, 20, 2000, 3, sensitivity=False)),
            ("representative_sensitivity", _request_dict(12, 20, 2000, 3)),
            ("representative_voi", _request_dict(12, 20, 2000, 3, evpi_factors=4)),
            ("boundary_level_frame", _level_framed_body(12, 20, 400, 3)),
            ("boundary_fragile_edges", _fragile_edge_body(14, 26, 1000, 4)),
            ("boundary_many_options", _fragile_edge_body(10, 18, 1000, 8)),
        ],
    )
    def test_advertised_cost_upper_bounds_evaluator_calls(self, label, body, count_evaluates):
        """The oracle across representative AND boundary shapes."""
        advertised, actual, W = _run(body, count_evaluates)
        assert advertised >= actual, (
            f"[{label}] UNDER-PRICED by {actual - advertised} units: "
            f"advertised={advertised}, actual={actual} "
            f"({count_evaluates.total} evaluate() calls x W={W})"
        )
