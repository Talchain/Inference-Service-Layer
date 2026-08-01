"""
ROADMAP 2.228-F3 — intervened-factor value flips + edge-strength winner capture.

Design: PHASE0-EVIDENCE-2026-07-28/design-2228-f3-flip-probe-redesign.md
(§2.1 candidate rule, §2.2 probe method, §2.4 wire fields, §4 Lane ISL).
Background: diagnosis-2228-enrichment-values.md §1 — the live control that proved
a flip search over a non-intervened factor can never succeed (43 probe rows,
zero `found`).

WHAT IS UNDER TEST

1. Winner capture on edge e-values. `_compute_edge_e_values` already knew the
   argmax on the flipped side of every bisection bracket and threw it away
   (analyzer :4727-4728 at 7c681fda). It is now retained as
   `alternative_winner_id`, alongside a per-row `baseline_winner_id` — zero
   extra evaluations.

2. A new, request-gated factor-flip phase. For each ROOT factor the per-option
   transmission slope T_o = (goal_o(F=1) - goal_o(F=0)) / (1 - 0) is measured
   with 2*O deterministic evaluations. This is EXACT, not an approximation:
   epsilon noise is disabled before every post-MC structural analysis
   (analyzer :1725), so the SCM is exactly affine in a root factor's value.

       F is a flip candidate  <=>  max_o T_o - min_o T_o > 1e-9

   Non-candidates are ATTESTED, never silently skipped: flip_reason
   'structurally_invariant' is a mathematical proof of no-flip, not a failed
   probe. That is exactly the class the diagnosis measured 43 wasted MC probes
   against.

   For candidates the crossing is CLOSED FORM per rival,
   F* = (A_i - A_j)/(T_j - T_i), confirmed by one argmax evaluation strictly on
   the far side (R6: a pairwise crossing is not necessarily an argmax change
   when a third option is above both lines there).

TRAP DISCIPLINE OBSERVED HERE

- trap 13 (an absence assertion must first prove it can see a presence): the
  zero-evaluation test asserts an EXACT call count so it fails in both
  directions; the "flag absent -> no block" pin is paired with a positive
  control; the unflippable-edge test asserts that an unflippable edge exists.
- trap 12b (a control pinned to "current" decays into a tautology): the band
  test's positive control is a MUTANT that pins every background to the
  expected-value configuration and must collapse the band.
- Every VALUE asserted below is hand-computed from the SCM in the docstring of
  the graph builder that produces it — never read back from the implementation.

⚠ A DESIGN INCONSISTENCY WAS FOUND AND IS RESOLVED IN FAVOUR OF §2.1 — see
  TestControlGraphCrossing.test_literal_diagnosis_s1_control_lever_is_invariant.
"""

import time
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.response_v2 import FactorFlipValueV2
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    FLIP_STABILITY_N_SEEDS,
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
)

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}

SEED = 4242
N_SAMPLES = 200  # the factor-flip phase is deterministic; MC size is irrelevant to it


# ---------------------------------------------------------------------------
# Graph builders — every constant here is load-bearing for a hand computation.
# ---------------------------------------------------------------------------


def _node(node_id: str, kind: str, value: Optional[float] = None) -> Dict[str, Any]:
    node: Dict[str, Any] = {"id": node_id, "kind": kind, "label": node_id}
    if value is not None:
        node["observed_state"] = {"value": value}
    return node


# StrengthDistribution.std is constrained gt=0.001, so a truly deterministic edge
# cannot be expressed. NEAR_ZERO_STD is the smallest legal value that keeps every
# hand computation below exact: the expected-value baseline every flip is searched
# against is mean * exists_probability and does not depend on std at all.
NEAR_ZERO_STD = 0.002


def _edge(
    src: str, dst: str, mean: float, std: float = NEAR_ZERO_STD, ep: float = 1.0
) -> Dict[str, Any]:
    return {
        "from": src,
        "to": dst,
        "strength": {"mean": mean, "std": std},
        "exists_probability": ep,
    }



# The route-level RequestValidator rejects an option whose interventions are
# EMPTY (EMPTY_INTERVENTIONS) or whose intervention targets have no effective path
# to the goal (NO_EFFECTIVE_PATH_TO_GOAL). But every graph here needs at least one
# option that does NOT intervene on the factor under test — that asymmetry is the
# whole mechanism §2.1 relies on.
#
# FAC_PAD resolves the tension without disturbing a single hand computation: a
# root factor with observed value 0.0 and a real edge to the goal, on which the
# "no intervention" option does do(FAC_PAD = 0.0) — i.e. it sets the value the
# node already has. Its contribution to the goal is 0.0 * strength = 0.0 for
# EVERY option (the others read the same 0.0 from observed_state), so every
# affine coefficient below is exactly as documented.
#
# It also carries two free cases: FAC_PAD's own row exercises the
# 'no_effect_within_bounds' arm (its only crossing is at a negative value), and
# its edge is an unflippable e-value entry.
FAC_PAD = "fac_pad"


def _pad_only() -> Dict[str, float]:
    """A validator-satisfying intervention that changes no outcome."""
    return {FAC_PAD: 0.0}


def _request(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    options: List[Dict[str, Any]],
    *,
    goal: str = "outcome",
    include_factor_flips: bool = True,
) -> RobustnessRequestV2:
    return RobustnessRequestV2(
        **{
            "request_id": "factor-flip-test",
            "graph": {
                "nodes": [*nodes, _node(FAC_PAD, "factor", 0.0)],
                "edges": [*edges, _edge(FAC_PAD, goal, 0.5)],
            },
            "options": options,
            "goal_node_id": goal,
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "include_e_values": True,
            "include_factor_flips": include_factor_flips,
        }
    )


def _control_graph(**kw: Any) -> RobustnessRequestV2:
    """The diagnosis §1 control, with ONE option NOT intervening on the lever.

    Nodes:  fac_lever (root, observed 0.3), fac_x (root, observed 0.5), outcome (goal)
    Edges:  fac_lever -> outcome  mean 0.5, exists_probability 1.0, std 0
            fac_x     -> outcome  mean 0.4, exists_probability 1.0, std 0

    The SCM (analyzer :1152, epsilon disabled) is
        outcome = lever_value * 0.5 + x_value * 0.4
    so with fac_x at its observed 0.5 the per-option goals as functions of the
    lever value L are exactly affine:

        opt_a  do(fac_lever = 0.8):  0.8*0.5 + 0.5*0.4 = 0.60          (T = 0)
        opt_b  do(fac_lever = 0.2):  0.2*0.5 + 0.5*0.4 = 0.30          (T = 0)
        opt_c  no intervention:      L*0.5   + 0.5*0.4 = 0.20 + 0.5*L  (T = 0.5)

    At the observed L = 0.3: a 0.60, b 0.30, c 0.35 -> baseline winner opt_a.

    HAND-COMPUTED CROSSING (opt_a vs opt_c):
        0.60 = 0.20 + 0.5*L   =>   L* = (0.60 - 0.20) / 0.5 = 0.80
    opt_b is PARALLEL to opt_a (both T = 0) and must never yield a crossing.
    Just above L* the argmax is opt_c, so the row is
        flip_value 0.8 · direction 'increase' · alternative_winner_id 'opt_c'.

    fac_x is intervened by NOBODY and lies on no severed path, so T = 0.4 for
    every option: spread 0 -> 'structurally_invariant'. That is the diagnosis §1
    class, mathematically attested instead of probed.

    Edge side: the fac_x -> outcome edge contributes 0.5*s_X to EVERY option, so
    no perturbation of it can move the argmax -> an unflippable e-value entry.
    """
    return _request(
        nodes=[
            _node("fac_lever", "factor", 0.3),
            _node("fac_x", "factor", 0.5),
            _node("outcome", "outcome"),
        ],
        edges=[_edge("fac_lever", "outcome", 0.5), _edge("fac_x", "outcome", 0.4)],
        options=[
            {"id": "opt_a", "label": "A", "interventions": {"fac_lever": 0.8}},
            {"id": "opt_b", "label": "B", "interventions": {"fac_lever": 0.2}},
            {"id": "opt_c", "label": "C", "interventions": _pad_only()},
        ],
        **kw,
    )


def _literal_s1_graph(**kw: Any) -> RobustnessRequestV2:
    """The diagnosis §1 control EXACTLY as written: BOTH options do() the lever."""
    return _request(
        nodes=[
            _node("fac_lever", "factor", 0.3),
            _node("fac_x", "factor", 0.5),
            _node("outcome", "outcome"),
        ],
        edges=[_edge("fac_lever", "outcome", 0.5), _edge("fac_x", "outcome", 0.4)],
        options=[
            {"id": "opt_a", "label": "A", "interventions": {"fac_lever": 0.8}},
            {"id": "opt_b", "label": "B", "interventions": {"fac_lever": 0.2}},
        ],
        **kw,
    )


def _all_invariant_graph(**kw: Any) -> RobustnessRequestV2:
    """No candidate factors at all — the whole diagnosis §1 class in one graph.

        outcome = f1*0.5 + f2*0.4 + lever*0.3
    Both options intervene ONLY on `lever`, so:
      - f1, f2: intervened by nobody, no severing -> identical slopes -> invariant
      - lever : intervened by ALL options -> every slope 0 -> invariant
        (§2.1's getFactorsOverriddenByAllOptions case, derived rather than listed)

    Nothing here is a candidate, so the phase must consume ZERO crossing,
    confirmation and stability-background evaluations.
    """
    return _request(
        nodes=[
            _node("f1", "factor", 0.4),
            _node("f2", "factor", 0.6),
            _node("lever", "factor", 0.5),
            _node("outcome", "outcome"),
        ],
        edges=[
            _edge("f1", "outcome", 0.5),
            _edge("f2", "outcome", 0.4),
            _edge("lever", "outcome", 0.3),
        ],
        options=[
            {"id": "opt_a", "label": "A", "interventions": {"lever": 0.9}},
            {"id": "opt_b", "label": "B", "interventions": {"lever": 0.1}},
        ],
        **kw,
    )


def _out_of_bounds_graph(**kw: Any) -> RobustnessRequestV2:
    """A genuine candidate whose only crossing lies OUTSIDE [0, 1].

    Nodes:  fac_lever (root, observed 0.2), fac_boost (root, observed 0.0), outcome
    Edges:  fac_lever -> outcome 0.5 · fac_boost -> outcome 0.5

        opt_a  do(fac_lever=1.0, fac_boost=1.0): 1.0*0.5 + 1.0*0.5 = 1.0  (T_lever = 0)
        opt_c  no intervention:                  0.5*L + 0.0*0.5 = 0.5*L  (T_lever = 0.5)

    Slopes differ (spread 0.5) so fac_lever IS a candidate — this exercises the
    honest-absence path, not the structurally-invariant one. The crossing is
        1.0 = 0.5*L   =>   L* = 2.0
    which is outside [0, 1]: the row must carry flip_value None and
    'no_effect_within_bounds'. A fabricated in-range value here is precisely the
    failure this roadmap item exists to remove.

    fac_boost is also a candidate (T_a = 0, T_c = 0.5) and its crossing is
        A_a = 1.0, A_c = 0.2*0.5 = 0.1  =>  L* = (1.0 - 0.1)/0.5 = 1.8
    also out of bounds. Two candidates, so the cap test has something to cap.
    """
    return _request(
        nodes=[
            _node("fac_lever", "factor", 0.2),
            _node("fac_boost", "factor", 0.0),
            _node("outcome", "outcome"),
        ],
        edges=[_edge("fac_lever", "outcome", 0.5), _edge("fac_boost", "outcome", 0.5)],
        options=[
            {"id": "opt_a", "label": "A", "interventions": {"fac_lever": 1.0, "fac_boost": 1.0}},
            {"id": "opt_c", "label": "C", "interventions": _pad_only()},
        ],
        **kw,
    )


def _upstream_partial_intervention_graph(**kw: Any) -> RobustnessRequestV2:
    """The case the founder ruling's literal wording misses (design §2.1 bullet 4).

        fac_up -> mid -> outcome,  strengths 0.5 and 0.6
        outcome = mid*0.6,  mid = fac_up*0.5

    `fac_up` is intervened by NOBODY, so a rule phrased as "factors intervened by
    some but not all options" would skip it. But do(mid=...) SEVERS fac_up's only
    path to the goal, so the intervening option's slope is 0 while the other's is
    0.5*0.6 = 0.3. The derived slope rule catches it automatically.

        opt_a  do(mid = 0.4):  0.4*0.6 = 0.24        (T_up = 0)
        opt_c  no do():        U*0.5*0.6 = 0.3*U     (T_up = 0.3)
    At the observed U = 0.4: a 0.24, c 0.12 -> baseline winner opt_a.
    HAND-COMPUTED CROSSING: 0.24 = 0.3*U  =>  U* = 0.8, strictly inside (0.4, 1).
    """
    return _request(
        nodes=[
            _node("fac_up", "factor", 0.4),
            _node("mid", "factor"),
            _node("outcome", "outcome"),
        ],
        edges=[_edge("fac_up", "mid", 0.5), _edge("mid", "outcome", 0.6)],
        options=[
            {"id": "opt_a", "label": "A", "interventions": {"mid": 0.4}},
            {"id": "opt_c", "label": "C", "interventions": _pad_only()},
        ],
        **kw,
    )


def _edge_flip_graph(**kw: Any) -> RobustnessRequestV2:
    """A graph whose fac_lever edge flips the winner at a comfortably non-zero mean.

        outcome = lever_value*s_L + x_value*s_X,  s_L = 0.5, s_X = 0.4
        opt_a  do(lever=0.8):              0.8*s_L + 0.5*s_X
        opt_b  do(lever=0.2, fac_x=1.0):   0.2*s_L + 1.0*s_X
        opt_c  no do():                    0.3*s_L + 0.5*s_X

    At the expected values a 0.60, b 0.50, c 0.35 -> baseline winner opt_a.
    Sweeping s_L: goal_a - goal_b = 0.6*s_L - 0.5*s_X = 0.6*s_L - 0.2, which is
    zero at s_L = 1/3 -> HAND-COMPUTED edge flip at 0.3333..., alternative
    winner opt_b. Chosen deliberately away from 0 so the 6-dp rounding of
    flip_mean cannot land on a three-way tie.
    """
    return _request(
        nodes=[
            _node("fac_lever", "factor", 0.3),
            _node("fac_x", "factor", 0.5),
            _node("outcome", "outcome"),
        ],
        edges=[_edge("fac_lever", "outcome", 0.5), _edge("fac_x", "outcome", 0.4)],
        options=[
            {"id": "opt_a", "label": "A", "interventions": {"fac_lever": 0.8}},
            {"id": "opt_b", "label": "B", "interventions": {"fac_lever": 0.2, "fac_x": 1.0}},
            {"id": "opt_c", "label": "C", "interventions": _pad_only()},
        ],
        **kw,
    )


def _uncertain_edge_graph(**kw: Any) -> RobustnessRequestV2:
    """A candidate whose crossing genuinely MOVES with the sampled edge background.

        outcome = lever*s_L + x*s_X,  s_L ~ N(0.5, 0.15), s_X ~ N(0.4, 0.15), both ep 1
        opt_a  do(lever=0.8, fac_x=0.0):  0.8*s_L
        opt_b  do(lever=0.2):             0.2*s_L + 0.5*s_X
        opt_c  no do():                   L*s_L   + 0.5*s_X

    Crossing (opt_a vs opt_c):  0.8*s_L = 0.5*s_X + s_L*L
        =>  L* = 0.8 - 0.5 * (s_X / s_L)
    which depends on the RATIO of the two sampled strengths — so it is NOT
    invariant across backgrounds. At the expected values
    L* = 0.8 - 0.5*(0.4/0.5) = 0.40, and the observed L is 0.3, so the base row
    is an 'increase' flip at 0.40.

    (The symmetric variant without the fac_x intervention has the s_X term
    cancel exactly, giving the SAME crossing under every background — a band
    that looks stable because the graph is degenerate, not because the finding
    is robust. That variant would have made the band test vacuous, which is
    the trap-12b failure mode this builder exists to avoid.)
    """
    return _request(
        nodes=[
            _node("fac_lever", "factor", 0.3),
            _node("fac_x", "factor", 0.5),
            _node("outcome", "outcome"),
        ],
        edges=[
            _edge("fac_lever", "outcome", 0.5, std=0.15),
            _edge("fac_x", "outcome", 0.4, std=0.15),
        ],
        options=[
            {"id": "opt_a", "label": "A", "interventions": {"fac_lever": 0.8, "fac_x": 0.0}},
            {"id": "opt_b", "label": "B", "interventions": {"fac_lever": 0.2}},
            {"id": "opt_c", "label": "C", "interventions": _pad_only()},
        ],
        **kw,
    )


def _multi_crossing_edge_graph(**kw: Any) -> RobustnessRequestV2:
    """An EDGE whose winner changes TWICE as its strength sweeps down.

    Added after a mutation check: with only one argmax change, a winner-capture
    that froze at the boundary probe and never updated during the bisection gave
    the SAME answer as the correct one, so the capture test could not see the
    defect. Here the two answers differ.

        outcome = lever_value * s_L + x_value * s_X,  s_X fixed at 0.4
        opt_a  do(lever=0.8):             goal = 0.8*s_L + 0.5*0.4 = 0.8*s_L + 0.20
        opt_b  do(lever=0.3, fac_x=0.75): goal = 0.3*s_L + 0.75*0.4 = 0.3*s_L + 0.30
        opt_c  do(lever=0.0, fac_x=0.80): goal = 0.0*s_L + 0.80*0.4 = 0.32

    At the baseline s_L = 0.5: a 0.60, b 0.45, c 0.32 -> winner opt_a.
    At the search BOUNDARY s_L = -1:  a -0.60, b 0.00, c 0.32 -> winner opt_c.
    But the FIRST loss of the lead, sweeping down from 0.5, is to opt_b:
        0.8*s + 0.20 = 0.3*s + 0.30  =>  s = 0.10/0.50 = 0.20
    and opt_c only overtakes opt_b further down (at 0.30*s + 0.30 = 0.32, s = 0.0667).

    So the correct row is flip_mean 0.20 with alternative_winner_id 'opt_b'; an
    implementation that reported the BOUNDARY winner would say 'opt_c'.
    """
    return _request(
        nodes=[
            _node("fac_lever", "factor", 0.3),
            _node("fac_x", "factor", 0.5),
            _node("outcome", "outcome"),
        ],
        edges=[_edge("fac_lever", "outcome", 0.5), _edge("fac_x", "outcome", 0.4)],
        options=[
            {"id": "opt_a", "label": "A", "interventions": {"fac_lever": 0.8}},
            {"id": "opt_b", "label": "B", "interventions": {"fac_lever": 0.3, "fac_x": 0.75}},
            {"id": "opt_c", "label": "C", "interventions": {"fac_lever": 0.0, "fac_x": 0.8}},
        ],
        **kw,
    )


def _multi_rival_graph(**kw: Any) -> RobustnessRequestV2:
    """TWO in-bounds crossings above the current value, listed FARTHEST-FIRST.

    Added after a mutation check: with a single crossing, "nearest crossing" and
    "any crossing" are the same answer, so the selection rule was untested (R6).
    Three distinct slopes need differential severing, which a two-path graph
    gives:

        fac_f -> mid_m -> outcome   (0.6, then 0.5)
        fac_f -> outcome            (0.25)
        =>  outcome = fac_f*(0.6*0.5) + fac_f*0.25 = 0.55 * fac_f  when nothing is pinned

        opt_lead    do(fac_f = 0.9):  0.55*0.9 = 0.495          (T = 0   )
        opt_direct  no pin on either: 0.55*F                    (T = 0.55)
        opt_mid     do(mid_m = 0.6):  0.6*0.5 + 0.25*F = 0.30 + 0.25*F   (T = 0.25)

    At the observed F = 0.3: lead 0.495, direct 0.165, mid 0.375 -> leader opt_lead.
    Two crossings with the leader, both inside [0, 1]:
        vs opt_mid:     0.495 = 0.30 + 0.25*F  =>  F = 0.78   <- NEAREST, the answer
        vs opt_direct:  0.495 = 0.55*F         =>  F = 0.90
    The options are ordered so the FARTHER crossing is enumerated first: an
    implementation that took crossings in rival order instead of sorting outward
    from the current value would answer 0.90.
    """
    return _request(
        nodes=[
            _node("fac_f", "factor", 0.3),
            _node("mid_m", "factor"),
            _node("outcome", "outcome"),
        ],
        edges=[
            _edge("fac_f", "mid_m", 0.6),
            _edge("mid_m", "outcome", 0.5),
            _edge("fac_f", "outcome", 0.25),
        ],
        options=[
            {"id": "opt_lead", "label": "Lead", "interventions": {"fac_f": 0.9}},
            {"id": "opt_direct", "label": "Direct", "interventions": _pad_only()},
            {"id": "opt_mid", "label": "Mid", "interventions": {"mid_m": 0.6}},
        ],
        **kw,
    )


def _row(rows: Optional[List[Dict[str, Any]]], factor_id: str) -> Dict[str, Any]:
    assert rows is not None, "factor_flip_values block missing entirely"
    matches = [r for r in rows if r["factor_id"] == factor_id]
    assert len(matches) == 1, f"expected exactly one row for {factor_id}, got {len(matches)}"
    return matches[0]


def _independent_winner(
    request: RobustnessRequestV2,
    *,
    factor_values: Optional[Dict[str, float]] = None,
    edge_overrides: Optional[Dict[Any, float]] = None,
) -> str:
    """Re-derive the argmax with a FRESH evaluator, never the producer's own."""
    evaluator = SCMEvaluatorV2(request.graph)
    config = {(e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges}
    config.update(edge_overrides or {})
    outcomes = {
        o.id: evaluator.evaluate(
            edge_strengths=config,
            interventions=o.interventions,
            goal_node=request.goal_node_id,
            factor_values=factor_values,
        )
        for o in request.options
    }
    return sorted(outcomes.items(), key=lambda x: (-x[1], x[0]))[0][0]


# ---------------------------------------------------------------------------
# (a) Golden hand-computable graph — assert the VALUE, not just presence
# ---------------------------------------------------------------------------


class TestControlGraphCrossing:
    def test_lever_flip_value_matches_the_hand_computed_crossing(self):
        """Design §4 Lane ISL test (a): crossing matches the analytic value to 1e-6.

        Hand computation in _control_graph's docstring: L* = 0.80 exactly.
        """
        resp = RobustnessAnalyzerV2().analyze(_control_graph())
        row = _row(resp.factor_flip_values, "fac_lever")

        assert row["flip_reason"] == "found"
        assert row["flip_value"] is not None
        assert abs(row["flip_value"] - 0.8) < 1e-6, f"expected 0.8, got {row['flip_value']}"
        assert row["direction"] == "increase"
        assert row["alternative_winner_id"] == "opt_c"
        assert row["baseline_winner_id"] == "opt_a"
        assert row["current_value"] == 0.3

    def test_non_intervened_factor_is_attested_structurally_invariant(self):
        """The diagnosis §1 class: fac_x transmits identically to every option."""
        resp = RobustnessAnalyzerV2().analyze(_control_graph())
        row = _row(resp.factor_flip_values, "fac_x")

        assert row["flip_reason"] == "structurally_invariant"
        assert row["flip_value"] is None
        assert row["direction"] is None
        assert row["alternative_winner_id"] is None
        assert row["baseline_winner_id"] == "opt_a"
        assert "stability" not in row, "an attested no-flip has no band to report"

    def test_argmax_confirmation_is_a_real_argmax_not_a_pairwise_gap(self):
        """R6 — opt_b is parallel to the leader and must never yield a crossing.

        A pairwise implementation dividing by a degenerate (T_j - T_i) would emit
        a spurious crossing or raise. The emitted alternative winner is checked
        against an INDEPENDENT re-derivation just past the reported crossing.
        """
        request = _control_graph()
        row = _row(RobustnessAnalyzerV2().analyze(request).factor_flip_values, "fac_lever")
        winner = _independent_winner(
            request, factor_values={"fac_lever": row["flip_value"] + 1e-6}
        )
        assert winner == row["alternative_winner_id"] == "opt_c"

    def test_nearest_of_several_in_bounds_crossings_wins(self):
        """MUTATION-DRIVEN (M12/R6). With a single crossing, "nearest" and "any"
        are the same answer, so the selection rule was untested. Here two
        crossings sit above the current value and the options are ordered
        farthest-first, so taking crossings in rival order answers 0.90 instead
        of the correct 0.78.
        """
        request = _multi_rival_graph()
        row = _row(RobustnessAnalyzerV2().analyze(request).factor_flip_values, "fac_f")

        assert row["flip_reason"] == "found"
        assert abs(row["flip_value"] - 0.78) < 1e-6, f"expected 0.78, got {row['flip_value']}"
        assert row["baseline_winner_id"] == "opt_lead"
        assert row["alternative_winner_id"] == "opt_mid"
        # Positive control: the FARTHER crossing is real too (0.90 is a genuine
        # crossing, not a nonsense number), so "0.78" is the nearest-of-two and
        # not merely the only one.
        assert (
            _independent_winner(request, factor_values={"fac_f": 0.90 + 1e-6}) != "opt_lead"
        )

    def test_literal_diagnosis_s1_control_lever_is_invariant(self):
        """⚠ DESIGN INCONSISTENCY, resolved in favour of the ratified §2.1 rule.

        The design's live-witness plan (§4, step 1) expects `fac_lever` to come
        back `found` on the diagnosis §1 control graph. But that graph is
        described as "two options that intervene ONLY on `fac_lever`" — i.e.
        BOTH options do() the lever. Under the ratified §2.1 rule every option's
        transmission slope for that lever is 0, the spread is 0, and the honest
        answer is 'structurally_invariant' — which is what §2.1 itself says
        ("intervened-by-ALL-options factor -> all T identical -> skipped"). It is
        also plainly correct: if every option overrides the lever, its observed
        value cannot move the winner.

        §2.1 is the ratified rule and wins here. The consequence is recorded
        loudly in the PR body: the live-witness graph needs ONE option that does
        NOT intervene on the lever (what _control_graph builds) before
        `fac_lever` can ever read `found`.
        """
        resp = RobustnessAnalyzerV2().analyze(_literal_s1_graph())
        assert _row(resp.factor_flip_values, "fac_lever")["flip_reason"] == (
            "structurally_invariant"
        )
        assert _row(resp.factor_flip_values, "fac_x")["flip_reason"] == "structurally_invariant"


# ---------------------------------------------------------------------------
# (b) Zero-evaluation spy on the provably-inert class
# ---------------------------------------------------------------------------


def _counting_evaluator(request: RobustnessRequestV2):
    evaluator = SCMEvaluatorV2(request.graph)
    calls = {"n": 0}
    real_evaluate = evaluator.evaluate

    def counting(*args: Any, **kwargs: Any) -> float:
        calls["n"] += 1
        return real_evaluate(*args, **kwargs)

    evaluator.evaluate = counting  # type: ignore[method-assign]
    return evaluator, calls


class TestNoWastedComputeOnInertFactors:
    def test_invariant_class_consumes_zero_crossing_evaluations(self):
        """Design §4 Lane ISL test (b) — spy on the evaluator call count.

        An EXACT count, not an upper bound, so the assertion fails in BOTH
        directions (trap 13). The phase is allowed exactly:
            O            baseline-winner evaluations, plus
            2 * O * F    slope-screen evaluations over F eligible root factors
        and NOTHING else — no crossing confirmation, no stability backgrounds —
        because no factor in this graph passes the candidate screen.
        """
        request = _all_invariant_graph()
        evaluator, calls = _counting_evaluator(request)

        rows = RobustnessAnalyzerV2()._compute_factor_flip_values(request, evaluator, SEED)

        n_options = len(request.options)
        has_parent = {e.to for e in request.graph.edges}
        n_root_factors = len(
            [
                n
                for n in request.graph.nodes
                if n.id != request.goal_node_id
                and n.id not in has_parent
                and n.observed_state is not None
            ]
        )
        expected = n_options + 2 * n_options * n_root_factors
        assert calls["n"] == expected, (
            f"factor-flip phase spent {calls['n']} evaluations on a graph with no "
            f"candidates; the screen alone costs {expected}. Any excess is compute "
            f"burned on the provably-inert class the diagnosis measured."
        )
        assert rows is not None
        assert {r["flip_reason"] for r in rows} == {"structurally_invariant"}
        assert all(r["flip_value"] is None for r in rows)

    def test_positive_control_a_candidate_does_spend_more_than_the_screen(self):
        """trap 13 — prove the counter above can SEE a presence."""
        request = _control_graph()
        evaluator, calls = _counting_evaluator(request)

        RobustnessAnalyzerV2()._compute_factor_flip_values(request, evaluator, SEED)

        has_parent = {e.to for e in request.graph.edges}
        n_root_factors = len(
            [
                n
                for n in request.graph.nodes
                if n.id != request.goal_node_id
                and n.id not in has_parent
                and n.observed_state is not None
            ]
        )
        screen_only = len(request.options) * (1 + 2 * n_root_factors)
        assert calls["n"] > screen_only, (
            "a graph with a real candidate must cost more than the screen — "
            "otherwise the zero-evaluation assertion above is vacuous"
        )


# ---------------------------------------------------------------------------
# (c) The case the ruling's literal wording misses
# ---------------------------------------------------------------------------


class TestUpstreamOfPartialIntervention:
    def test_factor_upstream_of_a_partially_intervened_node_is_a_candidate(self):
        """Design §2.1 bullet 4 — differential severing makes slopes differ."""
        row = _row(
            RobustnessAnalyzerV2()
            .analyze(_upstream_partial_intervention_graph())
            .factor_flip_values,
            "fac_up",
        )
        assert row["flip_reason"] == "found", (
            "fac_up is intervened by nobody, so an 'intervened by some but not all "
            "options' rule would have skipped it — the derived slope rule must not"
        )
        assert abs(row["flip_value"] - 0.8) < 1e-6, f"expected 0.8, got {row['flip_value']}"
        assert row["direction"] == "increase"
        assert row["baseline_winner_id"] == "opt_a"
        assert row["alternative_winner_id"] == "opt_c"


# ---------------------------------------------------------------------------
# (d) Band vacuity — the positive control is a MUTANT (trap 12b)
# ---------------------------------------------------------------------------


class TestStabilityBandIsNotVacuous:
    def test_bands_move_across_sampled_backgrounds(self):
        """The band must reflect real background variation, not one repeated point."""
        row = _row(
            RobustnessAnalyzerV2().analyze(_uncertain_edge_graph()).factor_flip_values,
            "fac_lever",
        )
        assert row["flip_reason"] == "found"
        band = row["stability"]
        assert band["n_seeds"] == FLIP_STABILITY_N_SEEDS
        assert len(band["seed_flip_values"]) == FLIP_STABILITY_N_SEEDS
        assert band["n_seeds_flipped"] >= 2, "need >= 2 flipped seeds for a width to mean anything"
        distinct = {v for v in band["seed_flip_values"] if v is not None}
        assert len(distinct) > 1, (
            "every sampled background produced the SAME flip value — the band is "
            "vacuous and would report false stability"
        )
        assert band["band_width"] > 0.0
        assert band["band_min"] <= band["band_median"] <= band["band_max"]

    def test_band_collapses_when_backgrounds_are_pinned_to_expected_values(self, monkeypatch):
        """trap 12b POSITIVE CONTROL — the mutant must be caught.

        Pin every sampled background to the expected-value configuration (the
        exact degeneracy that turns a band into a decorative constant) and prove
        the three assertions above go RED. A band test that cannot fail is the
        guarantee-theatre this programme exists to remove.
        """
        request = _uncertain_edge_graph()
        expected_value_config = {
            (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
        }
        analyzer = RobustnessAnalyzerV2()
        monkeypatch.setattr(
            analyzer,
            "_sample_flip_backgrounds",
            lambda request, master_seed, n_seeds, tag: [
                dict(expected_value_config) for _ in range(n_seeds)
            ],
        )
        rows = analyzer._compute_factor_flip_values(request, SCMEvaluatorV2(request.graph), SEED)
        band = _row(rows, "fac_lever")["stability"]

        distinct = {v for v in band["seed_flip_values"] if v is not None}
        assert len(distinct) == 1, "mutant must collapse the band to a single point"
        assert band["band_width"] == 0.0, "mutant must collapse the width to zero"


# ---------------------------------------------------------------------------
# (e) Winner-capture pin on a flipping EDGE
# ---------------------------------------------------------------------------


class TestEdgeAlternativeWinnerCapture:
    def test_alternative_winner_id_is_present_and_correct_on_a_flipping_edge(self):
        """Design §1.1 gap 1 / §2.2 — the argmax on the flipped side of the final
        bracket was computed at :4727 and discarded.

        Correctness is re-derived INDEPENDENTLY: rebuild the expected-value
        background, set the flipping edge to its reported flip_mean, take the
        argmax with the analyzer's tie-break, and require equality. Asserting
        mere presence would pass for a hard-coded string.
        """
        request = _edge_flip_graph()
        resp = RobustnessAnalyzerV2().analyze(request)
        assert resp.edge_e_values
        flipping = [e for e in resp.edge_e_values if e["from_id"] == "fac_lever"]
        assert len(flipping) == 1
        entry = flipping[0]

        assert entry["flip_direction"] == "decrease"
        assert abs(entry["flip_mean"] - 1.0 / 3.0) < 1e-5, entry["flip_mean"]
        assert entry["baseline_winner_id"] == "opt_a"
        assert entry["alternative_winner_id"] == "opt_b"
        assert entry["alternative_winner_id"] != entry["baseline_winner_id"]

        edge = next(
            e for e in request.graph.edges if e.from_ == "fac_lever" and e.to == "outcome"
        )
        assert entry["alternative_winner_id"] == _independent_winner(
            request,
            edge_overrides={("fac_lever", "outcome"): entry["flip_mean"] * edge.exists_probability},
        )

    def test_alternative_winner_is_the_winner_at_the_flip_point_not_at_the_boundary(self):
        """MUTATION-DRIVEN (M2). The bisection probes the extreme boundary first,
        and on a graph with only one argmax change the boundary winner and the
        flip-point winner coincide — so a capture frozen at the boundary passed
        the test above. On _multi_crossing_edge_graph they DIFFER: the boundary
        (s_L = -1) is led by opt_c, the flip point (s_L = 0.20) by opt_b.
        """
        request = _multi_crossing_edge_graph()
        resp = RobustnessAnalyzerV2().analyze(request)
        assert resp.edge_e_values
        entry = next(e for e in resp.edge_e_values if e["from_id"] == "fac_lever")

        assert entry["flip_direction"] == "decrease"
        assert abs(entry["flip_mean"] - 0.20) < 1e-5, entry["flip_mean"]
        assert entry["baseline_winner_id"] == "opt_a"
        assert entry["alternative_winner_id"] == "opt_b", (
            "reported the winner at the search BOUNDARY (opt_c) instead of at "
            "flip_mean — a user would be told the wrong option takes over"
        )
        # And the boundary really is led by someone else — otherwise this test
        # would be the same weak assertion as the one above (trap 13).
        assert (
            _independent_winner(
                request, edge_overrides={("fac_lever", "outcome"): -1.0}
            )
            == "opt_c"
        )
        assert entry["alternative_winner_id"] == _independent_winner(
            request, edge_overrides={("fac_lever", "outcome"): entry["flip_mean"]}
        )

    def test_baseline_winner_id_is_on_every_edge_entry(self):
        """§3.2's card gate reads the ROW's baseline_winner_id, so every row carries it."""
        resp = RobustnessAnalyzerV2().analyze(_edge_flip_graph())
        assert resp.edge_e_values
        assert all(e["baseline_winner_id"] == "opt_a" for e in resp.edge_e_values)

    def test_unflippable_edge_reports_no_alternative_winner(self):
        """An edge that cannot flip must not invent one — with a presence control."""
        resp = RobustnessAnalyzerV2().analyze(_control_graph())
        assert resp.edge_e_values
        unflippable = [e for e in resp.edge_e_values if e["flip_mean"] == e["current_mean"]]
        assert unflippable, (
            "positive control: the control graph must contain an unflippable edge, "
            "otherwise this absence assertion tests nothing"
        )
        assert all(e["alternative_winner_id"] is None for e in unflippable)
        assert all(e["baseline_winner_id"] == "opt_a" for e in unflippable)

    def test_alternative_winner_reaches_the_v2_wire(self):
        """The field must survive the v1-dict -> EdgeEValueV2 mapping in the API."""
        client = TestClient(app)
        resp = client.post(
            ENDPOINT,
            json=_edge_flip_graph().model_dump(by_alias=True, mode="json"),
            headers=V2_HEADERS,
        )
        assert resp.status_code == 200, resp.text
        entries = resp.json()["robustness"]["edge_e_values"]
        assert entries
        assert any(e.get("alternative_winner_id") == "opt_b" for e in entries)
        assert all(e.get("baseline_winner_id") == "opt_a" for e in entries)


# ---------------------------------------------------------------------------
# (f) Request gate — absent flag leaves the response byte-shape unchanged
# ---------------------------------------------------------------------------


class TestRequestGate:
    def test_flag_defaults_to_false(self):
        request = RobustnessRequestV2(
            **{
                "graph": {
                    "nodes": [_node("fac_lever", "factor", 0.3), _node("outcome", "outcome")],
                    "edges": [_edge("fac_lever", "outcome", 0.5)],
                },
                "options": [{"id": "opt_a", "label": "A", "interventions": {}}],
                "goal_node_id": "outcome",
                "n_samples": N_SAMPLES,
            }
        )
        assert request.include_factor_flips is False

    def test_absent_flag_emits_no_factor_flip_block_anywhere_on_the_wire(self):
        """REGRESSION PIN — a consumer that does not ask sees no factor-flip bytes."""
        payload = _control_graph(include_factor_flips=False).model_dump(by_alias=True, mode="json")
        assert payload["include_factor_flips"] is False
        resp = TestClient(app).post(ENDPOINT, json=payload, headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        assert "factor_flip_values" not in resp.text, (
            "the factor-flip block must be absent (not null, not empty) when unrequested"
        )

    def test_positive_control_the_block_is_visible_when_requested(self):
        """trap 13 — the absence assertion above must be able to see a presence."""
        payload = _control_graph(include_factor_flips=True).model_dump(by_alias=True, mode="json")
        resp = TestClient(app).post(ENDPOINT, json=payload, headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        assert "factor_flip_values" in resp.text
        rows = resp.json()["factor_flip_values"]
        assert {r["factor_id"] for r in rows} == {"fac_lever", "fac_x", FAC_PAD}
        pad = next(r for r in rows if r["factor_id"] == FAC_PAD)
        assert pad["flip_reason"] == "no_effect_within_bounds", (
            "FAC_PAD's slopes DO differ (opt_c pins it, the others do not), so it is a "
            "real candidate — but its only crossing is negative, so the honest answer "
            "is an absence, not the 'structurally_invariant' attestation"
        )
        # WIRE SHAPE, pinned deliberately: the V2 envelope serialises with
        # exclude_none, so a null flip_value is OMITTED rather than emitted as
        # JSON null — identical to the sibling EdgeEValueV2.e_value and to the
        # band_* keys. Consumers must read flip_reason, not key presence.
        assert "flip_value" not in pad
        assert "direction" not in pad
        assert "alternative_winner_id" not in pad
        assert pad["baseline_winner_id"] == "opt_a"
        assert FactorFlipValueV2(**pad).flip_value is None

        found = next(r for r in rows if r["factor_id"] == "fac_lever")
        assert found["flip_reason"] == "found"
        assert abs(found["flip_value"] - 0.8) < 1e-6
        assert found["alternative_winner_id"] == "opt_c"
        for row in rows:
            assert FactorFlipValueV2(**row)

    def test_analyzer_returns_none_when_unrequested(self):
        resp = RobustnessAnalyzerV2().analyze(_control_graph(include_factor_flips=False))
        assert resp.factor_flip_values is None


# ---------------------------------------------------------------------------
# (g) Crossing outside bounds -> honest absence
# ---------------------------------------------------------------------------


class TestHonestAbsence:
    def test_crossing_outside_bounds_yields_no_fabricated_value(self):
        """Slopes differ (a real candidate) but the crossing is at L* = 2.0."""
        row = _row(
            RobustnessAnalyzerV2().analyze(_out_of_bounds_graph()).factor_flip_values, "fac_lever"
        )
        assert row["flip_reason"] == "no_effect_within_bounds", (
            "a candidate whose crossing is out of range is NOT structurally "
            "invariant — conflating the two would destroy the attestation's meaning"
        )
        assert row["flip_value"] is None
        assert row["direction"] is None
        assert row["alternative_winner_id"] is None

    @pytest.mark.parametrize(
        "builder", [_control_graph, _out_of_bounds_graph, _uncertain_edge_graph]
    )
    def test_no_emitted_value_ever_lies_outside_the_domain(self, builder):
        resp = RobustnessAnalyzerV2().analyze(builder())
        assert resp.factor_flip_values, "trap 13: an empty block satisfies this vacuously"
        for row in resp.factor_flip_values or []:
            if row["flip_value"] is not None:
                assert 0.0 <= row["flip_value"] <= 1.0
            for value in (row.get("stability") or {}).get("seed_flip_values", []):
                if value is not None:
                    assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Budget degradation + determinism (design §2.2 / §4.4)
# ---------------------------------------------------------------------------


class TestBudgetDisclosure:
    def test_internal_budget_trip_attaches_nothing_and_discloses(self, monkeypatch):
        """All-or-nothing, mirroring E_VALUES_UNAVAILABLE."""
        monkeypatch.setattr(RobustnessAnalyzerV2, "FACTOR_FLIP_BUDGET_MS", -1)
        resp = RobustnessAnalyzerV2().analyze(_control_graph())
        assert resp.results, "core survives"
        assert resp.edge_e_values, "an unrelated optional phase is untouched"
        assert resp.factor_flip_values is None
        warning = next(
            (w for w in resp.inference_warnings if w.code == "FACTOR_FLIPS_UNAVAILABLE"), None
        )
        assert warning is not None
        assert warning.severity == "warning"
        assert warning.field == "factor_flip_values"
        assert warning.detail["reason"] == "factor_flip_budget_exceeded"
        assert isinstance(warning.detail["elapsed_ms"], (int, float))

    def test_internal_budget_trip_is_caught_before_any_band_runs(self, monkeypatch):
        """MUTATION-DRIVEN (M10). On a graph WITH candidates the band sweep has
        its own deadline check, which caught the trip even with the screen-loop
        guard disabled — so that guard was unproven. _all_invariant_graph has no
        candidates and therefore no band sweep: only the screen-loop guard can
        catch this.
        """
        monkeypatch.setattr(RobustnessAnalyzerV2, "FACTOR_FLIP_BUDGET_MS", -1)
        resp = RobustnessAnalyzerV2().analyze(_all_invariant_graph())
        assert resp.results, "core survives"
        assert resp.factor_flip_values is None
        warning = next(
            (w for w in resp.inference_warnings if w.code == "FACTOR_FLIPS_UNAVAILABLE"), None
        )
        assert warning is not None
        assert warning.detail["reason"] == "factor_flip_budget_exceeded"

    def test_positive_control_the_same_graph_completes_under_the_real_budget(self):
        """trap 13 — the absence above must be able to see a presence."""
        resp = RobustnessAnalyzerV2().analyze(_all_invariant_graph())
        assert resp.factor_flip_values, "the no-candidate graph still emits attested rows"
        assert all(w.code != "FACTOR_FLIPS_UNAVAILABLE" for w in resp.inference_warnings)

    def test_band_sweep_aborts_on_its_own_deadline_rather_than_attaching_a_partial_band(self):
        """MUTATION-DRIVEN (M13). The band sweep carries its OWN deadline check,
        but on every graph reachable through analyze() the screen-loop guard
        fires first and masks it — so the band guard was unproven at the level
        of the whole phase. Driven directly here, with an already-expired
        deadline, so no wall-clock timing is involved.
        """
        request = _uncertain_edge_graph()
        analyzer = RobustnessAnalyzerV2()
        evaluator = SCMEvaluatorV2(request.graph)
        node = next(n for n in request.graph.nodes if n.id == "fac_lever")
        backgrounds = analyzer._sample_flip_backgrounds(
            request, SEED, FLIP_STABILITY_N_SEEDS, "factor_flip_stability"
        )

        # POSITIVE CONTROL FIRST (trap 13): under a real budget the identical
        # call produces a full band, so the None below means "aborted", not
        # "this call never worked".
        band = analyzer._factor_flip_band(
            request, evaluator, node, 0.3, backgrounds, time.monotonic(), 8000.0
        )
        assert band is not None
        assert band["n_seeds"] == FLIP_STABILITY_N_SEEDS

        # A t0 ten seconds in the past against a zero budget: already expired.
        assert (
            analyzer._factor_flip_band(
                request, evaluator, node, 0.3, backgrounds, time.monotonic() - 10.0, 0.0
            )
            is None
        ), "an expired deadline must abort the sweep, not attach a partial band"

    def test_request_budget_exhaustion_discloses_before_entry(self, monkeypatch):
        monkeypatch.setattr(RobustnessAnalyzerV2, "OVERALL_REQUEST_BUDGET_MS", 0)
        resp = RobustnessAnalyzerV2().analyze(_control_graph())
        assert resp.factor_flip_values is None
        warning = next(
            (w for w in resp.inference_warnings if w.code == "FACTOR_FLIPS_UNAVAILABLE"), None
        )
        assert warning is not None
        assert warning.detail["reason"] == "request_budget_exhausted"

    def test_no_disclosure_when_the_phase_was_never_requested(self):
        resp = RobustnessAnalyzerV2().analyze(_control_graph(include_factor_flips=False))
        assert all(w.code != "FACTOR_FLIPS_UNAVAILABLE" for w in resp.inference_warnings)


class TestDeterminism:
    def test_same_request_and_seed_yields_an_identical_block(self):
        first = RobustnessAnalyzerV2().analyze(_uncertain_edge_graph()).factor_flip_values
        second = RobustnessAnalyzerV2().analyze(_uncertain_edge_graph()).factor_flip_values
        assert first, "trap 13: two absent blocks compare equal — prove a presence first"
        assert any(r.get("stability") for r in first), "bands must be part of what is pinned"
        assert first == second

    def test_factor_band_uses_a_child_seed_tag_distinct_from_the_edge_band(self, monkeypatch):
        """§2.2 — a distinct tag, so the factor sweep never consumes the edge stream.

        DERIVED, not mirrored: the tags are captured from the LIVE call sites
        during a full analyze(), so reusing the edge tag in the factor phase
        collapses the set to one element and turns this RED.
        """
        analyzer = RobustnessAnalyzerV2()
        seen: List[str] = []
        real = RobustnessAnalyzerV2._sample_flip_backgrounds

        def spy(self, request, master_seed, n_seeds, tag):
            seen.append(tag)
            return real(self, request, master_seed, n_seeds, tag)

        monkeypatch.setattr(RobustnessAnalyzerV2, "_sample_flip_backgrounds", spy)
        analyzer.analyze(_uncertain_edge_graph())

        assert len(set(seen)) == 2, f"expected two distinct child-seed tags, saw {sorted(set(seen))}"


class TestCandidateCap:
    def test_candidates_beyond_the_cap_are_emitted_not_silently_dropped(self, monkeypatch):
        """§2.2's cap must not become a silent skip — the diagnosis' whole point."""
        monkeypatch.setattr(RobustnessAnalyzerV2, "FACTOR_FLIP_MAX_CANDIDATES", 1)
        rows = RobustnessAnalyzerV2().analyze(_out_of_bounds_graph()).factor_flip_values
        assert rows is not None
        reasons = {r["factor_id"]: r["flip_reason"] for r in rows}
        assert set(reasons) == {"fac_lever", "fac_boost", FAC_PAD}, (
            "every eligible factor keeps a row; the cap changes what a row SAYS, "
            "never whether it exists"
        )
        capped = [r for r in rows if r["flip_reason"] == "candidate_cap_exceeded"]
        evaluated = [r for r in rows if r["flip_reason"] != "candidate_cap_exceeded"]
        # All three factors are candidates here (each has one option pinning it and
        # one not), so a cap of 1 must evaluate exactly one and disclose two.
        assert len(evaluated) == 1, [r["factor_id"] for r in evaluated]
        assert len(capped) == 2, [r["factor_id"] for r in capped]
        for row in capped:
            assert row["flip_value"] is None
            assert row["direction"] is None
            assert "stability" not in row, "an unevaluated candidate has no band to show"


@pytest.mark.parametrize(
    "builder",
    [_control_graph, _all_invariant_graph, _out_of_bounds_graph, _uncertain_edge_graph],
)
def test_every_eligible_root_factor_gets_exactly_one_row(builder):
    """Never silently skipped: one attested row per eligible root factor."""
    request = builder()
    resp = RobustnessAnalyzerV2().analyze(request)
    has_parent = {e.to for e in request.graph.edges}
    eligible = {
        n.id
        for n in request.graph.nodes
        if n.id != request.goal_node_id
        and n.id not in has_parent
        and n.observed_state is not None
        and n.observed_state.value is not None
    }
    assert {r["factor_id"] for r in resp.factor_flip_values or []} == eligible
