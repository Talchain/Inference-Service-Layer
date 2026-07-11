"""Analytic flip-threshold derivation — Track S Phase 1 SPIKE.

STATUS: experimental. NOT wired into any production path — nothing under
src/ imports this module (pinned by
tests/unit/test_analytic_flip.py::TestImportInertness). Consumers are the
spike benchmark (benchmarks/analytic_flip_agreement.py) and its tests.

Why this is possible
--------------------
The shipped flip thresholds (``edge_e_values[].flip_mean`` from
``RobustnessAnalyzerV2._compute_edge_e_values``, and the per-seed values
inside the #71 stability bands from ``_flip_mean_under_background``) are
found by boundary-check + 20-step bisection on the outcome function —
roughly 22 full option-sweep evaluations per edge per direction. But both
searches run on the DETERMINISTIC ``SCMEvaluatorV2`` (``analyze()`` sets
``evaluator._epsilon_rng = None`` before every structural analysis), whose
structural equations are linear-additive:

    node_value = base + intercept + sum(parent_value * edge_strength)

Vary ONE edge's ``strength.mean`` m (effective strength ``m * ep``) while
every other edge is held fixed (expected values on the base path; a sampled
background in the band sweep) and each option's goal value is an EXACTLY
affine function of m: the perturbed strength enters the recursion exactly
once, upstream values cannot depend on it (DAG), and everything downstream
is linear in the perturbed node's value. Consequences:

- each pairwise winner-margin is affine in m => at most one crossing;
- the region where the baseline winner stays the winner is an interval
  (intersection of half-lines), so the flip point in each direction is a
  single closed-form line crossing — bisection approximates a number that
  has an exact formula.

Honesty boundary (matters more than coverage)
---------------------------------------------
The closed form is INVALID — and this module must refuse loudly rather than
return a plausible number — when the outcome function is not affine in m:

1. Epsilon noise: an evaluator holding an ``_epsilon_rng`` while some node
   has ``epsilon_std > 0`` adds per-node noise and clamps node values to
   [0, 1] — stochastic AND piecewise: rejected structurally
   (``AnalyticInvalidityError``) before any arithmetic. (An epsilon RNG with
   all ``epsilon_std == 0`` can never fire and stays exactly affine.)
2. Any future non-linear structural equation (sigmoid/threshold/saturating
   nodes): caught at runtime by a three-point affinity tripwire — for each
   option the outcome at m = -1 must equal the affine prediction from
   m = 0 and m = +1 within ``AFFINITY_TOLERANCE``. Three collinear points do
   not PROVE affinity in general; the proof is the structural argument
   above, which holds for today's evaluator by construction. The tripwire
   exists so that a future evaluator change fails loud here instead of
   silently disagreeing with the MC search.
3. Internal inconsistency (boundary flips but no in-range crossing exists —
   impossible for an affine system): also raises, never guesses.

Semantics mirrored from the MC search (deliberately identical):
- winner tie-break: sorted by (-value, option_id);
- boundary pre-check per direction using evaluator-computed outcomes (so
  tie-breaks at the [-1, 1] boundary are decided by the very same floats
  the MC check sees);
- "increase" direction tried first, first flipping direction wins;
- bidirected edges skipped; same output keys, rounding and e_value floor.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.models.robustness_v2 import EdgeV2, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    EDGE_STRENGTH_MAX,
    EDGE_STRENGTH_MIN,
    SCMEvaluatorV2,
)

logger = logging.getLogger(__name__)

# Absolute tolerance for the three-point affinity tripwire, scaled by the
# magnitude of the values involved. Node values are O(1); accumulated float
# error over a production-sized DAG is ~1e-15 relative, real non-linearities
# are orders of magnitude above 1e-9.
AFFINITY_TOLERANCE = 1e-9

EdgeKey = Tuple[str, str]
EdgeConfig = Dict[EdgeKey, float]


class AnalyticInvalidityError(ValueError):
    """The analytic closed form does not apply — caller must fall back to MC.

    Raised instead of returning a number so an invalid structure can never
    masquerade as a computed flip threshold.
    """


def _winner(outcomes: Dict[str, float]) -> str:
    """Deterministic winner — identical tie-break to the MC search."""
    return sorted(outcomes.items(), key=lambda x: (-x[1], x[0]))[0][0]


def _evaluate_all_options(
    request: RobustnessRequestV2,
    evaluator: SCMEvaluatorV2,
    config: EdgeConfig,
) -> Dict[str, float]:
    return {
        option.id: evaluator.evaluate(
            edge_strengths=config,
            interventions=option.interventions,
            goal_node=request.goal_node_id,
        )
        for option in request.options
    }


def _guard_structural_validity(evaluator: SCMEvaluatorV2) -> None:
    """Reject evaluators whose outcomes are not deterministic-affine.

    Epsilon noise both randomises the outcome and clamps node values to
    [0, 1] (piecewise) — the closed form is meaningless there. The guard
    only trips when the noise can actually fire: an RNG is present AND some
    node carries epsilon_std > 0.
    """
    if evaluator._epsilon_rng is not None and any(
        node.epsilon_std > 0 for node in evaluator.graph.nodes
    ):
        raise AnalyticInvalidityError(
            "Evaluator carries an epsilon RNG and the graph has nodes with "
            "epsilon_std > 0: outcomes are stochastic and clamped, not affine "
            "in an edge strength. Use the MC bisection search instead. "
            "(The production flip search runs after analyze() sets "
            "evaluator._epsilon_rng = None, where the closed form is exact.)"
        )


def _affine_coefficients(
    request: RobustnessRequestV2,
    evaluator: SCMEvaluatorV2,
    edge_key: EdgeKey,
    exists_probability: float,
    background: EdgeConfig,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float], Dict[str, float]]:
    """Exact affine coefficients of each option's outcome in the edge's mean m.

    Uses the evaluator itself as a two-point oracle: with effective strength
    m * ep, outcome(m) = A + B*m where A = outcome(m=0) and
    B = outcome(m=1) - outcome(m=0). A third evaluation at m = -1 feeds the
    affinity tripwire AND doubles as the decrease-direction boundary check.

    Returns (coefficients {option_id: (A, B)}, outcomes at m=+1, outcomes at
    m=-1); the boundary outcome dicts are the same floats the MC boundary
    check would compute, so winner tie-breaks stay bit-identical.
    """
    config_zero = dict(background)
    config_zero[edge_key] = 0.0
    config_hi = dict(background)
    config_hi[edge_key] = EDGE_STRENGTH_MAX * exists_probability
    config_lo = dict(background)
    config_lo[edge_key] = EDGE_STRENGTH_MIN * exists_probability

    outcomes_zero = _evaluate_all_options(request, evaluator, config_zero)
    outcomes_hi = _evaluate_all_options(request, evaluator, config_hi)
    outcomes_lo = _evaluate_all_options(request, evaluator, config_lo)

    coefficients: Dict[str, Tuple[float, float]] = {}
    for option in request.options:
        intercept = outcomes_zero[option.id]
        slope = outcomes_hi[option.id] - intercept
        predicted_lo = intercept - slope
        scale = max(1.0, abs(intercept), abs(outcomes_hi[option.id]), abs(outcomes_lo[option.id]))
        if abs(outcomes_lo[option.id] - predicted_lo) > AFFINITY_TOLERANCE * scale:
            raise AnalyticInvalidityError(
                f"Affinity tripwire: option '{option.id}' outcome at m=-1 is "
                f"{outcomes_lo[option.id]!r} but the affine prediction from "
                f"m=0 and m=+1 is {predicted_lo!r} (edge {edge_key[0]}->"
                f"{edge_key[1]}). The outcome function is not affine in this "
                "edge's strength — the structural equations are no longer "
                "linear-additive. Use the MC bisection search instead."
            )
        coefficients[option.id] = (intercept, slope)

    return coefficients, outcomes_hi, outcomes_lo


def analytic_flip_search(
    request: RobustnessRequestV2,
    evaluator: SCMEvaluatorV2,
    edge: EdgeV2,
    background: EdgeConfig,
) -> Optional[Tuple[float, str]]:
    """Closed-form counterpart of the MC boundary-check + bisection.

    ``background`` holds every edge at the value the search runs against
    (expected values on the base path; a sampled configuration in the band
    sweep); this edge's own entry is overwritten internally.

    Returns (flip_mean, direction) with direction in {"increase",
    "decrease"} — "increase" tried first exactly like the MC search — or
    None when no perturbation of this edge's mean within
    [EDGE_STRENGTH_MIN, EDGE_STRENGTH_MAX] flips the winner.

    Raises AnalyticInvalidityError where the closed form does not apply
    (see module docstring — the honesty boundary).
    """
    _guard_structural_validity(evaluator)

    edge_key: EdgeKey = (edge.from_, edge.to)
    current_mean = edge.strength.mean
    exists_probability = edge.exists_probability

    baseline_config = dict(background)
    baseline_config[edge_key] = current_mean * exists_probability
    baseline_outcomes = _evaluate_all_options(request, evaluator, baseline_config)
    baseline_winner = _winner(baseline_outcomes)

    coefficients, outcomes_hi, outcomes_lo = _affine_coefficients(
        request, evaluator, edge_key, exists_probability, background
    )
    winner_intercept, winner_slope = coefficients[baseline_winner]

    for direction in ("increase", "decrease"):
        # Boundary pre-check — same evaluator floats and tie-break the MC
        # search uses, so "this direction cannot flip" agrees bit-for-bit.
        boundary_outcomes = outcomes_hi if direction == "increase" else outcomes_lo
        if _winner(boundary_outcomes) == baseline_winner:
            continue

        candidates: List[float] = []
        for option in request.options:
            if option.id == baseline_winner:
                continue
            rival_intercept, rival_slope = coefficients[option.id]
            if rival_slope == winner_slope:
                continue  # parallel margins — no crossing
            crossing = (winner_intercept - rival_intercept) / (rival_slope - winner_slope)
            if direction == "increase":
                # Rival overtakes as m grows only if it gains on the winner.
                if rival_slope > winner_slope and current_mean <= crossing <= EDGE_STRENGTH_MAX:
                    candidates.append(crossing)
            else:
                if rival_slope < winner_slope and EDGE_STRENGTH_MIN <= crossing <= current_mean:
                    candidates.append(crossing)

        if not candidates:
            # The boundary flipped, so an affine system MUST have an in-range
            # crossing; reaching here means the affine model and the
            # evaluator disagree. Refuse rather than guess.
            raise AnalyticInvalidityError(
                f"Internal inconsistency on edge {edge_key[0]}->{edge_key[1]} "
                f"({direction}): the {direction} boundary flips the winner "
                "but no affine crossing lies in range. The outcome function "
                "is not behaving affinely — use the MC bisection search."
            )

        # Nearest crossing to current_mean = boundary of the winner's
        # interval = the flip point the bisection converges to.
        flip_mean = min(candidates) if direction == "increase" else max(candidates)
        return flip_mean, direction

    return None


def analytic_flip_mean_under_background(
    request: RobustnessRequestV2,
    evaluator: SCMEvaluatorV2,
    edge: EdgeV2,
    background: EdgeConfig,
) -> Optional[float]:
    """Closed-form counterpart of ``_flip_mean_under_background`` (#71 bands).

    Same contract: the flip mean of one edge with the other edges held at a
    sampled background, or None when that background admits no flip within
    [EDGE_STRENGTH_MIN, EDGE_STRENGTH_MAX].
    """
    found = analytic_flip_search(request, evaluator, edge, background)
    return found[0] if found is not None else None


def analytic_edge_e_values(
    request: RobustnessRequestV2,
    evaluator: SCMEvaluatorV2,
) -> List[Dict[str, Any]]:
    """Closed-form counterpart of ``_compute_edge_e_values``.

    Entry-for-entry the same output contract: same keys, same rounding
    (flip_mean to 6 dp, e_value to 4 dp), same e_value >= 1.0 floor, same
    near-zero current_mean handling, same no-flip sentinel (e_value inf,
    flip_mean == current_mean, direction "increase"), same bidirected-edge
    skip. No wall-clock budget: the analytic path needs 4 option-sweeps per
    edge instead of up to ~44 for the bisection.

    Raises AnalyticInvalidityError where the closed form does not apply.
    """
    baseline_config: EdgeConfig = {
        (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
    }

    results: List[Dict[str, Any]] = []
    for edge in request.graph.edges:
        if getattr(edge, "edge_type", None) == "bidirected":
            continue

        current_mean = edge.strength.mean
        found = analytic_flip_search(request, evaluator, edge, baseline_config)

        if found is None:
            results.append(
                {
                    "edge_id": f"{edge.from_}->{edge.to}",
                    "from_id": edge.from_,
                    "to_id": edge.to,
                    "e_value": float("inf"),
                    "flip_direction": "increase",
                    "current_mean": current_mean,
                    "flip_mean": current_mean,
                }
            )
            continue

        flip_mean, direction = found
        if abs(current_mean) > 1e-10:
            e_value = abs(flip_mean / current_mean)
        else:
            # current_mean ~ 0 — any nonzero flip_mean is infinite leverage
            e_value = abs(flip_mean) / 1e-6 if abs(flip_mean) > 1e-10 else 1.0
        e_value = max(1.0, e_value)

        results.append(
            {
                "edge_id": f"{edge.from_}->{edge.to}",
                "from_id": edge.from_,
                "to_id": edge.to,
                "e_value": round(e_value, 4),
                "flip_direction": direction,
                "current_mean": current_mean,
                "flip_mean": round(flip_mean, 6),
            }
        )

    return results
