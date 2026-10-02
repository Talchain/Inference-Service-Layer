"""
Decision-level flip threshold (SCIENCE ROBUSTNESS step 1; EXPERIMENT. SCIENCE/DSK, programme-docs #85 5948121821).

THE QUESTION a user can be told: "the recommendation would change if <link> were weaker than X". The recommendation is
the analyser's ``recommended_option_id``. That is a statement about the WHOLE Monte Carlo, with every other link's
stated uncertainty integrated over.

WHY NOT TODAY'S ``flip_mean``. ``_compute_edge_e_values`` searches the flip point in ONE world: every other link held
at ``mean × exists_probability``, one deterministic evaluation per option. It is stable (identical across seeds), but it
answers a different question. On D1 it says capacity→availability can fall to 0.025 before the lead changes, while the
recommendation itself changes at ≈0.05–0.075 (#85 5948121821). Its stability band samples 10 single worlds per run, so
the band moves from run to run and still does not describe the decision.

THE ESTIMATOR. ``leader_at(x)`` runs the analyser's own Monte Carlo with the link's central value set to ``x`` and the
request seed unchanged. CORRECTION (2 Oct, step 2): that seed does NOT give every probe the same random numbers. A
tied draw is broken from the shared edge stream, so moving the mean desynchronised later draws (D1: from draw 55).
Step 1's 0-miss results were measured, not derived, and stand; the fast path below restores exact CRN in capture
mode. The search is in two stages:
- a coarse grid from the current value towards the bound finds the NEAREST change of recommendation, which guards
  against a curve that changes more than once;
- bisection then narrows that bracket to ``tol``.
The answer is a bracket ``[hold, flip]`` (the recommendation holds at ``hold`` and has changed at ``flip``) and its
midpoint, or an honest "no change between here and the bound".
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

LeaderAt = Callable[[float], str]


def decision_flip_threshold(
    leader_at: LeaderAt,
    current: float,
    bound: float,
    tol: float = 0.0025,
    grid: int = 8,
    max_bisect: int = 30,
) -> Dict[str, object]:
    """Nearest value between ``current`` and ``bound`` at which ``leader_at`` stops returning today's leader."""
    if grid < 1 or tol <= 0:
        raise ValueError("DECISION_FLIP_BAD_PARAMS")
    calls = 0

    def probe(x: float) -> str:
        nonlocal calls
        calls += 1
        return leader_at(x)

    lead = probe(current)
    hold: float = current
    flip: Optional[float] = None
    to: Optional[str] = None
    for i in range(1, grid + 1):
        x = current + (bound - current) * i / grid
        who = probe(x)
        if who != lead:
            flip, to = x, who
            break
        hold = x
    if flip is None:
        return {"exists": False, "leader": lead, "threshold": None, "bracket": None, "to_option_id": None, "evaluations": calls}

    steps = 0
    while abs(flip - hold) > tol and steps < max_bisect:
        mid = (hold + flip) / 2
        who = probe(mid)
        if who == lead:
            hold = mid
        else:
            flip, to = mid, who
        steps += 1
    bracket: List[float] = sorted([hold, flip])
    return {
        "exists": True,
        "leader": lead,
        "threshold": (hold + flip) / 2,
        "bracket": bracket,
        "to_option_id": to,
        "evaluations": calls,
    }


# ── Step 2: the affine fast path (#85 lease 5948579361; DL guard 2 Oct) ─────────────────────────────────────────────
#
# WHY IT IS EXACT. ISL's evaluator is linear in every edge strength (a node is Σ parent · strength + intercept, a DAG
# uses each edge at most once on a path), and on the SAME draws (CRN) the sampled strength of the moved link is
# `exists · (mean + std · z)`. So each draw's goal value, for every option and for the status-quo reading, is AFFINE
# in the link's mean: two Monte Carlo runs (mean = current, mean = 0) give it for every mean in between.
#
# WHERE IT IS NOT. A node with epsilon_std > 0 is clamped to [0, 1] after its noise, and an evaluated PRODUCT identity
# multiplies its operands. Either DOWNSTREAM of the link makes a draw's value piecewise or quadratic in the mean, and a
# crossing can then hide where the straight line says "no change". Such a link never takes this path: it is an honest
# absence (`downstream_nonlinearity`). Every quoted point is then re-run for real and must reproduce the predicted win
# shares (`affine_check_failed` otherwise).

import hashlib  # noqa: E402
import math  # noqa: E402
from collections import defaultdict  # noqa: E402
from typing import Any, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

AFFINE_METHOD = "affine_crn_replicates_v1"
BOUND_ABS = 0.02
BOUND_REL = 0.15
GRID_STEP = 0.0025
CHECK_TOL = 1e-9
STRIPPED = {
    "include_e_values": False,
    "include_voi": False,
    "include_factor_flips": False,
    "include_path_decomposition": False,
    "analysis_types": ["comparison"],
}


def downstream_nonlinearity(request: Any, from_id: str, to_id: str) -> Optional[str]:
    """Why the goal is NOT affine in this link's strength, or None when it is. Static: read off the graph."""
    from src.services.robustness_analyzer_v2 import _resolve_structural_identity_plans

    children: Dict[str, List[str]] = defaultdict(list)
    for e in request.graph.edges:
        if getattr(e, "edge_type", None) == "bidirected":
            continue
        children[e.from_].append(e.to)
    cone: set = set()
    stack = [to_id]
    while stack:
        n = stack.pop()
        if n in cone:
            continue
        cone.add(n)
        stack.extend(children.get(n, []))
    nodes = {n.id: n for n in request.graph.nodes}
    for nid in sorted(cone):
        node = nodes.get(nid)
        if node is not None and (getattr(node, "epsilon_std", 0.0) or 0.0) > 0:
            return f"clamp:{nid}"
    for nid, plan in sorted(_resolve_structural_identity_plans(request.graph).items()):
        if nid in cone and plan.evaluated and plan.operation != "sum":
            return f"identity:{nid}"
    return None


def p_best(values: np.ndarray, sense: str, n_samples: int) -> np.ndarray:
    """P(best) per option for (options × draws) values: ISL's winner rule for maximise / minimise, vectorised.

    A draw credits only its finite options; the best value wins and an exact tie splits the draw; a draw with no finite
    option credits nobody. Pinned against the canonical ``_winners_for_draw`` (tests/unit/test_decision_flip.py).
    """
    finite = np.isfinite(values)
    if sense == "maximise":
        v = np.where(finite, values, -np.inf)
        best = v.max(axis=0)
    elif sense == "minimise":
        v = np.where(finite, values, np.inf)
        best = v.min(axis=0)
    else:
        raise ValueError(f"DECISION_FLIP_SENSE_UNSUPPORTED: {sense}")
    winners = finite & (v == best) & finite.any(axis=0)
    counts = winners.sum(axis=0)
    share = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    out: np.ndarray = (winners * share).sum(axis=1) / n_samples
    return out


def _leader(p: np.ndarray, option_ids: Sequence[str]) -> str:
    # ISL's recommendation rule: max(option_wins, key=...) → the FIRST option holding the maximum.
    return option_ids[int(np.argmax(p))]


def affine_threshold(
    x0: np.ndarray, x1: np.ndarray, current: float, option_ids: Sequence[str], lead: str, sense: str,
    n_samples: int, step: float = GRID_STEP,
) -> Dict[str, Any]:
    """Nearest mean between ``current`` and 0 where the recommendation stops being ``lead``, on a ``step`` grid.

    ``x0`` / ``x1``: (options × draws) pre-noise outcomes with the link's mean at ``current`` / at 0, same draws.
    """
    n = max(1, int(math.ceil(abs(current) / step)))
    hold = current
    for k in range(1, n + 1):
        x = current - current * k / n
        t = (current - x) / current
        who = _leader(p_best(x0 + t * (x1 - x0), sense, n_samples), option_ids)
        if who != lead:
            return {"exists": True, "hold": hold, "flip": x, "to_option_id": who}
        hold = x
    return {"exists": False, "hold": None, "flip": None, "to_option_id": None}


def _child_seed(master: int, i: int) -> int:
    return int(hashlib.sha256(f"{master}:decision_flip:{i}".encode()).hexdigest()[:8], 16)


def compute_decision_flip_block(dreq: Any) -> Any:
    """The on-demand block: per link quoted / absent / no_change, under the K-replicate licence."""
    from src.models.robustness_v2 import DecisionFlipBlockV2, DecisionFlipLinkV2, RobustnessRequestV2
    from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2, compute_effective_seed

    base = dreq.request.model_copy(update=STRIPPED, deep=True)
    edges = {(e.from_, e.to): e for e in base.graph.edges}
    for ref in dreq.links:
        if (ref.from_id, ref.to_id) not in edges:
            raise ValueError(f"DECISION_FLIP_UNKNOWN_LINK: {ref.from_id}->{ref.to_id}")
    option_ids = [o.id for o in base.options]
    master, _ = compute_effective_seed(base)

    def run(seed: int, link: Optional[Tuple[str, str]] = None, mean: Optional[float] = None, capture: bool = True) -> Any:
        q = base.model_copy(update={"seed": str(seed)}, deep=True)
        if link is not None:
            for e in q.graph.edges:
                if (e.from_, e.to) == link:
                    e.strength.mean = mean
        q = RobustnessRequestV2.model_validate(q.model_dump(by_alias=True))
        q._capture_draws = capture
        return RobustnessAnalyzerV2().analyze(q)

    def matrix(resp: Any) -> np.ndarray:
        oo = resp._mc_draws["option_outcomes"]
        return np.array([oo[o] for o in option_ids], dtype=float)

    head = run(master, capture=False)
    leader = head.recommended_option_id
    links_out: List[Any] = []
    sense: str = "maximise"
    seeds = [_child_seed(master, i) for i in range(dreq.replicates)]
    replicate_base: Dict[int, Any] = {}
    for s in seeds:
        r = run(s)
        replicate_base[s] = r
        sense = str(r._mc_draws["objective"].sense) if r._mc_draws["objective"] is not None else "maximise"
    unstable = any(r.recommended_option_id != leader for r in replicate_base.values())

    for ref in dreq.links:
        link = (ref.from_id, ref.to_id)
        current = float(edges[link].strength.mean)
        common = {"from_id": ref.from_id, "to_id": ref.to_id, "current_mean": current}
        why = None
        if leader is None or sense not in ("maximise", "minimise"):
            why = "ranking_not_supported"
        elif unstable:
            why = "leader_unstable"
        elif current == 0:
            why = "link_at_zero"
        else:
            why = downstream_nonlinearity(base, *link)
            why = f"nonlinear_downstream:{why}" if why else None
        if why is not None:
            links_out.append(DecisionFlipLinkV2(status="absent", reason=why, **common))
            continue

        per_seed: List[Dict[str, Any]] = []
        for s in seeds:
            x0 = matrix(replicate_base[s])
            # The capture must reproduce the analyser's own win shares before the line is trusted.
            wp = {o.option_id: o.win_probability for o in replicate_base[s].results}
            if not np.allclose(p_best(x0, sense, base.n_samples), [wp[o] for o in option_ids], atol=CHECK_TOL, rtol=0):
                raise RuntimeError("DECISION_FLIP_CAPTURE_MISMATCH")
            x1 = matrix(run(s, link, 0.0))
            res = affine_threshold(x0, x1, current, option_ids, leader, sense, base.n_samples)
            res["seed"] = s
            res["x1"] = x1
            per_seed.append(res)

        found = [r for r in per_seed if r["exists"]]
        ths = [((r["hold"] + r["flip"]) / 2) if r["exists"] else None for r in per_seed]
        if not found:
            links_out.append(DecisionFlipLinkV2(status="no_change", replicate_thresholds=ths, **common))
            continue
        if len(found) < len(per_seed):
            links_out.append(DecisionFlipLinkV2(status="absent", reason="replicates_disagree", replicate_thresholds=ths, **common))
            continue
        if len({r["to_option_id"] for r in found}) > 1:
            links_out.append(DecisionFlipLinkV2(status="absent", reason="replicates_disagree_on_option", replicate_thresholds=ths, **common))
            continue
        vals = sorted(t for t in ths if t is not None)
        median = float(np.median(vals))
        spread = vals[-1] - vals[0]
        if spread > BOUND_ABS or spread > BOUND_REL * abs(median):
            links_out.append(DecisionFlipLinkV2(status="absent", reason="replicates_spread", replicate_thresholds=ths,
                                                replicate_range=spread, **common))
            continue

        # EXACT RE-RUN at the quoted point: the replicate nearest the median, both sides of its bracket, for real.
        near = min(found, key=lambda r: abs((r["hold"] + r["flip"]) / 2 - median))
        x0 = matrix(replicate_base[near["seed"]])
        x1 = near["x1"]
        ok = True
        for x, expect in ((near["flip"], near["to_option_id"]), (near["hold"], leader)):
            real = run(near["seed"], link, x)  # capture mode: the SAME draws the prediction was read from
            t = (current - x) / current
            predicted = p_best(x0 + t * (x1 - x0), sense, base.n_samples)
            actual = np.array([{o.option_id: o.win_probability for o in real.results}[o] for o in option_ids])
            if real.recommended_option_id != expect or not np.allclose(predicted, actual, atol=CHECK_TOL, rtol=0):
                ok = False
        if not ok:
            links_out.append(DecisionFlipLinkV2(status="absent", reason="affine_check_failed", replicate_thresholds=ths,
                                                replicate_range=spread, **common))
            continue
        links_out.append(DecisionFlipLinkV2(status="quoted", threshold=median, replicate_thresholds=ths,
                                            replicate_range=spread, to_option_id=found[0]["to_option_id"], **common))

    return DecisionFlipBlockV2(method=AFFINE_METHOD, leader_option_id=leader, replicates=dreq.replicates,
                               bound_abs=BOUND_ABS, bound_rel=BOUND_REL, grid_step=GRID_STEP, links=links_out)
