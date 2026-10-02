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

THE ESTIMATOR. ``leader_at(x)`` runs the analyser's own Monte Carlo with the link's central value set to ``x``. The
request seed is unchanged, so every probe sees the same random numbers (ISL's CRN invariant, #218), and the
recommendation as a function of ``x`` is free of between-probe sampling noise. The search is in two stages:
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
