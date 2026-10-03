"""
"WHEN does it pay off?" — the horizon view (Science 2.0 v0 EXPERIMENT; SCIENCE/DSK, programme-docs #85 5947449217).

Today's engine answers "which option is best once its effect is fully realised". It has no time: an option that
pays off in week 2 and one that pays off in week 12 are ranked as if both were already in force. The horizon view
asks the same question at each week up to the user's horizon.

THE MODEL (closed form, no new sampling). Each option ``o`` has an onset ``d_o`` (weeks before any effect) and a
ramp ``r_o`` (weeks from the first effect to the full effect). The share of the option's full effect realised in
week ``u`` (1-based, counted from the decision) is

    f_o(u) = clamp((u - d_o) / r_o, 0, 1)      for r_o > 0
    f_o(u) = 1 if u > d_o else 0               for r_o = 0

and the option's goal value in draw ``s`` at week ``u`` is the draw's no-intervention world moved by that share of
the option's full effect in the SAME draw:

    v_o(u, s) = sq(s) + f_o(u) * (x_o(s) - sq(s))

``x_o`` is the analyser's pre-noise, common-random-numbers outcome (the population ``win_probability`` is read
from) and ``sq`` is the same draw's status-quo reference (no interventions). ``evaluate="at"`` ranks ``v`` in week
``t``; ``evaluate="cumulative"`` ranks the mean of ``v`` over weeks ``1..t``, i.e. ``f`` is replaced by
``F_o(t) = mean(f_o(1..t))`` (what the option has delivered over the period, not where it stands at its end).

THE ONE WINNER RULE. Every draw at every week is decided by the analyser's canonical owner
``_winners_for_draw`` (passed in as ``winners_for_draw``), never by a local ``max``: a minimise or target
objective ranks the horizon exactly as it ranks today's answer. Ties split the draw equally, as the main loop
does; no RNG is drawn here, so nothing downstream can move.

THE IDENTITY. With every onset 0 and ramp 0, ``f = 1`` in every week, ``v`` is ``x`` itself (taken verbatim, not
recomputed through ``sq``), and P(best) in every week equals today's ``win_probability`` exactly. A view that
cannot reproduce today's answer in that case is not a view of today's model.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

METHOD = "closed_form_onset_ramp_v0"

# (finite outcomes at this draw and week, this draw's status-quo reference) → the winning option ids.
WinnersForDraw = Callable[[Dict[str, float], Optional[float]], List[str]]


def realised_fraction(week: int, onset_weeks: float, ramp_weeks: float) -> float:
    """Share of the option's full effect realised in ``week`` (1-based)."""
    if ramp_weeks <= 0:
        return 1.0 if week > onset_weeks else 0.0
    return min(1.0, max(0.0, (week - onset_weeks) / ramp_weeks))


def fraction_for(week: int, onset_weeks: float, ramp_weeks: float, evaluate: str) -> float:
    if evaluate == "at":
        return realised_fraction(week, onset_weeks, ramp_weeks)
    if evaluate == "cumulative":
        total = sum(realised_fraction(u, onset_weeks, ramp_weeks) for u in range(1, week + 1))
        return total / week
    raise ValueError(f"HORIZON_EVALUATE_UNKNOWN: {evaluate!r}")


def _value(x: float, sq: Optional[float], f: float) -> float:
    # f == 1 returns the analyser's own outcome verbatim: sq + 1 * (x - sq) is not bitwise x in floating point,
    # and the identity (all onsets 0 → today's win_probability exactly) rests on it.
    if f >= 1.0:
        return x
    if sq is None or not math.isfinite(sq):
        return math.nan
    if f <= 0.0:
        return sq
    return sq + f * (x - sq)


def compute_horizon_view(
    option_ids: Sequence[str],
    option_outcomes: Mapping[str, Sequence[float]],
    status_quo: Sequence[float],
    onsets: Mapping[str, Tuple[float, float]],
    horizon_weeks: int,
    evaluate: str,
    n_samples: int,
    winners_for_draw: WinnersForDraw,
) -> Dict[str, object]:
    """P(best) per option at each week 1..horizon_weeks, from the analyser's own draws.

    ``onsets`` maps option id → (onset_weeks, ramp_weeks); an option it omits is in force from week 1 (0, 0),
    which is today's assumption for every option. ``n_samples`` is the analyser's denominator for
    ``win_probability`` (uninformative draws credit nobody, so the shares sum to the informative fraction).
    """
    unknown = sorted(set(onsets) - set(option_ids))
    if unknown:
        raise ValueError(f"HORIZON_ONSET_UNKNOWN_OPTION: {unknown}")
    if horizon_weeks < 1:
        raise ValueError("HORIZON_WEEKS_INVALID")
    n_draws = len(status_quo)
    for oid in option_ids:
        if len(option_outcomes[oid]) != n_draws:
            raise ValueError("HORIZON_DRAWS_MISALIGNED")

    applied = {oid: onsets.get(oid, (0.0, 0.0)) for oid in option_ids}
    checkpoints: List[Dict[str, object]] = []
    for week in range(1, horizon_weeks + 1):
        frac = {oid: fraction_for(week, d, r, evaluate) for oid, (d, r) in applied.items()}
        wins = {oid: 0.0 for oid in option_ids}
        for s in range(n_draws):
            sq = status_quo[s]
            finite: Dict[str, float] = {}
            for oid in option_ids:
                v = _value(option_outcomes[oid][s], sq, frac[oid])
                if math.isfinite(v):
                    finite[oid] = v
            winners = winners_for_draw(finite, sq) if finite else []
            if len(winners) == 1:
                wins[winners[0]] += 1.0
            elif winners:
                share = 1.0 / len(winners)
                for w in winners:
                    wins[w] += share
        p_best = {oid: wins[oid] / n_samples for oid in option_ids}
        top = max(p_best.values())
        leaders = [oid for oid, p in p_best.items() if p == top]
        checkpoints.append({
            "week": week,
            "p_best": p_best,
            "leader_option_id": leaders[0] if len(leaders) == 1 and top > 0 else None,
        })

    flips: List[Dict[str, object]] = []
    for prev, cur in zip(checkpoints, checkpoints[1:]):
        if prev["leader_option_id"] != cur["leader_option_id"]:
            flips.append({
                "week": cur["week"],
                "from_option_id": prev["leader_option_id"],
                "to_option_id": cur["leader_option_id"],
            })

    return {
        "method": METHOD,
        "horizon_weeks": horizon_weeks,
        "evaluate": evaluate,
        "onsets": [
            {"option_id": oid, "onset_weeks": float(d), "ramp_weeks": float(r)} for oid, (d, r) in applied.items()
        ],
        "checkpoints": checkpoints,
        "flips": flips,
        "leader_at_horizon": checkpoints[-1]["leader_option_id"],
    }
