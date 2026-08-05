"""B2 downside / tail-risk metrics computed from per-option MC outcome samples.

The v2 robustness engine ALREADY draws joint (Common Random Numbers) per-option
outcome samples: one edge configuration + one factor draw per Monte-Carlo
iteration, every option evaluated against those SAME draws
(``robustness_analyzer_v2.py::_run_monte_carlo``). The distribution is therefore
already computed — these helpers only READ it. They add no new sampling and
change no existing emitted value.

Two families:

* ``cvar_from_samples`` / percentile p05 — MARGINAL tail metrics of one option's
  own outcome samples (same array + convention as the existing p10/p50/p90).
* ``expected_regret_per_option`` — a JOINT metric: it compares options
  per-sample (best-option outcome minus this option's outcome at the same
  underlying draw). This is meaningful ONLY because the samples are CRN-aligned;
  it must never be approximated from marginal statistics.
"""

from __future__ import annotations

import math

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# CVaR / expected-shortfall tail mass
# ---------------------------------------------------------------------------
# DOCTRINE-PENDING(Neil): tail mass for the CVaR / expected-shortfall metric.
# ``cvar_10`` is the MEAN of the worst CVAR_LEVEL fraction (the lowest outcomes,
# since higher outcome = better) of an option's MC outcome samples. 0.10 = the
# worst decile — the conventional 10% expected shortfall. The VALUE (0.10) is a
# risk-modeling default reserved for Neil (mirrors RISK_AVERSION_COEFFICIENT in
# sequential_decision.py); do NOT read 0.10 as scientifically ratified.
CVAR_LEVEL = 0.10


def cvar_from_samples(samples: Sequence[float], level: float = CVAR_LEVEL) -> float:
    """Expected shortfall: the mean of the worst ``level`` fraction of samples.

    "Worst" = lowest outcome values (higher outcome is better in the engine).
    The count of worst samples is ``k = max(1, floor(level * n))`` — so for the
    canonical 100 samples at level 0.10 this is exactly the mean of the 10
    lowest, matching the hand-derivable spec. For continuous unimodal MC
    distributions this equals the mean of all samples at or below the 10th
    percentile (percentile-based ES).

    Order-independent and deterministic. Raises ``ValueError`` on empty input.
    """
    arr = np.asarray(samples, dtype=float)
    n = arr.size
    if n == 0:
        raise ValueError("cvar_from_samples requires at least one sample")
    k = max(1, int(math.floor(level * n)))
    # np.partition puts the k smallest in the first k positions (unsorted among
    # themselves, which is fine — we only take their mean). O(n), order-invariant.
    lowest_k = np.partition(arr, k - 1)[:k]
    return float(np.mean(lowest_k))


def expected_regret_per_option(
    option_samples: Mapping[str, Sequence[float]],
) -> Dict[str, Optional[float]]:
    """Joint expected regret per option from CRN-aligned per-option samples.

    ``expected_regret[o] = mean_i( best_i - o_i )`` where ``best_i`` is the
    highest outcome across options at sample ``i`` and ``o_i`` is this option's
    outcome at the same underlying draw. ``>= 0`` by construction (an option
    never beats the per-sample best), and ``0`` for an option that wins every
    sample. The option with the highest mean outcome has the lowest regret
    (``regret_o = E[best] - mean_o`` when all draws are finite).

    Non-finite entries are handled PER INDEX to preserve joint alignment: at each
    sample the per-sample best is taken over the options that are finite there,
    and an option's regret averages only over the samples where IT is finite.
    This never inpaints or reorders — alignment is by index, so the joint
    comparison stays honest.

    An option with NO finite sample gets ``None`` — ABSENCE, not a number. There
    is no honest regret for an option that was never measured, and ``0.0`` is the
    worst possible placeholder for this statistic: it is exactly the value of an
    option that WINS EVERY SAMPLE, i.e. the most favourable regret expressible.
    (It used to be returned here as a "defensive" default; see
    ``decision_evpi_from_regrets`` for the reader that consumed it as real.)

    All arrays must be the same length (the engine draws ``n_samples`` for every
    option). Returns ``{}`` for empty input.

    CALLER CONTRACT: pass the PRE-``_apply_auto_scaled_noise`` outcomes — the
    genuinely CRN-aligned population (the same one that produces
    ``win_probability``). The auto-scaled-noise step adds INDEPENDENT per-option
    noise, so its samples are NOT CRN-aligned at a given index; feeding them here
    inflates regret by a max-over-independent-noise premium (CODE-REVIEW-ISL F1).
    """
    if not option_samples:
        return {}

    ids = list(option_samples.keys())
    arrays = [np.asarray(option_samples[i], dtype=float) for i in ids]

    lengths = {a.size for a in arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"expected_regret_per_option requires equal-length (CRN-aligned) "
            f"sample arrays; got lengths {sorted(lengths)}"
        )

    matrix = np.vstack(arrays)  # (n_options, n_samples)
    finite_mask = np.isfinite(matrix)

    # Per-sample best over the options finite at that sample. -inf sentinel for
    # non-finite entries so they never win the column max; a column where an
    # option is finite always has that option as a candidate, so best_i >= o_i
    # wherever o is finite (regret >= 0).
    masked = np.where(finite_mask, matrix, -np.inf)
    best_per_sample = masked.max(axis=0)  # (n_samples,)

    regrets: Dict[str, Optional[float]] = {}
    for row, option_id in enumerate(ids):
        valid = finite_mask[row]
        if not valid.any():
            # Option has no finite sample — there is no honest regret to report,
            # so report NONE. Emitting 0.0 here was a fabrication: it is the
            # regret of an option that wins every draw, and a downstream reader
            # (the decision-EVPI bound) accepted it as a real measurement
            # because `math.isfinite(0.0)` is True. Absent != wrong (trap #13).
            regrets[option_id] = None
            continue
        per_sample_regret = best_per_sample[valid] - matrix[row][valid]
        regrets[option_id] = float(np.mean(per_sample_regret))

    return regrets


def decision_evpi_from_regrets(
    regrets: Mapping[str, Optional[float]],
) -> Optional[float]:
    """Whole-decision EVPI = ``min_o expected_regret[o]``, over the options that
    HAVE an honest regret.

    Exact identity on a complete population: ``min_o (E[max_o U] - E[U_o])`` is
    achieved at ``argmax_o E[U_o]``, so this equals ``E[max_o U] - max_o E[U_o]``
    — the value of perfect information about the decision, in outcome units.

    ABSENCE HANDLING — this is the whole point of the function existing. An
    option whose regret is ``None`` (no finite draw) or non-finite is EXCLUDED
    from the minimisation rather than counted as a low value. Including such an
    option would collapse the minimum toward the most favourable end and thereby
    UNDERSTATE the value of information — and, because this bound caps every
    per-factor EVPPI, would silently report that no factor is worth learning
    about. Returns ``None`` when no option carries an honest regret: the bound is
    then unknown, and an unknown cap is applied as "no cap" by callers rather
    than as "cap of zero".

    This is deliberately a NAMED function rather than an inline ``min(...)``: the
    exclusion rule is the load-bearing part and must be stated and testable in
    one place, not re-derived at each call site.
    """
    honest = [r for r in regrets.values() if r is not None and math.isfinite(r)]
    return min(honest) if honest else None
