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
from typing import Dict, Mapping, Sequence

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
) -> Dict[str, float]:
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

    All arrays must be the same length (the engine draws ``n_samples`` for every
    option). Returns ``{}`` for empty input.
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

    regrets: Dict[str, float] = {}
    for row, option_id in enumerate(ids):
        valid = finite_mask[row]
        if not valid.any():
            # Option has no finite sample — no honest regret. Callers only emit
            # downside when the option HAS finite samples, so this is defensive.
            regrets[option_id] = 0.0
            continue
        per_sample_regret = best_per_sample[valid] - matrix[row][valid]
        regrets[option_id] = float(np.mean(per_sample_regret))

    return regrets
