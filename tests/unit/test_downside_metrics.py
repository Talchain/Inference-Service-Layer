"""Unit tests for B2 downside / tail-risk metrics (src/utils/downside.py).

RED-first, hand-derivable fixtures. Every fixture's expected value is derived in
a comment so the assertion is checkable by inspection, not by running the code.

The v2 robustness engine draws JOINT (Common Random Numbers) per-option outcome
samples — one edge_config + one factor_values per sample iteration, every option
evaluated against those SAME draws (robustness_analyzer_v2.py::_run_monte_carlo,
line ~1820-1857). So option_outcomes[A][i] and option_outcomes[B][i] share the
underlying draw at index i, and per-sample cross-option regret is a real joint
quantity, not an approximation from marginals.
"""

import math

import numpy as np
import pytest

from src.utils.downside import (
    CVAR_LEVEL,
    cvar_from_samples,
    expected_regret_per_option,
)

# ---------------------------------------------------------------------------
# CVAR_LEVEL constant
# ---------------------------------------------------------------------------


def test_cvar_level_is_ten_percent():
    """DOCTRINE-PENDING(Neil) default tail mass is the worst decile."""
    assert CVAR_LEVEL == 0.10


# ---------------------------------------------------------------------------
# cvar_from_samples — expected shortfall = mean of the worst `level` fraction
# ---------------------------------------------------------------------------


def test_cvar_10_mean_of_worst_ten_of_hundred():
    """Frozen-spec fixture: 100 known samples → cvar_10 = mean of the worst 10.

    samples = [0, 1, 2, ..., 99]; n=100, level=0.10 → k = floor(0.10*100) = 10.
    The 10 worst (lowest) values are {0,1,2,3,4,5,6,7,8,9}.
    mean = (0+1+...+9)/10 = 45/10 = 4.5  (exactly).

    MUTATION SENTINEL: if the metric took the worst 90% instead of 10%
    (k=90 → lowest {0..89}), the mean would be 44.5 ≠ 4.5 → this REDs.
    """
    samples = list(range(100))
    assert cvar_from_samples(samples) == pytest.approx(4.5)


def test_cvar_10_order_independent():
    """Shuffling the same 100 values does not change cvar_10 (=4.5)."""
    rng = np.random.default_rng(20260722)
    samples = rng.permutation(np.arange(100.0))
    assert cvar_from_samples(samples) == pytest.approx(4.5)


def test_cvar_10_small_n_floor_to_min_one():
    """n=5, level=0.10 → floor(0.5)=0 → clamped to k=1 → cvar = the single min.

    samples = [10, 2, 8, 4, 6]; the 1 worst = 2 → cvar_10 = 2.0.
    """
    assert cvar_from_samples([10.0, 2.0, 8.0, 4.0, 6.0]) == pytest.approx(2.0)


def test_cvar_10_k_is_floor_of_level_times_n():
    """n=25, level=0.10 → floor(2.5) = 2 lowest.

    samples = [0,1,...,24]; the 2 worst = {0,1}; mean = 0.5.
    """
    assert cvar_from_samples(list(range(25))) == pytest.approx(0.5)


def test_cvar_custom_level():
    """level override: worst 20% of [0..99] = {0..19}; mean = 9.5."""
    assert cvar_from_samples(list(range(100)), level=0.20) == pytest.approx(9.5)


def test_cvar_empty_raises():
    with pytest.raises(ValueError):
        cvar_from_samples([])


def test_cvar_le_p10_hard_guarantee_normal_draws():
    """Property: cvar_10 ≤ p05 ≤ p10 ≤ p50 for MC (normal-ish) distributions.

    For a normal, the lowest decile has a genuine left tail, so its mean (=ES_10)
    sits BELOW the 5th percentile: ES_10 ≈ -1.755σ < p05 ≈ -1.645σ < p10 ≈
    -1.282σ < p50 = 0. The HARD guarantee (any distribution) is cvar_10 ≤ p10;
    the full chain holds for the continuous unimodal MC regime, which is what the
    engine produces.
    """
    rng = np.random.default_rng(7)
    for mu, sigma in [(0.0, 1.0), (100.0, 15.0), (-3.0, 2.5), (50.0, 0.5)]:
        s = rng.normal(mu, sigma, 20000)
        cvar = cvar_from_samples(s)
        p05, p10, p50 = (float(v) for v in np.percentile(s, [5, 10, 50]))
        assert cvar <= p10, (mu, sigma, cvar, p10)          # hard guarantee
        assert cvar <= p05 <= p10 <= p50, (mu, sigma, cvar, p05, p10, p50)


def test_cvar_deterministic():
    s = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 3.5]
    assert cvar_from_samples(s) == cvar_from_samples(list(s))


# ---------------------------------------------------------------------------
# expected_regret_per_option — JOINT: mean_i( best_i - option_i )
# ---------------------------------------------------------------------------


def test_regret_dominant_option_is_zero_others_positive():
    """A dominates every sample → regret_A = 0; B lags → regret_B > 0.

    A = [10, 10, 10, 10]   B = [0, 5, 8, 2]        (joint, n=4)
    best per sample = [max(10,0), max(10,5), max(10,8), max(10,2)] = [10,10,10,10]
    regret_A = mean([10-10, 10-10, 10-10, 10-10]) = 0.0    (winner every sample)
    regret_B = mean([10-0,  10-5,  10-8,  10-2 ]) = (10+5+2+8)/4 = 25/4 = 6.25

    MUTATION SENTINEL: if best-per-sample used min() instead of max(), best would
    be [0,5,8,2] and regret_A = mean([0-10,5-10,8-10,2-10]) = -3.75 (negative,
    ≠ 0) → this REDs.
    """
    out = expected_regret_per_option(
        {"A": [10.0, 10.0, 10.0, 10.0], "B": [0.0, 5.0, 8.0, 2.0]}
    )
    assert out["A"] == pytest.approx(0.0)
    assert out["B"] == pytest.approx(6.25)


def test_regret_three_way_equals_ebest_minus_mean():
    """Cross-check the linear identity regret_o = E[best] - mean_o (full masks).

    A = [8, 2, 6, 4]  B = [4, 6, 2, 8]  C = [1, 1, 1, 1]
    best = [max(8,4,1), max(2,6,1), max(6,2,1), max(4,8,1)] = [8, 6, 6, 8]
    E[best] = (8+6+6+8)/4 = 28/4 = 7
    mean_A = 5, mean_B = 5, mean_C = 1
    regret_A = 7-5 = 2.0 ; regret_B = 7-5 = 2.0 ; regret_C = 7-1 = 6.0
    winner (highest mean = A/B) regret 2.0 ≤ C's 6.0.
    """
    out = expected_regret_per_option(
        {
            "A": [8.0, 2.0, 6.0, 4.0],
            "B": [4.0, 6.0, 2.0, 8.0],
            "C": [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert out["A"] == pytest.approx(2.0)
    assert out["B"] == pytest.approx(2.0)
    assert out["C"] == pytest.approx(6.0)


def test_regret_nonneg_and_winner_lowest_normal_draws():
    """Property: expected_regret ≥ 0 for all; argmin(regret) == argmax(mean)."""
    rng = np.random.default_rng(101)
    opts = {name: rng.normal(mu, 1.0, 5000) for name, mu in
            {"a": 0.0, "b": 1.0, "c": 0.5, "d": -0.5}.items()}
    out = expected_regret_per_option(opts)
    assert all(v >= 0.0 for v in out.values())
    winner_by_mean = max(opts, key=lambda k: float(np.mean(opts[k])))
    winner_by_regret = min(out, key=lambda k: out[k])
    assert winner_by_mean == winner_by_regret
    # winner's regret ≤ every other option's regret
    assert all(out[winner_by_regret] <= out[k] for k in out)


def test_regret_single_option_is_zero():
    """One option → best == itself every sample → regret 0."""
    out = expected_regret_per_option({"only": [3.0, 7.0, 1.0]})
    assert out["only"] == pytest.approx(0.0)


def test_regret_per_index_finite_masking():
    """Non-finite entries are dropped PER INDEX, preserving joint alignment.

    A = [10, 10, 10]   B = [0, inf, 4]        (n=3)
    idx0: both finite → best = max(10,0) = 10
    idx1: B=inf, only A finite → best = 10 (A participates)
    idx2: both finite → best = max(10,4) = 10
    regret_A over A's finite idx {0,1,2} = mean([0,0,0]) = 0.0
    regret_B over B's finite idx {0,2}   = mean([10-0, 10-4]) = mean([10,6]) = 8.0
      (idx1 excluded — B is non-finite there; alignment is by index, not by
       filtering each array independently)
    """
    out = expected_regret_per_option(
        {"A": [10.0, 10.0, 10.0], "B": [0.0, math.inf, 4.0]}
    )
    assert out["A"] == pytest.approx(0.0)
    assert out["B"] == pytest.approx(8.0)


def test_regret_deterministic():
    opts = {"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]}
    first = expected_regret_per_option(opts)
    second = expected_regret_per_option({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    assert first == second


def test_regret_empty_input_returns_empty():
    assert expected_regret_per_option({}) == {}
