"""S2 (D-23.8) per-factor EVPPI via single-loop Strong-Oakley regression.

EVPPI (Expected Value of Partial Perfect Information) answers, per uncertain
factor: *how much better could the decision be if we learned this factor's true
value before choosing?* — in OUTCOME units (the same units as ``outcome.mean``
and ``decision_evpi``), NOT win-probability points.

Definition (Strong & Oakley 2014, single-loop regression estimator)::

    EVPPI_i = E_{theta_i}[ max_o  E[U_o | theta_i] ]  -  max_o E[U_o]

where ``U_o`` is option ``o``'s outcome and ``theta_i`` is factor ``i``. The inner
conditional expectation ``E[U_o | theta_i]`` is estimated by REGRESSING each
option's per-sample outcome on that factor's per-sample value with a flexible
smoother. Both terms are read from the samples the robustness engine ALREADY
draws (the retained pre-noise Common-Random-Numbers joint population): there is
NO nested Monte Carlo and NO new sampling.

Why this is honest under CORRELATION as well as independence
-----------------------------------------------------------
The regression conditions on ``theta_i`` over draws from the JOINT distribution
(under active correlation, those draws come from the Gaussian copula — see
``FactorSampler._draw_correlated``). Regressing on a single input over joint
samples consistently estimates the true joint conditional expectation
``E[U_o | theta_i]``, which legitimately includes everything ``theta_i`` implies
about the factors correlated with it. The estimator never assumes independence;
it only assumes the samples are from the joint, which they are. This is why
factor_evppi is EMITTED (not suppressed) under correlation while the independence-
assuming OAT attributions (p_win_sensitivity, factor_sensitivity,
conditional_winners) are suppressed.

Estimator choice (``regression_evppi_v1``)
------------------------------------------
A fixed-degree polynomial least-squares smoother via ``numpy.polynomial`` (which
scales the fitting domain internally, so it is well-conditioned regardless of the
factor's raw scale). numpy + scipy are the only numeric dependencies in the ISL
tree; statsmodels/scikit-learn are transitive-only (via y0) and MUST NOT be
depended on. A modest degree keeps the finite-sample overfit bias small: at
``n >= 100`` with 5 parameters the samples-per-parameter ratio is >= 20.

Honesty guardrails
------------------
* ``EVPPI_i >= 0`` (Howard non-negativity): DEAD-MAN'S-SWITCH. For this estimator
  the raw is NON-NEGATIVE BY CONSTRUCTION — least-squares with an intercept
  preserves the sample mean (``mean_s fitted_o = mean_s U_o = E[U_o]``), and
  ``mean_s max_o fitted_o >= max_o mean_s fitted_o`` (Jensen), so
  ``evppi_raw = inner - baseline >= 0`` for every input. The caller's
  ``max(0, .)`` clamp therefore never fires on this estimator; a future
  ``clamped_low: true`` in telemetry means the ESTIMATOR CHANGED (e.g. lost its
  intercept), not that a real negative occurred. Kept as defence-in-depth.
* ``EVPPI_i <= decision_evpi`` (theorem: learning ONE factor cannot be worth more
  than learning EVERYTHING): the caller caps at the total EVPI and DISCLOSES the
  cap. This module reports the RAW estimate so the bound can be checked/pinned.
* Below-resolution: a deterministic PERMUTATION-NULL floor. The EVPPI is recomputed
  on ``theta`` SHUFFLED (breaking the theta<->U association, so the TRUE EVPPI is
  0), K times; the floor is the MAX of those K nulls. This is a permutation test:
  under the null (theta independent of U) the real estimate is EXCHANGEABLE with
  the K shuffled ones, so a true-null factor exceeds the max with probability
  ~1/(K+1) — i.e. ``below_resolution`` catches ~1 - 1/(K+1) of pure-noise factors.
  The earlier MEAN-of-K aggregation was a ~50th-percentile threshold (~half of true
  nulls escaped as ``resolved``, F-1); MAX at K=16 is a ~1/17 ~= 6% level
  (empirically ~8% escape on 100 true-null datasets, down from ~45%). Deterministic
  (per-factor seed) and derived from the data — no hand-tuned magnitude constant, so
  it cannot silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.polynomial import polynomial as _npp
from numpy.polynomial import polyutils as _nppu

# The fitting window ``Polynomial.fit`` maps the data domain onto ([-1, 1]).
_FIT_WINDOW = np.array([-1.0, 1.0])

# Method version tag emitted on the wire so a new estimator is distinguishable.
REGRESSION_EVPPI_METHOD = "regression_evppi_v1"

# Polynomial degree of the conditional-expectation smoother. Degree 4 (5 params)
# is flexible enough for a smooth 1-D conditional mean while keeping overfit bias
# small at n >= 100. Reduced adaptively when the factor has few distinct values.
REGRESSION_EVPPI_POLY_DEGREE = 4

# Number of theta permutations for the below-resolution noise floor. The floor is
# the MAX of these K nulls (NOT the mean — see F-1). Under the null the real
# estimate is exchangeable with the K shuffled ones, so the floor is a permutation
# test at level ~1/(K+1): K=16 => ~5.9% (empirically ~8% escape on 100 true-null
# datasets, vs ~45% with the old mean aggregation). K=16 is the smallest value that
# clears a <10% true-null escape target with margin; larger K does not reduce the
# residual near-boundary tail further, and a larger MAX only raises the floor (more
# conservative). Deterministic (per-factor sha256-derived seed). Cost: K extra
# polynomial fits per option per factor (microseconds).
REGRESSION_EVPPI_NULL_PERMUTATIONS = 16

# A factor whose sampled values span fewer than this many distinct values carries
# no learnable signal for a regression (a constant/near-constant input), so its
# EVPPI is exactly 0 by construction.
_MIN_DISTINCT_THETA = 2


@dataclass(frozen=True)
class FactorEvppiEstimate:
    """Raw (pre-clamp) per-factor EVPPI estimate and its audit components."""

    evppi_raw: float
    """E[max_o E[U_o|theta]] - max_o E[U_o]. May be slightly negative (noise)."""
    conditional_max_expected_utility: float
    """E_theta[max_o E[U_o|theta]] — the with-perfect-info leg (inner term)."""
    baseline_max_expected_utility: float
    """max_o E[U_o] — the without-info leg (best option by mean outcome)."""
    noise_floor: float
    """Permutation-null overfit floor; EVPPI <= floor is below-resolution."""
    degree_used: int
    """Effective polynomial degree after adaptive reduction (0 = degenerate)."""
    n_samples: int
    degenerate: bool
    """True when theta is (near-)constant, so EVPPI is 0 by construction."""


def _effective_degree(theta: np.ndarray, degree: int) -> int:
    """Cap the polynomial degree at ``n_distinct - 1`` to avoid over-parameterising
    a factor with few distinct sampled values (min degree 1)."""
    n_distinct = int(np.unique(theta).size)
    return max(1, min(degree, n_distinct - 1))


def _inner_expected_max(
    theta: np.ndarray, outcome_matrix: np.ndarray, degree: int
) -> float:
    """E_theta[max_o E[U_o|theta]] estimated by regressing each option's outcomes
    on ``theta`` and averaging the per-sample max over options of the fitted values.

    EFFICIENCY (E1 SAFE variant): all ``n_options`` regress the SAME ``theta``, so
    they share ONE polynomial design matrix. Build the mapped Vandermonde once and
    solve every option in a single multi-RHS ``np.linalg.lstsq(V, Y.T)`` — one SVD
    instead of ``n_options`` independent ``Polynomial.fit`` SVDs (measured ~5x at the
    50-factor/10-option cap). This is NUMERICALLY EQUIVALENT to the per-option fit:
    the same data-domain→[-1, 1] mapping (``Polynomial.fit``'s internal conditioning),
    the same power-basis Vandermonde, the same least-squares min-norm solve — max
    fitted difference ~1e-14 over random shapes, absorbed by the wire's 6-dp rounding
    and every pin's tolerance. (The AGGRESSIVE normal-equations reuse ``A = VᵀV`` is
    deliberately NOT used — it squares ``cond(V)`` and overflows at degree 4.)

    Rank-deficient designs still fall back to numpy's least-squares min-norm solution
    (weak/absent relationship → fitted collapses toward ``E[y]``), unchanged.
    """
    theta = np.asarray(theta, dtype=float)
    # Replicate numpy.polynomial.Polynomial.fit EXACTLY so the shared solve is
    # byte-equivalent to the per-option fits (not merely close): (1) map theta from
    # its data domain [min, max] onto [-1, 1]; (2) build the power-basis Vandermonde;
    # (3) column-scale it by column norms (Polynomial.fit's conditioning step) — since
    # only ``fitted = V @ coef`` is needed and ``coef = coef_scaled / scl``, the fitted
    # values are just ``V_scaled @ coef_scaled``, so no un-scaling is required;
    # (4) least-squares with the SAME ``rcond = len(x) * eps`` numpy uses. lstsq shares
    # ONE SVD across all option columns, replacing n_options independent SVDs.
    domain = [float(theta.min()), float(theta.max())]
    scaled = _nppu.mapdomain(theta, domain, _FIT_WINDOW)
    vander = _npp.polyvander(scaled, degree)  # (n_samples, degree + 1)
    col_scale = np.sqrt(np.square(vander).sum(axis=0))
    col_scale[col_scale == 0] = 1.0  # guard an all-zero column (degenerate basis term)
    vander_scaled = vander / col_scale
    rcond = float(len(theta) * np.finfo(theta.dtype).eps)
    rhs = np.asarray(outcome_matrix, dtype=float).T  # (n_samples, n_options)
    coef_scaled, *_ = np.linalg.lstsq(vander_scaled, rhs, rcond=rcond)
    fitted = vander_scaled @ coef_scaled  # (n_samples, n_options), fitted at theta
    return float(np.mean(fitted.max(axis=1)))


def factor_evppi_estimate(
    theta: Sequence[float],
    option_outcomes: Mapping[str, Sequence[float]],
    *,
    seed: int,
    degree: int = REGRESSION_EVPPI_POLY_DEGREE,
    null_permutations: int = REGRESSION_EVPPI_NULL_PERMUTATIONS,
) -> FactorEvppiEstimate:
    """Single-loop Strong-Oakley regression EVPPI for one factor.

    Args:
        theta: per-sample values of this factor (length N), from the retained
            joint CRN population.
        option_outcomes: option_id -> per-sample PRE-noise outcome (each length N),
            the same CRN population (every option evaluated against the same draw).
        seed: deterministic seed for the permutation-null floor.
        degree / null_permutations: estimator constants (see module docstring).

    Returns:
        FactorEvppiEstimate with the RAW estimate and audit components. The caller
        applies the Howard (>= 0) and per-factor <= total-EVPI clamps and discloses.
    """
    theta_arr = np.asarray(theta, dtype=float)
    option_ids = sorted(option_outcomes.keys())
    matrix = np.vstack([np.asarray(option_outcomes[o], dtype=float) for o in option_ids])
    n_samples = theta_arr.size

    # baseline = max over options of the mean outcome (== the max_o E[U_o] leg of
    # decision_evpi, so EVPPI shares decision_evpi's without-info baseline).
    baseline = float(matrix.mean(axis=1).max())

    # Degenerate factor: a (near-)constant theta carries no learnable signal.
    if int(np.unique(theta_arr).size) < _MIN_DISTINCT_THETA:
        return FactorEvppiEstimate(
            evppi_raw=0.0,
            conditional_max_expected_utility=baseline,
            baseline_max_expected_utility=baseline,
            noise_floor=0.0,
            degree_used=0,
            n_samples=n_samples,
            degenerate=True,
        )

    deg = _effective_degree(theta_arr, degree)
    inner = _inner_expected_max(theta_arr, matrix, deg)
    evppi_raw = inner - baseline

    # Permutation-null floor: shuffle theta (breaking theta<->U association, so the
    # TRUE EVPPI is 0) and re-estimate; each residual is the estimator's overfit
    # noise under the null. The floor is the MAX over K permutations, NOT the mean:
    # the mean is a ~50th-percentile threshold that a true-null estimate exceeds
    # ~half the time (F-1), whereas MAX makes this a permutation test at level
    # ~1/(K+1) (the real estimate is exchangeable with the K shuffled nulls under
    # the null), so ~1 - 1/(K+1) of pure-noise factors are caught as below_resolution.
    # (max(0, .) is defence-in-depth: inner_perm - baseline is >= 0 by the same
    # mean-preservation + Jensen argument as evppi_raw, so it never fires.)
    rng = np.random.default_rng(seed)
    floor_samples = []
    for _ in range(null_permutations):
        perm = rng.permutation(n_samples)
        inner_perm = _inner_expected_max(theta_arr[perm], matrix, deg)
        floor_samples.append(max(0.0, inner_perm - baseline))
    noise_floor = float(np.max(floor_samples)) if floor_samples else 0.0

    return FactorEvppiEstimate(
        evppi_raw=float(evppi_raw),
        conditional_max_expected_utility=float(inner),
        baseline_max_expected_utility=baseline,
        noise_floor=noise_floor,
        degree_used=deg,
        n_samples=n_samples,
        degenerate=False,
    )
