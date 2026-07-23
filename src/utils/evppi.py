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
* ``EVPPI_i >= 0`` (Howard non-negativity): a negative estimate is finite-sample
  noise; the caller clamps it to 0 and DISCLOSES the clamp.
* ``EVPPI_i <= decision_evpi`` (theorem: learning ONE factor cannot be worth more
  than learning EVERYTHING): the caller caps at the total EVPI and DISCLOSES the
  cap. This module reports the RAW estimate so the bound can be checked/pinned.
* Below-resolution: a deterministic PERMUTATION-NULL floor — the EVPPI recomputed
  with ``theta`` shuffled (which breaks the theta<->U association, so the TRUE
  EVPPI is 0) measures the estimator's own overfit-noise floor. An estimate at or
  below that floor is indistinguishable from noise. No hand-tuned constant, so it
  cannot silently drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# Method version tag emitted on the wire so a new estimator is distinguishable.
REGRESSION_EVPPI_METHOD = "regression_evppi_v1"

# Polynomial degree of the conditional-expectation smoother. Degree 4 (5 params)
# is flexible enough for a smooth 1-D conditional mean while keeping overfit bias
# small at n >= 100. Reduced adaptively when the factor has few distinct values.
REGRESSION_EVPPI_POLY_DEGREE = 4

# Number of theta permutations averaged for the below-resolution noise floor.
# Averaging several reduces the floor estimate's own variance (a single
# permutation is a noisy estimate of the overfit floor); 8 is a cheap, stable
# default (each permutation is one extra polynomial fit per option).
REGRESSION_EVPPI_NULL_PERMUTATIONS = 8

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


def _fitted_conditional_expectation(theta: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Regress ``y`` on ``theta`` with a degree-``degree`` polynomial and return the
    fitted values at each sample's ``theta``.

    Uses ``numpy.polynomial.Polynomial.fit``, which maps the fitting domain to
    ``[-1, 1]`` internally (well-conditioned for any factor scale). Rank-deficient
    designs fall back to the numpy least-squares min-norm solution, which for a
    weak/absent relationship collapses toward the mean of ``y`` (fitted ~ E[y]).
    """
    series = np.polynomial.Polynomial.fit(theta, y, degree)
    return np.asarray(series(theta), dtype=float)


def _inner_expected_max(
    theta: np.ndarray, outcome_matrix: np.ndarray, degree: int
) -> float:
    """E_theta[max_o E[U_o|theta]] estimated by regressing each option's outcomes
    on ``theta`` and averaging the per-sample max over options of the fitted values.
    """
    fitted = np.vstack(
        [_fitted_conditional_expectation(theta, outcome_matrix[o], degree) for o in range(outcome_matrix.shape[0])]
    )  # (n_options, n_samples)
    return float(np.mean(fitted.max(axis=0)))


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
    # TRUE EVPPI is 0) and re-estimate; the residual is the estimator's overfit
    # noise. Average several permutations for a stable, deterministic floor.
    rng = np.random.default_rng(seed)
    floor_samples = []
    for _ in range(null_permutations):
        perm = rng.permutation(n_samples)
        inner_perm = _inner_expected_max(theta_arr[perm], matrix, deg)
        floor_samples.append(max(0.0, inner_perm - baseline))
    noise_floor = float(np.mean(floor_samples)) if floor_samples else 0.0

    return FactorEvppiEstimate(
        evppi_raw=float(evppi_raw),
        conditional_max_expected_utility=float(inner),
        baseline_max_expected_utility=baseline,
        noise_floor=noise_floor,
        degree_used=deg,
        n_samples=n_samples,
        degenerate=False,
    )
