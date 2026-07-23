"""S2 (D-23.8) unit tests for the pure regression-EVPPI estimator.

Hand-derivable shapes with ANALYTIC EVPPI, fed as synthetic CRN sample arrays
directly to the estimator (no MC), so the numbers are pinned to closed-form
values with tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.utils.evppi import (
    REGRESSION_EVPPI_METHOD,
    factor_evppi_estimate,
)

# Standard-normal E|theta| = sqrt(2/pi): the analytic EVPPI of the sign-flip shape.
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # ~= 0.7978845608


def _standard_normal(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


class TestFlipShapeAnalyticEvppi:
    """theta ~ N(0,1); U_A = theta, U_B = -theta. The optimal option flips at
    theta = 0, so with perfect info you pick max(theta, -theta) = |theta|.

        max_o E[U_o]              = max(E[theta], E[-theta]) = 0
        E[max_o E[U_o|theta]]     = E[|theta|]               = sqrt(2/pi)
        EVPPI                     = sqrt(2/pi) ~= 0.79788

    (A degree>=1 polynomial fits U_A=theta and U_B=-theta exactly, so the
    estimator recovers E|theta| up to Monte Carlo error.)
    """

    def test_flip_evppi_matches_sqrt_2_over_pi(self):
        n = 20000
        theta = _standard_normal(n, seed=1)
        est = factor_evppi_estimate(
            theta, {"A": theta, "B": -theta}, seed=42
        )
        assert est.evppi_raw == pytest.approx(SQRT_2_OVER_PI, abs=0.02)
        assert est.baseline_max_expected_utility == pytest.approx(0.0, abs=0.02)
        assert est.conditional_max_expected_utility == pytest.approx(SQRT_2_OVER_PI, abs=0.02)
        assert not est.degenerate

    def test_flip_evppi_well_above_noise_floor(self):
        """A genuine high-VOI factor is resolved, not below-resolution."""
        n = 20000
        theta = _standard_normal(n, seed=2)
        est = factor_evppi_estimate(theta, {"A": theta, "B": -theta}, seed=7)
        assert est.evppi_raw > est.noise_floor
        # The overfit floor is tiny relative to the ~0.8 signal.
        assert est.noise_floor < 0.05


class TestDominantOptionZeroEvppi:
    """U_A = 10 + theta dominates U_B = theta pointwise for every theta, so no
    information can change the choice: EVPPI = 0 for the factor."""

    def test_dominant_option_evppi_is_zero(self):
        n = 8000
        theta = _standard_normal(n, seed=3)
        est = factor_evppi_estimate(theta, {"A": 10.0 + theta, "B": theta}, seed=9)
        # Raw estimate hugs 0 (only overfit noise); at/below the null floor.
        assert abs(est.evppi_raw) < 0.05
        assert est.evppi_raw <= est.noise_floor + 1e-9


class TestIrrelevantFactorReturnsZero:
    """An outcome that does not depend on theta at all has EVPPI ~ 0, and the
    estimator discriminates it from a high-VOI factor (mutation-style)."""

    def test_irrelevant_factor_near_zero_and_below_floor(self):
        n = 8000
        rng = np.random.default_rng(11)
        theta = rng.standard_normal(n)
        # Outcomes depend on an INDEPENDENT driver, not theta. Ranking still flips
        # across samples (so decision EVPI > 0) but NOT because of theta.
        driver = rng.standard_normal(n)
        est = factor_evppi_estimate(theta, {"A": driver, "B": -driver}, seed=13)
        assert abs(est.evppi_raw) < 0.05
        assert est.evppi_raw <= est.noise_floor + 1e-9

    def test_discriminates_high_voi_from_zero_voi(self):
        n = 12000
        rng = np.random.default_rng(17)
        theta = rng.standard_normal(n)
        driver = rng.standard_normal(n)
        high = factor_evppi_estimate(theta, {"A": theta, "B": -theta}, seed=1)
        zero = factor_evppi_estimate(theta, {"A": driver, "B": -driver}, seed=1)
        assert high.evppi_raw > 0.5
        assert zero.evppi_raw < 0.05
        assert high.evppi_raw > zero.evppi_raw


class TestBoundAndMethod:
    def test_single_option_evppi_zero(self):
        """With one option there is no decision to switch, so EVPPI = 0."""
        n = 4000
        theta = _standard_normal(n, seed=5)
        est = factor_evppi_estimate(theta, {"only": theta}, seed=1)
        assert est.evppi_raw == pytest.approx(0.0, abs=1e-9)

    def test_degenerate_constant_theta_is_zero(self):
        n = 2000
        theta = np.full(n, 0.7)
        est = factor_evppi_estimate(theta, {"A": np.arange(n) * 0.1, "B": -np.arange(n) * 0.1}, seed=1)
        assert est.degenerate is True
        assert est.evppi_raw == 0.0
        assert est.degree_used == 0

    def test_deterministic_given_seed(self):
        n = 5000
        theta = _standard_normal(n, seed=8)
        oo = {"A": theta, "B": -theta}
        a = factor_evppi_estimate(theta, oo, seed=99)
        b = factor_evppi_estimate(theta, oo, seed=99)
        assert a == b

    def test_method_tag_value(self):
        assert REGRESSION_EVPPI_METHOD == "regression_evppi_v1"
