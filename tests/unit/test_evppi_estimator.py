"""S2 (D-23.8) unit tests for the pure regression-EVPPI estimator.

Hand-derivable shapes with ANALYTIC EVPPI, fed as synthetic CRN sample arrays
directly to the estimator (no MC), so the numbers are pinned to closed-form
values with tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.utils.canonical_hash import canonical_json_hash
from src.utils.evppi import (
    REGRESSION_EVPPI_METHOD,
    REGRESSION_EVPPI_NULL_PERMUTATIONS,
    REGRESSION_EVPPI_POLY_DEGREE,
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


def _true_null_dataset(ds: int):
    """A TRUE-NULL factor: theta is INDEPENDENT of the option outcomes, but the
    decision genuinely flips on a hidden independent driver (so decision_evpi > 0).
    A correct below_resolution floor must catch these — the true EVPPI of theta is
    0. Varies theta's distribution (normal/heavy-tail/leverage-lumpy/uniform) and
    the outcome noise. Deterministic (fixed per-ds seeds)."""
    rng = np.random.default_rng(1000 + ds)
    n = 2000
    kind = ds % 4
    if kind == 0:
        theta = rng.standard_normal(n)
    elif kind == 1:
        theta = rng.standard_t(3, n)  # heavy tail => high-leverage points
    elif kind == 2:
        theta = rng.standard_normal(n)
        theta[rng.integers(0, n, 20)] *= 8  # leverage-lumpy
    else:
        theta = rng.uniform(-2, 2, n)
    noise = [0.2, 0.5, 1.0, 2.0][ds % 4]
    driver = rng.standard_normal(n)  # independent of theta
    return theta, {
        "A": driver + noise * rng.standard_normal(n),
        "B": -driver + noise * rng.standard_normal(n),
    }


class TestBelowResolutionFloorCalibration:
    """F-1 fix: the permutation-null floor is the MAX (not the mean) of K nulls, a
    permutation test at level ~1/(K+1). This pins the TRUE-NULL escape rate — the
    fraction of zero-information factors that wrongly ship as ``resolved`` — well
    below 10%. Reverting the aggregation to the mean (the pre-fix behaviour) drives
    the escape rate to ~45% and turns this test RED (mutation guard)."""

    N = 40
    TARGET = 0.10

    def test_true_null_escape_rate_below_10pct(self):
        escapes = 0
        for ds in range(self.N):
            theta, oo = _true_null_dataset(ds)
            est = factor_evppi_estimate(theta, oo, seed=7000 + ds)
            # A true-null factor "escapes" when it is labelled resolved
            # (evppi_raw strictly above its own noise floor).
            if est.evppi_raw > est.noise_floor:
                escapes += 1
        rate = escapes / self.N
        assert rate <= self.TARGET, (
            f"true-null escape rate {rate:.0%} exceeds {self.TARGET:.0%} — the "
            f"below_resolution floor is too permissive (mean-of-K aggregation?)."
        )

    def test_genuine_signal_not_floored_by_conservative_floor(self):
        """The more conservative (larger) MAX floor must NOT newly floor a genuine
        signal: the sign-flip shape's ~0.80 EVPPI stays resolved."""
        n = 20000
        theta = _standard_normal(n, seed=1)
        est = factor_evppi_estimate(theta, {"A": theta, "B": -theta}, seed=42)
        assert est.evppi_raw > est.noise_floor  # resolved
        assert est.evppi_raw > 0.5
        assert est.noise_floor < 0.05  # floor stays tiny relative to the signal


# ---------------------------------------------------------------------------
# Q6 (altitude): fail-loud fingerprint guard for the method-defining EVPPI
# constants, mirroring the STABILITY_CONFIDENCE_MAP pattern in
# tests/unit/test_confidence_provenance.py. The new estimator adopted the wire
# version TAG ("regression_evppi_v1") but NOT the fail-loud guard, so silently
# retuning degree=4 or K=16 would shift the below_resolution level (and the
# emitted evppi) under a stable method tag with no test failing. Bind the two
# METHOD-DEFINING constants to the version. The operational _STAGE_EVPI_JOINT_CELL_CAP
# (4096) is EXCLUDED — it governs WHEN a metric is skipped, not the VALUE of a
# computed one, and already has literal boundary tests.
#
# If either constant changes, this fails loud with the fresh fingerprint to paste
# and instructions: bump REGRESSION_EVPPI_METHOD and re-pin here. Do NOT regenerate
# the pin blindly to make the test pass — that defeats the disclosure contract.
PINNED_EVPPI_METHOD = "regression_evppi_v1"
PINNED_EVPPI_CONSTANTS_FINGERPRINT = (
    "934d8504deda533e7f3f9d61f2c5a3cd1f781fe4145743a69fce8ad625989a61"
)


def _current_evppi_constants_fingerprint() -> str:
    """sha256 over a canonical repr of the method-defining EVPPI constants,
    computed LIVE from source (the pin is a hardcoded literal, so this never
    self-heals). Same repo canonical_json_hash canonicalisation (sort_keys) the
    confidence fingerprint uses."""
    return canonical_json_hash(
        {
            "REGRESSION_EVPPI_POLY_DEGREE": REGRESSION_EVPPI_POLY_DEGREE,
            "REGRESSION_EVPPI_NULL_PERMUTATIONS": REGRESSION_EVPPI_NULL_PERMUTATIONS,
        }
    )


class TestEvppiMethodFingerprintGuard:
    def test_method_version_matches_pin(self):
        """The served method tag must equal the pinned version (they move together)."""
        assert REGRESSION_EVPPI_METHOD == PINNED_EVPPI_METHOD

    def test_constants_fingerprint_pinned_to_version(self):
        """If degree or K changed while REGRESSION_EVPPI_METHOD stayed put, fail
        loud and tell the editor exactly what to do."""
        current = _current_evppi_constants_fingerprint()
        assert current == PINNED_EVPPI_CONSTANTS_FINGERPRINT, (
            "The method-defining EVPPI constants (REGRESSION_EVPPI_POLY_DEGREE / "
            "REGRESSION_EVPPI_NULL_PERMUTATIONS) changed but REGRESSION_EVPPI_METHOD "
            f"is still '{REGRESSION_EVPPI_METHOD}'. Retuning degree or K changes the "
            "below_resolution level and the emitted evppi, and MUST be disclosed via a "
            "new method version:\n"
            "  1. bump REGRESSION_EVPPI_METHOD in src/utils/evppi.py, and\n"
            "  2. update PINNED_EVPPI_CONSTANTS_FINGERPRINT + PINNED_EVPPI_METHOD here.\n"
            f"Current fingerprint to pin: {current}\n"
            f"Old pinned fingerprint:     {PINNED_EVPPI_CONSTANTS_FINGERPRINT}"
        )
