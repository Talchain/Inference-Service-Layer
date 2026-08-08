"""
ROADMAP 2.720 — range→distribution converter (2.521 Q1, Neil-ratified).

Spec: parallel-briefs/RANGE-TO-DISTRIBUTION-SPEC-2026-08-08.md (validated by
execution while being written; solver-space choice, acceptance rule, refusal
taxonomy and this battery are ratified there).

Science basis (spec §1): a user-stated range is an approximately 50% credible
interval (Quillien, Bramley & Lucas 2025); the fit places the distribution's
QUARTILES on the stated bounds — beta for bounded quantities, normal for
unbounded scalars.

WHAT IS UNDER TEST (spec §5):
  T1  middle-mass property        — CDF(b) − CDF(a) = 0.50 within 2×acceptance
  T2  quartile placement          — PPF(.25)=a, PPF(.75)=b (equal 25% tails)
  T3  median strictly inside (a,b) — instrument check (provable, §3 E10)
  T4  normal closed-form exactness
  T4a beta known answers          — (0.25,0.75) → Beta(1,1); symmetry
  T4b beta round-trip across the corpus
  T5  refusal taxonomy            — each code, identity-bound, constructor-spied
  T6  determinism                 — bit-identical parameters call-to-call
  T7  purity / zero RNG           — no draw consumed from any stream
  plus hypothesis property sweeps and the resolver's every-code-has-a-bucket
  derived assertion (spec §4.3 / mutant M5).

TRAP DISCIPLINE:
- Fixtures are identity-bound by corpus KEY, never by a value predicate another
  row could satisfy (trap 19).
- Expectations are derived from the producer's declared semantics — Neil's
  recorded ruling and the closed-form/CDF definitions — never from this lane's
  own classification (trap 13c).
- The refusal battery asserts the EXACT code, and that no FittedDistribution is
  constructed (constructor spy, not absence-of-return).
- The every-code bucket test derives the code vocabulary from the type
  (typing.get_args), never from a hand list (trap 12).
"""

import math
from typing import Any, ClassVar, Dict, Optional, Tuple, get_args

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from src.models.range_fit import (
    RANGE_FIT_METHOD_VERSION,
    Domain,
    FittedDistribution,
    RangeFitDisclosure,
    RangeFitRefusal,
    RangeFitRefusalCode,
    RangeFitRefusalPayload,
    UserStatedRange,
)
from src.services import range_fit as range_fit_module
from src.services.range_fit import (
    ACCEPTANCE_RESIDUAL_TOLERANCE,
    RATIFIED_COVERAGE,
    fit_range_distribution,
    resolve_range_fits,
)

# Derived, never restated (spec §2.1): the middle-mass tolerance is the sum of
# the two acceptance residual bounds — not a third constant. (The acceptance
# bound itself is 1e−8, corrected from the spec's 1e−9 at the deployed scipy
# 1.16.3 — see the measured premise-correction note at its definition site.)
MIDDLE_MASS_TOL = 2 * ACCEPTANCE_RESIDUAL_TOLERANCE
QUARTILE_ABS_TOL = 1e-6  # spec §5 T2/T4a/T4b
Z75 = 0.6744897501960817  # Φ⁻¹(0.75), spec §2.3 — used here as the ORACLE only


# ---------------------------------------------------------------------------
# Corpus — identity-keyed (trap 19). The 11 beta rows include every named row
# from the spec's measured §2.4 corpus run.
# ---------------------------------------------------------------------------

BETA_CORPUS: Dict[str, Tuple[float, float]] = {
    "uniform_identity": (0.25, 0.75),  # exact known answer: Beta(1,1) (§3 E9)
    "u_shape": (0.001, 0.999),
    "near_edge_low": (1e-6, 0.3),  # §3 E6: extreme-but-valid, no epsilon floor
    "j_shape": (0.9, 0.99),
    "narrow_asymmetric": (0.01, 0.02),
    "near_full_domain": (1e-6, 1.0 - 1e-6),
    "hairline": (0.5, 0.500001),  # the row that killed (ln α, ln β) space
    "wide_symmetric": (0.1, 0.9),  # §3 E9: U-shaped, α = β < 1
    "mid_left": (0.2, 0.6),
    "tight_center": (0.4, 0.5),
    "symmetric_wide95": (0.05, 0.95),
}

NORMAL_CORPUS: Dict[str, Tuple[float, float]] = {
    "classic_unit": (-1.0, 1.0),  # → (0, 1.4826022): the IQR consistency constant
    "decade": (0.0, 10.0),
    "large_magnitude": (-1e6, 3e6),
    "hairline_normal": (2.5, 2.500001),
    "negative_band": (-3.7, -1.2),
}


def _fit_beta(key: str) -> FittedDistribution:
    a, b = BETA_CORPUS[key]
    return fit_range_distribution(lower=a, upper=b, domain="unit_interval")


def _fit_normal(key: str) -> FittedDistribution:
    a, b = NORMAL_CORPUS[key]
    return fit_range_distribution(lower=a, upper=b, domain="unbounded")


def _cdf(fitted: FittedDistribution, x: float) -> float:
    """Recompute the CDF from the RETURNED PARAMETERS — independent of any
    disclosure convenience fields the implementation also carries."""
    if fitted.family == "normal":
        return float(norm.cdf(x, loc=fitted.mu, scale=fitted.sigma))
    return float(beta_dist.cdf(x, fitted.alpha, fitted.beta))


def _ppf(fitted: FittedDistribution, q: float) -> float:
    if fitted.family == "normal":
        return float(norm.ppf(q, loc=fitted.mu, scale=fitted.sigma))
    return float(beta_dist.ppf(q, fitted.alpha, fitted.beta))


# ---------------------------------------------------------------------------
# T1 — middle-mass property (the ruling's headline content)
# ---------------------------------------------------------------------------


class TestT1MiddleMass:
    @pytest.mark.parametrize("key", sorted(BETA_CORPUS))
    def test_beta_middle_mass_is_half(self, key: str) -> None:
        a, b = BETA_CORPUS[key]
        fitted = _fit_beta(key)
        mass = _cdf(fitted, b) - _cdf(fitted, a)
        assert abs(mass - 0.5) <= MIDDLE_MASS_TOL, (key, mass)

    @pytest.mark.parametrize("key", sorted(NORMAL_CORPUS))
    def test_normal_middle_mass_is_half(self, key: str) -> None:
        a, b = NORMAL_CORPUS[key]
        fitted = _fit_normal(key)
        mass = _cdf(fitted, b) - _cdf(fitted, a)
        assert abs(mass - 0.5) <= MIDDLE_MASS_TOL, (key, mass)


# ---------------------------------------------------------------------------
# T2 — quartile placement (stronger than T1: pins EQUAL 25% tails, the ruled
# pair's second half — mutant M2's discriminating target)
# ---------------------------------------------------------------------------


class TestT2QuartilePlacement:
    @pytest.mark.parametrize("key", sorted(BETA_CORPUS))
    def test_beta_quartiles_on_bounds(self, key: str) -> None:
        a, b = BETA_CORPUS[key]
        fitted = _fit_beta(key)
        assert abs(_ppf(fitted, 0.25) - a) <= QUARTILE_ABS_TOL, key
        assert abs(_ppf(fitted, 0.75) - b) <= QUARTILE_ABS_TOL, key

    @pytest.mark.parametrize("key", sorted(NORMAL_CORPUS))
    def test_normal_quartiles_on_bounds(self, key: str) -> None:
        a, b = NORMAL_CORPUS[key]
        fitted = _fit_normal(key)
        assert abs(_ppf(fitted, 0.25) - a) <= QUARTILE_ABS_TOL, key
        assert abs(_ppf(fitted, 0.75) - b) <= QUARTILE_ABS_TOL, key


# ---------------------------------------------------------------------------
# T3 — median strictly inside (a, b): a correct fit GUARANTEES this (spec §3
# E10 derives it), so a firing here can only mean the fit machinery is wrong.
# The MEAN is deliberately NOT asserted (it may legitimately leave (a,b) for
# skewed fits — an assertion that can fire on a correct fit is a false-alarm
# generator).
# ---------------------------------------------------------------------------


class TestT3MedianBetweenBounds:
    @pytest.mark.parametrize("key", sorted(BETA_CORPUS))
    def test_beta_median_strictly_inside(self, key: str) -> None:
        a, b = BETA_CORPUS[key]
        fitted = _fit_beta(key)
        median = _ppf(fitted, 0.5)
        assert a < median < b, (key, median)

    @pytest.mark.parametrize("key", sorted(NORMAL_CORPUS))
    def test_normal_median_strictly_inside(self, key: str) -> None:
        a, b = NORMAL_CORPUS[key]
        fitted = _fit_normal(key)
        median = _ppf(fitted, 0.5)
        assert a < median < b, (key, median)


# ---------------------------------------------------------------------------
# T4 — normal closed form, exact (spec §2.3)
# ---------------------------------------------------------------------------


class TestT4NormalClosedForm:
    @pytest.mark.parametrize("key", sorted(NORMAL_CORPUS))
    def test_matches_closed_form(self, key: str) -> None:
        a, b = NORMAL_CORPUS[key]
        fitted = _fit_normal(key)
        mu_expected = (a + b) / 2.0
        sigma_expected = (b - a) / (2.0 * Z75)
        assert fitted.family == "normal"
        assert fitted.mu == pytest.approx(mu_expected, rel=1e-12, abs=1e-12)
        assert fitted.sigma == pytest.approx(sigma_expected, rel=1e-12)

    def test_classic_unit_fixture_by_name(self) -> None:
        """(−1, 1) → μ=0, σ=1/Φ⁻¹(0.75)=1.4826022 — the classic IQR/MAD
        normal-consistency constant (spec §2.3 known-answer fixture)."""
        fitted = _fit_normal("classic_unit")
        assert fitted.mu == pytest.approx(0.0, abs=1e-12)
        assert fitted.sigma == pytest.approx(1.4826022, abs=1e-7)
        assert fitted.coverage == RATIFIED_COVERAGE
        assert fitted.method_version == RANGE_FIT_METHOD_VERSION


# ---------------------------------------------------------------------------
# T4a / T4b — beta known answers and round-trip
# ---------------------------------------------------------------------------


class TestT4aBetaKnownAnswers:
    def test_uniform_identity_is_beta_1_1(self) -> None:
        """The uniform CDF is the identity, so its quartiles are exactly
        0.25/0.75 — (0.25, 0.75) MUST fit Beta(1, 1) (spec §3 E9)."""
        fitted = _fit_beta("uniform_identity")
        assert fitted.family == "beta"
        assert fitted.alpha == pytest.approx(1.0, abs=1e-6)
        assert fitted.beta == pytest.approx(1.0, abs=1e-6)

    def test_symmetric_range_gives_symmetric_shape(self) -> None:
        fitted = _fit_beta("wide_symmetric")
        assert fitted.alpha == pytest.approx(fitted.beta, rel=1e-6)

    def test_wide_symmetric_is_u_shaped(self) -> None:
        """(0.1, 0.9) is heavier-tailed than uniform: α = β < 1 (spec §3 E9 —
        correct and intended, not an anomaly to clamp)."""
        fitted = _fit_beta("wide_symmetric")
        assert fitted.alpha < 1.0
        assert fitted.beta < 1.0

    def test_disclosure_fields_match_params(self) -> None:
        """The derived read-only disclosure fields are the distribution's own
        moments/quantiles — recomputed here from the parameters."""
        fitted = _fit_beta("mid_left")
        a_, b_ = fitted.alpha, fitted.beta
        assert fitted.mean == pytest.approx(a_ / (a_ + b_), rel=1e-12)
        var = a_ * b_ / ((a_ + b_) ** 2 * (a_ + b_ + 1.0))
        assert fitted.std == pytest.approx(math.sqrt(var), rel=1e-12)
        assert fitted.q25 == pytest.approx(_ppf(fitted, 0.25), rel=1e-12)
        assert fitted.q75 == pytest.approx(_ppf(fitted, 0.75), rel=1e-12)


class TestT4bBetaRoundTrip:
    @pytest.mark.parametrize("key", sorted(BETA_CORPUS))
    def test_round_trip_recovers_bounds(self, key: str) -> None:
        """fit (a,b) → (α,β) → ppf(0.25/0.75) recovers (a,b) within 1e−6.
        T2 specialised to the numerical family, kept separate so a beta-only
        regression is NAMED (spec §5)."""
        a, b = BETA_CORPUS[key]
        fitted = _fit_beta(key)
        assert abs(float(beta_dist.ppf(0.25, fitted.alpha, fitted.beta)) - a) <= 1e-6, key
        assert abs(float(beta_dist.ppf(0.75, fitted.alpha, fitted.beta)) - b) <= 1e-6, key


# ---------------------------------------------------------------------------
# T5 — refusal taxonomy (spec §3). Identity-bound: the EXACT code is asserted,
# and a constructor spy proves no FittedDistribution was built on any refusal
# path (not merely that nothing was returned).
# ---------------------------------------------------------------------------


class _CountingFitted(FittedDistribution):
    """Constructor spy (spec §5 T5.x): counts every construction."""

    constructions: ClassVar[int] = 0

    def __init__(self, **data: Any) -> None:
        type(self).constructions += 1
        super().__init__(**data)


@pytest.fixture
def spy_fitted(monkeypatch: pytest.MonkeyPatch) -> type:
    _CountingFitted.constructions = 0
    monkeypatch.setattr(range_fit_module, "FittedDistribution", _CountingFitted)
    return _CountingFitted


def _assert_refuses(
    code: str,
    lower: Optional[float],
    upper: Optional[float],
    domain: Domain,
    spy: type,
) -> RangeFitRefusalPayload:
    before = spy.constructions
    with pytest.raises(RangeFitRefusal) as excinfo:
        fit_range_distribution(lower=lower, upper=upper, domain=domain)
    payload = excinfo.value.payload
    assert payload.code == code, payload
    assert spy.constructions == before, "a FittedDistribution was constructed on a refusal path"
    return payload


class TestT5RefusalTaxonomy:
    def test_t5_1_zero_width(self, spy_fitted: type) -> None:
        _assert_refuses("RANGE_ZERO_WIDTH", 0.4, 0.4, "unit_interval", spy_fitted)
        _assert_refuses("RANGE_ZERO_WIDTH", 5.0, 5.0, "unbounded", spy_fitted)

    def test_t5_2_invalid_order_never_swapped(self, spy_fitted: type) -> None:
        """Refuse, never silently swap — an inverted range is a slip the user
        should see, not a convention to normalise (spec §3 E2, contra PLoT's
        translator posture)."""
        _assert_refuses("RANGE_INVALID_ORDER", 0.7, 0.3, "unit_interval", spy_fitted)
        _assert_refuses("RANGE_INVALID_ORDER", 2.0, -1.0, "unbounded", spy_fitted)

    def test_t5_3_non_finite(self, spy_fitted: type) -> None:
        _assert_refuses("RANGE_NON_FINITE", float("nan"), 0.5, "unit_interval", spy_fitted)
        _assert_refuses("RANGE_NON_FINITE", 0.1, float("inf"), "unbounded", spy_fitted)
        _assert_refuses("RANGE_NON_FINITE", float("-inf"), 0.5, "unbounded", spy_fitted)

    def test_t5_4_out_of_domain_never_clamped(self, spy_fitted: type) -> None:
        payload = _assert_refuses("RANGE_OUT_OF_DOMAIN", -0.1, 0.5, "unit_interval", spy_fitted)
        assert "unit_interval" in payload.message  # names the declared domain
        _assert_refuses("RANGE_OUT_OF_DOMAIN", 0.5, 1.2, "unit_interval", spy_fitted)

    def test_t5_5_at_domain_edge(self, spy_fitted: type) -> None:
        """F(0)=0 and F(1)=1 for every (α,β) — the constraint system is
        unsatisfiable at the exact edge, ∀ parameters (spec §3 E5, derived)."""
        _assert_refuses("RANGE_AT_DOMAIN_EDGE", 0.0, 0.5, "unit_interval", spy_fitted)
        _assert_refuses("RANGE_AT_DOMAIN_EDGE", 0.5, 1.0, "unit_interval", spy_fitted)

    def test_t5_6_open_ended(self, spy_fitted: type) -> None:
        """One bound + a coverage under-determines a distribution; choosing a
        second constraint is science, not engineering (spec §3 E7 → Neil)."""
        _assert_refuses("RANGE_OPEN_ENDED", None, 0.2, "unit_interval", spy_fitted)
        _assert_refuses("RANGE_OPEN_ENDED", 0.05, None, "unit_interval", spy_fitted)
        _assert_refuses("RANGE_OPEN_ENDED", None, None, "unbounded", spy_fitted)

    def test_t5_6b_open_ended_takes_precedence(self, spy_fitted: type) -> None:
        """An absent bound refuses OPEN_ENDED even when the present bound is
        itself non-finite — absence is checked before value semantics."""
        _assert_refuses("RANGE_OPEN_ENDED", None, float("nan"), "unbounded", spy_fitted)

    def test_t5_7_nonconvergence_refuses_loud(
        self, spy_fitted: type, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both solver starts failing acceptance ⇒ typed refusal, NEVER a
        fallback distribution (spec §2.4 / §3 E8 — the fabrication mutant M4's
        target). The solver is forced to fail via monkeypatch; acceptance is
        residual-based so the garbage answer cannot pass."""
        calls = {"n": 0}

        class _FailedSolution:
            success = True  # the flag LIES — acceptance must not believe it
            x = np.array([0.0, 0.0])

        def _failing_root(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            return _FailedSolution()

        monkeypatch.setattr(range_fit_module.scipy_optimize, "root", _failing_root)
        payload = _assert_refuses("RANGE_FIT_NONCONVERGENT", 0.2, 0.6, "unit_interval", spy_fitted)
        assert payload.starts_tried == 2  # the full two-start ladder was tried
        assert calls["n"] == 2

    def test_acceptance_is_independent_of_solver_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The solver's success flag is the solver's CLAIM; the recomputed
        residual is the measurement (spec §2.4). A correct answer carrying
        success=False must still be accepted."""
        real_root = range_fit_module.scipy_optimize.root

        def _lying_root(*args: Any, **kwargs: Any) -> Any:
            sol = real_root(*args, **kwargs)
            sol.success = False  # flag lies; residual is what counts
            return sol

        monkeypatch.setattr(range_fit_module.scipy_optimize, "root", _lying_root)
        fitted = fit_range_distribution(lower=0.2, upper=0.6, domain="unit_interval")
        assert abs(_cdf(fitted, 0.6) - _cdf(fitted, 0.2) - 0.5) <= MIDDLE_MASS_TOL


# ---------------------------------------------------------------------------
# T6 — determinism: bit-identical parameters, call to call (spec §2.6)
# ---------------------------------------------------------------------------


class TestT6Determinism:
    @pytest.mark.parametrize("key", ["hairline", "u_shape", "near_edge_low"])
    def test_beta_bit_identical(self, key: str) -> None:
        first = _fit_beta(key)
        second = _fit_beta(key)
        assert first.alpha == second.alpha  # exact float equality, deliberately
        assert first.beta == second.beta

    def test_normal_bit_identical(self) -> None:
        first = _fit_normal("large_magnitude")
        second = _fit_normal("large_magnitude")
        assert first.mu == second.mu
        assert first.sigma == second.sigma


# ---------------------------------------------------------------------------
# T7 — purity / zero RNG (spec §2.6): fitting between two draws leaves the
# post-fit draw byte-identical to the no-fit control, on BOTH the modern
# Generator API and the legacy global stream (kin to the point_mass stream
# finding, ground-truth #6 — anything touching the stream shifts every
# downstream sample).
# ---------------------------------------------------------------------------


class TestT7ZeroRNG:
    def test_generator_stream_untouched(self) -> None:
        rng_fit = np.random.default_rng(20260808)
        pre_fit = rng_fit.normal()
        for key in sorted(BETA_CORPUS):
            _fit_beta(key)
        for key in sorted(NORMAL_CORPUS):
            _fit_normal(key)
        post_fit = rng_fit.normal()

        rng_ctl = np.random.default_rng(20260808)
        pre_ctl = rng_ctl.normal()
        post_ctl = rng_ctl.normal()
        assert pre_fit == pre_ctl
        assert post_fit == post_ctl

    def test_legacy_global_stream_untouched(self) -> None:
        np.random.seed(20260808)
        pre_fit = np.random.normal()
        _fit_beta("hairline")
        _fit_normal("classic_unit")
        post_fit = np.random.normal()

        np.random.seed(20260808)
        pre_ctl = np.random.normal()
        post_ctl = np.random.normal()
        assert pre_fit == pre_ctl
        assert post_fit == post_ctl


# ---------------------------------------------------------------------------
# ROADMAP 2.916 — the fitter's NUMERIC error path.
#
# THE DEFECT (measured at staging tip fcba3754, scipy 1.16.3): `math.exp` RAISES
# `OverflowError` where numpy would return `inf`, and the moment-matched start's
# own arithmetic can divide by an `s²` that underflowed to zero. Both escaped
# `_fit_beta` as RAW exceptions, breaking the module's whole contract — spec
# §2.4/§3: fit, or typed refusal, never anything else.
#
# THE CLASS, not the instance (the property sweep found ONE input; this battery
# pins the family it belongs to). Two disjoint OverflowError sub-families plus a
# ZeroDivisionError family, each measured at pristine:
#
#  (a) WIDE near-uniform ranges — the moment-match guard `s² < m(1−m)` tests
#      FEASIBILITY (ν₀ > 0) and says nothing about CONDITIONING. As s² → m(1−m)
#      from below, ν₀ → 0⁺, so ln ν₀ is large-negative and hybr's Powell step
#      overshoots ln ν past ln(DBL_MAX) = 709.782713. Measured θ₁ reached: 762.7,
#      795.9, 1039.3, 13573.9. 18 of 79,401 grid pairs (grid 1/400).
#  (b) NARROW ranges — ν₀ already astronomical, hybr steps further up. At m = 0.5
#      centred, EVERY width from 1e−5 to 1e−15 raised.
#  (c) DENORMAL-magnitude bounds — s² underflows to exactly 0.0, so the ν₀
#      division raises ZeroDivisionError before any solve begins.
#
# WHY (a) MUST FIT — measured, not decreed. The raw exception escaped the
# start-ladder loop, so the ladder's SECOND start was never reached: the module's
# own designed recovery was bypassed by the crash. Driving the fallback start
# directly at pristine, every (a) row converges to residual 1.1e−16 … 3.2e−15 —
# seven-plus orders INSIDE the 1e−8 acceptance bound, on both platforms measured.
# So the honest outcome for (a) is a real fit, and the fix's job is to let the
# ladder do what it was built to do.
#
# WHY (b) AND (c) ASSERT THE DISJUNCTION AND NOT "MUST REFUSE" — a correction
# this battery earned the hard way. Its first version pinned "narrow ⇒ refuse"
# from a macOS/arm64 sweep; CI's Linux/x86-64 runner FITTED `(0.5 ± 5e−9)` and
# went red. The fit-vs-refuse outcome at the numerical margin belongs to the
# platform's libm/BLAS, not to the spec — so the invariant is the one the module
# actually promises (fit OR typed refusal, never a raw escape), with each branch
# carrying its own full invariant. A mutant kit could never have caught this: it
# measures whether a test can DETECT a change, never whether the EXPECTATION is
# right (trap 13c). Only running on a second machine could, and did.
#
# WHY `RANGE_FIT_NONCONVERGENT` AND NOT `RANGE_AT_DOMAIN_EDGE`: the edge code
# asserts unsatisfiability for EVERY (α, β) — a mathematical claim that is FALSE
# here, since a beta distribution demonstrably exists (the fallback start finds
# it for family (a)). Borrowing that code would ship a fabricated derivation
# wearing a real refusal type. E8 is the spec's exact member for "both starts
# failed acceptance", and it alone carries `starts_tried`.
# ---------------------------------------------------------------------------

# The exact input the 2.916 property run falsified, at full double precision.
FALSIFYING_INPUT_2916: Tuple[float, float] = (0.12758646191766157, 0.8000422250461583)

# Family (a) representatives, measured raw-raising at pristine. Identity-keyed so
# an assertion binds to the row it names, never to a value another row satisfies.
WIDE_NEAR_UNIFORM_OVERFLOW: Dict[str, Tuple[float, float]] = {
    "falsifier": FALSIFYING_INPUT_2916,
    "band_low": (0.0225, 0.6625),
    "band_mid": (0.085, 0.75),
    "band_high": (0.1075, 0.7775),
}

# Family (b): narrow. Family (c): denormal-magnitude bounds.
NARROW_OVERFLOW: Dict[str, Tuple[float, float]] = {
    "centred_1e5": (0.5 - 5e-6, 0.5 + 5e-6),
    "centred_1e8": (0.5 - 5e-9, 0.5 + 5e-9),
    "centred_1e13": (0.5 - 5e-14, 0.5 + 5e-14),
}
DENORMAL_WIDTH: Dict[str, Tuple[float, float]] = {
    "denormal_tiny": (1e-320, 2e-320),
    "denormal_min": (5e-324, 1e-323),
    "subnormal_square": (1e-200, 2e-200),
}


def _residual_of(fitted: FittedDistribution, a: float, b: float) -> float:
    """The spec §2.4 measurement: both CDF residuals recomputed at the RETURNED
    parameters, independent of the solver's own success flag."""
    return max(abs(_cdf(fitted, a) - 0.25), abs(_cdf(fitted, b) - 0.75))


class TestR2916NumericErrorPath:
    """Every member of the numeric-failure class resolves to fit-or-typed-refusal.

    A raw exception is asserted against BY TYPE here, not merely "did not raise":
    `pytest.raises(RangeFitRefusal)` would pass on a crash in a suite that only
    counted errors, so each case names what escaped.
    """

    @pytest.mark.parametrize("key", sorted(WIDE_NEAR_UNIFORM_OVERFLOW))
    def test_wide_near_uniform_band_fits_via_the_fallback_start(self, key: str) -> None:
        """Family (a): the ladder recovers and returns a REAL fit.

        Precondition PINNED IN-TEST (trap 13b — a guard whose discrimination
        rests on an unpinned fixture decays silently): this asserts the row is
        genuinely in the ill-conditioned moment-match regime — the moment start
        is FEASIBLE (so it is constructed and attempted) yet ν₀ is small enough
        that ln ν₀ sits deep in negative log-space. If a future edit moved these
        rows out of that regime the test would still pass while testing nothing,
        so the regime itself is asserted, not assumed.
        """
        a, b = WIDE_NEAR_UNIFORM_OVERFLOW[key]
        m = (a + b) / 2.0
        s = (b - a) / (2.0 * Z75)
        assert s * s < m * (1.0 - m), f"{key}: moment start must be FEASIBLE to be attempted"
        nu0 = m * (1.0 - m) / (s * s) - 1.0
        assert 0.0 < nu0 < 1e-2, f"{key}: must be the ill-conditioned regime, got nu0={nu0}"

        fitted = fit_range_distribution(lower=a, upper=b, domain="unit_interval")

        assert fitted.family == "beta"
        assert fitted.alpha is not None and fitted.alpha > 0 and math.isfinite(fitted.alpha)
        assert fitted.beta is not None and fitted.beta > 0 and math.isfinite(fitted.beta)
        assert _residual_of(fitted, a, b) <= ACCEPTANCE_RESIDUAL_TOLERANCE
        assert abs(_cdf(fitted, b) - _cdf(fitted, a) - 0.5) <= MIDDLE_MASS_TOL
        assert a < _ppf(fitted, 0.5) < b

    def test_the_exact_falsifying_input_is_pinned_by_value(self) -> None:
        """The 2.916 regression pin: the precise double pair the property run
        falsified, asserted at full precision so a re-rounding cannot quietly
        retire it."""
        a, b = FALSIFYING_INPUT_2916
        assert (a, b) == (0.12758646191766157, 0.8000422250461583)
        fitted = fit_range_distribution(lower=a, upper=b, domain="unit_interval")
        assert fitted.family == "beta"
        assert _residual_of(fitted, a, b) <= ACCEPTANCE_RESIDUAL_TOLERANCE

    @pytest.mark.parametrize("key", sorted(NARROW_OVERFLOW))
    def test_narrow_range_never_escapes_raw(self, key: str, spy_fitted: type) -> None:
        """Family (b): the contract holds — fit, or typed E8 refusal, never a raw
        exception.

        ⚠ ORACLE CORRECTED BY MEASUREMENT (trap 13c — a mutant kit validates a
        test's SENSITIVITY, never the truth of its EXPECTATION). This assertion
        was first written as "the narrow family must REFUSE", derived from a
        macOS/arm64 sweep. CI's Linux/x86-64 runner then FITTED `(0.5 ± 5e−9)`
        and the test went red — correctly. Whether a given narrow range
        converges is a property of the platform's libm/BLAS, NOT a promise this
        module makes; the promise is the DISJUNCTION (spec §2.4/§3), and pinning
        one branch of it pinned an accident of one machine.

        The disjunction is not a licence for "anything goes": the defect under
        test is the RAW ESCAPE, and each branch below carries the full invariant
        for that branch. Mutation-checked — reverting the fix REDs this on the
        raw-escape assertion, and `continue`→`break` leaves it green while REDing
        the wide family, so the two tests discriminate different properties.
        """
        a, b = NARROW_OVERFLOW[key]
        before = spy_fitted.constructions
        try:
            fitted = fit_range_distribution(lower=a, upper=b, domain="unit_interval")
        except RangeFitRefusal as refusal:
            payload = refusal.payload
            assert payload.code == "RANGE_FIT_NONCONVERGENT", payload
            assert payload.lower == a and payload.upper == b  # raw bounds echoed
            assert payload.starts_tried is not None and payload.starts_tried >= 1
            assert (
                spy_fitted.constructions == before
            ), "a FittedDistribution was constructed on a refusal path"
            return
        except Exception as raw:  # noqa: BLE001 — this IS the 2.916 defect
            raise AssertionError(
                f"RAW {type(raw).__name__} escaped for {key} (a={a!r}, b={b!r}): {raw}. "
                f"The contract is fit-or-typed-refusal; a numeric failure must never "
                f"reach the product as an exception."
            ) from raw
        # Fit branch: a fit is only acceptable if it is a GOOD fit.
        assert fitted.family == "beta"
        assert fitted.alpha is not None and fitted.alpha > 0 and math.isfinite(fitted.alpha)
        assert fitted.beta is not None and fitted.beta > 0 and math.isfinite(fitted.beta)
        assert _residual_of(fitted, a, b) <= ACCEPTANCE_RESIDUAL_TOLERANCE

    @pytest.mark.parametrize("key", sorted(DENORMAL_WIDTH))
    def test_denormal_width_never_escapes_raw(self, key: str, spy_fitted: type) -> None:
        """Family (c): s² underflows to 0.0 and the ν₀ division raised
        ZeroDivisionError before any solve. These are IN-CONTRACT inputs — spec
        §3 E6 forbids an epsilon floor, so tiny-but-interior bounds must be
        answered, not crashed on.

        Same disjunction as family (b), for the same measured reason: the
        underflow itself is IEEE-754 deterministic, but what the fallback start
        then does with the range is the platform's business, not the spec's.
        """
        a, b = DENORMAL_WIDTH[key]
        assert 0.0 < a < b < 1.0, f"{key} must be interior to the declared domain"
        before = spy_fitted.constructions
        try:
            fitted = fit_range_distribution(lower=a, upper=b, domain="unit_interval")
        except RangeFitRefusal as refusal:
            assert refusal.payload.code == "RANGE_FIT_NONCONVERGENT", refusal.payload
            assert (
                spy_fitted.constructions == before
            ), "a FittedDistribution was constructed on a refusal path"
            return
        except Exception as raw:  # noqa: BLE001 — this IS the 2.916 defect
            raise AssertionError(
                f"RAW {type(raw).__name__} escaped for {key} (a={a!r}, b={b!r}): {raw}. "
                f"The contract is fit-or-typed-refusal; a numeric failure must never "
                f"reach the product as an exception."
            ) from raw
        assert fitted.alpha is not None and fitted.alpha > 0 and math.isfinite(fitted.alpha)
        assert _residual_of(fitted, a, b) <= ACCEPTANCE_RESIDUAL_TOLERANCE

    @pytest.mark.parametrize("key", sorted(BETA_CORPUS))
    def test_positive_control_healthy_fits_keep_their_residual(self, key: str) -> None:
        """POSITIVE CONTROL for the fix: the ratified corpus must still fit, and
        still fit WELL. Without this, the numeric error path could be 'fixed' by
        refusing everything and the class tests above would all pass.

        The bound is 1e−9 — a full order TIGHTER than the module's 1e−8
        acceptance — for every row except `hairline`, whose looser bound is not a
        concession to this change but the documented scipy-1.16.3 `beta.cdf`
        evaluation floor at α = β ≈ 2.27e11 (see ACCEPTANCE_RESIDUAL_TOLERANCE's
        premise-correction note). Binding by corpus KEY, never by magnitude.
        """
        a, b = BETA_CORPUS[key]
        fitted = _fit_beta(key)
        bound = ACCEPTANCE_RESIDUAL_TOLERANCE if key == "hairline" else 1e-9
        assert _residual_of(fitted, a, b) <= bound

    def test_normal_path_is_unaffected_by_extreme_magnitudes(self) -> None:
        """The class audit swept the normal path for the same failure and found
        it CLEAN — its closed form has no `exp` and its existing finiteness guard
        already refuses. Pinned so a later 'symmetry' edit cannot regress it into
        the beta path's shape."""
        for a, b in [(-1.5e308, 1.5e308), (1e-320, 2e-320), (-1e300, 1e300), (0.0, 1.7e308)]:
            try:
                fitted = fit_range_distribution(lower=a, upper=b, domain="unbounded")
                assert fitted.family == "normal"
                assert fitted.sigma is not None and fitted.sigma > 0
            except RangeFitRefusal as refusal:
                assert refusal.payload.code == "RANGE_FIT_NONCONVERGENT"


# ---------------------------------------------------------------------------
# Hypothesis property sweeps (spec §5 T1: hypothesis-generated interior pairs).
# The GUARANTEE under test is acceptance-or-refusal (spec §2.4 relies on no
# existence theorem): every generated pair must either fit correctly or refuse
# in type with RANGE_FIT_NONCONVERGENT — any other outcome fails. Vacuity is
# guarded by the fixed 11-row corpus above, which must ALL fit.
# ---------------------------------------------------------------------------


@st.composite
def _normal_pairs(draw: Any) -> Tuple[float, float]:
    """Width scales with magnitude (min relative width 1e−9): below that, μ's
    own double representation (ulp(a) vs σ) dominates — a REPRESENTATION limit
    of the inputs, not a property of the fit. Measured: (268435456.0,
    268435456.000001) puts half an ulp of μ at 4% of σ, displacing the middle
    mass by 0.215·(δ/σ)² ≈ 3.4e−4 with mathematically exact parameters."""
    a = draw(st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False))
    width_frac = draw(
        st.floats(min_value=1e-9, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return a, a + width_frac * max(1.0, abs(a))


@st.composite
def _beta_pairs(draw: Any) -> Tuple[float, float]:
    """Two independent interior draws, ordered — avoids the empty-interval
    boundary a dependent second draw hits when a is at the top of its range."""
    x = draw(st.floats(min_value=1e-6, max_value=1.0 - 1e-6))
    y = draw(st.floats(min_value=1e-6, max_value=1.0 - 1e-6))
    assume(abs(y - x) >= 1e-6)
    return min(x, y), max(x, y)


class TestPropertySweeps:
    @settings(max_examples=150, deadline=None)
    @given(pair=_normal_pairs())
    def test_normal_property(self, pair: Tuple[float, float]) -> None:
        a, b = pair
        if not (math.isfinite(a) and math.isfinite(b) and a < b):
            return
        fitted = fit_range_distribution(lower=a, upper=b, domain="unbounded")
        scale = max(1.0, abs(a), abs(b))
        assert abs(_cdf(fitted, b) - _cdf(fitted, a) - 0.5) <= MIDDLE_MASS_TOL
        assert abs(_ppf(fitted, 0.25) - a) <= 1e-6 * scale
        assert abs(_ppf(fitted, 0.75) - b) <= 1e-6 * scale
        assert a < _ppf(fitted, 0.5) < b

    @settings(max_examples=150, deadline=None)
    @given(pair=_beta_pairs())
    def test_beta_property_fit_or_typed_refusal(self, pair: Tuple[float, float]) -> None:
        a, b = pair
        if not (0.0 < a < b < 1.0):
            return
        try:
            fitted = fit_range_distribution(lower=a, upper=b, domain="unit_interval")
        except RangeFitRefusal as refusal:
            # Honest refusal is spec-conformant; a WRONG code is not.
            assert refusal.payload.code == "RANGE_FIT_NONCONVERGENT"
            return
        except Exception as raw:  # noqa: BLE001 — see ROADMAP 2.916
            # A RAW escape is the 2.916 defect itself, and it gets its OWN named
            # failure here rather than arriving as an anonymous hypothesis error.
            # This matters for diagnosis, not decoration: an uncaught OverflowError
            # surfaces as a bare "math range error" whose traceback points at
            # `math.exp` inside the residual — which reads like a solver problem
            # and hides both the contract that was broken (fit-or-typed-refusal)
            # and the input that broke it. `raise ... from raw` keeps the original
            # traceback attached while putting the contract violation in the
            # headline.
            raise AssertionError(
                f"RAW {type(raw).__name__} escaped fit_range_distribution for "
                f"(a={a!r}, b={b!r}, domain='unit_interval'): {raw}. The converter's "
                f"contract is fit-or-typed-refusal (spec §2.4/§3) — a numeric "
                f"failure must surface as RangeFitRefusal, never as an exception "
                f"the product would render as a 500."
            ) from raw
        assert fitted.alpha > 0 and math.isfinite(fitted.alpha)
        assert fitted.beta > 0 and math.isfinite(fitted.beta)
        assert abs(_cdf(fitted, b) - _cdf(fitted, a) - 0.5) <= MIDDLE_MASS_TOL
        assert a < _ppf(fitted, 0.5) < b


# ---------------------------------------------------------------------------
# Resolver: every refusal code the converter can emit gets a disclosure entry
# AND a named inference warning (spec §4.3, R-10 pattern; mutant M5's target —
# a refusal the user never sees is a silent default with extra steps).
# The code vocabulary is DERIVED from the type, never hand-listed (trap 12).
# ---------------------------------------------------------------------------


def _trigger_for(code: str) -> UserStatedRange:
    """A UserStatedRange that provokes exactly `code` through the resolver.
    RANGE_FIT_NONCONVERGENT is provoked via solver monkeypatch by its test."""
    triggers: Dict[str, UserStatedRange] = {
        "RANGE_ZERO_WIDTH": UserStatedRange(
            node_id="n_zero", lower=0.4, upper=0.4, domain="unit_interval"
        ),
        "RANGE_INVALID_ORDER": UserStatedRange(
            node_id="n_order", lower=0.7, upper=0.3, domain="unit_interval"
        ),
        "RANGE_NON_FINITE": UserStatedRange(
            node_id="n_nonfinite", lower=float("nan"), upper=0.5, domain="unbounded"
        ),
        "RANGE_OUT_OF_DOMAIN": UserStatedRange(
            node_id="n_domain", lower=-0.1, upper=0.5, domain="unit_interval"
        ),
        "RANGE_AT_DOMAIN_EDGE": UserStatedRange(
            node_id="n_edge", lower=0.0, upper=0.5, domain="unit_interval"
        ),
        "RANGE_OPEN_ENDED": UserStatedRange(
            node_id="n_open", lower=None, upper=0.2, domain="unit_interval"
        ),
        "RANGE_FIT_NONCONVERGENT": UserStatedRange(
            node_id="n_nonconv", lower=0.2, upper=0.6, domain="unit_interval"
        ),
    }
    return triggers[code]


class TestResolverEveryCodeHasABucket:
    def test_all_codes_surface_as_disclosure_and_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        all_codes = set(get_args(RangeFitRefusalCode))
        assert all_codes, "code vocabulary must be non-empty"

        # Trigger table completeness is asserted against the DERIVED vocabulary:
        # adding a code without a trigger REDs here (fail-loud, not assume-good).
        for code in sorted(all_codes):
            trigger = _trigger_for(code)  # KeyError = missing trigger = RED

            if code == "RANGE_FIT_NONCONVERGENT":

                class _Failed:
                    success = False
                    x = np.array([0.0, 0.0])

                monkeypatch.setattr(
                    range_fit_module.scipy_optimize,
                    "root",
                    lambda *a, **k: _Failed(),
                )

            disclosures, warnings = resolve_range_fits([trigger])

            if code == "RANGE_FIT_NONCONVERGENT":
                monkeypatch.undo()

            assert disclosures is not None and len(disclosures) == 1, code
            disclosure = disclosures[0]
            assert disclosure.node_id == trigger.node_id, code  # identity-bound
            assert disclosure.fitted is None, code
            assert disclosure.refusal is not None, code
            assert disclosure.refusal.code == code
            assert len(warnings) == 1, code
            warning = warnings[0]
            assert warning.code == code
            assert warning.severity == "warning"  # degradation class, never quiet
            assert warning.field == f"user_stated_ranges[{trigger.node_id}]"

    def test_successful_fit_produces_disclosure_and_no_warning(self) -> None:
        stated = UserStatedRange(
            node_id="n_ok", lower=0.2, upper=0.6, domain="unit_interval", source="user"
        )
        disclosures, warnings = resolve_range_fits([stated])
        assert disclosures is not None and len(disclosures) == 1
        disclosure = disclosures[0]
        assert disclosure.node_id == "n_ok"
        assert disclosure.refusal is None
        assert disclosure.fitted is not None
        assert disclosure.fitted.family == "beta"
        assert disclosure.lower == 0.2 and disclosure.upper == 0.6  # raw echo
        assert warnings == []

    def test_absent_ranges_resolve_to_nothing(self) -> None:
        assert resolve_range_fits(None) == (None, [])
        assert resolve_range_fits([]) == (None, [])

    def test_family_comes_from_declared_domain_not_values(self) -> None:
        """The SAME values fit different families under different DECLARED
        domains — family is metadata-derived, never value-sniffed (spec §2.2)."""
        in_unit = fit_range_distribution(lower=0.2, upper=0.6, domain="unit_interval")
        unbounded = fit_range_distribution(lower=0.2, upper=0.6, domain="unbounded")
        assert in_unit.family == "beta"
        assert unbounded.family == "normal"


# ---------------------------------------------------------------------------
# Model invariants
# ---------------------------------------------------------------------------


class TestModelInvariants:
    def test_disclosure_requires_exactly_one_of_fitted_or_refusal(self) -> None:
        fitted = fit_range_distribution(lower=0.2, upper=0.6, domain="unit_interval")
        payload = RangeFitRefusalPayload(
            code="RANGE_ZERO_WIDTH", message="m", lower=0.1, upper=0.1, domain="unit_interval"
        )
        with pytest.raises(ValueError):
            RangeFitDisclosure(
                node_id="n",
                lower=0.2,
                upper=0.6,
                domain="unit_interval",
                fitted=None,
                refusal=None,
            )
        with pytest.raises(ValueError):
            RangeFitDisclosure(
                node_id="n",
                lower=0.2,
                upper=0.6,
                domain="unit_interval",
                fitted=fitted,
                refusal=payload,
            )

    def test_coverage_constant_is_the_ratified_half(self) -> None:
        """RATIFIED_COVERAGE is Neil's ruling (2.521 Q1): a stated range is an
        ≈50% credible interval. A different value is a different ruling."""
        assert RATIFIED_COVERAGE == 0.5

    def test_fitted_distribution_family_params_are_coherent(self) -> None:
        with pytest.raises(ValueError):
            FittedDistribution(
                family="normal",
                mu=None,
                sigma=None,
                alpha=1.0,
                beta=1.0,
                mean=0.5,
                std=0.1,
                q25=0.4,
                q75=0.6,
                coverage=0.5,
                method_version=RANGE_FIT_METHOD_VERSION,
            )
