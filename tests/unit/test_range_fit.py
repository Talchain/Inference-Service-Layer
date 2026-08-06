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
