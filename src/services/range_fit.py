"""
Range→distribution converter — the pure fitter and the resolution seam
(ROADMAP 2.720; spec parallel-briefs/RANGE-TO-DISTRIBUTION-SPEC-2026-08-08.md).

Neil's 2.521 Q1 ruling, implemented exactly: a user-stated closed range
``[a, b]`` is an approximately 50% credible interval; the fit places the
distribution's QUARTILES on the stated bounds with equal 25% tails — beta for
bounded quantities, normal for unbounded scalars (Quillien, Bramley & Lucas
2025: listeners decode interval guesses as N(midpoint, k·width) with best-fit
k = 0.74; the interquartile constant 1/(2·Φ⁻¹(0.75)) = 0.7413 reproduces it).

PURITY IS LOAD-BEARING, not hygiene (spec §2.6): ISL analyses are
seed-reproducible and anything touching a draw stream shifts every downstream
sample. The converter consumes ZERO RNG draws, does no I/O, and identical
inputs produce bit-identical parameters (hybr is deterministic; the two-start
ladder is ordered; nothing samples). Tested (T7), not asserted.

In S3 the fit is computed, echoed and disclosed while compute stays
byte-identical (T9); only the application to the blend (S4) waits, on 2.521 Q2
plus the combination ruling — neither of which this module decides (spec §0).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import scipy.optimize as scipy_optimize  # type: ignore[import-untyped]
from scipy.special import expit, logit  # type: ignore[import-untyped]
from scipy.stats import beta as beta_dist  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]

from src.models.range_fit import (
    RANGE_FIT_METHOD_VERSION,
    Domain,
    FittedDistribution,
    RangeFitDisclosure,
    RangeFitRefusal,
    RangeFitRefusalPayload,
    UserStatedRange,
)
from src.models.response_v2 import InferenceWarning

# ---------------------------------------------------------------------------
# Constants — each written ONCE, everything else derived (trap 12)
# ---------------------------------------------------------------------------

# Neil's ruling, 2.521 Q1 (recorded verbatim in ROADMAP 2.521, grounded in
# Quillien, Bramley & Lucas 2025 §5.7: stated intervals are ≈50% likely to
# contain the answer; Yaniv & Foster 1997 measured 43–46% hit rates). A
# parameter with a ratified default, NOT a hardcoded quantile pair: a future
# "how sure are you?" elicitation becomes this argument with the same default.
RATIFIED_COVERAGE = 0.5

# Acceptance bound for the recomputed CDF residuals (spec §2.4 form: residual
# recomputed at the returned parameters, independent of the solver's flag).
#
# ⚠ PREMISE CORRECTION, measured 6 Aug 2026 (spec said 1e−9): the spec's §2.4
# corpus run was executed on scipy 1.11.x, but poetry.lock pins scipy 1.16.3 —
# the deployed bytes. At 1.16.3, `beta.cdf`'s own evaluation floor at extreme
# concentration is ABOVE 1e−9: the (0.5, 0.500001) corpus row (α = β ≈ 2.27e11)
# bottoms out at max-residual 7.55e−9, with ±5e−10 θ-perturbations bouncing to
# ~3.8e−6 (a Boost betainc branch edge). Every other corpus row measures
# ≤ 7.1e−11. A 1e−9 acceptance would therefore REFUSE a row the ratified corpus
# requires to FIT — not because the fit is wrong (the quantile displacement of
# a 1e−8 CDF residual is residual/pdf ≈ 2.4e−14 there) but because the ruler is
# coarser than the tolerance. 1e−8 clears the measured floor while preserving
# the guard's entire intent: a fabricated fallback (e.g. Beta(1,1) for a
# non-(0.25, 0.75) range) misses by ~1e−1 — seven orders of magnitude above.
# The middle-mass test tolerance DERIVES from two of these — never a third
# constant.
ACCEPTANCE_RESIDUAL_TOLERANCE = 1e-8


def _quantile_targets(coverage: float) -> Tuple[float, float]:
    """Quantile targets derived from coverage, never written twice (spec §2.1):
    at the ratified 0.5 these are exactly (0.25, 0.75) — the stated bounds are
    the fitted QUARTILES, equal tails."""
    return (1.0 - coverage) / 2.0, (1.0 + coverage) / 2.0


def _refuse(
    code: str,
    message: str,
    lower: Optional[float],
    upper: Optional[float],
    domain: Domain,
    starts_tried: Optional[int] = None,
) -> RangeFitRefusal:
    return RangeFitRefusal(
        RangeFitRefusalPayload(
            code=code,  # type: ignore[arg-type]  # closed vocabulary, callers pass literals
            message=message,
            lower=lower,
            upper=upper,
            domain=domain,
            starts_tried=starts_tried,
        )
    )


# ---------------------------------------------------------------------------
# The pure converter
# ---------------------------------------------------------------------------


def fit_range_distribution(
    lower: Optional[float],
    upper: Optional[float],
    domain: Domain,
    coverage: float = RATIFIED_COVERAGE,
) -> FittedDistribution:
    """Fit the distribution whose middle ``coverage`` of probability mass lies
    between the stated bounds, quartiles ON the bounds (at the ratified 0.5).

    Pure function: no I/O, no RNG, deterministic. Raises the typed
    ``RangeFitRefusal`` (spec §3) — NEVER returns a fallback.

    The ``Optional`` bounds exist so an open-ended statement is expressible
    and refused loudly (E7); the spec's contract is two-sided ranges only.
    """
    if not (0.0 < coverage < 1.0):
        # Programmer error, not user input — no other value is reachable today
        # (spec §2.5); the elicitation UI that would supply one is unspecced.
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")

    # E7 — open-ended: one bound absent. Checked FIRST: absence precedes value
    # semantics (a missing bound refuses OPEN_ENDED even if the present bound
    # is itself non-finite).
    if lower is None or upper is None:
        raise _refuse(
            "RANGE_OPEN_ENDED",
            "Open-ended ranges ('at least X' / 'no more than X') are not ruled: "
            "one bound plus a coverage does not determine a distribution. "
            "State both bounds.",
            lower,
            upper,
            domain,
        )

    # E3 — non-finite bounds. Refused, never silently skipped.
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise _refuse(
            "RANGE_NON_FINITE",
            "Range bounds must be finite numbers.",
            lower,
            upper,
            domain,
        )

    # E2 — inverted order. Refused, never silently swapped: an inverted range
    # is more likely a slip the user should see than a convention to normalise.
    if lower > upper:
        raise _refuse(
            "RANGE_INVALID_ORDER",
            "The lower bound is greater than the upper bound. Order is part of "
            "what was said — please restate the range.",
            lower,
            upper,
            domain,
        )

    # E1 — zero width. A zero-width '50% CI' asserts a certainty the ruling's
    # semantics cannot represent; certainty is the confirmed-value path.
    if lower == upper:
        raise _refuse(
            "RANGE_ZERO_WIDTH",
            "The range has zero width. State a range with some width, or "
            "confirm the value alone.",
            lower,
            upper,
            domain,
        )

    q_lo, q_hi = _quantile_targets(coverage)

    if domain == "unbounded":
        return _fit_normal(lower, upper, domain, coverage, q_lo, q_hi)
    return _fit_beta(lower, upper, domain, coverage, q_lo, q_hi)


def _fit_normal(
    a: float,
    b: float,
    domain: Domain,
    coverage: float,
    q_lo: float,
    q_hi: float,
) -> FittedDistribution:
    """Closed form, no solver (spec §2.3): the general two-quantile solution
    ``σ = (b−a)/(z_hi−z_lo), μ = a − z_lo·σ``, which at the ratified symmetric
    targets reduces exactly to ``μ = (a+b)/2, σ = (b−a)/(2·Φ⁻¹(0.75))``. The
    z constants are DERIVED from the coverage parameter — never a literal
    0.6745 beside a literal 0.25 (two mirrors of one decision, trap 12)."""
    z_lo = float(norm.ppf(q_lo))
    z_hi = float(norm.ppf(q_hi))
    sigma = (b - a) / (z_hi - z_lo)
    mu = a - z_lo * sigma

    # Post-fit assertions even in the closed form (spec §2.3): extreme-magnitude
    # doubles can overflow b−a. Failure refuses — never clamps.
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0.0):
        raise _refuse(
            "RANGE_FIT_NONCONVERGENT",
            "The normal fit produced non-finite parameters for this range.",
            a,
            b,
            domain,
        )

    return FittedDistribution(
        family="normal",
        mu=mu,
        sigma=sigma,
        mean=mu,
        std=sigma,
        q25=float(norm.ppf(q_lo, loc=mu, scale=sigma)),
        q75=float(norm.ppf(q_hi, loc=mu, scale=sigma)),
        coverage=coverage,
        method_version=RANGE_FIT_METHOD_VERSION,
    )


def _fit_beta(
    a: float,
    b: float,
    domain: Domain,
    coverage: float,
    q_lo: float,
    q_hi: float,
) -> FittedDistribution:
    """2-D quantile fit (spec §2.4): solve F_Beta(a)=q_lo, F_Beta(b)=q_hi with
    MINPACK hybrid Powell in MEAN/LOG-CONCENTRATION space.

    The parameterisation θ = (logit μ_B, ln ν), α = μ_B·ν, β = (1−μ_B)·ν is
    LOAD-BEARING and was measured during spec-writing: the naive (ln α, ln β)
    space FAILS on very narrow ranges ((0.5, 0.500001) stalls hybr from the
    moment-matched start α₀=β₀≈2.3e11) while this space converges the same
    case to residual 8.1e−10. Positivity is structural — no clipping, no
    penalty, no invalid parameter reachable. Do NOT "simplify" this back.
    """
    # E4 — outside the declared support. Never clamped: clamping invents a
    # different range than the user stated.
    if a < 0.0 or b > 1.0:
        raise _refuse(
            "RANGE_OUT_OF_DOMAIN",
            "The stated range lies outside the declared 'unit_interval' domain "
            "[0, 1]. Restate the range within the quantity's domain.",
            a,
            b,
            domain,
        )

    # E5 — exact edge. Derived, not decreed: F(0)=0 and F(1)=1 for EVERY
    # (α, β) ∈ (0,∞)², with no atoms — so F(a)=q_lo requires a>0 and F(b)=q_hi
    # requires b<1. Unsatisfiable ∀ parameters. (Near-edge bounds, e.g. 1e−6,
    # are legitimate extreme shapes and are FITTED — the guard is acceptance,
    # not an epsilon floor, which would be an undisclosed constant.)
    if a == 0.0 or b == 1.0:
        raise _refuse(
            "RANGE_AT_DOMAIN_EDGE",
            "A bound sits exactly at the domain edge; no beta distribution can "
            "place a quartile there. If the value could truly be at the edge, "
            "that is an open-ended statement (not yet ruled).",
            a,
            b,
            domain,
        )

    def residual(theta: Sequence[float]) -> List[float]:
        mu_b = float(expit(theta[0]))
        nu = float(math.exp(theta[1]))
        alpha = mu_b * nu
        beta_p = (1.0 - mu_b) * nu
        return [
            float(beta_dist.cdf(a, alpha, beta_p)) - q_lo,
            float(beta_dist.cdf(b, alpha, beta_p)) - q_hi,
        ]

    def recover(theta: Sequence[float]) -> Tuple[float, float]:
        mu_b = float(expit(theta[0]))
        nu = float(math.exp(theta[1]))
        return mu_b * nu, (1.0 - mu_b) * nu

    def accepted(alpha: float, beta_p: float) -> bool:
        """Acceptance — independent of the solver's own success flag (the flag
        is the solver's claim; the recomputed residual is the measurement)."""
        if not (math.isfinite(alpha) and math.isfinite(beta_p) and alpha > 0.0 and beta_p > 0.0):
            return False
        res_a = abs(float(beta_dist.cdf(a, alpha, beta_p)) - q_lo)
        res_b = abs(float(beta_dist.cdf(b, alpha, beta_p)) - q_hi)
        return max(res_a, res_b) <= ACCEPTANCE_RESIDUAL_TOLERANCE

    # Deterministic two-start ladder (spec §2.4) — no randomised multistart.
    m = (a + b) / 2.0
    z_lo = float(norm.ppf(q_lo))
    z_hi = float(norm.ppf(q_hi))
    s = (b - a) / (z_hi - z_lo)  # the §2.3 σ, moment-matched into (0, 1)

    starts: List[Tuple[float, float]] = []
    if s * s < m * (1.0 - m):
        nu0 = m * (1.0 - m) / (s * s) - 1.0
        starts.append((float(logit(m)), math.log(nu0)))
    # Fallback start (also the ONLY start when moment-match is infeasible for
    # a very wide range): concentration 1 at the midpoint.
    starts.append((float(logit(m)), 0.0))

    starts_tried = 0
    for theta0 in starts:
        starts_tried += 1
        solution = scipy_optimize.root(residual, list(theta0), method="hybr")
        alpha, beta_p = recover(solution.x)
        if accepted(alpha, beta_p):
            # Legitimate fits can carry very large concentrations (the
            # 1e−6-width corpus row fits α = β ≈ 2.27e11): large is not wrong,
            # and no cap is applied — a cap is an undisclosed constant.
            # Finiteness + the recomputed residual are the guards.
            mean = alpha / (alpha + beta_p)
            variance = (alpha * beta_p) / ((alpha + beta_p) ** 2 * (alpha + beta_p + 1.0))
            return FittedDistribution(
                family="beta",
                alpha=alpha,
                beta=beta_p,
                mean=mean,
                std=math.sqrt(variance),
                q25=float(beta_dist.ppf(q_lo, alpha, beta_p)),
                q75=float(beta_dist.ppf(q_hi, alpha, beta_p)),
                coverage=coverage,
                method_version=RANGE_FIT_METHOD_VERSION,
            )

    # E8 — both starts failed acceptance ⇒ REFUSE LOUD. Never a silent
    # fallback: a Beta(1,1) or moment-matched normal returned here would be a
    # fabricated value wearing real provenance (this estate's dominant defect).
    raise _refuse(
        "RANGE_FIT_NONCONVERGENT",
        "The beta quantile fit did not converge for this range.",
        a,
        b,
        domain,
        starts_tried=starts_tried,
    )


# ---------------------------------------------------------------------------
# The resolution seam (S3 half): fit once, disclose always
# ---------------------------------------------------------------------------


def resolve_range_fits(
    stated_ranges: Optional[Sequence[UserStatedRange]],
) -> Tuple[Optional[List[RangeFitDisclosure]], List[InferenceWarning]]:
    """Resolve every user-stated range to a disclosure entry: fitted when the
    converter accepts, the typed refusal PLUS a named 'warning'-severity
    inference warning when it refuses (spec §4.3 / R-10 pattern — a refusal
    the user never sees is a silent default with extra steps; mutant M5's
    target).

    Absent/empty input resolves to (None, []) — inert-when-absent, so every
    consumer that has not opted in observes a byte-identical response.

    In S3 the disclosures are echo only: compute NEVER reads them (T9 proves
    byte-identity). The future S4 resolver (the six-site free function over
    (node, uncertainty)) will call the same ``fit_range_distribution`` once
    per node when application is ruled.
    """
    if not stated_ranges:
        return None, []

    disclosures: List[RangeFitDisclosure] = []
    warnings: List[InferenceWarning] = []
    for stated in stated_ranges:
        try:
            fitted = fit_range_distribution(
                lower=stated.lower, upper=stated.upper, domain=stated.domain
            )
            disclosures.append(
                RangeFitDisclosure(
                    node_id=stated.node_id,
                    lower=stated.lower,
                    upper=stated.upper,
                    domain=stated.domain,
                    fitted=fitted,
                    refusal=None,
                )
            )
        except RangeFitRefusal as refusal:
            disclosures.append(
                RangeFitDisclosure(
                    node_id=stated.node_id,
                    lower=stated.lower,
                    upper=stated.upper,
                    domain=stated.domain,
                    fitted=None,
                    refusal=refusal.payload,
                )
            )
            warnings.append(
                InferenceWarning(
                    code=refusal.payload.code,
                    field=f"user_stated_ranges[{stated.node_id}]",
                    detail={
                        "message": refusal.payload.message,
                        "lower": refusal.payload.lower,
                        "upper": refusal.payload.upper,
                        "domain": refusal.payload.domain,
                        "starts_tried": refusal.payload.starts_tried,
                    },
                    # Degradation-disclosure class (the GOAL_THRESHOLD_* posture):
                    # the fitted distribution was WITHHELD — the reason is the
                    # whole payload; it must not be filed under the quiet 'info'
                    # that PLoT hides.
                    severity="warning",
                )
            )
    return disclosures, warnings
