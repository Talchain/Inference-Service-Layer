"""Gaussian-copula correlation support for factor-value sampling (B3-S1).

Builds a *factor-correlation plan* from client-supplied pairwise correlations:

1. assembles the correlation matrix over the correlated factor set,
2. checks positive-semidefiniteness (PSD),
3. when the assembled matrix is NOT PSD, applies the Higham (2002)
   nearest-correlation projection (alternating projections, eigen-based) and
   records a disclosure payload, and
4. Cholesky-factorises the (possibly projected) matrix for the copula draw.

Doctrine (D-23.4, research-sharpened):

- This module is only ever exercised when the request supplies
  ``factor_correlations`` — it is inert-when-absent. Nothing here consumes an
  RNG stream; the draw itself lives in ``FactorSampler``.
- Non-PSD inputs are PROJECTED with disclosure (Higham 2002), never repaired by
  the dormant PLoT "symmetrize+clamp+shrink" heuristic (which is not
  nearest-PSD).
- Cholesky (not an eigen square-root) is used deliberately: ``chol(I) == I``
  exactly, so an identity (all-``rho=0``) matrix leaves the normal draws
  untouched to full bit-identity — an eigen square-root would rotate by an
  arbitrary eigenbasis and break that guarantee.

Pure NumPy — no SciPy, no new dependency surface. The standard-normal CDF used
by the marginal (copula) transform lives in ``FactorSampler`` (``math.erfc``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Method identifiers surfaced on the wire (correlation_model disclosure block).
CORRELATION_METHOD = "gaussian_copula_v1"
HIGHAM_METHOD = "higham_2002_nearest_correlation"

# Higham alternating-projections controls — deterministic + iteration-capped.
HIGHAM_MAX_ITER = 100
HIGHAM_TOL = 1e-10

# PSD acceptance floor: the assembled matrix is treated as already-PSD (no
# projection) when its smallest eigenvalue is >= -PSD_EIGEN_TOL. Tiny negative
# eigenvalues from floating-point noise are tolerated; genuine indefiniteness is
# projected + disclosed.
PSD_EIGEN_TOL = 1e-10

# An UNSTATED (zero-filled) pair is disclosed in the effective-matrix payload (F-2)
# only when the projection moved its off-diagonal off 0 by more than this tolerance —
# far above the Higham residual float-noise floor (convergence ~1e-10), far below any
# meaningful correlation signal.
EFFECTIVE_DISCLOSURE_MOVE_TOL = 1e-9

# ---------------------------------------------------------------------------
# Hard-invalid admissibility band (F4, D-23.13 — enforcing D-23.4 +
# PARAMETER-RESEARCH-2026-07-23.md:49-59). The agreed doctrine is "reject
# hard-invalid, ELSE Higham-project near-PSD with disclosure". Higham (2002) was
# designed to REPAIR matrices that are *almost* valid (pairwise / partial-data
# estimation, differing sample windows, rounding) — where the smallest eigenvalue
# is only slightly negative. A matrix that is indefinite by a WIDE margin is not
# noise: it encodes mutually contradictory correlations (e.g. rho(a,b)=1,
# rho(a,c)=1, rho(b,c)=-1 has spectrum [-1,2,2] — a==b, a==c but rho(b,c)=-1), and
# projecting it silently analyses a DIFFERENT problem under materially different
# assumptions. Such inputs are rejected (typed 422) at request validation, BEFORE
# any projection reaches the sampler; near-PSD inputs still project + disclose.
#
# NEIL-PARAMETERS (operational defaults, NOT research-calibrated) — like
# CVAR_LEVEL / STABILITY_CONFIDENCE_MAP. A silent retune is caught by the
# fingerprint guard in tests/unit/test_correlation_hard_invalid.py, which pins the
# (constants fingerprint <-> version) pair and fails loud if a constant moves while
# the version stays put. To change a bound: bump CORRELATION_ADMISSION_METHOD_VERSION
# AND update the pinned fingerprint together.
#
# Rationale for the values (documented, defensible):
# - min-eigenvalue floor -0.05: on a unit-diagonal correlation matrix the
#   eigenvalues sum to n and are each in [0, n] when valid; a smallest eigenvalue
#   below -0.05 is ~5% of a full unit variance "borrowed from nowhere" — a genuine
#   spec inconsistency, orders of magnitude above the ~1e-10 float-noise scale and
#   above the mild slop of partial/pairwise estimation.
# - max-off-diagonal-adjustment 0.10: the interpretable materiality gate. If
#   reaching validity requires the nearest-correlation projection to silently move
#   any single STATED pairwise correlation by more than 0.10, the projected matrix
#   embodies a materially different dependence assumption than the caller declared.
# Reject iff EITHER fires (fail-closed OR). Both fire on the [-1,2,2] contradiction
# (lambda_min=-1.0, adj=0.5); a near-PSD noise case (frustrated 0.51: lambda_min=-0.02,
# adj=0.01) clears both and still projects. The verdict is PERMUTATION-INVARIANT
# (eigenvalues + max|off-diagonal adjustment| are invariant under symmetric
# relabeling), so it does not depend on factor draw order.
CORRELATION_ADMISSION_METHOD_VERSION = "corr_admission_v1"
CORRELATION_REJECT_MIN_EIGENVALUE = -0.05  # reject if lambda_min < this
CORRELATION_REJECT_MAX_ADJUSTMENT = 0.10  # reject if max |off-diag Higham adjustment| > this

# Eigenvalue floor used ONLY when a PSD-but-singular matrix (e.g. a perfect
# rho=1 pair, or a Higham projection whose smallest eigenvalue is clamped to ~0)
# trips Cholesky. The spectrum is lifted so its minimum reaches this floor, then
# the matrix is renormalised back to unit diagonal (correlation form) so the
# copula's standard-normal marginal assumption still holds. The floor is far
# below any meaningful correlation signal; bounded + deterministic.
_CHOLESKY_EIGEN_FLOOR = 1e-8


@dataclass(frozen=True)
class EffectivePair:
    """The EFFECTIVE (post-projection) correlation for one factor pair (F4).

    ``requested_rho`` is what the caller stated; ``effective_rho`` is the
    off-diagonal that the Higham-projected matrix ACTUALLY used to drive the copula
    draw; ``adjustment`` is ``effective_rho - requested_rho`` (how far, and which
    way, the projection silently moved that correlation). ``stated`` is True for a
    pair the caller supplied and False for an UNSTATED pair that defaulted to zero
    (assumed-independent) and was MOVED by the projection — those moved zero-fill
    pairs are disclosed too (F-2) so the effective matrix is fully reconstructable,
    not just the aggregate Frobenius distance.
    """

    factor_a: str
    factor_b: str
    requested_rho: float
    effective_rho: float
    adjustment: float
    stated: bool


@dataclass(frozen=True)
class AdmissibilityVerdict:
    """Hard-invalid admissibility verdict for an assembled correlation matrix (F4).

    ``admissible`` is False when the matrix is indefinite beyond the near-PSD repair
    band (see ``CORRELATION_REJECT_*``); the request validator turns that into a
    typed 422 BEFORE any projection. ``reasons`` names exactly which bound(s) fired
    (``min_eigenvalue`` and/or ``max_adjustment``). ``projected`` is the
    nearest-correlation matrix (equal to the input when it was already PSD) so the
    caller can name the offending pairs without recomputing the projection.
    """

    admissible: bool
    min_eigenvalue: float
    max_abs_off_diagonal_adjustment: float
    frobenius_distance: float
    reasons: Tuple[str, ...]
    projected: np.ndarray


def evaluate_correlation_admissibility(matrix: np.ndarray) -> AdmissibilityVerdict:
    """Decide whether an assembled correlation matrix is HARD-INVALID (F4, D-23.13).

    Already-PSD (up to float noise) → trivially admissible, no projection. Otherwise
    the Higham nearest-correlation projection is computed and the matrix is rejected
    iff its smallest eigenvalue is below ``CORRELATION_REJECT_MIN_EIGENVALUE`` OR the
    largest single off-diagonal adjustment exceeds ``CORRELATION_REJECT_MAX_ADJUSTMENT``
    (fail-closed OR). Reasons are recorded so the caller can name the criteria that
    fired. Permutation-invariant: the verdict does not depend on row/column order.
    """
    lam_min = min_eigenvalue(matrix)
    if lam_min >= -PSD_EIGEN_TOL:
        # Already PSD (or float-noise negative): admissible, no projection needed.
        return AdmissibilityVerdict(
            admissible=True,
            min_eigenvalue=lam_min,
            max_abs_off_diagonal_adjustment=0.0,
            frobenius_distance=0.0,
            reasons=(),
            projected=np.array(matrix, dtype=float),
        )
    projected, _iterations = nearest_correlation_higham(matrix)
    diff = projected - matrix
    frobenius_distance = float(np.linalg.norm(diff, ord="fro"))
    off = diff.copy()
    np.fill_diagonal(off, 0.0)
    max_off = float(np.max(np.abs(off))) if off.size else 0.0
    reasons = []
    if lam_min < CORRELATION_REJECT_MIN_EIGENVALUE:
        reasons.append("min_eigenvalue")
    if max_off > CORRELATION_REJECT_MAX_ADJUSTMENT:
        reasons.append("max_adjustment")
    return AdmissibilityVerdict(
        admissible=not reasons,
        min_eigenvalue=lam_min,
        max_abs_off_diagonal_adjustment=max_off,
        frobenius_distance=frobenius_distance,
        reasons=tuple(reasons),
        projected=projected,
    )


@dataclass(frozen=True)
class ProjectionInfo:
    """On-wire disclosure payload for a Higham nearest-correlation projection.

    ``effective_pairs`` (F4) discloses the EFFECTIVE post-projection off-diagonal for
    each supplied pair — the correlations that actually drove the copula — so the
    caller sees more than the aggregate Frobenius distance.
    """

    applied: bool
    method: str
    frobenius_distance: float
    max_abs_off_diagonal_adjustment: float
    iterations: int
    effective_pairs: Tuple[EffectivePair, ...]


@dataclass(frozen=True)
class CorrelationPlan:
    """Everything ``FactorSampler`` needs for a joint copula draw.

    ``factor_order`` is the canonical order the correlated factors are drawn in
    (first appearance in ``parameter_uncertainties``). ``cholesky`` is the
    lower-triangular ``L`` with ``L @ L.T`` == the (possibly projected)
    correlation matrix. ``projection`` is ``None`` when the assembled matrix was
    already PSD, else the disclosure payload for the Higham projection.
    """

    factor_order: List[str]
    cholesky: np.ndarray
    projection: Optional[ProjectionInfo]


def assemble_correlation_matrix(
    factor_order: List[str], pairs: List[Tuple[str, str, float]]
) -> np.ndarray:
    """Assemble the symmetric correlation matrix over ``factor_order``.

    Diagonal is 1. Each supplied pair sets the symmetric off-diagonal entry.
    Unsupplied off-diagonal entries stay 0 (independent within the set). A
    self-pair (a == b) is a validated no-op — the diagonal is already 1.
    """
    n = len(factor_order)
    idx = {f: i for i, f in enumerate(factor_order)}
    matrix = np.eye(n, dtype=float)
    for a, b, rho in pairs:
        ia = idx[a]
        ib = idx[b]
        if ia == ib:
            continue
        matrix[ia, ib] = rho
        matrix[ib, ia] = rho
    return matrix


def min_eigenvalue(matrix: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix (via ``eigvalsh``)."""
    return float(np.linalg.eigvalsh(matrix)[0])


def is_positive_semidefinite(matrix: np.ndarray, tol: float = PSD_EIGEN_TOL) -> bool:
    """True when the smallest eigenvalue is >= -tol (PSD up to float noise)."""
    return min_eigenvalue(matrix) >= -tol


def nearest_correlation_higham(
    matrix: np.ndarray,
    max_iter: int = HIGHAM_MAX_ITER,
    tol: float = HIGHAM_TOL,
) -> Tuple[np.ndarray, int]:
    """Higham (2002) nearest correlation matrix via alternating projections.

    Alternates projection onto the PSD cone (clamp negative eigenvalues to 0)
    and onto the unit-diagonal set, with Dykstra's correction, until the update
    is below ``tol`` or ``max_iter`` is reached.

    Returns ``(projected, iterations)`` where ``projected`` is symmetric, PSD
    (up to float noise), with unit diagonal.
    """
    y = np.array(matrix, dtype=float)
    ds = np.zeros_like(y)
    prev_y = y.copy()
    iterations = 0
    for k in range(1, max_iter + 1):
        iterations = k
        r = y - ds  # Dykstra correction
        # Project onto the PSD cone: clamp negative eigenvalues to zero.
        eigvals, eigvecs = np.linalg.eigh(r)
        clamped = np.clip(eigvals, 0.0, None)
        x = (eigvecs * clamped) @ eigvecs.T
        x = (x + x.T) / 2.0  # re-symmetrise against round-off
        ds = x - r
        # Project onto the unit-diagonal set.
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        if float(np.max(np.abs(y - prev_y))) < tol:
            break
        prev_y = y.copy()
    return y, iterations


def _safe_cholesky(matrix: np.ndarray) -> np.ndarray:
    """Cholesky ``L`` (lower).

    Raw Cholesky is used whenever it succeeds — this keeps ``chol(I) == I`` and
    every PD case bit-exact. Only when the matrix is PSD-but-singular (a perfect
    rho=1 pair, or a Higham projection whose smallest eigenvalue sits at ~0) does
    Cholesky fail; then the spectrum is lifted so its minimum reaches
    ``_CHOLESKY_EIGEN_FLOOR`` and the matrix is renormalised back to unit
    diagonal. Lifting adds a constant to every eigenvalue (strictly PD); the
    unit-diagonal renormalisation is a positive-diagonal congruence, so PSD-ness
    is preserved and each drawn factor remains standard-normal for the copula."""
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        n = matrix.shape[0]
        min_eig = float(np.linalg.eigvalsh(matrix)[0])
        ridge = _CHOLESKY_EIGEN_FLOOR - min_eig
        lifted = matrix + ridge * np.eye(n)
        diag = np.sqrt(np.diag(lifted))
        lifted = lifted / np.outer(diag, diag)
        return np.linalg.cholesky(lifted)


def build_correlation_plan(
    factor_order: List[str], pairs: List[Tuple[str, str, float]]
) -> CorrelationPlan:
    """Assemble → PSD-check → (Higham project + disclose) → Cholesky."""
    matrix = assemble_correlation_matrix(factor_order, pairs)
    projection: Optional[ProjectionInfo] = None
    used = matrix
    if not is_positive_semidefinite(matrix):
        projected, iterations = nearest_correlation_higham(matrix)
        diff = projected - matrix
        frobenius_distance = float(np.linalg.norm(diff, ord="fro"))
        off = diff.copy()
        np.fill_diagonal(off, 0.0)
        max_off = float(np.max(np.abs(off))) if off.size else 0.0
        # F4: per-pair EFFECTIVE off-diagonals — the correlations the copula actually
        # used after projection, and how far each moved from what the caller stated.
        idx = {f: i for i, f in enumerate(factor_order)}
        stated_keys = {frozenset((a, b)) for a, b, _ in pairs if a != b}
        effective_list = [
            EffectivePair(
                factor_a=a,
                factor_b=b,
                requested_rho=float(rho),
                effective_rho=float(projected[idx[a], idx[b]]),
                adjustment=float(projected[idx[a], idx[b]]) - float(rho),
                stated=True,
            )
            for a, b, rho in pairs
            if a != b
        ]
        # F-2: an UNSTATED pair defaults to correlation 0 (assumed-independent). When
        # the projection MOVES such a zero-filled entry off 0, disclose it too — else
        # the effective matrix the copula actually used is not reconstructable. The
        # threshold filters Higham residual float-noise (converges to ~1e-10).
        n = len(factor_order)
        for i in range(n):
            for j in range(i + 1, n):
                if frozenset((factor_order[i], factor_order[j])) in stated_keys:
                    continue
                eff = float(projected[i, j])
                if abs(eff) > EFFECTIVE_DISCLOSURE_MOVE_TOL:
                    effective_list.append(
                        EffectivePair(
                            factor_a=factor_order[i],
                            factor_b=factor_order[j],
                            requested_rho=0.0,
                            effective_rho=eff,
                            adjustment=eff,
                            stated=False,
                        )
                    )
        effective_pairs = tuple(effective_list)
        projection = ProjectionInfo(
            applied=True,
            method=HIGHAM_METHOD,
            frobenius_distance=frobenius_distance,
            max_abs_off_diagonal_adjustment=max_off,
            iterations=iterations,
            effective_pairs=effective_pairs,
        )
        used = projected
    cholesky = _safe_cholesky(used)
    return CorrelationPlan(
        factor_order=list(factor_order), cholesky=cholesky, projection=projection
    )
