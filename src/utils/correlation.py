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
from typing import Dict, List, Optional, Tuple

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

# Eigenvalue floor used ONLY when a PSD-but-singular matrix (e.g. a perfect
# rho=1 pair, or a Higham projection whose smallest eigenvalue is clamped to ~0)
# trips Cholesky. The spectrum is lifted so its minimum reaches this floor, then
# the matrix is renormalised back to unit diagonal (correlation form) so the
# copula's standard-normal marginal assumption still holds. The floor is far
# below any meaningful correlation signal; bounded + deterministic.
_CHOLESKY_EIGEN_FLOOR = 1e-8


@dataclass(frozen=True)
class CorrelationPlan:
    """Everything ``FactorSampler`` needs for a joint copula draw.

    ``factor_order`` is the canonical order the correlated factors are drawn in
    (first appearance in ``parameter_uncertainties``). ``cholesky`` is the
    lower-triangular ``L`` with ``L @ L.T`` == the (possibly projected)
    correlation matrix. ``projection`` is ``None`` when the assembled matrix was
    already PSD, else the on-wire disclosure payload for the Higham projection.
    """

    factor_order: List[str]
    cholesky: np.ndarray
    projection: Optional[Dict[str, object]]


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
    projection: Optional[Dict[str, object]] = None
    used = matrix
    if not is_positive_semidefinite(matrix):
        projected, iterations = nearest_correlation_higham(matrix)
        diff = projected - matrix
        frobenius_distance = float(np.linalg.norm(diff, ord="fro"))
        off = diff.copy()
        np.fill_diagonal(off, 0.0)
        max_off = float(np.max(np.abs(off))) if off.size else 0.0
        projection = {
            "applied": True,
            "method": HIGHAM_METHOD,
            "frobenius_distance": frobenius_distance,
            "max_abs_off_diagonal_adjustment": max_off,
            "iterations": iterations,
        }
        used = projected
    cholesky = _safe_cholesky(used)
    return CorrelationPlan(
        factor_order=list(factor_order), cholesky=cholesky, projection=projection
    )
