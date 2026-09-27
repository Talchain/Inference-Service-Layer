"""Deadline metrics, sensitivity and EVPPI.

EVPPI uses ISL's served estimator (``src/utils/evppi.py``), loaded READ-ONLY by file path so no
``src`` package initialisation runs. Its sha256 is recorded with every result.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .corpus import ROOT, sha256_file

ISL_EVPPI_PATH = ROOT.parent.parent / "src" / "utils" / "evppi.py"
_EVPPI_MODULE: ModuleType | None = None


def isl_evppi() -> ModuleType:
    global _EVPPI_MODULE
    if _EVPPI_MODULE is None:
        spec = importlib.util.spec_from_file_location("r3b_isl_evppi", ISL_EVPPI_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules["r3b_isl_evppi"] = module  # dataclasses resolve the module by name
        spec.loader.exec_module(module)
        _EVPPI_MODULE = module
    return _EVPPI_MODULE


def isl_evppi_sha256() -> str:
    return sha256_file(Path(ISL_EVPPI_PATH))


def r6(x: float) -> float:
    return float(round(float(x), 6))


def attained(y: np.ndarray, threshold: float, operator: str, first_passage: bool) -> np.ndarray:
    """Per-draw indicator. First passage = met at some month <= H; otherwise at H only."""
    series = y if first_passage else y[:, -1:]
    hit = series >= threshold if operator == ">=" else series <= threshold
    return np.asarray(hit.any(axis=1), dtype=float)


def goal_summary(y: np.ndarray, threshold: float, operator: str, has_path: bool) -> dict[str, Any]:
    n = y.shape[0]
    at_h = attained(y, threshold, operator, first_passage=False)
    out: dict[str, Any] = {
        "n_draws": n,
        "p_at_H": r6(at_h.mean()),
        "p_at_H_mc_se": r6(math.sqrt(at_h.mean() * (1 - at_h.mean()) / n)) if n > 1 else 0.0,
        "outcome_at_H_mean": r6(y[:, -1].mean()),
        "outcome_at_H_quantiles": {
            q: r6(np.quantile(y[:, -1], float(q) / 100)) for q in ("5", "25", "50", "75", "95")
        },
    }
    if has_path:
        by_h = attained(y, threshold, operator, first_passage=True)
        out["p_by_H"] = r6(by_h.mean())
        out["p_by_H_mc_se"] = r6(math.sqrt(by_h.mean() * (1 - by_h.mean()) / n)) if n > 1 else 0.0
        out["month0_outcome_mean"] = r6(y[:, 0].mean())
        out["p_at_month0"] = r6(attained(y[:, :1], threshold, operator, False).mean())
        out["path_mean"] = [r6(v) for v in y.mean(axis=0)]
        out["at_vs_by_disagree"] = bool(out["p_by_H"] != out["p_at_H"])
    else:
        out["p_by_H"] = None
    return out


def decision_evpi(outcomes: dict[str, np.ndarray]) -> float:
    matrix = np.vstack([outcomes[o] for o in sorted(outcomes)])
    return float(matrix.max(axis=0).mean() - matrix.mean(axis=1).max())


def evppi_status(theta: np.ndarray, outcomes: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    """ISL-identical: estimator, K=16 permutation-max floor, clamp to [0, EVPI], 6-dp rule."""
    est = isl_evppi().factor_evppi_estimate(theta, outcomes, seed=seed)
    evpi = max(0.0, decision_evpi(outcomes))
    val = min(max(0.0, est.evppi_raw), evpi)
    emitted = round(val, 6)
    resolved = (not est.degenerate) and emitted > round(est.noise_floor, 6)
    return {
        "evppi": r6(emitted),
        "noise_floor": r6(est.noise_floor),
        "decision_evpi": r6(evpi),
        "status": "resolved" if resolved else "below_resolution",
        "degenerate": bool(est.degenerate),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return r6(np.corrcoef(rx, ry)[0, 1])


def validate_evppi_estimator(seed: int = 20260927, n: int = 20000) -> dict[str, Any]:
    """Point 8: the estimator must reproduce an analytic EVPPI and a legitimate zero."""
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    cases = []
    ok = True
    for mu, sigma in ((0.0, 1.0), (0.5, 1.0)):
        theta = rng.normal(mu, sigma, n)
        truth = sigma * norm.pdf(mu / sigma) + mu * norm.cdf(mu / sigma) - max(mu, 0.0)
        res = evppi_status(theta, {"A": theta, "B": np.zeros(n)}, seed)
        passed = abs(res["evppi"] - truth) <= 0.03 * truth + 0.005 and res["status"] == "resolved"
        ok = ok and passed
        cases.append(
            {
                "case": f"U_A=theta~N({mu},{sigma}^2), U_B=0",
                "analytic": r6(truth),
                **res,
                "pass": passed,
            }
        )
    theta = rng.normal(0.0, 1.0, n)
    noise_a = rng.normal(0.0, 1.0, n)
    noise_b = rng.normal(0.0, 1.0, n)
    res0 = evppi_status(theta, {"A": theta + noise_a + 0.1, "B": theta + noise_b}, seed)
    passed0 = res0["status"] == "below_resolution"
    ok = ok and passed0
    cases.append(
        {
            "case": "zero control: theta shifts both options equally (+ independent noise)",
            "analytic": 0.0,
            **res0,
            "pass": passed0,
        }
    )
    return {
        "estimator": "ISL src/utils/evppi.py factor_evppi_estimate (Strong-Oakley regression)",
        "sha256": isl_evppi_sha256(),
        "cases": cases,
        "all_pass": ok,
        "metric_name": "EVPPI" if ok else "ISL-compatible information-value proxy",
    }
