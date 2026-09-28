"""Does ISL's per-factor EVPPI status separate decision-relevant factors from irrelevant ones?

ISL marks a factor ``resolved`` when its regression EVPPI exceeds a permutation-null floor
(``src/utils/evppi.py``; status at ``robustness_analyzer_v2.py`` "below_resolution =
evppi_emitted <= noise_floor_emitted"). Shuffling theta tests theta-outcome ASSOCIATION. It
does not test whether theta can change the DECISION. A factor that moves only an option that
never wins has true EVPPI exactly 0, yet its fitted curves can cross the leader's through
fit noise, while the shuffled fits stay flat and apart (floor ~0).

This study measures the ``resolved`` rate on synthetic two-option cases with known true
EVPPI, using ISL's OWN estimator (loaded by file path, unmodified). It also measures a
candidate status gate: a 2-fold cross-fitted test that the decision rule LEARNED from theta
beats the best fixed option on held-out draws. Under "theta cannot change the decision" the
expected gain of any learned rule is <= 0, whatever the association.

Usage, from the ISL repo root::

    poetry run python experiments/r3b_sim/evppi_gate/study.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.polynomial import polynomial as npp
from numpy.polynomial import polyutils as nppu

HERE = Path(__file__).resolve().parent
ISL_ROOT = HERE.parents[2]
SEEDS = 200
SIZES = (500, 2000)
Z_ONE_SIDED = 1.645


def load_isl_evppi() -> Any:
    path = ISL_ROOT / "src" / "utils" / "evppi.py"
    spec = importlib.util.spec_from_file_location("isl_evppi_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def isl_status(est: Any) -> str:
    """ISL's wire status rule: evppi and floor compared on their emitted 6-dp values."""
    evppi = round(max(0.0, est.evppi_raw), 6)
    return "below_resolution" if evppi <= round(est.noise_floor, 6) else "resolved"


def _fit_predict(th_tr: np.ndarray, y_tr: np.ndarray, th_te: np.ndarray, deg: int) -> np.ndarray:
    dom = [float(th_tr.min()), float(th_tr.max())]
    v = npp.polyvander(nppu.mapdomain(th_tr, dom, [-1.0, 1.0]), deg)
    scale = np.sqrt((v**2).sum(axis=0))
    scale[scale == 0] = 1.0
    coef, *_ = np.linalg.lstsq(v / scale, y_tr, rcond=len(th_tr) * np.finfo(float).eps)
    vt = npp.polyvander(nppu.mapdomain(np.clip(th_te, *dom), dom, [-1.0, 1.0]), deg) / scale
    return np.asarray(vt @ coef)


def crossfit_gain_passes(
    theta: np.ndarray, outcomes: dict[str, np.ndarray], seed: int, deg: int = 4
) -> bool:
    """Held-out value of the learned decision rule minus the best fixed option > z * SE."""
    ids = sorted(outcomes)
    y = np.column_stack([outcomes[o] for o in ids])
    n = theta.size
    idx = np.random.default_rng(seed).permutation(n)
    folds = (idx[: n // 2], idx[n // 2 :])
    gain = np.empty(n)
    for k in (0, 1):
        tr, te = folds[1 - k], folds[k]
        best_fixed = int(np.argmax(y[tr].mean(axis=0)))
        rule = np.argmax(_fit_predict(theta[tr], y[tr], theta[te], deg), axis=1)
        gain[te] = y[te, rule] - y[te, best_fixed]
    se = float(gain.std(ddof=1) / np.sqrt(n))
    return bool(gain.mean() > Z_ONE_SIDED * se) if se > 0 else bool(gain.mean() > 0)


# name -> (E[U_b | theta] as a function of theta, true EVPPI against E[U_a] = 10, sd 1 noise)
CASES: dict[str, tuple[Callable[[np.ndarray], np.ndarray], float, str]] = {
    "dominated_dependent": (
        lambda th: 9.0 + 0.8 * np.tanh(th),
        0.0,
        "theta moves B only; B never beats A (sup E[B|theta] = 9.8 < 10)",
    ),
    "independent_null": (
        lambda th: 9.0 + 0.0 * th,
        0.0,
        "theta unrelated to either option",
    ),
    "true_positive": (
        lambda th: 9.0 + 1.0 * th,
        0.0833155,
        "B wins for theta > 1: EVPPI = E[(theta - 1)+]",
    ),
    "moderate_positive": (
        lambda th: 9.5 + 0.5 * th,
        0.0416577,
        "B wins for theta > 1: EVPPI = 0.5 E[(theta - 1)+]",
    ),
    "weak_tail_positive": (
        lambda th: 9.0 + 0.5 * th,
        0.0042454,
        "B wins only for theta > 2: EVPPI = 0.5 E[(theta - 2)+]",
    ),
}


def run() -> dict[str, Any]:
    isl = load_isl_evppi()
    out: dict[str, Any] = {
        "estimator": "ISL src/utils/evppi.py factor_evppi_estimate, unmodified",
        "seeds": SEEDS,
        "gate": f"2-fold cross-fitted learned-rule gain > {Z_ONE_SIDED} SE (one-sided)",
        "cases": {},
    }
    for name, (mean_b, true_evppi, note) in CASES.items():
        rows = {}
        for n in SIZES:
            current = gated = zero_floor = 0
            for s in range(SEEDS):
                rng = np.random.default_rng(s)
                theta = rng.normal(0.0, 1.0, n)
                outcomes = {
                    "a": 10.0 + rng.normal(0.0, 1.0, n),
                    "b": mean_b(theta) + rng.normal(0.0, 1.0, n),
                }
                est = isl.factor_evppi_estimate(theta, outcomes, seed=1000 + s)
                resolved = isl_status(est) == "resolved"
                current += resolved
                zero_floor += round(est.noise_floor, 6) == 0.0
                gated += resolved and crossfit_gain_passes(theta, outcomes, seed=5000 + s)
            rows[str(n)] = {
                "resolved_current": current,
                "resolved_with_gate": gated,
                "floor_zero": zero_floor,
            }
        out["cases"][name] = {"true_evppi": true_evppi, "note": note, "by_n": rows}
    return out


def main() -> None:
    res = run()
    (HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for name, c in res["cases"].items():
        for n, r in c["by_n"].items():
            print(
                f"{name:20s} true EVPPI {c['true_evppi']:.4f}  n={n:>5}  resolved "
                f"{r['resolved_current']:3d}/{SEEDS} -> {r['resolved_with_gate']:3d}/{SEEDS} "
                f"with gate  (floor 0 in {r['floor_zero']})"
            )


if __name__ == "__main__":
    main()
