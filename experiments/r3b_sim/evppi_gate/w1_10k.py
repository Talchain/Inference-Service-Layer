"""W1 at the served default draw count (n = 10,000): the production gate on #198 (c054ce4)."""
import json, math, sys, time
import numpy as np
from src.utils.evppi import factor_evppi_estimate

CASES = {
    "dominated_dependent (true 0)": (lambda th: [10 + 0 * th, 9.0 + 0.8 * np.tanh(th)], 0.0),
    "independent_null (true 0)": (lambda th: [10 + 0 * th, 9.0 + 0 * th], 0.0),
    "near_tie_null (true 0)": (lambda th: [10 + 0 * th, 9.98 + 0 * th], 0.0),
    "three_option_null (true 0)": (lambda th: [10 + 0 * th, 9.95 + 0 * th, 9.0 + 0.8 * np.tanh(th)], 0.0),
    "true_positive (0.0833)": (lambda th: [10 + 0 * th, 9.0 + 1.0 * th], 0.0833155),
    "moderate (0.0417)": (lambda th: [10 + 0 * th, 9.5 + 0.5 * th], 0.0416577),
    "weak_tail (0.0042)": (lambda th: [10 + 0 * th, 9.0 + 0.5 * th], 0.0042454),
    "nonlinear_U (true > 0)": (lambda th: [10 + 0 * th, 9.6 + 0.4 * th**2], None),
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0, c - h), 3), round(min(1, c + h), 3)


SEEDS, N = int(sys.argv[1]), int(sys.argv[2])
out = {"n_draws": N, "seeds": SEEDS, "cases": {}}
t0 = time.time()
for name, (f, true) in CASES.items():
    before = after = 0
    for s in range(SEEDS):
        r = np.random.default_rng(s)
        th = r.normal(0, 1, N)
        Y = [mu + r.normal(0, 1, N) for mu in f(th)]
        e = factor_evppi_estimate(th, {str(i): y for i, y in enumerate(Y)}, seed=1000 + s)
        res = round(max(0, e.evppi_raw), 6) > round(e.noise_floor, 6)
        before += res
        after += res and e.decision_gain_passes
    out["cases"][name] = {"true_evppi": true, "resolved_current": before, "resolved_gated": after,
                          "ci95_current": wilson(before, SEEDS), "ci95_gated": wilson(after, SEEDS)}
    print(f"{name:30s} n={N}: resolved {before:3d}/{SEEDS} {wilson(before, SEEDS)} -> {after:3d}/{SEEDS} {wilson(after, SEEDS)}", flush=True)
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("seconds", out["seconds"])
