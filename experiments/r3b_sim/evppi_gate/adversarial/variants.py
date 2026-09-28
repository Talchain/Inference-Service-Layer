import numpy as np, json, sys
from src.utils import evppi as E
from src.utils.evppi import factor_evppi_estimate
Z = 1.645

def gate(th, Y, seed, deg=4, comparator="train", clip=True, repeats=1):
    n = th.size; means = []; ses = []
    for r in range(repeats):
        order = np.random.default_rng((seed, 1 + r)).permutation(n); folds = np.array_split(order, 2)
        g = np.empty(n)
        full_best = int(np.argmax(Y.mean(0)))
        for k, te in enumerate(folds):
            tr = np.concatenate([f for j, f in enumerate(folds) if j != k])
            d = E._effective_degree(th[tr], deg)
            if clip:
                pred = E._fit_predict(th[tr], Y[tr], th[te], d)
            else:
                from numpy.polynomial import polynomial as P, polyutils as PU
                dom = [th[tr].min(), th[tr].max()]
                V = P.polyvander(PU.mapdomain(th[tr], dom, [-1, 1]), d); sc = np.sqrt((V**2).sum(0)); sc[sc == 0] = 1
                c, *_ = np.linalg.lstsq(V / sc, Y[tr], rcond=None)
                pred = (P.polyvander(PU.mapdomain(th[te], dom, [-1, 1]), d) / sc) @ c
            rule = np.argmax(pred, axis=1)
            best = {"train": int(np.argmax(Y[tr].mean(0))), "full": full_best, "heldout": int(np.argmax(Y[te].mean(0)))}[comparator]
            g[te] = Y[te, rule] - Y[te, best]
        means.append(g.mean()); ses.append(g.std(ddof=1) / np.sqrt(n))
    m, s = float(np.mean(means)), float(np.mean(ses))
    return m > Z * s if s > 0 else m > 0

CASES = {
  "independent_null":      (lambda th: [10 + 0*th, 9.0 + 0*th], 0),
  "dominated_dependent":   (lambda th: [10 + 0*th, 9.0 + 0.8*np.tanh(th)], 0),
  "close_null (tie-ish)":  (lambda th: [10 + 0*th, 9.98 + 0*th], 0),
  "close_dominated":       (lambda th: [10 + 0*th, 9.90 + 0.09*np.tanh(th)], 0),
  "common_shift_null":     (lambda th: [10 + 0.5*th, 9.9 + 0.5*th], 0),
  "three_opt_null":        (lambda th: [10 + 0*th, 9.95 + 0*th, 9.0 + 0.8*np.tanh(th)], 0),
  "true_positive_0.083":   (lambda th: [10 + 0*th, 9.0 + 1.0*th], 1),
  "moderate_0.042":        (lambda th: [10 + 0*th, 9.5 + 0.5*th], 1),
  "weak_tail_0.004":       (lambda th: [10 + 0*th, 9.0 + 0.5*th], 1),
  "nonlinear_U_0.05":      (lambda th: [10 + 0*th, 9.6 + 0.4*th**2], 1),
}
VARIANTS = [("current (train comparator, clip, deg4, 1 split)", {}),
            ("full-sample comparator", {"comparator": "full"}),
            ("held-out comparator", {"comparator": "heldout"}),
            ("no clipping", {"clip": False}),
            ("degree 2", {"deg": 2}),
            ("5 repeated splits", {"repeats": 5})]
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out = {}
for name, (f, pos) in CASES.items():
    for n in (500, 2000):
        rows = {v: 0 for v, _ in VARIANTS}; rows["gate_alone(current)"] = 0; rows["floor_only(current rule)"] = 0
        for s in range(SEEDS):
            r = np.random.default_rng(s); th = r.normal(0, 1, n)
            mus = f(th); Y = np.column_stack([mu + r.normal(0, 1, n) for mu in mus])
            est = factor_evppi_estimate(th, {str(i): Y[:, i] for i in range(Y.shape[1])}, seed=1000 + s)
            floor_ok = round(max(0, est.evppi_raw), 6) > round(est.noise_floor, 6)
            rows["floor_only(current rule)"] += floor_ok
            g0 = gate(th, Y, 1000 + s)
            rows["gate_alone(current)"] += g0
            for v, kw in VARIANTS:
                rows[v] += floor_ok and (g0 if not kw else gate(th, Y, 1000 + s, **kw))
        out[f"{name} n={n}"] = rows
        print(f"{name:22s} n={n:5d} " + " | ".join(f"{k.split(' (')[0][:18]}={v}" for k, v in rows.items()), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
