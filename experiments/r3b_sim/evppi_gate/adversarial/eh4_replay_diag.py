import json, sys, numpy as np
import src.services.robustness_analyzer_v2 as A
import src.utils.evppi as E
from src.models.robustness_v2 import RobustnessRequestV2
S = sys.argv[1]
payload = json.load(open(f"{S}/eh4-isl-request.json"))
req = RobustnessRequestV2.model_validate(payload)
captured = {}
orig = A.factor_evppi_estimate
def spy(theta, oo, *, seed, **kw):
    est = orig(theta, oo, seed=seed, **kw)
    captured[len(captured)] = (np.asarray(theta, float), {k: np.asarray(v, float) for k, v in oo.items()}, seed, est)
    return est
A.factor_evppi_estimate = spy
r = A.RobustnessAnalyzerV2().analyze(req)
rows = {e["factor_id"]: e for e in r.factor_evppi}
for i, (th, oo, seed, est) in captured.items():
    ids = sorted(oo); Y = np.vstack([oo[o] for o in ids]).T; n = th.size
    order = np.random.default_rng((seed, 1)).permutation(n); folds = np.array_split(order, 2)
    gain = np.empty(n)
    for k, te in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != k])
        deg = E._effective_degree(th[tr], 4)
        best = int(np.argmax(Y.mean(0)))
        rule = np.argmax(E._fit_predict(th[tr], Y[tr], th[te], deg), axis=1)
        gain[te] = Y[te, rule] - Y[te, best]
    se = gain.std(ddof=1) / np.sqrt(n)
    print(f"row {i}: evppi_raw {est.evppi_raw:.6f} floor {est.noise_floor:.6f} gate {est.decision_gain_passes} "
          f"| held-out gain {gain.mean():.6f} SE {se:.6f} z {gain.mean()/se if se>0 else float('nan'):.2f} "
          f"| rule switches away from best-fixed on {np.mean(gain!=0):.1%} of draws")
print({k: (v["evppi"], v["noise_floor"], v["status"], v["status_reason"]) for k, v in rows.items()})
