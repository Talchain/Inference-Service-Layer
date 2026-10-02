"""
SCIENCE ROBUSTNESS step 1 (EXPERIMENT; #85 lease 5948121821): BEFORE vs AFTER on D1, the metric named before the fix.

Fixture: tests/fixtures/robustness/d1-isl-request.json = R3's served D1 graph translated by PLoT /v2/run (PLoT staging
4526e432); see tests/fixtures/horizon provenance on ISL #219. Path links: the plan's path to the goal.

  BEFORE  today's outputs over 10 master seeds: base flip_mean (one world) + its 10-world band median.
  AFTER   decision_flip_threshold over 10 master seeds (10k draws each, CRN within a seed).
  REF     the same search with leader = argmax of P(best) averaged over 10 further seeds (100k draws), tol 0.001.
  METRIC  per link: (a) spread max-min <= 0.01; (b) exists/none agreement 10/10;
          (c) misses vs REF: exists mismatch, or |x - x_ref| > 0.01, target 0/30; (d) runtime per link.

usage: ISL_AUTH_DISABLED=true python scripts/measure_decision_flip.py <out.json>
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
from src.models.robustness_v2 import RobustnessRequestV2  # noqa: E402
from src.services.decision_flip import decision_flip_threshold  # noqa: E402
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2  # noqa: E402

FIXTURE = "tests/fixtures/robustness/d1-isl-request.json"
PATH = [
    ("sprint_capacity_for_ai_reporting", "ai_reporting_module_availability"),
    ("ai_reporting_module_availability", "enterprise_prospect_signing_likelihood"),
    ("enterprise_prospect_signing_likelihood", "quarterly_revenue"),
]
SEEDS = range(1, 11)
REF_SEEDS = range(1001, 1011)
SPREAD_MAX, MISS_TOL = 0.01, 0.01
STRIP = {"include_e_values": False, "include_voi": False, "include_factor_flips": False, "analysis_types": ["comparison"]}

BASE = json.load(open(FIXTURE))


def edge_of(q, link):
    hit = [e for e in q["graph"]["edges"] if (e.get("from") or e.get("from_"), e["to"]) == link]
    assert len(hit) == 1, link
    return hit[0]


def run(link, x, seed, strip=True):
    q = copy.deepcopy(BASE)
    q["seed"] = str(seed)
    if strip:
        q.update(STRIP)
    if link is not None:
        edge_of(q, link)["strength"]["mean"] = x
    return RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**q))


def ref_leader(link, x):
    tot = {}
    for s in REF_SEEDS:
        for o in run(link, x, s).results:
            tot[o.option_id] = tot.get(o.option_id, 0.0) + o.win_probability
    return max(tot, key=lambda k: tot[k])


def spread(xs):
    xs = [x for x in xs if x is not None]
    return round(max(xs) - min(xs), 6) if xs else None


def main(out_path):
    out = {"before": {}, "after": {}, "ref": {}, "metric": {}}
    # BEFORE: today's served outputs, unstripped (e-values + bands on), per master seed.
    for seed in SEEDS:
        r = run(None, None, seed, strip=False)
        ev = {(e["from_id"], e["to_id"]): e for e in (r.edge_e_values or [])}
        for link in PATH:
            e = ev.get(link) or {}
            st = e.get("stability") or {}
            out["before"].setdefault("->".join(link), []).append(
                {"seed": seed, "flip_mean": e.get("flip_mean"), "dir": e.get("flip_direction"),
                 "n_flipped": st.get("n_seeds_flipped"), "band_median": st.get("band_median")})
    for link in PATH:
        key = "->".join(link)
        cur = edge_of(BASE, link)["strength"]["mean"]
        rows = []
        for seed in SEEDS:
            t = time.perf_counter()
            res = decision_flip_threshold(lambda x: run(link, x, seed).recommended_option_id, cur, 0.0)
            res["ms"] = round((time.perf_counter() - t) * 1000)
            res["seed"] = seed
            rows.append(res)
            print(key[:40], "seed", seed, res["exists"], res["threshold"], res["evaluations"], res["ms"], "ms", flush=True)
        out["after"][key] = rows
        t = time.perf_counter()
        ref = decision_flip_threshold(lambda x: ref_leader(link, x), cur, 0.0, tol=0.001)
        ref["ms"] = round((time.perf_counter() - t) * 1000)
        out["ref"][key] = ref
        print(key[:40], "REF", ref["exists"], ref["threshold"], ref["ms"], "ms", flush=True)

        b = out["before"][key]
        def misses(vals, exists_flags):
            n = 0
            for v, ex in zip(vals, exists_flags):
                if ex != ref["exists"]:
                    n += 1
                elif ex and abs(v - ref["threshold"]) > MISS_TOL:
                    n += 1
            return n
        a_x = [r["threshold"] for r in rows]
        a_ex = [r["exists"] for r in rows]
        # flip_mean "exists" = a decrease-direction flip at or above 0 (the "weaker" claim); below 0 = a reversal.
        ZERO = -1e-5  # a flip reported at −1e-06 is a flip at 0 (the link gone), not a reversal
        b_fm = [max(x["flip_mean"], 0.0) if x["flip_mean"] is not None else None for x in b]
        b_fm_ex = [x["flip_mean"] is not None and x["dir"] == "decrease" and x["flip_mean"] >= ZERO for x in b]
        b_md = [max(x["band_median"], 0.0) if x["band_median"] is not None else None for x in b]
        b_md_ex = [x["band_median"] is not None and x["band_median"] >= ZERO for x in b]
        out["metric"][key] = {
            "ref": ref["threshold"] if ref["exists"] else None,
            "after": {"spread": spread(a_x), "exists_agree": max(a_ex.count(True), a_ex.count(False)),
                      "misses": misses(a_x, a_ex), "ms_mean": round(sum(r["ms"] for r in rows) / len(rows))},
            "before_flip_mean": {"spread": spread(b_fm), "exists_agree": max(b_fm_ex.count(True), b_fm_ex.count(False)),
                                 "misses": misses(b_fm, b_fm_ex), "value": b_fm[0]},
            "before_band_median": {"spread": spread(b_md), "exists_agree": max(b_md_ex.count(True), b_md_ex.count(False)),
                                   "misses": misses(b_md, b_md_ex)},
        }
        print(key[:40], json.dumps(out["metric"][key]), flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
