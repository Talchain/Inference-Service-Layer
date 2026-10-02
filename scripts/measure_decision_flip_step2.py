"""
SCIENCE ROBUSTNESS step 2 (EXPERIMENT; #85 lease 5948579361 + DL conditions 2-4): the on-demand block, measured.

Per case, 10 on-demand requests (master seeds 1..10, K=4): each link quoted / absent / no_change under the licence
(median of 4, range <= 0.02 AND <= 15%), scored against an INDEPENDENT 100k-draw reference: step 1's grid+bisection
on ordinary-mode analyses with P(best) pooled over seeds 1001..1010 (tol 0.001). A miss = a quoted link more than 0.01
from the reference, or quoted / no_change where the reference disagrees on existence. Wall clock per request.

Fixtures (tests/fixtures/robustness/): R3's served graphs translated by PLoT /v2/run at 4526e432 (D1 = A-Q-D1-BUILD,
D3 = A-Q-D3-BUILD). D3-CLAMPED is a CONSTRUCTED negative control (no served model carries a clamp): D3 with
epsilon_std 0.05 on monthly_cloud_savings, so the link INTO it must be withheld and the link OUT of it stays clean.

usage: ISL_AUTH_DISABLED=true python scripts/measure_decision_flip_step2.py <out.json> [<step1-result.json>]
"""

import copy
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
from src.models.robustness_v2 import DecisionFlipRequestV2, RobustnessRequestV2  # noqa: E402
from src.services import decision_flip as df  # noqa: E402
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2  # noqa: E402

FIX = "tests/fixtures/robustness/"
D1_PATH = [("sprint_capacity_for_ai_reporting", "ai_reporting_module_availability"),
           ("ai_reporting_module_availability", "enterprise_prospect_signing_likelihood"),
           ("enterprise_prospect_signing_likelihood", "quarterly_revenue")]
D3_PATH = [("gcp_workload_share", "monthly_cloud_overspend_during_migration"),
           ("gcp_workload_share", "monthly_cloud_savings"),
           ("migration_preparation_effort", "monthly_cloud_overspend_during_migration"),
           ("monthly_cloud_overspend_during_migration", "monthly_spend"),
           ("monthly_cloud_savings", "monthly_spend")]
REF_SEEDS = range(1001, 1011)
MISS_TOL = 0.01


def load(name, eps=None):
    q = json.load(open(FIX + name))
    for n in q["graph"]["nodes"]:
        if eps and n["id"] in eps:
            n["epsilon_std"] = eps[n["id"]]
    return q


def ref_threshold(q, link):
    base = copy.deepcopy(q)
    base.update(df.STRIPPED)
    cur = [e for e in base["graph"]["edges"] if (e.get("from") or e.get("from_"), e["to"]) == link][0]["strength"]["mean"]

    def leader(x):
        tot = {}
        for s in REF_SEEDS:
            qq = copy.deepcopy(base)
            qq["seed"] = str(s)
            for e in qq["graph"]["edges"]:
                if (e.get("from") or e.get("from_"), e["to"]) == link:
                    e["strength"]["mean"] = x
            for o in RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**qq)).results:
                tot[o.option_id] = tot.get(o.option_id, 0.0) + o.win_probability
        return max(tot, key=lambda k: tot[k])

    r = df.decision_flip_threshold(leader, cur, 0.0, tol=0.001)
    return r["threshold"] if r["exists"] else None


def main(out_path, step1_path=None):
    cases = {"D1": (load("d1-isl-request.json"), D1_PATH), "D3": (load("d3-isl-request.json"), D3_PATH),
             "D3-CLAMPED": (load("d3-isl-request.json", {"monthly_cloud_savings": 0.05}), D3_PATH)}
    prior = json.load(open(step1_path)) if step1_path else {}
    out = {}
    for name, (q, path) in cases.items():
        refs = {}
        for link in path:
            key = "->".join(link)
            if name == "D1" and prior.get("ref", {}).get(key):
                refs[key] = prior["ref"][key]["threshold"] if prior["ref"][key]["exists"] else None
            else:
                t = time.perf_counter()
                refs[key] = ref_threshold(q, link)
                print(name, key[:50], "REF", refs[key], round((time.perf_counter() - t)), "s", flush=True)
        reqs = []
        for seed in range(1, 11):
            qq = copy.deepcopy(q)
            qq["seed"] = str(seed)
            t = time.perf_counter()
            block = df.compute_decision_flip_block(DecisionFlipRequestV2.model_validate(
                {"request": qq, "links": [{"from_id": a, "to_id": b} for a, b in path], "replicates": 4}))
            ms = round((time.perf_counter() - t) * 1000)
            row = {"seed": seed, "ms": ms, "leader": block.leader_option_id, "links": {}}
            for lk in block.links:
                key = f"{lk.from_id}->{lk.to_id}"
                ref = refs[key]
                miss = (lk.status == "quoted" and (ref is None or abs(lk.threshold - ref) > MISS_TOL)) or \
                       (lk.status == "no_change" and ref is not None)
                row["links"][key] = {"status": lk.status, "reason": lk.reason, "threshold": lk.threshold,
                                     "range": lk.replicate_range, "miss": miss}
            reqs.append(row)
            print(name, "seed", seed, ms, "ms", {k[:28]: (v["status"], v["reason"], v["threshold"] and round(v["threshold"], 4), v["miss"]) for k, v in row["links"].items()}, flush=True)
        summary = {}
        for link in path:
            key = "->".join(link)
            st = [r["links"][key] for r in reqs]
            summary[key] = {"ref": refs[key], "quoted": sum(s["status"] == "quoted" for s in st),
                            "absent": sum(s["status"] == "absent" for s in st), "no_change": sum(s["status"] == "no_change" for s in st),
                            "misses": sum(s["miss"] for s in st), "reasons": sorted({s["reason"] for s in st if s["reason"]})}
        out[name] = {"summary": summary, "ms_median": statistics.median(r["ms"] for r in reqs), "ms_max": max(r["ms"] for r in reqs), "requests": reqs}
        print(name, json.dumps({"summary": summary, "ms_median": out[name]["ms_median"], "ms_max": out[name]["ms_max"]}), flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
