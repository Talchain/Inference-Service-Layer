#!/usr/bin/env python3
"""SCI-STRUCTURAL-ROBUSTNESS 30 Sep 2026: extract comparable outputs from the A/B/C run matrix.

Reads runs/<model>-<optset>-s<seed>.{isl-calls,plot-response}.json and prints/writes tables.
Engine outputs are read from ISL's RAW v2 response (the same engine for A, B, C); product-surface status
(what PLoT/CEE would show) is read from PLoT's response. Goal values are converted to GBP with the goal's
own frame (the goal node's observed cap, 106,250), which is how PLoT denormalises them (control: A reproduces
PLoT's served GBP figures exactly).
"""
import json
import os
import sys

OUT = os.environ.get("SSR_OUT", "/private/tmp/ssr-20260930/out")
RUNS = f"{OUT}/runs"
GOAL_FRAME = 106250.0
PRICE_FRAME = 200.0
MODELS = ["A", "B", "C"]
OPTSETS = ["cur", "served3"]
SEEDS = ["1254899477", "1", "20260930"]
PHASE_WARNING_HINTS = ("UNAVAILABLE", "BUDGET", "DEADLINE", "SKIPPED", "DEGRADED", "TIMEOUT")


DIRECT = f"{OUT}/runs-direct"
PLOT_SEED = "1254899477"
VOLATILE = {"processing_time_ms", "timestamp", "request_id", "request_echo", "build", "seed_used", "seed_source"}


def load(tag):
    """Engine output: the in-process ISL run (budgets raised) for every seed. Product surface: PLoT's response,
    available for the seed PLoT ran (1254899477); for other seeds the PLoT fields are reported as not run."""
    d = json.load(open(f"{DIRECT}/{tag}.isl-response.json"))
    v2 = [{"url": "direct", "status": d["status"], "ms": d["ms"], "request": d["request"], "response": d["response"]}]
    plot_path = f"{RUNS}/{tag}.plot-response.json"
    plot = json.load(open(plot_path)) if tag.endswith(f"s{PLOT_SEED}") and os.path.exists(plot_path) else {"status": None, "body": {}}
    return v2, plot


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def control(tag):
    """Direct (budgets raised) vs the ISL response PLoT actually received, same request: equal on every science field?"""
    calls = json.load(open(f"{RUNS}/{tag}.isl-calls.json"))
    via_plot = [c for c in calls if "analyze/v2" in c["url"]][-1]["response"]
    direct = json.load(open(f"{DIRECT}/{tag}.isl-response.json"))["response"]
    a, b = strip(via_plot), strip(direct)
    diffs = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return diffs


def gbp(x):
    return None if x is None else x * GOAL_FRAME


def summarise(tag):
    v2, plot = load(tag)
    final = v2[-1]
    isl = final["response"]
    req = final["request"]
    row = {"tag": tag, "isl_calls": len(v2), "isl_status": final["status"], "isl_ms": final["ms"],
           "seed_sent": req.get("seed"), "n_samples": req.get("n_samples"),
           "seed_used": isl.get("seed_used"),
           "identity_forwarded": [n["id"] for n in req["graph"]["nodes"] if n.get("nonlinear_identity")],
           "identity_evaluations": isl.get("identity_evaluations"),
           "options": {}}
    for o in isl.get("options", []):
        cons = (o.get("constraint_analysis") or {}).get("constraints") or []
        churn = next((c for c in cons if c.get("node_id") == "monthly_churn"), {})
        row["options"][o["id"]] = {
            "mean_gbp": gbp(o["outcome"].get("mean")),
            "p10_gbp": gbp(o["outcome"].get("p10")),
            "p50_gbp": gbp(o["outcome"].get("p50")),
            "p90_gbp": gbp(o["outcome"].get("p90")),
            "p_goal": o.get("probability_of_goal"),
            "p_churn_ok": churn.get("prob_satisfied"),
            "p_joint": (o.get("constraint_analysis") or {}).get("joint_probability"),
            "win": o.get("win_probability"),
        }
    rob = isl.get("robustness") or {}
    row["robustness"] = {
        "level": rob.get("level"),
        "is_robust": rob.get("is_robust"),
        "recommendation_stability": rob.get("recommendation_stability"),
        "fragile_edges_v1": rob.get("fragile_edges_v1"),
        "flippable_edges": [
            {"edge": e["edge_id"], "current": e.get("current_mean"), "flip": e.get("flip_mean")}
            for e in (rob.get("edge_e_values") or []) if not e.get("is_unflippable")
        ],
    }
    ranking = sorted(isl.get("options", []), key=lambda o: -(o.get("win_probability") or 0))
    row["isl_top_option"] = ranking[0]["id"] if ranking else None
    row["factor_influence"] = {
        f["node_id"]: {"score": f.get("influence_score"), "rank": f.get("influence_rank"), "importance_rank": f.get("importance_rank")}
        for f in (isl.get("factor_sensitivity") or [])
    }
    row["factor_flips"] = [
        {k: f.get(k) for k in ("factor_id", "current_value", "flip_value", "flip_reason", "alternative_winner_id")}
        for f in (isl.get("factor_flip_values") or [])
    ]
    warns = [w.get("code") for w in (isl.get("inference_warnings") or [])]
    crits = [(c.get("code"), c.get("severity")) for c in (isl.get("critiques") or [])]
    row["isl_warnings"] = warns
    row["isl_critiques"] = crits
    row["phase_degraded"] = [w for w in warns + [c for c, _ in crits] if w and any(h in w for h in PHASE_WARNING_HINTS)]
    row["statuses"] = {k: isl.get(k) for k in ("analysis_status", "robustness_status", "factor_sensitivity_status")}
    # --- product surface (PLoT) ---
    body = plot["body"]
    pr = body.get("robustness") or {}
    oc = {(o.get("option_id") or o.get("id")): o for o in (body.get("option_comparison") or [])}
    row["plot"] = {
        "status": plot["status"],
        "recommended": pr.get("recommended_option_id"),
        "display_verdict": pr.get("display_verdict"),
        "level": pr.get("level"),
        "fragile_edges": [e.get("edge_id") for e in (pr.get("fragile_edges") or [])],
        "goal_figures_shown": {k: ("probability_of_goal" in v) for k, v in oc.items()},
        "outcome_mean_shown": {k: (v.get("outcome") or {}).get("mean") for k, v in oc.items()},
        "withheld_codes": sorted({w.get("code") for w in (body.get("inference_warnings") or []) if "WITHHELD" in str(w.get("code")) or "NOT_" in str(w.get("code"))}),
        "identities_not_forwarded": (body.get("_meta") or {}).get("identities_not_forwarded"),
        "driver_order": body.get("driver_order"),
    }
    return row


def main():
    rows = {}
    for seed in SEEDS:
        for m in MODELS:
            for o in OPTSETS:
                tag = f"{m}-{o}-s{seed}"
                if os.path.exists(f"{DIRECT}/{tag}.isl-response.json"):
                    rows[tag] = summarise(tag)
                    if seed == PLOT_SEED:
                        rows[tag]["control_direct_vs_plot_isl_diff_keys"] = control(tag)
    json.dump(rows, open(f"{OUT}/summary.json", "w"), indent=2)
    # compact print
    for tag, r in rows.items():
        print(f"== {tag}  isl_calls={r['isl_calls']} status={r['isl_status']} ms={r['isl_ms']} seed={r['seed_used']} n={r['n_samples']} "
              f"identity_fwd={r['identity_forwarded']} degraded={r['phase_degraded']} top={r['isl_top_option']} "
              f"rob={r['robustness']['level']}/{r['robustness']['recommendation_stability']}")
        for oid, o in r["options"].items():
            print(f"   {oid:14s} mean=£{o['mean_gbp']:,.0f} p10=£{o['p10_gbp']:,.0f} p50=£{o['p50_gbp']:,.0f} p90=£{o['p90_gbp']:,.0f} "
                  f"P(goal)={o['p_goal']} P(churn<=5%)={o['p_churn_ok']} win={o['win']}")
        print(f"   influence: " + ", ".join(f"{k}:{v['rank']}({v['score']:.3f})" for k, v in sorted(r['factor_influence'].items(), key=lambda kv: kv[1]['rank'] or 99)))
        print(f"   flippable edges: {r['robustness']['flippable_edges']}")
        print(f"   factor flips: {r['factor_flips']}")
        print(f"   plot: {r['plot']}")
        if "control_direct_vs_plot_isl_diff_keys" in r:
            print(f"   CONTROL direct-vs-PLoT ISL response differing top-level keys: {r['control_direct_vs_plot_isl_diff_keys']}")


if __name__ == "__main__":
    main()
