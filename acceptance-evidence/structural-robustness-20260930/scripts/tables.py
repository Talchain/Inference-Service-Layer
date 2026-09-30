#!/usr/bin/env python3
"""SCI-STRUCTURAL-ROBUSTNESS 30 Sep 2026: render the results tables (markdown) from summary.json.
Numbers are never hand-transcribed into the report: this script prints them."""
import json
import os

OUT = os.environ.get("SSR_OUT", "/private/tmp/ssr-20260930/out")
S = json.load(open(f"{OUT}/summary.json"))
SEEDS = ["1254899477", "1", "20260930"]
LABEL = {"raise_to_59": "Raise to £59", "keep_49_price": "Keep £49", "raise_to_54": "Raise to £54"}


def gbp(x):
    return "—" if x is None else f"£{x:,.0f}"


def pct(x):
    return "—" if x is None else f"{x:.4f}".rstrip("0").rstrip(".") if x not in (0, 1) else f"{x:.0f}"


def rng(vals, fmt):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    return fmt(lo) if abs(hi - lo) < 1e-12 else f"{fmt(lo)}–{fmt(hi)}"


def row(model, optset, oid, key):
    return [S[f"{model}-{optset}-s{s}"]["options"].get(oid, {}).get(key) for s in SEEDS if f"{model}-{optset}-s{s}" in S]


def table_options(optset):
    oids = ["raise_to_59", "keep_49_price"] + (["raise_to_54"] if optset == "served3" else [])
    print(f"\n| Model | Option | MRR mean (£/month, 12-mo horizon) | p10–p90 (seed {SEEDS[0]}) | P(MRR > £85k) | P(churn ≤ 5%) | Win share |")
    print("|---|---|---|---|---|---|---|")
    for m in ["A", "B", "C"]:
        for oid in oids:
            base = S[f"{m}-{optset}-s{SEEDS[0]}"]["options"].get(oid, {})
            print(f"| {m} | {LABEL[oid]} | {rng(row(m, optset, oid, 'mean_gbp'), gbp)} | {gbp(base.get('p10_gbp'))}–{gbp(base.get('p90_gbp'))} "
                  f"| {rng(row(m, optset, oid, 'p_goal'), pct)} | {rng(row(m, optset, oid, 'p_churn_ok'), pct)} | {rng(row(m, optset, oid, 'win'), pct)} |")


def table_structure(optset):
    print(f"\n| Model | Top option (all 3 seeds) | ISL robustness level | Recommendation stability | Flippable belief edges (strength now → flip) | Influence ranking (ISL factor_sensitivity) |")
    print("|---|---|---|---|---|---|")
    for m in ["A", "B", "C"]:
        rs = [S[f"{m}-{optset}-s{s}"] for s in SEEDS]
        tops = sorted({r["isl_top_option"] for r in rs})
        levels = sorted({r["robustness"]["level"] for r in rs})
        stab = rng([r["robustness"]["recommendation_stability"] for r in rs], pct)
        fl = rs[0]["robustness"]["flippable_edges"]
        fl_s = "; ".join(f"{e['edge']} ({e['current']} → {e['flip']:.4f})" for e in fl) or "none (price → MRR is a definition, not a belief)"
        inf = rs[0]["factor_influence"]
        inf_s = " > ".join(f"{k} ({v['score']:.2f})" for k, v in sorted(inf.items(), key=lambda kv: kv[1]["rank"] or 99))
        same_inf = all(r["factor_influence"] == rs[0]["factor_influence"] for r in rs)
        print(f"| {m} | {', '.join(LABEL[t] for t in tops)} | {', '.join(levels)} | {stab} | {fl_s} | {inf_s}{'' if same_inf else ' (varies by seed)'} |")


def table_product(optset):
    print(f"\n| Model | PLoT verdict (seed {SEEDS[0]}) | Recommended | Goal figures shown? | Withheld code | PLoT fragile edges |")
    print("|---|---|---|---|---|---|")
    for m in ["A", "B", "C"]:
        p = S[f"{m}-{optset}-s{SEEDS[0]}"]["plot"]
        shown = "yes" if all(p["goal_figures_shown"].values()) else ("no" if not any(p["goal_figures_shown"].values()) else "partly")
        print(f"| {m} | {p['display_verdict']} | {LABEL.get(p['recommended'], p['recommended'])} | {shown} | {', '.join(p['withheld_codes']) or '—'} | {', '.join(p['fragile_edges']) or '—'} |")


def controls():
    print("\n| Run (seed " + SEEDS[0] + ") | ISL fields differing, in-process (budgets raised) vs the ISL response PLoT received |")
    print("|---|---|")
    for m in ["A", "B", "C"]:
        for o in ["cur", "served3"]:
            r = S[f"{m}-{o}-s{SEEDS[0]}"]
            print(f"| {m}-{o} | {r.get('control_direct_vs_plot_isl_diff_keys')} |")
    degraded = {t: r["phase_degraded"] for t, r in S.items() if r["phase_degraded"]}
    print(f"\nOptional-phase degradation warnings across all {len(S)} runs: {degraded or 'none'}")
    print(f"n_samples per run: {sorted({r['n_samples'] for r in S.values()})}; seeds: {sorted({str(r['seed_used']) for r in S.values()})}")


for optset, title in [("cur", "Option set 1 — today's CEE submission (Keep £49, Raise to £59)"),
                      ("served3", "Option set 2 — the served-era set (+ Olumi's 'Raise to £54')")]:
    print(f"\n#### {title}")
    table_options(optset)
    table_structure(optset)
    table_product(optset)
print("\n#### Controls")
controls()
