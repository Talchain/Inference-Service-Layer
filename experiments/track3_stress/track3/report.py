"""Machine-readable results plus a short readable report. Model-relative wording only:
no 'best', 'winner' or 'recommend'; counts of tested settings are never called chances."""
import json, os
from fractions import Fraction as Q

from .fixtures import OPTIONS, TARGET, UNIT
from .stress_metrics import BRIEF_REFERENCE, Z


def _f(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def _compact(c):
    e = c["engine"]
    return {k: c[k] for k in ("group", "hypothesis", "scenario_id", "option", "oracle_side", "engine_side", "checks", "meta")} | {
        "x": str(c["x"]), "y": str(c["y"]), "oracle_level": str(c["oracle_level"]), "graph_level": str(c["graph_level"]),
        "engine": e, "request_id": c["request"]["request_id"]}


def write_report(out, manifest, cells, clean, controls):
    calls_dir = os.path.join(out, "calls"); os.makedirs(calls_dir, exist_ok=True)
    seen = set()
    for c in cells:                                   # one file per evaluated request: body, change record, PLoT reply, ISL exchange
        if c["scenario_id"] in seen:
            continue
        seen.add(c["scenario_id"])
        with open(os.path.join(calls_dir, c["scenario_id"].replace(":", "_").replace(",", "_") + ".json"), "w") as f:
            json.dump({"request": c["request"], "change_record": c["change_record"], "plot_response": c["result"]["plot"],
                       "isl_exchanges": c["result"]["isl_exchanges"], "elapsed_s": c["result"]["elapsed_s"]}, f, indent=1)
    floor = sorted({(e["from"], e["to"], e["strength"]["mean"], e["strength"]["std"])
                    for c in cells for x in c["result"]["isl_exchanges"] for e in x["request"]["graph"]["edges"]})
    results = {"manifest": manifest, "analysis": clean, "controls": controls, "cells": [_compact(c) for c in cells],
               "translated_edge_spreads": [dict(zip(("from", "to", "mean", "std_sent_to_isl"), f)) for f in floor]}
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=str)

    s, fails = clean["summary"], clean["reference_failures"]
    ties = [c for c in cells if c["checks"]["side"] == "tie_unresolved_on_engine_path"]
    L = []
    w = L.append
    ok_controls = sum(c["detected"] == c.get("expected_detected", True) and c.get("verdict") == c.get("expected_verdict") for c in controls)
    w("# Track 3: G1–G4 on the local PLoT→ISL path (Milestones 1 and 2)\n")
    w(f"**Verdict: {clean['verdict']}.** {len(fails)} reference failures against the exact oracle; {len(clean['gaps'])} completeness gaps. "
      f"{len(ties)} exact-tie rows are **UNRESOLVED (engine floor)** on the floored path, never counted as met or missed; "
      f"{sum(r['resolved'] for r in clean['sum_route'])}/{len(clean['sum_route'])} resolve on the core's exact-sum route (below). "
      f"Negative controls behaved as expected: {ok_controls}/{len(controls)}.\n")
    for g in clean["gaps"]:
        w(f"- GAP: {g}")
    idn = manifest["identity"]
    w(f"- Path: {manifest['path']}.")
    w(f"- PLoT `{idn['plot_commit']}` · ISL `{idn['isl_commit']}` · @talchain/schemas {idn['plot_talchain_schemas']} · "
      f"ISL src identical to pin: {idn.get('isl_src_identical_to_pin')} · ISL analyser `{idn['isl_analyzer_module']}` · runner `{idn.get('runner_commit', '?')[:8]}`{' (+ uncommitted)' if idn.get('runner_dirty') else ''} · seed {manifest['seed']} · n = {manifest['n_samples']} draws per G1/G2 call, 10,000 per G3/G4 call · Z = {Z}.")
    w(f"- Units: {UNIT}, 12 months, target Y ≥ {TARGET} (inclusive). PLoT scale: goal_threshold_cap = {manifest['scale_gbp000_per_unit']:g} {UNIT} per unit.")
    rt = manifest["runtime"]
    w(f"- Runtime: {rt['wall_s']} s wall for {rt['plot_calls']} PLoT calls (clean run and controls); "
      f"ISL {rt.get('isl_rss_mb')} MB, PLoT {rt.get('plot_rss_mb')} MB resident at the end.\n")

    w("## G1: option A (Growth-first), Y = 120 − 20x − 20y, under M0\n")
    w("| x | y | Exact Y | Exact side | Engine level ± SE | Engine P(goal) | Engine side | Check |")
    w("|---:|---:|---:|---|---:|---:|---|---|")
    g1 = [c for c in cells if c["option"] == "A" and c["hypothesis"] == "M0"]
    for c in sorted(g1, key=lambda c: (c["x"] + c["y"], c["x"])):
        e = c["engine"] or {}
        w(f"| {float(c['x']):g} | {float(c['y']):g} | {float(c['oracle_level']):g} | {c['oracle_side']} | "
          f"{_f(e.get('level'))} ± {_f(e.get('se'), 3)} | {_f(e.get('p_goal'), 3)} | {c['engine_side']} | {c['checks']['side']} |")
    wit = clean["witnesses"]["engine"]
    names = ", ".join(x["option"] + " under " + x["hypothesis"] for x in wit) or "none"
    w(f"\n**Joint failure (engine rows):** {names} — "
      "meets at (0,0), at x alone and at y alone, falls short at x and y together. An additive threshold, not an interaction term.\n")

    w("## G2: eight settings, three options, two explanations\n")
    w("| Hypothesis | x | y | " + " | ".join(f"{o} exact / engine" for o in OPTIONS) + " | Highest tested level |")
    w("|---|---:|---:|" + "---:|" * len(OPTIONS) + "---|")
    g2 = [c for c in cells if c["group"] == "G2"]
    for h, x, y in sorted({(c["hypothesis"], c["x"], c["y"]) for c in g2}):
        row = {c["option"]: c for c in g2 if (c["hypothesis"], c["x"], c["y"]) == (h, x, y)}
        cellstr = lambda c: f"{float(c['oracle_level']):g} / {_f(c['engine'] and c['engine']['level'])}" + (" UNRESOLVED (engine floor)" if c["checks"]["side"].startswith("tie") else "")
        ht = s["highest_tested"].get(str((h, x, y)), {})
        w(f"| {h} | {float(x):g} | {float(y):g} | " + " | ".join(cellstr(row[o]) for o in OPTIONS) +
          f" | {'/'.join(ht.get('engine', [])) if isinstance(ht.get('engine'), list) else ht.get('engine')} |")
    w("\n| Measure over the 8 tested settings | " + " | ".join(OPTIONS) + " |")
    w("|---|" + "---:|" * len(OPTIONS))
    w("| Meets ≥ 90, exact (brief reference) | " + " | ".join(f"{s['options'][o]['oracle']['meets']}/8 ({BRIEF_REFERENCE['meets'][o]}/8)" for o in OPTIONS) + " |")
    w("| Meets / UNRESOLVED (engine floor) / falls short, engine | " + " | ".join(
        f"{s['options'][o]['engine']['meets']} / {s['options'][o]['engine']['indeterminate']} / {s['options'][o]['engine']['falls_short']}" for o in OPTIONS) + " |")
    w("| Worst tested outcome, exact / engine | " + " | ".join(f"{_f(s['options'][o]['oracle']['worst'], 0)} / {_f(s['options'][o]['engine']['worst'])}" for o in OPTIONS) + " |")
    w("| Maximum tested regret, exact / engine | " + " | ".join(f"{_f(s['options'][o]['oracle']['max_regret'], 0)} / {_f(s['options'][o]['engine']['max_regret'])}" for o in OPTIONS) + " |")
    ws = clean.get("withstand", {})
    w("\n**Withstands every tested setting:** " + ("; ".join(f"{o} ({basis})" for o, (ok, basis) in ws.items() if ok) or "none")
      + ". **Does not:** " + ("; ".join(f"{o} ({basis})" for o, (ok, basis) in ws.items() if not ok) or "none") + ".")
    det = {"meets", "falls_short"}
    dis = [d for d in clean["disagreements"]["engine"] if {d["M0"], d["M1"]} <= det]
    w(f"\n**Explanations disagree (engine, both rows determinate):** " + "; ".join(f"{d['option']} at ({d['x']:g},{d['y']:g}) {d['M0']} under M0, {d['M1']} under M1" for d in dis) + ".")
    moved = []
    for x, y in sorted({(c["x"], c["y"]) for c in g2}):
        h0, h1 = (s["highest_tested"].get(str((h, x, y)), {}).get("engine") for h in ("M0", "M1"))
        if isinstance(h0, list) and isinstance(h1, list) and h0 != h1:
            moved.append(f"({float(x):g},{float(y):g}) {'/'.join(h0)} → {'/'.join(h1)}")
    w(f"**Adding the M1 mechanism changes which option holds the highest tested level** at: {'; '.join(moved) or 'no setting'}.")
    w("B has the smallest maximum tested regret; C has the highest worst tested outcome; A has the largest favourable-case "
      "outcome but falls short at the joint stress under M0 and at every stressed setting under M1. These are model-relative "
      "findings over 8 tested settings: no option is a universal winner, and no probability is placed on M0 or M1.\n")

    w("## Located boundary losses (the engine path, not the fixture)\n")
    w(f"- **Exact ties are undecidable on this path.** {len(ties)} rows sit exactly on the target "
      f"({', '.join(sorted({c['option'] + ' ' + c['scenario_id'] for c in ties}))}). ISL rejects any edge spread ≤ 0.001 and PLoT floors "
      "causal edges at min(0.05, max(0.001, |mean|/2)) (`graph-normaliser.ts:1078`), so a declared-exact model reaches ISL with spread; "
      "at an exact tie the level estimate cannot certify ≥ 90 and P(goal) is a coin flip. They are reported as unresolved, never as met.")
    w("- **Raw £000 graphs are not the supported input.** Sent un-normalised, PLoT capped a strength of 100 to 1, re-scaled interventions, "
      "synthesised lever spreads for `std: 0`, and clipped the target 90 to 1 (probe, this session). The runner therefore uses PLoT's own goal ruler (`goal_threshold_cap`).")
    art = {c["scenario_id"]: c for c in cells if c["option"] == "A" and c["group"] == "G1"}
    near = [c for c in art.values() if c["oracle_level"] != TARGET and c["engine"]]
    if near:
        w("- **On this path P(goal) is an artefact for an exact model.** " + "; ".join(
            f"exact £{float(c['oracle_level']):g}k shows P(goal) {c['engine']['p_goal']:.2f}" for c in near) +
          ". The spread comes only from PLoT's floor, so the runner decides the side from the level and never reports these as chances.")
    for b in manifest.get("boundary_probes", []):
        w(f"- Engine-only probe: {b}")
    sr = clean["sum_route"]
    if sr:
        w("\n### The core's exact-sum route on the UNRESOLVED rows (goal = today + change; CEE `mapper.ts` exact sum)\n")
        w("| Row | Exact | ISL level (normalised) | Threshold | Strict sent | P(goal) | Identity evaluated | Resolved |")
        w("|---|---:|---:|---:|---|---:|---|---|")
        for r in sr:
            w(f"| {r['option']} {r['scenario_id']} | {r['exact_level']} | {r['isl_level_normalised']} | {r['threshold_normalised']} | "
              f"{r['strict_sent']} | {r['p_goal']} | {r['identity_evaluated']} | {'yes' if r['resolved'] else 'NO'} |")
        w("\nThis route admits no parent of the goal except option-set changes and no negative change, so the runner computes each "
          "option's stressed change exactly and the engine only adds and compares: it proves the exact comparison, not stress "
          "propagation. The inclusive tie holds by one unit in the last place (level 0.4500000000000001 against 0.45), which comes from the cap round-trip.")
    w("")
    m = clean.get("m2")
    if m:
        w("## Milestone 2 · G3: identical marginals (0.9), different joints\n")
        w("| Declared joint law | Exact P(both) | Sent to engine | Engine P(both) | Engine marginals | n |")
        w("|---|---:|---|---:|---|---:|")
        for r in m["g3"]["rows"]:
            if r.get("unmapped"):
                w(f"| {r['law'].replace('_', ' ')} | {float(Q(r['exact_p_both'])):.2f} | **unsupported**: {r.get('reason')} | — | — | — |")
                continue
            sent = "no correlation" if r["rho_sent"] is None else f"Gaussian copula rho = {r['rho_sent']:g}"
            w(f"| {r['law'].replace('_', ' ')} | {float(Q(r['exact_p_both'])):.2f} | {sent} | {_f(r['engine_joint'], 4)} | "
              f"{', '.join(f'{k} {v:.4f}' for k, v in r['engine_marginals'].items())} | {r['n']} |")
        bounds = m["g3"]["marginals_only"].get("bounds")
        w(f"\n- Marginals alone: **not identified**; bounds [{float(Q(bounds[0])):.2f}, {float(Q(bounds[1])):.2f}]. Never 0.81 by default.")
        w("- Invalid tables refused, never repaired: " + "; ".join(f"{k} ({v.get('reason')})" for k, v in m["g3"]["admission"].items()) + ".")
        ref = m["g3"]["engine_refusal_rho_1_2"]
        w(f"- Engine refuses rho = 1.2: {ref['analysis_status']} (HTTP {ref['http_status']}).")
        w("- Scope: thresholded Normal drivers at rho 0, +1 and −1 realise these three laws in exact arithmetic. The engine "
          "regularises a singular ±1 pair to about ±0.99999999 (`src/utils/correlation.py:289`), so its joint is numerically approximate; "
          "the check is the stated Monte Carlo tolerance. Other joint laws have no mapping here and come back unsupported, unevaluated. "
          "ISL's copula note says 'zero tail dependence': an ideal rho = +1 has perfect same-tail dependence, and rho = −1 has opposite-tail "
          "dependence (zero same-tail).\n")
        w("## Milestone 2 · G4: Y ~ Normal(mu, 10²) £000, target Y ≥ 100\n")
        w("| mu | Exact P(Y ≥ 100) | Engine (target on the uncertain root, Channel B) | n | Within 4 SE + 1/n |")
        w("|---:|---:|---:|---:|---|")
        for r in m["g4"]["rows"]:
            w(f"| {r['mu']:g} | {r['exact']:.10f} | {_f(r['engine_p'], 4)} | {r['n']} | {r['within_tolerance']} |")
        b = m["g4_boundary"]
        w("\n**Boundary findings (reported, not gated):**")
        w("- **The headline chance (Channel A) distorts the declared law.** Through a goal node, PLoT's edge floor widens sd 10 to "
          + ", ".join(f"{x['engine_sd']:.2f}" for x in b["channel_a_floored"]) + ". P(goal) "
          + "; ".join(f"mu {x['mu']}: {x['engine_p']:.4f} vs exact {x['exact']:.4f}" for x in b["channel_a_floored"]) + ".")
        w("- **The exact-sum route with zero spread is determinate** (the brief's zero-spread counterpart): "
          + "; ".join(f"mu {x['mu']}: level {x['level']:.2f}, margin {x['level'] - 100:+.2f}, P(goal) {x['p_goal']}" for x in b["sum_zero_spread"]) + ".")
        d = b["sum_declared_spread"]
        w(f"- **The exact-sum route silently drops a declared spread.** PLoT sent {d['spread_sent_to_isl']} for today's level, but ISL returned "
          + "; ".join(f"mu {x['mu']}: sd {x['sd']}, P(goal) {x['p_goal']} (exact if drawn {x['exact_if_drawn']:.4f})" for x in d["rows"])
          + ". No warning says the declared spread was lost (only generic DEGENERATE_OPTION_ZERO_VARIANCE critiques).\n")
    w("## Negative controls (each fault must turn the reference check red)\n")
    w("| Fault | Brief | Expected | Observed | First failure it caused |")
    w("|---|---|---|---|---|")
    for c in controls:
        exp = ("RED" if c.get("expected_detected", True) else "GREEN") + (f", verdict {c['expected_verdict']}" if c.get("expected_verdict") else "")
        obs = ("RED" if c["detected"] else "GREEN") + (f", verdict {c['verdict']}" if c.get("verdict") else "")
        w(f"| {c['fault']} | {c['group']}: {c['meaning']} | {exp} | {obs} | {((c['first_failures'] or c.get('gaps') or ['—'])[0])[:140]} |")
    w("\n## Scope and what this does not show\n")
    w("- Synthetic fixtures from brief v1.0 §6; not customer evidence. Stresses are explicit coordinates, not probabilities; "
      "'met at 6 of 8 tested settings' is not a 75 % chance.")
    w("- Stresses are scenario parameter assumptions (edge strengths −x, −y on controllable exposures); options are interventions. "
      "M1's £30k is an illustrative assumption carried on a named intervention → retention → contribution path.")
    w("- In G1/G2 the engine spread is PLoT's floor artefact, not uncertainty in the fixture. G3/G4 carry genuinely declared laws. "
      "Monte Carlo precision is not model validity.")
    w(f"- Status: isolated prototype with numerical and service-boundary evidence. Not integrated, not served.")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
