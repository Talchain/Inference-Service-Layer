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
    ok_controls = sum(c["detected"] == c.get("expected_detected", True) for c in controls)
    w("# Track 3 Milestone 1: G1 and G2 on the local PLoT→ISL path\n")
    w(f"**Verdict:** {'PASS' if not fails else 'FAIL'} against the exact oracle on every determinate row "
      f"({len(fails)} reference failures). {len(ties)} exact-tie rows are **unresolved on the engine path** (a located boundary loss, below). "
      f"Negative controls behaved as expected: {ok_controls}/{len(controls)}.\n")
    idn = manifest["identity"]
    w(f"- Path: {manifest['path']}.")
    w(f"- PLoT `{idn['plot_commit']}` · ISL `{idn['isl_commit']}` · @talchain/schemas {idn['plot_talchain_schemas']} · "
      f"ISL analyser `{idn['isl_analyzer_module']}` · seed {manifest['seed']} · n = {manifest['n_samples']} draws per call · Z = {Z}.")
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
        cellstr = lambda c: f"{float(c['oracle_level']):g} / {_f(c['engine'] and c['engine']['level'])}" + (" (tie)" if c["checks"]["side"].startswith("tie") else "")
        ht = s["highest_tested"].get(str((h, x, y)), {})
        w(f"| {h} | {float(x):g} | {float(y):g} | " + " | ".join(cellstr(row[o]) for o in OPTIONS) +
          f" | {'/'.join(ht.get('engine', [])) if isinstance(ht.get('engine'), list) else ht.get('engine')} |")
    w("\n| Measure over the 8 tested settings | " + " | ".join(OPTIONS) + " |")
    w("|---|" + "---:|" * len(OPTIONS))
    w("| Meets ≥ 90, exact (brief reference) | " + " | ".join(f"{s['options'][o]['oracle']['meets']}/8 ({BRIEF_REFERENCE['meets'][o]}/8)" for o in OPTIONS) + " |")
    w("| Meets / unresolved tie / falls short, engine | " + " | ".join(
        f"{s['options'][o]['engine']['meets']} / {s['options'][o]['engine']['indeterminate']} / {s['options'][o]['engine']['falls_short']}" for o in OPTIONS) + " |")
    w("| Worst tested outcome, exact / engine | " + " | ".join(f"{_f(s['options'][o]['oracle']['worst'], 0)} / {_f(s['options'][o]['engine']['worst'])}" for o in OPTIONS) + " |")
    w("| Maximum tested regret, exact / engine | " + " | ".join(f"{_f(s['options'][o]['oracle']['max_regret'], 0)} / {_f(s['options'][o]['engine']['max_regret'])}" for o in OPTIONS) + " |")
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
    w("")
    w("## Negative controls (each fault must turn the reference check red)\n")
    w("| Fault | Brief | Expected | Observed | First failure it caused |")
    w("|---|---|---|---|---|")
    for c in controls:
        exp = "RED" if c.get("expected_detected", True) else "GREEN"
        obs = "RED" if c["detected"] else "GREEN"
        w(f"| {c['fault']} | {c['group']}: {c['meaning']} | {exp} | {obs} | {(c['first_failures'] or ['—'])[0][:140]} |")
    w("\n## Scope and what this does not show\n")
    w("- Synthetic fixtures from brief v1.0 §6; not customer evidence. Stresses are explicit coordinates, not probabilities; "
      "'met at 6 of 8 tested settings' is not a 75 % chance.")
    w("- Stresses are scenario parameter assumptions (edge strengths −x, −y on controllable exposures); options are interventions. "
      "M1's £30k is an illustrative assumption carried on a named intervention → retention → contribution path.")
    w("- Engine spread here is PLoT's floor artefact, not uncertainty in the fixture. Monte Carlo precision is not model validity.")
    w(f"- Status: isolated prototype with numerical and service-boundary evidence. Not integrated, not served.")
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
