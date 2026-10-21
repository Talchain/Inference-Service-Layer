"""One command: G1 + the eight G2 settings through local PLoT->ISL, the exact oracle beside
every row, and the negative controls (each must turn the reference check RED).

  cd experiments/track3_stress && <isl-venv>/bin/python -m track3.run --out <fresh dir>
"""
import argparse, json, os, platform, resource, subprocess, sys, time
from fractions import Fraction as Q

from . import fixtures as F
from .evaluator_adapter import ROOT, Engine, rows_from
from . import m2
from .report import write_report
from .scenario_builder import BUILDER_FAULTS, SCALE, build_request, build_sum_route_request, graph_level
from .stress_metrics import (METRIC_FAULTS, Z, cell_checks, check_reference, completeness_gaps, engine_side,
                             hypothesis_disagreements, joint_failure_witnesses, oracle_side, sum_route_failures, summarise)

SEED, N_SAMPLES = 20261021, 2000
SET_FAULTS = {"drop_adverse": ("M1", Q(0), Q(1)), "oat_only": None}
NEGATIVE_CONTROLS = [  # (fault, brief group, what it breaks)
    ("omit_cost_stress", "G1", "omit one stress"), ("reverse_demand_sign", "G1", "reverse a stress's sign"),
    ("strict_comparison", "G1", "change >= to >"), ("missing_as_success", "G1", "replace a missing row with success"),
    ("stale_cache", "G1", "reuse a cached result after a stress changes"),
    ("remove_m1_path", "G2", "remove the M1 path"), ("drop_adverse", "G2", "drop an adverse scenario"),
    ("average_models", "G2", "average the models without weights"), ("oat_only", "G2", "perturb only one coordinate"),
    ("unique_winner", "G2", "select a unique winner at the A/B tie"), ("horizon_mismatch", "G2", "compare different horizons"),
    ("reorder_options", "G2", "reorder options (A/B/C differ, so a swapped binding would show; must stay GREEN)"),
]
EXPECT_GREEN = {"reorder_options"}


def requested_settings(faults):
    settings = [(h, x, y, "G2") for h in F.HYPOTHESES for x, y in F.G2_SETTINGS]
    if "oat_only" in faults:
        settings = [s for s in settings if not (s[1] == 1 and s[2] == 1)]
    if "drop_adverse" in faults:
        settings = [s for s in settings if s[:3] != SET_FAULTS["drop_adverse"]]
    return settings + [("M0", x, y, "G1") for x, y in F.G1_EXTRA]


def evaluate_all(engine, faults, calls_log):
    cells = []
    for h, x, y, group in requested_settings(faults):
        body, change_record = build_request(h, x, y, seed=SEED, n_samples=N_SAMPLES, faults=faults & BUILDER_FAULTS,
                                            strict="strict_comparison" in faults)
        if "reorder_options" in faults:
            body["options"] = list(reversed(body["options"]))       # results must still bind by option id
        result = engine.evaluate(body, h)
        rows = rows_from(result, SCALE)
        horizon = 24 if ("horizon_mismatch" in faults and h == "M1") else F.HORIZON_MONTHS
        calls_log.append({"scenario_id": F.scenario_id(h, x, y), "faults": sorted(faults), "http_status": result["http_status"],
                          "analysis_status": result["plot"].get("analysis_status"), "elapsed_s": round(result["elapsed_s"], 3),
                          "cache_hit": result["cache_hit"], "isl_calls": len(result["isl_exchanges"])})
        for oid in (["A"] if group == "G1" else F.OPTIONS):
            engine_row = rows.get(oid)
            if {"missing_as_success", "missing_row"} & faults and (h, x, y, oid) == ("M0", 1, 1, "A"):
                engine_row = None                            # the control: this row failed to evaluate
            o_level = F.oracle_level(oid, h, x, y)
            cell = {"group": group, "hypothesis": h, "x": x, "y": y, "scenario_id": F.scenario_id(h, x, y), "option": oid,
                    "oracle_level": o_level, "oracle_side": oracle_side(o_level),
                    "graph_level": graph_level(body, oid), "engine": engine_row,
                    "engine_side": engine_side(engine_row["level"], engine_row["se"]) if engine_row and engine_row["status"] == "supported" else "missing",
                    "meta": {"unit": F.UNIT, "horizon_months": horizon, "target": float(F.TARGET), "operator": F.TARGET_OPERATOR},
                    "change_record": change_record, "request": body, "result": result}
            cell["checks"] = cell_checks(cell)
            cells.append(cell)
    return cells


def evaluate_sum_route(engine, cells, faults, calls_log):
    """Re-ask every exact-tie row on the core's exact-sum route. The strict_comparison fault sends '>' to the engine."""
    out = []
    for c in [c for c in cells if c["oracle_level"] == F.TARGET]:
        body = build_sum_route_request(c["hypothesis"], c["x"], c["y"], seed=SEED, n_samples=N_SAMPLES)
        if "strict_comparison" in faults:
            body["goal_threshold_strict"] = True
        res = engine.evaluate(body, "sum-route")
        calls_log.append({"scenario_id": "sum-route " + c["scenario_id"], "faults": sorted(faults), "http_status": res["http_status"],
                          "analysis_status": res["plot"].get("analysis_status"), "elapsed_s": round(res["elapsed_s"], 3),
                          "cache_hit": res["cache_hit"], "isl_calls": len(res["isl_exchanges"])})
        isl = res["isl_exchanges"][-1] if res["isl_exchanges"] else {"request": {}, "response": {}}
        raw = next((o for o in (isl.get("response") or {}).get("options", []) if o["id"] == c["option"]), {})
        ident = (res["plot"].get("identity_evaluations") or [{}])[0]
        out.append({"scenario_id": c["scenario_id"], "option": c["option"], "exact_level": str(c["oracle_level"]),
                    "isl_level_normalised": repr((raw.get("outcome") or {}).get("mean")),
                    "threshold_normalised": repr(isl["request"].get("goal_threshold")),
                    "strict_sent": bool(isl["request"].get("goal_threshold_strict")),
                    "p_goal": raw.get("probability_of_goal"), "identity_evaluated": ident.get("evaluated")})
    for r in out:
        r["resolved"] = not sum_route_failures([r])
    return out


def comparability_of(cells):
    keys = ["unit", "horizon_months", "target", "operator"]
    differs = [k for k in keys if len({c["meta"][k] for c in cells}) > 1]
    return {"comparable": not differs, "differs": differs, "shared": {k: cells[0]["meta"][k] for k in keys if k not in differs}}


def analyse(cells, faults=frozenset(), sum_route=()):
    summary = summarise(cells, faults & METRIC_FAULTS)
    comparability = comparability_of(cells)
    out = {"summary": summary, "comparability": comparability, "sum_route": list(sum_route),
           "witnesses": {"oracle": joint_failure_witnesses(cells, "oracle_side"), "engine": joint_failure_witnesses(cells, "engine_side")},
           "disagreements": {"oracle": hypothesis_disagreements(cells, "oracle_side"), "engine": hypothesis_disagreements(cells, "engine_side")},
           "reference_failures": check_reference(cells, summary, comparability) + sum_route_failures(sum_route)}
    out["gaps"] = completeness_gaps(cells, out)
    out["verdict"] = "FAIL" if out["reference_failures"] else ("INCOMPLETE" if out["gaps"] else "PASS")
    return out


def engine_only_probe(capture_dir):
    """Engine-only (labelled): resend one captured translated request straight to ISL with the
    declared spread 0 restored on its edges. Records whether ISL can carry an exact edge."""
    import urllib.request, urllib.error
    first = sorted(f for f in os.listdir(capture_dir))[0]
    rec = json.load(open(os.path.join(capture_dir, first)))
    body = rec["request"]
    for e in body["graph"]["edges"]:
        e["strength"]["std"] = 0.0
    req = urllib.request.Request("http://127.0.0.1:18731" + rec["path"], data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return f"ISL accepted edge std 0 (HTTP {r.status}) for captured request {first}"
    except urllib.error.HTTPError as e:
        msg = json.loads(e.read() or b"{}").get("status_reason", "")
        return f"ISL refused edge std 0 for captured request {first}: HTTP {e.code} '{msg}'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-controls", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit(f"--out {a.out} must be a fresh directory")
    os.makedirs(os.path.join(a.out, "capture"), exist_ok=True)
    load = os.getloadavg()[0]
    if load >= 25:
        sys.exit(f"1-minute load {load:.1f} >= 25: not starting the engine")
    t0, calls = time.perf_counter(), []
    with Engine(os.path.join(a.out, "capture")) as engine:
        clean_cells = evaluate_all(engine, frozenset(), calls)
        clean = analyse(clean_cells, sum_route=evaluate_sum_route(engine, clean_cells, frozenset(), calls))
        controls = []
        if not a.skip_controls:
            stale = Engine(engine.capture_dir, faults={"stale_cache"}); stale.identity = engine.identity
            for fault, group, meaning in NEGATIVE_CONTROLS:
                runner = stale if fault == "stale_cache" else engine
                cells = evaluate_all(runner, frozenset({fault}), calls)
                sr = evaluate_sum_route(engine, cells, frozenset({fault}), calls) if fault == "strict_comparison" else ()
                fails = analyse(cells, frozenset({fault}), sr)["reference_failures"]
                controls.append({"fault": fault, "group": group, "meaning": meaning, "detected": bool(fails),
                                 "first_failures": fails[:3], **({"expected_detected": False} if fault in EXPECT_GREEN else {})})
            # Contrast for missing_as_success: the same failed row, counted honestly, must NOT go red.
            honest = analyse(evaluate_all(engine, frozenset({"missing_row"}), calls))
            controls.append({"fault": "missing_row (contrast, no fault)", "group": "G1",
                             "meaning": "same failed row, reported as missing", "detected": bool(honest["reference_failures"]),
                             "expected_detected": False, "first_failures": honest["reference_failures"][:3],
                             "verdict": honest["verdict"], "expected_verdict": "INCOMPLETE", "gaps": honest["gaps"],
                             "A_summary": honest["summary"]["options"]["A"]})
        probes = [engine_only_probe(engine.capture_dir)]
        # ── Milestone 2: G3 and G4 on the same engine session.
        g3 = m2.run_g3(engine, frozenset(), calls, SEED, 10000)
        g4 = m2.run_g4(engine, frozenset(), calls, SEED, 10000)
        g4_probes = m2.g4_boundary_probes(engine, calls, SEED, 10000)
        clean["m2"] = {"g3": g3, "g4": g4, "g4_boundary": g4_probes,
                       "failures": m2.g3_failures(g3) + m2.g4_failures(g4)}
        if clean["m2"]["failures"]:
            clean["reference_failures"] += clean["m2"]["failures"]
            clean["verdict"] = "FAIL"
        if not a.skip_controls:
            controls += m2.m2_controls(engine, g4, calls, SEED, 10000)
        rss = {}
        for name in ("isl", "plot"):
            try:
                pid = open(os.path.join(ROOT, "logs", f"{name}.pid")).read().strip()
                kids = subprocess.run(["pgrep", "-P", pid], capture_output=True, text=True).stdout.split()
                rss[name + "_rss_mb"] = round(sum(int(subprocess.run(["ps", "-o", "rss=", "-p", p], capture_output=True, text=True).stdout.strip() or 0)
                                                  for p in [pid, *kids]) / 1024, 1)
            except Exception as exc:
                rss[name + "_rss_mb"] = f"UNKNOWN ({exc})"
        identity, n_calls = engine.identity, engine.calls + (0 if a.skip_controls else stale.calls)
        first_isl = clean_cells[0]["result"]["isl_exchanges"][0]["response"]
        identity["isl_reported"] = {k: first_isl.get(k) for k in ("version", "endpoint_version", "engine_version", "build")}
        identity["plot_reported_build"] = (clean_cells[0]["result"]["plot"].get("meta") or {}).get("build")
    wall = time.perf_counter() - t0
    manifest = {
        "brief": "Track 3 brief v1.0, 21 Oct 2026 (outbox/TRACK3-BRIEF-v1-20261021.md)", "milestone": 1,
        "path": "service boundary: runner -> local PLoT /v2/run -> capture proxy -> local ISL (auth off, no network beyond loopback)",
        "identity": identity, "seed": SEED, "n_samples": N_SAMPLES, "z": Z, "scale_gbp000_per_unit": float(SCALE),
        "python": platform.python_version(), "load_at_start": load, "boundary_probes": probes,
        "runtime": {"wall_s": round(wall, 1), "plot_calls": n_calls, "per_call_s": calls, **rss,
                    "runner_max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)},
    }
    write_report(a.out, manifest, clean_cells, clean, controls)
    as_expected = lambda c: c["detected"] == c.get("expected_detected", True) and c.get("verdict") == c.get("expected_verdict")
    ok = clean["verdict"] == "PASS" and all(as_expected(c) for c in controls)
    print(json.dumps({"verdict": clean["verdict"], "clean_reference_failures": len(clean["reference_failures"]), "gaps": len(clean["gaps"]),
                      "controls_as_expected": f"{sum(map(as_expected, controls))}/{len(controls)}",
                      "plot_calls": n_calls, "wall_s": round(wall, 1), "out": a.out}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
