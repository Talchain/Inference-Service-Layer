"""Verdicts, worst tested level, regret, witnesses and disagreements over the REQUESTED population.

Exact oracle values are Fractions; engine values are Monte Carlo means of a declared-exact
model whose only spread is PLoT's edge floor. A side is decided only when the level is more
than Z standard errors from the target; otherwise it is `indeterminate` (never rounded into
a pass). Missing rows stay missing. Nothing is averaged across hypotheses.
"""
from fractions import Fraction as Q
import math

from .fixtures import G2_SETTINGS, HYPOTHESES, OPTIONS, TARGET, TARGET_OPERATOR, meets, oracle_level

Z = 4.0                                    # predeclared before the first run
METRIC_FAULTS = {"strict_comparison", "missing_as_success", "average_models", "unique_winner"}
# Brief v1.0 §6 G2 reference, checked by hand by the brief's author and the DL. Our oracle must reproduce it.
BRIEF_REFERENCE = {"meets": {"A": 4, "B": 8, "C": 8}, "worst": {"A": 50, "B": 90, "C": 91},
                   "max_regret": {"A": 41, "B": 15, "C": 25}}


def engine_side(level, se, operator=TARGET_OPERATOR):
    if level is None:
        return "missing"
    margin = level - float(TARGET)
    if abs(margin) <= Z * se:
        return "indeterminate"
    return "meets" if (margin > 0 if operator == ">" else margin >= 0) else "falls_short"


def oracle_side(level, operator=TARGET_OPERATOR):
    return "meets" if meets(level, operator) else "falls_short"


def cell_checks(cell):
    """Per-cell agreement between the engine row and the exact oracle."""
    o, e = cell["oracle_level"], cell["engine"]
    if e is None or e["status"] != "supported":
        return {"level_agrees": None, "side": "missing" if e is None else "failed"}
    tol = Z * e["se"] + 1e-9
    agrees = abs(e["level"] - float(o)) <= tol
    eside, oside = cell["engine_side"], cell["oracle_side"]
    if eside == oside:
        side = "agrees"
    elif eside == "indeterminate" and o == TARGET:
        side = "tie_unresolved_on_engine_path"          # the located boundary loss, expected
    else:
        side = "DISAGREES"
    return {"level_agrees": agrees, "level_error": e["level"] - float(o), "tolerance": tol, "side": side}


def summarise(cells, faults=frozenset()):
    """G2 summary per option over the 8 REQUESTED settings, oracle and engine side by side."""
    op = ">" if "strict_comparison" in faults else TARGET_OPERATOR
    g2 = [c for c in cells if c["group"] == "G2"]
    settings = sorted({(c["hypothesis"], c["x"], c["y"]) for c in g2})
    out = {"requested_settings": len(settings), "options": {}, "highest_tested": {}}
    for oid in OPTIONS:
        mine = {(c["hypothesis"], c["x"], c["y"]): c for c in g2 if c["option"] == oid}
        e_rows = {k: c["engine"] for k, c in mine.items() if c["engine"] and c["engine"]["status"] == "supported"}
        sides = {k: engine_side(r["level"], r["se"], op) for k, r in e_rows.items()}
        missing = [k for k in settings if k not in e_rows]
        meets_n = sum(s == "meets" for s in sides.values())
        if "missing_as_success" in faults:
            meets_n += len(missing)                       # FAULT: a missing row is not a success
        o_levels = [mine[k]["oracle_level"] for k in settings if k in mine]
        levels = [r["level"] for r in e_rows.values()]
        if "average_models" in faults:                    # FAULT: unweighted model average
            by_s = {}
            for (h, x, y), r in e_rows.items():
                by_s.setdefault((x, y), []).append(r["level"])
            levels = [sum(v) / len(v) for v in by_s.values()]
        out["options"][oid] = {
            "requested": len(settings), "completed": len(e_rows), "missing": [list(map(str, k)) for k in missing],
            "oracle": {"meets": sum(meets(l, op) for l in o_levels), "worst": float(min(o_levels)) if o_levels else None},
            "engine": {"meets": meets_n, "indeterminate": sum(s == "indeterminate" for s in sides.values()),
                       "falls_short": sum(s == "falls_short" for s in sides.values()),
                       "worst": min(levels) if levels else None,
                       "worst_se": max((r["se"] for r in e_rows.values()), default=None),
                       "complete": not missing},
        }
    # Regret: only where every option completed at that setting; otherwise that setting is incomplete.
    o_reg = {o: [] for o in OPTIONS}; e_reg = {o: [] for o in OPTIONS}; incomplete = []
    for key in settings:
        row = {c["option"]: c for c in g2 if (c["hypothesis"], c["x"], c["y"]) == key}
        best_o = max(row[o]["oracle_level"] for o in OPTIONS)
        for o in OPTIONS:
            o_reg[o].append(best_o - row[o]["oracle_level"])
        tied_o = sorted(o for o in OPTIONS if row[o]["oracle_level"] == best_o)
        if any(not row[o]["engine"] or row[o]["engine"]["status"] != "supported" for o in OPTIONS):
            incomplete.append(list(map(str, key)))
            out["highest_tested"][str(key)] = {"oracle": tied_o, "engine": "incomplete"}
            continue
        lv = {o: row[o]["engine"]["level"] for o in OPTIONS}
        best = max(lv.values())
        for o in OPTIONS:
            e_reg[o].append((best - lv[o], row[o]["engine"]["se"]))
        if "unique_winner" in faults:                     # FAULT: a tie forced to one option
            tied_e = [max(lv, key=lv.get)]
        else:
            tied_e = sorted(o for o in OPTIONS
                            if best - lv[o] <= Z * math.hypot(row[o]["engine"]["se"], max(r["engine"]["se"] for r in row.values())))
        out["highest_tested"][str(key)] = {"oracle": tied_o, "engine": tied_e}
    for o in OPTIONS:
        out["options"][o]["oracle"]["max_regret"] = float(max(o_reg[o])) if o_reg[o] else None
        out["options"][o]["engine"]["max_regret"] = max((r for r, _ in e_reg[o]), default=None)
    out["regret_incomplete_settings"] = incomplete
    return out


def joint_failure_witnesses(cells, side_key):
    """Baseline passes; x alone passes; y alone passes; x and y together fall short."""
    found = []
    corners = {(Q(0), Q(0)): "baseline", (Q(1), Q(0)): "x_alone", (Q(0), Q(1)): "y_alone", (Q(1), Q(1)): "joint"}
    for h in HYPOTHESES:
        for o in OPTIONS:
            row = {(c["x"], c["y"]): c for c in cells if c["group"] == "G2" and c["hypothesis"] == h and c["option"] == o}
            if not all(k in row for k in corners):
                continue
            s = {name: row[k][side_key] for k, name in corners.items()}
            if s["baseline"] == s["x_alone"] == s["y_alone"] == "meets" and s["joint"] == "falls_short":
                found.append({"hypothesis": h, "option": o,
                              "rows": {name: row[k]["scenario_id"] for k, name in corners.items()}})
    return found


def hypothesis_disagreements(cells, side_key):
    out = []
    for o in OPTIONS:
        for x, y in G2_SETTINGS:
            pair = {c["hypothesis"]: c for c in cells if c["group"] == "G2" and c["option"] == o and (c["x"], c["y"]) == (x, y)}
            if set(pair) == {"M0", "M1"} and pair["M0"][side_key] != pair["M1"][side_key]:
                out.append({"option": o, "x": float(x), "y": float(y),
                            "M0": pair["M0"][side_key], "M1": pair["M1"][side_key]})
    return out


def check_reference(cells, summary, comparability):
    """Every way the computed outputs can deviate from the independent exact reference.
    Empty list = clean. Tie cells unresolved on the engine path are a disclosed loss, not a failure."""
    fails = []
    for c in cells:
        k = c["checks"]
        if k["side"] == "DISAGREES" or k["level_agrees"] is False:
            fails.append(f"cell {c['scenario_id']} {c['option']}: level/side deviates from the exact oracle "
                         f"(oracle {float(c['oracle_level'])}, engine {c['engine'] and c['engine']['level']})")
    fixture_settings = {(h, x, y) for h in HYPOTHESES for x, y in G2_SETTINGS}
    present = {(c["hypothesis"], c["x"], c["y"]) for c in cells if c["group"] == "G2"}
    if present != fixture_settings:
        fails.append(f"requested G2 population is {len(present)} settings, fixture defines {len(fixture_settings)}: "
                     f"missing {sorted(map(str, fixture_settings - present))}")
    for o, s in summary["options"].items():
        if s["oracle"]["meets"] != BRIEF_REFERENCE["meets"][o] and present == fixture_settings:
            fails.append(f"{o}: oracle meets {s['oracle']['meets']}/8 but the brief's reference is {BRIEF_REFERENCE['meets'][o]}/8")
        mine = [c for c in cells if c["group"] == "G2" and c["option"] == o]
        recount = sum(c["engine_side"] == "meets" for c in mine)
        if s["engine"]["meets"] != recount:
            fails.append(f"{o}: summary counts {s['engine']['meets']} meets but its rows show {recount}")
        supported = [c for c in mine if c["engine"] and c["engine"]["status"] == "supported"]
        if supported:
            true_min = min(c["engine"]["level"] for c in supported)
            if s["engine"]["worst"] is None or abs(s["engine"]["worst"] - true_min) > 1e-9:
                fails.append(f"{o}: worst tested level {s['engine']['worst']} is not the minimum of its rows ({true_min})")
        if present == fixture_settings:
            if s["oracle"]["worst"] != BRIEF_REFERENCE["worst"][o] or s["oracle"]["max_regret"] != BRIEF_REFERENCE["max_regret"][o]:
                fails.append(f"{o}: oracle worst/regret {s['oracle']['worst']}/{s['oracle']['max_regret']} "
                             f"differ from the brief's {BRIEF_REFERENCE['worst'][o]}/{BRIEF_REFERENCE['max_regret'][o]}")
            er = s["engine"]["max_regret"]
            if er is not None and abs(er - BRIEF_REFERENCE["max_regret"][o]) > Z * 2 * (s["engine"]["worst_se"] or 0) + 1e-9:
                fails.append(f"{o}: engine max regret {er:.4f} vs reference {BRIEF_REFERENCE['max_regret'][o]}")
    for key, v in summary["highest_tested"].items():
        if v["engine"] != "incomplete" and v["engine"] != v["oracle"]:
            fails.append(f"setting {key}: highest tested level held by {v['engine']}, exact reference {v['oracle']}")
    if not comparability["comparable"]:
        fails.append("rows compared across different " + ", ".join(comparability["differs"]))
    return fails


def sum_route_failures(sum_route):
    """The UNRESOLVED (engine floor) rows re-asked on the core's exact-sum route must come back exact and on
    the fixture's own side (inclusive >=). A strict comparison sent to the engine flips a tie and is caught here."""
    fails = []
    for r in sum_route:
        p, want = r["p_goal"], meets(Q(r["exact_level"]))
        if r["identity_evaluated"] is not True or p not in (0.0, 1.0) or (p == 1.0) != want:
            fails.append(f"exact-sum route {r['option']} {r['scenario_id']}: P(goal) {p}, identity evaluated "
                         f"{r['identity_evaluated']}, but exact {r['exact_level']} {'meets' if want else 'falls short'} (inclusive)")
    return fails


def completeness_gaps(cells, analysis):
    """What stops a clean reference check from being a PASS: unevaluated rows, and any of the brief's §0
    demonstrations 1-3 the engine rows do not show. An incomplete run is INCOMPLETE, never PASS."""
    gaps = [f"row not evaluated: {c['option']} {c['scenario_id']} ({c['checks']['side']})"
            for c in cells if c["engine"] is None or c["engine"]["status"] != "supported"]
    if not analysis["witnesses"]["engine"]:
        gaps.append("demonstration 1 missing: no joint-failure witness on engine rows")
    det = {"meets", "falls_short"}
    if not [d for d in analysis["disagreements"]["engine"] if {d["M0"], d["M1"]} <= det]:
        gaps.append("demonstration 2 missing: no determinate disagreement between hypotheses on engine rows")
    s = analysis["summary"]["options"]
    withstand = [o for o in s if s[o]["engine"]["complete"] and s[o]["engine"]["falls_short"] == 0]
    if len(withstand) < 2:
        gaps.append(f"demonstration 3 missing: fewer than two options withstand every tested setting ({withstand})")
    else:
        low_regret = min(withstand, key=lambda o: s[o]["engine"]["max_regret"])
        high_worst = max(withstand, key=lambda o: s[o]["engine"]["worst"])
        if low_regret == high_worst:
            gaps.append(f"demonstration 3 missing: {low_regret} leads on both regret and worst outcome, so no trade-off is shown")
    return gaps
