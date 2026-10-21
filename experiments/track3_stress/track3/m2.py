"""Milestone 2: G3 (identical marginals, different joints) and G4 (a declared Normal law), on the same
local PLoT->ISL path, with the brief's negative and cross-cutting controls. Called from run.py."""
import json
import math
from fractions import Fraction as Q

from . import fixtures as F
from .scenario_builder import SCALE, build_g3_request, build_g4_request, build_sum_route_request, g3_rho_for

TOL_Z = 4.0


def _tol(p, n):
    return TOL_Z * math.sqrt(max(p * (1 - p), 0.0) / n) + 1.0 / n


def _first(res):
    return ((res["plot"].get("option_comparison") or [{}])[0], res)


def _log(calls, tag, res, faults):
    calls.append({"scenario_id": tag, "faults": sorted(faults), "http_status": res["http_status"],
                  "analysis_status": res["plot"].get("analysis_status"), "elapsed_s": round(res["elapsed_s"], 3),
                  "cache_hit": res["cache_hit"], "isl_calls": len(res["isl_exchanges"])})


def run_g3(engine, faults, calls, seed, n):
    admission = {name: F.admit_joint(t, (F.G3_MARGINAL, F.G3_MARGINAL)) for name, t in F.G3_INVALID.items()}
    if "accept_invalid_table" in faults:                                 # FAULT: an invalid table is admitted
        admission = {name: {"admitted": True} for name in F.G3_INVALID}
    marginals_only = F.joint_from_marginals_only(F.G3_MARGINAL, F.G3_MARGINAL)
    if "infer_independence" in faults:                                   # FAULT: missing dependence read as independence
        marginals_only = {"identified": True, "point": F.G3_MARGINAL ** 2}
    rows = []
    for law, table in F.G3_LAWS.items():
        adm = F.admit_joint(table)
        rho = g3_rho_for(adm)
        if "independent_in_shared" in faults and law == "always_together":
            rho = None                                                   # FAULT: shared case drawn independently
        body = build_g3_request(law, rho, seed=seed, n_samples=n, p=adm["p1"])
        res = engine.evaluate(body, "g3:" + law)
        _log(calls, "g3 " + law, res, faults)
        o, _ = _first(res)
        cp = o.get("constraint_probabilities") or {}
        nn = (o.get("probability_of_joint_goal_precision") or {}).get("n_informative") or 0
        reported = o.get("probability_of_joint_goal")
        if "multiply_marginals" in faults and cp:                        # FAULT: joint = product of marginals
            reported = cp.get("S1", 0) * cp.get("S2", 0)
        rows.append({"law": law, "exact_p_both": str(adm["p_both"]), "rho_sent": rho, "engine_joint": reported,
                     "engine_marginals": cp, "n": nn, "status": res["plot"].get("analysis_status"),
                     "copula": bool(res["isl_exchanges"]) and bool(res["isl_exchanges"][-1]["request"].get("factor_correlations")),
                     "options_agree": len({json.dumps(x.get("constraint_probabilities"), sort_keys=True)
                                           for x in res["plot"].get("option_comparison") or []}) == 1})
    if "average_dependence" in faults:                                   # FAULT: the three laws averaged into one
        mean = sum(r["engine_joint"] for r in rows) / len(rows)
        for r in rows:
            r["engine_joint"] = mean
    # Engine admission: an impossible correlation must be refused, not repaired.
    bad = build_g3_request("invalid_rho", 1.2, seed=seed, n_samples=n, p=F.G3_MARGINAL)
    res = engine.evaluate(bad, "g3:invalid_rho"); _log(calls, "g3 invalid rho 1.2", res, faults)
    engine_refusal = {"http_status": res["http_status"], "analysis_status": res["plot"].get("analysis_status"),
                      "reason": (res["plot"].get("status_reason") or "")[:200]}
    return {"rows": rows, "admission": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in admission.items()},
            "marginals_only": {k: (tuple(map(str, v)) if isinstance(v, tuple) else str(v)) for k, v in marginals_only.items()},
            "engine_refusal_rho_1_2": engine_refusal}


def g3_failures(g3):
    fails = []
    for r in g3["rows"]:
        exact = float(Q(r["exact_p_both"]))
        if r["engine_joint"] is None or r["n"] == 0:
            fails.append(f"G3 {r['law']}: no engine joint (status {r['status']})"); continue
        if abs(r["engine_joint"] - exact) > _tol(exact, r["n"]):
            fails.append(f"G3 {r['law']}: engine joint {r['engine_joint']:.4f} vs exact {exact} (tolerance {_tol(exact, r['n']):.4f})")
        for k, m in r["engine_marginals"].items():
            if abs(m - 0.9) > _tol(0.9, r["n"]):
                fails.append(f"G3 {r['law']}: marginal {k} {m:.4f} vs 0.9")
    if len({r["engine_joint"] for r in g3["rows"]}) < len(g3["rows"]):
        fails.append("G3: the three laws report one shared joint (averaged or collapsed)")
    for name, a in g3["admission"].items():
        if a.get("admitted") == "True":
            fails.append(f"G3 admission: invalid table '{name}' was admitted")
    mo = g3["marginals_only"]
    if mo.get("identified") != "False" or mo.get("bounds") != ("4/5", "9/10"):
        fails.append(f"G3: marginals only must give bounds [0.8, 0.9], not {mo}")
    ref = g3["engine_refusal_rho_1_2"]
    if ref["analysis_status"] == "computed":
        fails.append("G3: the engine computed with rho 1.2 instead of refusing it")
    return fails


def g4_row(engine, mu, faults, calls, seed, n, tag="", **kw):
    body = build_g4_request(mu, seed=seed, n_samples=n, **kw)
    if "reorder_options" in faults:
        body["options"] = list(reversed(body["options"]))
    if "unsupported_condition" in faults:
        body["goal_constraints"][0]["operator"] = "between"
    timeout = 0.001 if "timeout" in faults else 60
    res = engine.evaluate(body, f"g4:{mu}{tag}", timeout_s=timeout, use_cache="timeout" not in faults)
    _log(calls, f"g4 mu={float(mu):g}{tag}", res, faults)
    o = next((x for x in res["plot"].get("option_comparison") or [] if x.get("option_id") == "o1"), {})
    p = (o.get("constraint_probabilities") or {}).get("Y_target")
    nn = ((o.get("constraint_probabilities_precision") or {}).get("Y_target") or {}).get("n_informative") or 0
    exact = F.normal_goal_probability(Q(mu))
    status = "supported" if p is not None and nn else ("failed" if res["http_status"] in (None, 500, 502, 503, 504) or res["plot"].get("analysis_status") == "failed" else "unsupported")
    return {"mu": float(mu), "exact": exact, "engine_p": p, "n": nn, "status": status,
            "reason": None if status == "supported" else (res["plot"].get("status_reason") or str([c.get("code") for c in res["plot"].get("critiques", [])][:4]))[:200],
            "within_tolerance": None if p is None else abs(p - exact) <= _tol(exact, nn)}


def run_g4(engine, faults, calls, seed, n):
    rows = [g4_row(engine, mu, faults, calls, seed, n) for mu in F.G4_MUS]
    return {"rows": rows}


def g4_failures(g4):
    fails = []
    for r in g4["rows"]:
        if r["status"] != "supported":
            fails.append(f"G4 mu={r['mu']:g}: {r['status']} ({r['reason']})")
        elif not r["within_tolerance"]:
            fails.append(f"G4 mu={r['mu']:g}: engine {r['engine_p']:.4f} vs exact {r['exact']:.10f}")
    return fails


def g4_boundary_probes(engine, calls, seed, n):
    """Reported findings, not gates: the headline Channel A on a floored goal path, and the exact-sum route."""
    out = {}
    # Channel A: goal fed by the uncertain driver through a (floored) edge; options shift the mean by 0/+10/+20.
    body = {"request_id": "track3-g4:channelA-floored", "graph": {"nodes": [
              {"id": "y", "kind": "goal", "label": "Outcome Y", "intercept": 0, "goal_threshold": float(F.G4_TARGET / SCALE),
               "goal_threshold_frame": "delta", "goal_threshold_cap": float(SCALE)},
              {"id": "y_level", "kind": "factor", "label": "Outcome driver", "category": "observable",
               "observed_state": {"value": float(Q(90) / SCALE), "std": float(F.G4_SD / SCALE)}},
              {"id": "lever", "kind": "factor", "label": "Lever", "observed_state": {"value": 0, "std": 1e-4}}],
            "edges": [{"from": "y_level", "to": "y", "exists_probability": 1, "strength": {"mean": 1, "std": 0}},
                      {"from": "lever", "to": "y", "exists_probability": 1, "strength": {"mean": 1, "std": 0}}]},
            "options": [{"id": f"mu{90 + c}", "label": f"mu {90 + c}", "interventions": {"lever": c / float(SCALE)}} for c in (0, 10, 20)],
            "goal_node_id": "y", "goal_threshold": float(F.G4_TARGET / SCALE), "goal_direction": "maximise", "seed": seed, "n_samples": n}
    res = engine.evaluate(body, "g4:channelA"); _log(calls, "g4 channel A floored", res, frozenset())
    out["channel_a_floored"] = [{"mu": int(o["option_id"][2:]), "exact": F.normal_goal_probability(Q(int(o["option_id"][2:]))),
                                 "engine_p": o.get("probability_of_goal"), "engine_sd": (o.get("outcome") or {}).get("std", 0) * float(SCALE),
                                 "declared_sd": float(F.G4_SD)} for o in res["plot"].get("option_comparison") or []]
    # Exact-sum route, zero spread: determinate levels and margins (the brief's zero-spread counterpart).
    zs = build_sum_route_request("M0", 0, 0, seed=seed, n_samples=n, today=Q(90))
    zs["request_id"] = "track3-g4:sum-zero-spread"
    zs["graph"]["nodes"][0]["goal_threshold"] = zs["goal_threshold"] = float(F.G4_TARGET / SCALE)
    zs["graph"]["nodes"][0]["goal_threshold_raw"] = float(F.G4_TARGET)
    zs["options"] = [{"id": f"mu{90 + c}", "label": f"mu {90 + c}", "interventions": {"net_contribution__change": c / float(SCALE)}} for c in (0, 10, 20)]
    res = engine.evaluate(zs, "g4:sum-zero"); _log(calls, "g4 exact-sum zero spread", res, frozenset())
    out["sum_zero_spread"] = [{"mu": int(o["option_id"][2:]), "level": (o.get("outcome") or {}).get("mean", 0) * float(SCALE),
                               "sd": (o.get("outcome") or {}).get("std"), "p_goal": o.get("probability_of_goal")}
                              for o in res["plot"].get("option_comparison") or []]
    # Exact-sum route with a DECLARED Normal(90, 10^2) on today: does the spread reach the result?
    ds = json.loads(json.dumps(zs)); ds["request_id"] = "track3-g4:sum-declared-spread"
    for node in ds["graph"]["nodes"]:
        if node["id"] in ("net_contribution", "net_contribution_today"):
            node["observed_state"]["std"] = float(F.G4_SD / SCALE)
    res = engine.evaluate(ds, "g4:sum-spread"); _log(calls, "g4 exact-sum declared spread", res, frozenset())
    sent = [p for x in res["isl_exchanges"] for p in x["request"].get("parameter_uncertainties", []) if p["node_id"] == "net_contribution_today"]
    out["sum_declared_spread"] = {"spread_sent_to_isl": sent, "warnings": [w.get("code") for w in res["plot"].get("inference_warnings", [])],
                                  "rows": [{"mu": int(o["option_id"][2:]), "sd": (o.get("outcome") or {}).get("std"), "p_goal": o.get("probability_of_goal"),
                                            "exact_if_drawn": F.normal_goal_probability(Q(int(o["option_id"][2:])))}
                                           for o in res["plot"].get("option_comparison") or []]}
    return out



G3_CONTROLS = [("multiply_marginals", "multiply the marginals"), ("independent_in_shared", "draw S1/S2 independently in the shared case"),
               ("accept_invalid_table", "accept an invalid mass table"), ("infer_independence", "infer independence from missing dependence"),
               ("average_dependence", "average the three dependence hypotheses")]


def m2_controls(engine, g4_clean, calls, seed, n):
    out = []
    for fault, meaning in G3_CONTROLS:
        fails = g3_failures(run_g3(engine, frozenset({fault}), calls, seed, n))
        out.append({"fault": fault, "group": "G3", "meaning": meaning, "detected": bool(fails), "first_failures": fails[:3]})
    base = {r["mu"]: r["engine_p"] for r in g4_clean["rows"]}
    for fault, meaning, kw in [("reorder_options", "reorder options", {}), ("lineage_only", "add only lineage metadata", {"label_suffix": " [lineage: track3]"}),
                               ("unit_reexpression", "re-express £000 as £ with matching cap and target", {"unit": "£"})]:
        f = frozenset({fault}) if fault == "reorder_options" else frozenset()
        rows = [g4_row(engine, mu, f, calls, seed, n, tag=":" + fault, **kw) for mu in F.G4_MUS]
        moved = [f"mu={r['mu']:g}: {r['engine_p']} vs {base.get(r['mu'])}" for r in rows if r["engine_p"] != base.get(r["mu"])]
        fails = g4_failures({"rows": rows}) + moved
        out.append({"fault": fault, "group": "G4", "meaning": meaning + " (semantic result must not move)", "expected_detected": False,
                    "detected": bool(fails), "first_failures": fails[:3]})
    r = g4_row(engine, Q(95), frozenset(), calls, seed, n, tag=":real_change")
    fails = [] if (r["status"] == "supported" and r["within_tolerance"] and r["engine_p"] != base.get(90.0)) else [f"mu=95 row {r}"]
    out.append({"fault": "real_change", "group": "G4", "meaning": "change one real input (mu 90 -> 95): result must move to exact 0.3085",
                "expected_detected": False, "detected": bool(fails), "first_failures": fails[:3], "row": r})
    for fault, meaning, want in [("unsupported_condition", "pass an unsupported condition (operator 'between')", "unsupported"),
                                 ("timeout", "a timed-out evaluator", "failed")]:
        rows = [g4_row(engine, F.G4_MUS[0], frozenset({fault}), calls, seed, n, tag=":" + fault)]
        visible = all(x["status"] == want and x["engine_p"] is None and x["reason"] for x in rows)
        out.append({"fault": fault, "group": "G4", "meaning": meaning + f": row must be visibly '{want}', never a number",
                    "expected_detected": True, "detected": bool(g4_failures({"rows": rows})) and visible,
                    "first_failures": g4_failures({"rows": rows})[:1]})
    return out
