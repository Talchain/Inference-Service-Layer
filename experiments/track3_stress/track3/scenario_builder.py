"""Turn (hypothesis, stress setting) into one validated temporary PLoT /v2/run request.

Representation (recorded in every change record):
* PLoT's supported scale is normalised; the goal's ruler is `goal_threshold_cap`
  (user units = normalised x cap). SCALE = £200k per unit, so the target 90 is 0.45.
* Options are interventions on controllable characteristics: planned uplift over a
  common base, demand-stress exposure, cost-stress exposure.
* Stresses x, y are scenario PARAMETER ASSUMPTIONS, not actions: the per-unit-exposure
  loss (edge strength) is -x and -y. They are never interventions.
* M1 adds a named path: growth_commitment (set by Growth-first) -> retention_loss ->
  net_contribution. The outcome is never written into the graph as a number.
* Lever values are kept small and edge strengths large (|s| <= 1) because PLoT floors
  every causal edge spread at min(0.05, max(0.001, |mean|/2)); the floor's effect is
  measured and reported, never hidden.
"""
from fractions import Fraction as Q

from .fixtures import HORIZON_MONTHS, HYPOTHESES, OPTIONS, TARGET, TARGET_OPERATOR, UNIT, scenario_id

SCALE = Q(200)                      # £000 per normalised unit (= goal_threshold_cap)
COMMON_BASE = min(base for _, base, _, _ in OPTIONS.values())
LEVER_STD = 1e-4                    # explicit, so PLoT does not synthesise a status-quo spread
GOAL = "net_contribution"
BUILDER_FAULTS = {"omit_cost_stress", "reverse_demand_sign", "remove_m1_path"}


def _n(value):
    return float(Q(value) / SCALE)


def build_request(hypothesis, x, y, *, seed, n_samples, faults=frozenset(), strict=False):
    x, y = Q(x), Q(y)
    lever = lambda i, label: {"id": i, "kind": "factor", "label": label,
                              "observed_state": {"value": 0, "std": LEVER_STD}}
    edge = lambda f, t, m: {"id": f"e_{f}", "from": f, "to": t, "exists_probability": 1.0,
                            "strength": {"mean": float(m), "std": 0}}
    nodes = [
        {"id": GOAL, "kind": "goal", "label": f"{HORIZON_MONTHS}-month net contribution ({UNIT})",
         "intercept": _n(COMMON_BASE), "goal_threshold": _n(TARGET),
         "goal_threshold_frame": "delta", "goal_threshold_cap": float(SCALE)},
        lever("planned_uplift", "Planned uplift over the common base"),
        lever("demand_exposure", "Demand-stress exposure"),
        lever("cost_exposure", "Cost-stress exposure"),
    ]
    demand_strength = x if "reverse_demand_sign" in faults else -x
    edges = [edge("planned_uplift", GOAL, 1), edge("demand_exposure", GOAL, demand_strength)]
    if "omit_cost_stress" not in faults:
        edges.append(edge("cost_exposure", GOAL, -y))
    options = []
    for oid, (label, base, dx, dy) in OPTIONS.items():
        options.append({"id": oid, "label": label, "interventions": {
            "planned_uplift": _n(base - COMMON_BASE), "demand_exposure": _n(dx), "cost_exposure": _n(dy)}})
    retention = HYPOTHESES[hypothesis]["retention_loss"]
    if retention and "remove_m1_path" not in faults:
        nodes += [lever("growth_commitment", "Growth-first commitment"),
                  {"id": "retention_loss", "kind": "factor", "label": "Persistent retention loss"}]
        edges += [edge("growth_commitment", "retention_loss", 1), edge("retention_loss", GOAL, -1)]
        for option in options:
            option["interventions"]["growth_commitment"] = _n(retention.get(option["id"], 0))
    sid = scenario_id(hypothesis, x, y)
    body = {"request_id": f"track3:{sid}", "graph": {"nodes": nodes, "edges": edges}, "options": options,
            "goal_node_id": GOAL, "goal_threshold": _n(TARGET), "goal_direction": "maximise",
            "seed": seed, "n_samples": n_samples}
    if strict or TARGET_OPERATOR == ">":
        body["goal_threshold_strict"] = True
    change_record = [
        {"element": "goal", "change": f"target {TARGET_OPERATOR} {TARGET} {UNIT} as {float(TARGET / SCALE)} on cap {SCALE}",
         "origin": "brief v1.0 §6"},
        {"element": "e_demand_exposure", "change": f"strength {float(demand_strength)} (demand stress x={float(x):g})",
         "kind": "scenario parameter assumption", "origin": "synthetic stress coordinate"},
        {"element": "e_cost_exposure",
         "change": "OMITTED" if "omit_cost_stress" in faults else f"strength {float(-y)} (cost stress y={float(y):g})",
         "kind": "scenario parameter assumption", "origin": "synthetic stress coordinate"},
        {"element": "hypothesis", "change": hypothesis + (" WITHOUT its path" if "remove_m1_path" in faults and retention else ""),
         "origin": HYPOTHESES[hypothesis].get("origin", "brief v1.0 §6 base hypothesis")},
    ]
    return body, change_record


def _q(v):
    return Q(repr(float(v)))          # the wire decimal, exactly


def graph_level(body, option_id):
    """Exact linear read of the BUILT request (no engine): an independent check that the
    graph encodes the fixture. Returns the option's goal level in £000 as a Fraction."""
    nodes = {n["id"]: n for n in body["graph"]["nodes"]}
    parents = {}
    for e in body["graph"]["edges"]:
        parents.setdefault(e["to"], []).append((e["from"], _q(e["strength"]["mean"])))
    setting = {k: _q(v) for k, v in next(o for o in body["options"] if o["id"] == option_id)["interventions"].items()}

    def value(node_id):
        if node_id in setting:
            return setting[node_id]
        node = nodes[node_id]
        if node_id not in parents:
            return _q((node.get("observed_state") or {}).get("value", 0))
        return _q(node.get("intercept", 0)) + sum(s * value(p) for p, s in parents[node_id])

    return value(body["goal_node_id"]) * SCALE


def build_sum_route_request(hypothesis, x, y, *, seed, n_samples, today=Q(0)):
    """The core's exact route (CEE mapper.ts exact sum): goal = <goal>_today + <goal>__change, where each
    option SETS the change. The route admits no other parent of the goal and no negative change, so the
    stress arithmetic is done HERE, exactly, by the builder: the engine proves the exact comparison,
    not the stress propagation. Used only to test whether the UNRESOLVED (engine floor) rows resolve."""
    obs = lambda v: {"value": float(Q(v) / SCALE), "baseline": float(Q(v) / SCALE), "raw_value": float(v), "cap": float(SCALE), "unit": UNIT}
    today_id, change_id = f"{GOAL}_today", f"{GOAL}__change"
    nodes = [{"id": GOAL, "kind": "goal", "label": f"{HORIZON_MONTHS}-month net contribution ({UNIT})", "observed_state": obs(today),
              "goal_threshold": _n(TARGET), "goal_threshold_raw": float(TARGET), "goal_threshold_unit": UNIT,
              "goal_threshold_cap": float(SCALE), "goal_threshold_frame": "level",
              "nonlinear_identity": {"operation": "sum", "factor_ids": [today_id, change_id], "stated_in_brief": False}},
             {"id": today_id, "kind": "factor", "label": "Net contribution today", "category": "observable", "observed_state": obs(today)},
             {"id": change_id, "kind": "factor", "label": "Change in net contribution", "category": "controllable",
              "quantity_frame": "change", "observed_state": {"value": 0, "raw_value": 0, "cap": float(SCALE), "unit": UNIT}}]
    edges = [{"id": f"{p}__{GOAL}", "from": p, "to": GOAL, "strength": {"mean": 1, "std": 0.05}, "exists_probability": 1,
              "effect_direction": "positive"} for p in (today_id, change_id)]
    from .fixtures import oracle_level
    options = [{"id": o, "label": OPTIONS[o][0], "interventions": {change_id: _n(oracle_level(o, hypothesis, x, y) - today)}} for o in OPTIONS]
    return {"request_id": f"track3-sumroute:{scenario_id(hypothesis, Q(x), Q(y))}", "graph": {"nodes": nodes, "edges": edges},
            "options": options, "goal_node_id": GOAL, "goal_threshold": _n(TARGET), "goal_direction": "maximise",
            "seed": seed, "n_samples": n_samples}


# ── G3: each admitted law is carried as two thresholded Normal drivers joined by ISL's Gaussian copula.
# For equal marginals p, P(both) = p^2, p and 2p-1 are exactly rho = 0, +1 and -1. Any other P(both) has no
# mapping here and is returned as unsupported, never approximated.
G3_DRIVER_MEAN, G3_DRIVER_SD = 0.5, 0.1


def g3_rho_for(admitted):
    p, both = admitted["p1"], admitted["p_both"]
    if admitted["p1"] != admitted["p2"]:
        return None
    return {p * p: 0.0, p: 1.0, 2 * p - 1: -1.0}.get(both)


def build_g3_request(law, rho, *, seed, n_samples, p):
    from statistics import NormalDist
    t = G3_DRIVER_MEAN + NormalDist().inv_cdf(1 - float(p)) * G3_DRIVER_SD
    drv = lambda i: {"id": i, "kind": "factor", "label": f"Driver of target indicator {i[-1]}", "category": "observable",
                     "observed_state": {"value": G3_DRIVER_MEAN, "std": G3_DRIVER_SD}}
    edge = lambda f: {"id": f"e_{f}", "from": f, "to": "carrier", "exists_probability": 1, "strength": {"mean": 0.5, "std": 0}}
    body = {"request_id": f"track3-g3:{law}", "graph": {"nodes": [
                {"id": "carrier", "kind": "goal", "label": "Carrier outcome (not a target)", "intercept": 0},
                drv("driver_s1"), drv("driver_s2"),
                {"id": "lever", "kind": "factor", "label": "Unrelated lever", "observed_state": {"value": 0, "std": LEVER_STD}}],
            "edges": [edge("driver_s1"), edge("driver_s2"), edge("lever")]},
            "options": [{"id": "o1", "label": "Option 1", "interventions": {"lever": 0.0}},
                        {"id": "o2", "label": "Option 2", "interventions": {"lever": 0.5}}],
            "goal_node_id": "carrier", "goal_direction": "maximise", "seed": seed, "n_samples": n_samples,
            "goal_constraints": [{"constraint_id": f"S{i}", "node_id": f"driver_s{i}", "operator": ">=", "value": t,
                                  "value_frame": "delta", "label": f"Target {i}"} for i in (1, 2)]}
    if rho is not None:
        body["factor_correlations"] = [{"factor_a": "driver_s1", "factor_b": "driver_s2", "rho": rho}]
    return body


# ── G4: the declared Normal law on an uncertain root, the target as a goal constraint on its raw draws
# (Channel B). Channel A through a goal node is floored by PLoT and is measured separately as a boundary probe.
def build_g4_request(mu, *, seed, n_samples, unit=UNIT, label_suffix=""):
    """unit '£' re-expresses every figure in pounds with the matching cap (the unit-invariance control)."""
    from .fixtures import G4_SD, G4_TARGET
    k = Q(1000) if unit == "£" else Q(1)
    scale = SCALE * k
    n = lambda v: float(Q(v) * k / scale)
    return {"request_id": f"track3-g4:mu={float(mu):g}{label_suffix}", "graph": {"nodes": [
                {"id": "y_level", "kind": "factor", "label": f"Outcome Y ({unit}){label_suffix}", "category": "observable",
                 "observed_state": {"value": n(mu), "std": n(G4_SD)}},
                {"id": "carrier", "kind": "goal", "label": "Carrier outcome (not a target)", "intercept": 0},
                {"id": "lever", "kind": "factor", "label": "Unrelated lever", "observed_state": {"value": 0, "std": LEVER_STD}}],
            "edges": [{"from": "lever", "to": "carrier", "exists_probability": 1, "strength": {"mean": 0.5, "std": 0}},
                      {"from": "y_level", "to": "carrier", "exists_probability": 1, "strength": {"mean": 0.5, "std": 0}}]},
            "options": [{"id": "o1", "label": "Option 1", "interventions": {"lever": 0.0}},
                        {"id": "o2", "label": "Option 2", "interventions": {"lever": 0.5}}],
            "goal_node_id": "carrier", "goal_direction": "maximise", "seed": seed, "n_samples": n_samples,
            "goal_constraints": [{"constraint_id": "Y_target", "node_id": "y_level", "operator": ">=", "value": n(G4_TARGET),
                                  "value_frame": "delta", "label": f"Y >= {G4_TARGET * k} {unit}"}]}
