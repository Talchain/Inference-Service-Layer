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
