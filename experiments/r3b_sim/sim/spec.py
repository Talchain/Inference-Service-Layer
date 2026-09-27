"""Hand-audited per-graph mapping specs: loading and strict validation.

The engine takes no defaults: a missing spec key raises. Values are never re-typed in a spec;
every evidence pointer is resolved against the corpus graph and every brief quote must be a
verbatim substring of the brief.
"""

from __future__ import annotations

import json
from typing import Any

from .corpus import MAPPING_DIR, Graph

CLASSES = frozenset(
    {
        "stock",
        "flow",
        "algebraic_identity",
        "behavioural_parameter",
        "exogenous",
        "unsupported_or_ambiguous",
    }
)
EVIDENCE_KINDS = frozenset(
    {"brief", "typed_field", "declared_identity", "accounting_identity", "prototype_assumption"}
)
TIERS = {"T0": 0, "T1": 1, "T2": 2, "X": 3}
TIER_NAMES = {v: k for k, v in TIERS.items()}
DEFAULTS = frozenset(
    {
        "level_persistence",
        "rate_persistence",
        "onset_month_0",
        "no_lag",
        "linear_scaling",
        "template_uncertainty",
    }
)
GAP_CODES = frozenset(
    {
        "HORIZON_AMBIGUOUS",
        "GOAL_NOT_QUANTIFIED",
        "GOAL_TEMPORAL_SEMANTICS_AMBIGUOUS",
        "BASELINE_MISSING",
        "STOCK_LEVEL_MISSING",
        "FLOW_NET_VS_GROSS_AMBIGUOUS",
        "INFLOW_MISSING",
        "STATIC_COEFFICIENT_NO_TEMPORAL_MEANING",
        "UNQUANTIFIED_NODE",
        "IDENTITY_INCONSISTENT_WITH_HELD_LEVEL",
        "IDENTITY_INCOMPLETE",
        "SUM_CARRIER_ABSENT",
        "EFFECT_TIMING_UNSPECIFIED",
        "ONSET_UNSPECIFIED",
        "RATE_PERSISTENCE_UNSPECIFIED",
        "OPTION_LEVELS_MISSING",
        "OPTION_TARGET_INVALID",
        "FRAME_MISSING",
        "FRAME_INCONSISTENT",
        "UNIT_MISMATCH",
        "HORIZON_BAKED_INTO_NODE",
        "UNCERTAINTY_NOT_SPECIFIED",
        "INTERNAL_INCONSISTENCY",
    }
)
_TOP_KEYS = (
    "schema",
    "graph",
    "journey",
    "horizon",
    "goal",
    "quantities",
    "declared_identities",
    "derived_identities",
    "goal_model_t2",
    "x_effects",
    "dynamic_models",
    "brief_decision_pair",
    "graph_gaps",
    "notes",
)
_FLOW_FORMS = frozenset({"level", "rate_x_stock", "rate_delta_x_stock", "derived_gross"})


class SpecError(ValueError):
    pass


def load_spec(graph_id: str) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((MAPPING_DIR / f"{graph_id}.json").read_text())
    return spec


def _check_evidence(graph: Graph, ev: dict[str, Any], where: str, errors: list[str]) -> None:
    kind = ev.get("kind")
    if kind not in EVIDENCE_KINDS:
        errors.append(f"{where}: unknown evidence kind {kind!r}")
        return
    if kind == "brief":
        quote = ev.get("quote")
        if not isinstance(quote, str) or quote not in graph.brief:
            errors.append(f"{where}: brief quote not verbatim: {quote!r}")
    elif kind in ("typed_field", "declared_identity"):
        try:
            graph.resolve(str(ev.get("ptr")))
        except KeyError:
            errors.append(f"{where}: pointer does not resolve: {ev.get('ptr')!r}")
    elif kind == "accounting_identity":
        if not ev.get("basis"):
            errors.append(f"{where}: accounting_identity needs a basis")


def _check_gaps(gaps: Any, where: str, errors: list[str]) -> None:
    if not isinstance(gaps, list):
        errors.append(f"{where}: gaps must be a list")
        return
    for g in gaps:
        if g not in GAP_CODES:
            errors.append(f"{where}: unknown GAP code {g!r}")


def validate(spec: dict[str, Any], graph: Graph) -> list[str]:
    """Every problem found; an empty list means the spec is admissible."""
    errors: list[str] = []
    for key in _TOP_KEYS:
        if key not in spec:
            errors.append(f"missing top-level key {key!r}")
    if errors:
        return errors
    if spec["graph"] != graph.id or spec["journey"] != graph.journey:
        errors.append("graph id / journey mismatch")

    hz = spec["horizon"]
    _check_evidence(graph, hz["evidence"], "horizon", errors)
    _check_gaps(hz["gaps"], "horizon", errors)
    if hz["status"] == "resolved" and not isinstance(hz["months"], int):
        errors.append("horizon: resolved without integer months")
    if hz["status"] not in ("resolved", "HORIZON_AMBIGUOUS"):
        errors.append(f"horizon: bad status {hz['status']!r}")

    goal = spec["goal"]
    if goal["node"] not in graph.nodes or graph.kind(goal["node"]) != "goal":
        errors.append("goal: node is not the graph's goal")
    if goal["threshold_ptr"] is not None:
        try:
            float(graph.resolve(goal["threshold_ptr"]))
        except (KeyError, TypeError, ValueError):
            errors.append("goal: threshold pointer does not resolve to a number")
    if goal["temporal_semantics"] not in ("attain_by_H", "withheld"):
        errors.append("goal: bad temporal_semantics")
    _check_evidence(graph, goal["evidence"], "goal", errors)
    _check_gaps(goal["gaps"], "goal", errors)

    quantities = spec["quantities"]
    expected = set(graph.quantity_ids)
    if set(quantities) != expected:
        errors.append(
            f"quantities must cover every non-structural node exactly: "
            f"missing={sorted(expected - set(quantities))} extra={sorted(set(quantities) - expected)}"
        )
    for qid, q in quantities.items():
        for key in ("class", "evidence", "gaps", "note"):
            if key not in q:
                errors.append(f"quantities.{qid}: missing {key!r}")
        if q.get("class") not in CLASSES:
            errors.append(f"quantities.{qid}: bad class {q.get('class')!r}")
        for i, ev in enumerate(q.get("evidence", [])):
            _check_evidence(graph, ev, f"quantities.{qid}.evidence[{i}]", errors)
        _check_gaps(q.get("gaps"), f"quantities.{qid}", errors)

    if set(spec["declared_identities"]) != set(graph.identity_targets):
        errors.append("declared_identities must list exactly the graph's nonlinear_identity nodes")
    for tid, d in spec["declared_identities"].items():
        if d.get("status") not in ("admitted", "withheld"):
            errors.append(f"declared_identities.{tid}: bad status")
        _check_gaps(d.get("gaps"), f"declared_identities.{tid}", errors)

    for d in spec["derived_identities"]:
        if d.get("tier") != "X":
            errors.append(f"derived identity {d.get('id')}: must be tier X (no sum carrier)")
        if d.get("operation") != "sum":
            errors.append(f"derived identity {d.get('id')}: only 'sum' is modelled")
        for op in [d.get("target"), *d.get("operands", [])]:
            if op not in graph.nodes:
                errors.append(f"derived identity {d.get('id')}: unknown node {op!r}")
        if not isinstance(d.get("identity_overrides_held"), bool):
            errors.append(f"derived identity {d.get('id')}: identity_overrides_held required")
        _check_gaps(d.get("gaps"), f"derived identity {d.get('id')}", errors)

    for xe in spec["x_effects"]:
        path = xe.get("path", [])
        if len(path) < 2:
            errors.append(f"x_effect {xe.get('id')}: path too short")
        for a, b in zip(path, path[1:]):
            try:
                graph.edge(f"{a}->{b}")
            except KeyError:
                errors.append(f"x_effect {xe.get('id')}: no edge {a}->{b}")
        if not xe.get("sweep") or len(xe.get("basis", [])) != len(xe["sweep"]):
            errors.append(f"x_effect {xe.get('id')}: every sweep value needs a basis")

    model_ids = set()
    for m in spec["dynamic_models"]:
        model_ids.add(m.get("id"))
        if m.get("tier") not in ("T2", "X"):
            errors.append(f"model {m.get('id')}: tier must be T2 or X")
        if m.get("type") not in ("stock_flow", "static_at_H"):
            errors.append(f"model {m.get('id')}: bad type")
        for dflt in m.get("defaults", []):
            if dflt not in DEFAULTS:
                errors.append(f"model {m.get('id')}: unknown default {dflt!r}")
        if m.get("tier") == "T2" and m.get("assumptions"):
            errors.append(f"model {m.get('id')}: T2 models may not carry prototype assumptions")
        for ek in m.get("replaces_edges", []):
            try:
                graph.edge(ek)
            except KeyError:
                errors.append(f"model {m.get('id')}: replaces unknown edge {ek}")
        if m.get("output") not in graph.nodes:
            errors.append(f"model {m.get('id')}: unknown output")
        if m.get("type") == "stock_flow":
            for sid, s in m.get("stocks", {}).items():
                if sid not in graph.nodes:
                    errors.append(f"model {m.get('id')}: unknown stock {sid}")
                if s.get("init") not in ("held", "inversion"):
                    errors.append(f"model {m.get('id')}: bad stock init")
                if s.get("init") == "inversion" and m.get("tier") != "X":
                    errors.append(f"model {m.get('id')}: identity inversion is Mode X only")
                if s.get("option_delta_mode") not in ("jump", "overlay"):
                    errors.append(f"model {m.get('id')}: option_delta_mode required")
            for f in m.get("flows", []):
                if f.get("form") not in _FLOW_FORMS:
                    errors.append(f"model {m.get('id')}: bad flow form {f.get('form')!r}")
                if f.get("stock") not in m.get("stocks", {}):
                    errors.append(f"model {m.get('id')}: flow into undeclared stock")
            for a in m.get("algebraic", []):
                if a.get("source") == "derived" and m.get("tier") != "X":
                    errors.append(f"model {m.get('id')}: derived identity outside Mode X")

    t2 = spec["goal_model_t2"]
    if t2.get("status") == "model" and t2.get("model") not in model_ids:
        errors.append("goal_model_t2 names an unknown model")
    _check_gaps(t2.get("gaps"), "goal_model_t2", errors)

    pair = spec["brief_decision_pair"]
    if pair is not None and (len(pair) != 2 or any(o not in graph.option_ids for o in pair)):
        errors.append("brief_decision_pair must name two options")
    _check_gaps(spec["graph_gaps"], "graph_gaps", errors)
    return errors


def load_valid_spec(graph: Graph) -> dict[str, Any]:
    spec = load_spec(graph.id)
    errors = validate(spec, graph)
    if errors:
        raise SpecError(f"{graph.id}: " + "; ".join(errors))
    return spec
