"""Mechanical edge classification. No judgement lives here: an edge is quantified in user units
only if it carries a typed ``natural_effect`` whose units match both of its endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .corpus import USER_SOURCES, Edge, Graph
from .frames import canon_unit, node_unit


@dataclass(frozen=True)
class EdgeClass:
    key: str
    cls: str
    evidence: str
    tier: int | None
    gaps: tuple[str, ...]
    ne_amount: float | None
    ne_per: float | None
    units_fit: bool
    magnitude: str | None
    sd_over_mean: float | None


def definitional_edges(graph: Graph, spec: dict[str, Any]) -> set[str]:
    """Operand -> target edges of ADMITTED declared identities (a definition, never sampled)."""
    keys: set[str] = set()
    for target, decl in spec["declared_identities"].items():
        if decl["status"] != "admitted":
            continue
        ident = graph.identity(target)
        assert ident is not None
        for op in ident["factor_ids"]:
            keys.add(f"{op}->{target}")
    return keys


def classify_edge(graph: Graph, spec: dict[str, Any], edge: Edge) -> EdgeClass:
    ratio = abs(edge.strength_std / edge.strength_mean) if edge.strength_mean else None
    if edge.key in definitional_edges(graph, spec):
        return EdgeClass(
            edge.key,
            "algebraic_identity",
            "declared_identity (operand edge; definitional, never sampled)",
            None,
            (),
            None,
            None,
            True,
            edge.magnitude,
            ratio,
        )
    ne = edge.natural_effect
    if ne is None:
        return EdgeClass(
            edge.key,
            "unsupported_or_ambiguous",
            "range-normalised strength only",
            None,
            ("STATIC_COEFFICIENT_NO_TEMPORAL_MEANING",),
            None,
            None,
            False,
            edge.magnitude,
            ratio,
        )
    dst_u = node_unit(graph, edge.dst)
    src_u = node_unit(graph, edge.src)
    fit = (
        canon_unit(ne.get("amount_unit")) is not None
        and canon_unit(ne.get("amount_unit")) == dst_u
        and canon_unit(ne.get("per_source_change_unit")) == src_u
    )
    gaps: list[str] = ["EFFECT_TIMING_UNSPECIFIED"]
    if not fit:
        gaps.append("UNIT_MISMATCH")
    tier = 0 if edge.source in USER_SOURCES else 1
    return EdgeClass(
        edge.key,
        "behavioural_parameter",
        "typed natural_effect",
        tier,
        tuple(gaps),
        float(ne["amount"]),
        float(ne["per_source_change"]),
        fit,
        edge.magnitude,
        ratio,
    )
