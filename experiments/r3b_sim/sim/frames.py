"""User-unit levels (R3-8 frame reader: raw_value, else value x cap, else value x scale_frame).

A level with no frame is WITHHELD (``FRAME_MISSING``), never approximated. Unit strings are
mapped to canonical dimensions through an explicit table; an unknown unit is ``None`` and a
natural effect touching it is treated as not unit-checked (``UNIT_MISMATCH``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .corpus import USER_SOURCES, Graph

# Canonical dimensions. Every corpus unit string must appear here or be reported as unknown.
_UNIT_TABLE: dict[str, str] = {
    "gbp per month": "GBP/month",
    "gbp/month": "GBP/month",
    "gbp/year": "GBP/year",
    "gbp/year per engineer": "GBP/year/engineer",
    "gbp": "GBP",
    "gbp over 6 months": "GBP",
    "gbp over six months": "GBP",
    "%": "percent",
    "percent": "percent",
    "percent per month": "percent",
    "% monthly churn": "percent",
    "percentage points": "percent",
    "%/month": "percent/month",
    "score out of 100": "score100",
    "subscribers": "subscribers",
    "customers": "subscribers",
    "subscribers per month": "subscribers/month",
    "new pro subscribers per month": "subscribers/month",
    "off/on": "switch",
    "binary": "switch",
    "switch": "switch",
    "share of pro subscribers": "switch",
    "engineers": "engineers",
    "engineers hired (change from today)": "engineers",
    "senior-engineer-equivalent fte": "FTE",
    "platform delivery units/month": "delivery_units/month",
}

# Dimensions whose normalised value IS the user value (0/1 switches and 0-1 shares).
NATIVE_DIMENSIONS = frozenset({"switch"})

_REL_TOL = 1e-6


def canon_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return _UNIT_TABLE.get(unit.strip().lower())


def node_unit(graph: Graph, node_id: str) -> str | None:
    node = graph.nodes[node_id]
    os_ = node.get("observed_state") or {}
    unit = os_.get("unit") or node.get("goal_threshold_unit")
    return canon_unit(unit) if isinstance(unit, str) else None


def source_tier(source: str | None) -> int:
    """0 = user/brief-stated; 1 = Olumi estimate (cee_inference, cee_hypothesis or unknown)."""
    return 0 if source in USER_SOURCES else 1


@dataclass(frozen=True)
class Level:
    value: float | None
    source: str | None
    tier: int
    frame: str
    gaps: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.value is not None


def _frame(graph: Graph, node_id: str) -> tuple[str, float | None]:
    node = graph.nodes[node_id]
    os_ = node.get("observed_state") or {}
    cap = os_.get("cap")
    if isinstance(cap, (int, float)) and cap:
        return "cap", float(cap)
    sf = node.get("scale_frame")
    if isinstance(sf, (int, float)) and sf:
        return "scale_frame", float(sf)
    if node_unit(graph, node_id) in NATIVE_DIMENSIONS:
        return "native", 1.0
    return "none", None


def held_level(graph: Graph, node_id: str) -> Level | None:
    """Today's held level in user units, or None when the node holds no observed_state."""
    os_ = graph.nodes[node_id].get("observed_state")
    if not isinstance(os_, dict):
        return None
    source = os_.get("source")
    tier = source_tier(source)
    frame, factor = _frame(graph, node_id)
    raw = os_.get("raw_value")
    norm = os_.get("value")
    if isinstance(raw, (int, float)):
        if factor is not None and isinstance(norm, (int, float)):
            implied = float(norm) * factor
            if abs(implied - float(raw)) > _REL_TOL * max(1.0, abs(float(raw))):
                return Level(None, source, tier, frame, ("FRAME_INCONSISTENT",))
        return Level(float(raw), source, tier, "raw_value", ())
    if factor is not None and isinstance(norm, (int, float)):
        return Level(float(norm) * factor, source, tier, frame, ())
    return Level(None, source, tier, frame, ("FRAME_MISSING",))


def intervention_level(graph: Graph, node_id: str, spec: dict[str, Any]) -> Level:
    """An option's typed intervention in the target's own frame (never another node's)."""
    source = spec.get("source")
    tier = source_tier(source)
    value = spec.get("value")
    frame, factor = _frame(graph, node_id)
    if factor is None or not isinstance(value, (int, float)):
        return Level(None, source, tier, frame, ("FRAME_MISSING",))
    return Level(float(value) * factor, source, tier, frame, ())
