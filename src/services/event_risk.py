"""event_risk.v1: the ONE owner of event-risk occurrence semantics in ISL.

Science 393023, ``science-richness-P0-20261007.md`` ruling (a) + §4 PILOT. A risk is an EVENT that
may happen within a horizon. Three uncertainties are never merged:

- OCCURRENCE: does the event happen within the horizon. Owned here.
- mechanism EXISTENCE: each link's ``exists_probability``, drawn by ``DualUncertaintySampler``.
- EFFECT SIZE: each risk->child link's strength, read as the severity CONDITIONAL on occurrence.

Semantics. Per Monte Carlo draw and per event risk r, ONE draw from a dedicated stream:
``p ~ Uniform(p_low, p_high)``, ``u ~ Uniform(0, 1)``, ``z = u * p_mid / p``. Under an option the
risk's EXPECTED occurrence is ``L = p_mid * prod_k(1 - m_k * clamp(x_k, 0, 1))``, x_k being the
value of the k-th named preventer, and the risk OCCURS on that draw iff ``z < L``. So
``P(occurs) = E[p] * prod(1 - m x)``: Science's ``p x (1 - m)``. One z per draw is shared by every
option and by the status-quo reference (common random numbers): a mitigation can only remove an
occurrence, never create one. Each child then receives ``occurs x (drawn severity)`` through the
ordinary linear equation.

Where no draw is supplied (the central/deterministic phases), the risk takes its expected
occurrence ``clamp(L, 0, 1)``. That is exact for the linear SCM, so deterministic phases agree with
the Monte Carlo mean.

Every function is inert when no node carries ``event_risk``. Nothing is drawn, nothing is
rewritten, and no branch is taken, so a legacy request is byte-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Sequence, Tuple

from src.models.robustness_v2 import EdgeV2, GraphV2, NodeV2
from src.utils.rng import SeededRNG

# The occurrence stream is SeededRNG(factor_stream_seed + OCCURRENCE_STREAM_OFFSET). It is separate
# from the edge stream (seed), the factor stream (seed+1), noise (seed+2), epsilon (seed+3) and the
# EVPI streams (seed+100/+101), so adding an event risk moves no other draw.
OCCURRENCE_STREAM_OFFSET = 7_919


@dataclass(frozen=True)
class EventRiskPlan:
    """One event risk, resolved from its node's ``event_risk`` block."""

    node_id: str
    p_low: float
    p_high: float
    # ((preventer factor id, occurrence_reduction m), ...) in the order the block states them.
    mitigations: Tuple[Tuple[str, float], ...]

    @property
    def p_mid(self) -> float:
        return (self.p_low + self.p_high) / 2.0


def resolve_event_risk_plans(nodes: Sequence[NodeV2]) -> Dict[str, EventRiskPlan]:
    """Every event risk among ``nodes``, in node order. Empty for every legacy request."""
    plans: Dict[str, EventRiskPlan] = {}
    for node in nodes:
        block = node.event_risk
        if block is None:
            continue
        plans[node.id] = EventRiskPlan(
            node_id=node.id,
            p_low=block.occurrence.p_low,
            p_high=block.occurrence.p_high,
            mitigations=tuple(
                (m.factor_id, m.occurrence_reduction) for m in block.mitigations or []
            ),
        )
    return plans


def occurrence_p_key(node_id: str) -> str:
    """The ``factor_values`` key of a draw's p for event risk ``node_id``. It can never collide
    with a node id (node ids match ``^[a-z0-9_:-]+$``; this key carries ``@``)."""
    return f"{node_id}@p"


def draw_occurrence_state(plan: EventRiskPlan, rng: SeededRNG) -> Tuple[float, float]:
    """One draw's occurrence state for ``plan``: ``(z, p)``, two draws from ``rng`` in this order.

    - ``p ~ Uniform(p_low, p_high)``: which probability is true. EPISTEMIC: evidence could narrow
      it, so it is recorded per draw (Science Q6: the slice-3 p-EVPPI regroups draws by p).
    - ``u ~ Uniform(0, 1)``: whether the event happens. ALEATORY: only time reveals it, so it is
      never a value-of-information target.

    The risk occurs under an option iff ``z = u * p_mid / p < L`` (module docstring). ``p = 0`` (or
    ``p_mid = 0``) can never occur: ``z = +inf``."""
    p = rng.uniform(plan.p_low, plan.p_high) if plan.p_high > plan.p_low else plan.p_low
    u = rng.random()
    if p <= 0.0 or plan.p_mid <= 0.0:
        return math.inf, p
    return u * plan.p_mid / p, p


def expected_occurrence(plan: EventRiskPlan, node_values: Mapping[str, float]) -> float:
    """``L = p_mid * prod_k(1 - m_k * clamp(x_k, 0, 1))``: the option's expected occurrence.

    ``node_values`` holds each preventer's value on this evaluation (a parent of the risk, so it
    is already computed in topological order). An unset preventer reads 0 (not in place).
    Several preventers multiply: independent preventions, each cut applied separately (Science
    Q2)."""
    level = plan.p_mid
    for factor_id, reduction in plan.mitigations:
        x = min(1.0, max(0.0, node_values.get(factor_id, 0.0)))
        level *= 1.0 - reduction * x
    return min(1.0, max(0.0, level))


# How an evaluator reads an event risk on a Monte Carlo draw.
#
# - "realised": occurs (1/0) iff z < L. Every outcome the user sees (P(goal), mean, downside,
#   win share) is computed this way.
# - "p_conditional": the draw's p-conditional EXPECTATION p * prod(1 - m x) = (p / p_mid) * L.
#   Every PER-DRAW CHOICE made with perfect information reads the risk this way (Science C-EVPI):
#   expected regret and the whole-decision EVPI bound (min regret), and the factor EVPPI / EVPC
#   regressions that share that population. A choice may be credited with knowing which
#   probability is true, never with knowing whether the event happens: no research can buy that.
#   It is NEVER thresholded (Science Q4): a goal chance or a constraint probability read off a
#   conditional mean is wrong (0.50 instead of 0.90 at threshold 0.76; Codex buddy r1). So the
#   per-factor EVPI arms, which hold the policy FIXED and count goal attainment or wins, read
#   "realised".
OccurrenceMode = Literal["realised", "p_conditional"]


def occurrence_value(
    plan: EventRiskPlan,
    node_values: Mapping[str, float],
    factor_values: Mapping[str, float],
    mode: "OccurrenceMode" = "realised",
) -> float:
    """The risk node's value on one evaluation.

    With no draw in ``factor_values`` (a central/deterministic evaluation) it is the EXPECTED
    occurrence L, exact for MEAN-based phases only; no goal-threshold figure is read from it
    (Science Q4). On a Monte Carlo draw it is 1.0/0.0 ("realised") or ``(p / p_mid) * L``
    ("p_conditional")."""
    level = expected_occurrence(plan, node_values)
    z = factor_values.get(plan.node_id)
    if z is None:
        return level
    if mode == "realised":
        return 1.0 if z < level else 0.0
    p = factor_values[occurrence_p_key(plan.node_id)]
    if plan.p_mid <= 0.0:
        return 0.0
    return min(1.0, max(0.0, p / plan.p_mid * level))


def today_levels(plans: Mapping[str, EventRiskPlan]) -> Dict[str, float]:
    """Each event risk's level TODAY: 0, because the event has not happened yet.

    A level-framed figure reads ``B + (option_i - reference_i)``, where the status-quo reference
    cancels what today's level B already embodies. B never embodies a future event, so the
    reference holds every event risk at 0. Otherwise the event would cancel out of the status quo
    and the figure would read as certain. Empty for every legacy request."""
    return {node_id: 0.0 for node_id in plans}


def strip_occurrence_state(
    factor_values_per_sample: List[Dict[str, float]], plans: Mapping[str, EventRiskPlan]
) -> List[Dict[str, float]]:
    """The per-draw factor values without the occurrence state (z and p). z and p are internal
    draw state, not quantities a reader may attribute a goal chance to; a z "driver" would
    describe a uniform draw, not the event (Codex buddy r1). Returns the SAME list when no
    event risk exists (legacy requests are untouched)."""
    if not plans:
        return factor_values_per_sample
    internal = set(plans) | {occurrence_p_key(node_id) for node_id in plans}
    return [
        {key: value for key, value in values.items() if key not in internal}
        for values in factor_values_per_sample
    ]


def mitigation_edges(graph: GraphV2) -> set:
    """Each event risk's preventer -> risk links as ``(from, to)``. They are DEFINITIONAL: the
    evaluator computes the risk from the stated reduction, never from the link's sampled
    strength, so every sampler holds them and no edge-level output lists them (as R3-9 does for
    an identity's operands)."""
    return {
        (factor_id, plan.node_id)
        for plan in resolve_event_risk_plans(graph.nodes).values()
        for factor_id, _ in plan.mitigations
    }


def resolve_event_risk_graph(graph: GraphV2) -> GraphV2:
    """The graph every analysis phase reads. Returns ``graph`` itself (same object) when there is
    no event risk.

    Each preventer -> risk link is rewritten to its exact linear coefficient
    ``dL/dx = -p_mid * m`` with existence 1.0. Phases that read only coefficients (path
    decomposition, structural influence) then agree with the evaluator. The producer's strength
    on that link is ignored by contract: the reduction m on the risk is the one source of truth.
    The link's std is kept; it is never drawn (held by ``mitigation_edges``)."""
    plans = resolve_event_risk_plans(graph.nodes)
    if not plans:
        return graph
    coefficient: Dict[Tuple[str, str], float] = {
        (factor_id, plan.node_id): -plan.p_mid * reduction
        for plan in plans.values()
        for factor_id, reduction in plan.mitigations
    }
    edges: List[EdgeV2] = []
    for edge in graph.edges:
        key = (edge.from_, edge.to)
        if key not in coefficient:
            edges.append(edge)
            continue
        strength = edge.strength.model_copy(update={"mean": coefficient[key]})
        edges.append(edge.model_copy(update={"strength": strength, "exists_probability": 1.0}))
    return graph.model_copy(update={"edges": edges})


__all__ = [
    "OCCURRENCE_STREAM_OFFSET",
    "EventRiskPlan",
    "draw_occurrence_state",
    "expected_occurrence",
    "mitigation_edges",
    "OccurrenceMode",
    "occurrence_p_key",
    "occurrence_value",
    "today_levels",
    "resolve_event_risk_graph",
    "resolve_event_risk_plans",
    "strip_occurrence_state",
]
