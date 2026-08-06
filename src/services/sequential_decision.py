"""
Sequential Decision Support Service for Phase 4.

Implements backward induction for multi-stage decision problems,
computing optimal policies and value of flexibility.
"""

import itertools
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.models.requests import (
    DecisionStage,
    SequentialAnalysisRequest,
    SequentialGraph,
    SequentialGraphEdge,
    SequentialGraphNode,
)
from src.models.responses import (
    ConditionalAction,
    DecisionRule,
    ExplanationMetadata,
    Policy,
    SequentialAnalysisResponse,
    StageAnalysis,
    StageOption,
    StagePolicy,
)

logger = logging.getLogger(__name__)


def _discounted_edge_value(immediate: float, discount_factor: float, child_value: float) -> float:
    """Value of traversing an edge: its immediate payoff plus the discounted
    continuation value of the child it points to.

    This is the single edge-valuation convention used throughout the engine --
    the backward-induction decision (max) and chance (expectation) branches and
    the policy's conditional-action values all read an edge's worth the same way.
    Keeping the formula in one place stops any consumer silently reporting
    continuation-only and dropping the edge's immediate payoff (RW-6b).
    """
    return immediate + discount_factor * child_value


def _edge_probability(edge: Dict[str, Any], num_siblings: int) -> float:
    """Transition probability for a chance edge, defaulting an UNSPECIFIED
    probability to an equal split across the node's outgoing edges.

    The request model's `probability` field is Optional with default None, and
    `_build_graph_data` always writes the key -- so `edge.get("probability",
    1/len)` returned None (key present, F-3), and `None * value` raised TypeError
    while the equal-split fallback was dead code. Treat None as 'unspecified' here,
    at every probability read, so omitting a probability means an equal split.
    """
    prob = edge.get("probability")
    if prob is None:
        return 1.0 / num_siblings
    return float(prob)


def _edge_immediate_payoff(edge: Dict[str, Any]) -> float:
    """Immediate payoff of traversing an edge, defaulting a missing or falsy
    value to 0.

    Single convention for the immediate-payoff read off a graph_data edge dict:
    `edge.get("immediate_payoff", 0) or 0` -- a missing key OR a None/0 stored
    value both read as 0. `_build_graph_data` always writes the key (as
    `edge.immediate_payoff or 0`), so the `.get` default is defensive; keeping
    the read in one place stops any consumer diverging on how an absent-or-None
    payoff is treated (the absent-as-0 convention this engine enforces).
    """
    return edge.get("immediate_payoff", 0) or 0


# Risk-aversion coefficient for the mean-standard-deviation adjustment applied to a
# chance node's value under risk_tolerance="averse": value = mean - k * sqrt(variance).
# DOCTRINE-PENDING(Neil): ruling D-13 fixes only the UNITS here — variance (currency^2)
# -> sqrt(variance) (sigma / currency units), symmetric with the 'seeking' branch and
# _calculate_resolved_uncertainty, both of which already use sqrt(variance). That is a
# consistency restoration, not a modeling change. The coefficient VALUE (k=0.5) is a
# risk-modeling decision reserved for Neil; do NOT read 0.5 as ratified.
RISK_AVERSION_COEFFICIENT = 0.5

# F4d (A3, 2026-07-22): tolerance for validating that a chance node's branch
# probabilities form a proper distribution (sum to 1). Loose enough to absorb
# float error in the equal-split default (e.g. 3 * (1/3) = 0.999999999999…) yet
# tight enough to reject a genuinely malformed distribution.
_PROB_SUM_TOLERANCE = 1e-6

# F-3 (A3 VOI adversarial): safety cap on the number of joint cells the per-stage
# stage_evpi decide-after leg enumerates (∏ branch-counts over the decision's chance
# children = B^K for K chance-child actions with B branches each). A legal request
# (SequentialGraph caps: ≤100 nodes / ≤300 edges) can build a decision with ~97
# chance-child actions ⇒ 2^97 cells, an un-preemptable CPU DoS on the async route.
# 4096 = 2^12 caps the enumeration at trivial cost (each cell is O(K); 4096 cells is
# well under a millisecond) while never rejecting a legitimate decision — real
# decisions face a handful of independent chance nodes (6 binary = 64 cells, even
# 12 binary = 4096); the DoS shapes need K ≥ 13. Over the cap, stage_evpi honestly
# SKIPS (null + status) rather than 422-ing the whole (valid, O(nodes)) analysis.
_STAGE_EVPI_JOINT_CELL_CAP = 4096


class SequentialDecisionEngine:
    """
    Engine for solving sequential decision problems via backward induction.

    Supports multi-stage decisions where later choices depend on earlier outcomes.
    """

    def __init__(self) -> None:
        """Initialize the sequential decision engine."""
        self.logger = logger

    def analyze(self, request: SequentialAnalysisRequest) -> SequentialAnalysisResponse:
        """
        Analyze a sequential decision problem using backward induction.

        Args:
            request: Request with graph, stages, and parameters

        Returns:
            SequentialAnalysisResponse with optimal policy and analysis
        """
        self.logger.info(f"Analyzing sequential decision with {len(request.stages)} stages")

        # Build internal representation
        graph_data = self._build_graph_data(request.graph)

        # Run backward induction
        node_values, optimal_actions = self._backward_induction(
            graph_data, request.stages, request.discount_factor, request.risk_tolerance or "neutral"
        )

        # Build optimal policy
        policy = self._build_policy(
            graph_data, request.stages, node_values, optimal_actions, request.discount_factor
        )

        # Generate stage analyses
        stage_analyses = self._generate_stage_analyses(
            graph_data, request.stages, node_values, optimal_actions, request.discount_factor
        )

        # Arch step 1 (2026-07-26): `value_of_flexibility` and the
        # `sensitivity_to_timing` label derived from it are OMITTED, and the
        # estimators behind them (_compute_value_of_flexibility,
        # _calculate_committed_value, _calculate_average_continuation,
        # _assess_timing_sensitivity) are removed rather than left unmounted.
        # The committed leg took np.mean over a chance node's branches while the
        # flexible leg used _edge_probability, so the difference between the two
        # was an estimator gap reported as economic value (40.0 where the true
        # value is 0 — see SequentialAnalysisResponse's docstring and
        # CODEX-SCIENCE-CLAIMS-VERIFY-2026-07-26.md claim 1). A correct
        # committed value needs information sets this request schema cannot
        # express; without them it collapses onto backward induction and the
        # field is identically 0. Nothing else in this response depends on it.
        return SequentialAnalysisResponse(
            optimal_policy=policy,
            stage_analyses=stage_analyses,
        )



    def _build_graph_data(self, graph: SequentialGraph) -> Dict[str, Any]:
        """Build internal graph representation."""
        nodes = {}
        edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # from_node -> list of edges
        incoming_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(
            list
        )  # to_node -> list of edges

        for node in graph.nodes:
            nodes[node.id] = {
                "id": node.id,
                "type": node.type,
                "label": node.label,
                "payoff": node.payoff,
                "probabilities": node.probabilities,
            }

        for edge in graph.edges:
            edge_data = {
                "from": edge.from_node,
                "to": edge.to_node,
                "action": edge.action,
                "outcome": edge.outcome,
                "probability": edge.probability,
                "immediate_payoff": edge.immediate_payoff or 0,
            }
            edges[edge.from_node].append(edge_data)
            incoming_edges[edge.to_node].append(edge_data)

        return {
            "nodes": nodes,
            "edges": dict(edges),
            "incoming_edges": dict(incoming_edges),
            "stage_assignments": graph.stage_assignments,
        }

    def _backward_induction(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        discount_factor: float,
        risk_tolerance: str,
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Perform backward induction to find optimal policy.

        Returns:
            Tuple of (node_values, optimal_actions)
        """
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        stage_assignments = graph_data["stage_assignments"]

        # Initialize values for terminal nodes
        node_values: Dict[str, float] = {}
        optimal_actions: Dict[str, str] = {}

        for node_id, node in nodes.items():
            if node["type"] == "terminal":
                payoff = node.get("payoff", 0) or 0
                node_values[node_id] = self._risk_adjust_value(payoff, 0, risk_tolerance)

        # Sort stages so the driver visits nodes leaf-first (highest stage_index
        # first). This ordering is NOT merely cosmetic and must NOT be dropped as
        # "no longer required for correctness": seeding later-stage node_values
        # before earlier stages ALSO bounds resolve()'s recursion depth in staged
        # graphs — each resolve() call then finds most of its children already
        # valued instead of recursing the whole chain. Correctness survives without
        # the sort (resolve() values any child on demand), but bounded recursion
        # depth does not. Kept for both bounded depth and stable, low-surprise order.
        sorted_stages = sorted(stages, key=lambda s: s.stage_index, reverse=True)

        # Valuation is dependency-ordered, NOT stage-ordered: a node is valued by
        # first (recursively) valuing every child it feeds. This guarantees
        # children-before-parents even when a chance node and the decision it feeds
        # share a stage_index (the request schema's OWN canonical example:
        # market:1, pricing:1). The previous reverse-stage sweep valued nodes in
        # arbitrary intra-stage order and read a not-yet-valued child as
        # continuation 0, silently dropping its entire subtree and sign-inverting
        # the root value. A child that genuinely cannot be resolved (dangling edge
        # or cycle) now FAILS LOUD instead of being fabricated as 0.
        resolving: Set[str] = set()

        def edge_value(edge: Dict[str, Any]) -> float:
            """Immediate payoff plus the discounted continuation value of the
            edge's child. Shared by the decision (max) and chance (expectation)
            branches so the `immediate + discount_factor * resolve(child)` formula
            lives in exactly one place (the module-level `_discounted_edge_value`).
            Closes over resolve(), defined below."""
            immediate = _edge_immediate_payoff(edge)
            return _discounted_edge_value(immediate, discount_factor, resolve(edge["to"]))

        def resolve(node_id: str) -> float:
            if node_id in node_values:
                return node_values[node_id]  # terminal (pre-seeded) or already valued
            if node_id not in nodes:
                raise ValueError(
                    f"Sequential graph references undefined node '{node_id}' "
                    f"(dangling edge); cannot value backward induction."
                )
            if node_id in resolving:
                raise ValueError(
                    f"Sequential graph contains a cycle through node '{node_id}'; "
                    f"backward induction requires a directed acyclic graph."
                )

            node = nodes[node_id]
            outgoing = edges.get(node_id, [])

            if not outgoing:
                # No outgoing edges - treat as terminal with 0 payoff
                node_values[node_id] = 0.0
                return 0.0

            resolving.add(node_id)
            try:
                if node["type"] == "decision":
                    # Decision node: maximize over actions
                    best_value = float("-inf")
                    best_action: str = ""

                    for edge in outgoing:
                        total = edge_value(edge)
                        if total > best_value:
                            best_value = total
                            best_action = str(edge.get("action", edge["to"]))

                    node_values[node_id] = best_value
                    optimal_actions[node_id] = best_action
                    return best_value

                elif node["type"] == "chance":
                    # Chance node: expected value over outcomes
                    expected_value: float = 0.0
                    total_prob: float = 0.0

                    for edge in outgoing:
                        prob = _edge_probability(edge, len(outgoing))
                        expected_value += prob * edge_value(edge)
                        total_prob += prob

                    # F4d (A3, 2026-07-22): reject a chance node whose total
                    # probability mass is effectively zero (<= _PROB_SUM_TOLERANCE)
                    # — it is not a valid distribution and its expected value is
                    # undefined. The prior code skipped normalisation when
                    # total_prob == 0 and valued the node at 0, silently collapsing
                    # the root to a wrong value at HTTP 200. Fail loud -> 422 (route
                    # D-12). Omitted probabilities default to an equal split
                    # (mass == 1), so that path is unaffected.
                    #
                    # A non-zero but non-unit sum is renormalised below. Behaviour
                    # UNCHANGED by 2.704, but its justification has moved and the
                    # note is corrected rather than left to rot: this leniency used
                    # to be load-bearing for the dark stage-sensitivity perturbation
                    # (`_perturb_parameter` scaled a node's probabilities and re-ran
                    # induction, relying on renormalisation to keep the distribution
                    # proper). That route and that helper were RETIRED in 2.704, so
                    # nothing internal depends on the leniency any more.
                    #
                    # It is deliberately NOT tightened here: renormalising a
                    # non-unit sum is caller-visible behaviour on the LIVE
                    # /api/v1/analysis/sequential mount, and changing it is a
                    # separate, testable decision — not a side effect of a
                    # retirement. A future lane may now tighten to a strict
                    # sum-to-1 check without the blocker this comment used to name.
                    if total_prob <= _PROB_SUM_TOLERANCE:
                        raise ValueError(
                            f"Chance node '{node_id}' has effectively zero total "
                            f"probability mass ({total_prob:g} <= {_PROB_SUM_TOLERANCE:g}) "
                            f"across its {len(outgoing)} outgoing branches: it is not "
                            f"a valid distribution and its expected value is undefined. "
                            f"Provide branch probabilities that sum to 1 (or omit "
                            f"them for an equal split)."
                        )

                    # Normalize if probabilities don't sum to 1
                    if abs(total_prob - 1.0) > 0.01:
                        expected_value /= total_prob

                    # Apply risk adjustment
                    variance = self._estimate_outcome_variance(
                        outgoing, node_values, discount_factor
                    )
                    value = self._risk_adjust_value(expected_value, variance, risk_tolerance)
                    node_values[node_id] = value
                    return value

                # Unsupported node type - unreachable through the pattern-validated
                # request model (type is one of decision/chance/terminal), but fail
                # loud rather than fabricate a value if one ever slips through.
                raise ValueError(
                    f"Sequential graph node '{node_id}' has unsupported type "
                    f"'{node['type']}'."
                )
            finally:
                resolving.discard(node_id)

        # Bucket every node by its stage_index in a SINGLE pass, so the driver loop
        # does one dict lookup per stage instead of re-scanning all stage_assignments
        # for every stage. Coverage is identical: only nodes whose stage_index
        # appears in `stages` are driven (buckets for other stage indices are never
        # looked up), and intra-stage order matches stage_assignments iteration
        # order exactly.
        nodes_by_stage: Dict[int, List[str]] = {}
        for node_id, s in stage_assignments.items():
            nodes_by_stage.setdefault(s, []).append(node_id)

        # Drive valuation over the same node set the previous stage sweep covered:
        # every node assigned to a stage that appears in `stages`. resolve() values
        # any reachable child on demand, so intra-stage ordering no longer matters.
        for stage in sorted_stages:
            for node_id in nodes_by_stage.get(stage.stage_index, []):
                resolve(node_id)

        return node_values, optimal_actions

    def _risk_adjust_value(self, mean: float, variance: float, risk_tolerance: str) -> float:
        """Apply risk adjustment to expected value."""
        if risk_tolerance == "neutral" or variance == 0:
            return mean
        elif risk_tolerance == "averse":
            # Mean-standard-deviation penalty (sigma units), symmetric with the
            # 'seeking' branch and _calculate_resolved_uncertainty which already use
            # sqrt(variance). D-13 units fix; coefficient is DOCTRINE-PENDING(Neil).
            # (variance == 0 is handled by the first branch, so sqrt is safe here.)
            return float(mean - RISK_AVERSION_COEFFICIENT * np.sqrt(variance))
        elif risk_tolerance == "seeking":
            # Risk-seeking: slight bonus for variance
            return mean + 0.1 * np.sqrt(variance) if variance > 0 else mean
        return mean

    def _estimate_outcome_variance(
        self, outgoing_edges: List[Dict], node_values: Dict[str, float], discount_factor: float
    ) -> float:
        """Estimate variance of outcomes from a chance node."""
        if not outgoing_edges:
            return 0

        values = []
        probs = []

        for edge in outgoing_edges:
            child_id = edge["to"]
            prob = _edge_probability(edge, len(outgoing_edges))
            immediate = _edge_immediate_payoff(edge)

            if child_id not in node_values:
                # RW-6a: absent != zero. A child missing from node_values has an
                # UNKNOWN continuation, not a zero one. The backward-induction
                # caller resolves every child before computing variance, so this
                # only fires from _calculate_resolved_uncertainty when a resolution
                # chance node feeds a non-terminal that induction never valued.
                # Fabricating `value = immediate` (treating the unknown
                # continuation as 0) silently corrupts resolved_uncertainty. Fail
                # loud, matching the engine's absent-as-0 doctrine (see resolve()).
                raise ValueError(
                    f"Sequential variance estimate references unvalued node "
                    f"'{child_id}': its continuation value is unknown (absent from "
                    f"backward induction), not zero. The node must be reachable "
                    f"from a staged node before its variance can be estimated."
                )
            value = _discounted_edge_value(immediate, discount_factor, node_values[child_id])

            values.append(value)
            probs.append(prob)

        # Normalize probabilities
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]

        # Calculate variance
        mean = sum(p * v for p, v in zip(probs, values))
        variance = sum(p * (v - mean) ** 2 for p, v in zip(probs, values))

        return float(variance)

    @staticmethod
    def _require_valued_decision_node(node_id: str, node_values: Dict[str, float]) -> None:
        """Fail loud (F-2 / F-1a) when a decision node listed in a stage's
        decision_nodes was never valued by backward induction.

        `decision_nodes` is an independent list the request model never
        cross-validates against `stage_assignments`; a node listed here but not
        driven (missing from stage_assignments under an analysed stage, and
        unreachable from any staged node) is a client-input STAGING defect, not an
        internal failure. Raising ValueError maps it to 422 via D-12 with an
        actionable message, instead of a mislabeled KeyError-500 (F-2) or a
        fabricated absent-as-0 StageOption (F-1a). Single source of the message so
        both call sites stay in lockstep.
        """
        if node_id not in node_values:
            raise ValueError(
                f"Sequential staging defect: decision node '{node_id}' is listed in "
                f"a stage's decision_nodes but backward induction never valued it — "
                f"its stage_assignments entry is missing (or maps to a stage not "
                f"among the analysed `stages`), and it is unreachable from any staged "
                f"node. Assign '{node_id}' to an analysed stage so its subtree can be "
                f"valued."
            )

    def _build_policy(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        node_values: Dict[str, float],
        optimal_actions: Dict[str, str],
        discount_factor: float,
    ) -> Policy:
        """Build policy from backward induction results."""
        stage_policies = []
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        for stage in sorted(stages, key=lambda s: s.stage_index):
            # Get decision nodes at this stage
            decision_nodes = stage.decision_nodes

            if not decision_nodes:
                continue

            # Build the decision rule for this stage's decision node.
            # Arch step 1 (2026-07-26): `DecisionStage.decision_nodes` is now
            # capped at ONE node and >1 is rejected at the request boundary
            # (MULTI_DECISION_STAGE_UNSUPPORTED), so the `break` below can no
            # longer silently discard decisions the client supplied. The loop
            # remains because a declared node may be absent from the graph or
            # not typed `decision`, in which case this stage yields no policy.
            for node_id in decision_nodes:
                if node_id not in nodes:
                    continue

                node = nodes[node_id]
                if node["type"] != "decision":
                    continue

                # F-2: fail loud on a mis-staged decision node before the child
                # index below can raise a mislabeled KeyError-500 (-> 422 via D-12).
                self._require_valued_decision_node(node_id, node_values)

                # Get optimal action
                default_action = optimal_actions.get(node_id, "none")

                # Build conditional actions from edges
                conditional_actions = []
                outgoing = edges.get(node_id, [])

                for edge in outgoing:
                    action = edge.get("action", edge["to"])
                    child_id = edge["to"]

                    # RW-6b: the value of taking this action is the edge's own
                    # value -- immediate payoff plus discounted continuation.
                    # Reporting the bare child value dropped the immediate payoff
                    # (and the discount) for every already-valued child (post-#85:
                    # always).
                    #
                    # Direct-index node_values[child_id], NOT .get(child_id, 0): the
                    # F-2 guard above proved this decision node was resolved by
                    # backward induction, and resolve() values every child before
                    # returning, so each child here is guaranteed present. Past that
                    # guard a KeyError would be a true internal invariant breach --
                    # still fail loud, never fabricate continuation 0 (the
                    # absent-as-0 class this lane kills; cf. RW-6a's fail-loud).
                    immediate = _edge_immediate_payoff(edge)
                    ev = _discounted_edge_value(
                        immediate, discount_factor, node_values[child_id]
                    )

                    # Add as conditional action if not default
                    if action != default_action:
                        # Generate a condition based on context
                        condition = self._generate_condition_string(edge, graph_data)

                        conditional_actions.append(
                            ConditionalAction(
                                condition=condition, action=action, expected_value_if_taken=ev
                            )
                        )

                decision_rule = DecisionRule(
                    default_action=default_action, conditional_actions=conditional_actions
                )

                # Determine what this stage is contingent on
                contingent_on = stage.resolution_nodes or []

                stage_policies.append(
                    StagePolicy(
                        stage_index=stage.stage_index,
                        stage_label=stage.stage_label,
                        decision_rule=decision_rule,
                        contingent_on=contingent_on,
                    )
                )

                # One policy per stage. Not "for simplicity" any more: the
                # request boundary rejects >1 decision node per stage, so at
                # most one iteration can ever reach here and nothing is dropped.
                break

        # Calculate expected total value
        root_value = self._get_root_value(graph_data, node_values)

        # F4b (A3, 2026-07-22): the fabricated `value_distribution` is OMITTED. It
        # reported normal params with std = 0.2 * |root_value| — a made-up spread
        # presented as if measured, while backward induction is deterministic (it
        # draws no noise). A genuine policy-value distribution is a modeling
        # roadmap item.
        return Policy(
            stages=stage_policies,
            expected_total_value=root_value,
        )

    def _generate_condition_string(self, edge: Dict, graph_data: Dict) -> str:
        """Generate human-readable condition string for an edge."""
        outcome = edge.get("outcome")
        action = edge.get("action")
        to_node = edge["to"]

        if outcome:
            return f"If {outcome}"
        elif to_node in graph_data["nodes"]:
            node_label = graph_data["nodes"][to_node].get("label", to_node)
            return f"If choosing {action or node_label}"
        else:
            return f"If {action or 'alternative'}"

    def _get_root_value(self, graph_data: Dict[str, Any], node_values: Dict[str, float]) -> float:
        """Get value at the root node (the stage-0 decision, else any stage-0 node).

        F-1b: absent != zero. The prior `.get(node_id, 0)` / `max(...)` / `return 0`
        fallbacks fabricated expected_total_value 0.0 whenever there was no root to
        read — no node assigned to stage 0, or a stage-0 root that backward
        induction never valued (stage 0 omitted from the analysed `stages`). Both
        are client-input STAGING defects; fail loud (-> 422 via D-12) instead. A
        root LEGITIMATELY worth 0.0 is present in node_values and is returned
        normally below — only a genuinely-absent root raises.
        """
        stage_assignments = graph_data["stage_assignments"]
        nodes = graph_data["nodes"]

        # Prefer the stage-0 decision node (the true root); else any stage-0 node.
        root_id: Optional[str] = None
        for node_id, stage in stage_assignments.items():
            if stage == 0 and nodes[node_id]["type"] == "decision":
                root_id = node_id
                break
        if root_id is None:
            for node_id, stage in stage_assignments.items():
                if stage == 0:
                    root_id = node_id
                    break

        if root_id is None:
            raise ValueError(
                "Sequential staging defect: no node is assigned to stage 0, so the "
                "decision problem has no root stage and expected_total_value is "
                "undefined. Assign the initial decision to stage_index 0."
            )
        if root_id not in node_values:
            raise ValueError(
                f"Sequential staging defect: the stage-0 root node '{root_id}' was "
                f"not valued by backward induction (stage 0 is not among the analysed "
                f"`stages`, or the node is unreachable). Ensure stage 0 is analysed."
            )
        return node_values[root_id]

    def _generate_stage_analyses(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        node_values: Dict[str, float],
        optimal_actions: Dict[str, str],
        discount_factor: float,
    ) -> List[StageAnalysis]:
        """Generate detailed analysis for each stage."""
        analyses = []
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        for stage in sorted(stages, key=lambda s: s.stage_index):
            options = []

            for node_id in stage.decision_nodes:
                if node_id not in nodes:
                    continue

                node = nodes[node_id]
                if node["type"] != "decision":
                    continue

                # F-1a: _build_policy `break`s after the FIRST decision node per
                # stage, so a second mis-staged decision node in decision_nodes
                # reaches only here. Fail loud on it too (same guard, -> 422) rather
                # than fabricating a StageOption from an unvalued continuation.
                self._require_valued_decision_node(node_id, node_values)

                # Analyze each available action
                outgoing = edges.get(node_id, [])

                for edge in outgoing:
                    action = edge.get("action", edge["to"])
                    child_id = edge["to"]
                    immediate = _edge_immediate_payoff(edge)

                    # Direct-index, NOT node_values.get(child_id, 0): past the guard
                    # above this decision node is valued, so every child is present
                    # (resolve() values them all). A missing child is a breach -->
                    # fail loud, never fabricate continuation 0 (absent != zero).
                    continuation = node_values[child_id]
                    total = _discounted_edge_value(immediate, discount_factor, continuation)

                    options.append(
                        StageOption(
                            option_id=action,
                            label=action.replace("_", " ").title(),
                            immediate_value=immediate,
                            continuation_value=continuation,
                            total_value=total,
                        )
                    )

            # F4c: magnitude of outcome dispersion resolved at this stage (sqrt of
            # summed chance-node variance) — NOT a value of information.
            resolved_uncertainty = self._calculate_resolved_uncertainty(
                stage, graph_data, node_values, discount_factor
            )

            # S3 (A3 VOI honesty, D-23.8): honest per-stage EVPI in outcome units,
            # replacing the deleted `optimal_waiting_value` (discount × sqrt(Σvar)
            # heuristic). Computed for this stage's decision node (the first valid
            # one, matching _build_policy) as E_C[max_a Q] − max_a E_C[Q] on the
            # backward-induction tree. `stage_evpi_status` discloses the value's
            # status: absent (None) when COMPUTED and EXACT (incl. a real EVPI of 0,
            # the identified single-chance case); 'no_decision_node' when the stage
            # has no decision to inform (null value); 'skipped_joint_space_too_large'
            # when the decide-after joint enumeration would exceed the safety cap
            # (F-3 honest skip of an auxiliary metric, null value); or
            # 'assumed_independent_coupling' with coupling_assumption populated (F1,
            # D-23.11) when the value is COMPUTED but under an independence assumption
            # across >=2 action-specific chance nodes that the tree does not identify.
            primary_decision = next(
                (
                    nid
                    for nid in stage.decision_nodes
                    if nid in nodes and nodes[nid]["type"] == "decision"
                ),
                None,
            )
            if primary_decision is not None:
                stage_evpi, stage_evpi_status, coupling_assumption = (
                    self._compute_stage_evpi(
                        primary_decision, graph_data, node_values, discount_factor
                    )
                )
            else:
                stage_evpi, stage_evpi_status, coupling_assumption = (
                    None,
                    "no_decision_node",
                    None,
                )

            analyses.append(
                StageAnalysis(
                    stage_index=stage.stage_index,
                    stage_label=stage.stage_label,
                    options_at_stage=options,
                    resolved_uncertainty=resolved_uncertainty,
                    stage_evpi=stage_evpi,
                    stage_evpi_status=stage_evpi_status,
                    coupling_assumption=coupling_assumption,
                )
            )

        return analyses

    def _calculate_resolved_uncertainty(
        self,
        stage: DecisionStage,
        graph_data: Dict[str, Any],
        node_values: Dict[str, float],
        discount_factor: float,
    ) -> float:
        """Magnitude of outcome dispersion resolved at this stage.

        F4c (A3, 2026-07-22): this was named `_calculate_information_value` and its
        result was emitted as `information_value` with the claim that it is "the
        difference between value with and without information." It is NOT: it
        computes sqrt(Σ variance) over the chance nodes resolving at this stage —
        a payoff-unit magnitude of how dispersed the outcomes are, with no
        with-information vs without-information comparison anywhere. A true value
        of information (E[value|info] − E[value]) is a modeling roadmap item; the
        honest quantity is relabeled accordingly.
        """
        # Get chance nodes that resolve at this stage
        resolution_nodes = stage.resolution_nodes
        if not resolution_nodes:
            return 0

        total_variance: float = 0.0
        for node_id in resolution_nodes:
            if node_id in graph_data["nodes"]:
                node = graph_data["nodes"][node_id]
                if node["type"] == "chance":
                    outgoing = graph_data["edges"].get(node_id, [])
                    variance = self._estimate_outcome_variance(
                        outgoing, node_values, discount_factor
                    )
                    total_variance += variance

        # sqrt of total variance = standard-deviation-scale dispersion magnitude
        return np.sqrt(total_variance) if total_variance > 0 else 0

    def _subtree_chance_nodes(
        self,
        start_id: str,
        nodes: Dict[str, Any],
        edges: Dict[str, List[Dict[str, Any]]],
    ) -> set:
        """All chance-node ids reachable from ``start_id`` (including itself).

        Primitive for the F1 identifiability logic in ``_compute_stage_evpi``:
        the SETS (not just a boolean) are needed because a chance node shared
        between two actions' subtrees means the tree IDENTIFIES their coupling
        (same id = same random variable) — emitting the independence value there
        contradicts the submitted graph (Codex re-confirm, D-23.19). Iterative
        DFS with a visited set so graph cycles terminate; cost is O(subtree)
        and the graph is capped at 100 nodes.
        """
        stack = [start_id]
        seen: set = set()
        chance: set = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            if nodes.get(nid, {}).get("type") == "chance":
                chance.add(nid)
            for e in edges.get(nid, []):
                target = e.get("to")
                if target is not None and target not in seen:
                    stack.append(target)
        return chance

    def _subtree_reaches_chance(
        self,
        start_id: str,
        nodes: Dict[str, Any],
        edges: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        """Does the subtree rooted at ``start_id`` contain any reachable chance node
        (including ``start_id`` itself)? Derived from ``_subtree_chance_nodes`` —
        one traversal, one truth (see that docstring for why the sets exist).
        """
        return bool(self._subtree_chance_nodes(start_id, nodes, edges))

    def _compute_stage_evpi(
        self,
        decision_node_id: str,
        graph_data: Dict[str, Any],
        node_values: Dict[str, float],
        discount_factor: float,
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """Honest per-stage EVPI (S3 — A3 VOI, D-23.8): the expected value of
        perfectly resolving the chance node(s) this decision faces BEFORE choosing,
        minus deciding without that information.

            stage_evpi = E_C[max_a Q(a | C)] − max_a E_C[Q(a)]   (>= 0, outcome units)

        Returns ``(stage_evpi, status, coupling_assumption)``:
        * ``(value, None, None)`` — computed EXACT value (may be a real 0.0); the
          IDENTIFIED case (see below);
        * ``(value, "assumed_independent_coupling", "independence_across_actions")``
          — computed but under an UNIDENTIFIED independence assumption (F1, D-23.11);
          the value is present but NOT exact for the supplied tree — see below;
        * ``(None, "skipped_joint_space_too_large", None)`` — the decide-after joint
          enumeration ∏(branch counts) exceeds ``_STAGE_EVPI_JOINT_CELL_CAP``, an
          F-3 honest skip of this auxiliary metric (the exact O(nodes) analysis is
          untouched). The join size is measured multiplicatively with an
          overflow-safe early exit — the ∏ is NEVER materialised.
        * ``(None, "skipped_shared_chance_nodes_unsupported", None)`` — the SAME
          chance-node id is reachable from >=2 actions' subtrees, so the graph
          IDENTIFIES their coupling and the independence enumeration would
          contradict it (Codex re-confirm, D-23.19); skipped until conditional
          subtree re-valuation ships (rowed refinement).

        This REPLACES the deleted ``optimal_waiting_value`` (a discount × sqrt(Σvar)
        dispersion heuristic dressed as an option value) with a real value of
        information. Both legs read the backward-induction tree the engine already
        builds — no new sampling, no heuristic:

        * decide-now leg = ``node_values[decision]`` = ``max_a E_C[Q(a)]`` (the value
          the optimal policy maximises: commit the action before the chance resolves).
        * decide-after leg = ``E_C[max_a Q(a | C=outcome)]``: for each realised
          outcome of the immediate chance node(s) under the decision's actions,
          re-pick the best action, then average over the chance outcomes. Jensen's
          inequality makes it >= the decide-now leg.

        It is exactly 0 when one action dominates in every chance branch (perfect
        information never changes the choice) and when the decision faces no
        immediate chance node at all (nothing to resolve).

        SCOPE & IDENTIFIABILITY (F1, D-23.11 — the joint law across actions is NOT in
        the tree; detection broadened in the Fable adversarial round; scope clarified
        in round 2). stage_evpi is scoped to THIS stage's IMMEDIATE chance — the chance
        node(s) directly under the decision's actions. Chance deeper in the tree is a
        LATER stage's EVPI (computed when that stage is analysed), NOT folded in here.
        Hence the two early returns above (no action edges / no immediate chance) yield
        an EXACT 0.0 with no status: a stage facing no immediate uncertainty has nothing
        to resolve, whatever chance lies deeper (round-2 T4). Only once the decision
        DOES face immediate chance does identifiability bite — and there the tree
        supplies only the MARGINAL outcome distribution under each action:

        * IDENTIFIED (EXACT, no status) — at most ONE action faces reachable chance
          (every other action's subtree is chance-free). Resolving that single
          uncertainty and re-maxing over actions — each deterministic action carrying
          its exact ``Q`` — is an exact ``E_C[max_a Q(a|C)]``.
        * UNIDENTIFIED — >=2 actions each have a chance node reachable in their subtree
          (one immediate, another immediate OR one level deeper). The decide-after leg
          then mixes one action's per-branch realised value against another action's
          AVERAGED value (its ``unconditional_q`` = ``E[Q_b]`` substituted for
          ``E[Q_b | C]``), i.e. it ASSUMES the resolved chance is INDEPENDENT of the
          other action's (possibly deeper) uncertainty. The tree does not identify that
          joint law — the same marginals also admit same-state coupling (EVPI can be 0),
          opposite coupling (larger EVPI), etc. So the number is one unrequested
          modelling choice, NOT exact for the supplied tree. Per D-23.11 (disclose-and-
          status, NOT nullify — the value stays useful and the endpoint stays live) it
          is returned WITH ``status='assumed_independent_coupling'`` and
          ``coupling_assumption='independence_across_actions'`` so a consumer never
          reads it as identified/exact. NOTE (value-refinement, flagged for Paul/Neil):
          when the deeper node is the SAME chance node reused under another action, the
          tree DOES identify the joint (same-state) and the exact EVPI differs from the
          emitted independence value (e.g. 0 vs 25); this round SAFELY CONTAINS that by
          disclosing (never claiming exact), not by recomputing the shared-state value.
          (Doctrine alternatives — require a shared scenario variable; emit status-only
          with a null value; or give the no-immediate-chance-but-deeper case its own
          status [metric-scope question, round-2 R2-F1] — are Paul/Neil flags; this
          implements disclose+value+status with per-immediate-stage scope.)

        RISK POSTURE (documented choice; flagged for review): the legs use the
        engine's own risk-ADJUSTED ``node_values`` for every continuation, so each
        per-outcome argmax selects exactly the action the optimal policy would —
        consistent with how the policy chooses actions (argmax over risk-adjusted
        values). The outer average over the RESOLVED chance is NOT re-penalised for
        that chance's own variance (perfect information has removed it), while every
        deeper continuation keeps its ``node_values`` adjustment. For
        ``risk_tolerance='neutral'`` this is the standard risk-neutral EVPI. Because
        mean−kσ is a DOCTRINE-PENDING(Neil) heuristic rather than a coherent utility,
        the ``max(0, ·)`` clamp guards its rare negative under risk aversion; for
        neutral it never binds.
        """
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        action_edges = edges.get(decision_node_id, [])
        if not action_edges:
            return 0.0, None, None

        # The immediate chance node(s) directly under the decision's actions = the
        # uncertainty this decision faces AT THIS STAGE. If none, this stage has no
        # immediate uncertainty to resolve, so its EVPI is EXACTLY 0 (status None) —
        # per-immediate-stage scope (round-2 R2-F1): any chance deeper in the tree
        # belongs to a LATER stage's EVPI, computed when that stage is analysed, and is
        # NOT an unidentified-coupling case here (no independence product runs when
        # there is no immediate chance to product over). The subtree scan below only
        # discriminates identified vs unidentified once immediate chance DOES exist.
        has_chance = any(
            nodes.get(e["to"], {}).get("type") == "chance" for e in action_edges
        )
        if not has_chance:
            return 0.0, None, None

        # decide-now leg is exactly what backward induction stored for the decision.
        decide_now = node_values[decision_node_id]

        # Each action's UNCONDITIONAL Q(a) = immediate + γ·V(child) — used when the
        # action's child is not one of the chance nodes resolved in a joint outcome.
        unconditional_q: List[float] = []
        for e in action_edges:
            child = e["to"]
            if child not in node_values:
                # absent != zero (engine doctrine): a staged decision has every
                # child valued by backward induction; a missing one is a breach.
                raise ValueError(
                    f"stage_evpi references unvalued child '{child}' of decision "
                    f"'{decision_node_id}'."
                )
            unconditional_q.append(
                _discounted_edge_value(
                    _edge_immediate_payoff(e), discount_factor, node_values[child]
                )
            )

        # For each resolved chance node (a chance child of an action), its outcome
        # branches with NORMALISED probabilities and the value AT the chance node
        # given that outcome (edge_value(C→branch)) — the same normalisation and
        # edge valuation resolve() uses, so the leg is consistent with node_values.
        resolved: List[Tuple[str, List[Tuple[float, float]]]] = []
        for e in action_edges:
            cid = e["to"]
            if nodes.get(cid, {}).get("type") != "chance":
                continue
            out = edges.get(cid, [])
            weighted = [(_edge_probability(b, len(out)), b) for b in out]
            total_p = sum(p for p, _ in weighted)
            if total_p <= _PROB_SUM_TOLERANCE:
                # Mirrors resolve()'s F4d guard: a zero-mass distribution has no EVPI.
                raise ValueError(
                    f"stage_evpi: chance node '{cid}' has effectively zero total "
                    f"probability mass; its expected value is undefined."
                )
            branches: List[Tuple[float, float]] = []
            for p, b in weighted:
                bchild = b["to"]
                if bchild not in node_values:
                    raise ValueError(
                        f"stage_evpi: unvalued branch child '{bchild}' of chance "
                        f"node '{cid}'."
                    )
                branch_value = _discounted_edge_value(
                    _edge_immediate_payoff(b), discount_factor, node_values[bchild]
                )
                branches.append((p / total_p, branch_value))
            resolved.append((cid, branches))

        # F-A3 (Fable adversarial round): a chance node reached by k action edges lands
        # in `resolved` k times, but it is ONE random variable — resolving it once fixes
        # its branch for every action that reads it (`realised[cid]` is keyed by cid, so
        # duplicate entries marginalise out exactly: Σ p_i = 1). Dedupe by cid BEFORE the
        # cap count and the product so a shared node costs B cells, not B^k — otherwise a
        # 65-branch node shared by 2 actions is counted 65^2 > cap and falsely skipped.
        seen_cids: set = set()
        deduped_resolved: List[Tuple[str, List[Tuple[float, float]]]] = []
        for cid, branches in resolved:
            if cid not in seen_cids:
                seen_cids.add(cid)
                deduped_resolved.append((cid, branches))
        resolved = deduped_resolved

        # F-3 SAFETY GUARD: the decide-after leg enumerates ∏(branch counts) joint
        # cells = B^K for K chance-child actions. A legal request can drive K ~97
        # (2^97 cells) — an un-preemptable CPU freeze of the async event loop.
        # Measure the joint size MULTIPLICATIVELY with an overflow-safe early exit
        # (never materialise the ∏, so K=97 costs ~13 multiplications then bails). If
        # it exceeds the cap, honestly SKIP this auxiliary metric (null + status)
        # rather than 422-ing the valid, O(nodes) analysis (over-rejection).
        joint_cells = 1
        for _, branches in resolved:
            joint_cells *= len(branches)
            if joint_cells > _STAGE_EVPI_JOINT_CELL_CAP:
                return None, "skipped_joint_space_too_large", None

        # F1 IDENTIFIABILITY (D-23.11; detection BROADENED in the Fable adversarial
        # round — it must scan the WHOLE per-action subtree, not just the immediate
        # child). The tree gives only the per-action MARGINAL chance distribution; it
        # does NOT identify the JOINT law of outcomes across mutually-exclusive actions.
        # The decide-after leg is EXACT only when at most ONE action faces any resolvable
        # uncertainty — then resolving that single chance and re-maxing (every other
        # action being deterministic, its Q is its exact value) is identified. When >=2
        # actions each have a chance node ANYWHERE in their subtree, the leg mixes one
        # action's per-branch realised value against another action's AVERAGED value
        # (its `unconditional_q`, = E[Q_b] substituted for E[Q_b | C]) — i.e. it ASSUMES
        # the resolved chance is independent of the other action's (possibly deeper)
        # uncertainty. That is the SAME unidentified cross-action joint whether the
        # second action's chance is immediate (Codex) or one level down (adversarial
        # FN-1). A prior version counted only DISTINCT IMMEDIATE chance children and
        # labelled FN-1/FN-2 exact — a false negative. Disclose whenever >=2 actions
        # face DISJOINT reachable chance; when the SAME chance id is reachable from
        # >=2 actions the coupling is IDENTIFIED by the graph and the value is
        # SKIPPED instead (see the shared_chance guard below — Codex re-confirm
        # D-23.19 superseded the earlier disclose-not-recompute posture for FN-2).
        per_action_chance = [
            self._subtree_chance_nodes(e["to"], nodes, edges) for e in action_edges
        ]

        # F1 SHARPENED (Codex re-confirm, D-23.19): if the SAME chance-node id is
        # reachable from >=2 actions' subtrees, the tree IDENTIFIES the coupling
        # (same id = same random variable — the dedupe above already treats it as
        # one variable for probability mass). The independence label would then
        # CONTRADICT the submitted graph, and the enumeration below would mix one
        # action's REALISED copy against another action's AVERAGED copy of the
        # same variable (their repro: two routes to one 0/100 coin — identified
        # EVPI 0, we emitted 25). Until conditional subtree re-valuation ships
        # (rowed refinement: pin realised nodes and re-derive each action's Q per
        # joint cell), honestly SKIP: null + a dedicated status. Codex-blessed
        # containment ("if that refinement cannot ship immediately, emit null
        # with a shared-state unsupported status").
        shared_chance: set = set()
        for i in range(len(per_action_chance)):
            for j in range(i + 1, len(per_action_chance)):
                shared_chance |= per_action_chance[i] & per_action_chance[j]
        if shared_chance:
            return None, "skipped_shared_chance_nodes_unsupported", None

        actions_facing_chance = sum(1 for s in per_action_chance if s)
        assumes_independence = actions_facing_chance >= 2

        # Enumerate the JOINT outcome space of the resolved chance nodes. When >=2
        # actions face reachable chance this product treats the resolved marginals as
        # INDEPENDENT of the other actions' uncertainty (the F1 assumption disclosed
        # above, `assumes_independence`); when at most one action faces uncertainty it is
        # the exact E_C[max_a Q(a|C)] over that single resolved state. Under each joint
        # outcome an action keeps its unconditional Q unless its child IS one of the
        # resolved chance nodes, in which case it takes that node's realised branch
        # value; then max over actions, average by joint prob.
        e_after = 0.0
        for combo in itertools.product(*[branches for _, branches in resolved]):
            joint_p = 1.0
            realised: Dict[str, float] = {}
            for (cid, _), (branch_p, branch_value) in zip(resolved, combo):
                joint_p *= branch_p
                realised[cid] = branch_value
            best = float("-inf")
            for idx, e in enumerate(action_edges):
                child = e["to"]
                if child in realised:
                    q = _discounted_edge_value(
                        _edge_immediate_payoff(e), discount_factor, realised[child]
                    )
                else:
                    q = unconditional_q[idx]
                if q > best:
                    best = q
            e_after += joint_p * best

        stage_evpi_value = max(0.0, e_after - decide_now)
        if assumes_independence:
            return (
                stage_evpi_value,
                "assumed_independent_coupling",
                "independence_across_actions",
            )
        return stage_evpi_value, None, None

    # Arch step 1 (2026-07-26): _compute_value_of_flexibility,
    # _calculate_committed_value, _calculate_average_continuation and
    # _assess_timing_sensitivity were REMOVED here. They produced
    # `value_of_flexibility` and the `sensitivity_to_timing` bucket derived
    # from it, both omitted from SequentialAnalysisResponse. The committed
    # leg walked the tree with np.mean over a chance node's outgoing edges
    # while the flexible leg used _edge_probability (:53), so the field
    # published the gap between two estimators of one quantity as the value
    # of waiting. See SequentialAnalysisResponse's docstring for why the
    # repair is a modelling item (information sets) and not a one-liner, and
    # CODEX-SCIENCE-CLAIMS-VERIFY-2026-07-26.md claim 1 for the reproduction.
    # The normative invariant is pinned, xfailed, at
    # tests/unit/test_arch_step1_claims.py::
    # TestValueOfFlexibilityOmitted::test_no_future_choice_implies_zero_flexibility






