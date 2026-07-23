"""
Sequential Decision Support Service for Phase 4.

Implements backward induction for multi-stage decision problems,
computing optimal policies and value of flexibility.
"""

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
    StageSensitivityRequest,
)
from src.models.responses import (
    ConditionalAction,
    DecisionRule,
    ExplanationMetadata,
    Policy,
    PolicyTreeNode,
    PolicyTreeResponse,
    SequentialAnalysisResponse,
    StageAnalysis,
    StageOption,
    StagePolicy,
    StageSensitivityResponse,
    StageSensitivityResult,
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

        # Calculate value of flexibility
        value_of_flexibility = self._compute_value_of_flexibility(
            graph_data, request.stages, node_values, request.discount_factor
        )

        # Determine sensitivity to timing
        sensitivity_to_timing = self._assess_timing_sensitivity(
            stage_analyses, value_of_flexibility
        )

        return SequentialAnalysisResponse(
            optimal_policy=policy,
            stage_analyses=stage_analyses,
            value_of_flexibility=value_of_flexibility,
            sensitivity_to_timing=sensitivity_to_timing,
        )

    def get_policy_tree(self, request: SequentialAnalysisRequest) -> PolicyTreeResponse:
        """
        Generate policy tree representation.

        Args:
            request: Sequential analysis request

        Returns:
            PolicyTreeResponse with tree structure
        """
        # Build internal representation
        graph_data = self._build_graph_data(request.graph)

        # Run backward induction
        node_values, optimal_actions = self._backward_induction(
            graph_data, request.stages, request.discount_factor, request.risk_tolerance or "neutral"
        )

        # Find root node (decision node at stage 0)
        root_node_id = None
        for node_id, stage in request.graph.stage_assignments.items():
            if stage == 0:
                node = graph_data["nodes"][node_id]
                if node["type"] == "decision":
                    root_node_id = node_id
                    break

        if root_node_id is None:
            # Fallback to first node
            root_node_id = list(graph_data["nodes"].keys())[0]

        # Build tree recursively
        root = self._build_tree_node(
            root_node_id,
            graph_data,
            request.graph.stage_assignments,
            node_values,
            optimal_actions,
            visited=set(),
        )

        # Count nodes
        total_nodes = self._count_tree_nodes(root)

        return PolicyTreeResponse(
            root=root, total_stages=len(request.stages), total_nodes=total_nodes
        )

    def stage_sensitivity(self, request: StageSensitivityRequest) -> StageSensitivityResponse:
        """
        Perform stage-by-stage sensitivity analysis.

        Args:
            request: Stage sensitivity request

        Returns:
            StageSensitivityResponse with sensitivity results
        """
        # Build internal representation
        graph_data = self._build_graph_data(request.graph)

        # Get baseline policy
        baseline_values, baseline_actions = self._backward_induction(
            graph_data, request.stages, discount_factor=0.95, risk_tolerance="neutral"
        )

        # Identify parameters to vary
        parameters = request.parameters_to_vary or self._auto_detect_parameters(graph_data)

        # Analyze each stage
        stage_results = []
        all_sensitivities: Dict[str, List[float]] = {}

        for stage in request.stages:
            stage_result = self._analyze_stage_sensitivity(
                graph_data,
                stage,
                request.stages,
                parameters,
                request.variation_range,
                baseline_values,
                baseline_actions,
            )
            stage_results.append(stage_result)

            # Collect all sensitivities
            for param, sens in stage_result.parameter_sensitivities.items():
                if param not in all_sensitivities:
                    all_sensitivities[param] = []
                all_sensitivities[param].append(sens)

        # Calculate overall robustness
        if stage_results:
            overall_robustness = float(np.mean([r.robustness_score for r in stage_results]))
        else:
            overall_robustness = 1.0

        # Find most sensitive parameters
        avg_sensitivities: Dict[str, float] = {
            param: float(np.mean(values)) for param, values in all_sensitivities.items()
        }
        most_sensitive = sorted(
            avg_sensitivities.keys(), key=lambda x: avg_sensitivities[x], reverse=True
        )[:3]

        explanation = ExplanationMetadata(
            summary=f"Policy is {'robust' if overall_robustness > 0.7 else 'moderately robust' if overall_robustness > 0.4 else 'fragile'} to parameter changes",
            reasoning=self._generate_sensitivity_reasoning(most_sensitive, avg_sensitivities),
            technical_basis="One-at-a-time sensitivity analysis with backward induction",
            assumptions=["Parameters vary independently", "Linear approximation to sensitivity"],
        )

        return StageSensitivityResponse(
            stage_results=stage_results,
            overall_robustness=round(overall_robustness, 3),
            most_sensitive_parameters=most_sensitive,
            explanation=explanation,
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
            immediate = edge.get("immediate_payoff", 0) or 0
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
                    # A non-zero but non-unit sum is renormalised below (unchanged):
                    # this is a deliberate, load-bearing behaviour — the internal
                    # stage-sensitivity perturbation (_perturb_parameter) scales a
                    # node's probabilities and re-runs induction, relying on this
                    # renormalisation to keep the distribution proper. Rejecting a
                    # non-unit sum here would break that (dark) path; tightening it
                    # to a strict sum-to-1 check belongs with a rework of that
                    # perturbation (currently a no-op under renormalisation).
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
            immediate = edge.get("immediate_payoff", 0) or 0

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

            # Build decision rule for first decision node at stage
            # (simplified - in practice might need multiple rules)
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
                    immediate = edge.get("immediate_payoff", 0) or 0
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

                break  # One policy per stage for simplicity

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
                    immediate = edge.get("immediate_payoff", 0) or 0

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

            # Calculate waiting value if applicable
            waiting_value = None
            if stage.stage_index == 0 and len(stages) > 1:
                waiting_value = self._calculate_waiting_value(
                    graph_data, stages, node_values, discount_factor
                )

            analyses.append(
                StageAnalysis(
                    stage_index=stage.stage_index,
                    stage_label=stage.stage_label,
                    options_at_stage=options,
                    resolved_uncertainty=resolved_uncertainty,
                    optimal_waiting_value=waiting_value,
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

    def _calculate_waiting_value(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        node_values: Dict[str, float],
        discount_factor: float,
    ) -> float:
        """Calculate value of waiting/delaying first stage decision."""
        # Simplified: value of waiting is related to information gained
        # by observing chance node outcomes before committing

        if len(stages) < 2:
            return 0

        # Get value at stage 0 without waiting
        root_value = self._get_root_value(graph_data, node_values)

        # F4c: proxy = discount_factor × the next stage's resolved-uncertainty
        # magnitude. This is a discounted outcome-dispersion magnitude, NOT a true
        # option value (the wait-and-decide vs commit-now difference); the field
        # description discloses that.
        next_stage_uncertainty = self._calculate_resolved_uncertainty(
            stages[1] if len(stages) > 1 else stages[0], graph_data, node_values, discount_factor
        )

        waiting_value = discount_factor * next_stage_uncertainty

        return waiting_value

    def _compute_value_of_flexibility(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        node_values: Dict[str, float],
        discount_factor: float,
    ) -> float:
        """
        Compute value of flexibility (waiting vs committing now).

        Compares:
        - V_flexible: Value of optimal policy (decide at each stage)
        - V_committed: Value of committing to stage-0 decision for all stages
        """
        # Get flexible value (optimal policy)
        v_flexible = self._get_root_value(graph_data, node_values)

        # Calculate committed value (ignore future information)
        v_committed = self._calculate_committed_value(graph_data, stages, discount_factor)

        value_of_flexibility = max(0, v_flexible - v_committed)

        return round(value_of_flexibility, 2)

    def _calculate_committed_value(
        self, graph_data: Dict[str, Any], stages: List[DecisionStage], discount_factor: float
    ) -> float:
        """Calculate value when committing upfront (ignoring future information)."""
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        stage_assignments = graph_data["stage_assignments"]

        # Find first decision node
        first_decision = None
        for node_id, stage in stage_assignments.items():
            if stage == 0 and nodes[node_id]["type"] == "decision":
                first_decision = node_id
                break

        if first_decision is None:
            return 0

        # Calculate expected value of each action without conditioning on information
        outgoing = edges.get(first_decision, [])
        best_committed_value = float("-inf")

        for edge in outgoing:
            immediate = edge.get("immediate_payoff", 0) or 0

            # Calculate expected continuation without optimal future decisions
            # (use average outcomes instead of max)
            continuation = self._calculate_average_continuation(
                edge["to"], graph_data, discount_factor, visited=set()
            )

            total = _discounted_edge_value(immediate, discount_factor, continuation)

            if total > best_committed_value:
                best_committed_value = total

        return best_committed_value if best_committed_value > float("-inf") else 0

    def _calculate_average_continuation(
        self, node_id: str, graph_data: Dict[str, Any], discount_factor: float, visited: Set[str]
    ) -> float:
        """Calculate average continuation value (non-optimal)."""
        if node_id in visited:
            return 0
        visited.add(node_id)

        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        if node_id not in nodes:
            return 0

        node = nodes[node_id]

        if node["type"] == "terminal":
            return node.get("payoff", 0) or 0

        outgoing = edges.get(node_id, [])
        if not outgoing:
            return 0

        # For both decision and chance nodes, take average (not max or expected)
        values = []
        for edge in outgoing:
            immediate = edge.get("immediate_payoff", 0) or 0
            continuation = self._calculate_average_continuation(
                edge["to"], graph_data, discount_factor, visited.copy()
            )
            values.append(_discounted_edge_value(immediate, discount_factor, continuation))

        return float(np.mean(values)) if values else 0

    def _assess_timing_sensitivity(
        self, stage_analyses: List[StageAnalysis], value_of_flexibility: float
    ) -> str:
        """Assess how sensitive results are to timing."""
        if not stage_analyses:
            return "low"

        # Get total value from first stage
        first_stage_options = stage_analyses[0].options_at_stage if stage_analyses else []

        if not first_stage_options:
            return "low"

        # Calculate relative flexibility value
        best_value = max(opt.total_value for opt in first_stage_options)

        if best_value != 0:
            relative_flexibility = value_of_flexibility / abs(best_value)
        else:
            relative_flexibility = 0

        if relative_flexibility > 0.3:
            return "high"
        elif relative_flexibility > 0.1:
            return "medium"
        else:
            return "low"

    def _build_tree_node(
        self,
        node_id: str,
        graph_data: Dict[str, Any],
        stage_assignments: Dict[str, int],
        node_values: Dict[str, float],
        optimal_actions: Dict[str, str],
        visited: Set[str],
    ) -> PolicyTreeNode:
        """Recursively build a policy tree node."""
        if node_id in visited:
            # Prevent infinite loops
            return PolicyTreeNode(
                node_id=node_id,
                stage=stage_assignments.get(node_id, 0),
                node_type="terminal",
                label="(cyclic reference)",
                optimal_action=None,
                expected_value=0,
                children=[],
            )

        visited.add(node_id)

        nodes = graph_data["nodes"]
        edges = graph_data["edges"]

        if node_id not in nodes:
            return PolicyTreeNode(
                node_id=node_id,
                stage=0,
                node_type="terminal",
                label=node_id,
                optimal_action=None,
                expected_value=0,
                children=[],
            )

        node = nodes[node_id]
        stage = stage_assignments.get(node_id, 0)
        node_type = node["type"]
        label = node.get("label", node_id)
        optimal_action = optimal_actions.get(node_id)
        # TODO(A3 F4:990, pre-mount hardening): this is the absent-as-0 fabrication
        # class — a node not valued by backward induction is reported with
        # expected_value 0 rather than failing loud. The live paths (_build_policy
        # :575-585, _generate_stage_analyses) direct-index node_values[...] and let
        # a missing value raise. This method feeds ONLY the policy-tree route,
        # which is DARK (not mounted; see main.py selective-mount note), so the
        # blast radius is 0 today. Left as .get(...,0) deliberately: the route's
        # fallback root (:164-166) can select an unvalued node, so a naive
        # direct-index would raise on a legitimate-but-unmounted request; a proper
        # fix (guarantee the chosen root is valued, then direct-index + fail-loud)
        # is deferred to when policy-tree is runtime-verified and mounted.
        expected_value = node_values.get(node_id, 0)

        # Build children
        children = []
        outgoing = edges.get(node_id, [])

        for edge in outgoing:
            child_id = edge["to"]
            action = edge.get("action")
            outcome = edge.get("outcome")
            probability = edge.get("probability")

            child_info = {
                "child_id": child_id,
            }

            if action:
                child_info["action"] = action
            if outcome:
                child_info["outcome"] = outcome
            if probability is not None:
                child_info["probability"] = probability

            children.append(child_info)

        return PolicyTreeNode(
            node_id=node_id,
            stage=stage,
            node_type=node_type,
            label=label,
            optimal_action=optimal_action,
            expected_value=expected_value,
            children=children,
        )

    def _count_tree_nodes(self, root: PolicyTreeNode) -> int:
        """Count total nodes in tree."""
        return 1 + len(root.children)  # Simplified - doesn't recurse into children

    def _auto_detect_parameters(self, graph_data: Dict[str, Any]) -> List[str]:
        """Auto-detect parameters to vary for sensitivity analysis."""
        parameters = []

        # Add probability parameters from chance edges
        for node_id, node in graph_data["nodes"].items():
            if node["type"] == "chance":
                for edge in graph_data["edges"].get(node_id, []):
                    if edge.get("probability") is not None:
                        param_name = f"{node_id}_{edge.get('outcome', 'outcome')}_prob"
                        parameters.append(param_name)

        # Add payoff parameters from terminal nodes
        for node_id, node in graph_data["nodes"].items():
            if node["type"] == "terminal" and node.get("payoff") is not None:
                parameters.append(f"{node_id}_payoff")

        return parameters[:10]  # Limit

    def _analyze_stage_sensitivity(
        self,
        graph_data: Dict[str, Any],
        stage: DecisionStage,
        all_stages: List[DecisionStage],
        parameters: List[str],
        variation_range: float,
        baseline_values: Dict[str, float],
        baseline_actions: Dict[str, str],
    ) -> StageSensitivityResult:
        """Analyze sensitivity for a single stage."""
        sensitivities = {}
        policy_changes = {}

        # Get baseline value at this stage's decision nodes
        baseline_stage_value: float = 0.0
        for node_id in stage.decision_nodes:
            if node_id in baseline_values:
                baseline_stage_value = baseline_values[node_id]
                break

        # Test each parameter
        for param in parameters:
            # Perturb parameter and re-run backward induction
            perturbed_graph = self._perturb_parameter(graph_data, param, variation_range)

            perturbed_values, perturbed_actions = self._backward_induction(
                perturbed_graph, all_stages, discount_factor=0.95, risk_tolerance="neutral"
            )

            # Calculate sensitivity
            perturbed_stage_value: float = 0.0
            for node_id in stage.decision_nodes:
                if node_id in perturbed_values:
                    perturbed_stage_value = perturbed_values[node_id]
                    break

            if baseline_stage_value != 0:
                value_change = abs(perturbed_stage_value - baseline_stage_value)
                sensitivity = value_change / abs(baseline_stage_value)
            else:
                sensitivity = 0

            sensitivities[param] = round(min(1.0, sensitivity), 3)

            # Check if policy changed
            for node_id in stage.decision_nodes:
                if baseline_actions.get(node_id) != perturbed_actions.get(node_id):
                    # Policy changed - record threshold
                    policy_changes[param] = 1.0 - variation_range

        # Calculate robustness score
        if sensitivities:
            avg_sensitivity = float(np.mean(list(sensitivities.values())))
            robustness = max(0.0, 1.0 - avg_sensitivity)
        else:
            robustness = 1.0

        return StageSensitivityResult(
            stage_index=stage.stage_index,
            stage_label=stage.stage_label,
            parameter_sensitivities=sensitivities,
            policy_changes_at=policy_changes if policy_changes else None,
            robustness_score=round(robustness, 3),
        )

    def _perturb_parameter(
        self, graph_data: Dict[str, Any], param: str, variation: float
    ) -> Dict[str, Any]:
        """Create perturbed copy of graph data."""
        import copy

        perturbed = copy.deepcopy(graph_data)

        # Parse parameter name to find what to perturb
        parts = param.split("_")

        if param.endswith("_payoff"):
            # Perturb terminal payoff
            node_id = param.replace("_payoff", "")
            if node_id in perturbed["nodes"]:
                original = perturbed["nodes"][node_id].get("payoff", 0) or 0
                perturbed["nodes"][node_id]["payoff"] = original * (1 - variation)

        elif "_prob" in param:
            # Perturb probability
            for node_id in perturbed["edges"]:
                for edge in perturbed["edges"][node_id]:
                    if edge.get("probability") is not None:
                        edge["probability"] *= 1 - variation

        return perturbed

    def _generate_sensitivity_reasoning(
        self, most_sensitive: List[str], sensitivities: Dict[str, float]
    ) -> str:
        """Generate reasoning text for sensitivity analysis."""
        if not most_sensitive:
            return "No significant parameter sensitivities detected."

        top_param = most_sensitive[0]
        top_sens = sensitivities.get(top_param, 0)

        if top_sens > 0.5:
            return f"{top_param} has high sensitivity ({top_sens:.2f}) - small changes could significantly affect optimal decisions."
        elif top_sens > 0.2:
            return f"{top_param} shows moderate sensitivity ({top_sens:.2f}) - changes may affect value but likely not optimal actions."
        else:
            return f"All parameters show low sensitivity - policy is robust to reasonable parameter variations."
