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
    PolicyDistribution,
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


class SequentialDecisionEngine:
    """
    Engine for solving sequential decision problems via backward induction.

    Supports multi-stage decisions where later choices depend on earlier outcomes.
    """

    def __init__(self):
        """Initialize the sequential decision engine."""
        self.logger = logger

    def analyze(
        self,
        request: SequentialAnalysisRequest
    ) -> SequentialAnalysisResponse:
        """
        Analyze a sequential decision problem using backward induction.

        Args:
            request: Request with graph, stages, and parameters

        Returns:
            SequentialAnalysisResponse with optimal policy and analysis
        """
        self.logger.info(
            f"Analyzing sequential decision with {len(request.stages)} stages"
        )

        # Build internal representation
        graph_data = self._build_graph_data(request.graph)

        # Run backward induction
        node_values, optimal_actions = self._backward_induction(
            graph_data,
            request.stages,
            request.discount_factor,
            request.risk_tolerance
        )

        # Build optimal policy
        policy = self._build_policy(
            graph_data,
            request.stages,
            node_values,
            optimal_actions
        )

        # Generate stage analyses
        stage_analyses = self._generate_stage_analyses(
            graph_data,
            request.stages,
            node_values,
            optimal_actions,
            request.discount_factor
        )

        # Calculate value of flexibility
        value_of_flexibility = self._compute_value_of_flexibility(
            graph_data,
            request.stages,
            node_values,
            request.discount_factor
        )

        # Determine sensitivity to timing
        sensitivity_to_timing = self._assess_timing_sensitivity(
            stage_analyses,
            value_of_flexibility
        )

        return SequentialAnalysisResponse(
            optimal_policy=policy,
            stage_analyses=stage_analyses,
            value_of_flexibility=value_of_flexibility,
            sensitivity_to_timing=sensitivity_to_timing
        )

    def get_policy_tree(
        self,
        request: SequentialAnalysisRequest
    ) -> PolicyTreeResponse:
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
            graph_data,
            request.stages,
            request.discount_factor,
            request.risk_tolerance
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
            visited=set()
        )

        # Count nodes
        total_nodes = self._count_tree_nodes(root)

        return PolicyTreeResponse(
            root=root,
            total_stages=len(request.stages),
            total_nodes=total_nodes
        )

    def stage_sensitivity(
        self,
        request: StageSensitivityRequest
    ) -> StageSensitivityResponse:
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
            graph_data,
            request.stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        # Identify parameters to vary
        parameters = request.parameters_to_vary or self._auto_detect_parameters(graph_data)

        # Analyze each stage
        stage_results = []
        all_sensitivities = {}

        for stage in request.stages:
            stage_result = self._analyze_stage_sensitivity(
                graph_data,
                stage,
                request.stages,
                parameters,
                request.variation_range,
                baseline_values,
                baseline_actions
            )
            stage_results.append(stage_result)

            # Collect all sensitivities
            for param, sens in stage_result.parameter_sensitivities.items():
                if param not in all_sensitivities:
                    all_sensitivities[param] = []
                all_sensitivities[param].append(sens)

        # Calculate overall robustness
        if stage_results:
            overall_robustness = np.mean([r.robustness_score for r in stage_results])
        else:
            overall_robustness = 1.0

        # Find most sensitive parameters
        avg_sensitivities = {
            param: np.mean(values)
            for param, values in all_sensitivities.items()
        }
        most_sensitive = sorted(
            avg_sensitivities.keys(),
            key=lambda x: avg_sensitivities[x],
            reverse=True
        )[:3]

        explanation = ExplanationMetadata(
            summary=f"Policy is {'robust' if overall_robustness > 0.7 else 'moderately robust' if overall_robustness > 0.4 else 'fragile'} to parameter changes",
            reasoning=self._generate_sensitivity_reasoning(most_sensitive, avg_sensitivities),
            technical_basis="One-at-a-time sensitivity analysis with backward induction",
            assumptions=["Parameters vary independently", "Linear approximation to sensitivity"]
        )

        return StageSensitivityResponse(
            stage_results=stage_results,
            overall_robustness=round(overall_robustness, 3),
            most_sensitive_parameters=most_sensitive,
            explanation=explanation
        )

    def _build_graph_data(self, graph: SequentialGraph) -> Dict[str, Any]:
        """Build internal graph representation."""
        nodes = {}
        edges = defaultdict(list)  # from_node -> list of edges
        incoming_edges = defaultdict(list)  # to_node -> list of edges

        for node in graph.nodes:
            nodes[node.id] = {
                "id": node.id,
                "type": node.type,
                "label": node.label,
                "payoff": node.payoff,
                "probabilities": node.probabilities
            }

        for edge in graph.edges:
            edge_data = {
                "from": edge.from_node,
                "to": edge.to_node,
                "action": edge.action,
                "outcome": edge.outcome,
                "probability": edge.probability,
                "immediate_payoff": edge.immediate_payoff or 0
            }
            edges[edge.from_node].append(edge_data)
            incoming_edges[edge.to_node].append(edge_data)

        return {
            "nodes": nodes,
            "edges": dict(edges),
            "incoming_edges": dict(incoming_edges),
            "stage_assignments": graph.stage_assignments
        }

    def _backward_induction(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        discount_factor: float,
        risk_tolerance: str
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
        node_values = {}
        optimal_actions = {}

        for node_id, node in nodes.items():
            if node["type"] == "terminal":
                payoff = node.get("payoff", 0) or 0
                node_values[node_id] = self._risk_adjust_value(
                    payoff, 0, risk_tolerance
                )

        # Sort stages in reverse order
        sorted_stages = sorted(stages, key=lambda s: s.stage_index, reverse=True)

        # Process stages from end to beginning
        for stage in sorted_stages:
            # Get all nodes at this stage
            stage_nodes = [
                node_id for node_id, s in stage_assignments.items()
                if s == stage.stage_index
            ]

            for node_id in stage_nodes:
                if node_id in node_values:
                    continue  # Already processed (terminal)

                node = nodes[node_id]
                outgoing = edges.get(node_id, [])

                if not outgoing:
                    # No outgoing edges - treat as terminal with 0 payoff
                    node_values[node_id] = 0
                    continue

                if node["type"] == "decision":
                    # Decision node: maximize over actions
                    best_value = float('-inf')
                    best_action = None

                    for edge in outgoing:
                        child_id = edge["to"]
                        immediate = edge.get("immediate_payoff", 0) or 0

                        if child_id in node_values:
                            continuation = node_values[child_id]
                        else:
                            continuation = 0

                        total = immediate + discount_factor * continuation

                        if total > best_value:
                            best_value = total
                            best_action = edge.get("action", child_id)

                    node_values[node_id] = best_value
                    optimal_actions[node_id] = best_action

                elif node["type"] == "chance":
                    # Chance node: expected value over outcomes
                    expected_value = 0
                    total_prob = 0

                    for edge in outgoing:
                        child_id = edge["to"]
                        prob = edge.get("probability", 1.0 / len(outgoing))
                        immediate = edge.get("immediate_payoff", 0) or 0

                        if child_id in node_values:
                            continuation = node_values[child_id]
                        else:
                            continuation = 0

                        expected_value += prob * (immediate + discount_factor * continuation)
                        total_prob += prob

                    # Normalize if probabilities don't sum to 1
                    if total_prob > 0 and abs(total_prob - 1.0) > 0.01:
                        expected_value /= total_prob

                    # Apply risk adjustment
                    variance = self._estimate_outcome_variance(
                        outgoing, node_values, discount_factor
                    )
                    node_values[node_id] = self._risk_adjust_value(
                        expected_value, variance, risk_tolerance
                    )

        return node_values, optimal_actions

    def _risk_adjust_value(
        self,
        mean: float,
        variance: float,
        risk_tolerance: str
    ) -> float:
        """Apply risk adjustment to expected value."""
        if risk_tolerance == "neutral" or variance == 0:
            return mean
        elif risk_tolerance == "averse":
            # Mean-variance with risk aversion coefficient
            return mean - 0.5 * variance
        elif risk_tolerance == "seeking":
            # Risk-seeking: slight bonus for variance
            return mean + 0.1 * np.sqrt(variance) if variance > 0 else mean
        return mean

    def _estimate_outcome_variance(
        self,
        outgoing_edges: List[Dict],
        node_values: Dict[str, float],
        discount_factor: float
    ) -> float:
        """Estimate variance of outcomes from a chance node."""
        if not outgoing_edges:
            return 0

        values = []
        probs = []

        for edge in outgoing_edges:
            child_id = edge["to"]
            prob = edge.get("probability", 1.0 / len(outgoing_edges))
            immediate = edge.get("immediate_payoff", 0) or 0

            if child_id in node_values:
                value = immediate + discount_factor * node_values[child_id]
            else:
                value = immediate

            values.append(value)
            probs.append(prob)

        # Normalize probabilities
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]

        # Calculate variance
        mean = sum(p * v for p, v in zip(probs, values))
        variance = sum(p * (v - mean) ** 2 for p, v in zip(probs, values))

        return variance

    def _build_policy(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        node_values: Dict[str, float],
        optimal_actions: Dict[str, str]
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

                # Get optimal action
                default_action = optimal_actions.get(node_id, "none")

                # Build conditional actions from edges
                conditional_actions = []
                outgoing = edges.get(node_id, [])

                for edge in outgoing:
                    action = edge.get("action", edge["to"])
                    child_id = edge["to"]

                    if child_id in node_values:
                        ev = node_values[child_id]
                    else:
                        ev = edge.get("immediate_payoff", 0) or 0

                    # Add as conditional action if not default
                    if action != default_action:
                        # Generate a condition based on context
                        condition = self._generate_condition_string(
                            edge, graph_data
                        )

                        conditional_actions.append(ConditionalAction(
                            condition=condition,
                            action=action,
                            expected_value_if_taken=ev
                        ))

                decision_rule = DecisionRule(
                    default_action=default_action,
                    conditional_actions=conditional_actions
                )

                # Determine what this stage is contingent on
                contingent_on = stage.resolution_nodes or []

                stage_policies.append(StagePolicy(
                    stage_index=stage.stage_index,
                    stage_label=stage.stage_label,
                    decision_rule=decision_rule,
                    contingent_on=contingent_on
                ))

                break  # One policy per stage for simplicity

        # Calculate expected total value
        root_value = self._get_root_value(graph_data, node_values)

        # Estimate value distribution
        value_std = abs(root_value) * 0.2  # Simplified estimate

        return Policy(
            stages=stage_policies,
            expected_total_value=root_value,
            value_distribution=PolicyDistribution(
                type="normal",
                parameters={"mean": root_value, "std": value_std}
            )
        )

    def _generate_condition_string(
        self,
        edge: Dict,
        graph_data: Dict
    ) -> str:
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

    def _get_root_value(
        self,
        graph_data: Dict[str, Any],
        node_values: Dict[str, float]
    ) -> float:
        """Get value at root node (earliest stage decision)."""
        stage_assignments = graph_data["stage_assignments"]
        nodes = graph_data["nodes"]

        # Find decision node at stage 0
        for node_id, stage in stage_assignments.items():
            if stage == 0 and nodes[node_id]["type"] == "decision":
                return node_values.get(node_id, 0)

        # Fallback to any stage 0 node
        for node_id, stage in stage_assignments.items():
            if stage == 0:
                return node_values.get(node_id, 0)

        # Last resort
        if node_values:
            return max(node_values.values())
        return 0

    def _generate_stage_analyses(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        node_values: Dict[str, float],
        optimal_actions: Dict[str, str],
        discount_factor: float
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

                # Analyze each available action
                outgoing = edges.get(node_id, [])

                for edge in outgoing:
                    action = edge.get("action", edge["to"])
                    child_id = edge["to"]
                    immediate = edge.get("immediate_payoff", 0) or 0

                    continuation = node_values.get(child_id, 0)
                    total = immediate + discount_factor * continuation

                    options.append(StageOption(
                        option_id=action,
                        label=action.replace("_", " ").title(),
                        immediate_value=immediate,
                        continuation_value=continuation,
                        total_value=total
                    ))

            # Calculate information value
            info_value = self._calculate_information_value(
                stage, graph_data, node_values, discount_factor
            )

            # Calculate waiting value if applicable
            waiting_value = None
            if stage.stage_index == 0 and len(stages) > 1:
                waiting_value = self._calculate_waiting_value(
                    graph_data, stages, node_values, discount_factor
                )

            analyses.append(StageAnalysis(
                stage_index=stage.stage_index,
                stage_label=stage.stage_label,
                options_at_stage=options,
                information_value=info_value,
                optimal_waiting_value=waiting_value
            ))

        return analyses

    def _calculate_information_value(
        self,
        stage: DecisionStage,
        graph_data: Dict[str, Any],
        node_values: Dict[str, float],
        discount_factor: float
    ) -> float:
        """
        Calculate value of information revealed at this stage.

        This is the difference between value with and without information.
        """
        # Get chance nodes that resolve at this stage
        resolution_nodes = stage.resolution_nodes
        if not resolution_nodes:
            return 0

        # Simplified: estimate as variance reduction
        total_variance = 0
        for node_id in resolution_nodes:
            if node_id in graph_data["nodes"]:
                node = graph_data["nodes"][node_id]
                if node["type"] == "chance":
                    outgoing = graph_data["edges"].get(node_id, [])
                    variance = self._estimate_outcome_variance(
                        outgoing, node_values, discount_factor
                    )
                    total_variance += variance

        # Value of information is roughly sqrt of variance reduction
        return np.sqrt(total_variance) if total_variance > 0 else 0

    def _calculate_waiting_value(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        node_values: Dict[str, float],
        discount_factor: float
    ) -> float:
        """Calculate value of waiting/delaying first stage decision."""
        # Simplified: value of waiting is related to information gained
        # by observing chance node outcomes before committing

        if len(stages) < 2:
            return 0

        # Get value at stage 0 without waiting
        root_value = self._get_root_value(graph_data, node_values)

        # Estimate value with perfect information (upper bound)
        # This would require resolving all uncertainty first
        info_value_stage1 = self._calculate_information_value(
            stages[1] if len(stages) > 1 else stages[0],
            graph_data,
            node_values,
            discount_factor
        )

        # Waiting value is discounted information value
        waiting_value = discount_factor * info_value_stage1

        return waiting_value

    def _compute_value_of_flexibility(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        node_values: Dict[str, float],
        discount_factor: float
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
        v_committed = self._calculate_committed_value(
            graph_data, stages, discount_factor
        )

        value_of_flexibility = max(0, v_flexible - v_committed)

        return round(value_of_flexibility, 2)

    def _calculate_committed_value(
        self,
        graph_data: Dict[str, Any],
        stages: List[DecisionStage],
        discount_factor: float
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
        best_committed_value = float('-inf')

        for edge in outgoing:
            immediate = edge.get("immediate_payoff", 0) or 0

            # Calculate expected continuation without optimal future decisions
            # (use average outcomes instead of max)
            continuation = self._calculate_average_continuation(
                edge["to"], graph_data, discount_factor, visited=set()
            )

            total = immediate + discount_factor * continuation

            if total > best_committed_value:
                best_committed_value = total

        return best_committed_value if best_committed_value > float('-inf') else 0

    def _calculate_average_continuation(
        self,
        node_id: str,
        graph_data: Dict[str, Any],
        discount_factor: float,
        visited: Set[str]
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
            values.append(immediate + discount_factor * continuation)

        return np.mean(values) if values else 0

    def _assess_timing_sensitivity(
        self,
        stage_analyses: List[StageAnalysis],
        value_of_flexibility: float
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
        visited: Set[str]
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
                children=[]
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
                children=[]
            )

        node = nodes[node_id]
        stage = stage_assignments.get(node_id, 0)
        node_type = node["type"]
        label = node.get("label", node_id)
        optimal_action = optimal_actions.get(node_id)
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
            children=children
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
        baseline_actions: Dict[str, str]
    ) -> StageSensitivityResult:
        """Analyze sensitivity for a single stage."""
        sensitivities = {}
        policy_changes = {}

        # Get baseline value at this stage's decision nodes
        baseline_stage_value = 0
        for node_id in stage.decision_nodes:
            if node_id in baseline_values:
                baseline_stage_value = baseline_values[node_id]
                break

        # Test each parameter
        for param in parameters:
            # Perturb parameter and re-run backward induction
            perturbed_graph = self._perturb_parameter(
                graph_data, param, variation_range
            )

            perturbed_values, perturbed_actions = self._backward_induction(
                perturbed_graph,
                all_stages,
                discount_factor=0.95,
                risk_tolerance="neutral"
            )

            # Calculate sensitivity
            perturbed_stage_value = 0
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
            avg_sensitivity = np.mean(list(sensitivities.values()))
            robustness = max(0, 1.0 - avg_sensitivity)
        else:
            robustness = 1.0

        return StageSensitivityResult(
            stage_index=stage.stage_index,
            stage_label=stage.stage_label,
            parameter_sensitivities=sensitivities,
            policy_changes_at=policy_changes if policy_changes else None,
            robustness_score=round(robustness, 3)
        )

    def _perturb_parameter(
        self,
        graph_data: Dict[str, Any],
        param: str,
        variation: float
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
                        edge["probability"] *= (1 - variation)

        return perturbed

    def _generate_sensitivity_reasoning(
        self,
        most_sensitive: List[str],
        sensitivities: Dict[str, float]
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
