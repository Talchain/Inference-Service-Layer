"""
FACET-based robustness analyzer with dual uncertainty support (v2.3).

Implements Monte Carlo robustness analysis that samples both:
- Structural uncertainty: Edge existence (Bernoulli)
- Parametric uncertainty: Effect magnitude (Normal)

This enables answering:
- "Is my decision robust to uncertainty about whether this relationship exists?"
- "Is my decision robust to the effect being stronger/weaker than estimated?"
- "If an edge is weaker than modelled, which alternative option would win?"
"""

import hashlib
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models.robustness_v2 import (
    BucketResult,
    ClampMetrics,
    ConditionalWinner,
    ConstraintAnalysis,
    ConstraintResult,
    EdgeV2,
    FactorSensitivityResult,
    FragileEdgeEnhanced,
    GoalConstraint,
    GraphV2,
    InferenceWarning,
    InterventionOption,
    NodeV2,
    OptionResult,
    OutcomeDistribution,
    ParameterUncertainty,
    ResponseMetadataV2,
    RobustnessRequestV2,
    RobustnessResponseV2,
    RobustnessResult,
    SensitivityResult,
    StabilityThresholdsResponse,
)
from src.models.response_v2 import ZeroSensitivityReason
from src.constants import (
    ELASTICITY_CLAMP_MAX,
    FACTOR_SENSITIVITY_BASELINE_EPSILON,
    FACTOR_SENSITIVITY_VALUE_EPSILON,
    ZERO_VARIANCE_TOLERANCE,
)
from src.models.critique import (
    CONSTRAINT_NODE_DEFAULT_BASE,
    DEGENERATE_OPTION_ZERO_VARIANCE,
    HIGH_TIE_RATE,
)
from src.models.response_v2 import CritiqueV2
from src.utils.rng import SEED_HASH_VERSION, SeededRNG, compute_seed_from_graph
from src.__version__ import __version__
from src.models.metadata import generate_config_fingerprint
from src.config.stability_thresholds import (
    STABILITY_THRESHOLDS,
    classify_attribution_stability,
)

logger = logging.getLogger(__name__)

# Safety net: nodes that must not participate in inference
NON_INFERENCE_KINDS = {"decision", "option", "constraint"}

# Edge strength bounds from schema v2.6
EDGE_STRENGTH_MIN = -1.0
EDGE_STRENGTH_MAX = 1.0

# Default samples for marginal switch probability calculation
MARGINAL_K_SAMPLES = 100


def filter_inference_graph(graph: GraphV2) -> GraphV2:
    """Filter out non-inference nodes and incident edges as a safety net."""
    filtered_nodes = [node for node in graph.nodes if node.kind.lower() not in NON_INFERENCE_KINDS]
    removed_nodes = len(graph.nodes) - len(filtered_nodes)

    if removed_nodes == 0:
        return graph

    kept_node_ids = {node.id for node in filtered_nodes}
    filtered_edges = [
        edge for edge in graph.edges if edge.from_ in kept_node_ids and edge.to in kept_node_ids
    ]
    removed_edges = len(graph.edges) - len(filtered_edges)
    removed_node_ids = [node.id for node in graph.nodes if node.kind.lower() in NON_INFERENCE_KINDS]

    logger.warning(
        "robustness_v2_filtered_non_inference_nodes",
        extra={
            "removed_node_count": removed_nodes,
            "removed_edge_count": removed_edges,
            "removed_node_ids": removed_node_ids,
            "remaining_node_count": len(filtered_nodes),
            "remaining_edge_count": len(filtered_edges),
        },
    )

    return GraphV2(nodes=filtered_nodes, edges=filtered_edges)


# =============================================================================
# Fragile Edge with Alternative Winner
# =============================================================================


@dataclass
class FragileEdge:
    """Internal representation of a fragile edge with alternative winner analysis."""

    edge_id: str  # "from->to" format
    from_id: str
    to_id: str
    alternative_winner_id: Optional[str] = None  # Option that wins when edge is weak
    switch_probability: Optional[float] = None  # P(alternative wins | edge weak)
    marginal_switch_probability: Optional[float] = None  # P(flip | only this edge varies)


# =============================================================================
# Dual Uncertainty Sampler
# =============================================================================


class DualUncertaintySampler:
    """
    Samples edge configurations with structural + parametric uncertainty.

    For each edge:
    1. Sample existence from Bernoulli(exists_probability)
    2. If exists, sample strength from Normal(mean, std)
    3. If not exists, effective_strength = 0

    This enables Monte Carlo integration over both uncertainty dimensions.
    """

    def __init__(self, edges: List[EdgeV2], rng: SeededRNG):
        """
        Initialize sampler.

        Args:
            edges: List of edges with dual uncertainty
            rng: Seeded random number generator
        """
        self.edges = edges
        self.rng = rng
        self._existence_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._sample_count = 0

    def sample_edge_configuration(self) -> Dict[Tuple[str, str], float]:
        """
        Sample one edge configuration.

        Returns:
            Dict mapping (from, to) -> effective_strength
            If edge doesn't exist in this sample, strength = 0
        """
        config = {}
        self._sample_count += 1

        for edge in self.edges:
            edge_key = (edge.from_, edge.to)

            # Structural uncertainty: does edge exist?
            if self.rng.bernoulli(edge.exists_probability):
                # Parametric uncertainty: what's the effect size?
                strength = self.rng.normal(edge.strength.mean, edge.strength.std)
                # Clip to schema bounds [-1, 1] for consistency with marginal analysis.
                # This is a clipped normal, not a true truncated normal — it shifts
                # effective mean and reduces variance at distribution tails. Acceptable
                # for PoC; truncated normal sampling is a post-pilot refinement.
                strength = np.clip(strength, EDGE_STRENGTH_MIN, EDGE_STRENGTH_MAX)
                config[edge_key] = strength
                self._existence_counts[edge_key] += 1
            else:
                # Edge doesn't exist in this sample
                config[edge_key] = 0.0

        return config

    def sample_n_configurations(self, n: int) -> List[Dict[Tuple[str, str], float]]:
        """
        Sample n independent edge configurations.

        Args:
            n: Number of configurations to sample

        Returns:
            List of edge configuration dictionaries
        """
        return [self.sample_edge_configuration() for _ in range(n)]

    def get_existence_rates(self) -> Dict[str, float]:
        """
        Get actual existence rates from sampling.

        Returns:
            Dict mapping "from->to" -> observed existence rate
        """
        if self._sample_count == 0:
            return {}

        return {
            f"{edge.from_}->{edge.to}": (
                self._existence_counts[(edge.from_, edge.to)] / self._sample_count
            )
            for edge in self.edges
        }


# =============================================================================
# Factor Sampler (Phase 2A Part 2)
# =============================================================================


class FactorSampler:
    """
    Samples factor node values with parameter uncertainty.

    For each factor with specified uncertainty:
    1. Get mean from node's observed_state.value
    2. Sample from distribution (normal, uniform, or point_mass)

    This enables Monte Carlo integration over factor value uncertainty,
    complementing edge uncertainty (structural + magnitude).
    """

    def __init__(
        self,
        nodes: List[NodeV2],
        uncertainties: Optional[List[ParameterUncertainty]],
        rng: SeededRNG,
    ):
        """
        Initialize factor sampler.

        Args:
            nodes: List of graph nodes (may include observed_state)
            uncertainties: List of factor uncertainty specifications
            rng: Seeded random number generator
        """
        self.rng = rng
        self._node_map: Dict[str, NodeV2] = {n.id: n for n in nodes}
        self._uncertainty_map: Dict[str, ParameterUncertainty] = {
            u.node_id: u for u in (uncertainties or [])
        }
        self._sample_count = 0
        self._value_sums: Dict[str, float] = defaultdict(float)

    def sample_factor_values(self) -> Dict[str, float]:
        """
        Sample factor values for one Monte Carlo iteration.

        Returns:
            Dict mapping node_id -> sampled value for all factor nodes
            with specified uncertainty.
        """
        self._sample_count += 1
        factor_values: Dict[str, float] = {}

        for node_id, uncertainty in self._uncertainty_map.items():
            node = self._node_map.get(node_id)
            if not node:
                # Node doesn't exist - skip (should have been caught by validation)
                continue

            # Get mean from observed_state.value, default to 0
            mean = 0.0
            if node.observed_state and node.observed_state.value is not None:
                mean = node.observed_state.value

            # Sample from specified distribution
            sampled_value = self._sample_from_distribution(uncertainty, mean, node_id)
            factor_values[node_id] = sampled_value
            self._value_sums[node_id] += sampled_value

        return factor_values

    def _sample_from_distribution(
        self, uncertainty: ParameterUncertainty, mean: float, node_id: str
    ) -> float:
        """
        Sample a value from the specified distribution.

        Args:
            uncertainty: Distribution specification
            mean: Mean value (from observed_state.value)
            node_id: Node ID for error messages

        Returns:
            Sampled value
        """
        dist = uncertainty.distribution

        if dist == "point_mass":
            # No sampling - use observed value directly
            return mean

        elif dist == "normal":
            # Sample from Normal(mean, std)
            std = uncertainty.std or 0.0
            return self.rng.normal(mean, std)

        elif dist == "uniform":
            # Sample uniformly from [range_min, range_max]
            range_min = uncertainty.range_min
            range_max = uncertainty.range_max
            if range_min is None or range_max is None:
                raise ValueError(
                    f"Uniform distribution for node '{node_id}' requires range_min and range_max"
                )
            return self.rng.uniform(range_min, range_max)

        else:
            # Unknown distribution - fail fast instead of silent fallback
            raise ValueError(
                f"Unknown distribution '{dist}' for node '{node_id}'. "
                f"Supported: point_mass, normal, uniform"
            )

    def has_uncertainties(self) -> bool:
        """Check if any factor uncertainties are specified."""
        return len(self._uncertainty_map) > 0

    def get_mean_sampled_values(self) -> Dict[str, float]:
        """
        Get mean values from sampling for diagnostics.

        Returns:
            Dict mapping node_id -> mean sampled value
        """
        if self._sample_count == 0:
            return {}

        return {node_id: total / self._sample_count for node_id, total in self._value_sums.items()}


# =============================================================================
# SCM Evaluator
# =============================================================================


class SCMEvaluatorV2:
    """
    Evaluates structural causal model outcomes given edge configuration.

    Implements a simplified linear SCM evaluation where:
    - Node value = sum of (parent_value * edge_strength) for all incoming edges
    - Intervention nodes have fixed values
    - Evaluation follows topological order

    For more complex models, this could integrate with a full SCM engine.

    Note: Nodes may contain `observed_state` with actual factor values from
    CEE extraction. Phase 2A Part 2 will use these for:
    - Anchoring factor distributions to observed values
    - Computing realistic outcome distributions
    - Providing value-aware robustness analysis
    """

    def __init__(self, graph: GraphV2):
        """
        Initialize evaluator.

        Args:
            graph: Causal graph structure (nodes may include observed_state)
        """
        self.graph = graph
        self._node_order = self._compute_topological_order()
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, List[str]] = defaultdict(list)
        # Build node lookup for quick access to observed_state
        # Phase 2A Part 2: Will use this for factor value sampling
        self._nodes_by_id: Dict[str, NodeV2] = {node.id: node for node in graph.nodes}

        for edge in graph.edges:
            self._children[edge.from_].append(edge.to)
            self._parents[edge.to].append(edge.from_)

    def _compute_topological_order(self) -> List[str]:
        """Compute topological order of nodes for evaluation."""
        # Build adjacency list
        in_degree = {node.id: 0 for node in self.graph.nodes}
        adj = defaultdict(list)

        for edge in self.graph.edges:
            adj[edge.from_].append(edge.to)
            in_degree[edge.to] += 1

        # Kahn's algorithm — use deque for O(1) popleft (list.pop(0) is O(n))
        queue: deque = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)

            for child in adj[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.graph.nodes):
            # Defence-in-depth: this branch should NEVER execute in production.
            # Cycles are blocked upstream by RequestValidator.validate() which
            # calls detect_graph_cycle() and returns a blocker critique, preventing
            # analysis entirely.  This fallback exists purely as a safety net in
            # case the validator is bypassed (e.g. direct internal calls or future
            # refactors that skip validation).
            # If this warning fires in production, investigate why RequestValidator
            # did not catch the cycle before analysis reached this point.
            logger.warning("Graph has cycles, using arbitrary node order")
            return [n.id for n in self.graph.nodes]

        return order

    def evaluate(
        self,
        edge_strengths: Dict[Tuple[str, str], float],
        interventions: Dict[str, float],
        goal_node: str,
        base_values: Optional[Dict[str, float]] = None,
        factor_values: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Evaluate outcome under given edge configuration and interventions.

        Uses a linear additive model:
        node_value = base_value + sum(parent_value * edge_strength)

        Args:
            edge_strengths: Sampled edge strengths (0 if edge doesn't exist)
            interventions: Node interventions (do(X=x))
            goal_node: Target outcome node
            base_values: Optional base values for nodes (default: 0)
            factor_values: Optional sampled factor values (overrides observed_state.value)

        Returns:
            Value at goal_node

        Phase 2A Part 2 (ACTIVE):
            Root factor nodes use observed_state.value as their base value.
            If factor_values is provided, those take precedence (for sampling).
        """
        if base_values is None:
            base_values = {}
        if factor_values is None:
            factor_values = {}

        node_values: Dict[str, float] = {}

        for node_id in self._node_order:
            if node_id in interventions:
                # Interventional value overrides structural equations
                node_values[node_id] = interventions[node_id]
            else:
                # Get node object (used for observed_state and intercept)
                node = self._nodes_by_id.get(node_id)

                # Determine base value for this node
                # Priority: factor_values > observed_state.value > base_values > 0
                if node_id in factor_values:
                    # Sampled factor value takes highest priority
                    base = factor_values[node_id]
                elif node_id in base_values:
                    # Explicitly provided base value
                    base = base_values[node_id]
                else:
                    # Check for observed_state.value on root nodes
                    is_root = len(self._parents.get(node_id, [])) == 0
                    if (
                        is_root
                        and node
                        and node.observed_state
                        and node.observed_state.value is not None
                    ):
                        base = node.observed_state.value
                    else:
                        base = 0.0

                # Compute contribution from parents
                parents_contribution = 0.0
                for parent in self._parents[node_id]:
                    edge_key = (parent, node_id)
                    strength = edge_strengths.get(edge_key, 0.0)
                    parent_value = node_values.get(parent, 0.0)
                    parents_contribution += parent_value * strength

                # Get node intercept (default 0.0 if not set)
                intercept = getattr(node, "intercept", 0.0) if node else 0.0

                node_values[node_id] = base + intercept + parents_contribution

        return node_values.get(goal_node, 0.0)

    def evaluate_multi(
        self,
        edge_strengths: Dict[Tuple[str, str], float],
        interventions: Dict[str, float],
        target_nodes: List[str],
        base_values: Optional[Dict[str, float]] = None,
        factor_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate and return values for multiple target nodes.

        Same computational model as evaluate(), but returns a dict of values
        for the specified target nodes instead of a single goal node value.

        Args:
            edge_strengths: Sampled edge strengths (0 if edge doesn't exist)
            interventions: Node interventions (do(X=x))
            target_nodes: List of node IDs to return values for
            base_values: Optional base values for nodes (default: 0)
            factor_values: Optional sampled factor values (overrides observed_state.value)

        Returns:
            Dict mapping target_node_id -> computed value
        """
        if base_values is None:
            base_values = {}
        if factor_values is None:
            factor_values = {}

        node_values: Dict[str, float] = {}

        for node_id in self._node_order:
            if node_id in interventions:
                # Interventional value overrides structural equations
                node_values[node_id] = interventions[node_id]
            else:
                # Get node object (used for observed_state and intercept)
                node = self._nodes_by_id.get(node_id)

                # Determine base value for this node
                # Priority: factor_values > observed_state.value > base_values > 0
                if node_id in factor_values:
                    base = factor_values[node_id]
                elif node_id in base_values:
                    base = base_values[node_id]
                else:
                    is_root = len(self._parents.get(node_id, [])) == 0
                    if (
                        is_root
                        and node
                        and node.observed_state
                        and node.observed_state.value is not None
                    ):
                        base = node.observed_state.value
                    else:
                        base = 0.0

                # Compute contribution from parents
                parents_contribution = 0.0
                for parent in self._parents[node_id]:
                    edge_key = (parent, node_id)
                    strength = edge_strengths.get(edge_key, 0.0)
                    parent_value = node_values.get(parent, 0.0)
                    parents_contribution += parent_value * strength

                # Get node intercept (default 0.0 if not set)
                intercept = getattr(node, "intercept", 0.0) if node else 0.0

                node_values[node_id] = base + intercept + parents_contribution

        # Return only the requested target nodes
        return {node_id: node_values.get(node_id, 0.0) for node_id in target_nodes}


# =============================================================================
# Robustness Analyzer V2
# =============================================================================


class RobustnessAnalyzerV2:
    """
    Robustness analysis with dual uncertainty (v2.2).

    For each Monte Carlo sample:
    1. Sample edge configuration (existence + strength)
    2. Evaluate each option's outcome given that configuration
    3. Track which option wins

    Aggregates across samples to compute:
    - Outcome distributions per option
    - Win probabilities
    - Sensitivity to edge existence vs magnitude
    - Overall robustness assessment
    """

    # Thresholds for robustness assessment
    ROBUST_THRESHOLD = 0.7  # Win probability for "robust" recommendation
    FRAGILE_THRESHOLD = 0.1  # Elasticity threshold for fragile edges
    HIGH_SENSITIVITY_THRESHOLD = 0.2  # Elasticity for "high sensitivity"

    def __init__(self):
        """Initialize analyzer."""
        self.logger = logger
        # Test-only override: set to an int to fix bootstrap count and skip
        # adaptive timing. None = use adaptive budget (production default).
        self._n_bootstrap_override: Optional[int] = None

    def analyze(self, request: RobustnessRequestV2) -> RobustnessResponseV2:
        """
        Perform complete robustness analysis.

        Args:
            request: v2.2 robustness request

        Returns:
            Complete analysis response
        """
        start_time = time.time()

        # Generate request_id if not provided
        request_id = request.request_id or f"robustness-{uuid.uuid4().hex[:12]}"

        # Safety net: remove non-inference nodes/edges before analysis
        filtered_graph = filter_inference_graph(request.graph)
        if filtered_graph is not request.graph:
            request = request.model_copy(update={"graph": filtered_graph})

            # Post-filter validation: ensure goal node still exists
            filtered_node_ids = {node.id for node in filtered_graph.nodes}
            if request.goal_node_id not in filtered_node_ids:
                raise ValueError(
                    f"Goal node '{request.goal_node_id}' was filtered out as non-inference node. "
                    f"Goal nodes must be of kind: factor, chance, outcome, or risk."
                )

            # Post-filter validation: warn if intervention nodes were filtered
            for option in request.options:
                missing_interventions = [
                    node_id
                    for node_id in option.interventions.keys()
                    if node_id not in filtered_node_ids
                ]
                if missing_interventions:
                    self.logger.warning(
                        "robustness_v2_interventions_filtered",
                        extra={
                            "request_id": request_id,
                            "option_id": option.id,
                            "filtered_intervention_nodes": missing_interventions,
                        },
                    )

        # Setup - use separate RNG streams for edge and factor sampling
        # to prevent fragile determinism coupling
        # Use explicit None check: seed=0 is a valid explicit seed and must not
        # be treated as falsy (the old `or` expression would discard it).
        seed = request.seed if request.seed is not None else compute_seed_from_graph(request.graph)
        rng_edge = SeededRNG(seed)
        rng_factor = SeededRNG(seed + 1)
        sampler = DualUncertaintySampler(request.graph.edges, rng_edge)
        factor_sampler = FactorSampler(
            request.graph.nodes, request.parameter_uncertainties, rng_factor
        )
        evaluator = SCMEvaluatorV2(request.graph)

        self.logger.info(
            "robustness_v2_analysis_started",
            extra={
                "request_id": request_id,
                "n_samples": request.n_samples,
                "n_options": len(request.options),
                "n_edges": len(request.graph.edges),
                "n_factor_uncertainties": len(request.parameter_uncertainties or []),
                "seed": seed,
            },
        )

        # Determine constraint target nodes for multi-constraint analysis
        constraint_target_nodes: Optional[List[str]] = None
        if request.goal_constraints:
            constraint_target_nodes = sorted(set(gc.node_id for gc in request.goal_constraints))

        # Collect parse-time warnings (e.g. STRENGTH_MEAN_CLAMPED) from graph construction.
        # These are generated during Pydantic model validation of EdgeV2.strength.mean.
        inference_warnings: List[InferenceWarning] = request.graph.collect_parse_warnings()

        # Detect constraint target nodes that will silently default to base=0.0
        # (non-root nodes without ParameterUncertainty, not fully covered by
        # interventions across all options)
        constraint_default_base_critiques: List[CritiqueV2] = []
        if constraint_target_nodes:
            parent_map: dict[str, list[str]] = defaultdict(list)
            for edge in request.graph.edges:
                parent_map[edge.to].append(edge.from_)
            uncertainty_node_ids = set(u.node_id for u in (request.parameter_uncertainties or []))
            for node_id in constraint_target_nodes:
                is_root = len(parent_map.get(node_id, [])) == 0
                has_uncertainty = node_id in uncertainty_node_ids
                # Skip warning if every option intervenes on this node
                # (intervention value overrides the base, so base=0.0 is never used)
                all_options_intervene = all(node_id in opt.interventions for opt in request.options)
                if not is_root and not has_uncertainty and not all_options_intervene:
                    inference_warnings.append(
                        InferenceWarning(
                            code="CONSTRAINT_NODE_DEFAULT_BASE",
                            field=f"nodes[{node_id}].base",
                            detail={
                                "node_id": node_id,
                                "defaulted_to": 0.0,
                                "reason": "no_parameter_uncertainty",
                                "message": (
                                    f"Node '{node_id}' has no ParameterUncertainty "
                                    f"— defaulted to base=0.0, constraint probability "
                                    f"may be unreliable"
                                ),
                            },
                        )
                    )
                    constraint_default_base_critiques.append(
                        CONSTRAINT_NODE_DEFAULT_BASE.build(
                            node_id=node_id,
                            affected_node_ids=[node_id],
                        )
                    )
                    self.logger.warning(
                        "isl.constraint.default_base",
                        extra={
                            "node_id": node_id,
                            "defaulted_to": 0.0,
                            "reason": "no_parameter_uncertainty",
                        },
                    )

        # Run Monte Carlo simulation
        (
            option_outcomes,
            option_wins,
            winner_per_sample,
            edge_configs_per_sample,
            tie_count,
            constraint_node_values,
            factor_values_per_sample,
        ) = self._run_monte_carlo(
            request, sampler, factor_sampler, evaluator, constraint_target_nodes
        )

        # Compute tie rate
        tie_rate = tie_count / request.n_samples

        # Apply auto-scaled noise to outcome/risk nodes (V08 scientific accuracy)
        # Uses separate RNG stream (seed + 2) for determinism
        rng_noise = SeededRNG(seed + 2)
        option_outcomes = self._apply_auto_scaled_noise(
            option_outcomes,
            request.goal_node_id,
            request.graph.nodes,
            rng_noise,
        )

        # Compute results (including constraint analysis if goal_constraints provided)
        results = self._compute_option_results(
            option_outcomes, option_wins, request, constraint_node_values
        )

        # Build critiques for analysis warnings
        critiques: List[CritiqueV2] = []

        # Check for zero-variance options (degenerate outcomes)
        # Use tolerance to catch near-zero values from floating point arithmetic
        option_labels = {opt.id: (opt.label or opt.id) for opt in request.options}
        for result in results:
            if result.outcome_distribution.std < ZERO_VARIANCE_TOLERANCE:
                critiques.append(
                    DEGENERATE_OPTION_ZERO_VARIANCE.build(
                        option_label=option_labels.get(result.option_id, result.option_id),
                        affected_option_ids=[result.option_id],
                    )
                )

        # Check for high tie rate
        if tie_rate > 0.5:
            critiques.append(
                HIGH_TIE_RATE.build(
                    tie_rate_pct=int(tie_rate * 100),
                )
            )

        # Add constraint default-base critiques (also surfaced via inference_warnings)
        critiques.extend(constraint_default_base_critiques)

        # Compute sensitivity if requested
        sensitivity = []
        if "sensitivity" in request.analysis_types:
            sensitivity = self._compute_sensitivity(
                request, option_outcomes, sampler, rng_edge, evaluator
            )

        # Compute factor sensitivity if factor uncertainties are specified
        factor_sensitivity = []
        if factor_sampler.has_uncertainties() and "sensitivity" in request.analysis_types:
            factor_sensitivity = self._compute_factor_sensitivity(
                request, option_outcomes, rng_factor, evaluator
            )

        # Compute conditional winners (factor-partitioned win probabilities)
        conditional_winners = None
        if factor_sampler.has_uncertainties() and len(request.options) > 1:
            conditional_winners = self._compute_conditional_winners(
                factor_values_per_sample,
                winner_per_sample,
                option_outcomes,
                factor_sampler,
                request,
            )

        # Compute robustness assessment (with alternative winner analysis)
        robustness = self._compute_robustness(
            option_wins,
            winner_per_sample,
            sensitivity,
            request,
            edge_configs_per_sample,
            evaluator,
            seed,
        )

        execution_time = int((time.time() - start_time) * 1000)

        # Find recommended option
        recommended_option_id = max(option_wins, key=option_wins.get)
        recommendation_confidence = option_wins[recommended_option_id] / request.n_samples

        # Include stability thresholds when bootstrap stability was computed
        has_bootstrap = any(fs.attribution_stability is not None for fs in factor_sensitivity)
        stability_thresholds = (
            StabilityThresholdsResponse(
                high_moderate_boundary=STABILITY_THRESHOLDS.high_moderate_boundary,
                moderate_low_boundary=STABILITY_THRESHOLDS.moderate_low_boundary,
                version=STABILITY_THRESHOLDS.version,
                provisional=STABILITY_THRESHOLDS.provisional,
            )
            if has_bootstrap
            else None
        )

        response = RobustnessResponseV2(
            request_id=request_id,
            results=results,
            recommended_option_id=recommended_option_id,
            recommendation_confidence=recommendation_confidence,
            sensitivity=sensitivity,
            factor_sensitivity=factor_sensitivity,
            robustness=robustness,
            metadata=ResponseMetadataV2(
                isl_version=__version__,
                n_samples_used=request.n_samples,
                seed_used=seed,
                execution_time_ms=execution_time,
                edge_existence_rates=sampler.get_existence_rates(),
                config_fingerprint=generate_config_fingerprint(),
                tie_count=tie_count,
                tie_rate=tie_rate,
                seed_hash_version=SEED_HASH_VERSION,
            ),
            critiques=critiques,
            inference_warnings=inference_warnings,
            conditional_winners=conditional_winners,
            stability_thresholds=stability_thresholds,
        )

        self.logger.info(
            "robustness_v2_analysis_complete",
            extra={
                "request_id": request_id,
                "recommended_option": recommended_option_id,
                "recommendation_confidence": recommendation_confidence,
                "is_robust": robustness.is_robust,
                "execution_time_ms": execution_time,
                "conditional_winners_count": len(conditional_winners) if conditional_winners else 0,
            },
        )

        return response

    def _run_monte_carlo(
        self,
        request: RobustnessRequestV2,
        sampler: DualUncertaintySampler,
        factor_sampler: FactorSampler,
        evaluator: SCMEvaluatorV2,
        constraint_target_nodes: Optional[List[str]] = None,
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, float],
        List[str],
        List[Dict[Tuple[str, str], float]],
        int,
        Optional[Dict[str, Dict[str, List[float]]]],
        List[Dict[str, float]],
    ]:
        """
        Run Monte Carlo simulation with dual edge uncertainty and factor uncertainty.

        Args:
            request: The robustness analysis request
            sampler: Edge configuration sampler
            factor_sampler: Factor value sampler
            evaluator: SCM evaluator
            constraint_target_nodes: Optional list of node IDs to track for constraint analysis

        Returns:
            Tuple of:
            - option_outcomes: Dict[option_id, List[outcome_value]]
            - option_wins: Dict[option_id, win_count] (float for tie splitting)
            - winner_per_sample: List of winning option ID per sample
            - edge_configs_per_sample: Edge configurations per sample
            - tie_count: Number of samples with ties
            - constraint_node_values: Dict[option_id, Dict[node_id, List[value]]] or None
            - factor_values_per_sample: List of sampled factor value dicts per MC iteration

        Note: option_wins uses float to support split-tie handling where ties are
        divided equally among tied options.
        """
        option_outcomes: Dict[str, List[float]] = {opt.id: [] for opt in request.options}
        option_wins: Dict[str, float] = {opt.id: 0.0 for opt in request.options}
        winner_per_sample: List[str] = []
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]] = []
        factor_values_per_sample: List[Dict[str, float]] = []
        tie_count = 0

        # Initialize constraint node values tracking if needed
        constraint_node_values: Optional[Dict[str, Dict[str, List[float]]]] = None
        if constraint_target_nodes:
            constraint_node_values = {
                opt.id: {node_id: [] for node_id in constraint_target_nodes}
                for opt in request.options
            }

        for _ in range(request.n_samples):
            # Sample edge configuration (structural + parametric uncertainty)
            edge_config = sampler.sample_edge_configuration()

            # Sample factor values (parameter uncertainty)
            factor_values = factor_sampler.sample_factor_values()
            factor_values_per_sample.append(factor_values)

            # Evaluate each option
            sample_outcomes = {}
            for option in request.options:
                if constraint_target_nodes:
                    # Use evaluate_multi to get both goal and constraint node values
                    all_target_nodes = list(set([request.goal_node_id] + constraint_target_nodes))
                    node_values = evaluator.evaluate_multi(
                        edge_strengths=edge_config,
                        interventions=option.interventions,
                        target_nodes=all_target_nodes,
                        factor_values=factor_values,
                    )
                    outcome = node_values.get(request.goal_node_id, 0.0)
                    # Store constraint node values
                    for node_id in constraint_target_nodes:
                        constraint_node_values[option.id][node_id].append(
                            node_values.get(node_id, 0.0)
                        )
                else:
                    # Standard evaluation for goal node only
                    outcome = evaluator.evaluate(
                        edge_strengths=edge_config,
                        interventions=option.interventions,
                        goal_node=request.goal_node_id,
                        factor_values=factor_values,
                    )

                option_outcomes[option.id].append(outcome)
                sample_outcomes[option.id] = outcome

            # Track winner with fair tie-breaking (split ties equally)
            max_outcome = max(sample_outcomes.values())
            winners = [opt_id for opt_id, val in sample_outcomes.items() if val == max_outcome]

            if len(winners) == 1:
                # Clear winner
                option_wins[winners[0]] += 1.0
                winner_per_sample.append(winners[0])
            else:
                # Tie: split win equally among tied options
                tie_count += 1
                split_value = 1.0 / len(winners)
                for winner in winners:
                    option_wins[winner] += split_value
                # Use deterministic random tie-breaking via the existing sampler RNG.
                # This preserves full determinism (same seed = same tie-break result)
                # while eliminating insertion-order bias that arises from always
                # picking winners[0].  We reuse sampler.rng so the RNG stream
                # remains a single deterministic sequence — no new sub-seed needed.
                winner_per_sample.append(str(sampler.rng.choice(winners)))

            # Store edge config for alternative winner analysis
            edge_configs_per_sample.append(edge_config)

        return (
            option_outcomes,
            option_wins,
            winner_per_sample,
            edge_configs_per_sample,
            tie_count,
            constraint_node_values,
            factor_values_per_sample,
        )

    def _apply_auto_scaled_noise(
        self,
        option_outcomes: Dict[str, List[float]],
        goal_node_id: str,
        graph_nodes: List,
        rng: "SeededRNG",
    ) -> Dict[str, List[float]]:
        """
        Apply auto-scaled noise to outcome/risk node samples.

        Per Neil Bramley's heuristic: "Match unexplained noise to explained variance"
        - Only outcome and risk nodes receive noise
        - Noise std = std(samples) from the model
        - If std = 0, skip noise entirely (no model uncertainty)

        Args:
            option_outcomes: Dict of option_id -> list of outcome samples
            goal_node_id: The goal node being measured
            graph_nodes: List of graph nodes to check node kind
            rng: Seeded RNG for determinism

        Returns:
            Modified option_outcomes with noise applied
        """
        # Find the goal node and check its kind
        goal_node = None
        for node in graph_nodes:
            if node.id == goal_node_id:
                goal_node = node
                break

        if goal_node is None:
            return option_outcomes

        # Only apply noise to outcome and risk nodes
        node_kind = getattr(goal_node, "kind", "").lower()
        if node_kind not in ("outcome", "risk"):
            return option_outcomes

        # Apply noise to each option's samples
        for option_id, samples in option_outcomes.items():
            if not samples:
                continue

            samples_array = np.array(samples)
            outcome_std = float(np.std(samples_array))

            # If std = 0, skip noise (no model uncertainty to match)
            if outcome_std <= 0:
                continue

            # Deliberate modelling choice (Neil Bramley heuristic):
            # Add noise ~ N(0, outcome_std) to each sample.
            # Mathematical effect: var(X + N) = var(X) + var(N) ≈ 2·var(X) when
            # var(N) = var(X), so the p10/p90 spread is approximately √2 wider than
            # the purely model-driven distribution.
            # Rationale: the noise term represents unexplained variance not captured
            # by the structural causal model (measurement error, omitted variables, etc.).
            # This makes the outcome intervals more conservative (wider) which is the
            # correct direction under uncertainty.
            # WARNING: Changing this constant factor affects ALL downstream percentile
            # and confidence computations.  Any change requires re-validation of the
            # full calibration suite.
            noise = np.array([rng.normal(0, outcome_std) for _ in range(len(samples))])
            option_outcomes[option_id] = (samples_array + noise).tolist()

        return option_outcomes

    def _compute_option_results(
        self,
        outcomes: Dict[str, List[float]],
        wins: Dict[str, float],
        request: RobustnessRequestV2,
        constraint_node_values: Optional[Dict[str, Dict[str, List[float]]]] = None,
    ) -> List[OptionResult]:
        """Compute distribution statistics for each option.

        Args:
            outcomes: Dict[option_id, List[outcome_samples]]
            wins: Dict[option_id, win_count]
            request: The analysis request
            constraint_node_values: Optional dict of constraint node sample values
                for multi-constraint analysis
        """
        results = []

        for option in request.options:
            samples = outcomes[option.id]
            if not samples:
                continue

            samples_array = np.array(samples)
            ci_lower, ci_upper = self._compute_confidence_interval(
                samples_array, request.confidence_level
            )

            # Compute probability_of_goal if threshold is provided
            probability_of_goal = None
            if request.goal_threshold is not None:
                n_meets_threshold = int(np.sum(samples_array >= request.goal_threshold))
                probability_of_goal = n_meets_threshold / len(samples)

            # Compute constraint analysis if constraints provided
            constraint_analysis_result: Optional[ConstraintAnalysis] = None
            if request.goal_constraints and constraint_node_values:
                analysis_dict = self._compute_constraint_analysis(
                    constraint_node_values,
                    request.goal_constraints,
                    option.id,
                )
                if analysis_dict:
                    # Convert dict to ConstraintAnalysis model
                    constraint_results = [
                        ConstraintResult(
                            node_id=c["node_id"],
                            operator=c["operator"],
                            threshold=c["threshold"],
                            label=c["label"],
                            prob_satisfied=c["prob_satisfied"],
                            failure_margin_median=c["failure_margin_median"],
                            near_miss_fraction=c["near_miss_fraction"],
                            binding=c["binding"],
                        )
                        for c in analysis_dict["constraints"]
                    ]
                    constraint_analysis_result = ConstraintAnalysis(
                        constraints=constraint_results,
                        joint_probability=analysis_dict["joint_probability"],
                        conditional_probabilities=analysis_dict["conditional_probabilities"],
                    )

            results.append(
                OptionResult(
                    option_id=option.id,
                    outcome_distribution=OutcomeDistribution(
                        mean=float(np.mean(samples_array)),
                        std=float(np.std(samples_array)),
                        median=float(np.median(samples_array)),
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                        # Task 2: Store raw samples so the V2 API layer can compute
                        # actual p10/p50/p90 percentiles instead of aliasing CI bounds.
                        samples=samples,
                    ),
                    win_probability=wins[option.id] / request.n_samples,
                    probability_of_goal=probability_of_goal,
                    constraint_analysis=constraint_analysis_result,
                )
            )

        return results

    def _compute_confidence_interval(
        self, samples: np.ndarray, confidence_level: float
    ) -> Tuple[float, float]:
        """Compute percentile-based prediction interval from Monte Carlo samples.

        Note: Returns percentile bounds (not a frequentist confidence interval).
        For 95% level, returns 2.5th and 97.5th percentiles of the sample distribution.
        """
        alpha = 1 - confidence_level
        lower = float(np.percentile(samples, alpha / 2 * 100))
        upper = float(np.percentile(samples, (1 - alpha / 2) * 100))
        return lower, upper

    def _compute_sensitivity(
        self,
        request: RobustnessRequestV2,
        baseline_outcomes: Dict[str, List[float]],
        sampler: DualUncertaintySampler,
        rng: SeededRNG,
        evaluator: SCMEvaluatorV2,
    ) -> List[SensitivityResult]:
        """
        Compute sensitivity to edge existence and magnitude.

        For each edge, measures:
        1. Existence sensitivity: Impact of forcing edge on vs off
        2. Magnitude sensitivity: Impact of varying strength mean
        """
        sensitivities = []

        # Compute baseline mean outcome for reference option
        ref_option = request.options[0]
        baseline_mean = np.mean(baseline_outcomes[ref_option.id])

        for edge in request.graph.edges:
            # Existence sensitivity
            existence_sens = self._compute_existence_sensitivity(
                request, edge, baseline_mean, rng, evaluator
            )
            sensitivities.append(
                {
                    "edge_from": edge.from_,
                    "edge_to": edge.to,
                    "sensitivity_type": "existence",
                    "elasticity": existence_sens,
                    "interpretation": self._interpret_existence_sensitivity(edge, existence_sens),
                }
            )

            # Magnitude sensitivity
            magnitude_sens = self._compute_magnitude_sensitivity(
                request, edge, baseline_mean, rng, evaluator
            )
            sensitivities.append(
                {
                    "edge_from": edge.from_,
                    "edge_to": edge.to,
                    "sensitivity_type": "magnitude",
                    "elasticity": magnitude_sens,
                    "interpretation": self._interpret_magnitude_sensitivity(edge, magnitude_sens),
                }
            )

        # Rank by absolute elasticity
        sensitivities.sort(key=lambda x: abs(x["elasticity"]), reverse=True)

        # Convert to SensitivityResult with ranks
        results = []
        for i, s in enumerate(sensitivities):
            results.append(
                SensitivityResult(
                    edge_from=s["edge_from"],
                    edge_to=s["edge_to"],
                    sensitivity_type=s["sensitivity_type"],
                    elasticity=s["elasticity"],
                    importance_rank=i + 1,
                    interpretation=s["interpretation"],
                )
            )

        return results

    def _compute_existence_sensitivity(
        self,
        request: RobustnessRequestV2,
        edge: EdgeV2,
        baseline_mean: float,
        rng: SeededRNG,
        evaluator: SCMEvaluatorV2,
    ) -> float:
        """
        Compute sensitivity to edge existence.

        Compares outcomes when edge is forced to exist vs forced to not exist.
        """
        n_sensitivity_samples = min(100, request.n_samples // 10)
        ref_option = request.options[0]

        # Sample with edge forced to exist
        outcomes_on = []
        for _ in range(n_sensitivity_samples):
            edge_config = self._sample_with_forced_existence(
                request.graph.edges, edge, exists=True, rng=rng
            )
            outcome = evaluator.evaluate(
                edge_strengths=edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
            )
            outcomes_on.append(outcome)

        # Sample with edge forced to not exist
        outcomes_off = []
        for _ in range(n_sensitivity_samples):
            edge_config = self._sample_with_forced_existence(
                request.graph.edges, edge, exists=False, rng=rng
            )
            outcome = evaluator.evaluate(
                edge_strengths=edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
            )
            outcomes_off.append(outcome)

        # Compute elasticity
        mean_on = np.mean(outcomes_on)
        mean_off = np.mean(outcomes_off)
        outcome_diff = mean_on - mean_off

        # Use epsilon-stabilised denominator to handle near-zero baselines
        baseline_denom = max(abs(baseline_mean), FACTOR_SENSITIVITY_BASELINE_EPSILON)

        # Elasticity: relative change in outcome for existence change (0 -> 1)
        raw_elasticity = outcome_diff / baseline_denom
        return max(-ELASTICITY_CLAMP_MAX, min(ELASTICITY_CLAMP_MAX, raw_elasticity))

    def _compute_magnitude_sensitivity(
        self,
        request: RobustnessRequestV2,
        edge: EdgeV2,
        baseline_mean: float,
        rng: SeededRNG,
        evaluator: SCMEvaluatorV2,
    ) -> float:
        """
        Compute sensitivity to edge magnitude.

        Varies strength mean by ±1 std and measures outcome change.
        """
        n_sensitivity_samples = min(100, request.n_samples // 10)
        ref_option = request.options[0]

        # Sample with strength mean + std
        outcomes_high = []
        for _ in range(n_sensitivity_samples):
            edge_config = self._sample_with_shifted_mean(
                request.graph.edges, edge, shift=+edge.strength.std, rng=rng
            )
            outcome = evaluator.evaluate(
                edge_strengths=edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
            )
            outcomes_high.append(outcome)

        # Sample with strength mean - std
        outcomes_low = []
        for _ in range(n_sensitivity_samples):
            edge_config = self._sample_with_shifted_mean(
                request.graph.edges, edge, shift=-edge.strength.std, rng=rng
            )
            outcome = evaluator.evaluate(
                edge_strengths=edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
            )
            outcomes_low.append(outcome)

        # Compute elasticity (change per 2*std shift)
        mean_high = np.mean(outcomes_high)
        mean_low = np.mean(outcomes_low)
        outcome_diff = mean_high - mean_low

        # Use epsilon-stabilised denominator to handle near-zero baselines
        baseline_denom = max(abs(baseline_mean), FACTOR_SENSITIVITY_BASELINE_EPSILON)

        # Normalize by 2*std range
        raw_elasticity = (outcome_diff / baseline_denom) / 2.0
        return max(-ELASTICITY_CLAMP_MAX, min(ELASTICITY_CLAMP_MAX, raw_elasticity))

    def _sample_with_forced_existence(
        self,
        edges: List[EdgeV2],
        target_edge: EdgeV2,
        exists: bool,
        rng: SeededRNG,
    ) -> Dict[Tuple[str, str], float]:
        """Sample edge configuration with one edge's existence forced."""
        config = {}

        for edge in edges:
            edge_key = (edge.from_, edge.to)

            if edge.from_ == target_edge.from_ and edge.to == target_edge.to:
                # Force this edge's existence
                if exists:
                    strength = rng.normal(edge.strength.mean, edge.strength.std)
                    config[edge_key] = np.clip(strength, EDGE_STRENGTH_MIN, EDGE_STRENGTH_MAX)
                else:
                    config[edge_key] = 0.0
            else:
                # Sample normally
                if rng.bernoulli(edge.exists_probability):
                    strength = rng.normal(edge.strength.mean, edge.strength.std)
                    config[edge_key] = np.clip(strength, EDGE_STRENGTH_MIN, EDGE_STRENGTH_MAX)
                else:
                    config[edge_key] = 0.0

        return config

    def _sample_with_shifted_mean(
        self,
        edges: List[EdgeV2],
        target_edge: EdgeV2,
        shift: float,
        rng: SeededRNG,
    ) -> Dict[Tuple[str, str], float]:
        """
        Sample edge configuration with one edge's mean shifted.

        CRITICAL: The target edge is FORCED to exist so we isolate
        magnitude effect from existence effect. Otherwise, magnitude
        sensitivity would be conflated with structural uncertainty.
        """
        config = {}

        for edge in edges:
            edge_key = (edge.from_, edge.to)

            if edge.from_ == target_edge.from_ and edge.to == target_edge.to:
                # TARGET EDGE: Force to exist and apply shifted mean
                # This isolates magnitude sensitivity from existence sensitivity
                strength = rng.normal(edge.strength.mean + shift, edge.strength.std)
                config[edge_key] = np.clip(strength, EDGE_STRENGTH_MIN, EDGE_STRENGTH_MAX)
            else:
                # OTHER EDGES: Sample normally (both existence and strength)
                if rng.bernoulli(edge.exists_probability):
                    strength = rng.normal(edge.strength.mean, edge.strength.std)
                    config[edge_key] = np.clip(strength, EDGE_STRENGTH_MIN, EDGE_STRENGTH_MAX)
                else:
                    config[edge_key] = 0.0

        return config

    def _interpret_existence_sensitivity(self, edge: EdgeV2, elasticity: float) -> str:
        """Generate human-readable interpretation for existence sensitivity."""
        edge_name = f"{edge.from_}->{edge.to}"

        if abs(elasticity) < 0.05:
            return f"Decision is robust to whether {edge_name} exists"
        elif abs(elasticity) < self.HIGH_SENSITIVITY_THRESHOLD:
            return f"Decision is moderately sensitive to {edge_name} existence"
        else:
            return (
                f"Decision is highly sensitive to {edge_name} existence - "
                "consider validating this relationship"
            )

    def _interpret_magnitude_sensitivity(self, edge: EdgeV2, elasticity: float) -> str:
        """Generate human-readable interpretation for magnitude sensitivity."""
        edge_name = f"{edge.from_}->{edge.to}"

        if abs(elasticity) < 0.05:
            return f"Decision is robust to effect size variation in {edge_name}"
        elif abs(elasticity) < self.HIGH_SENSITIVITY_THRESHOLD:
            return f"Decision is moderately sensitive to {edge_name} effect size"
        else:
            return (
                f"Decision is highly sensitive to {edge_name} effect size - "
                "consider narrowing uncertainty"
            )

    def _compute_factor_sensitivity(
        self,
        request: RobustnessRequestV2,
        baseline_outcomes: Dict[str, List[float]],
        rng: SeededRNG,
        evaluator: SCMEvaluatorV2,
    ) -> List[FactorSensitivityResult]:
        """
        Compute sensitivity to factor node values.

        For each factor with uncertainty specified, measures how much
        the outcome changes when the factor value is varied by ±1 std
        (or ±10% of range for uniform distributions).

        Intervention vs Non-Intervention Factor Behavior
        ------------------------------------------------
        This function computes sensitivity for NON-INTERVENTION factors only
        (contextual variables like market conditions, user attributes, etc.).

        - **Non-intervention factors**: Variables that affect the outcome but
          are not directly controlled by the decision options. Sensitivity
          reflects how much uncertainty in these factors affects the decision.

        - **Intervention factors**: Variables set by options (e.g., marketing
          spend, pricing). These are NOT included in parameter_uncertainties
          because their values are determined by the option, not estimated.

        If a factor that is also an intervention target appears in
        parameter_uncertainties, its sensitivity may be zero or near-zero
        because the intervention value overrides the perturbed value.
        This is expected behavior—use zero_reason="intervention_override"
        to diagnose such cases.

        Debug Fields
        ------------
        - elasticity: Raw (unclamped) value for determinism/audit
        - elasticity_display: Clamped to [-100, 100] for UI safety
        - zero_reason: Explains why sensitivity is zero (if applicable)
        - baseline_near_zero: True if epsilon denominator was applied
        """
        # Diagnostic: log entry point
        self.logger.info(
            "factor_sensitivity_entry",
            extra={
                "has_parameter_uncertainties": bool(request.parameter_uncertainties),
                "num_uncertainties": len(request.parameter_uncertainties)
                if request.parameter_uncertainties
                else 0,
                "uncertainties": [
                    {"node_id": u.node_id, "distribution": u.distribution, "std": u.std}
                    for u in (request.parameter_uncertainties or [])
                ],
            },
        )

        if not request.parameter_uncertainties:
            return []

        sensitivities = []
        ref_option = request.options[0]
        baseline_mean = np.mean(baseline_outcomes[ref_option.id])

        # Build set of intervention factor IDs for INTERVENTION_OVERRIDE detection
        intervention_factor_ids = (
            set(ref_option.interventions.keys()) if ref_option.interventions else set()
        )

        # Diagnostic: log baseline
        self.logger.info(
            "factor_sensitivity_baseline",
            extra={
                "ref_option_id": ref_option.id,
                "baseline_mean": baseline_mean,
                "baseline_outcomes_count": len(baseline_outcomes.get(ref_option.id, [])),
            },
        )

        # Build node map for labels
        node_map = {n.id: n for n in request.graph.nodes}

        # Sample mean edge configuration for sensitivity analysis
        # (isolate factor sensitivity from edge uncertainty)
        mean_edge_config = {
            (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
        }

        for uncertainty in request.parameter_uncertainties:
            node = node_map.get(uncertainty.node_id)
            if not node:
                continue

            # Get observed value
            observed_value = None
            if node.observed_state and node.observed_state.value is not None:
                observed_value = node.observed_state.value

            mean_value = observed_value if observed_value is not None else 0.0

            # Determine perturbation amount based on distribution
            if uncertainty.distribution == "normal":
                delta = uncertainty.std or 0.0
            elif uncertainty.distribution == "uniform":
                range_min = uncertainty.range_min or 0.0
                range_max = uncertainty.range_max or 0.0
                delta = (range_max - range_min) * 0.1  # 10% of range
            else:
                # point_mass - no sensitivity
                delta = 0.0

            if delta == 0.0:
                # No uncertainty to measure
                sensitivities.append(
                    {
                        "node_id": uncertainty.node_id,
                        "node_label": node.label,
                        "elasticity": 0.0,
                        "elasticity_display": 0.0,
                        "observed_value": observed_value,
                        "interpretation": f"Factor {node.label} has no uncertainty (point mass)",
                        "zero_reason": ZeroSensitivityReason.POINT_MASS,
                        "baseline_near_zero": False,
                    }
                )
                continue

            # Evaluate with high and low values (single evaluation - deterministic given fixed inputs)
            factor_values_high = {uncertainty.node_id: mean_value + delta}
            outcome_high = evaluator.evaluate(
                edge_strengths=mean_edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
                factor_values=factor_values_high,
            )

            factor_values_low = {uncertainty.node_id: mean_value - delta}
            outcome_low = evaluator.evaluate(
                edge_strengths=mean_edge_config,
                interventions=ref_option.interventions,
                goal_node=request.goal_node_id,
                factor_values=factor_values_low,
            )

            outcome_diff = outcome_high - outcome_low

            # Compute true elasticity: (%Δ outcome) / (%Δ factor)
            # Use epsilon-stabilised denominators to handle near-zero baselines
            # (e.g., binary factors 0/1 where observed_state.value = 0)
            baseline_near_zero = abs(baseline_mean) < FACTOR_SENSITIVITY_BASELINE_EPSILON
            baseline_denom = max(abs(baseline_mean), FACTOR_SENSITIVITY_BASELINE_EPSILON)
            factor_denom = max(abs(mean_value), FACTOR_SENSITIVITY_VALUE_EPSILON)

            pct_outcome_change = outcome_diff / baseline_denom
            pct_factor_change = (2 * delta) / factor_denom
            # Raw elasticity is canonical (unclamped) for determinism
            elasticity = (
                pct_outcome_change / pct_factor_change if abs(pct_factor_change) > 1e-10 else 0.0
            )
            # Display elasticity is clamped for UI safety
            elasticity_display = max(-ELASTICITY_CLAMP_MAX, min(ELASTICITY_CLAMP_MAX, elasticity))

            # Determine zero_reason if elasticity is effectively zero
            # Priority order: INTERVENTION_OVERRIDE > ZERO_DELTA > ZERO_OUTCOME_DIFF > BASELINE_NORMALISED
            # (DISCONNECTED and POINT_MASS are handled elsewhere)
            zero_reason = None
            if abs(elasticity) < 1e-10:
                # Check if factor is overridden by intervention (highest priority)
                if uncertainty.node_id in intervention_factor_ids:
                    zero_reason = ZeroSensitivityReason.INTERVENTION_OVERRIDE
                # Check if delta is too small to perturb meaningfully
                elif abs(delta) < 1e-10:
                    zero_reason = ZeroSensitivityReason.ZERO_DELTA
                # Check if perturbation didn't affect outcome
                elif abs(outcome_diff) < 1e-10:
                    zero_reason = ZeroSensitivityReason.ZERO_OUTCOME_DIFF
                # Check if epsilon denominator was applied
                elif baseline_near_zero:
                    zero_reason = ZeroSensitivityReason.BASELINE_NORMALISED
                else:
                    # Computation resulted in zero (rare edge case)
                    zero_reason = ZeroSensitivityReason.ZERO_OUTCOME_DIFF

            # Diagnostic logging for factor sensitivity computation
            self.logger.info(
                "factor_sensitivity_computation",
                extra={
                    "node_id": uncertainty.node_id,
                    "node_label": node.label,
                    "baseline_mean": baseline_mean,
                    "mean_value": mean_value,
                    "delta": delta,
                    "outcome_high": outcome_high,
                    "outcome_low": outcome_low,
                    "outcome_diff": outcome_diff,
                    "baseline_denom": baseline_denom,
                    "factor_denom": factor_denom,
                    "pct_outcome_change": pct_outcome_change,
                    "pct_factor_change": pct_factor_change,
                    "elasticity": elasticity,
                    "elasticity_display": elasticity_display,
                    "zero_reason": zero_reason.value if zero_reason else None,
                    "baseline_near_zero": baseline_near_zero,
                },
            )

            sensitivities.append(
                {
                    "node_id": uncertainty.node_id,
                    "node_label": node.label,
                    "elasticity": elasticity,
                    "elasticity_display": elasticity_display,
                    "observed_value": observed_value,
                    "interpretation": self._interpret_factor_sensitivity(node.label, elasticity),
                    "zero_reason": zero_reason,
                    "baseline_near_zero": baseline_near_zero,
                }
            )

        # Compute structural influence for all factors
        factor_node_ids = [s["node_id"] for s in sensitivities]
        influence_scores = self._compute_structural_influence(
            request.graph, factor_node_ids, request.goal_node_id
        )

        # Add influence scores to sensitivities
        for s in sensitivities:
            s["influence_score"] = influence_scores.get(s["node_id"], 0.0)

        # Sort by absolute elasticity for importance_rank
        sensitivities.sort(key=lambda x: abs(x["elasticity"]), reverse=True)

        # Compute influence_rank (sort by influence_score descending)
        sorted_by_influence = sorted(
            sensitivities, key=lambda x: x["influence_score"], reverse=True
        )
        influence_rank_map = {s["node_id"]: i + 1 for i, s in enumerate(sorted_by_influence)}

        # --- Bootstrap stability analysis (3C) ---
        # Measures stability of attribution under model and sampling uncertainty:
        # how consistently each factor ranks as important when we resample edge
        # configurations. This is NOT "confidence in the causal relationship"
        # (which requires observational/experimental data we don't have).
        primary_elasticities = {s["node_id"]: s["elasticity"] for s in sensitivities}
        # sensitivities is already sorted by |elasticity| desc, so index+1 = importance_rank
        primary_ranks = {s["node_id"]: i + 1 for i, s in enumerate(sensitivities)}
        bootstrap_stability = self._compute_bootstrap_stability(
            request,
            baseline_mean,
            ref_option,
            evaluator,
            rng,
            primary_elasticities,
            primary_ranks,
            n_bootstrap_override=self._n_bootstrap_override,
        )

        # Convert to results with ranks
        results = []
        for i, s in enumerate(sensitivities):
            # Update zero_reason: DISCONNECTED takes priority if factor has no causal path
            zero_reason = s.get("zero_reason")
            if abs(s["elasticity"]) < 1e-10 and s["influence_score"] < 1e-10:
                # Factor is disconnected (no causal path to goal)
                # This overrides ZERO_OUTCOME_DIFF since disconnection is the root cause
                zero_reason = ZeroSensitivityReason.DISCONNECTED

            node_id = s["node_id"]
            bs = bootstrap_stability.get(node_id, {})

            results.append(
                FactorSensitivityResult(
                    node_id=node_id,
                    node_label=s["node_label"],
                    elasticity=s["elasticity"],
                    elasticity_display=s.get("elasticity_display"),
                    importance_rank=i + 1,
                    observed_value=s["observed_value"],
                    interpretation=s["interpretation"],
                    zero_reason=zero_reason,
                    baseline_near_zero=s.get("baseline_near_zero"),
                    influence_score=s["influence_score"],
                    influence_rank=influence_rank_map[s["node_id"]],
                    elasticity_std=bs.get("elasticity_std"),
                    attribution_stability=bs.get("attribution_stability"),
                    rank_flip_rate=bs.get("rank_flip_rate"),
                    stability_method=bs.get("stability_method"),
                )
            )

        return results

    def _interpret_factor_sensitivity(self, node_label: str, elasticity: float) -> str:
        """Generate human-readable interpretation for factor sensitivity."""
        if abs(elasticity) < 0.05:
            return f"Decision is robust to {node_label} value variation"
        elif abs(elasticity) < self.HIGH_SENSITIVITY_THRESHOLD:
            return f"Decision is moderately sensitive to {node_label} value"
        else:
            return (
                f"Decision is highly sensitive to {node_label} value - "
                "consider narrowing uncertainty or gathering more data"
            )

    def _compute_conditional_winners(
        self,
        factor_values_per_sample: List[Dict[str, float]],
        winner_per_sample: List[str],
        option_outcomes: Dict[str, List[float]],
        factor_sampler: "FactorSampler",
        request: RobustnessRequestV2,
        min_bucket_size: int = 50,
    ) -> Optional[List[ConditionalWinner]]:
        """
        Compute conditional win probabilities by partitioning MC samples at
        each factor's median value.

        For each non-point-mass factor, splits samples into low (< median) and
        high (>= median) buckets and computes win probabilities within each.
        Reports only factors where the winner flips between buckets.

        Limitation: median split is simplistic. It cannot detect non-monotonic
        effects, factor interactions, or flips at extreme quantiles.

        Args:
            factor_values_per_sample: Factor values sampled per MC iteration
            winner_per_sample: Winning option ID per MC sample
            option_outcomes: Per-option outcome values (for tie-breaking by mean)
            factor_sampler: FactorSampler (for _uncertainty_map and _node_map)
            request: The analysis request (for option labels)
            min_bucket_size: Minimum samples per bucket (skip if fewer)

        Returns:
            List of ConditionalWinner where winner_flips is True,
            or None if no flips found.
        """
        if len(request.options) <= 1:
            return None
        if not factor_values_per_sample or not factor_sampler.has_uncertainties():
            return None

        option_labels = {opt.id: (opt.label or opt.id) for opt in request.options}
        option_means = {opt_id: float(np.mean(vals)) for opt_id, vals in option_outcomes.items()}

        results: List[ConditionalWinner] = []

        for factor_id, uncertainty in factor_sampler._uncertainty_map.items():
            if uncertainty.distribution == "point_mass":
                continue

            # Extract factor values across all samples
            values = np.array([fv.get(factor_id, np.nan) for fv in factor_values_per_sample])
            if np.any(np.isnan(values)):
                continue

            median = float(np.median(values))

            # Partition into low/high buckets
            low_mask = values < median
            high_mask = ~low_mask

            low_count = int(np.sum(low_mask))
            high_count = int(np.sum(high_mask))

            if low_count < min_bucket_size or high_count < min_bucket_size:
                continue

            # Compute bucket winners
            low_bucket = self._compute_bucket_result(
                low_mask, winner_per_sample, option_labels, option_means
            )
            high_bucket = self._compute_bucket_result(
                high_mask, winner_per_sample, option_labels, option_means
            )

            winner_flips = low_bucket.winner_id != high_bucket.winner_id

            if not winner_flips:
                continue

            # Get node metadata
            node = factor_sampler._node_map.get(factor_id)
            factor_label = node.label if node else factor_id
            split_unit = (
                node.observed_state.unit
                if node and node.observed_state and node.observed_state.unit
                else None
            )

            results.append(
                ConditionalWinner(
                    factor_id=factor_id,
                    factor_label=factor_label,
                    split_value=median,
                    split_unit=split_unit,
                    low_bucket=low_bucket,
                    high_bucket=high_bucket,
                    winner_flips=True,
                )
            )

        return results if results else None

    def _compute_bucket_result(
        self,
        mask: np.ndarray,
        winner_per_sample: List[str],
        option_labels: Dict[str, str],
        option_means: Dict[str, float],
    ) -> BucketResult:
        """Compute win probabilities within a bucket of MC samples."""
        indices = np.where(mask)[0]
        bucket_size = len(indices)

        # Count wins per option in this bucket
        win_counts: Dict[str, int] = {}
        for idx in indices:
            winner = winner_per_sample[idx]
            win_counts[winner] = win_counts.get(winner, 0) + 1

        # Determine bucket winner (ties broken by higher mean outcome)
        sorted_options = sorted(
            win_counts.items(),
            key=lambda x: (x[1], option_means.get(x[0], 0.0)),
            reverse=True,
        )

        winner_id = sorted_options[0][0]
        winner_prob = sorted_options[0][1] / bucket_size

        runner_up_id = None
        runner_up_prob = None
        if len(sorted_options) > 1:
            runner_up_id = sorted_options[1][0]
            runner_up_prob = sorted_options[1][1] / bucket_size

        return BucketResult(
            n_samples=bucket_size,
            winner_id=winner_id,
            winner_label=option_labels.get(winner_id, winner_id),
            winner_probability=winner_prob,
            runner_up_id=runner_up_id,
            runner_up_probability=runner_up_prob,
        )

    def _compute_bootstrap_stability(
        self,
        request: RobustnessRequestV2,
        baseline_mean: float,
        ref_option: InterventionOption,
        evaluator: SCMEvaluatorV2,
        rng: SeededRNG,
        primary_elasticities: Dict[str, float],
        primary_ranks: Dict[str, int],
        n_bootstrap_override: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute bootstrap stability metrics for factor sensitivity (3C).

        Measures stability of attribution under model and sampling uncertainty:
        how consistently each factor ranks as important when we resample edge
        configurations. This is NOT "confidence in the causal relationship"
        (which requires observational/experimental data we don't have).

        Adaptive budget (when n_bootstrap_override is None):
        - Start at N_BOOTSTRAP=10. If wall-clock < 100ms, increase to 20.
        - If > 200ms at 10, keep 10 and log budget_exceeded.
        - Jackknife fallback is not implemented: the current sensitivity code
          uses deterministic mean-edge-config evaluation (not per-MC-sample
          elasticities), so delete-d jackknife cannot recompute elasticity
          without re-running the full MC. Bootstrap at 10 iterations is
          sub-millisecond per iteration, making the fallback unnecessary.

        Args:
            request: The robustness request (graph, parameter_uncertainties, etc.)
            baseline_mean: Mean outcome from the primary MC run
            ref_option: Reference option for evaluation
            evaluator: SCM evaluator
            rng: Primary RNG (bootstrap seeds derived from rng.seed)
            primary_elasticities: Dict[node_id -> elasticity] from deterministic computation
            primary_ranks: Dict[node_id -> importance_rank] from the main sensitivity run
            n_bootstrap_override: If set, skip adaptive budget and run exactly this many
                iterations. Used by tests to ensure timing-independent determinism.

        Returns:
            Dict[node_id -> {elasticity_std, attribution_stability, rank_flip_rate, stability_method}]
        """
        if not request.parameter_uncertainties:
            return {}

        primary_seed = rng.seed
        budget_exceeded = False
        t0 = time.time()

        if n_bootstrap_override is not None:
            # Fixed count — skip adaptive budget (used by tests)
            bootstrap_elasticities = self._run_bootstrap_iterations(
                request,
                baseline_mean,
                ref_option,
                evaluator,
                primary_seed,
                n_bootstrap_override,
            )
            n_bootstrap = n_bootstrap_override
        else:
            # --- Phase 1: run 10 bootstrap iterations ---
            n_bootstrap_initial = 10
            bootstrap_elasticities = self._run_bootstrap_iterations(
                request,
                baseline_mean,
                ref_option,
                evaluator,
                primary_seed,
                n_bootstrap_initial,
            )
            t1 = time.time()
            elapsed_ms = (t1 - t0) * 1000

            # --- Phase 2: adaptive budget ---
            n_bootstrap = n_bootstrap_initial
            if elapsed_ms < 100:
                # Budget allows 20 runs; add 10 more
                extra = self._run_bootstrap_iterations(
                    request,
                    baseline_mean,
                    ref_option,
                    evaluator,
                    primary_seed + n_bootstrap_initial,
                    10,
                )
                # Merge: append elasticities for each node
                for node_id in bootstrap_elasticities:
                    bootstrap_elasticities[node_id].extend(extra.get(node_id, []))
                n_bootstrap = 20
            elif elapsed_ms > 200:
                budget_exceeded = True
                self.logger.info(
                    "factor_sensitivity_bootstrap_budget_exceeded",
                    extra={
                        "elapsed_ms": round(elapsed_ms, 1),
                        "n_bootstrap": n_bootstrap_initial,
                        "budget_exceeded": True,
                    },
                )

        stability_method = f"bootstrap_{n_bootstrap}"

        # --- Phase 3: compute stability metrics ---
        # Collect per-bootstrap rank arrays for rank_flip_rate
        n_factors = len(bootstrap_elasticities)
        node_ids = list(bootstrap_elasticities.keys())

        # Build per-bootstrap-run elasticity matrix: [run_index][factor_index]
        n_runs = n_bootstrap
        # bootstrap_elasticities[node_id] is a list of length n_runs
        elasticity_matrix = []
        for run_idx in range(n_runs):
            run_elasticities = {}
            for nid in node_ids:
                vals = bootstrap_elasticities[nid]
                run_elasticities[nid] = vals[run_idx] if run_idx < len(vals) else 0.0
            elasticity_matrix.append(run_elasticities)

        # Compute per-bootstrap importance ranks (by |elasticity|, descending)
        per_run_ranks: Dict[str, List[int]] = {nid: [] for nid in node_ids}
        for run_elasticities in elasticity_matrix:
            sorted_ids = sorted(node_ids, key=lambda nid: abs(run_elasticities[nid]), reverse=True)
            for rank_idx, nid in enumerate(sorted_ids):
                per_run_ranks[nid].append(rank_idx + 1)

        result: Dict[str, Dict[str, Any]] = {}
        for nid in node_ids:
            elasticities = np.array(bootstrap_elasticities[nid])
            e_std = float(np.std(elasticities, ddof=1)) if len(elasticities) > 1 else 0.0

            # Use primary (deterministic) elasticity for the negligible check,
            # not the bootstrap mean — the primary value is the reported number.
            primary_e = primary_elasticities.get(nid, 0.0)

            # Attribution stability from coefficient of variation
            # Thresholds are configurable via STABILITY_THRESHOLDS (provisional)
            attribution_stability = classify_attribution_stability(
                primary_e, e_std, STABILITY_THRESHOLDS
            )

            # Rank flip rate: fraction of runs where rank shifts by >= 2
            # compared to the reported primary importance_rank (the rank users see).
            base_rank = primary_ranks.get(nid, 1)
            ranks = per_run_ranks[nid]
            flips = sum(1 for r in ranks if abs(r - base_rank) >= 2)
            rank_flip_rate = flips / len(ranks) if ranks else 0.0

            result[nid] = {
                "elasticity_std": round(e_std, 8),
                "attribution_stability": attribution_stability,
                "rank_flip_rate": round(rank_flip_rate, 4),
                "stability_method": stability_method,
            }

        self.logger.info(
            "factor_sensitivity_bootstrap_complete",
            extra={
                "n_bootstrap": n_bootstrap,
                "stability_method": stability_method,
                "elapsed_ms": round((time.time() - t0) * 1000, 1),
                "budget_exceeded": budget_exceeded,
                "n_factors": n_factors,
            },
        )

        return result

    def _run_bootstrap_iterations(
        self,
        request: RobustnessRequestV2,
        baseline_mean: float,
        ref_option: InterventionOption,
        evaluator: SCMEvaluatorV2,
        seed_offset: int,
        n_iterations: int,
    ) -> Dict[str, List[float]]:
        """
        Run N bootstrap iterations of factor sensitivity with resampled edge configs.

        Each iteration:
        1. Create a new RNG from (seed_offset + iteration_index)
        2. Sample a fresh edge configuration via DualUncertaintySampler
        3. Compute elasticity for each factor using that edge config

        Args:
            request: Robustness request
            baseline_mean: Baseline outcome mean from primary MC
            ref_option: Reference option
            evaluator: SCM evaluator
            seed_offset: Base seed for this batch of iterations
            n_iterations: Number of iterations to run

        Returns:
            Dict[node_id -> List[elasticity_values]] across iterations
        """
        node_map = {n.id: n for n in request.graph.nodes}
        result: Dict[str, List[float]] = {
            u.node_id: [] for u in request.parameter_uncertainties if node_map.get(u.node_id)
        }

        for i in range(n_iterations):
            # Deterministic seed derived from primary seed + bootstrap index
            boot_rng = SeededRNG(seed_offset + i)
            boot_sampler = DualUncertaintySampler(request.graph.edges, boot_rng)
            edge_config = boot_sampler.sample_edge_configuration()

            for uncertainty in request.parameter_uncertainties:
                node = node_map.get(uncertainty.node_id)
                if not node:
                    continue

                observed_value = None
                if node.observed_state and node.observed_state.value is not None:
                    observed_value = node.observed_state.value
                mean_value = observed_value if observed_value is not None else 0.0

                if uncertainty.distribution == "normal":
                    delta = uncertainty.std or 0.0
                elif uncertainty.distribution == "uniform":
                    range_min = uncertainty.range_min or 0.0
                    range_max = uncertainty.range_max or 0.0
                    delta = (range_max - range_min) * 0.1
                else:
                    delta = 0.0

                if delta == 0.0:
                    result[uncertainty.node_id].append(0.0)
                    continue

                # Evaluate with this bootstrap's edge config
                factor_values_high = {uncertainty.node_id: mean_value + delta}
                outcome_high = evaluator.evaluate(
                    edge_strengths=edge_config,
                    interventions=ref_option.interventions,
                    goal_node=request.goal_node_id,
                    factor_values=factor_values_high,
                )

                factor_values_low = {uncertainty.node_id: mean_value - delta}
                outcome_low = evaluator.evaluate(
                    edge_strengths=edge_config,
                    interventions=ref_option.interventions,
                    goal_node=request.goal_node_id,
                    factor_values=factor_values_low,
                )

                outcome_diff = outcome_high - outcome_low
                baseline_denom = max(abs(baseline_mean), FACTOR_SENSITIVITY_BASELINE_EPSILON)
                factor_denom = max(abs(mean_value), FACTOR_SENSITIVITY_VALUE_EPSILON)
                pct_outcome_change = outcome_diff / baseline_denom
                pct_factor_change = (2 * delta) / factor_denom
                elasticity = (
                    pct_outcome_change / pct_factor_change
                    if abs(pct_factor_change) > 1e-10
                    else 0.0
                )

                result[uncertainty.node_id].append(elasticity)

        return result

    def _compute_structural_influence(
        self,
        graph: GraphV2,
        factor_node_ids: List[str],
        goal_node_id: str,
    ) -> Dict[str, float]:
        """
        Compute structural influence score for each factor based on causal path strengths.

        Algorithm:
        1. For each factor, find all paths to goal_node_id
        2. For each path, compute path_strength = product of edge.strength.mean * exists_probability
        3. Factor influence = sum of absolute path strengths (multiple paths add)
        4. Normalize to 0-1 scale across all factors

        Args:
            graph: Causal graph with edges
            factor_node_ids: List of factor node IDs to compute influence for
            goal_node_id: Target goal node ID

        Returns:
            Dict mapping node_id -> influence_score (0-1, normalized)
        """
        # Build adjacency list for path finding
        adjacency: Dict[str, List[Tuple[str, float]]] = {}
        for edge in graph.edges:
            from_node = edge.from_
            to_node = edge.to
            # Effective strength = mean * exists_probability
            effective_strength = edge.strength.mean * edge.exists_probability
            if from_node not in adjacency:
                adjacency[from_node] = []
            adjacency[from_node].append((to_node, effective_strength))

        def find_all_paths_strengths(
            start: str,
            end: str,
            visited: set,
        ) -> List[float]:
            """
            Find all paths from start to end and return list of path strengths.
            Each path strength is the product of edge strengths along the path.
            """
            if start == end:
                return [1.0]  # Base case: path of strength 1

            if start in visited:
                return []  # Cycle detection

            if start not in adjacency:
                return []  # No outgoing edges

            visited.add(start)
            path_strengths = []

            for next_node, edge_strength in adjacency[start]:
                sub_paths = find_all_paths_strengths(next_node, end, visited.copy())
                for sub_strength in sub_paths:
                    path_strengths.append(edge_strength * sub_strength)

            return path_strengths

        # Compute raw influence for each factor
        raw_influences: Dict[str, float] = {}
        for node_id in factor_node_ids:
            path_strengths = find_all_paths_strengths(node_id, goal_node_id, set())
            # Sum of absolute path strengths (multiple paths add)
            raw_influences[node_id] = sum(abs(s) for s in path_strengths)

        # Normalize to 0-1 scale
        max_influence = max(raw_influences.values()) if raw_influences else 0.0
        if max_influence < 1e-10:
            # All factors have zero influence
            return {node_id: 0.0 for node_id in factor_node_ids}

        return {node_id: raw_influences[node_id] / max_influence for node_id in factor_node_ids}

    def _compute_robustness(
        self,
        option_wins: Dict[str, float],
        winner_per_sample: List[str],
        sensitivity: List[SensitivityResult],
        request: RobustnessRequestV2,
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]],
        evaluator: SCMEvaluatorV2,
        global_seed: int,
    ) -> RobustnessResult:
        """Compute overall robustness assessment with alternative winner analysis."""
        # Recommendation stability: fraction of samples with same winner
        n_samples = request.n_samples
        most_frequent_winner = max(option_wins, key=option_wins.get)
        recommendation_stability = option_wins[most_frequent_winner] / n_samples

        # Identify fragile and robust edges (by edge_id string)
        # IMPORTANT: Aggregate sensitivities per edge BEFORE categorization
        # Each edge may have multiple sensitivity entries (existence + magnitude)
        # Use max(abs(elasticity)) to determine the edge's sensitivity level
        edge_max_elasticity: Dict[str, float] = {}
        edge_info: Dict[str, Tuple[str, str]] = {}  # edge_id -> (from_id, to_id)

        for sens in sensitivity:
            edge_id = f"{sens.edge_from}->{sens.edge_to}"
            current_max = edge_max_elasticity.get(edge_id, 0.0)
            edge_max_elasticity[edge_id] = max(current_max, abs(sens.elasticity))
            edge_info[edge_id] = (sens.edge_from, sens.edge_to)

        # Now categorize edges based on their max elasticity
        # Thresholds: fragile > 0.1, robust < 0.05, moderate = [0.05, 0.1]
        fragile_edge_ids = set()
        robust_edge_ids = set()
        fragile_edge_info: Dict[str, Tuple[str, str]] = {}

        for edge_id, max_elasticity in edge_max_elasticity.items():
            if max_elasticity > self.FRAGILE_THRESHOLD:
                fragile_edge_ids.add(edge_id)
                fragile_edge_info[edge_id] = edge_info[edge_id]
            elif max_elasticity < 0.05:
                robust_edge_ids.add(edge_id)
            # Edges with 0.05 <= elasticity <= 0.1 are implicitly "moderate" (uncategorized)

        fragile_edges = list(fragile_edge_ids)
        robust_edges = list(robust_edge_ids)

        # Compute alternative winners for fragile edges (includes marginal calculation)
        fragile_edges_enhanced = self._compute_alternative_winners(
            fragile_edge_info,
            edge_configs_per_sample,
            winner_per_sample,
            most_frequent_winner,
            request,
            evaluator,
            global_seed,
        )

        # Overall robustness
        # Per Decision Model Schema v2.6: is_robust = recommendation_stability >= 0.7
        # fragile_edges is a separate indicator of edge-level sensitivity
        is_robust = recommendation_stability >= self.ROBUST_THRESHOLD

        # Confidence based on sample size and stability
        confidence = min(0.99, recommendation_stability * (1 - 1 / np.sqrt(n_samples)))

        # Interpretation
        if is_robust:
            if fragile_edges:
                interpretation = (
                    f"Recommendation is ROBUST with {confidence:.0%} confidence. "
                    f"{most_frequent_winner} wins in {recommendation_stability:.0%} of scenarios. "
                    f"({len(fragile_edges)} sensitive edge{'s' if len(fragile_edges) > 1 else ''} identified)"
                )
            else:
                interpretation = (
                    f"Recommendation is ROBUST with {confidence:.0%} confidence. "
                    f"{most_frequent_winner} wins in {recommendation_stability:.0%} of scenarios."
                )
        elif recommendation_stability >= 0.5:
            interpretation = (
                f"Recommendation is MODERATELY ROBUST. "
                f"{most_frequent_winner} wins in {recommendation_stability:.0%} of scenarios, "
                f"but is sensitive to: {', '.join(fragile_edges[:3])}"
            )
        else:
            interpretation = (
                f"Recommendation is FRAGILE. No clear winner - "
                f"best option wins in only {recommendation_stability:.0%} of scenarios. "
                f"High sensitivity to: {', '.join(fragile_edges[:3])}"
            )

        return RobustnessResult(
            is_robust=is_robust,
            confidence=confidence,
            fragile_edges=fragile_edges,
            fragile_edges_enhanced=fragile_edges_enhanced,
            robust_edges=robust_edges,
            recommendation_stability=recommendation_stability,
            interpretation=interpretation,
        )

    def _compute_alternative_winners(
        self,
        fragile_edge_info: Dict[str, Tuple[str, str]],
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]],
        winner_per_sample: List[str],
        overall_winner: str,
        request: Optional[RobustnessRequestV2] = None,
        evaluator: Optional[SCMEvaluatorV2] = None,
        global_seed: Optional[int] = None,
    ) -> List[FragileEdgeEnhanced]:
        """
        Compute alternative winners for fragile edges.

        For each fragile edge, identifies which option wins most often when
        the edge is "weak" (bottom 25% of sampled strengths). Also computes
        marginal switch probability (isolated edge contribution) when
        request, evaluator, and global_seed are provided.

        Args:
            fragile_edge_info: Map of edge_id -> (from_id, to_id)
            edge_configs_per_sample: Edge strengths for each MC sample
            winner_per_sample: Winner option ID for each MC sample
            overall_winner: The overall recommended option
            request: Full robustness request with graph and options (optional)
            evaluator: SCM evaluator instance (optional)
            global_seed: Request-level seed for reproducibility (optional)

        Returns:
            List of FragileEdgeEnhanced objects with enhanced fragile edge information
        """
        # Check if marginal calculation is possible
        can_compute_marginal = (
            request is not None and evaluator is not None and global_seed is not None
        )

        results = []

        for edge_id, (from_id, to_id) in fragile_edge_info.items():
            edge_key = (from_id, to_id)

            # Compute marginal switch probability (isolated edge contribution)
            # Only computed when all required parameters are provided
            # Note: marginal computes its own baseline winner under expected-value config
            marginal_prob: Optional[float] = None
            if can_compute_marginal:
                marginal_prob = self._compute_marginal_switch_probability(
                    edge_key=edge_key,
                    request=request,
                    evaluator=evaluator,
                    global_seed=global_seed,
                )

            # Collect edge strengths across all samples
            strengths = [config.get(edge_key, 0.0) for config in edge_configs_per_sample]

            if not strengths:
                # No data for this edge (joint sampling unavailable)
                results.append(
                    FragileEdgeEnhanced(
                        edge_id=edge_id,
                        from_id=from_id,
                        to_id=to_id,
                        alternative_winner_id=None,
                        switch_probability=None,
                        marginal_switch_probability=marginal_prob,
                    )
                )
                continue

            # Find bottom 25% threshold (weak edge samples)
            strength_array = np.array(strengths)
            weak_threshold = np.percentile(strength_array, 25)

            # Get samples where edge is weak
            weak_sample_indices = [i for i, s in enumerate(strengths) if s <= weak_threshold]

            if not weak_sample_indices:
                results.append(
                    FragileEdgeEnhanced(
                        edge_id=edge_id,
                        from_id=from_id,
                        to_id=to_id,
                        alternative_winner_id=None,
                        switch_probability=None,
                        marginal_switch_probability=marginal_prob,
                    )
                )
                continue

            # Count winner distribution in weak-edge samples
            weak_winner_counts: Dict[str, int] = defaultdict(int)
            for idx in weak_sample_indices:
                weak_winner_counts[winner_per_sample[idx]] += 1

            # Find most frequent winner in weak-edge samples
            weak_winner = max(weak_winner_counts, key=weak_winner_counts.get)
            weak_winner_count = weak_winner_counts[weak_winner]
            total_weak_samples = len(weak_sample_indices)

            # Determine alternative winner and switch probability
            # The alternative is the best option OTHER than the overall winner
            # switch_probability is the probability of that alternative in weak scenarios
            if weak_winner != overall_winner:
                # Clear case: a different option wins when edge is weak
                alternative_winner_id = weak_winner
                switch_probability = weak_winner_count / total_weak_samples
            else:
                # Same option wins most often, but we want to show the risk
                # Find the best alternative (second most frequent) and its probability
                alternatives = {
                    opt: count for opt, count in weak_winner_counts.items() if opt != overall_winner
                }
                if alternatives:
                    # There's at least one alternative winner in weak scenarios
                    best_alt = max(alternatives, key=alternatives.get)
                    alternative_winner_id = best_alt
                    switch_probability = alternatives[best_alt] / total_weak_samples
                else:
                    # Only the overall winner appeared in weak scenarios - truly stable
                    alternative_winner_id = None
                    switch_probability = 0.0

            results.append(
                FragileEdgeEnhanced(
                    edge_id=edge_id,
                    from_id=from_id,
                    to_id=to_id,
                    alternative_winner_id=alternative_winner_id,
                    switch_probability=switch_probability,
                    marginal_switch_probability=marginal_prob,
                )
            )

        return results

    def _compute_marginal_switch_probability(
        self,
        edge_key: Tuple[str, str],
        request: RobustnessRequestV2,
        evaluator: SCMEvaluatorV2,
        global_seed: int,
        k_samples: int = MARGINAL_K_SAMPLES,
    ) -> float:
        """Compute probability of decision flip when ONLY this edge varies.

        Samples this edge K times from its uncertainty distribution while holding
        all other edges at their expected values (mean * exists_probability).
        Returns fraction of samples where the winner changes.

        Baseline semantics:
        - Other edges: held at expected value (mean * exists_probability)
        - Target edge: samples BOTH existence (Bernoulli) AND strength (Normal)
        - Baseline winner: computed under the same baseline config (not MC overall_winner)

        This ensures the marginal calculation is self-consistent: we compare
        sampled outcomes against the winner under the same baseline assumptions.

        Args:
            edge_key: (from_id, to_id) tuple
            request: Full robustness request with graph and options
            evaluator: SCM evaluator instance
            global_seed: Request-level seed for reproducibility
            k_samples: Number of samples (default 100)

        Returns:
            Probability in [0.0, 1.0] that this edge alone flips the recommendation
        """
        from_id, to_id = edge_key

        # Deterministic seed: SHA256 (process-safe, NOT Python hash())
        edge_seed_str = f"{global_seed}:{from_id}->{to_id}"
        edge_seed = int(hashlib.sha256(edge_seed_str.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(edge_seed)

        # Get target edge's parameters
        edge = next((e for e in request.graph.edges if (e.from_, e.to) == edge_key), None)
        if edge is None:
            self.logger.warning(
                "marginal_switch_edge_not_found",
                extra={"edge_key": f"{from_id}->{to_id}"},
            )
            return 0.0

        # Build baseline config: all edges at expected value (mean * exists_probability)
        # This is consistent with how existence is a sampling gate in the rest of the system
        baseline_config = {
            (e.from_, e.to): e.strength.mean * e.exists_probability for e in request.graph.edges
        }

        # Compute baseline winner under this config (not overall_winner from MC)
        # This ensures we compare against the correct reference point
        baseline_outcomes = {}
        for option in request.options:
            baseline_outcomes[option.id] = evaluator.evaluate(
                edge_strengths=baseline_config,
                interventions=option.interventions,
                goal_node=request.goal_node_id,
            )
        # Deterministic tie-breaking for baseline winner
        sorted_baseline = sorted(baseline_outcomes.items(), key=lambda x: (-x[1], x[0]))
        marginal_baseline_winner = sorted_baseline[0][0]

        flip_count = 0

        for _ in range(k_samples):
            # Sample existence (Bernoulli)
            exists = rng.random() < edge.exists_probability

            if not exists:
                # Edge doesn't exist in this sample → effective strength is 0
                sampled_strength = 0.0
            else:
                # Sample strength from Normal(mean, std), clamped to schema bounds
                sampled_strength = rng.normal(edge.strength.mean, edge.strength.std)
                sampled_strength = np.clip(sampled_strength, EDGE_STRENGTH_MIN, EDGE_STRENGTH_MAX)

            # Build counterfactual config: this edge sampled, others at baseline
            counterfactual_config = baseline_config.copy()
            counterfactual_config[edge_key] = sampled_strength

            # Evaluate all options under counterfactual
            outcomes = {}
            for option in request.options:
                outcomes[option.id] = evaluator.evaluate(
                    edge_strengths=counterfactual_config,
                    interventions=option.interventions,
                    goal_node=request.goal_node_id,
                )

            # Determine winner with deterministic tie-breaking (sort by option_id)
            sorted_options = sorted(outcomes.items(), key=lambda x: (-x[1], x[0]))
            sample_winner = sorted_options[0][0]

            if sample_winner != marginal_baseline_winner:
                flip_count += 1

        return flip_count / k_samples

    # =========================================================================
    # Multi-Constraint Goal Analysis (Phase 2)
    # =========================================================================

    def _check_constraint_satisfied(
        self,
        value: float,
        constraint: GoalConstraint,
    ) -> bool:
        """
        Check if a value satisfies a constraint.

        Args:
            value: The node value to check
            constraint: The constraint to check against

        Returns:
            True if value satisfies constraint, False otherwise
        """
        if constraint.operator == ">=":
            return value >= constraint.threshold
        elif constraint.operator == "<=":
            return value <= constraint.threshold
        else:
            # This should never happen due to Pydantic validation
            raise ValueError(f"Unknown operator: {constraint.operator}")

    def _compute_constraint_probabilities(
        self,
        constraint_node_values: Dict[str, Dict[str, List[float]]],
        constraints: List[GoalConstraint],
        option_id: str,
    ) -> Tuple[Dict[str, float], float, List[List[bool]]]:
        """
        Compute per-constraint and joint probabilities for an option.

        Args:
            constraint_node_values: Dict[option_id, Dict[node_id, List[sample_values]]]
            constraints: List of GoalConstraint objects
            option_id: The option to compute probabilities for

        Returns:
            Tuple of:
            - per_constraint_probs: Dict[constraint_index_str, prob_satisfied]
            - joint_probability: P(all constraints satisfied)
            - satisfaction_matrix: List[sample_idx][constraint_idx] -> bool (for conditional prob)
        """
        if not constraints:
            return {}, 1.0, []

        option_values = constraint_node_values[option_id]
        n_samples = len(next(iter(option_values.values())))

        # Build satisfaction matrix: [sample_idx][constraint_idx] -> bool
        satisfaction_matrix: List[List[bool]] = []
        for sample_idx in range(n_samples):
            sample_satisfactions = []
            for constraint in constraints:
                value = option_values[constraint.node_id][sample_idx]
                satisfied = self._check_constraint_satisfied(value, constraint)
                sample_satisfactions.append(satisfied)
            satisfaction_matrix.append(sample_satisfactions)

        # Per-constraint probabilities
        per_constraint_probs = {}
        for c_idx, constraint in enumerate(constraints):
            satisfied_count = sum(1 for sample in satisfaction_matrix if sample[c_idx])
            per_constraint_probs[str(c_idx)] = satisfied_count / n_samples

        # Joint probability: all constraints satisfied
        joint_satisfied_count = sum(1 for sample in satisfaction_matrix if all(sample))
        joint_probability = joint_satisfied_count / n_samples

        return per_constraint_probs, joint_probability, satisfaction_matrix

    def _compute_conditional_probabilities(
        self,
        satisfaction_matrix: List[List[bool]],
        constraints: List[GoalConstraint],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise conditional probabilities: P(C_j | C_i).

        Args:
            satisfaction_matrix: [sample_idx][constraint_idx] -> bool
            constraints: List of GoalConstraint objects

        Returns:
            Dict[constraint_i_idx, Dict[constraint_j_idx, conditional_prob]]
            Where conditional_prob is P(C_j | C_i) = P(C_i and C_j) / P(C_i)
        """
        if len(constraints) < 2:
            return {}

        n_samples = len(satisfaction_matrix)
        n_constraints = len(constraints)

        conditional_probs: Dict[str, Dict[str, float]] = {}

        for i in range(n_constraints):
            conditional_probs[str(i)] = {}
            # Count samples where constraint i is satisfied
            count_i = sum(1 for sample in satisfaction_matrix if sample[i])

            if count_i == 0:
                # P(C_j | C_i) is undefined when P(C_i) = 0 - omit these entries
                # The dict for this constraint remains empty, indicating undefined
                pass
            else:
                for j in range(n_constraints):
                    if i != j:
                        # Count samples where both i and j are satisfied
                        count_ij = sum(
                            1 for sample in satisfaction_matrix if sample[i] and sample[j]
                        )
                        conditional_probs[str(i)][str(j)] = count_ij / count_i

        return conditional_probs

    def _compute_near_miss_diagnostics(
        self,
        constraint_node_values: Dict[str, Dict[str, List[float]]],
        constraints: List[GoalConstraint],
        option_id: str,
        satisfaction_matrix: List[List[bool]],
        near_miss_fraction_threshold: float = 0.1,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute near-miss diagnostics for each constraint.

        For each constraint, computes:
        - failure_margin_median: Median distance from threshold when constraint fails
        - near_miss_fraction: Fraction of failures within near_miss_fraction_threshold of threshold
        - binding: True if prob_satisfied ∈ [0.4, 0.6] (constraint is borderline)

        Args:
            constraint_node_values: Dict[option_id, Dict[node_id, List[sample_values]]]
            constraints: List of GoalConstraint objects
            option_id: The option to compute diagnostics for
            satisfaction_matrix: Precomputed satisfaction matrix
            near_miss_fraction_threshold: Relative threshold for "near miss" (default 10%)

        Returns:
            Dict[constraint_idx, {failure_margin_median, near_miss_fraction, binding}]
        """
        if not constraints:
            return {}

        option_values = constraint_node_values[option_id]
        n_samples = len(satisfaction_matrix)

        diagnostics: Dict[int, Dict[str, Any]] = {}

        for c_idx, constraint in enumerate(constraints):
            values = option_values[constraint.node_id]
            threshold = constraint.threshold

            # Get failure samples
            failure_margins = []
            near_miss_count = 0

            for sample_idx, satisfied in enumerate(sample[c_idx] for sample in satisfaction_matrix):
                if not satisfied:
                    value = values[sample_idx]
                    # Compute margin (distance from threshold)
                    if constraint.operator == ">=":
                        # For >= threshold, margin is threshold - value (positive when failing)
                        margin = threshold - value
                    else:  # <=
                        # For <= threshold, margin is value - threshold (positive when failing)
                        margin = value - threshold

                    failure_margins.append(margin)

                    # Check if near-miss (within threshold% of the threshold value)
                    threshold_abs = abs(threshold) if threshold != 0 else 1.0
                    if margin <= near_miss_fraction_threshold * threshold_abs:
                        near_miss_count += 1

            # Compute diagnostics
            n_failures = len(failure_margins)
            failure_margin_median = float(np.median(failure_margins)) if failure_margins else None
            near_miss_fraction = near_miss_count / n_failures if n_failures > 0 else None

            # Compute prob_satisfied for binding determination
            satisfied_count = sum(1 for sample in satisfaction_matrix if sample[c_idx])
            prob_satisfied = satisfied_count / n_samples
            binding = 0.4 <= prob_satisfied <= 0.6

            diagnostics[c_idx] = {
                "failure_margin_median": failure_margin_median,
                "near_miss_fraction": near_miss_fraction,
                "binding": binding,
            }

        return diagnostics

    def _compute_constraint_analysis(
        self,
        constraint_node_values: Optional[Dict[str, Dict[str, List[float]]]],
        constraints: Optional[List[GoalConstraint]],
        option_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute full constraint analysis for an option.

        Args:
            constraint_node_values: Dict[option_id, Dict[node_id, List[sample_values]]]
            constraints: List of GoalConstraint objects
            option_id: The option to compute analysis for

        Returns:
            Dict with constraint analysis results, or None if no constraints
        """
        if not constraints or not constraint_node_values:
            return None

        # T3: Per-constraint and joint probability
        (
            per_constraint_probs,
            joint_probability,
            satisfaction_matrix,
        ) = self._compute_constraint_probabilities(constraint_node_values, constraints, option_id)

        # T4: Pairwise conditional probabilities
        conditional_probs = self._compute_conditional_probabilities(
            satisfaction_matrix, constraints
        )

        # T5: Near-miss diagnostics
        near_miss_diagnostics = self._compute_near_miss_diagnostics(
            constraint_node_values, constraints, option_id, satisfaction_matrix
        )

        # Build constraint results
        constraint_results = []
        for c_idx, constraint in enumerate(constraints):
            diag = near_miss_diagnostics.get(c_idx, {})
            constraint_results.append(
                {
                    "node_id": constraint.node_id,
                    "operator": constraint.operator,
                    "threshold": constraint.threshold,
                    "label": constraint.label,
                    "prob_satisfied": per_constraint_probs.get(str(c_idx), 0.0),
                    "failure_margin_median": diag.get("failure_margin_median"),
                    "near_miss_fraction": diag.get("near_miss_fraction"),
                    "binding": diag.get("binding", False),
                }
            )

        return {
            "constraints": constraint_results,
            "joint_probability": joint_probability,
            "conditional_probabilities": conditional_probs if conditional_probs else None,
        }
