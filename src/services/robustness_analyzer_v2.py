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

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models.robustness_v2 import (
    ClampMetrics,
    EdgeV2,
    FactorSensitivityResult,
    FragileEdgeEnhanced,
    GraphV2,
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
)
from src.constants import ZERO_VARIANCE_TOLERANCE
from src.models.critique import (
    DEGENERATE_OPTION_ZERO_VARIANCE,
    HIGH_TIE_RATE,
)
from src.models.response_v2 import CritiqueV2
from src.utils.rng import SeededRNG, compute_seed_from_graph
from src.__version__ import __version__
from src.models.metadata import generate_config_fingerprint

logger = logging.getLogger(__name__)


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
                config[edge_key] = strength
                self._existence_counts[edge_key] += 1
            else:
                # Edge doesn't exist in this sample
                config[edge_key] = 0.0

        return config

    def sample_n_configurations(
        self, n: int
    ) -> List[Dict[Tuple[str, str], float]]:
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
            sampled_value = self._sample_from_distribution(uncertainty, mean)
            factor_values[node_id] = sampled_value
            self._value_sums[node_id] += sampled_value

        return factor_values

    def _sample_from_distribution(
        self, uncertainty: ParameterUncertainty, mean: float
    ) -> float:
        """
        Sample a value from the specified distribution.

        Args:
            uncertainty: Distribution specification
            mean: Mean value (from observed_state.value)

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
                    f"Uniform distribution for node {node_id} requires range_min and range_max"
                )
            return self.rng.uniform(range_min, range_max)

        else:
            # Unknown distribution - fail fast instead of silent fallback
            raise ValueError(
                f"Unknown distribution '{dist}' for node {node_id}. "
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

        return {
            node_id: total / self._sample_count
            for node_id, total in self._value_sums.items()
        }


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

        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for child in adj[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.graph.nodes):
            # Graph has cycles - return arbitrary order
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
                    if is_root and node and node.observed_state and node.observed_state.value is not None:
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
                intercept = getattr(node, 'intercept', 0.0) if node else 0.0

                node_values[node_id] = base + intercept + parents_contribution

        return node_values.get(goal_node, 0.0)


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

        # Setup - use separate RNG streams for edge and factor sampling
        # to prevent fragile determinism coupling
        seed = request.seed or compute_seed_from_graph(request.graph)
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

        # Run Monte Carlo simulation
        (
            option_outcomes,
            option_wins,
            winner_per_sample,
            edge_configs_per_sample,
            tie_count,
        ) = self._run_monte_carlo(request, sampler, factor_sampler, evaluator)

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

        # Compute results
        results = self._compute_option_results(
            option_outcomes, option_wins, request
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

        # Compute robustness assessment (with alternative winner analysis)
        robustness = self._compute_robustness(
            option_wins,
            winner_per_sample,
            sensitivity,
            request,
            edge_configs_per_sample,
        )

        execution_time = int((time.time() - start_time) * 1000)

        # Find recommended option
        recommended_option_id = max(option_wins, key=option_wins.get)
        recommendation_confidence = option_wins[recommended_option_id] / request.n_samples

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
            ),
            critiques=critiques,
        )

        self.logger.info(
            "robustness_v2_analysis_complete",
            extra={
                "request_id": request_id,
                "recommended_option": recommended_option_id,
                "recommendation_confidence": recommendation_confidence,
                "is_robust": robustness.is_robust,
                "execution_time_ms": execution_time,
            },
        )

        return response

    def _run_monte_carlo(
        self,
        request: RobustnessRequestV2,
        sampler: DualUncertaintySampler,
        factor_sampler: FactorSampler,
        evaluator: SCMEvaluatorV2,
    ) -> Tuple[
        Dict[str, List[float]],
        Dict[str, float],
        List[str],
        List[Dict[Tuple[str, str], float]],
        int,
    ]:
        """
        Run Monte Carlo simulation with dual edge uncertainty and factor uncertainty.

        Returns:
            (option_outcomes, option_wins, winner_per_sample, edge_configs_per_sample, tie_count)

        Note: option_wins uses float to support split-tie handling where ties are
        divided equally among tied options.
        """
        option_outcomes: Dict[str, List[float]] = {
            opt.id: [] for opt in request.options
        }
        option_wins: Dict[str, float] = {opt.id: 0.0 for opt in request.options}
        winner_per_sample: List[str] = []
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]] = []
        tie_count = 0

        for _ in range(request.n_samples):
            # Sample edge configuration (structural + parametric uncertainty)
            edge_config = sampler.sample_edge_configuration()

            # Sample factor values (parameter uncertainty)
            factor_values = factor_sampler.sample_factor_values()

            # Evaluate each option
            sample_outcomes = {}
            for option in request.options:
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
                # For winner_per_sample, use first winner (for backward compat in alternative winner analysis)
                winner_per_sample.append(winners[0])

            # Store edge config for alternative winner analysis
            edge_configs_per_sample.append(edge_config)

        return option_outcomes, option_wins, winner_per_sample, edge_configs_per_sample, tie_count

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
        node_kind = getattr(goal_node, 'kind', '').lower()
        if node_kind not in ('outcome', 'risk'):
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

            # Add noise ~ N(0, outcome_std) to each sample
            noise = np.array([rng.normal(0, outcome_std) for _ in range(len(samples))])
            option_outcomes[option_id] = (samples_array + noise).tolist()

        return option_outcomes

    def _compute_option_results(
        self,
        outcomes: Dict[str, List[float]],
        wins: Dict[str, float],
        request: RobustnessRequestV2,
    ) -> List[OptionResult]:
        """Compute distribution statistics for each option."""
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

            results.append(
                OptionResult(
                    option_id=option.id,
                    outcome_distribution=OutcomeDistribution(
                        mean=float(np.mean(samples_array)),
                        std=float(np.std(samples_array)),
                        median=float(np.median(samples_array)),
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                    ),
                    win_probability=wins[option.id] / request.n_samples,
                    probability_of_goal=probability_of_goal,
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
            sensitivities.append({
                "edge_from": edge.from_,
                "edge_to": edge.to,
                "sensitivity_type": "existence",
                "elasticity": existence_sens,
                "interpretation": self._interpret_existence_sensitivity(
                    edge, existence_sens
                ),
            })

            # Magnitude sensitivity
            magnitude_sens = self._compute_magnitude_sensitivity(
                request, edge, baseline_mean, rng, evaluator
            )
            sensitivities.append({
                "edge_from": edge.from_,
                "edge_to": edge.to,
                "sensitivity_type": "magnitude",
                "elasticity": magnitude_sens,
                "interpretation": self._interpret_magnitude_sensitivity(
                    edge, magnitude_sens
                ),
            })

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

        if abs(baseline_mean) < 1e-10:
            return 0.0

        # Elasticity: relative change in outcome for existence change (0 -> 1)
        return (outcome_diff / baseline_mean) if baseline_mean != 0 else 0.0

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

        if abs(baseline_mean) < 1e-10:
            return 0.0

        # Normalize by 2*std range
        return (outcome_diff / baseline_mean) / 2.0 if baseline_mean != 0 else 0.0

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
                    config[edge_key] = rng.normal(edge.strength.mean, edge.strength.std)
                else:
                    config[edge_key] = 0.0
            else:
                # Sample normally
                if rng.bernoulli(edge.exists_probability):
                    config[edge_key] = rng.normal(edge.strength.mean, edge.strength.std)
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
                config[edge_key] = rng.normal(
                    edge.strength.mean + shift, edge.strength.std
                )
            else:
                # OTHER EDGES: Sample normally (both existence and strength)
                if rng.bernoulli(edge.exists_probability):
                    config[edge_key] = rng.normal(edge.strength.mean, edge.strength.std)
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
        """
        if not request.parameter_uncertainties:
            return []

        sensitivities = []
        ref_option = request.options[0]
        baseline_mean = np.mean(baseline_outcomes[ref_option.id])

        # Build node map for labels
        node_map = {n.id: n for n in request.graph.nodes}

        # Sample mean edge configuration for sensitivity analysis
        # (isolate factor sensitivity from edge uncertainty)
        mean_edge_config = {
            (e.from_, e.to): e.strength.mean * e.exists_probability
            for e in request.graph.edges
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
                sensitivities.append({
                    "node_id": uncertainty.node_id,
                    "node_label": node.label,
                    "elasticity": 0.0,
                    "observed_value": observed_value,
                    "interpretation": f"Factor {node.label} has no uncertainty (point mass)",
                })
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
            if abs(baseline_mean) < 1e-10:
                elasticity = 0.0
            else:
                pct_outcome_change = outcome_diff / baseline_mean
                pct_factor_change = (2 * delta) / mean_value if abs(mean_value) > 1e-10 else 1.0
                elasticity = pct_outcome_change / pct_factor_change if abs(pct_factor_change) > 1e-10 else 0.0

            sensitivities.append({
                "node_id": uncertainty.node_id,
                "node_label": node.label,
                "elasticity": elasticity,
                "observed_value": observed_value,
                "interpretation": self._interpret_factor_sensitivity(
                    node.label, elasticity
                ),
            })

        # Sort by absolute elasticity
        sensitivities.sort(key=lambda x: abs(x["elasticity"]), reverse=True)

        # Convert to results with ranks
        results = []
        for i, s in enumerate(sensitivities):
            results.append(
                FactorSensitivityResult(
                    node_id=s["node_id"],
                    node_label=s["node_label"],
                    elasticity=s["elasticity"],
                    importance_rank=i + 1,
                    observed_value=s["observed_value"],
                    interpretation=s["interpretation"],
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

    def _compute_robustness(
        self,
        option_wins: Dict[str, float],
        winner_per_sample: List[str],
        sensitivity: List[SensitivityResult],
        request: RobustnessRequestV2,
        edge_configs_per_sample: List[Dict[Tuple[str, str], float]],
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

        # Compute alternative winners for fragile edges
        fragile_edges_enhanced = self._compute_alternative_winners(
            fragile_edge_info,
            edge_configs_per_sample,
            winner_per_sample,
            most_frequent_winner,
        )

        # Overall robustness
        is_robust = (
            recommendation_stability >= self.ROBUST_THRESHOLD
            and len(fragile_edges) == 0
        )

        # Confidence based on sample size and stability
        confidence = min(0.99, recommendation_stability * (1 - 1 / np.sqrt(n_samples)))

        # Interpretation
        if is_robust:
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
    ) -> List[FragileEdgeEnhanced]:
        """
        Compute alternative winners for fragile edges.

        For each fragile edge, identifies which option wins most often when
        the edge is "weak" (bottom 25% of sampled strengths).

        Args:
            fragile_edge_info: Map of edge_id -> (from_id, to_id)
            edge_configs_per_sample: Edge strengths for each MC sample
            winner_per_sample: Winner option ID for each MC sample
            overall_winner: The overall recommended option

        Returns:
            List of FragileEdgeEnhanced objects with enhanced fragile edge information
        """
        results = []

        for edge_id, (from_id, to_id) in fragile_edge_info.items():
            edge_key = (from_id, to_id)

            # Collect edge strengths across all samples
            strengths = [
                config.get(edge_key, 0.0) for config in edge_configs_per_sample
            ]

            if not strengths:
                # No data for this edge
                results.append(FragileEdgeEnhanced(
                    edge_id=edge_id,
                    from_id=from_id,
                    to_id=to_id,
                    alternative_winner_id=None,
                    switch_probability=None,
                ))
                continue

            # Find bottom 25% threshold (weak edge samples)
            strength_array = np.array(strengths)
            weak_threshold = np.percentile(strength_array, 25)

            # Get samples where edge is weak
            weak_sample_indices = [
                i for i, s in enumerate(strengths) if s <= weak_threshold
            ]

            if not weak_sample_indices:
                results.append(FragileEdgeEnhanced(
                    edge_id=edge_id,
                    from_id=from_id,
                    to_id=to_id,
                    alternative_winner_id=None,
                    switch_probability=None,
                ))
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
                    opt: count for opt, count in weak_winner_counts.items()
                    if opt != overall_winner
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

            results.append(FragileEdgeEnhanced(
                edge_id=edge_id,
                from_id=from_id,
                to_id=to_id,
                alternative_winner_id=alternative_winner_id,
                switch_probability=switch_probability,
            ))

        return results
