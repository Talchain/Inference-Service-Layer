"""
FACET-based robustness analyzer with dual uncertainty support (v2.2).

Implements Monte Carlo robustness analysis that samples both:
- Structural uncertainty: Edge existence (Bernoulli)
- Parametric uncertainty: Effect magnitude (Normal)

This enables answering:
- "Is my decision robust to uncertainty about whether this relationship exists?"
- "Is my decision robust to the effect being stronger/weaker than estimated?"
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.robustness_v2 import (
    ClampMetrics,
    EdgeV2,
    GraphV2,
    InterventionOption,
    OptionResult,
    OutcomeDistribution,
    ResponseMetadataV2,
    RobustnessRequestV2,
    RobustnessResponseV2,
    RobustnessResult,
    SensitivityResult,
)
from src.utils.rng import SeededRNG, compute_seed_from_graph
from src.__version__ import __version__
from src.models.metadata import generate_config_fingerprint

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, graph: GraphV2):
        """
        Initialize evaluator.

        Args:
            graph: Causal graph structure
        """
        self.graph = graph
        self._node_order = self._compute_topological_order()
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, List[str]] = defaultdict(list)

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

        Returns:
            Value at goal_node
        """
        if base_values is None:
            base_values = {}

        node_values: Dict[str, float] = {}

        for node_id in self._node_order:
            if node_id in interventions:
                # Interventional value overrides structural equations
                node_values[node_id] = interventions[node_id]
            else:
                # Compute from parents
                base = base_values.get(node_id, 0.0)
                parents_contribution = 0.0

                for parent in self._parents[node_id]:
                    edge_key = (parent, node_id)
                    strength = edge_strengths.get(edge_key, 0.0)
                    parent_value = node_values.get(parent, 0.0)
                    parents_contribution += parent_value * strength

                node_values[node_id] = base + parents_contribution

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

        # Setup
        seed = request.seed or compute_seed_from_graph(request.graph)
        rng = SeededRNG(seed)
        sampler = DualUncertaintySampler(request.graph.edges, rng)
        evaluator = SCMEvaluatorV2(request.graph)

        self.logger.info(
            "robustness_v2_analysis_started",
            extra={
                "request_id": request.request_id,
                "n_samples": request.n_samples,
                "n_options": len(request.options),
                "n_edges": len(request.graph.edges),
                "seed": seed,
            },
        )

        # Run Monte Carlo simulation
        option_outcomes, option_wins, winner_per_sample = self._run_monte_carlo(
            request, sampler, evaluator
        )

        # Compute results
        results = self._compute_option_results(
            option_outcomes, option_wins, request
        )

        # Compute sensitivity if requested
        sensitivity = []
        if "sensitivity" in request.analysis_types:
            sensitivity = self._compute_sensitivity(
                request, option_outcomes, sampler, rng, evaluator
            )

        # Compute robustness assessment
        robustness = self._compute_robustness(
            option_wins, winner_per_sample, sensitivity, request
        )

        execution_time = int((time.time() - start_time) * 1000)

        # Find recommended option
        recommended_option_id = max(option_wins, key=option_wins.get)
        recommendation_confidence = option_wins[recommended_option_id] / request.n_samples

        response = RobustnessResponseV2(
            request_id=request.request_id,
            results=results,
            recommended_option_id=recommended_option_id,
            recommendation_confidence=recommendation_confidence,
            sensitivity=sensitivity,
            robustness=robustness,
            metadata=ResponseMetadataV2(
                isl_version=__version__,
                n_samples_used=request.n_samples,
                seed_used=seed,
                execution_time_ms=execution_time,
                edge_existence_rates=sampler.get_existence_rates(),
                config_fingerprint=generate_config_fingerprint(),
            ),
        )

        self.logger.info(
            "robustness_v2_analysis_complete",
            extra={
                "request_id": request.request_id,
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
        evaluator: SCMEvaluatorV2,
    ) -> Tuple[Dict[str, List[float]], Dict[str, int], List[str]]:
        """
        Run Monte Carlo simulation.

        Returns:
            (option_outcomes, option_wins, winner_per_sample)
        """
        option_outcomes: Dict[str, List[float]] = {
            opt.id: [] for opt in request.options
        }
        option_wins: Dict[str, int] = {opt.id: 0 for opt in request.options}
        winner_per_sample: List[str] = []

        for _ in range(request.n_samples):
            # Sample edge configuration
            edge_config = sampler.sample_edge_configuration()

            # Evaluate each option
            sample_outcomes = {}
            for option in request.options:
                outcome = evaluator.evaluate(
                    edge_strengths=edge_config,
                    interventions=option.interventions,
                    goal_node=request.goal_node_id,
                )
                option_outcomes[option.id].append(outcome)
                sample_outcomes[option.id] = outcome

            # Track winner (highest outcome)
            winner = max(sample_outcomes, key=sample_outcomes.get)
            option_wins[winner] += 1
            winner_per_sample.append(winner)

        return option_outcomes, option_wins, winner_per_sample

    def _compute_option_results(
        self,
        outcomes: Dict[str, List[float]],
        wins: Dict[str, int],
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
                )
            )

        return results

    def _compute_confidence_interval(
        self, samples: np.ndarray, confidence_level: float
    ) -> Tuple[float, float]:
        """Compute confidence interval from samples."""
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
        """Sample edge configuration with one edge's mean shifted."""
        config = {}

        for edge in edges:
            edge_key = (edge.from_, edge.to)

            if rng.bernoulli(edge.exists_probability):
                if edge.from_ == target_edge.from_ and edge.to == target_edge.to:
                    # Shift this edge's mean
                    config[edge_key] = rng.normal(
                        edge.strength.mean + shift, edge.strength.std
                    )
                else:
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

    def _compute_robustness(
        self,
        option_wins: Dict[str, int],
        winner_per_sample: List[str],
        sensitivity: List[SensitivityResult],
        request: RobustnessRequestV2,
    ) -> RobustnessResult:
        """Compute overall robustness assessment."""
        # Recommendation stability: fraction of samples with same winner
        n_samples = request.n_samples
        most_frequent_winner = max(option_wins, key=option_wins.get)
        recommendation_stability = option_wins[most_frequent_winner] / n_samples

        # Identify fragile and robust edges
        fragile_edges = []
        robust_edges = []

        for sens in sensitivity:
            edge_id = f"{sens.edge_from}->{sens.edge_to}"
            if abs(sens.elasticity) > self.FRAGILE_THRESHOLD:
                if edge_id not in fragile_edges:
                    fragile_edges.append(edge_id)
            elif abs(sens.elasticity) < 0.05:
                if edge_id not in robust_edges:
                    robust_edges.append(edge_id)

        # Remove duplicates (edges appear twice: existence + magnitude)
        fragile_edges = list(set(fragile_edges))
        robust_edges = list(set(robust_edges))

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
            robust_edges=robust_edges,
            recommendation_stability=recommendation_stability,
            interpretation=interpretation,
        )
