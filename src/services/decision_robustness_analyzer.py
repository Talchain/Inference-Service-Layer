"""
Decision Robustness Suite Analyzer.

Implements unified robustness analysis combining:
- Sensitivity analysis (parameter impact)
- Robustness bounds (flip thresholds)
- Value of Information (EVPI/EVSI)
- Pareto frontier (multi-goal)
- Narrative generation

Brief 7: ISL — Decision Robustness Suite
"""

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.config import get_settings
from src.models.decision_robustness import (
    AnalysisOptions,
    ConfidenceLevelEnum,
    DecisionOption,
    ImpactDirectionEnum,
    ParetoPoint,
    ParetoResult,
    RankedOption,
    Recommendation,
    RecommendationStatusEnum,
    RobustnessBound,
    RobustnessLabelEnum,
    RobustnessRequest,
    RobustnessResult,
    SensitiveParameter,
    UtilityDistribution,
    UtilitySpecification,
    ValueOfInformation,
)
from src.models.shared import GraphV1
from src.utils.cache import get_cache
from src.utils.determinism import make_deterministic
from src.utils.rng import SeededRNG

logger = logging.getLogger(__name__)

# Cache for simulation traces
_robustness_cache = get_cache("decision_robustness", max_size=500, ttl=3600)


class DecisionRobustnessAnalyzer:
    """
    Unified robustness analyzer for decision support.

    Computes sensitivity, robustness bounds, VoI, and Pareto frontier
    in a single analysis call. Designed for performance (<5s typical).
    """

    # Robustness thresholds
    ROBUST_THRESHOLD = 0.5  # ±50% required to flip
    MODERATE_THRESHOLD = 0.2  # ±20% required to flip
    # Below 20% = fragile

    def __init__(
        self,
        settings: Optional[Any] = None,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            settings: Application settings (optional)
        """
        self.settings = settings or get_settings()
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _validate_request(
        self,
        request: RobustnessRequest,
        request_id: str,
    ) -> None:
        """
        Validate request inputs (Brief 20 Task 3).

        Checks for:
        - Empty options list
        - Single option (needs 2+ for comparison)
        - Missing goal node
        - Circular graph
        - Disconnected nodes (warning only)

        Args:
            request: The robustness request
            request_id: Request ID for logging

        Raises:
            ValueError: If validation fails
        """
        # Check options count
        if not request.options or len(request.options) == 0:
            logger.warning(
                "validation_failed",
                extra={"request_id": request_id, "reason": "empty_options"},
            )
            raise ValueError(
                "Options list is empty. At least 2 options are required for "
                "robustness analysis to compare alternatives."
            )

        if len(request.options) == 1:
            logger.warning(
                "validation_failed",
                extra={"request_id": request_id, "reason": "single_option"},
            )
            raise ValueError(
                "Only 1 option provided. Robustness analysis requires at least "
                "2 options to compare. Consider adding a baseline/status quo option."
            )

        # Check goal node exists in graph
        node_ids = {node.id for node in request.graph.nodes}
        goal_id = request.utility.goal_node_id
        if goal_id not in node_ids:
            logger.warning(
                "validation_failed",
                extra={
                    "request_id": request_id,
                    "reason": "missing_goal_node",
                    "goal_id": goal_id,
                },
            )
            raise ValueError(
                f"Goal node '{goal_id}' not found in graph. "
                f"Available nodes: {sorted(node_ids)}"
            )

        # Check additional goals exist
        if request.utility.additional_goals:
            missing_goals = [g for g in request.utility.additional_goals if g not in node_ids]
            if missing_goals:
                logger.warning(
                    "validation_failed",
                    extra={
                        "request_id": request_id,
                        "reason": "missing_additional_goals",
                        "missing": missing_goals,
                    },
                )
                raise ValueError(
                    f"Additional goal nodes not found: {missing_goals}. "
                    f"Available nodes: {sorted(node_ids)}"
                )

        # Check for cycles (simple detection)
        if self._has_cycle(request.graph):
            logger.warning(
                "validation_failed",
                extra={"request_id": request_id, "reason": "cyclic_graph"},
            )
            raise ValueError(
                "Graph contains a cycle. Robustness analysis requires a "
                "directed acyclic graph (DAG) for causal inference."
            )

        # Check intervention nodes exist (warning only, don't fail)
        for option in request.options:
            missing_interventions = [
                k for k in option.interventions.keys() if k not in node_ids
            ]
            if missing_interventions:
                logger.warning(
                    "validation_warning",
                    extra={
                        "request_id": request_id,
                        "reason": "missing_intervention_nodes",
                        "option": option.id,
                        "missing": missing_interventions,
                    },
                )

        # Check for disconnected nodes (warning only)
        connected = self._get_connected_nodes(request.graph, goal_id)
        disconnected = node_ids - connected
        if disconnected:
            logger.info(
                "validation_info",
                extra={
                    "request_id": request_id,
                    "reason": "disconnected_nodes",
                    "disconnected": list(disconnected),
                },
            )

    def _has_cycle(self, graph: GraphV1) -> bool:
        """Check if graph has a cycle using DFS."""
        adjacency = {}
        for node in graph.nodes:
            adjacency[node.id] = []
        for edge in graph.edges:
            adjacency[edge.from_].append(edge.to)

        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for node in adjacency:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def _get_connected_nodes(self, graph: GraphV1, goal_id: str) -> set:
        """Get all nodes connected to the goal node (ancestors)."""
        reverse_adj = {}
        for node in graph.nodes:
            reverse_adj[node.id] = []
        for edge in graph.edges:
            reverse_adj[edge.to].append(edge.from_)

        connected = {goal_id}
        queue = [goal_id]
        while queue:
            current = queue.pop(0)
            for parent in reverse_adj.get(current, []):
                if parent not in connected:
                    connected.add(parent)
                    queue.append(parent)
        return connected

    def analyze(
        self,
        request: RobustnessRequest,
        request_id: str,
    ) -> RobustnessResult:
        """
        Perform unified robustness analysis.

        Args:
            request: Robustness analysis request
            request_id: Request ID for tracing

        Returns:
            Complete RobustnessResult with all metrics

        Raises:
            ValueError: If validation fails
        """
        start_time = time.time()

        # Input validation (Brief 20 Task 3)
        self._validate_request(request, request_id)

        # Create per-request RNG for thread-safe determinism
        rng = make_deterministic(request.model_dump())

        options = request.analysis_options or AnalysisOptions()
        timeout_sec = (options.timeout_ms or 5000) / 1000

        # Track completed and skipped analyses
        completed_analyses: List[str] = []
        skipped_analyses: List[str] = []

        logger.info(
            "decision_robustness_analysis_started",
            extra={
                "request_id": request_id,
                "num_options": len(request.options),
                "num_nodes": len(request.graph.nodes),
                "num_edges": len(request.graph.edges),
                "goal_node": request.utility.goal_node_id,
                "timeout_ms": options.timeout_ms,
                "monte_carlo_samples": options.monte_carlo_samples,
            },
        )

        try:
            # Step 1: Build graph model for simulation
            graph_model = self._build_graph_model(request.graph)

            # Step 2: Compute utility for all options
            step_start = time.time()
            option_utilities = self._compute_all_utilities(
                request.options,
                graph_model,
                request.utility,
                options.monte_carlo_samples,
                rng,
            )
            completed_analyses.append("rankings")
            logger.debug(
                "analysis_step_completed",
                extra={"step": "rankings", "elapsed_ms": (time.time() - step_start) * 1000},
            )

            # Step 3: Rank options and create recommendation
            option_rankings = self._rank_options(option_utilities, request.options)
            recommendation = self._create_recommendation(
                option_rankings,
                graph_model,
                request.utility,
            )

            # Step 4: Sensitivity analysis
            step_start = time.time()
            sensitivity_params = self._compute_sensitivity(
                option_rankings[0],  # Top option
                graph_model,
                request.utility,
                options.sensitivity_top_n,
                options.perturbation_range,
                rng,
            )
            completed_analyses.append("sensitivity")
            logger.debug(
                "analysis_step_completed",
                extra={"step": "sensitivity", "elapsed_ms": (time.time() - step_start) * 1000},
            )

            # Step 5: Robustness bounds
            step_start = time.time()
            robustness_bounds = self._compute_robustness_bounds(
                option_rankings,
                graph_model,
                request.utility,
                sensitivity_params,
                options.perturbation_range,
                rng,
            )
            completed_analyses.append("robustness_bounds")
            logger.debug(
                "analysis_step_completed",
                extra={"step": "robustness_bounds", "elapsed_ms": (time.time() - step_start) * 1000},
            )

            # Step 6: Determine robustness label
            robustness_label, robustness_summary = self._classify_robustness(
                robustness_bounds,
                sensitivity_params,
            )

            # Check timeout before expensive operations
            elapsed_so_far = (time.time() - start_time) * 1000
            time_remaining_ms = (timeout_sec * 1000) - elapsed_so_far

            # Step 7: Value of Information (if enabled and time permits)
            voi_results: List[ValueOfInformation] = []
            if options.include_voi and request.parameter_uncertainties:
                if time_remaining_ms > 1000:  # Need at least 1s for VoI
                    step_start = time.time()
                    voi_results = self._compute_value_of_information(
                        option_rankings,
                        graph_model,
                        request.utility,
                        request.parameter_uncertainties,
                        options.sample_sizes_for_evsi,
                        rng,
                    )
                    completed_analyses.append("voi")
                    logger.debug(
                        "analysis_step_completed",
                        extra={"step": "voi", "elapsed_ms": (time.time() - step_start) * 1000},
                    )
                else:
                    skipped_analyses.append("voi")
                    logger.info(
                        "analysis_step_skipped",
                        extra={"step": "voi", "reason": "timeout_approaching"},
                    )
            elif not options.include_voi:
                skipped_analyses.append("voi")
            elif not request.parameter_uncertainties:
                skipped_analyses.append("voi")

            # Step 8: Pareto frontier (if multi-goal, enabled, and time permits)
            pareto_result: Optional[ParetoResult] = None
            if (
                options.include_pareto
                and request.utility.additional_goals
                and len(request.utility.additional_goals) > 0
            ):
                elapsed_so_far = (time.time() - start_time) * 1000
                time_remaining_ms = (timeout_sec * 1000) - elapsed_so_far
                if time_remaining_ms > 500:  # Need at least 500ms for Pareto
                    step_start = time.time()
                    pareto_result = self._compute_pareto_frontier(
                        request.options,
                        graph_model,
                        request.utility,
                        option_rankings[0].option_id,
                        rng,
                    )
                    completed_analyses.append("pareto")
                    logger.debug(
                        "analysis_step_completed",
                        extra={"step": "pareto", "elapsed_ms": (time.time() - step_start) * 1000},
                    )
                else:
                    skipped_analyses.append("pareto")
                    logger.info(
                        "analysis_step_skipped",
                        extra={"step": "pareto", "reason": "timeout_approaching"},
                    )
            elif not options.include_pareto:
                skipped_analyses.append("pareto")
            else:
                skipped_analyses.append("pareto")

            # Step 9: Generate narrative (improved quality - Brief 20 Task 4)
            narrative = self._generate_narrative(
                robustness_label,
                sensitivity_params,
                robustness_bounds,
                voi_results,
                recommendation,
            )

            # Update recommendation status based on robustness
            if robustness_label == RobustnessLabelEnum.FRAGILE:
                recommendation = Recommendation(
                    option_id=recommendation.option_id,
                    option_label=recommendation.option_label,
                    confidence=ConfidenceLevelEnum.LOW,
                    recommendation_status=RecommendationStatusEnum.EXPLORATORY,
                )

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                "decision_robustness_analysis_completed",
                extra={
                    "request_id": request_id,
                    "elapsed_ms": elapsed_ms,
                    "completed_analyses": completed_analyses,
                    "skipped_analyses": skipped_analyses,
                    "robustness_label": robustness_label.value,
                    "top_option": option_rankings[0].option_id,
                },
            )

            return RobustnessResult(
                option_rankings=option_rankings,
                recommendation=recommendation,
                sensitivity=sensitivity_params,
                robustness_label=robustness_label,
                robustness_summary=robustness_summary,
                robustness_bounds=robustness_bounds,
                value_of_information=voi_results,
                pareto=pareto_result,
                narrative=narrative,
                partial=len(skipped_analyses) > 0 and "voi" not in skipped_analyses,
                completed_analyses=completed_analyses,
                skipped_analyses=skipped_analyses,
                elapsed_ms=elapsed_ms,
            )

        except FuturesTimeoutError:
            logger.warning(
                "decision_robustness_timeout",
                extra={"request_id": request_id, "timeout_sec": timeout_sec},
            )
            # Return partial results
            return self._build_partial_result(
                request, graph_model if 'graph_model' in dir() else {}, rng
            )

        except Exception as e:
            logger.error(
                "decision_robustness_error",
                exc_info=True,
                extra={"request_id": request_id},
            )
            raise

    def _build_graph_model(self, graph: GraphV1) -> Dict[str, Any]:
        """
        Build an internal graph model for simulation.

        Extracts node parameters and edge weights into a format
        suitable for Monte Carlo simulation.

        Args:
            graph: GraphV1 input

        Returns:
            Internal graph model dictionary
        """
        model = {
            "nodes": {},
            "edges": [],
            "adjacency": {},  # node_id -> list of (target_id, weight)
            "reverse_adjacency": {},  # node_id -> list of (source_id, weight)
        }

        # Process nodes
        for node in graph.nodes:
            node_data = {
                "id": node.id,
                "kind": node.kind.value if hasattr(node.kind, 'value') else node.kind,
                "label": node.label or node.id,
                "belief": node.belief,
                "metadata": node.metadata or {},
            }
            model["nodes"][node.id] = node_data
            model["adjacency"][node.id] = []
            model["reverse_adjacency"][node.id] = []

        # Process edges
        for edge in graph.edges:
            # Handle alias: GraphEdgeV1 uses from_ internally but exposes as 'from'
            from_node = edge.from_
            to_node = edge.to
            weight = edge.weight if edge.weight is not None else 1.0

            edge_data = {
                "from": from_node,
                "to": to_node,
                "weight": weight,
                "label": edge.label,
            }
            model["edges"].append(edge_data)
            model["adjacency"][from_node].append((to_node, weight))
            model["reverse_adjacency"][to_node].append((from_node, weight))

        return model

    def _compute_all_utilities(
        self,
        options: List[DecisionOption],
        graph_model: Dict[str, Any],
        utility_spec: UtilitySpecification,
        n_samples: int,
        rng: SeededRNG,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute utility distributions for all options.

        Uses Monte Carlo simulation to propagate uncertainty
        through the graph structure.

        Args:
            options: Decision options
            graph_model: Internal graph model
            utility_spec: Utility specification
            n_samples: Number of MC samples
            rng: Per-request random number generator

        Returns:
            Dict mapping option_id to utility statistics
        """
        results = {}

        for i, option in enumerate(options):
            # Spawn independent RNG for each option
            option_rng = rng.spawn()

            # Simulate utility distribution
            samples = self._simulate_option_utility(
                option,
                graph_model,
                utility_spec,
                n_samples,
                option_rng,
            )

            # Compute statistics
            results[option.id] = {
                "expected": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "p5": float(np.percentile(samples, 5)),
                "p25": float(np.percentile(samples, 25)),
                "p50": float(np.percentile(samples, 50)),
                "p75": float(np.percentile(samples, 75)),
                "p95": float(np.percentile(samples, 95)),
                "samples": samples,
                "is_baseline": option.is_baseline,
            }

        return results

    def _simulate_option_utility(
        self,
        option: DecisionOption,
        graph_model: Dict[str, Any],
        utility_spec: UtilitySpecification,
        n_samples: int,
        rng: SeededRNG,
    ) -> np.ndarray:
        """
        Simulate utility for a single option using Monte Carlo.

        Propagates interventions through the graph structure,
        adding noise based on edge weights and node uncertainties.

        Args:
            option: Decision option
            graph_model: Internal graph model
            utility_spec: Utility specification
            n_samples: Number of samples
            rng: Per-request random number generator

        Returns:
            Array of utility samples
        """
        samples = np.zeros(n_samples)

        # Get goal node
        goal_node_id = utility_spec.goal_node_id

        for i in range(n_samples):
            # Start with intervention values
            node_values = dict(option.interventions)

            # Propagate through graph (topological order approximation)
            # Simple forward propagation
            visited = set(option.interventions.keys())
            queue = list(option.interventions.keys())

            while queue:
                current = queue.pop(0)

                # Get downstream nodes
                for target_id, weight in graph_model["adjacency"].get(current, []):
                    if target_id not in visited:
                        # Compute target value from all incoming edges
                        incoming_sum = 0.0
                        all_sources_ready = True

                        for source_id, src_weight in graph_model["reverse_adjacency"].get(target_id, []):
                            if source_id in node_values:
                                # Add weighted contribution with noise
                                noise = rng.normal(0, 0.1 * abs(src_weight))
                                incoming_sum += node_values[source_id] * src_weight + noise
                            else:
                                all_sources_ready = False

                        if all_sources_ready:
                            # Add node-specific baseline
                            node_belief = graph_model["nodes"].get(target_id, {}).get("belief", 0.5)
                            baseline = node_belief * 100 if node_belief else 50

                            node_values[target_id] = baseline + incoming_sum
                            visited.add(target_id)
                            queue.append(target_id)

            # Extract goal value
            if goal_node_id in node_values:
                samples[i] = node_values[goal_node_id]
            else:
                # Goal not reached - use default
                samples[i] = 0.0

        # Handle additional goals if weighted
        if utility_spec.additional_goals and utility_spec.weights:
            # Multi-goal weighted utility (simplified)
            primary_weight = utility_spec.weights.get(goal_node_id, 1.0)
            samples = samples * primary_weight

        return samples

    def _rank_options(
        self,
        utilities: Dict[str, Dict[str, Any]],
        options: List[DecisionOption],
    ) -> List[RankedOption]:
        """
        Rank options by expected utility.

        Args:
            utilities: Utility statistics for each option
            options: Original option definitions

        Returns:
            List of RankedOption sorted by rank
        """
        # Find baseline option
        baseline_id = None
        baseline_utility = None
        for opt in options:
            if opt.is_baseline:
                baseline_id = opt.id
                baseline_utility = utilities[opt.id]["expected"]
                break

        # If no explicit baseline, use lowest utility option
        if baseline_id is None:
            baseline_id = min(utilities.keys(), key=lambda k: utilities[k]["expected"])
            baseline_utility = utilities[baseline_id]["expected"]

        # Sort by expected utility (descending)
        sorted_options = sorted(
            options,
            key=lambda o: utilities[o.id]["expected"],
            reverse=True,
        )

        ranked = []
        for rank, opt in enumerate(sorted_options, 1):
            util = utilities[opt.id]

            vs_baseline = None
            vs_baseline_pct = None
            if baseline_utility is not None and baseline_utility != 0:
                vs_baseline = util["expected"] - baseline_utility
                vs_baseline_pct = (vs_baseline / abs(baseline_utility)) * 100

            ranked.append(
                RankedOption(
                    option_id=opt.id,
                    option_label=opt.label,
                    expected_utility=util["expected"],
                    utility_distribution=UtilityDistribution(
                        p5=util["p5"],
                        p25=util["p25"],
                        p50=util["p50"],
                        p75=util["p75"],
                        p95=util["p95"],
                    ),
                    rank=rank,
                    vs_baseline=vs_baseline,
                    vs_baseline_pct=vs_baseline_pct,
                )
            )

        return ranked

    def _create_recommendation(
        self,
        rankings: List[RankedOption],
        graph_model: Dict[str, Any],
        utility_spec: UtilitySpecification,
    ) -> Recommendation:
        """
        Create top recommendation with confidence assessment.

        Args:
            rankings: Ranked options
            graph_model: Internal graph model
            utility_spec: Utility specification

        Returns:
            Recommendation object
        """
        top = rankings[0]
        second = rankings[1] if len(rankings) > 1 else None

        # Assess confidence based on gap to second place
        if second:
            gap_pct = abs(top.expected_utility - second.expected_utility) / max(
                abs(top.expected_utility), 1e-10
            )

            if gap_pct > 0.2:  # >20% gap
                confidence = ConfidenceLevelEnum.HIGH
            elif gap_pct > 0.1:  # 10-20% gap
                confidence = ConfidenceLevelEnum.MEDIUM
            else:
                confidence = ConfidenceLevelEnum.LOW
        else:
            confidence = ConfidenceLevelEnum.HIGH  # Only one option

        # Default to actionable - will be updated if fragile
        return Recommendation(
            option_id=top.option_id,
            option_label=top.option_label,
            confidence=confidence,
            recommendation_status=RecommendationStatusEnum.ACTIONABLE,
        )

    def _compute_sensitivity(
        self,
        top_option: RankedOption,
        graph_model: Dict[str, Any],
        utility_spec: UtilitySpecification,
        top_n: int,
        perturbation_range: float,
        rng: SeededRNG,
    ) -> List[SensitiveParameter]:
        """
        Compute parameter sensitivity using perturbation analysis.

        For each edge weight and node parameter, computes the
        sensitivity of expected utility to parameter changes.

        Args:
            top_option: Top ranked option
            graph_model: Internal graph model
            utility_spec: Utility specification
            top_n: Number of top parameters to return
            perturbation_range: Perturbation magnitude
            rng: Per-request random number generator

        Returns:
            List of most sensitive parameters
        """
        sensitivities = []
        base_utility = top_option.expected_utility

        # Analyze edge weight sensitivity
        for edge in graph_model["edges"]:
            edge_id = f"edge_{edge['from']}_{edge['to']}"
            current_weight = edge["weight"]

            # Perturb up
            edge["weight"] = current_weight * (1 + perturbation_range)
            perturbed_model = graph_model.copy()
            utility_up = self._quick_utility_estimate(
                perturbed_model, utility_spec.goal_node_id, rng.spawn()
            )

            # Perturb down
            edge["weight"] = current_weight * (1 - perturbation_range)
            utility_down = self._quick_utility_estimate(
                perturbed_model, utility_spec.goal_node_id, rng.spawn()
            )

            # Restore
            edge["weight"] = current_weight

            # Calculate sensitivity
            delta = (utility_up - utility_down) / 2
            sensitivity_score = abs(delta) / max(abs(base_utility), 1e-10)

            # Determine impact direction
            if delta > 0:
                direction = ImpactDirectionEnum.POSITIVE
                desc = f"Increasing {edge['from']}→{edge['to']} connection improves outcome"
            else:
                direction = ImpactDirectionEnum.NEGATIVE
                desc = f"Increasing {edge['from']}→{edge['to']} connection decreases outcome"

            sensitivities.append(
                SensitiveParameter(
                    parameter_id=edge_id,
                    parameter_label=f"{edge['from']} → {edge['to']} weight",
                    sensitivity_score=min(1.0, sensitivity_score),
                    current_value=current_weight,
                    impact_direction=direction,
                    description=desc,
                )
            )

        # Sort by sensitivity and return top N
        sensitivities.sort(key=lambda s: s.sensitivity_score, reverse=True)
        return sensitivities[:top_n]

    def _quick_utility_estimate(
        self,
        graph_model: Dict[str, Any],
        goal_node_id: str,
        rng: SeededRNG,
        n_samples: int = 100,
    ) -> float:
        """
        Quick utility estimate for sensitivity analysis.

        Uses fewer samples for speed.

        Args:
            graph_model: Internal graph model
            goal_node_id: Goal node ID
            rng: Per-request random number generator
            n_samples: Number of samples

        Returns:
            Estimated utility value
        """
        # Simple propagation starting from random initial values
        samples = []

        for _ in range(n_samples):
            values = {}

            # Initialize all nodes with baseline
            for node_id, node_data in graph_model["nodes"].items():
                belief = node_data.get("belief", 0.5)
                values[node_id] = (belief or 0.5) * 100 + rng.normal(0, 10)

            # Propagate edges
            for edge in graph_model["edges"]:
                from_val = values.get(edge["from"], 50)
                weight = edge["weight"]
                contribution = from_val * weight * 0.01  # Scale factor

                if edge["to"] in values:
                    values[edge["to"]] += contribution

            samples.append(values.get(goal_node_id, 50))

        return float(np.mean(samples))

    def _compute_robustness_bounds(
        self,
        rankings: List[RankedOption],
        graph_model: Dict[str, Any],
        utility_spec: UtilitySpecification,
        sensitivity_params: List[SensitiveParameter],
        perturbation_range: float,
        rng: SeededRNG,
    ) -> List[RobustnessBound]:
        """
        Compute robustness bounds (flip thresholds).

        For each sensitive parameter, finds the threshold value
        that would flip the top recommendation.

        Args:
            rankings: Ranked options
            graph_model: Internal graph model
            utility_spec: Utility specification
            sensitivity_params: Most sensitive parameters
            perturbation_range: Maximum perturbation to search
            rng: Per-request random number generator

        Returns:
            List of robustness bounds
        """
        if len(rankings) < 2:
            return []

        bounds = []
        top_option = rankings[0]
        second_option = rankings[1]
        utility_gap = top_option.expected_utility - second_option.expected_utility

        for param in sensitivity_params[:5]:  # Top 5 most sensitive
            # Extract edge info from parameter_id
            if param.parameter_id.startswith("edge_"):
                parts = param.parameter_id.replace("edge_", "").split("_", 1)
                if len(parts) == 2:
                    from_node, to_node = parts

                    # Find flip threshold via binary search
                    flip_pct = self._binary_search_flip(
                        graph_model,
                        from_node,
                        to_node,
                        param.current_value,
                        utility_gap,
                        param.sensitivity_score,
                        perturbation_range,
                        rng,
                    )

                    if flip_pct is not None:
                        flip_threshold = param.current_value * (flip_pct / 100)

                        bounds.append(
                            RobustnessBound(
                                parameter_id=param.parameter_id,
                                parameter_label=param.parameter_label,
                                flip_threshold=abs(flip_threshold),
                                flip_threshold_pct=abs(flip_pct),
                                flip_to_option=second_option.option_id,
                            )
                        )

        return bounds

    def _binary_search_flip(
        self,
        graph_model: Dict[str, Any],
        from_node: str,
        to_node: str,
        current_value: float,
        utility_gap: float,
        sensitivity: float,
        max_pct: float,
        rng: SeededRNG,
    ) -> Optional[float]:
        """
        Binary search to find flip threshold percentage.

        Args:
            graph_model: Internal graph model
            from_node: Source node of edge
            to_node: Target node of edge
            current_value: Current parameter value
            utility_gap: Gap between top two options
            sensitivity: Sensitivity score
            max_pct: Maximum percentage to search
            rng: Per-request random number generator (unused, for API consistency)

        Returns:
            Percentage change required to flip, or None if not within range
        """
        if sensitivity < 0.01:
            return None

        # Estimate flip point using linear approximation
        # delta_utility = sensitivity * delta_param
        # We need delta_utility >= utility_gap
        estimated_pct = (utility_gap / (sensitivity + 0.01)) * 10

        # Clamp to reasonable range
        flip_pct = min(max_pct * 100, max(5.0, estimated_pct))

        if flip_pct > max_pct * 100:
            return None  # Beyond search range

        return flip_pct

    def _classify_robustness(
        self,
        bounds: List[RobustnessBound],
        sensitivity_params: List[SensitiveParameter],
    ) -> Tuple[RobustnessLabelEnum, str]:
        """
        Classify overall robustness and generate summary.

        Classification rules:
        - Robust: No single parameter flip within ±50%
        - Moderate: Flip requires ±20-50% change
        - Fragile: Flip requires <±20% change

        Args:
            bounds: Robustness bounds
            sensitivity_params: Sensitive parameters

        Returns:
            (robustness_label, summary_string)
        """
        if not bounds:
            return (
                RobustnessLabelEnum.ROBUST,
                "Your decision is robust to typical parameter uncertainty. "
                "No parameters would flip the recommendation within tested ranges.",
            )

        # Find minimum flip threshold
        min_flip_pct = min(b.flip_threshold_pct for b in bounds)

        if min_flip_pct >= 50:
            label = RobustnessLabelEnum.ROBUST
            summary = (
                f"Your decision is robust. Even {min_flip_pct:.0f}% changes "
                "to key parameters wouldn't change the recommendation."
            )
        elif min_flip_pct >= 20:
            label = RobustnessLabelEnum.MODERATE
            # Find the parameter with lowest flip threshold
            min_bound = min(bounds, key=lambda b: b.flip_threshold_pct)
            summary = (
                f"Your decision is moderately robust. "
                f"Changing {min_bound.parameter_label} by ~{min_flip_pct:.0f}% "
                f"would flip to {min_bound.flip_to_option}."
            )
        else:
            label = RobustnessLabelEnum.FRAGILE
            min_bound = min(bounds, key=lambda b: b.flip_threshold_pct)
            summary = (
                f"Your decision is fragile. Small changes (~{min_flip_pct:.0f}%) "
                f"to {min_bound.parameter_label} could change the recommendation. "
                "Treat this as exploratory analysis."
            )

        return label, summary

    def _compute_value_of_information(
        self,
        rankings: List[RankedOption],
        graph_model: Dict[str, Any],
        utility_spec: UtilitySpecification,
        uncertainties: Dict[str, Dict[str, float]],
        sample_sizes: List[int],
        rng: SeededRNG,
    ) -> List[ValueOfInformation]:
        """
        Compute Expected Value of Perfect Information (EVPI) and
        Expected Value of Sample Information (EVSI).

        EVPI: Maximum value of resolving all uncertainty about a parameter
        EVSI: Value of partial information from sampling

        Args:
            rankings: Ranked options
            graph_model: Internal graph model
            utility_spec: Utility specification
            uncertainties: Parameter uncertainties {param_id: {mean, std}}
            sample_sizes: Sample sizes for EVSI
            rng: Per-request random number generator

        Returns:
            List of VoI results for each uncertain parameter
        """
        voi_results = []

        for param_id, uncertainty in uncertainties.items():
            mean = uncertainty.get("mean", 0)
            std = uncertainty.get("std", 1)

            # Compute EVPI
            # EVPI = E[max utility with perfect info] - max E[utility under uncertainty]
            evpi = self._compute_evpi(
                param_id, mean, std, graph_model, utility_spec, rng.spawn()
            )

            # Compute EVSI for each sample size
            best_evsi = 0.0
            best_sample_size = sample_sizes[0] if sample_sizes else 50

            for n in sample_sizes:
                evsi = self._compute_evsi(
                    param_id, mean, std, n, graph_model, utility_spec, rng.spawn()
                )
                if evsi > best_evsi:
                    best_evsi = evsi
                    best_sample_size = n

            # Generate recommendation
            if evpi > 10000:
                rec = "High value - consider gathering data"
                suggestion = f"Gather data on {param_id}. Sample size of {best_sample_size} recommended."
            elif evpi > 1000:
                rec = "Moderate value - gather data if convenient"
                suggestion = f"Consider sampling {best_sample_size} observations for {param_id}."
            else:
                rec = "Low value - current information sufficient"
                suggestion = f"Current uncertainty about {param_id} has minimal impact."

            # Get readable label
            param_label = param_id.replace("_", " ").title()

            voi_results.append(
                ValueOfInformation(
                    parameter_id=param_id,
                    parameter_label=param_label,
                    evpi=evpi,
                    evsi=best_evsi,
                    current_uncertainty=std,
                    recommendation=rec,
                    data_collection_suggestion=suggestion,
                )
            )

        # Sort by EVPI descending
        voi_results.sort(key=lambda v: v.evpi, reverse=True)
        return voi_results

    def _compute_evpi(
        self,
        param_id: str,
        mean: float,
        std: float,
        graph_model: Dict[str, Any],
        utility_spec: UtilitySpecification,
        rng: SeededRNG,
    ) -> float:
        """
        Compute EVPI for a parameter.

        EVPI = E[max_a U(a, θ)] - max_a E[U(a, θ)]

        Args:
            param_id: Parameter identifier
            mean: Parameter mean
            std: Parameter standard deviation
            graph_model: Internal graph model
            utility_spec: Utility specification
            rng: Per-request random number generator

        Returns:
            EVPI value
        """
        n_scenarios = 100

        # Sample parameter values
        param_samples = rng.normal_array(mean, std, n_scenarios)

        # For each scenario, compute max utility across actions
        max_utilities_per_scenario = []

        for param_val in param_samples:
            # Simplified: estimate utility impact of parameter
            base_utility = self._quick_utility_estimate(
                graph_model, utility_spec.goal_node_id, rng.spawn(), 50
            )
            # Parameter affects utility proportionally
            utility_with_param = base_utility * (1 + (param_val - mean) / max(abs(mean), 1))
            max_utilities_per_scenario.append(utility_with_param)

        # E[max U with perfect info]
        e_max_with_info = float(np.mean(np.maximum(max_utilities_per_scenario, 0)))

        # max E[U under uncertainty] - just the mean
        max_e_under_uncertainty = float(np.mean(max_utilities_per_scenario))

        evpi = max(0, e_max_with_info - max_e_under_uncertainty)

        return evpi

    def _compute_evsi(
        self,
        param_id: str,
        mean: float,
        std: float,
        sample_size: int,
        graph_model: Dict[str, Any],
        utility_spec: UtilitySpecification,
        rng: SeededRNG,
    ) -> float:
        """
        Compute EVSI for a parameter with given sample size.

        EVSI approximates value of sampling N observations.

        Args:
            param_id: Parameter identifier
            mean: Parameter mean
            std: Parameter standard deviation
            sample_size: Number of samples
            graph_model: Internal graph model
            utility_spec: Utility specification
            rng: Per-request random number generator

        Returns:
            EVSI value
        """
        # EVSI is typically a fraction of EVPI based on sample size
        # Approximation: EVSI ≈ EVPI * (1 - 1/sqrt(n))
        evpi = self._compute_evpi(param_id, mean, std, graph_model, utility_spec, rng)

        reduction_factor = 1 - 1 / np.sqrt(sample_size + 1)
        evsi = evpi * reduction_factor * 0.7  # Conservative factor

        return max(0, evsi)

    def _compute_pareto_frontier(
        self,
        options: List[DecisionOption],
        graph_model: Dict[str, Any],
        utility_spec: UtilitySpecification,
        current_selection: str,
        rng: SeededRNG,
    ) -> ParetoResult:
        """
        Compute Pareto frontier for multi-goal decisions.

        Args:
            options: Decision options
            graph_model: Internal graph model
            utility_spec: Utility specification with multiple goals
            current_selection: Currently recommended option
            rng: Per-request random number generator

        Returns:
            ParetoResult with frontier analysis
        """
        all_goals = [utility_spec.goal_node_id]
        if utility_spec.additional_goals:
            all_goals.extend(utility_spec.additional_goals)

        # Compute goal values for each option
        option_goal_values = {}

        for opt in options:
            goal_values = {}
            for goal in all_goals:
                # Estimate goal value
                value = self._quick_utility_estimate(
                    graph_model, goal, rng.spawn(), 100
                )
                goal_values[goal] = value

            option_goal_values[opt.id] = {
                "label": opt.label,
                "values": goal_values,
            }

        # Find Pareto frontier
        frontier_points = []
        dominated_ids = set()

        # Check dominance
        for opt_id, data in option_goal_values.items():
            is_dominated = False

            for other_id, other_data in option_goal_values.items():
                if opt_id == other_id:
                    continue

                # Check if other dominates this option
                # (at least as good in all goals, strictly better in at least one)
                at_least_as_good = all(
                    other_data["values"][g] >= data["values"][g]
                    for g in all_goals
                )
                strictly_better = any(
                    other_data["values"][g] > data["values"][g]
                    for g in all_goals
                )

                if at_least_as_good and strictly_better:
                    is_dominated = True
                    dominated_ids.add(opt_id)
                    break

            # Generate trade-off description
            if not is_dominated:
                # Compare to best in each goal
                trade_offs = []
                for goal in all_goals:
                    best_for_goal = max(
                        option_goal_values.keys(),
                        key=lambda k: option_goal_values[k]["values"][goal],
                    )
                    if best_for_goal != opt_id:
                        diff_pct = (
                            (option_goal_values[best_for_goal]["values"][goal] - data["values"][goal])
                            / max(abs(data["values"][goal]), 1)
                            * 100
                        )
                        trade_offs.append(f"{diff_pct:.0f}% lower {goal}")

                trade_off_desc = (
                    "Optimal trade-off"
                    if not trade_offs
                    else f"Trade-offs: {', '.join(trade_offs[:2])}"
                )
            else:
                trade_off_desc = "Dominated by other options"

            frontier_points.append(
                ParetoPoint(
                    option_id=opt_id,
                    option_label=data["label"],
                    goal_values=data["values"],
                    is_dominated=is_dominated,
                    trade_off_description=trade_off_desc,
                )
            )

        # Check if current selection is Pareto efficient
        current_pareto_efficient = current_selection not in dominated_ids

        return ParetoResult(
            goals=all_goals,
            frontier_options=frontier_points,
            current_selection_pareto_efficient=current_pareto_efficient,
        )

    def _generate_narrative(
        self,
        robustness_label: RobustnessLabelEnum,
        sensitivity_params: List[SensitiveParameter],
        robustness_bounds: List[RobustnessBound],
        voi_results: List[ValueOfInformation],
        recommendation: Recommendation,
    ) -> str:
        """
        Generate plain-language narrative combining all analyses (Brief 20 Task 4).

        Improved narrative quality:
        - Natural language flow
        - Appropriate confidence hedging
        - Actionable guidance
        - No jargon

        Args:
            robustness_label: Robustness classification
            sensitivity_params: Sensitive parameters
            robustness_bounds: Flip thresholds
            voi_results: VoI analysis
            recommendation: Top recommendation

        Returns:
            Narrative string
        """
        parts = []
        opt_label = recommendation.option_label

        # Opening statement based on robustness
        if robustness_label == RobustnessLabelEnum.ROBUST:
            parts.append(
                f"Based on the analysis, {opt_label} is the recommended choice, "
                "and this recommendation holds up well under uncertainty."
            )
        elif robustness_label == RobustnessLabelEnum.MODERATE:
            parts.append(
                f"Based on the analysis, {opt_label} appears to be the best choice, "
                "though the recommendation depends on some assumptions that may need verification."
            )
        else:
            parts.append(
                f"The analysis suggests {opt_label} as a starting point, "
                "but the recommendation is sensitive to assumptions and should be treated "
                "as exploratory rather than definitive."
            )

        # Sensitivity insight with actionable framing
        if sensitivity_params:
            top_param = sensitivity_params[0]
            direction_word = "improves" if top_param.impact_direction == ImpactDirectionEnum.POSITIVE else "decreases"

            if top_param.sensitivity_score > 0.7:
                parts.append(
                    f"Your assumption about {top_param.parameter_label} has a major influence "
                    f"on the outcome—if it {direction_word}, the expected results change significantly."
                )
            elif top_param.sensitivity_score > 0.4:
                parts.append(
                    f"The outcome is moderately sensitive to {top_param.parameter_label}."
                )

        # Flip threshold with concrete guidance
        if robustness_bounds and robustness_label != RobustnessLabelEnum.ROBUST:
            top_bound = min(robustness_bounds, key=lambda b: b.flip_threshold_pct)
            if top_bound.flip_threshold_pct < 20:
                parts.append(
                    f"Be aware: if {top_bound.parameter_label} turns out to be just "
                    f"{top_bound.flip_threshold_pct:.0f}% different from your estimate, "
                    f"{top_bound.flip_to_option} would become the better choice."
                )
            elif top_bound.flip_threshold_pct < 50:
                parts.append(
                    f"The recommendation would change to {top_bound.flip_to_option} "
                    f"if {top_bound.parameter_label} shifts by about {top_bound.flip_threshold_pct:.0f}%."
                )

        # VoI with actionable suggestion
        if voi_results:
            high_value_voi = [v for v in voi_results if v.evpi > 5000]
            if high_value_voi:
                top_voi = high_value_voi[0]
                parts.append(
                    f"Consider gathering more data on {top_voi.parameter_label}—reducing "
                    f"uncertainty here could be worth up to ${top_voi.evpi:,.0f} in better decisions."
                )
                if top_voi.data_collection_suggestion:
                    parts.append(f"One approach: {top_voi.data_collection_suggestion}")

        # Closing guidance based on status
        if recommendation.recommendation_status == RecommendationStatusEnum.EXPLORATORY:
            parts.append(
                "Given the uncertainty, consider running a small pilot or gathering "
                "more information before committing fully."
            )

        return " ".join(parts)

    def _build_partial_result(
        self,
        request: RobustnessRequest,
        graph_model: Dict[str, Any],
        rng: SeededRNG,
    ) -> RobustnessResult:
        """
        Build partial result when timeout occurs.

        Returns basic ranking and sensitivity without VoI/Pareto.

        Args:
            request: Original request
            graph_model: Internal graph model
            rng: Per-request random number generator (unused in partial result)

        Returns:
            Partial RobustnessResult
        """
        # Create minimal rankings
        rankings = []
        for i, opt in enumerate(request.options):
            rankings.append(
                RankedOption(
                    option_id=opt.id,
                    option_label=opt.label,
                    expected_utility=0.0,
                    utility_distribution=UtilityDistribution(
                        p5=0, p25=0, p50=0, p75=0, p95=0
                    ),
                    rank=i + 1,
                    vs_baseline=None,
                    vs_baseline_pct=None,
                )
            )

        return RobustnessResult(
            option_rankings=rankings,
            recommendation=Recommendation(
                option_id=request.options[0].id,
                option_label=request.options[0].label,
                confidence=ConfidenceLevelEnum.LOW,
                recommendation_status=RecommendationStatusEnum.EXPLORATORY,
            ),
            sensitivity=[],
            robustness_label=RobustnessLabelEnum.FRAGILE,
            robustness_summary="Analysis timed out. Results are incomplete.",
            robustness_bounds=[],
            value_of_information=[],
            pareto=None,
            narrative=(
                "The analysis timed out before completing all calculations. "
                "The option rankings shown are preliminary. For more reliable results, "
                "try simplifying the graph or increasing the timeout."
            ),
            partial=True,
            completed_analyses=["rankings"],
            skipped_analyses=["sensitivity", "robustness_bounds", "voi", "pareto"],
            elapsed_ms=None,
        )


def get_graph_hash(graph: GraphV1) -> str:
    """
    Compute hash of graph structure for caching.

    Args:
        graph: GraphV1 input

    Returns:
        SHA256 hash string
    """
    # Create canonical representation
    nodes_repr = sorted([
        (n.id, n.kind.value if hasattr(n.kind, 'value') else n.kind)
        for n in graph.nodes
    ])
    edges_repr = sorted([
        (e.from_, e.to, e.weight)
        for e in graph.edges
    ])

    canonical = json.dumps({"nodes": nodes_repr, "edges": edges_repr}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
