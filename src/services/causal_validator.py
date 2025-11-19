"""
Causal Validator service using Y₀ for causal identification.

Validates causal models and identifies valid adjustment sets for
estimating causal effects.
"""

import logging
from typing import List, Optional

import networkx as nx
from y0.algorithm.identify import Identification, identify_outcomes
from y0.dsl import P, Variable
from y0.graph import NxMixedGraph

from src.models.requests import CausalValidationRequest
from src.models.responses import (
    CausalValidationResponse,
    ValidationIssue,
)
from src.models.shared import ConfidenceLevel, ValidationIssueType, ValidationStatus
from src.services.explanation_generator import ExplanationGenerator
from src.utils.determinism import canonical_hash, make_deterministic
from src.utils.graph_parser import edge_list_to_networkx, edge_list_to_y0, find_backdoor_paths
from src.utils.validation import validate_dag_structure, validate_node_in_graph

logger = logging.getLogger(__name__)


class CausalValidator:
    """
    Validates causal models using Y₀ library.

    Determines whether causal effects can be identified and provides
    adjustment sets when possible.
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        self.explanation_generator = ExplanationGenerator()

    def validate(self, request: CausalValidationRequest) -> CausalValidationResponse:
        """
        Validate causal model for identifiability.

        Args:
            request: Validation request with DAG and variables

        Returns:
            CausalValidationResponse: Validation results
        """
        # Make computation deterministic
        request_hash = make_deterministic(request.model_dump())

        logger.info(
            "causal_validation_started",
            extra={
                "request_hash": canonical_hash(request.model_dump()),
                "treatment": request.treatment,
                "outcome": request.outcome,
            },
        )

        try:
            # Validate inputs
            validate_dag_structure(request.dag.nodes, request.dag.edges)
            validate_node_in_graph(request.treatment, request.dag.nodes, "Treatment")
            validate_node_in_graph(request.outcome, request.dag.nodes, "Outcome")

            # Convert to NetworkX for analysis
            nx_graph = edge_list_to_networkx(request.dag.nodes, request.dag.edges)

            # Check if there's any path from treatment to outcome
            if not nx.has_path(nx_graph, request.treatment, request.outcome):
                # No causal path exists
                return self._create_no_path_response(request)

            # Try to find adjustment sets using backdoor criterion
            adjustment_sets = self._find_adjustment_sets(
                nx_graph,
                request.treatment,
                request.outcome,
                request.dag.nodes,
            )

            if adjustment_sets is not None:
                # Effect is identifiable
                return self._create_identifiable_response(
                    request,
                    nx_graph,
                    adjustment_sets,
                )
            else:
                # Try Y₀ identification
                y0_result = self._try_y0_identification(request)
                if y0_result:
                    return y0_result

                # Cannot identify
                return self._create_cannot_identify_response(request, nx_graph)

        except Exception as e:
            logger.error("causal_validation_failed", exc_info=True)
            raise

    def _find_adjustment_sets(
        self,
        graph: nx.DiGraph,
        treatment: str,
        outcome: str,
        all_nodes: List[str],
    ) -> Optional[List[List[str]]]:
        """
        Find valid adjustment sets using backdoor criterion.

        Args:
            graph: NetworkX graph
            treatment: Treatment node
            outcome: Outcome node
            all_nodes: All nodes in the graph

        Returns:
            List of valid adjustment sets, or None if none exist
        """
        # Get all potential adjustment variables (everything except treatment and outcome)
        potential_adjusters = [
            node for node in all_nodes if node not in [treatment, outcome]
        ]

        # Find backdoor paths
        backdoor_paths = find_backdoor_paths(graph, treatment, outcome)

        if not backdoor_paths:
            # No backdoor paths - no adjustment needed
            return [[]]

        # Try different adjustment sets
        valid_sets = []

        # Try each individual node
        for node in potential_adjusters:
            if self._blocks_all_backdoors(graph, treatment, outcome, [node]):
                valid_sets.append([node])

        # Try pairs (if no single node works)
        if not valid_sets and len(potential_adjusters) >= 2:
            for i, node1 in enumerate(potential_adjusters):
                for node2 in potential_adjusters[i + 1 :]:
                    if self._blocks_all_backdoors(graph, treatment, outcome, [node1, node2]):
                        valid_sets.append([node1, node2])

        return valid_sets if valid_sets else None

    def _blocks_all_backdoors(
        self,
        graph: nx.DiGraph,
        treatment: str,
        outcome: str,
        adjustment_set: List[str],
    ) -> bool:
        """
        Check if an adjustment set blocks all backdoor paths.

        Args:
            graph: NetworkX graph
            treatment: Treatment node
            outcome: Outcome node
            adjustment_set: Proposed adjustment set

        Returns:
            bool: True if all backdoor paths are blocked
        """
        # Get parents of treatment
        parents = list(graph.predecessors(treatment))

        for parent in parents:
            # Check if there's a path from parent to outcome that doesn't go through adjustment set
            # Create a copy of graph with adjustment set nodes removed
            temp_graph = graph.copy()
            temp_graph.remove_nodes_from(adjustment_set)

            # If there's still a path, this backdoor is not blocked
            if temp_graph.has_node(parent) and temp_graph.has_node(outcome):
                if nx.has_path(temp_graph, parent, outcome):
                    return False

        return True

    def _try_y0_identification(
        self, request: CausalValidationRequest
    ) -> Optional[CausalValidationResponse]:
        """
        Try to identify causal effect using Y₀ algorithm.

        Args:
            request: Validation request

        Returns:
            Validation response if successful, None otherwise
        """
        try:
            # Convert to Y₀ graph format
            y0_graph = edge_list_to_y0(request.dag.nodes, request.dag.edges)

            # Create treatment and outcome variables
            treatment_var = Variable(request.treatment)
            outcome_var = Variable(request.outcome)

            # Try identification
            result = identify_outcomes(
                graph=y0_graph,
                treatments={treatment_var},
                outcomes={outcome_var},
            )

            if result and result != Identification.UNKNOWN:
                # Y₀ successfully identified the effect
                logger.info("y0_identification_successful")

                # Return identifiable response (Y₀ found a valid identification)
                explanation = self.explanation_generator.generate_causal_validation_explanation(
                    status="identifiable",
                    treatment=request.treatment,
                    outcome=request.outcome,
                    adjustment_sets=[[]],
                    minimal_set=[],
                )

                return CausalValidationResponse(
                    status=ValidationStatus.IDENTIFIABLE,
                    adjustment_sets=[[]],
                    minimal_set=[],
                    backdoor_paths=[],
                    confidence=ConfidenceLevel.MEDIUM,
                    explanation=explanation,
                )

            return None

        except Exception as e:
            logger.warning(f"y0_identification_failed: {e}")
            return None

    def _create_identifiable_response(
        self,
        request: CausalValidationRequest,
        graph: nx.DiGraph,
        adjustment_sets: List[List[str]],
    ) -> CausalValidationResponse:
        """Create response for identifiable case."""
        # Find minimal set (smallest adjustment set)
        minimal_set = min(adjustment_sets, key=len) if adjustment_sets else []

        # Get backdoor paths for explanation
        backdoor_paths_list = find_backdoor_paths(
            graph, request.treatment, request.outcome
        )
        backdoor_paths = [" → ".join(path) for path in backdoor_paths_list]

        # Generate explanation
        explanation = self.explanation_generator.generate_causal_validation_explanation(
            status="identifiable",
            treatment=request.treatment,
            outcome=request.outcome,
            adjustment_sets=adjustment_sets,
            minimal_set=minimal_set,
        )

        return CausalValidationResponse(
            status=ValidationStatus.IDENTIFIABLE,
            adjustment_sets=adjustment_sets,
            minimal_set=minimal_set,
            backdoor_paths=backdoor_paths if backdoor_paths else None,
            confidence=ConfidenceLevel.HIGH,
            explanation=explanation,
        )

    def _create_no_path_response(
        self, request: CausalValidationRequest
    ) -> CausalValidationResponse:
        """Create response when there's no causal path."""
        issue = ValidationIssue(
            type=ValidationIssueType.MISSING_CONNECTION,
            description=f"No causal path exists from {request.treatment} to {request.outcome}",
            affected_nodes=[request.treatment, request.outcome],
            suggested_action=f"Add causal edges showing how {request.treatment} affects {request.outcome}",
        )

        explanation = self.explanation_generator.generate_causal_validation_explanation(
            status="cannot_identify",
            treatment=request.treatment,
            outcome=request.outcome,
            issues=[issue.model_dump()],
        )

        return CausalValidationResponse(
            status=ValidationStatus.CANNOT_IDENTIFY,
            issues=[issue],
            confidence=ConfidenceLevel.HIGH,
            explanation=explanation,
        )

    def _create_cannot_identify_response(
        self, request: CausalValidationRequest, graph: nx.DiGraph
    ) -> CausalValidationResponse:
        """Create response for cannot identify case."""
        # Try to diagnose why
        backdoor_paths_list = find_backdoor_paths(
            graph, request.treatment, request.outcome
        )

        issues = []
        if backdoor_paths_list:
            issue = ValidationIssue(
                type=ValidationIssueType.CONFOUNDING,
                description="Backdoor paths exist but no valid adjustment set found",
                affected_nodes=[request.treatment, request.outcome],
                suggested_action="Review graph structure for unmeasured confounders",
            )
            issues.append(issue)

        explanation = self.explanation_generator.generate_causal_validation_explanation(
            status="cannot_identify",
            treatment=request.treatment,
            outcome=request.outcome,
            issues=[i.model_dump() for i in issues],
        )

        return CausalValidationResponse(
            status=ValidationStatus.CANNOT_IDENTIFY,
            issues=issues if issues else None,
            confidence=ConfidenceLevel.MEDIUM,
            explanation=explanation,
        )
