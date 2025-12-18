"""
Enhanced Causal Validator service with comprehensive Y₀ integration.

Provides rich identifiability analysis including:
- Method determination (backdoor, front-door, IV, do-calculus)
- Identification formulas
- Structured assumptions
- Alternative method consideration
- Failure diagnosis
- Graceful degradation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from y0.algorithm.identify import Identification, identify_outcomes
from y0.dsl import P, Variable
from y0.graph import NxMixedGraph

from src.models.requests import CausalValidationRequest
from src.models.responses import (
    AlternativeMethod,
    AssumptionDetail,
    CausalValidationResponse,
    ValidationIssue,
)
from src.models.shared import ConfidenceLevel, ValidationIssueType, ValidationStatus
from src.services.explanation_generator import ExplanationGenerator
from src.utils.determinism import canonical_hash, make_deterministic
from src.utils.graph_parser import edge_list_to_networkx, edge_list_to_y0, find_backdoor_paths
from src.utils.validation import validate_dag_structure, validate_node_in_graph

logger = logging.getLogger(__name__)


class EnhancedCausalValidator:
    """
    Enhanced Y₀-powered causal validator.

    Provides comprehensive identifiability analysis with rich explanation metadata.
    """

    def __init__(self) -> None:
        """Initialize the enhanced validator."""
        self.explanation_generator = ExplanationGenerator()

    def validate(self, request: CausalValidationRequest) -> CausalValidationResponse:
        """
        Validate causal model for identifiability with comprehensive Y₀ analysis.

        Args:
            request: Validation request with DAG and variables

        Returns:
            CausalValidationResponse: Enhanced validation results
        """
        # Make computation deterministic
        rng = make_deterministic(request.model_dump())

        logger.info(
            "enhanced_causal_validation_started",
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

            # Convert to NetworkX for basic analysis
            nx_graph = edge_list_to_networkx(request.dag.nodes, request.dag.edges)

            # Check if there's any path from treatment to outcome
            if not nx.has_path(nx_graph, request.treatment, request.outcome):
                return self._create_no_path_response(request, nx_graph)

            # Try comprehensive Y₀ identification
            y0_result = self._try_comprehensive_y0_identification(request, nx_graph)

            if y0_result:
                return y0_result

            # Fallback: Try basic backdoor criterion
            adjustment_sets = self._find_adjustment_sets(
                nx_graph,
                request.treatment,
                request.outcome,
                request.dag.nodes,
            )

            if adjustment_sets is not None:
                return self._create_backdoor_response(request, nx_graph, adjustment_sets)

            # Cannot identify
            return self._create_cannot_identify_response(request, nx_graph)

        except Exception as e:
            logger.error("enhanced_causal_validation_failed", exc_info=True)
            # Graceful degradation
            return self._create_degraded_response(request, error=e)

    def _try_comprehensive_y0_identification(
        self,
        request: CausalValidationRequest,
        nx_graph: nx.DiGraph
    ) -> Optional[CausalValidationResponse]:
        """
        Try comprehensive Y₀ identification with rich metadata extraction.

        Args:
            request: Validation request
            nx_graph: NetworkX graph for fallback analysis

        Returns:
            Enhanced validation response if successful, None otherwise
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
                logger.info("y0_comprehensive_identification_successful")

                # Determine which method Y₀ used
                method = self._determine_y0_method(result, nx_graph, request)

                # Extract adjustment set (for backdoor cases)
                adjustment_set = self._extract_y0_adjustment_set(
                    result, nx_graph, request
                )

                # Generate identification formula
                formula = self._generate_identification_formula(
                    method, adjustment_set, request
                )

                # Extract structured assumptions
                assumptions = self._extract_assumptions(method, adjustment_set)

                # Check alternative methods
                alternatives = self._check_alternative_methods(
                    request, nx_graph
                )

                # Generate explanation
                explanation = self.explanation_generator.generate_causal_validation_explanation(
                    status="identifiable",
                    treatment=request.treatment,
                    outcome=request.outcome,
                    adjustment_sets=[adjustment_set] if adjustment_set else [[]],
                    minimal_set=adjustment_set if adjustment_set else [],
                )

                return CausalValidationResponse(
                    status=ValidationStatus.IDENTIFIABLE,
                    method=method,
                    adjustment_sets=[adjustment_set] if adjustment_set else [[]],
                    minimal_set=adjustment_set if adjustment_set else [],
                    identification_formula=formula,
                    structured_assumptions=assumptions,
                    alternative_methods=alternatives,
                    backdoor_paths=self._get_backdoor_paths(nx_graph, request),
                    confidence=ConfidenceLevel.HIGH,
                    explanation=explanation,
                )

            return None

        except Exception as e:
            logger.warning(f"y0_comprehensive_identification_failed: {e}")
            return None

    def _determine_y0_method(
        self,
        y0_result: Any,
        nx_graph: nx.DiGraph,
        request: CausalValidationRequest
    ) -> str:
        """
        Determine which identification method Y₀ used.

        Args:
            y0_result: Y₀ identification result
            nx_graph: NetworkX graph
            request: Validation request

        Returns:
            Method name: "backdoor", "front_door", "instrumental_variables", or "do_calculus"
        """
        # Check if backdoor criterion applies
        backdoor_paths = find_backdoor_paths(
            nx_graph, request.treatment, request.outcome
        )

        # If no backdoor paths or we found valid adjustment set, it's backdoor
        if not backdoor_paths or self._find_adjustment_sets(
            nx_graph, request.treatment, request.outcome, request.dag.nodes
        ):
            return "backdoor"

        # For now, assume do-calculus for complex cases
        # TODO: Add front-door and IV detection when Y₀ exposes that info
        return "do_calculus"

    def _extract_y0_adjustment_set(
        self,
        y0_result: Any,
        nx_graph: nx.DiGraph,
        request: CausalValidationRequest
    ) -> List[str]:
        """
        Extract adjustment set from Y₀ result or networkx analysis.

        Args:
            y0_result: Y₀ identification result
            nx_graph: NetworkX graph
            request: Validation request

        Returns:
            List of variables to adjust for
        """
        # Try to find backdoor adjustment set
        adjustment_sets = self._find_adjustment_sets(
            nx_graph,
            request.treatment,
            request.outcome,
            request.dag.nodes,
        )

        if adjustment_sets:
            # Return minimal set
            return min(adjustment_sets, key=len)

        return []

    def _generate_identification_formula(
        self,
        method: str,
        adjustment_set: List[str],
        request: CausalValidationRequest
    ) -> str:
        """
        Generate human-readable identification formula.

        Args:
            method: Identification method used
            adjustment_set: Adjustment set variables
            request: Validation request

        Returns:
            Human-readable formula string
        """
        treatment = request.treatment
        outcome = request.outcome

        if method == "backdoor":
            if not adjustment_set:
                return f"P({outcome}|do({treatment})) = P({outcome}|{treatment})"
            else:
                z_vars = ", ".join(adjustment_set)
                return f"P({outcome}|do({treatment})) = Σ_{{{z_vars}}} P({outcome}|{treatment}, {z_vars}) P({z_vars})"

        elif method == "front_door":
            return f"P({outcome}|do({treatment})) = Σ_M P(M|{treatment}) Σ_{treatment}' P({outcome}|M, {treatment}') P({treatment}')"

        elif method == "instrumental_variables":
            return f"P({outcome}|do({treatment})) identifiable via instrumental variable Z"

        else:  # do_calculus
            return f"P({outcome}|do({treatment})) identifiable via general do-calculus"

    def _extract_assumptions(
        self,
        method: str,
        adjustment_set: List[str]
    ) -> List[AssumptionDetail]:
        """
        Extract structured assumptions for the identification method.

        Args:
            method: Identification method used
            adjustment_set: Adjustment set variables

        Returns:
            List of structured assumption details
        """
        assumptions = []

        if method == "backdoor":
            assumptions.extend([
                AssumptionDetail(
                    type="no_unmeasured_confounding",
                    description=(
                        f"No unmeasured confounders after adjusting for {', '.join(adjustment_set)}"
                        if adjustment_set
                        else "No unmeasured confounders between treatment and outcome"
                    ),
                    critical=True
                ),
                AssumptionDetail(
                    type="positivity",
                    description="All treatment values possible at all covariate levels",
                    critical=True
                ),
                AssumptionDetail(
                    type="consistency",
                    description="Well-defined interventions and potential outcomes",
                    critical=True
                ),
                AssumptionDetail(
                    type="causal_structure",
                    description="DAG correctly represents causal relationships",
                    critical=True
                )
            ])

        elif method == "front_door":
            assumptions.extend([
                AssumptionDetail(
                    type="mediator_completeness",
                    description="All causal pathways go through identified mediators",
                    critical=True
                ),
                AssumptionDetail(
                    type="no_confounding_mediator_outcome",
                    description="No unmeasured confounding between mediator and outcome",
                    critical=True
                ),
                AssumptionDetail(
                    type="causal_structure",
                    description="DAG correctly represents causal relationships",
                    critical=True
                )
            ])

        elif method == "instrumental_variables":
            assumptions.extend([
                AssumptionDetail(
                    type="instrument_validity",
                    description="Instrument affects outcome only through treatment",
                    critical=True
                ),
                AssumptionDetail(
                    type="instrument_relevance",
                    description="Instrument sufficiently affects treatment",
                    critical=True
                )
            ])

        else:  # do_calculus
            assumptions.extend([
                AssumptionDetail(
                    type="causal_structure",
                    description="DAG correctly represents all causal relationships",
                    critical=True
                ),
                AssumptionDetail(
                    type="markov_property",
                    description="Conditional independencies implied by DAG hold",
                    critical=True
                )
            ])

        return assumptions

    def _check_alternative_methods(
        self,
        request: CausalValidationRequest,
        nx_graph: nx.DiGraph
    ) -> List[AlternativeMethod]:
        """
        Check which alternative identification methods are applicable.

        Args:
            request: Validation request
            nx_graph: NetworkX graph

        Returns:
            List of alternative method assessments
        """
        alternatives = []

        # Check backdoor
        backdoor_applicable, backdoor_reason = self._try_backdoor_method(
            nx_graph, request
        )
        alternatives.append(AlternativeMethod(
            method="backdoor",
            applicable=backdoor_applicable,
            reason=backdoor_reason
        ))

        # Check front-door (simplified heuristic)
        # TODO: Implement proper front-door criterion check
        alternatives.append(AlternativeMethod(
            method="front_door",
            applicable=False,
            reason="Front-door criterion check not yet implemented"
        ))

        # Check instrumental variables (not typically applicable without explicit IV)
        alternatives.append(AlternativeMethod(
            method="instrumental_variables",
            applicable=False,
            reason="No instrumental variable specified"
        ))

        return alternatives

    def _try_backdoor_method(
        self,
        nx_graph: nx.DiGraph,
        request: CausalValidationRequest
    ) -> Tuple[bool, str]:
        """
        Check if backdoor criterion is applicable.

        Args:
            nx_graph: NetworkX graph
            request: Validation request

        Returns:
            Tuple of (applicable, reason)
        """
        adjustment_sets = self._find_adjustment_sets(
            nx_graph,
            request.treatment,
            request.outcome,
            request.dag.nodes,
        )

        if adjustment_sets is not None:
            if not adjustment_sets[0]:  # Empty adjustment set
                return True, "No confounding - direct identification possible"
            else:
                minimal = min(adjustment_sets, key=len)
                return True, f"Valid adjustment set exists: {{{', '.join(minimal)}}}"
        else:
            backdoor_paths = find_backdoor_paths(
                nx_graph, request.treatment, request.outcome
            )
            if backdoor_paths:
                return False, "Backdoor paths exist but no valid adjustment set with measured variables"
            else:
                return True, "No backdoor paths to block"

    def _find_adjustment_sets(
        self,
        graph: nx.DiGraph,
        treatment: str,
        outcome: str,
        all_nodes: List[str],
    ) -> Optional[List[List[str]]]:
        """
        Find valid adjustment sets using backdoor criterion.

        (Same implementation as original - kept for backward compatibility)
        """
        potential_adjusters = [
            node for node in all_nodes if node not in [treatment, outcome]
        ]

        backdoor_paths = find_backdoor_paths(graph, treatment, outcome)

        if not backdoor_paths:
            return [[]]

        valid_sets = []

        # Try each individual node
        for node in potential_adjusters:
            if self._blocks_all_backdoors(graph, treatment, outcome, [node]):
                valid_sets.append([node])

        # Try pairs if no single node works
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

        (Same implementation as original - kept for backward compatibility)
        """
        parents = list(graph.predecessors(treatment))

        for parent in parents:
            temp_graph = graph.copy()
            temp_graph.remove_nodes_from(adjustment_set)

            if temp_graph.has_node(parent) and temp_graph.has_node(outcome):
                if nx.has_path(temp_graph, parent, outcome):
                    return False

        return True

    def _get_backdoor_paths(
        self,
        nx_graph: nx.DiGraph,
        request: CausalValidationRequest
    ) -> Optional[List[str]]:
        """Get formatted backdoor paths."""
        paths_list = find_backdoor_paths(
            nx_graph, request.treatment, request.outcome
        )
        if paths_list:
            return [" → ".join(path) for path in paths_list]
        return None

    def _create_backdoor_response(
        self,
        request: CausalValidationRequest,
        graph: nx.DiGraph,
        adjustment_sets: List[List[str]],
    ) -> CausalValidationResponse:
        """Create enhanced response for backdoor identification."""
        minimal_set = min(adjustment_sets, key=len) if adjustment_sets else []

        # Generate formula
        formula = self._generate_identification_formula(
            "backdoor", minimal_set, request
        )

        # Extract assumptions
        assumptions = self._extract_assumptions("backdoor", minimal_set)

        # Check alternatives
        alternatives = self._check_alternative_methods(request, graph)

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
            method="backdoor",
            adjustment_sets=adjustment_sets,
            minimal_set=minimal_set,
            identification_formula=formula,
            structured_assumptions=assumptions,
            alternative_methods=alternatives,
            backdoor_paths=self._get_backdoor_paths(graph, request),
            confidence=ConfidenceLevel.HIGH,
            explanation=explanation,
        )

    def _create_no_path_response(
        self,
        request: CausalValidationRequest,
        nx_graph: nx.DiGraph
    ) -> CausalValidationResponse:
        """Create enhanced response when there's no causal path."""
        issue = ValidationIssue(
            type=ValidationIssueType.MISSING_CONNECTION,
            description=f"No causal path exists from {request.treatment} to {request.outcome}",
            affected_nodes=[request.treatment, request.outcome],
            suggested_action=f"Add causal edges showing how {request.treatment} affects {request.outcome}",
        )

        suggestions = [
            f"Add edges from {request.treatment} to {request.outcome} or through mediators",
            f"Verify that {request.treatment} actually affects {request.outcome} in your domain",
            "Review the causal structure - is the treatment-outcome relationship missing?"
        ]

        explanation = self.explanation_generator.generate_causal_validation_explanation(
            status="cannot_identify",
            treatment=request.treatment,
            outcome=request.outcome,
            issues=[issue.model_dump()],
        )

        return CausalValidationResponse(
            status=ValidationStatus.CANNOT_IDENTIFY,
            reason="no_causal_path",
            suggestions=suggestions,
            attempted_methods=["structural_analysis"],
            issues=[issue],
            confidence=ConfidenceLevel.HIGH,
            explanation=explanation,
        )

    def _create_cannot_identify_response(
        self,
        request: CausalValidationRequest,
        graph: nx.DiGraph
    ) -> CausalValidationResponse:
        """Create enhanced response for cannot identify case with diagnosis."""
        # Diagnose why identification failed
        backdoor_paths_list = find_backdoor_paths(
            graph, request.treatment, request.outcome
        )

        if backdoor_paths_list:
            reason = "unmeasured_confounding"
            suggestions = [
                "Add measured confounders to the model to block backdoor paths",
                "Consider using instrumental variables if available",
                "Explore front-door criterion if mediators fully capture the pathway",
                "Collect data on potential confounders"
            ]
            description = "Backdoor paths exist but no valid adjustment set found with measured variables"
        else:
            reason = "identification_failed"
            suggestions = [
                "Simplify causal structure if possible",
                "Verify DAG structure is correct",
                "Consult with causal inference expert for complex cases"
            ]
            description = "Effect cannot be identified with standard methods"

        issue = ValidationIssue(
            type=ValidationIssueType.CONFOUNDING,
            description=description,
            affected_nodes=[request.treatment, request.outcome],
            suggested_action="Review graph structure for unmeasured confounders",
        )

        explanation = self.explanation_generator.generate_causal_validation_explanation(
            status="cannot_identify",
            treatment=request.treatment,
            outcome=request.outcome,
            issues=[issue.model_dump()],
        )

        return CausalValidationResponse(
            status=ValidationStatus.CANNOT_IDENTIFY,
            reason=reason,
            suggestions=suggestions,
            attempted_methods=["backdoor", "y0_identification"],
            issues=[issue],
            confidence=ConfidenceLevel.MEDIUM,
            explanation=explanation,
        )

    def _create_degraded_response(
        self,
        request: CausalValidationRequest,
        error: Exception
    ) -> CausalValidationResponse:
        """
        Create graceful degraded response when Y₀ errors.

        Args:
            request: Validation request
            error: Exception that occurred

        Returns:
            Degraded response with fallback assessment
        """
        logger.warning(
            "creating_degraded_response",
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
                "treatment": request.treatment,
                "outcome": request.outcome,
            }
        )

        # Basic fallback: check for direct path
        try:
            nx_graph = edge_list_to_networkx(request.dag.nodes, request.dag.edges)
            direct_path = nx.has_path(nx_graph, request.treatment, request.outcome)

            # Find potential confounders (nodes with paths to both treatment and outcome)
            potential_confounders = []
            for node in request.dag.nodes:
                if node not in [request.treatment, request.outcome]:
                    if (nx.has_path(nx_graph, node, request.treatment) and
                        nx.has_path(nx_graph, node, request.outcome)):
                        potential_confounders.append(node)

            fallback_assessment = {
                "direct_path_exists": direct_path,
                "potential_confounders": potential_confounders,
                "recommendation": (
                    "Manual review recommended - advanced analysis unavailable"
                    if direct_path
                    else "No causal path found - treatment may not affect outcome"
                )
            }
        except Exception as fallback_error:
            logger.error(f"fallback_assessment_failed: {fallback_error}")
            fallback_assessment = {
                "error": "Fallback assessment also failed",
                "recommendation": "Contact support with error details"
            }

        explanation = self.explanation_generator.generate_causal_validation_explanation(
            status="degraded",
            treatment=request.treatment,
            outcome=request.outcome,
        )

        return CausalValidationResponse(
            status=ValidationStatus.DEGRADED,
            reason="y0_analysis_failed",
            fallback_assessment=fallback_assessment,
            suggestions=[
                "Try simplifying the DAG structure",
                "Verify all node names are valid identifiers",
                "Contact support if error persists"
            ],
            attempted_methods=["y0_identification", "fallback_structural_analysis"],
            confidence=ConfidenceLevel.LOW,
            explanation=explanation,
        )
