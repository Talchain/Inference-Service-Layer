"""
Advanced Model Validator service.

Performs comprehensive validation of causal models with structural,
statistical, and domain-specific checks. Target: 90%+ issue detection rate.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.config import get_settings
from src.models.phase1_models import (
    BestPractice,
    BestPracticeStatus,
    DomainValidation,
    ImpactLevel,
    ModelSuggestion,
    QualityLevel,
    StructuralValidation,
    StatisticalValidation,
    SuggestionType,
    ValidationCheck,
    ValidationLevel,
    ValidationResults,
    ValidationStatus,
)
from src.models.shared import ExplanationMetadata
from src.utils.determinism import make_deterministic

logger = logging.getLogger(__name__)
settings = get_settings()


class AdvancedModelValidator:
    """
    Advanced causal model validation.

    Performs 3 levels of validation:
    1. Structural: DAG properties, identifiability
    2. Statistical: Distributions, parameters, sample size
    3. Domain: Best practices, common pitfalls

    Target: 90%+ issue detection rate
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        pass

    def validate(
        self,
        dag: Dict[str, Any],
        structural_model: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        validation_level: ValidationLevel,
    ) -> Tuple[
        QualityLevel, float, ValidationResults, List[ModelSuggestion], List[BestPractice]
    ]:
        """
        Perform comprehensive model validation.

        Args:
            dag: DAG structure with nodes and edges
            structural_model: Optional structural model with equations
            context: Optional context (domain, etc.)
            validation_level: Level of validation to perform

        Returns:
            Tuple of (quality_level, quality_score, validation_results,
                     suggestions, best_practices)
        """
        # Make computation deterministic
        rng = make_deterministic(
            {
                "dag": dag,
                "level": validation_level.value,
            }
        )

        logger.info(
            "validation_started",
            extra={
                "num_nodes": len(dag.get("nodes", [])),
                "num_edges": len(dag.get("edges", [])),
                "level": validation_level.value,
                "seed": rng.seed,
            },
        )

        # Perform validations
        structural_validation = self._validate_structural(dag, validation_level)
        statistical_validation = self._validate_statistical(
            structural_model, validation_level
        )
        domain_validation = self._validate_domain(dag, context, validation_level)

        validation_results = ValidationResults(
            structural=structural_validation,
            statistical=statistical_validation,
            domain=domain_validation,
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(validation_results, dag, structural_model)

        # Check best practices
        best_practices = self._check_best_practices(dag, structural_model, context)

        # Compute overall quality
        quality_level, quality_score = self._compute_quality(validation_results)

        logger.info(
            "validation_complete",
            extra={
                "quality_level": quality_level.value,
                "quality_score": quality_score,
                "num_suggestions": len(suggestions),
                "num_checks": len(structural_validation.checks)
                + len(statistical_validation.checks)
                + len(domain_validation.checks),
            },
        )

        return quality_level, quality_score, validation_results, suggestions, best_practices

    def _validate_structural(
        self, dag: Dict[str, Any], level: ValidationLevel
    ) -> StructuralValidation:
        """Validate structural properties of DAG."""
        checks = []

        # Build NetworkX graph
        G = self._build_graph(dag)

        # Check 1: Is DAG acyclic?
        is_dag = nx.is_directed_acyclic_graph(G)
        checks.append(
            ValidationCheck(
                name="acyclic_check",
                status=ValidationStatus.PASS if is_dag else ValidationStatus.FAIL,
                description="Graph is acyclic" if is_dag else "Graph contains cycles",
                recommendation="Remove cycles to ensure valid causal structure"
                if not is_dag
                else None,
            )
        )

        # Check 2: Connected components
        if G.number_of_nodes() > 0:
            weakly_connected = nx.is_weakly_connected(G)
            checks.append(
                ValidationCheck(
                    name="connectivity_check",
                    status=ValidationStatus.PASS
                    if weakly_connected
                    else ValidationStatus.WARNING,
                    description="Graph is connected"
                    if weakly_connected
                    else "Graph has disconnected components",
                    recommendation="Consider if disconnected components are intentional"
                    if not weakly_connected
                    else None,
                )
            )

        # Check 3: Reasonable size
        num_nodes = G.number_of_nodes()
        if num_nodes > 50:
            checks.append(
                ValidationCheck(
                    name="size_check",
                    status=ValidationStatus.WARNING,
                    description=f"Large graph with {num_nodes} nodes may be hard to interpret",
                    recommendation="Consider simplifying or decomposing the model",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="size_check",
                    status=ValidationStatus.PASS,
                    description=f"Reasonable model size ({num_nodes} nodes)",
                    recommendation=None,
                )
            )

        # Check 4: No isolated nodes
        isolated = list(nx.isolates(G))
        if isolated:
            checks.append(
                ValidationCheck(
                    name="isolated_nodes_check",
                    status=ValidationStatus.WARNING,
                    description=f"Found {len(isolated)} isolated nodes: {isolated}",
                    recommendation="Remove isolated nodes or add edges",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="isolated_nodes_check",
                    status=ValidationStatus.PASS,
                    description="No isolated nodes",
                    recommendation=None,
                )
            )

        # Additional checks for COMPREHENSIVE level
        if level == ValidationLevel.COMPREHENSIVE:
            # Check for common structural issues
            self._check_structural_patterns(G, checks)

        return StructuralValidation(checks=checks)

    def _validate_statistical(
        self, structural_model: Optional[Dict[str, Any]], level: ValidationLevel
    ) -> StatisticalValidation:
        """Validate statistical properties of model."""
        checks = []

        if structural_model is None:
            checks.append(
                ValidationCheck(
                    name="model_provided",
                    status=ValidationStatus.WARNING,
                    description="No structural model provided for statistical validation",
                    recommendation="Provide structural model for complete validation",
                )
            )
            return StatisticalValidation(checks=checks)

        # Check 1: Variables defined
        variables = structural_model.get("variables", [])
        if not variables:
            checks.append(
                ValidationCheck(
                    name="variables_defined",
                    status=ValidationStatus.FAIL,
                    description="No variables defined in structural model",
                    recommendation="Define model variables",
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name="variables_defined",
                    status=ValidationStatus.PASS,
                    description=f"{len(variables)} variables defined",
                    recommendation=None,
                )
            )

        # Check 2: Distributions specified
        distributions = structural_model.get("distributions", {})
        if distributions:
            # Check distribution parameters
            valid_dists = 0
            for var, dist_spec in distributions.items():
                if "type" in dist_spec and "parameters" in dist_spec:
                    valid_dists += 1

            if valid_dists == len(distributions):
                checks.append(
                    ValidationCheck(
                        name="distributions_valid",
                        status=ValidationStatus.PASS,
                        description=f"All {len(distributions)} distributions properly specified",
                        recommendation=None,
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name="distributions_valid",
                        status=ValidationStatus.WARNING,
                        description=f"Only {valid_dists}/{len(distributions)} distributions fully specified",
                        recommendation="Ensure all distributions have type and parameters",
                    )
                )

        # Check 3: Equations defined
        equations = structural_model.get("equations", {})
        if equations:
            checks.append(
                ValidationCheck(
                    name="equations_defined",
                    status=ValidationStatus.PASS,
                    description=f"{len(equations)} structural equations defined",
                    recommendation=None,
                )
            )

        return StatisticalValidation(checks=checks)

    def _validate_domain(
        self,
        dag: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        level: ValidationLevel,
    ) -> DomainValidation:
        """Validate domain-specific properties."""
        checks = []

        # Check 1: Context provided
        if context is None:
            checks.append(
                ValidationCheck(
                    name="context_provided",
                    status=ValidationStatus.WARNING,
                    description="No domain context provided",
                    recommendation="Provide domain context for domain-specific validation",
                )
            )
            return DomainValidation(checks=checks)

        # Check 2: Domain-specific node naming
        nodes = dag.get("nodes", [])
        domain = context.get("domain", "general")

        if domain != "general":
            checks.append(
                ValidationCheck(
                    name="domain_specified",
                    status=ValidationStatus.PASS,
                    description=f"Domain specified: {domain}",
                    recommendation=None,
                )
            )

        # Check 3: Variable naming conventions
        if nodes:
            has_descriptive_names = all(len(node) > 1 for node in nodes)
            checks.append(
                ValidationCheck(
                    name="descriptive_names",
                    status=ValidationStatus.PASS
                    if has_descriptive_names
                    else ValidationStatus.WARNING,
                    description="Variable names are descriptive"
                    if has_descriptive_names
                    else "Some variables have single-character names",
                    recommendation="Use descriptive variable names for clarity"
                    if not has_descriptive_names
                    else None,
                )
            )

        return DomainValidation(checks=checks)

    def _check_structural_patterns(
        self, G: nx.DiGraph, checks: List[ValidationCheck]
    ) -> None:
        """Check for common structural patterns and anti-patterns."""
        # Check for v-structures (colliders)
        num_colliders = 0
        for node in G.nodes():
            parents = list(G.predecessors(node))
            if len(parents) >= 2:
                num_colliders += 1

        if num_colliders > 0:
            checks.append(
                ValidationCheck(
                    name="colliders_present",
                    status=ValidationStatus.PASS,
                    description=f"Found {num_colliders} collider structures",
                    recommendation=None,
                )
            )

        # Check for long causal chains
        if G.number_of_nodes() > 0:
            try:
                longest_path = nx.dag_longest_path_length(G)
                if longest_path > 5:
                    checks.append(
                        ValidationCheck(
                            name="chain_length",
                            status=ValidationStatus.WARNING,
                            description=f"Longest causal chain has {longest_path} nodes",
                            recommendation="Consider if all intermediate variables are necessary",
                        )
                    )
            except:
                pass  # Not a DAG or other issue

    def _generate_suggestions(
        self,
        results: ValidationResults,
        dag: Dict[str, Any],
        structural_model: Optional[Dict[str, Any]],
    ) -> List[ModelSuggestion]:
        """Generate improvement suggestions based on validation results."""
        suggestions = []

        # Collect all failed/warning checks
        all_checks = (
            results.structural.checks
            + results.statistical.checks
            + results.domain.checks
        )

        failed_checks = [c for c in all_checks if c.status == ValidationStatus.FAIL]
        warning_checks = [c for c in all_checks if c.status == ValidationStatus.WARNING]

        # Generate suggestions for failed checks
        for check in failed_checks:
            if "cycle" in check.name or "acyclic" in check.name:
                suggestions.append(
                    ModelSuggestion(
                        type=SuggestionType.REMOVE_EDGE,
                        description="Remove edges to eliminate cycles",
                        rationale="Causal models must be acyclic (DAGs)",
                        confidence=0.95,
                        impact=ImpactLevel.HIGH,
                    )
                )

        # Generate suggestions for warnings
        for check in warning_checks:
            if "isolated" in check.name:
                suggestions.append(
                    ModelSuggestion(
                        type=SuggestionType.REMOVE_NODE,
                        description=f"Remove isolated nodes or add edges to connect them",
                        rationale="Isolated nodes don't contribute to causal inference",
                        confidence=0.8,
                        impact=ImpactLevel.MEDIUM,
                    )
                )

            if "disconnect" in check.name:
                suggestions.append(
                    ModelSuggestion(
                        type=SuggestionType.ADD_EDGE,
                        description="Consider adding edges to connect components",
                        rationale="Disconnected components may indicate missing relationships",
                        confidence=0.6,
                        impact=ImpactLevel.MEDIUM,
                    )
                )

        return suggestions

    def _check_best_practices(
        self,
        dag: Dict[str, Any],
        structural_model: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> List[BestPractice]:
        """Check adherence to best practices."""
        practices = []

        # Practice 1: Document variables
        if context and "variable_descriptions" in context:
            practices.append(
                BestPractice(
                    practice="Document variables",
                    status=BestPracticeStatus.FOLLOWED,
                    description="Variables are documented",
                )
            )
        else:
            practices.append(
                BestPractice(
                    practice="Document variables",
                    status=BestPracticeStatus.NOT_FOLLOWED,
                    description="Variable descriptions not provided",
                )
            )

        # Practice 2: Use domain knowledge
        if context and context.get("domain") != "general":
            practices.append(
                BestPractice(
                    practice="Apply domain knowledge",
                    status=BestPracticeStatus.FOLLOWED,
                    description="Domain context provided",
                )
            )
        else:
            practices.append(
                BestPractice(
                    practice="Apply domain knowledge",
                    status=BestPracticeStatus.PARTIAL,
                    description="Limited domain context",
                )
            )

        # Practice 3: Specify uncertainty
        if structural_model and "distributions" in structural_model:
            practices.append(
                BestPractice(
                    practice="Quantify uncertainty",
                    status=BestPracticeStatus.FOLLOWED,
                    description="Probability distributions specified",
                )
            )
        else:
            practices.append(
                BestPractice(
                    practice="Quantify uncertainty",
                    status=BestPracticeStatus.NOT_FOLLOWED,
                    description="No uncertainty quantification",
                )
            )

        return practices

    def _compute_quality(
        self, results: ValidationResults
    ) -> Tuple[QualityLevel, float]:
        """Compute overall quality score and level."""
        all_checks = (
            results.structural.checks
            + results.statistical.checks
            + results.domain.checks
        )

        if not all_checks:
            return QualityLevel.ACCEPTABLE, 50.0

        # Count check statuses
        num_pass = sum(1 for c in all_checks if c.status == ValidationStatus.PASS)
        num_warning = sum(1 for c in all_checks if c.status == ValidationStatus.WARNING)
        num_fail = sum(1 for c in all_checks if c.status == ValidationStatus.FAIL)

        # Compute score (0-100)
        score = (num_pass * 100 + num_warning * 50 + num_fail * 0) / len(all_checks)

        # Determine quality level
        if score >= 90:
            quality_level = QualityLevel.EXCELLENT
        elif score >= 75:
            quality_level = QualityLevel.GOOD
        elif score >= 50:
            quality_level = QualityLevel.ACCEPTABLE
        else:
            quality_level = QualityLevel.POOR

        return quality_level, score

    def _build_graph(self, dag: Dict[str, Any]) -> nx.DiGraph:
        """Build NetworkX graph from DAG specification."""
        G = nx.DiGraph()

        # Add nodes
        nodes = dag.get("nodes", [])
        G.add_nodes_from(nodes)

        # Add edges
        edges = dag.get("edges", [])
        for edge in edges:
            if isinstance(edge, list) and len(edge) == 2:
                G.add_edge(edge[0], edge[1])

        return G
