"""
Critique definitions for ISL V2 response format.

Provides structured critique types with explicit source classification
for validation, analysis, and engine errors.
"""

import uuid
from dataclasses import dataclass
from typing import List, Literal, Optional

from src.models.response_v2 import CritiqueV2


@dataclass
class CritiqueDefinition:
    """Definition for a critique type."""

    code: str
    severity: Literal["info", "warning", "error", "blocker"]
    source: Literal["validation", "analysis", "engine"]
    message_template: str
    default_suggestion: Optional[str] = None

    def build(
        self,
        affected_option_ids: Optional[List[str]] = None,
        affected_node_ids: Optional[List[str]] = None,
        suggestion: Optional[str] = None,
        **template_vars,
    ) -> CritiqueV2:
        """Build a CritiqueV2 instance from this definition."""
        return CritiqueV2(
            id=f"critique_{uuid.uuid4().hex[:8]}",
            code=self.code,
            severity=self.severity,
            source=self.source,
            message=self.message_template.format(**template_vars),
            affected_option_ids=affected_option_ids,
            affected_node_ids=affected_node_ids,
            suggestion=suggestion or self.default_suggestion,
        )


# =============================================================================
# Validation Critiques (request structure issues)
# =============================================================================

MISSING_GOAL_NODE = CritiqueDefinition(
    code="MISSING_GOAL_NODE",
    severity="blocker",
    source="validation",
    message_template="Goal node not found in graph",
    default_suggestion="Ensure goal_node_id references a node in the graph",
)

NO_OPTIONS = CritiqueDefinition(
    code="NO_OPTIONS",
    severity="blocker",
    source="validation",
    message_template="No options provided for comparison",
    default_suggestion="Provide at least one option with interventions",
)

EMPTY_INTERVENTIONS = CritiqueDefinition(
    code="EMPTY_INTERVENTIONS",
    severity="blocker",
    source="validation",
    message_template="Option '{label}' has no interventions specified",
    default_suggestion="Add intervention mappings specifying which variables to change",
)

INVALID_INTERVENTION_TARGET = CritiqueDefinition(
    code="INVALID_INTERVENTION_TARGET",
    severity="blocker",
    source="validation",
    message_template="Option '{label}' targets non-existent node",
    default_suggestion="Check that intervention targets reference valid node IDs",
)

NO_EFFECTIVE_PATH_TO_GOAL = CritiqueDefinition(
    code="NO_EFFECTIVE_PATH_TO_GOAL",
    severity="blocker",
    source="validation",
    message_template=(
        "Option '{label}' has no interventions that can effectively affect the goal"
    ),
    default_suggestion=(
        "Add causal edges connecting intervention targets to the goal, "
        "or target different factors"
    ),
)

IDENTICAL_OPTIONS = CritiqueDefinition(
    code="IDENTICAL_OPTIONS",
    severity="blocker",
    source="validation",
    message_template="Options '{label_a}' and '{label_b}' have identical interventions",
    default_suggestion="Ensure each option specifies different intervention values",
)

GRAPH_CYCLE_DETECTED = CritiqueDefinition(
    code="GRAPH_CYCLE_DETECTED",
    severity="blocker",
    source="validation",
    message_template="Graph contains a cycle",
    default_suggestion="Remove cyclic dependencies from the causal graph",
)

# P2-ISL-4: Additional graph structure critiques
GRAPH_EMPTY = CritiqueDefinition(
    code="GRAPH_EMPTY",
    severity="blocker",
    source="validation",
    message_template="Graph contains no nodes",
    default_suggestion="Add nodes to the causal graph",
)

GRAPH_DISCONNECTED = CritiqueDefinition(
    code="GRAPH_DISCONNECTED",
    severity="warning",
    source="validation",
    message_template="Graph has {count} disconnected components",
    default_suggestion="Verify graph connectivity; disconnected nodes won't affect goal",
)

# P2-ISL-4: Node validation
INVALID_NODE_ID = CritiqueDefinition(
    code="INVALID_NODE_ID",
    severity="blocker",
    source="validation",
    message_template='Node ID "{id}" contains invalid characters',
    default_suggestion="Node IDs must contain only lowercase letters, numbers, underscores, colons, and hyphens",
)

DUPLICATE_NODE_ID = CritiqueDefinition(
    code="DUPLICATE_NODE_ID",
    severity="blocker",
    source="validation",
    message_template='Duplicate node ID: "{id}"',
    default_suggestion="Ensure all node IDs are unique",
)

# P2-ISL-4: Edge validation
EDGE_STRENGTH_OUT_OF_RANGE = CritiqueDefinition(
    code="EDGE_STRENGTH_OUT_OF_RANGE",
    severity="warning",
    source="validation",
    message_template="Edge {from_node}→{to_node} strength {value} outside [-3, 3] range",
    default_suggestion="Edge strengths should typically be between -3 and 3",
)

EDGE_STD_INVALID = CritiqueDefinition(
    code="EDGE_STD_INVALID",
    severity="blocker",
    source="validation",
    message_template="Edge {from_node}→{to_node} std must be > 0, got {value}",
    default_suggestion="Edge uncertainty (std) must be a positive number",
)

EDGE_ENDPOINT_MISSING = CritiqueDefinition(
    code="EDGE_ENDPOINT_MISSING",
    severity="blocker",
    source="validation",
    message_template="Edge references missing node: {endpoint}",
    default_suggestion="Ensure both edge endpoints exist as nodes in the graph",
)

NEGLIGIBLE_EDGE_STRENGTH = CritiqueDefinition(
    code="NEGLIGIBLE_EDGE_STRENGTH",
    severity="info",
    source="validation",
    message_template="Edge {from_node}→{to_node} has negligible strength ({value})",
    default_suggestion="This edge may have no practical effect on outcomes",
)

# P2-ISL-4: Option validation
INSUFFICIENT_OPTIONS = CritiqueDefinition(
    code="INSUFFICIENT_OPTIONS",
    severity="blocker",
    source="validation",
    message_template="At least 2 options required for comparison, got {count}",
    default_suggestion="Add at least one more option to enable comparison",
)

OPTION_NO_INTERVENTIONS = CritiqueDefinition(
    code="OPTION_NO_INTERVENTIONS",
    severity="info",
    source="validation",
    message_template='Option "{id}" has no interventions (treated as status quo)',
    default_suggestion="This option represents the baseline/status quo scenario",
)

DUPLICATE_OPTION_ID = CritiqueDefinition(
    code="DUPLICATE_OPTION_ID",
    severity="blocker",
    source="validation",
    message_template='Duplicate option ID: "{id}"',
    default_suggestion="Ensure all option IDs are unique",
)

INTERVENTION_VALUE_INVALID = CritiqueDefinition(
    code="INTERVENTION_VALUE_INVALID",
    severity="blocker",
    source="validation",
    message_template="Intervention value must be finite number, got: {value}",
    default_suggestion="Ensure intervention values are valid finite numbers",
)

# P2-ISL-4: Inference errors
MONTE_CARLO_FAILED = CritiqueDefinition(
    code="MONTE_CARLO_FAILED",
    severity="blocker",
    source="analysis",
    message_template="Monte Carlo simulation failed: {reason}",
    default_suggestion="Check graph structure and edge values for numerical issues",
)

BASELINE_NEAR_ZERO = CritiqueDefinition(
    code="BASELINE_NEAR_ZERO",
    severity="warning",
    source="analysis",
    message_template="Baseline outcome near zero ({value}), sensitivity calculations may be unstable",
    default_suggestion="Results are epsilon-guarded but should be interpreted with caution",
)

INFERENCE_TIMEOUT = CritiqueDefinition(
    code="INFERENCE_TIMEOUT",
    severity="blocker",
    source="engine",
    message_template="Inference timed out after {seconds}s",
    default_suggestion="Try reducing n_samples or simplifying the graph",
)

SEED_INVALID = CritiqueDefinition(
    code="SEED_INVALID",
    severity="warning",
    source="validation",
    message_template='Invalid seed "{value}", using default "42"',
    default_suggestion="Provide a valid integer seed for reproducibility",
)


# =============================================================================
# Analysis Critiques (issues discovered during computation)
# =============================================================================

DEGENERATE_OUTCOMES = CritiqueDefinition(
    code="DEGENERATE_OUTCOMES",
    severity="warning",
    source="analysis",
    message_template="All options produce nearly identical outcomes",
    default_suggestion=(
        "Check that options specify different intervention values and that "
        "intervention targets are connected to the goal with non-zero effect"
    ),
)

NUMERICAL_INSTABILITY = CritiqueDefinition(
    code="NUMERICAL_INSTABILITY",
    severity="warning",
    source="analysis",
    message_template=(
        "Numerical instability detected in {invalid_count} of {total_count} samples"
    ),
    default_suggestion="Check for extreme values or edge weights in the graph",
)

LOW_EFFECTIVE_SAMPLES = CritiqueDefinition(
    code="LOW_EFFECTIVE_SAMPLES",
    severity="warning",
    source="analysis",
    message_template=(
        "Only {valid_count} of {total_count} samples were numerically valid"
    ),
    default_suggestion="Results may be unreliable. Consider simplifying the graph",
)

IDENTIFIABILITY_ISSUE = CritiqueDefinition(
    code="IDENTIFIABILITY_ISSUE",
    severity="warning",
    source="analysis",
    message_template="Causal effect may not be fully identifiable",
    default_suggestion="Results should be interpreted with caution",
)


# =============================================================================
# Engine Critiques (internal errors)
# =============================================================================

INTERNAL_ERROR = CritiqueDefinition(
    code="INTERNAL_ERROR",
    severity="blocker",
    source="engine",
    message_template="An internal error occurred during analysis",
    default_suggestion="Please retry. If the problem persists, contact support",
)


# =============================================================================
# Critique Registry
# =============================================================================

CRITIQUES = {
    # Validation - Graph structure
    "GRAPH_EMPTY": GRAPH_EMPTY,
    "GRAPH_DISCONNECTED": GRAPH_DISCONNECTED,
    "GRAPH_CYCLE_DETECTED": GRAPH_CYCLE_DETECTED,
    # Validation - Nodes
    "MISSING_GOAL_NODE": MISSING_GOAL_NODE,
    "INVALID_NODE_ID": INVALID_NODE_ID,
    "DUPLICATE_NODE_ID": DUPLICATE_NODE_ID,
    # Validation - Edges
    "EDGE_STRENGTH_OUT_OF_RANGE": EDGE_STRENGTH_OUT_OF_RANGE,
    "EDGE_STD_INVALID": EDGE_STD_INVALID,
    "EDGE_ENDPOINT_MISSING": EDGE_ENDPOINT_MISSING,
    "NEGLIGIBLE_EDGE_STRENGTH": NEGLIGIBLE_EDGE_STRENGTH,
    # Validation - Options
    "NO_OPTIONS": NO_OPTIONS,
    "INSUFFICIENT_OPTIONS": INSUFFICIENT_OPTIONS,
    "EMPTY_INTERVENTIONS": EMPTY_INTERVENTIONS,
    "OPTION_NO_INTERVENTIONS": OPTION_NO_INTERVENTIONS,
    "DUPLICATE_OPTION_ID": DUPLICATE_OPTION_ID,
    "INVALID_INTERVENTION_TARGET": INVALID_INTERVENTION_TARGET,
    "INTERVENTION_VALUE_INVALID": INTERVENTION_VALUE_INVALID,
    "NO_EFFECTIVE_PATH_TO_GOAL": NO_EFFECTIVE_PATH_TO_GOAL,
    "IDENTICAL_OPTIONS": IDENTICAL_OPTIONS,
    # Validation - Seed
    "SEED_INVALID": SEED_INVALID,
    # Analysis
    "DEGENERATE_OUTCOMES": DEGENERATE_OUTCOMES,
    "NUMERICAL_INSTABILITY": NUMERICAL_INSTABILITY,
    "LOW_EFFECTIVE_SAMPLES": LOW_EFFECTIVE_SAMPLES,
    "IDENTIFIABILITY_ISSUE": IDENTIFIABILITY_ISSUE,
    "MONTE_CARLO_FAILED": MONTE_CARLO_FAILED,
    "BASELINE_NEAR_ZERO": BASELINE_NEAR_ZERO,
    # Engine
    "INTERNAL_ERROR": INTERNAL_ERROR,
    "INFERENCE_TIMEOUT": INFERENCE_TIMEOUT,
}


def get_critique(code: str) -> CritiqueDefinition:
    """Get critique definition by code."""
    if code not in CRITIQUES:
        raise ValueError(f"Unknown critique code: {code}")
    return CRITIQUES[code]
