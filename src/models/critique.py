"""
Critique definitions for ISL V2 response format.

Provides structured critique types with explicit source classification
for validation, analysis, and engine errors.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from src.models.response_v2 import CritiqueV2


def deterministic_critique_id(
    code: str,
    message: str,
    affected_option_ids: Optional[List[str]] = None,
    affected_node_ids: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> str:
    """Derive a stable critique id from critique content (+ seed when known).

    Same-seed identical requests must produce byte-identical responses; a
    ``uuid.uuid4()`` id broke that on every response containing a critique
    (science-validation report §3 volatile-field catalogue, fix recommended
    in §5.7a). The formatted message participates in the hash because some
    critiques (e.g. EDGE_STRENGTH_OUT_OF_RANGE) carry their distinguishing
    detail only in template vars, not in the affected-id lists — hashing
    (seed, code, affected ids) alone would collide within one response.
    """
    material = "\x1f".join(
        [
            "" if seed is None else str(seed),
            code,
            message,
            "\x1e".join(affected_option_ids or []),
            "\x1e".join(affected_node_ids or []),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"critique_{digest}"


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
        seed: Optional[int] = None,
        **template_vars: Any,
    ) -> CritiqueV2:
        """Build a CritiqueV2 instance from this definition.

        ``seed`` (the analysis seed, when the call site has one) feeds the
        deterministic critique id; it is not part of the critique payload.
        """
        message = self.message_template.format(**template_vars)
        return CritiqueV2(
            id=deterministic_critique_id(
                code=self.code,
                message=message,
                affected_option_ids=affected_option_ids,
                affected_node_ids=affected_node_ids,
                seed=seed,
            ),
            code=self.code,
            severity=self.severity,
            source=self.source,
            message=message,
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
    message_template=("Option '{label}' has no interventions that can effectively affect the goal"),
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
    message_template=("Numerical instability detected in {invalid_count} of {total_count} samples"),
    default_suggestion="Check for extreme values or edge weights in the graph",
)

LOW_EFFECTIVE_SAMPLES = CritiqueDefinition(
    code="LOW_EFFECTIVE_SAMPLES",
    severity="warning",
    source="analysis",
    message_template=("Only {valid_count} of {total_count} samples were numerically valid"),
    default_suggestion="Results may be unreliable. Consider simplifying the graph",
)

IDENTIFIABILITY_ISSUE = CritiqueDefinition(
    code="IDENTIFIABILITY_ISSUE",
    severity="warning",
    source="analysis",
    message_template="Causal effect may not be fully identifiable",
    default_suggestion="Results should be interpreted with caution",
)

DEGENERATE_OPTION_ZERO_VARIANCE = CritiqueDefinition(
    code="DEGENERATE_OPTION_ZERO_VARIANCE",
    severity="warning",
    source="analysis",
    message_template=(
        "Option '{option_label}' has zero variance — "
        "intervention may have no causal path to goal"
    ),
    default_suggestion=(
        "Check that intervention targets are connected to the goal " "with non-zero edge strengths"
    ),
)

STRUCTURAL_INFLUENCE_TRUNCATED = CritiqueDefinition(
    code="STRUCTURAL_INFLUENCE_TRUNCATED",
    severity="warning",
    source="analysis",
    message_template=(
        "Structural influence for factor(s) {factor_ids} was computed from a "
        "truncated path enumeration ({budget} walk budget exhausted) — reported "
        "influence for these factors is a lower bound and influence_rank may be "
        "affected"
    ),
    default_suggestion=(
        "Dense graphs exceed the exact path-enumeration budget; consider reducing "
        "edge density if exact structural influence ranking matters"
    ),
)

HIGH_TIE_RATE = CritiqueDefinition(
    code="HIGH_TIE_RATE",
    severity="warning",
    source="analysis",
    message_template=(
        "{tie_rate_pct}% of samples resulted in ties between options — "
        "win probabilities may be misleading"
    ),
    default_suggestion=(
        "High tie rates often indicate sparse edge existence or "
        "interventions with no effective path to goal"
    ),
)

CONSTRAINT_NODE_DEFAULT_BASE = CritiqueDefinition(
    code="CONSTRAINT_NODE_DEFAULT_BASE",
    severity="warning",
    source="analysis",
    message_template=(
        "Constraint node '{node_id}' has no ParameterUncertainty and is non-root "
        "— defaulted to base=0.0, and its root ancestor(s) {gap_roots} carry no "
        "observed value or ParameterUncertainty (defaulted to 0.0); constraint "
        "probability may be unreliable"
    ),
    default_suggestion=(
        "Provide observed values or ParameterUncertainty entries for the listed "
        "root ancestors so the propagated composition is data-supported. Note: a "
        "ParameterUncertainty base on a non-root node is ADDED to parent "
        "propagation — it does not pin the node's value; use the node's "
        "intercept for a fixed exogenous offset"
    ),
)

# Cluster-2 (Track S Phase 0): fires for the SAME underlying condition as
# CONSTRAINT_NODE_DEFAULT_BASE (non-root, non-objective constraint target
# without ParameterUncertainty) but when every root ancestor of the target
# carries data (observed value, ParameterUncertainty, or an intervention in
# every option). There the target's samples are a fully-supported
# forward-propagated composition of its parents — the base=0.0 is a zero
# EXOGENOUS OFFSET, not a missing-data placeholder, and the old
# "may be unreliable" wording misrepresented expected propagation as a data
# gap. Same `code` and severity (downstream consumers key on code, not
# message text — ROADMAP 1.26b precedent); only message/suggestion differ.
CONSTRAINT_NODE_DEFAULT_BASE_SUPPORTED = CritiqueDefinition(
    code="CONSTRAINT_NODE_DEFAULT_BASE",
    severity="warning",
    source="analysis",
    message_template=(
        "Constraint node '{node_id}' has no ParameterUncertainty and is non-root "
        "— base offset defaulted to 0.0; its samples are the forward-propagated "
        "composition of its parents (all root ancestors carry data), so the "
        "constraint probability is model-derived, not a missing-data placeholder"
    ),
    default_suggestion=(
        "No action needed if parent propagation is the intended semantics. Use "
        "the node's intercept for a fixed exogenous offset; note a "
        "ParameterUncertainty base on a non-root node is ADDED to parent "
        "propagation — it does not pin the node's value"
    ),
)

# Doctrine-B variant (post-#204): fires for the SAME underlying condition
# (non-root constraint target, no ParameterUncertainty) but when the target IS
# the graph's objective node (goal_node_id). There, base=0.0 is the expected,
# ratified default — the constraint probability is scored from the same
# forward-propagated outcome-distribution series used for the outcome stats,
# not left unmeasured for lack of data. Using the generic "may be unreliable"
# wording here misrepresents expected behaviour as a data gap. Same `code`
# (downstream consumers key on code, not on message text) and severity — only
# the message/suggestion differ.
CONSTRAINT_NODE_DEFAULT_BASE_OBJECTIVE = CritiqueDefinition(
    code="CONSTRAINT_NODE_DEFAULT_BASE",
    severity="warning",
    source="analysis",
    message_template=(
        "Constraint node '{node_id}' is the objective node and has no "
        "ParameterUncertainty — defaulted to base=0.0 as expected for a "
        "non-root objective; its constraint probability is scored from the "
        "modelled outcome distribution, not a missing-data placeholder"
    ),
    default_suggestion=(
        "This is expected for a non-root objective node without an explicit "
        "baseline. Supply a ParameterUncertainty entry only if you intend an "
        "additive exogenous base on top of parent propagation — the sampled "
        "base is ADDED to parent contributions, it does not replace them"
    ),
)


# Cluster-2 (Track S Phase 0): the goal node's distribution is the
# forward-propagated composition of its parents (doctrine B). When one or
# more root ancestors of the goal carry NO data (no observed value, no
# ParameterUncertainty, not intervened by every option) they default to 0.0,
# so goal-level probabilities partially rest on placeholder zeros. Science
# honesty: no priors are invented — the gap is disclosed instead.
GOAL_ANCESTOR_DATA_GAP = CritiqueDefinition(
    code="GOAL_ANCESTOR_DATA_GAP",
    severity="warning",
    source="analysis",
    message_template=(
        "Goal node '{node_id}' is scored from its forward-propagated outcome "
        "distribution, but root ancestor(s) {gap_roots} carry no observed value "
        "or ParameterUncertainty and defaulted to 0.0 — goal-level "
        "probabilities partially rest on placeholder zeros (insufficient data)"
    ),
    default_suggestion=(
        "Provide observed values or ParameterUncertainty entries for the listed "
        "root ancestors; until then the honest reading of goal-fit is "
        "'insufficient data', not a calibrated probability"
    ),
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
    "DEGENERATE_OPTION_ZERO_VARIANCE": DEGENERATE_OPTION_ZERO_VARIANCE,
    "HIGH_TIE_RATE": HIGH_TIE_RATE,
    "CONSTRAINT_NODE_DEFAULT_BASE": CONSTRAINT_NODE_DEFAULT_BASE,
    "GOAL_ANCESTOR_DATA_GAP": GOAL_ANCESTOR_DATA_GAP,
    # Engine
    "INTERNAL_ERROR": INTERNAL_ERROR,
    "INFERENCE_TIMEOUT": INFERENCE_TIMEOUT,
}


def get_critique(code: str) -> CritiqueDefinition:
    """Get critique definition by code."""
    if code not in CRITIQUES:
        raise ValueError(f"Unknown critique code: {code}")
    return CRITIQUES[code]
