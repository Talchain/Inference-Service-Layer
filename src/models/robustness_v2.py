"""
Robustness analysis models for v2.2 dual uncertainty schema.

Supports both structural uncertainty (edge existence) and parametric
uncertainty (effect magnitude) for proper robustness analysis.
"""

from __future__ import annotations

import hashlib
import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
import re

from src.constants import (
    DEFAULT_EXISTS_PROBABILITY,
    MAX_CONTROL_CANDIDATES,
    MAX_CONTROL_VALUES,
    MAX_FACTOR_CORRELATIONS,
    MAX_GRAPH_EDGES,
    MAX_GRAPH_NODES,
    MAX_OPTIONS,
    MAX_PARAMETER_UNCERTAINTIES,
    NON_INFERENCE_KINDS,
)

# Import from response_v2 (no circular import since response_v2 doesn't import this module)
from src.models.response_v2 import (
    CorrelationModelV2,
    CritiqueV2,
    InferenceWarning,
    StabilityThresholdsResponse,
    ZeroSensitivityReason,
)


# Distribution families the Gaussian-copula marginal transform supports (B3-S1).
# ALLOWLIST, not a blocklist: the correlation validator rejects any family not in
# this set with a typed 422, so a future distribution family fails loud at request
# validation rather than reaching _copula_transform's fail-closed branch (which the
# enhanced handler maps to a 500). Keep in lockstep with FactorSampler._copula_transform.
_CORRELATION_SUPPORTED_DISTRIBUTIONS = frozenset({"normal", "uniform"})


# =============================================================================
# Enums
# =============================================================================


class NodeKindV2(str, Enum):
    """Node types in v2 causal graphs."""

    FACTOR = "factor"
    DECISION = "decision"
    CHANCE = "chance"
    OPTION = "option"
    OUTCOME = "outcome"
    GOAL = "goal"
    RISK = "risk"
    ACTION = "action"


class SensitivityType(str, Enum):
    """Types of sensitivity analysis."""

    EXISTENCE = "existence"
    MAGNITUDE = "magnitude"


# ZeroSensitivityReason is defined in response_v2.py to avoid circular imports
# Import it from there: from src.models.response_v2 import ZeroSensitivityReason


# =============================================================================
# Core V2 Schema Components
# =============================================================================


class StrengthDistribution(BaseModel):
    """
    Parametric uncertainty over edge effect magnitude.

    Represents a Normal distribution over the causal effect strength.
    Positive mean = positive causal effect (increase in cause -> increase in effect)
    Negative mean = negative causal effect (increase in cause -> decrease in effect)

    Validation:
    - mean is clamped to [-1.0, 1.0] at parse time.  The original value is
      stored in _pre_clamp_mean so that EdgeV2 can emit an InferenceWarning
      with the correct edge identity.
    - NaN or Inf values for mean or std are rejected with a ValidationError —
      they cannot be clamped to a meaningful range.
    """

    mean: float = Field(
        ...,
        description="Expected effect size (SIGNED: negative = negative effect), clamped to [-1, 1]",
    )
    std: float = Field(
        ..., gt=0.001, description="Standard deviation of effect size (must be > 0.001)"
    )

    # Private: records the original mean before clamping (None if no clamp occurred).
    # Checked by EdgeV2.model_validator to emit the STRENGTH_MEAN_CLAMPED InferenceWarning.
    _pre_clamp_mean: Optional[float] = PrivateAttr(default=None)

    @field_validator("mean", mode="before")
    @classmethod
    def validate_mean(cls, v: Any) -> float:
        """Reject NaN/Inf; clamp to [-1.0, 1.0]."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"mean must be a number, got {type(v).__name__}")
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"mean must be a finite number (no NaN or Inf), got {v}")
        return float(v)  # actual clamp happens in model_validator after all fields are set

    @field_validator("std", mode="before")
    @classmethod
    def validate_std(cls, v: Any) -> float:
        """Reject NaN/Inf for std."""
        if not isinstance(v, (int, float)):
            raise ValueError(f"std must be a number, got {type(v).__name__}")
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"std must be a finite number (no NaN or Inf), got {v}")
        return float(v)

    @model_validator(mode="after")
    def clamp_mean(self) -> "StrengthDistribution":
        """Clamp mean to [-1.0, 1.0] and record the original if clamped."""
        original = self.mean
        clamped = max(-1.0, min(1.0, original))
        if clamped != original:
            # Store original so EdgeV2 can emit InferenceWarning with edge identity
            object.__setattr__(self, "_pre_clamp_mean", original)
            object.__setattr__(self, "mean", clamped)
        return self

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {"extra": "ignore", "json_schema_extra": {"example": {"mean": 0.5, "std": 0.1}}}


class ObservedState(BaseModel):
    """
    Observed state for quantitative factor nodes.

    Captures the current observed value of a factor, along with optional
    baseline for comparison, display unit, and data provenance.

    This supports the v2.2 schema where factor nodes can carry actual
    observed values from CEE extraction or user input.
    """

    value: float = Field(
        ..., description="Current observed value in user units (e.g., 59 for £59k revenue)"
    )
    baseline: Optional[float] = Field(
        None, description="Reference/baseline value for comparison (e.g., 49 for £49k baseline)"
    )
    unit: Optional[str] = Field(
        None, max_length=50, description="Display unit (e.g., '£', '%', 'users', 'k')"
    )
    source: Optional[str] = Field(
        None,
        max_length=100,
        description="Data provenance (e.g., 'brief_extraction', 'user_input', 'computed')",
    )
    # CIL 0.2: declared std per v2.6 canonical schema (PLoT sends this field)
    std: Optional[float] = Field(
        None, description="Standard deviation / uncertainty of the observed value"
    )
    # CIL: passthrough fields from CEE — ISL preserves these for downstream consumers (ISL-6)
    raw_value: Optional[float] = Field(None, description="Original pre-normalised value from CEE")
    cap: Optional[float] = Field(None, description="Upper bound for normalisation range from CEE")
    extractionType: Optional[str] = Field(
        None,
        description="How the value was extracted (e.g., 'explicit', 'inferred'). "
        "camelCase matches CEE output — do not rename.",
    )
    factor_type: Optional[str] = Field(None, description="Factor classification from CEE")
    uncertainty_drivers: Optional[List[str]] = Field(
        None, description="List of uncertainty sources for this factor from CEE"
    )

    @field_validator("value")
    @classmethod
    def value_must_be_finite(cls, v: float) -> float:
        """Validate that value is a finite number (not NaN or infinity)."""
        if not math.isfinite(v):
            raise ValueError("value must be finite (not NaN or infinity)")
        return v

    @field_validator("baseline")
    @classmethod
    def baseline_must_be_finite(cls, v: Optional[float]) -> Optional[float]:
        """Validate that baseline, if provided, is finite."""
        if v is not None and not math.isfinite(v):
            raise ValueError("baseline must be finite (not NaN or infinity)")
        return v

    @field_validator("std", "raw_value", "cap")
    @classmethod
    def optional_floats_must_be_finite(cls, v: Optional[float]) -> Optional[float]:
        """Validate that optional numeric fields, if provided, are finite."""
        if v is not None and not math.isfinite(v):
            raise ValueError("value must be finite (not NaN or infinity)")
        return v

    # CIL 0.2: accept unknown fields per cross-service contract
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "value": 59.0,
                "baseline": 49.0,
                "unit": "£k",
                "source": "brief_extraction",
                "std": 5.0,
            }
        },
    }


class ParameterUncertainty(BaseModel):
    """
    Uncertainty specification for a factor node's value.

    Defines how to sample a factor's value during Monte Carlo analysis.
    The mean is typically taken from the node's `observed_state.value`.

    Supported distributions:
    - "normal": Sample from Normal(observed_value, std)
    - "uniform": Sample uniformly from [range_min, range_max]
    - "point_mass": Use observed_value exactly (no sampling)
    """

    node_id: str = Field(
        ...,
        pattern=r"^[a-z0-9_:-]+$",
        description="ID of the factor node this uncertainty applies to",
    )
    distribution: str = Field(
        default="normal", description="Distribution family: 'normal', 'uniform', 'point_mass'"
    )
    std: Optional[float] = Field(
        None, ge=0, description="Standard deviation for Normal sampling around observed_state.value"
    )
    # For uniform distribution
    range_min: Optional[float] = Field(None, description="Minimum value for uniform distribution")
    range_max: Optional[float] = Field(None, description="Maximum value for uniform distribution")

    @model_validator(mode="after")
    def validate_distribution_params(self) -> "ParameterUncertainty":
        """Validate distribution-specific parameters."""
        if self.distribution == "normal":
            if self.std is None or self.std <= 0:
                raise ValueError(
                    f"For normal distribution, 'std' must be provided and > 0 "
                    f"(got std={self.std})"
                )
        elif self.distribution == "uniform":
            if self.range_min is None or self.range_max is None:
                raise ValueError(
                    "For uniform distribution, both 'range_min' and 'range_max' must be provided"
                )
            if self.range_min >= self.range_max:
                raise ValueError(
                    f"For uniform distribution, range_min ({self.range_min}) "
                    f"must be less than range_max ({self.range_max})"
                )
        elif self.distribution == "point_mass":
            pass  # No additional params needed
        else:
            raise ValueError(
                f"Unknown distribution '{self.distribution}'. "
                f"Supported: 'normal', 'uniform', 'point_mass'"
            )
        return self

    # CIL 0.2: accept unknown fields per cross-service contract
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {"node_id": "marketing_spend", "distribution": "normal", "std": 2.5}
        },
    }


class FactorCorrelation(BaseModel):
    """A single pairwise correlation between two factor uncertainties (B3-S1).

    Correlated factors are sampled JOINTLY via a Gaussian copula over their
    existing marginals (D-23.4). Independence remains the default: correlation
    is inert-when-absent and activates ONLY when at least one of these is
    supplied. No default correlations are ever invented.

    Semantic cross-checks (both factors exist, both carry a sampled
    ``parameter_uncertainty``, distribution supported, no duplicate/conflicting
    or invalid self-pairs) are enforced by
    ``RobustnessRequestV2.validate_factor_correlations`` so the messages can name
    the offending factor ids without echoing any other request values.
    """

    factor_a: str = Field(
        ...,
        pattern=r"^[a-z0-9_:-]+$",
        description="ID of the first factor node in the correlated pair",
    )
    factor_b: str = Field(
        ...,
        pattern=r"^[a-z0-9_:-]+$",
        description="ID of the second factor node in the correlated pair",
    )
    rho: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Pearson correlation coefficient in [-1, 1] applied to the two "
        "factors' marginals via a Gaussian copula",
    )

    # CIL 0.2: accept unknown fields per cross-service contract
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {"factor_a": "marketing_spend", "factor_b": "demand", "rho": 0.6}
        },
    }


class EdgeV2(BaseModel):
    """
    Edge with dual uncertainty.

    Combines structural uncertainty (does the edge exist?) with
    parametric uncertainty (how strong is the effect?).
    """

    from_: str = Field(..., alias="from", pattern=r"^[a-z0-9_:-]+$", description="Source node ID")
    to: str = Field(..., pattern=r"^[a-z0-9_:-]+$", description="Target node ID")
    exists_probability: float = Field(
        default=DEFAULT_EXISTS_PROBABILITY,
        ge=0,
        le=1,
        description="P(edge exists) - structural uncertainty. Defaults to 0.8 when not provided.",
    )
    strength: StrengthDistribution = Field(
        ..., description="Effect magnitude distribution - parametric uncertainty"
    )
    label: Optional[str] = Field(
        None, description="Human-readable edge description", max_length=500
    )
    edge_type: Optional[Literal["directed", "bidirected"]] = Field(
        None,
        description="Edge directionality. 'directed' (default when absent) = causal edge. "
        "'bidirected' = unmeasured confounder between two nodes (used by identifiability analysis).",
    )

    # Private: populated by model_validator when strength.mean was clamped.
    # Callers (GraphV2 or the analyzer) must collect these to build inference_warnings.
    _strength_clamp_warning: Optional[InferenceWarning] = PrivateAttr(default=None)

    # Private: populated by model_validator when exists_probability used the default.
    _exists_probability_default_warning: Optional[InferenceWarning] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def emit_strength_clamp_warning(self) -> "EdgeV2":
        """Record an InferenceWarning if strength.mean was clamped during parsing."""
        pre_clamp = self.strength._pre_clamp_mean
        if pre_clamp is not None:
            warning = InferenceWarning(
                code="STRENGTH_MEAN_CLAMPED",
                field=f"edges[{self.from_}\u2192{self.to}].strength.mean",
                detail={"original": pre_clamp, "clamped": self.strength.mean},
            )
            object.__setattr__(self, "_strength_clamp_warning", warning)
        return self

    @model_validator(mode="after")
    def emit_exists_probability_default_warning(self) -> "EdgeV2":
        """Record an InferenceWarning if exists_probability was not explicitly provided."""
        if "exists_probability" not in self.model_fields_set:
            warning = InferenceWarning(
                code="EXISTS_PROBABILITY_DEFAULT",
                field=f"edges[{self.from_}\u2192{self.to}].exists_probability",
                detail={
                    "edge_from": self.from_,
                    "edge_to": self.to,
                    "default_value": DEFAULT_EXISTS_PROBABILITY,
                },
            )
            object.__setattr__(self, "_exists_probability_default_warning", warning)
        return self

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "from": "marketing",
                "to": "demand",
                "exists_probability": 0.9,
                "strength": {"mean": 0.6, "std": 0.15},
                "label": "Marketing increases demand",
            }
        },
        "populate_by_name": True,
    }


class NodeV2(BaseModel):
    """Node in the v2 causal graph."""

    id: str = Field(..., pattern=r"^[a-z0-9_:-]+$", description="Unique node identifier")
    kind: str = Field(..., description="Node type (factor, decision, chance, outcome, etc.)")
    label: str = Field(..., description="Human-readable node name", max_length=500)
    body: Optional[str] = Field(None, description="Detailed description", max_length=5000)
    observed_state: Optional[ObservedState] = Field(
        None,
        description="Observed state for quantitative factor nodes (value, baseline, unit, source)",
    )
    intercept: float = Field(
        default=0.0,
        description="Node intercept term (constant added to structural equation). "
        "Represents the baseline value when all parent contributions are zero.",
    )
    epsilon_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Per-node noise standard deviation. When > 0, adds N(0, epsilon_std) "
        "noise to the structural equation output each MC sample, representing "
        "unexplained variance (measurement error, omitted variables). "
        "Uses dedicated RNG stream (seed+3) for determinism.",
    )
    # CIL: preserve CEE node categorisation for downstream consumers (ISL-5)
    category: Optional[str] = Field(
        None,
        description="Node category from CEE (e.g., 'market', 'operational'). "
        "Passthrough only — not used by ISL computation.",
    )
    # CIL: node-level factor type from CEE. Also present on ObservedState for
    # backward compat — PLoT may send at either or both levels. ISL preserves
    # both; no precedence rule (neither is used by ISL computation).
    factor_type: Optional[str] = Field(
        None,
        description="Factor classification from CEE (e.g., 'market', 'operational'). "
        "Passthrough only — not used by ISL computation.",
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "id": "revenue",
                "kind": "outcome",
                "label": "Total Revenue",
                "intercept": 0.0,
                "epsilon_std": 0.0,
                "category": "financial",
                "factor_type": "market",
                "observed_state": {"value": 59.0, "baseline": 49.0, "unit": "£k"},
            }
        },
    }


class GraphV2(BaseModel):
    """
    Causal graph with dual uncertainty edges.

    Represents a directed acyclic graph where each edge has both
    structural uncertainty (exists_probability) and parametric
    uncertainty (strength distribution).
    """

    nodes: List[NodeV2] = Field(
        ..., min_length=1, max_length=MAX_GRAPH_NODES, description="List of graph nodes"
    )
    edges: List[EdgeV2] = Field(
        ..., max_length=MAX_GRAPH_EDGES, description="List of directed edges with dual uncertainty"
    )

    @field_validator("nodes")
    @classmethod
    def validate_unique_node_ids(cls, v: List[NodeV2]) -> List[NodeV2]:
        """Validate node IDs are unique."""
        node_ids = [node.id for node in v]
        if len(node_ids) != len(set(node_ids)):
            duplicates = [nid for nid in node_ids if node_ids.count(nid) > 1]
            raise ValueError(f"Duplicate node IDs found: {list(set(duplicates))}")
        return v

    @field_validator("edges")
    @classmethod
    def validate_edges_reference_nodes(cls, v: List[EdgeV2], info: Any) -> List[EdgeV2]:
        """Validate edges reference existing nodes."""
        if "nodes" in info.data:
            node_ids = {node.id for node in info.data["nodes"]}
            for edge in v:
                if edge.from_ not in node_ids:
                    raise ValueError(f"Edge references non-existent source node: {edge.from_}")
                if edge.to not in node_ids:
                    raise ValueError(f"Edge references non-existent target node: {edge.to}")
        return v

    @field_validator("edges")
    @classmethod
    def validate_no_self_loops(cls, v: List[EdgeV2]) -> List[EdgeV2]:
        """Validate no self-loops exist."""
        for edge in v:
            if edge.from_ == edge.to:
                raise ValueError(f"Self-loop detected on node: {edge.from_}")
        return v

    @field_validator("edges")
    @classmethod
    def validate_no_duplicate_edges(cls, v: List[EdgeV2]) -> List[EdgeV2]:
        """Reject exact duplicate directed edges.

        Two edges with the same (from, to) pair and the same edge_type would
        double-count the same causal effect in the linear SCM (both are
        sampled and both contribute parent_value * strength). Edges sharing
        endpoints but differing in edge_type (directed vs bidirected
        confounder) are semantically distinct and remain allowed.
        """
        seen: set = set()
        duplicates: List[str] = []
        for edge in v:
            key = (edge.from_, edge.to, edge.edge_type or "directed")
            if key in seen:
                duplicates.append(f"{edge.from_}->{edge.to}")
            seen.add(key)
        if duplicates:
            raise ValueError(f"Duplicate edges found: {sorted(set(duplicates))}")
        return v

    def collect_parse_warnings(self) -> List[InferenceWarning]:
        """
        Return all InferenceWarnings generated during graph parsing.

        Collects:
        - STRENGTH_MEAN_CLAMPED: edges whose strength.mean was clamped to [-1, 1]
        - EXISTS_PROBABILITY_DEFAULT: edges where exists_probability used the 0.8 default

        Call this after constructing the graph to retrieve warnings that must
        be forwarded to the response's inference_warnings field.
        """
        warnings: List[InferenceWarning] = []
        for edge in self.edges:
            if edge._strength_clamp_warning is not None:
                warnings.append(edge._strength_clamp_warning)
            if edge._exists_probability_default_warning is not None:
                warnings.append(edge._exists_probability_default_warning)
        return warnings

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "nodes": [
                    {"id": "price", "kind": "decision", "label": "Price"},
                    {"id": "revenue", "kind": "outcome", "label": "Revenue"},
                ],
                "edges": [
                    {
                        "from": "price",
                        "to": "revenue",
                        "exists_probability": 0.95,
                        "strength": {"mean": 0.5, "std": 0.1},
                    }
                ],
            }
        },
    }


class InterventionOption(BaseModel):
    """A decision option with its interventions."""

    id: str = Field(..., description="Unique option identifier")
    label: str = Field(..., description="Human-readable option name", max_length=500)
    interventions: Dict[str, float] = Field(
        ..., description="node_id -> intervention value mapping"
    )

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "id": "low_price",
                "label": "Keep price at $49",
                "interventions": {"price": 0.49},
            }
        },
    }


class GoalConstraint(BaseModel):
    """
    A constraint on a goal/outcome node for multi-constraint analysis.

    Allows specifying success criteria like "revenue >= 100k" or "cost <= 50k".
    Multiple constraints can be specified to compute joint probabilities.
    """

    node_id: str = Field(
        ...,
        pattern=r"^[a-z0-9_:-]+$",
        description="ID of the node this constraint applies to (must exist in graph)",
    )
    operator: Literal[">=", "<="] = Field(
        ...,
        description="Comparison operator: '>=' for minimum threshold, '<=' for maximum threshold",
    )
    value: float = Field(
        ..., description="Threshold value for the constraint (v2.7 contract field name)"
    )
    label: Optional[str] = Field(
        None,
        max_length=200,
        description="Human-readable label for coaching (e.g., 'Revenue target', 'Budget cap')",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_threshold(cls, values: Any) -> Any:
        """Accept legacy 'threshold' field and map it to 'value' for backward compat."""
        if isinstance(values, dict) and "value" not in values and "threshold" in values:
            values["value"] = values.pop("threshold")
        return values

    @property
    def threshold(self) -> float:
        """Alias so internal computation can still reference constraint.threshold."""
        return self.value

    @field_validator("value")
    @classmethod
    def validate_value_finite(cls, v: float) -> float:
        """Reject NaN and infinite values for value."""
        if not math.isfinite(v):
            raise ValueError("value must be a finite number, not NaN or infinite")
        return v

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "node_id": "revenue",
                "operator": ">=",
                "value": 100000.0,
                "label": "Revenue target",
            }
        },
    }


# =============================================================================
# Request Schema
# =============================================================================


class ControlCandidate(BaseModel):
    """A value-of-control (EVPC) candidate: a factor the user could pull to one of
    a bounded set of chosen values (A3 S4, D-23.8).

    For each candidate value ``x`` the engine evaluates ``do(factor_id = x)`` on the
    SAME retained joint Common-Random-Numbers draws that scored the options (the
    intervention overrides the drawn value for this factor while every other factor —
    including its correlated partners — keeps its per-sample joint draw), and reports
    ``EVPC = max_x E[U | do(factor_id=x)] − max_a E[U_a]`` in outcome units. This is a
    request-side input only; ISL lands it FIRST because the request models carry
    ``extra:"ignore"`` and would silently drop a producer-first field.
    """

    factor_id: str = Field(
        ...,
        description="ID of the graph node to control (do-intervene). Must exist in "
        "the graph and must not be the goal node itself.",
    )
    values: List[float] = Field(
        ...,
        min_length=1,
        max_length=MAX_CONTROL_VALUES,
        description="Candidate values to grid over for do(factor_id=value). At least "
        f"one, at most {MAX_CONTROL_VALUES}. Each value must be finite. More values "
        "tighten the grid approximation of the true (continuous) value of control.",
    )

    @field_validator("values")
    @classmethod
    def validate_values_finite(cls, v: List[float], info: Any) -> List[float]:
        """Reject NaN/inf candidate values with a factor-named 422.

        The message names only the offending factor id — never any other request
        value — mirroring the correlation validator's disclosure discipline.
        """
        factor_id = info.data.get("factor_id", "<unknown>")
        for value in v:
            if not math.isfinite(value):
                raise ValueError(
                    "control_candidates: values must all be finite numbers (not NaN "
                    f"or infinite) for factor '{factor_id}'"
                )
        return v

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {"factor_id": "price", "values": [0.3, 0.5, 0.7]},
        },
    }


class RobustnessRequestV2(BaseModel):
    """
    V2.2 robustness analysis request.

    Accepts a causal graph with dual uncertainty edges, a set of decision
    options, and configuration for Monte Carlo sampling and analysis.
    """

    request_id: Optional[str] = Field(
        None, description="Optional request ID for tracing. Generated if not provided."
    )
    graph: GraphV2 = Field(..., description="Causal graph with dual uncertainty edges")
    options: List[InterventionOption] = Field(
        ..., min_length=1, max_length=MAX_OPTIONS, description="Decision options to compare"
    )
    goal_node_id: str = Field(..., description="Target outcome node to optimize")

    # Sampling configuration
    n_samples: int = Field(
        default=1000, ge=100, le=10000, description="Number of Monte Carlo samples"
    )
    # Task 4: Accept str | int | None for cross-service compatibility.
    # CEE/UI/PLoT may send seed as string; normalised to int internally.
    seed: Optional[Union[int, str]] = Field(
        default=None,
        description="Random seed for reproducibility; if None, computed from graph. "
        "Accepts int or string. Numeric strings are converted via int(); "
        "non-numeric strings are hashed deterministically.",
    )

    @field_validator("seed", mode="before")
    @classmethod
    def normalise_seed_to_int(cls, v: Any) -> Optional[int]:
        """Normalise seed to int: numeric strings → int, non-numeric → deterministic hash."""
        if v is None:
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                # Stable deterministic hash for non-numeric strings (e.g. "my_seed").
                # Uses SHA-256 instead of Python's hash() which is randomized
                # per process (PEP 456) and would break cross-service determinism.
                return int(hashlib.sha256(v.encode("utf-8")).hexdigest(), 16) % (2**31)
        # Fallback: try int conversion
        return int(v)

    # Analysis configuration
    analysis_types: List[str] = Field(
        default=["comparison", "sensitivity", "robustness"],
        description="Types of analysis to perform",
    )
    confidence_level: float = Field(
        default=0.95, ge=0.5, le=0.99, description="Confidence level for intervals"
    )

    # Factor uncertainty configuration (Phase 2A Part 2)
    # F8: capped at MAX_PARAMETER_UNCERTAINTIES (eligible nodes <= graph.nodes cap).
    # Previously unbounded, which let duplicate/oversized lists multiply EVPI MC
    # passes at zero admitted cost (the "duplicate free-ride"). Uniqueness is
    # enforced by validate_parameter_uncertainties_reference_nodes below.
    parameter_uncertainties: Optional[List[ParameterUncertainty]] = Field(
        None,
        max_length=MAX_PARAMETER_UNCERTAINTIES,
        description="Uncertainty specifications for factor node values. "
        "If not provided, factor nodes use observed_state.value as fixed values. "
        "Each node_id may appear at most once (duplicates are redundant: EVPI on a "
        "repeated node_id produces byte-identical rows).",
    )

    # B3-S1 correlated factors (D-23.4). Additive + optional. When absent the
    # analysis is BYTE-IDENTICAL to the independent-factor path (inert-when-absent);
    # when supplied, the named factors are sampled JOINTLY via a Gaussian copula
    # over their marginals. No default correlations are ever invented.
    factor_correlations: Optional[List[FactorCorrelation]] = Field(
        None,
        max_length=MAX_FACTOR_CORRELATIONS,
        description="Pairwise correlations between factor uncertainties. Independence "
        "is the default; supplying any pair activates a Gaussian copula over the named "
        "factors' marginals. Each referenced factor must have a parameter_uncertainty "
        "with a supported distribution (normal or uniform; point_mass factors have zero "
        "variance and cannot be correlated). Under active correlation, independence-"
        "assuming per-factor attributions (factor_sensitivity, p_win_sensitivity, "
        "conditional_winners) are omitted with a disclosure marker; joint quantities "
        "(win_probability, downside, percentiles, factor_evppi) remain valid.",
    )

    # Goal threshold configuration (single constraint, legacy)
    goal_threshold: Optional[float] = Field(
        None,
        description="Success threshold for goal outcome. When provided, "
        "computes probability_of_goal (fraction of samples meeting/exceeding threshold).",
    )

    # Enhancement flags
    include_e_values: bool = Field(
        default=False,
        description="Compute E-value analogue per edge (minimum strength perturbation "
        "to flip recommendation). Gated for performance — adds up to 2s latency.",
    )
    include_voi: bool = Field(
        default=False,
        description="Compute Expected Value of Perfect Information (EVPI) per factor. "
        "Requires parameter_uncertainties.",
    )
    include_path_decomposition: bool = Field(
        default=False,
        description="Compute a structural pathway decomposition for the recommended "
        "option's retained intervention targets: the top-3 simple directed paths to the "
        "goal, each with its signed structural contribution (analytic path tracing). "
        "Gated; off by default. This is a structural decomposition of the modelled effect, "
        "not a real-world causal claim, and is not weighted by intervention magnitude.",
    )

    # Multi-constraint goal analysis (Phase 2)
    goal_constraints: Optional[List[GoalConstraint]] = Field(
        None,
        max_length=20,
        description="Multiple goal constraints for joint probability analysis. "
        "When provided, computes per-constraint probabilities, joint probability, "
        "and conditional probabilities. Requires nodes to exist in graph.",
    )

    # A3 S4 value-of-control (EVPC, D-23.8). Additive + optional, request-driven.
    # When absent the analysis is BYTE-IDENTICAL (inert-when-absent); when supplied,
    # ISL emits the top-level `factor_evpc` block. NOT coupled to include_voi: EVPI/
    # EVPPI (include_voi) measure the value of INFORMATION about a factor, EVPC
    # measures the value of CONTROLLING one — distinct capabilities, so control_
    # candidates presence is its own sufficient opt-in gate. Landed ISL-first because
    # request models carry extra:"ignore" (a producer-first field would be dropped).
    control_candidates: Optional[List[ControlCandidate]] = Field(
        None,
        max_length=MAX_CONTROL_CANDIDATES,
        description="Value-of-control (EVPC) candidates: factors the user could pull "
        "to chosen values. When provided, ISL grids do(factor=value) over each "
        "candidate's values on the retained joint CRN samples and emits factor_evpc "
        f"(per-lever EVPC in outcome units). At most {MAX_CONTROL_CANDIDATES} "
        "candidates; each factor_id must exist in the graph, must not be the goal "
        "node, and may appear at most once. Independent of include_voi.",
    )

    @field_validator("options")
    @classmethod
    def validate_unique_option_ids(cls, v: List[InterventionOption]) -> List[InterventionOption]:
        """Validate option IDs are unique (matches GraphV2.validate_unique_node_ids pattern)."""
        option_ids = [opt.id for opt in v]
        if len(option_ids) != len(set(option_ids)):
            duplicates = [oid for oid in option_ids if option_ids.count(oid) > 1]
            raise ValueError(f"Duplicate option IDs found: {list(set(duplicates))}")
        return v

    @field_validator("goal_threshold")
    @classmethod
    def validate_goal_threshold_finite(cls, v: Optional[float]) -> Optional[float]:
        """Reject NaN and infinite values for goal_threshold."""
        import math

        if v is not None and (math.isnan(v) or math.isinf(v)):
            raise ValueError("goal_threshold must be a finite number, not NaN or infinite")
        return v

    @field_validator("goal_node_id")
    @classmethod
    def validate_goal_node_exists(cls, v: str, info: Any) -> str:
        """Validate goal node exists in graph."""
        if "graph" in info.data:
            node_ids = {node.id for node in info.data["graph"].nodes}
            if v not in node_ids:
                raise ValueError(f"Goal node '{v}' not found in graph")
        return v

    @model_validator(mode="after")
    def validate_interventions_reference_nodes(self) -> "RobustnessRequestV2":
        """Validate all intervention nodes exist in graph."""
        node_ids = {node.id for node in self.graph.nodes}
        for option in self.options:
            for node_id in option.interventions.keys():
                if node_id not in node_ids:
                    raise ValueError(
                        f"Option '{option.id}' references non-existent node: {node_id}"
                    )
        return self

    @model_validator(mode="after")
    def validate_parameter_uncertainties_reference_nodes(self) -> "RobustnessRequestV2":
        """Validate parameter_uncertainties: node existence AND uniqueness (F8).

        Uniqueness is enforced here (fail-closed at parse time → typed 422 before
        any compute) because a repeated node_id is definitionally redundant: the
        win-probability sensitivity sweep seeds each per-factor pass on
        ``f"{seed}:evpi:{node_id}"``, so duplicate node_ids produce byte-identical
        p_win_sensitivity rows while still costing a full
        Monte Carlo pass each. Rejecting duplicates removes the free-ride and lets
        the admission gate price only the deduplicated factor count.
        """
        if self.parameter_uncertainties:
            node_ids = {node.id for node in self.graph.nodes}
            seen: set[str] = set()
            duplicates: list[str] = []
            for uncertainty in self.parameter_uncertainties:
                if uncertainty.node_id not in node_ids:
                    raise ValueError(
                        f"ParameterUncertainty references non-existent node: {uncertainty.node_id}"
                    )
                if uncertainty.node_id in seen:
                    duplicates.append(uncertainty.node_id)
                seen.add(uncertainty.node_id)
            if duplicates:
                raise ValueError(
                    "Duplicate parameter_uncertainties node_ids (each node may appear "
                    f"at most once): {sorted(set(duplicates))}"
                )
        return self

    @model_validator(mode="after")
    def validate_goal_constraints_reference_nodes(self) -> "RobustnessRequestV2":
        """Validate all goal_constraints node_ids exist in graph."""
        if self.goal_constraints:
            node_ids = {node.id for node in self.graph.nodes}
            for constraint in self.goal_constraints:
                if constraint.node_id not in node_ids:
                    raise ValueError(
                        f"GoalConstraint references non-existent node: {constraint.node_id}"
                    )
        return self

    @model_validator(mode="after")
    def validate_factor_correlations(self) -> "RobustnessRequestV2":
        """Validate factor_correlations (B3-S1) → fail closed with a typed 422.

        Hard-invalid inputs (D-23.4) are rejected here so a malformed correlation
        never reaches the copula sampler. Messages name the offending factor
        id(s) only — never any other request value:

        - a referenced factor is not a graph node (unknown factor id),
        - a referenced factor has no parameter_uncertainty (nothing to correlate),
        - a referenced factor's distribution is not in the supported allowlist
          {normal, uniform} (point_mass has zero variance; any other family has no
          copula marginal transform — allowlisted so a new family fails loud here),
        - a self-pair (factor_a == factor_b), any rho (a factor is trivially
          correlated with itself; the pair declares no cross-factor dependence yet
          would activate the correlation regime — B3 P3-2),
        - a duplicate unordered pair {a, b} (redundant or conflicting).

        ``rho`` bounds are enforced by the FactorCorrelation field itself.
        """
        if not self.factor_correlations:
            return self

        node_ids = {node.id for node in self.graph.nodes}
        # Distribution per factor that actually has a sampled uncertainty.
        dist_by_factor = {u.node_id: u.distribution for u in (self.parameter_uncertainties or [])}

        seen_pairs: set[tuple[str, str]] = set()
        for corr in self.factor_correlations:
            a, b = corr.factor_a, corr.factor_b

            # Self-pair: rejected outright, any rho (B3 P3-2). A factor is
            # trivially perfectly correlated with itself, so a self-pair expresses
            # no dependence between distinct factors — yet supplying one still
            # ACTIVATES the correlation regime (correlation_model disclosure +
            # suppression of the independence-assuming per-factor attributions).
            # That is a withhold-for-nothing, so a self-pair is never valid input.
            if a == b:
                raise ValueError(
                    "factor_correlations: self-correlation is not permitted for "
                    f"factor '{a}' (a factor is trivially correlated with itself; "
                    "declaring the pair expresses no dependence between distinct "
                    "factors)"
                )

            for factor_id in (a, b):
                if factor_id not in node_ids:
                    raise ValueError(
                        "factor_correlations references non-existent factor node: " f"{factor_id}"
                    )
                if factor_id not in dist_by_factor:
                    raise ValueError(
                        f"factor_correlations references factor '{factor_id}' which "
                        "has no parameter_uncertainty; correlation requires both "
                        "factors to carry a sampled uncertainty"
                    )
                dist = dist_by_factor[factor_id]
                if dist not in _CORRELATION_SUPPORTED_DISTRIBUTIONS:
                    # Allowlist (B3 INFO-2): only normal/uniform marginals have a
                    # copula transform. point_mass (zero variance) and any future
                    # family are rejected here rather than 500ing mid-analysis.
                    raise ValueError(
                        "factor_correlations: correlation is only supported for "
                        "factors with a normal or uniform distribution; factor "
                        f"'{factor_id}' has distribution '{dist}'"
                    )

            if a != b:
                key = (a, b) if a <= b else (b, a)
                if key in seen_pairs:
                    raise ValueError(
                        "factor_correlations contains a duplicate pair for factors "
                        f"'{key[0]}' and '{key[1]}' (each unordered pair may appear "
                        "at most once)"
                    )
                seen_pairs.add(key)

        return self

    @model_validator(mode="after")
    def validate_control_candidates(self) -> "RobustnessRequestV2":
        """Validate control_candidates (A3 S4 EVPC, D-23.8) → fail closed, typed 422.

        Rejected here (before any compute) so a malformed do()-grid never reaches
        the evaluator. Messages name the offending factor id only — never any other
        request value:

        - a candidate factor is not a graph node (unknown lever),
        - a candidate factor IS the goal node (do(goal=x) sets the outcome by fiat —
          not a lever; it would report a meaningless "value of control"),
        - a candidate factor is removed by the inference-node filter, i.e. its kind
          is in NON_INFERENCE_KINDS (decision/option/constraint). Such a node never
          reaches the evaluator, so do() would silently no-op and EVPC would clamp to
          a FALSE 0 (Codex F5, D-23.14) — reject before compute.
        - a duplicate factor_id across candidates (ambiguous value grid).

        The kind check validates against the POST-FILTER inference node set — exactly
        the nodes that survive filter_inference_graph and reach the evaluator, derived
        from the SAME NON_INFERENCE_KINDS the analyzer's filter uses so the validator
        and the filter cannot fork. A candidate that survives the filter (factor,
        chance, or a non-goal outcome mediator) computes a REAL, non-clamped EVPC and
        is ACCEPTED — the S4 design controls intermediate `chance` nodes (see
        tests/integration/test_factor_evpc_wire.py). Only the FILTERED kinds carry the
        Codex false-0 defect, so only they are rejected. (D-23.14's "kind==factor
        allowlist" phrasing is read here as its own positive-control intent — "filtered
        kinds fail before compute" / "reject if it won't reach the evaluator" — because
        a strict factor-only allowlist would regress the shipped chance-control
        capability; flagged for review as A3 Codex-fix-A decision.)

        Per-value finiteness is enforced on ControlCandidate.values; the candidate
        count (<= MAX_CONTROL_CANDIDATES) and per-candidate value count
        (<= MAX_CONTROL_VALUES) are enforced by the list max_length bounds, which
        raise a typed 422 automatically.
        """
        if not self.control_candidates:
            return self

        node_kind = {node.id: node.kind for node in self.graph.nodes}
        # Post-filter inference node set: exactly the nodes filter_inference_graph
        # keeps (kind NOT in NON_INFERENCE_KINDS), i.e. those that reach the evaluator.
        inference_node_ids = {
            nid for nid, kind in node_kind.items() if kind.lower() not in NON_INFERENCE_KINDS
        }
        seen: set[str] = set()
        for candidate in self.control_candidates:
            fid = candidate.factor_id
            if fid not in node_kind:
                raise ValueError(
                    "control_candidates references non-existent factor node: " f"{fid}"
                )
            if fid == self.goal_node_id:
                raise ValueError(
                    f"control_candidates may not target the goal node '{fid}': "
                    "do(goal=x) sets the outcome by fiat and is not a controllable "
                    "lever (its 'value of control' would be meaningless)"
                )
            if fid not in inference_node_ids:
                # kind in NON_INFERENCE_KINDS (decision/option/constraint) -> removed by
                # filter_inference_graph before the evaluator is built, so do(fid=x) never
                # reaches the compute and EVPC would clamp to a plausible FALSE 0 (Codex F5).
                raise ValueError(
                    f"control_candidates factor '{fid}' has kind '{node_kind[fid]}', which is "
                    "removed by the inference-node filter and would never reach the evaluator "
                    "(its value of control would clamp to a false 0); only nodes that survive "
                    "the inference filter are controllable levers"
                )
            if fid in seen:
                raise ValueError(
                    "control_candidates contains a duplicate factor_id "
                    f"'{fid}' (each factor may appear at most once)"
                )
            seen.add(fid)

        return self

    # CIL: explicit extra='ignore' — unknown fields are silently dropped.
    # This is a documented contract promise; do not change without cross-service coordination.
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "request_id": "req-001",
                "graph": {
                    "nodes": [
                        {"id": "price", "kind": "decision", "label": "Price"},
                        {"id": "revenue", "kind": "outcome", "label": "Revenue"},
                    ],
                    "edges": [
                        {
                            "from": "price",
                            "to": "revenue",
                            "exists_probability": 0.9,
                            "strength": {"mean": -0.5, "std": 0.15},
                        }
                    ],
                },
                "options": [
                    {"id": "low", "label": "Low price", "interventions": {"price": 0.3}},
                    {"id": "high", "label": "High price", "interventions": {"price": 0.7}},
                ],
                "goal_node_id": "revenue",
                "n_samples": 1000,
            }
        },
    }


# =============================================================================
# Response Schema
# =============================================================================


class OutcomeDistribution(BaseModel):
    """Distribution of outcomes from Monte Carlo sampling."""

    mean: float = Field(..., description="Mean outcome value")
    std: float = Field(..., description="Standard deviation")
    median: float = Field(..., description="Median outcome value")
    ci_lower: float = Field(..., description="Lower bound of confidence interval")
    ci_upper: float = Field(..., description="Upper bound of confidence interval")
    samples: Optional[List[float]] = Field(None, description="Raw samples if requested")

    model_config = {
        "json_schema_extra": {
            "example": {
                "mean": 50000.0,
                "std": 5000.0,
                "median": 49500.0,
                "ci_lower": 40000.0,
                "ci_upper": 60000.0,
            }
        }
    }


class ConstraintResult(BaseModel):
    """Internal result for a single goal constraint."""

    node_id: str = Field(..., description="Node ID the constraint applies to")
    operator: Literal[">=", "<="] = Field(..., description="Comparison operator")
    threshold: float = Field(..., description="Threshold value")
    label: Optional[str] = Field(None, description="Human-readable label for coaching")
    prob_satisfied: float = Field(
        ..., ge=0, le=1, description="Probability that this constraint is satisfied"
    )
    failure_margin_median: Optional[float] = Field(
        None, description="Median distance from threshold when constraint fails"
    )
    near_miss_fraction: Optional[float] = Field(
        None, ge=0, le=1, description="Fraction of failures within 10% of threshold"
    )
    binding: Optional[bool] = Field(
        None, description="True if constraint is borderline (prob_satisfied ∈ [0.4, 0.6])"
    )


class ConstraintAnalysis(BaseModel):
    """Internal multi-constraint analysis results for an option."""

    constraints: List[ConstraintResult] = Field(
        ..., description="Per-constraint probability results"
    )
    joint_probability: float = Field(
        ..., ge=0, le=1, description="P(all constraints satisfied simultaneously)"
    )
    conditional_probabilities: Optional[Dict[str, Dict[str, float]]] = Field(
        None, description="Pairwise conditional probabilities: P(C_j | C_i)"
    )


class OptionResult(BaseModel):
    """Results for a single decision option."""

    option_id: str = Field(..., description="Option identifier")
    outcome_distribution: OutcomeDistribution = Field(..., description="Distribution of outcomes")
    win_probability: float = Field(..., ge=0, le=1, description="P(this option is best)")
    probability_of_goal: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="P(outcome >= goal_threshold). Only present when goal_threshold is provided in request.",
    )
    constraint_analysis: Optional[ConstraintAnalysis] = Field(
        None,
        description="Multi-constraint analysis results. Only present when goal_constraints is provided.",
    )

    # B2 CRN-fix (CODE-REVIEW-ISL F1): the JOINT expected_regret, computed in the
    # analyzer from the PRE-noise CRN-aligned outcomes -- the SAME population that
    # produced win_probability (winner_per_sample). It MUST NOT be reconstructed
    # from outcome_distribution.samples, which are POST-`_apply_auto_scaled_noise`
    # (independent per-option noise that breaks CRN alignment and inflates regret).
    #
    # A REGULAR (serialized) field, NOT a PrivateAttr: the analyzer runs inside the
    # ProcessPoolExecutor worker (analysis_pool.run_offloaded), which returns the
    # response as `model_dump_json()` and the endpoint reconstructs it with
    # `model_validate_json()`. Pydantic DROPS private attrs across that
    # dump/validate boundary, so a PrivateAttr would silently become None on every
    # OFFLOADED request -> `downside` omitted in prod while present in in-process
    # tests (a test/prod divergence). A regular field survives serialization.
    # OptionResult (V1) is INTERNAL -- the client receives a separately-built
    # OptionResultV2, so this field never reaches the client wire (it appears only
    # in the internal OptionResult schema). Set in _compute_option_results; read in
    # api/robustness.py to build DownsideV2. None when regret was not computed
    # (e.g. no samples), in which case the V2 layer omits `downside`.
    pre_noise_expected_regret: Optional[float] = Field(
        default=None,
        description=(
            "Internal (V1, not client-facing): PRE-noise CRN-aligned joint "
            "expected regret, threaded analyzer -> V2 emission so the wire value "
            "matches win_probability's population. Survives the offload "
            "serialization boundary (a PrivateAttr would not)."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "option_id": "opt1",
                "outcome_distribution": {
                    "mean": 50000.0,
                    "std": 5000.0,
                    "median": 49500.0,
                    "ci_lower": 40000.0,
                    "ci_upper": 60000.0,
                },
                "win_probability": 0.65,
                "probability_of_goal": 0.72,
            }
        }
    }


class SensitivityResult(BaseModel):
    """Sensitivity to a single edge."""

    edge_from: str = Field(..., description="Source node of edge")
    edge_to: str = Field(..., description="Target node of edge")
    sensitivity_type: str = Field(..., description="Type: 'existence' or 'magnitude'")
    elasticity: float = Field(..., description="% change in outcome per % change in parameter")
    importance_rank: int = Field(..., ge=1, description="Rank by importance (1 = most important)")
    interpretation: str = Field(..., description="Human-readable explanation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "edge_from": "marketing",
                "edge_to": "demand",
                "sensitivity_type": "existence",
                "elasticity": 0.45,
                "importance_rank": 1,
                "interpretation": "Decision is moderately sensitive to marketing->demand existence",
            }
        }
    }


class FactorSensitivityResult(BaseModel):
    """Sensitivity to a factor node's value (Phase 2A Part 2)."""

    node_id: str = Field(..., description="Factor node ID")
    node_label: Optional[str] = Field(None, description="Human-readable node label")
    elasticity: float = Field(
        ..., description="% change in outcome per % change in factor value (raw, unclamped)"
    )
    elasticity_display: Optional[float] = Field(
        None, description="UI-safe elasticity clamped to [-100, 100] (debug/display only)"
    )
    importance_rank: int = Field(..., ge=1, description="Rank by importance (1 = most important)")
    observed_value: Optional[float] = Field(
        None, description="Observed value from node's observed_state"
    )
    interpretation: str = Field(..., description="Human-readable explanation")
    # Debug-only fields (not part of product contract)
    zero_reason: Optional[ZeroSensitivityReason] = Field(
        None,
        description="Debug: explains why sensitivity is zero (only present when elasticity ≈ 0)",
    )
    baseline_near_zero: Optional[bool] = Field(
        None, description="Debug: True if epsilon denominator was applied due to near-zero baseline"
    )
    # Structural influence fields (always computed from graph structure)
    influence_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Structural influence from causal path strengths (0-1, normalized)",
    )
    influence_rank: Optional[int] = Field(
        None, ge=1, description="Rank by influence_score (1 = highest influence)"
    )
    # Bootstrap uncertainty fields (3C — factor sensitivity confidence)
    elasticity_std: Optional[float] = Field(
        None,
        ge=0,
        description="Std dev of elasticity across bootstrap/jackknife runs. "
        "Measures stability of attribution under model and sampling uncertainty, "
        "NOT confidence in the causal relationship (which requires data we don't have).",
    )
    attribution_stability: Optional[Literal["high", "moderate", "low", "negligible"]] = Field(
        None,
        description="Categorical stability: 'high' (CV<0.1), 'moderate' (CV<0.3), "
        "'low' (CV>=0.3), or 'negligible' (|elasticity|<1e-6)",
    )
    rank_flip_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Fraction of bootstrap runs where this factor's importance rank "
        "shifts by >= 2 positions",
    )
    stability_method: Optional[str] = Field(
        None, description="Method used: 'bootstrap_20' or 'bootstrap_10'"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "node_id": "marketing_spend",
                "node_label": "Marketing Spend",
                "elasticity": 0.32,
                "elasticity_display": 0.32,
                "importance_rank": 2,
                "observed_value": 50000.0,
                "interpretation": "Decision is moderately sensitive to marketing_spend value",
                "influence_score": 0.85,
                "influence_rank": 1,
                "elasticity_std": 0.04,
                "attribution_stability": "high",
                "rank_flip_rate": 0.05,
                "stability_method": "bootstrap_20",
            }
        }
    }


class FragileEdgeEnhanced(BaseModel):
    """Enhanced fragile edge data with alternative winner analysis.

    Used internally by the analyzer. Maps 1:1 with FragileEdgeV2 in response_v2.py
    for API responses.
    """

    edge_id: str = Field(..., description="Edge identifier in 'from->to' format")
    from_id: str = Field(..., description="Source node ID")
    to_id: str = Field(..., description="Target node ID")
    alternative_winner_id: Optional[str] = Field(
        None,
        description="Option that wins most often when this edge is weak (bottom quartile). "
        "Null if same option wins regardless of edge strength.",
    )
    switch_probability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Probability of alternative winner in weak-edge scenarios. "
        "0.0 if same option wins (stable), null only if no data available.",
    )
    marginal_switch_probability: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Probability of decision flip when ONLY this edge varies "
        "(all other edges held at baseline). Isolates individual edge contribution.",
    )


class RobustnessResult(BaseModel):
    """Overall robustness assessment."""

    is_robust: bool = Field(..., description="Whether recommendation is robust")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in robustness assessment")
    fragile_edges: List[str] = Field(
        default_factory=list, description="Edges that could flip the decision (format: 'from->to')"
    )
    fragile_edges_enhanced: Optional[List[FragileEdgeEnhanced]] = Field(
        default=None, description="Enhanced fragile edge data with alternative winner analysis"
    )
    robust_edges: List[str] = Field(
        default_factory=list, description="Edges that don't significantly affect decision"
    )
    recommendation_stability: float = Field(
        ..., ge=0, le=1, description="P(same recommendation across samples)"
    )
    interpretation: str = Field(..., description="Human-readable robustness summary")
    # Trust penalty metadata (auditable when root nodes defaulted to 0.0)
    stability_penalty_factor: Optional[float] = Field(
        None,
        description="Multiplicative penalty applied to recommendation_stability "
        "due to missing root node values. 1.0 = no penalty.",
    )
    defaulted_root_node_ids: Optional[List[str]] = Field(
        None,
        description="Root node IDs that defaulted to 0.0 (trigger for stability penalty).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "is_robust": True,
                "confidence": 0.92,
                "fragile_edges": ["marketing->demand"],
                "robust_edges": ["price->revenue"],
                "recommendation_stability": 0.88,
                "interpretation": "Recommendation is robust with 92% confidence",
            }
        }
    }


class ClampMetrics(BaseModel):
    """Tracks out-of-bounds sampling for diagnostics."""

    total_node_samples: int = Field(..., description="Total node value samples")
    clamped_samples: int = Field(..., description="Samples that were clamped to bounds")
    clamp_rate: float = Field(..., ge=0, le=1, description="Fraction of samples clamped")
    nodes_with_high_clamp_rate: List[str] = Field(
        default_factory=list, description="Nodes with >10% clamp rate"
    )


class ResponseMetadataV2(BaseModel):
    """Execution metadata for v2 responses."""

    schema_version: str = Field(default="2.2", description="Schema version")
    isl_version: str = Field(..., description="ISL service version")
    n_samples_used: int = Field(..., description="Actual samples used")
    seed_used: int = Field(..., description="Random seed used")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    edge_existence_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Actual sampling rates per edge (format: 'from->to': rate)",
    )
    clamp_metrics: Optional[ClampMetrics] = Field(
        None, description="Out-of-bounds sampling metrics"
    )
    config_fingerprint: str = Field(..., description="Hash of determinism-critical config")
    tie_count: Optional[int] = Field(
        None, description="Number of Monte Carlo samples with tied outcomes"
    )
    tie_rate: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Fraction of samples with tied outcomes (tie_count / n_samples)",
    )
    seed_hash_version: int = Field(
        default=2,
        description="Version of the seed hash algorithm used. "
        "V1 omits edge_type; V2 includes it.",
    )
    auto_noise_applied: bool = Field(
        default=False,
        description="Whether auto-scaled noise (√2 variance inflation) was applied to "
        "outcome distributions. Only applies to outcome/risk goal nodes. "
        "When true, p10/p90 spreads are ~√2 wider than the purely "
        "model-driven distribution.",
    )
    n_defaulted_root_nodes: Optional[int] = Field(
        None,
        description="Number of root nodes that defaulted to 0.0 due to missing "
        "observed_state.value. Non-zero indicates missing model inputs.",
    )


class BucketResult(BaseModel):
    """Win probability results for one side of a factor median split."""

    n_samples: int = Field(..., ge=0, description="Number of MC samples in this bucket")
    winner_id: str = Field(..., description="Option ID with highest win rate in this bucket")
    winner_label: str = Field(..., description="Human-readable label of the winning option")
    winner_probability: float = Field(
        ..., ge=0, le=1, description="Win probability of the bucket winner"
    )
    runner_up_id: Optional[str] = Field(None, description="Second-place option ID")
    runner_up_probability: Optional[float] = Field(
        None, ge=0, le=1, description="Win probability of runner-up"
    )


class ConditionalWinner(BaseModel):
    """
    Conditional win probability analysis for a single factor.

    Splits MC samples at the factor's median value and computes win probabilities
    in each half. When the winner differs between halves, the decision is sensitive
    to this factor's value range.

    Limitations:
    - Median split is simplistic. Does not detect non-monotonic effects or
      factor interactions.
    - Flips at extreme quantiles (top/bottom 10%) may be missed by a 50/50 split.
    """

    factor_id: str = Field(..., description="Node ID of the factor")
    factor_label: str = Field(..., description="Human-readable label")
    split_value: float = Field(..., description="Median factor value used as split point")
    split_unit: Optional[str] = Field(None, description="Unit from observed_state if available")
    low_bucket: BucketResult = Field(..., description="Results for samples below median")
    high_bucket: BucketResult = Field(..., description="Results for samples at/above median")
    winner_flips: bool = Field(
        ..., description="True if winner differs between low and high buckets"
    )


class PathContribution(BaseModel):
    """
    One modelled pathway's signed structural contribution to the goal.

    A pathway is a simple directed sequence of node IDs from a retained intervention
    target to the goal.  ``path_effect`` is the signed product of per-edge
    coefficients (``strength.mean * exists_probability``) along the pathway — the
    same per-edge semantics ISL uses for structural influence — and is NOT scaled
    by the intervention magnitude.  This is a structural decomposition of the
    modelled effect, not a real-world causal claim.
    """

    path: List[str] = Field(
        ...,
        description="Node IDs from the retained intervention target to the goal, in directed path order.",
    )
    path_effect: float = Field(
        ...,
        description="Signed product of per-edge coefficients (strength.mean * exists_probability) "
        "along this path. Structural only — not scaled by the intervention magnitude.",
    )
    total_effect: float = Field(
        ...,
        description="Signed sum of path_effect across all enumerated intervention-target-to-goal "
        "paths. Identical on every entry, for auditability.",
    )
    signed_contribution: Optional[float] = Field(
        None,
        description="path_effect / total_effect when the net modelled effect is non-negligible; "
        "omitted when indeterminate. May be negative or exceed 1 when paths oppose.",
    )
    status: Literal["computed", "indeterminate"] = Field(
        ...,
        description="'computed' when |total_effect| >= 1e-10; 'indeterminate' when the net "
        "modelled effect is near zero and a relative share is not well defined.",
    )
    mechanism: str = Field(
        ...,
        description="Human-readable modelled-pathway-contribution statement. Describes modelled "
        "structure only; not a real-world causal claim.",
    )

    model_config = {"extra": "ignore"}


class PathDecomposition(BaseModel):
    """
    Structural pathway decomposition for the recommended option's retained
    intervention targets.

    The recommended option is carried as context/metadata via
    ``recommended_option_id``; it is never itself a path node.  Computed paths
    start at the retained intervention target/factor nodes (those that survived
    inference-graph filtering).  This decomposes the modelled structural effect,
    not the option-level causal effect size, and is not weighted by intervention
    magnitude.
    """

    recommended_option_id: str = Field(
        ...,
        description="The recommended option this decomposition explains (context/metadata; "
        "not a path node).",
    )
    entry_nodes: List[str] = Field(
        ...,
        description="Retained intervention target node IDs the paths start from "
        "(intervention targets that survived inference-graph filtering).",
    )
    truncated: bool = Field(
        default=False,
        description="True when the number of simple paths exceeded the safety budget, so the "
        "top-3 pathway ranking was suppressed for performance and paths is empty. This does "
        "NOT mean the modelled effect is zero — only that individual pathways were too "
        "numerous to rank. Distinct from an empty result with truncated=False, which means no "
        "reachable path from the retained intervention targets.",
    )
    path_count: int = Field(
        default=0,
        description="Number of simple intervention-target-to-goal paths enumerated. When "
        "truncated is True this equals the budget cap and the true count is higher.",
    )
    paths: List[PathContribution] = Field(
        default_factory=list,
        description="Top-3 intervention-target-to-goal paths, ranked by absolute path_effect.",
    )

    model_config = {"extra": "ignore"}


class RobustnessResponseV2(BaseModel):
    """V2.2 robustness analysis response."""

    request_id: str = Field(..., description="Request identifier for tracing")

    # Core results
    results: List[OptionResult] = Field(..., description="Results for each decision option")
    recommended_option_id: str = Field(..., description="ID of recommended option")
    recommendation_confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in recommendation"
    )

    # Sensitivity analysis
    sensitivity: List[SensitivityResult] = Field(
        default_factory=list, description="Sensitivity results per edge"
    )

    # Factor sensitivity analysis (Phase 2A Part 2)
    factor_sensitivity: List[FactorSensitivityResult] = Field(
        default_factory=list, description="Sensitivity results per factor node value"
    )

    # Robustness analysis
    robustness: RobustnessResult = Field(..., description="Overall robustness assessment")

    # Metadata
    metadata: ResponseMetadataV2 = Field(..., description="Execution metadata", alias="_metadata")

    # Analysis critiques (warnings about degenerate options, high tie rates, etc.)
    critiques: List[CritiqueV2] = Field(
        default_factory=list, description="Analysis critiques and warnings"
    )

    # Inference warnings (e.g. constraint nodes defaulting to base=0.0).
    # Contract: always present as a list — [] when empty, never absent.
    # This field survives exclude_none=True because it has a non-None default.
    inference_warnings: List[InferenceWarning] = Field(
        default_factory=list,
        description="Structured warnings about inference conditions that may affect result reliability",
    )

    # Conditional winners (factor-partitioned win probabilities)
    conditional_winners: Optional[List[ConditionalWinner]] = Field(
        None,
        description="Factors where the winning option flips depending on factor value range. "
        "Only present when at least one factor causes a winner flip.",
    )

    # Stability threshold metadata (3C-thresholds)
    stability_thresholds: Optional[StabilityThresholdsResponse] = Field(
        None,
        description="Thresholds used for attribution_stability classification. "
        "Provisional — pending scientific review. NOT included in response_hash.",
    )

    # E-value results (enhancement — optional, gated by budget and include_e_values flag)
    edge_e_values: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="E-value analogue per edge: minimum strength perturbation to flip "
        "recommendation. Only included when computed within time budget.",
    )

    # Per-factor win-probability sensitivity (enhancement — optional, gated by
    # include_voi). S2 (D-23.8) HONEST RELABEL: this was ``factor_evpi``, but it is
    # NOT value-of-information — see the wire description on ISLResponseV2.
    p_win_sensitivity: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Per-factor win-probability sensitivity: how much the recommended "
        "option's win probability (or P(joint_goal)) moves when this factor is fixed "
        "at its mean, with the decision held FIXED. A win-probability delta in "
        "PROBABILITY units — NOT value-of-information (it cannot capture option-"
        "switching). Method-tagged 'p_win_delta_at_mean_v1'. For decision value use "
        "decision_evpi (whole-decision EVPI) and factor_evppi (per-factor EVPPI), "
        "both in outcome units.",
    )

    # Per-factor EVPPI (S2 — A3 VOI, D-23.8). Regression EVPPI on the retained
    # joint CRN samples (no new sampling); see ISLResponseV2.factor_evppi.
    # NOTE (C3 scope): the V2 WIRE model ISLResponseV2.factor_evppi is typed as
    # List[FactorEvppiEntryV2] (fail-loud at the consumer boundary + openapi). This
    # analyzer/V1 model keeps Dict[str, Any] deliberately: its rows are built and
    # consumed internally as dicts, and typing it would force broad churn on
    # analyzer-level tests for no consumer benefit. Wire serialization is identical.
    factor_evppi: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Per-factor Expected Value of Partial Perfect Information "
        "(EVPPI) in OUTCOME units, via single-loop Strong-Oakley regression on the "
        "retained joint CRN samples. Method 'regression_evppi_v1'. Option-controlled "
        "levers are OMITTED (absent, not zero). Emitted under active correlation. "
        "See ISLResponseV2.factor_evppi.",
    )

    # Per-lever EVPC (S4 — A3 value-of-control, D-23.8). Grid do() on the retained
    # joint CRN samples (no new sampling); see ISLResponseV2.factor_evpc.
    factor_evpc: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Per-lever Expected Value of Control (EVPC) in OUTCOME units, "
        "via grid do(factor=value) on the retained joint CRN samples. Method "
        "'grid_do_v1'. Present only when control_candidates was supplied. See "
        "ISLResponseV2.factor_evpc.",
    )

    # Path decomposition (enhancement — optional, gated by include_path_decomposition flag)
    path_decomposition: Optional[PathDecomposition] = Field(
        None,
        description="Structural pathway decomposition for the recommended option's retained "
        "intervention targets: top-3 simple directed paths to the goal with signed structural "
        "contributions. Structural decomposition of the modelled effect, not a causal claim "
        "about realised outcomes. Gated by include_path_decomposition.",
    )

    # Correlated-factors disclosure (B3-S1). Present only when the request
    # supplied factor_correlations (Gaussian copula active). Carries the tail-
    # independence caveat, any Higham PSD projection, and the suppressed-
    # attribution manifest through to the wire.
    correlation_model: Optional[CorrelationModelV2] = Field(
        None,
        description="Disclosure of the active factor-correlation (Gaussian copula) model. "
        "Absent when correlation is inactive (independent-factor default).",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "request_id": "req-001",
                "results": [
                    {
                        "option_id": "opt1",
                        "outcome_distribution": {
                            "mean": 50000.0,
                            "std": 5000.0,
                            "median": 49500.0,
                            "ci_lower": 40000.0,
                            "ci_upper": 60000.0,
                        },
                        "win_probability": 0.65,
                    }
                ],
                "recommended_option_id": "opt1",
                "recommendation_confidence": 0.65,
                "sensitivity": [],
                "robustness": {
                    "is_robust": True,
                    "confidence": 0.92,
                    "fragile_edges": [],
                    "robust_edges": [],
                    "recommendation_stability": 0.88,
                    "interpretation": "Robust recommendation",
                },
                "_metadata": {
                    "schema_version": "2.2",
                    "isl_version": "1.0.0",
                    "n_samples_used": 1000,
                    "seed_used": 12345,
                    "execution_time_ms": 150,
                    "edge_existence_rates": {},
                    "config_fingerprint": "abc123",
                    "tie_count": 0,
                    "tie_rate": 0.0,
                },
                "critiques": [],
                "inference_warnings": [],
            }
        },
    }


# =============================================================================
# Schema Detection
# =============================================================================


def detect_schema_version(request: Dict[str, Any]) -> str:
    """
    Detect request schema version from request structure.

    Args:
        request: Raw request dictionary

    Returns:
        "v2" for v2.2 schema, "v1" for legacy schema

    Raises:
        ValueError: If schema cannot be determined
    """
    if "graph" in request and "options" in request:
        return "v2"
    elif "causal_model" in request:
        return "v1"
    else:
        raise ValueError(
            "Unknown request schema - must contain 'graph'+'options' (v2) " "or 'causal_model' (v1)"
        )
