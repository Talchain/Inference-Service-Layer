"""
Unit tests for ISL V2 Response Enhancement components.

Tests cover:
- Path validation (BFS algorithm)
- Request validation pipeline
- Degenerate outcome detection
- Response builder
- Critique definitions
"""

import pytest

from src.constants import (
    DEFAULT_EXISTS_PROBABILITY_THRESHOLD,
    DEFAULT_STRENGTH_THRESHOLD,
    DEGENERATE_RELATIVE_THRESHOLD,
    IDENTICAL_OPTIONS_VALUE_TOLERANCE,
    MIN_VALID_RATIO,
)
from src.models.critique import (
    DEGENERATE_OUTCOMES,
    EMPTY_INTERVENTIONS,
    IDENTICAL_OPTIONS,
    INTERNAL_ERROR,
    INVALID_INTERVENTION_TARGET,
    MISSING_GOAL_NODE,
    NO_EFFECTIVE_PATH_TO_GOAL,
    NO_OPTIONS,
    CritiqueDefinition,
    get_critique,
)
from src.models.response_v2 import (
    CritiqueV2,
    DiagnosticsV2,
    FactorSensitivityV2,
    ISLResponseV2,
    OptionDiagnosticV2,
    OptionResultV2,
    OutcomeDistributionV2,
    RequestEchoV2,
    RobustnessResultV2,
)
from src.utils.response_builder import (
    ResponseBuilder,
    build_request_echo,
    determine_option_status,
    hash_node_id,
)
from src.validation.degenerate_detector import detect_degenerate_outcomes
from src.validation.path_validator import PathValidationConfig, PathValidator
from src.validation.request_validator import (
    RequestValidator,
    ValidationResult,
    detect_graph_cycle,
    detect_identical_options,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test threshold constants."""

    def test_threshold_values(self):
        """Test default threshold values are sensible."""
        assert DEFAULT_EXISTS_PROBABILITY_THRESHOLD == 1e-6
        assert DEFAULT_STRENGTH_THRESHOLD == 1e-6
        assert IDENTICAL_OPTIONS_VALUE_TOLERANCE == 1e-9
        assert DEGENERATE_RELATIVE_THRESHOLD == 0.01
        assert MIN_VALID_RATIO == 0.8


# =============================================================================
# Critique Tests
# =============================================================================


class TestCritiqueDefinition:
    """Test critique definition building."""

    def test_build_basic_critique(self):
        """Test building a basic critique."""
        critique = MISSING_GOAL_NODE.build()

        assert critique.code == "MISSING_GOAL_NODE"
        assert critique.severity == "blocker"
        assert critique.source == "validation"
        assert "Goal node not found" in critique.message
        assert critique.id.startswith("critique_")
        assert critique.suggestion is not None

    def test_build_critique_with_template_vars(self):
        """Test building critique with template variables."""
        critique = EMPTY_INTERVENTIONS.build(label="Option A")

        assert "Option A" in critique.message
        assert critique.code == "EMPTY_INTERVENTIONS"

    def test_build_critique_with_affected_ids(self):
        """Test building critique with affected IDs."""
        critique = INVALID_INTERVENTION_TARGET.build(
            label="Option B",
            affected_option_ids=["opt_1", "opt_2"],
            affected_node_ids=["node_x"],
        )

        assert critique.affected_option_ids == ["opt_1", "opt_2"]
        assert critique.affected_node_ids == ["node_x"]

    def test_build_critique_with_custom_suggestion(self):
        """Test overriding default suggestion."""
        custom = "Check your graph structure"
        critique = MISSING_GOAL_NODE.build(suggestion=custom)

        assert critique.suggestion == custom

    def test_get_critique_by_code(self):
        """Test retrieving critique by code."""
        critique = get_critique("INTERNAL_ERROR")

        assert critique == INTERNAL_ERROR
        assert critique.severity == "blocker"
        assert critique.source == "engine"

    def test_get_critique_unknown_code(self):
        """Test error for unknown critique code."""
        with pytest.raises(ValueError, match="Unknown critique code"):
            get_critique("UNKNOWN_CODE")

    def test_critique_sources(self):
        """Test critique source classification."""
        assert MISSING_GOAL_NODE.source == "validation"
        assert NO_OPTIONS.source == "validation"
        assert DEGENERATE_OUTCOMES.source == "analysis"
        assert INTERNAL_ERROR.source == "engine"


# =============================================================================
# Path Validator Tests
# =============================================================================


class TestPathValidator:
    """Test BFS path validation."""

    @pytest.fixture
    def simple_nodes(self):
        """Simple linear graph nodes: A -> B -> C."""
        return [
            {"id": "A", "kind": "controllable"},
            {"id": "B", "kind": "intermediate"},
            {"id": "C", "kind": "outcome"},
        ]

    @pytest.fixture
    def simple_edges(self):
        """Simple linear graph edges: A -> B -> C."""
        return [
            {
                "source_id": "A",
                "target_id": "B",
                "exists_probability": 0.9,
                "strength": {"mean": 0.5, "std": 0.1},
            },
            {
                "source_id": "B",
                "target_id": "C",
                "exists_probability": 0.9,
                "strength": {"mean": 0.5, "std": 0.1},
            },
        ]

    @pytest.fixture
    def disconnected_nodes(self):
        """Graph with disconnected node: A -> B, C (isolated)."""
        return [
            {"id": "A", "kind": "controllable"},
            {"id": "B", "kind": "outcome"},
            {"id": "C", "kind": "intermediate"},
        ]

    @pytest.fixture
    def disconnected_edges(self):
        """Edges for disconnected graph."""
        return [
            {
                "source_id": "A",
                "target_id": "B",
                "exists_probability": 0.9,
                "strength": {"mean": 0.5, "std": 0.1},
            },
        ]

    @pytest.fixture
    def weak_edge_nodes(self):
        """Graph with weak edge: A -> B (weak) -> C."""
        return [
            {"id": "A", "kind": "controllable"},
            {"id": "B", "kind": "intermediate"},
            {"id": "C", "kind": "outcome"},
        ]

    @pytest.fixture
    def weak_edge_edges(self):
        """Edges for weak edge graph."""
        return [
            {
                "source_id": "A",
                "target_id": "B",
                "exists_probability": 0.9,
                "strength": {"mean": 0.5, "std": 0.1},
            },
            {
                "source_id": "B",
                "target_id": "C",
                "exists_probability": 0.9,
                "strength": {"mean": 0.0, "std": 0.0},  # Zero strength
            },
        ]

    @pytest.fixture
    def low_probability_nodes(self):
        """Graph with low probability edge."""
        return [
            {"id": "A", "kind": "controllable"},
            {"id": "B", "kind": "outcome"},
        ]

    @pytest.fixture
    def low_probability_edges(self):
        """Edges for low probability graph."""
        return [
            {
                "source_id": "A",
                "target_id": "B",
                "exists_probability": 1e-9,  # Below threshold
                "strength": {"mean": 0.5, "std": 0.1},
            },
        ]

    def test_has_structural_path_simple(self, simple_nodes, simple_edges):
        """Test structural path detection in simple graph."""
        config = PathValidationConfig()
        validator = PathValidator(simple_nodes, simple_edges, config)

        assert validator.has_structural_path("A", "C") is True
        assert validator.has_structural_path("B", "C") is True
        assert validator.has_structural_path("C", "C") is True  # Self

    def test_has_structural_path_disconnected(
        self, disconnected_nodes, disconnected_edges
    ):
        """Test no path from disconnected node."""
        config = PathValidationConfig()
        validator = PathValidator(disconnected_nodes, disconnected_edges, config)

        assert validator.has_structural_path("A", "B") is True
        assert validator.has_structural_path("C", "B") is False  # Isolated

    def test_has_structural_path_low_probability(
        self, low_probability_nodes, low_probability_edges
    ):
        """Test low probability edge excluded from structural path."""
        config = PathValidationConfig(exists_probability_threshold=1e-6)
        validator = PathValidator(
            low_probability_nodes, low_probability_edges, config
        )

        assert validator.has_structural_path("A", "B") is False

    def test_has_effective_path_simple(self, simple_nodes, simple_edges):
        """Test effective path detection in simple graph."""
        config = PathValidationConfig()
        validator = PathValidator(simple_nodes, simple_edges, config)

        assert validator.has_effective_path("A", "C") is True
        assert validator.has_effective_path("B", "C") is True

    def test_has_effective_path_weak_edge(self, weak_edge_nodes, weak_edge_edges):
        """Test no effective path through zero-strength edge."""
        config = PathValidationConfig(strength_threshold=1e-6)
        validator = PathValidator(weak_edge_nodes, weak_edge_edges, config)

        # Structural path exists
        assert validator.has_structural_path("A", "C") is True
        # But effective path doesn't (zero strength on B->C)
        assert validator.has_effective_path("A", "C") is False

    def test_validate_option(self, simple_nodes, simple_edges):
        """Test validating option with effective paths."""
        config = PathValidationConfig()
        validator = PathValidator(simple_nodes, simple_edges, config)

        diag = validator.validate_option(
            option_id="opt1",
            intervention_targets=["A", "B"],
            goal_id="C",
        )

        assert diag.has_effective_path is True
        assert diag.targets_with_effective_path_count == 2
        assert diag.targets_without_effective_path_count == 0

    def test_validate_option_with_missing_target(self, simple_nodes, simple_edges):
        """Test validating option with nonexistent target."""
        config = PathValidationConfig()
        validator = PathValidator(simple_nodes, simple_edges, config)

        diag = validator.validate_option(
            option_id="opt1",
            intervention_targets=["A", "nonexistent"],
            goal_id="C",
        )

        assert diag.has_effective_path is True  # A has path
        assert diag.targets_with_effective_path_count == 1
        assert diag.targets_without_effective_path_count == 1
        assert len(diag.warnings) == 1

    def test_self_path(self, simple_nodes, simple_edges):
        """Test path to self is valid."""
        config = PathValidationConfig()
        validator = PathValidator(simple_nodes, simple_edges, config)

        assert validator.has_structural_path("A", "A") is True
        assert validator.has_effective_path("A", "A") is True


# =============================================================================
# Request Validator Tests
# =============================================================================


class TestRequestValidator:
    """Test request validation pipeline."""

    @pytest.fixture
    def valid_request(self):
        """Valid request with path to goal."""
        return {
            "graph": {
                "nodes": [
                    {"id": "price", "kind": "controllable"},
                    {"id": "revenue", "kind": "outcome"},
                ],
                "edges": [
                    {
                        "source_id": "price",
                        "target_id": "revenue",
                        "exists_probability": 0.9,
                        "strength": {"mean": -0.5, "std": 0.1},
                    },
                ],
            },
            "options": [
                {
                    "id": "low_price",
                    "label": "Low Price",
                    "interventions": {"price": 50},
                },
                {
                    "id": "high_price",
                    "label": "High Price",
                    "interventions": {"price": 100},
                },
            ],
            "goal_node_id": "revenue",
        }

    @pytest.fixture
    def request_missing_goal(self):
        """Request with missing goal node."""
        return {
            "graph": {
                "nodes": [{"id": "price", "kind": "controllable"}],
                "edges": [],
            },
            "options": [
                {"id": "opt1", "label": "Option 1", "interventions": {"price": 50}},
            ],
            "goal_node_id": "nonexistent",
        }

    @pytest.fixture
    def request_no_options(self):
        """Request with no options."""
        return {
            "graph": {
                "nodes": [{"id": "revenue", "kind": "outcome"}],
                "edges": [],
            },
            "options": [],
            "goal_node_id": "revenue",
        }

    @pytest.fixture
    def request_empty_interventions(self):
        """Request with empty interventions."""
        return {
            "graph": {
                "nodes": [{"id": "revenue", "kind": "outcome"}],
                "edges": [],
            },
            "options": [
                {"id": "opt1", "label": "Empty Option", "interventions": {}},
            ],
            "goal_node_id": "revenue",
        }

    @pytest.fixture
    def request_invalid_target(self):
        """Request with invalid intervention target."""
        return {
            "graph": {
                "nodes": [{"id": "revenue", "kind": "outcome"}],
                "edges": [],
            },
            "options": [
                {
                    "id": "opt1",
                    "label": "Bad Target",
                    "interventions": {"nonexistent": 50},
                },
            ],
            "goal_node_id": "revenue",
        }

    @pytest.fixture
    def request_no_path(self):
        """Request with no path to goal."""
        return {
            "graph": {
                "nodes": [
                    {"id": "price", "kind": "controllable"},
                    {"id": "revenue", "kind": "outcome"},
                ],
                "edges": [],  # No edges = no path
            },
            "options": [
                {"id": "opt1", "label": "No Path", "interventions": {"price": 50}},
            ],
            "goal_node_id": "revenue",
        }

    @pytest.fixture
    def request_identical_options(self):
        """Request with identical options."""
        return {
            "graph": {
                "nodes": [
                    {"id": "price", "kind": "controllable"},
                    {"id": "revenue", "kind": "outcome"},
                ],
                "edges": [
                    {
                        "source_id": "price",
                        "target_id": "revenue",
                        "exists_probability": 0.9,
                        "strength": {"mean": 0.5, "std": 0.1},
                    },
                ],
            },
            "options": [
                {"id": "opt1", "label": "Option 1", "interventions": {"price": 50}},
                {"id": "opt2", "label": "Option 2", "interventions": {"price": 50}},
            ],
            "goal_node_id": "revenue",
        }

    def test_valid_request(self, valid_request):
        """Test validation passes for valid request."""
        config = PathValidationConfig()
        validator = RequestValidator(
            graph=valid_request["graph"],
            options=valid_request["options"],
            goal_node_id=valid_request["goal_node_id"],
            path_config=config,
        )
        result = validator.validate()

        assert result.is_valid is True
        assert result.has_blockers is False
        assert len(result.critiques) == 0

    def test_missing_goal_node(self, request_missing_goal):
        """Test blocker for missing goal node."""
        config = PathValidationConfig()
        validator = RequestValidator(
            graph=request_missing_goal["graph"],
            options=request_missing_goal["options"],
            goal_node_id=request_missing_goal["goal_node_id"],
            path_config=config,
        )
        result = validator.validate()

        assert result.is_valid is False
        assert result.has_blockers is True
        assert any(c.code == "MISSING_GOAL_NODE" for c in result.critiques)

    def test_no_options(self, request_no_options):
        """Test blocker for no options."""
        config = PathValidationConfig()
        validator = RequestValidator(
            graph=request_no_options["graph"],
            options=request_no_options["options"],
            goal_node_id=request_no_options["goal_node_id"],
            path_config=config,
        )
        result = validator.validate()

        assert result.is_valid is False
        assert result.has_blockers is True
        assert any(c.code == "NO_OPTIONS" for c in result.critiques)

    def test_empty_interventions(self, request_empty_interventions):
        """Test blocker for empty interventions."""
        config = PathValidationConfig()
        validator = RequestValidator(
            graph=request_empty_interventions["graph"],
            options=request_empty_interventions["options"],
            goal_node_id=request_empty_interventions["goal_node_id"],
            path_config=config,
        )
        result = validator.validate()

        assert result.is_valid is False
        assert result.has_blockers is True
        assert any(c.code == "EMPTY_INTERVENTIONS" for c in result.critiques)

    def test_invalid_intervention_target(self, request_invalid_target):
        """Test blocker for invalid intervention target."""
        config = PathValidationConfig()
        validator = RequestValidator(
            graph=request_invalid_target["graph"],
            options=request_invalid_target["options"],
            goal_node_id=request_invalid_target["goal_node_id"],
            path_config=config,
        )
        result = validator.validate()

        assert result.is_valid is False
        assert result.has_blockers is True
        assert any(c.code == "INVALID_INTERVENTION_TARGET" for c in result.critiques)

    def test_no_effective_path(self, request_no_path):
        """Test blocker for no effective path to goal."""
        config = PathValidationConfig()
        validator = RequestValidator(
            graph=request_no_path["graph"],
            options=request_no_path["options"],
            goal_node_id=request_no_path["goal_node_id"],
            path_config=config,
        )
        result = validator.validate()

        assert result.is_valid is False
        assert result.has_blockers is True
        assert any(c.code == "NO_EFFECTIVE_PATH_TO_GOAL" for c in result.critiques)

    def test_identical_options(self, request_identical_options):
        """Test blocker for identical options."""
        config = PathValidationConfig()
        validator = RequestValidator(
            graph=request_identical_options["graph"],
            options=request_identical_options["options"],
            goal_node_id=request_identical_options["goal_node_id"],
            path_config=config,
        )
        result = validator.validate()

        assert result.is_valid is False
        assert result.has_blockers is True
        assert any(c.code == "IDENTICAL_OPTIONS" for c in result.critiques)

    def test_option_diagnostics_generated(self, valid_request):
        """Test option diagnostics are generated."""
        config = PathValidationConfig()
        validator = RequestValidator(
            graph=valid_request["graph"],
            options=valid_request["options"],
            goal_node_id=valid_request["goal_node_id"],
            path_config=config,
        )
        result = validator.validate()

        assert len(result.option_diagnostics) == 2
        assert all(d.has_effective_path for d in result.option_diagnostics)

    def test_cycle_detection(self):
        """Test blocker for cyclic graph."""
        cyclic_request = {
            "graph": {
                "nodes": [
                    {"id": "A", "kind": "controllable"},
                    {"id": "B", "kind": "intermediate"},
                    {"id": "C", "kind": "outcome"},
                ],
                "edges": [
                    {
                        "source_id": "A",
                        "target_id": "B",
                        "exists_probability": 0.9,
                        "strength": {"mean": 0.5, "std": 0.1},
                    },
                    {
                        "source_id": "B",
                        "target_id": "C",
                        "exists_probability": 0.9,
                        "strength": {"mean": 0.5, "std": 0.1},
                    },
                    {
                        "source_id": "C",
                        "target_id": "A",  # Creates cycle: A -> B -> C -> A
                        "exists_probability": 0.9,
                        "strength": {"mean": 0.5, "std": 0.1},
                    },
                ],
            },
            "options": [
                {"id": "opt1", "label": "Option 1", "interventions": {"A": 50}},
            ],
            "goal_node_id": "C",
        }
        config = PathValidationConfig()
        validator = RequestValidator(
            graph=cyclic_request["graph"],
            options=cyclic_request["options"],
            goal_node_id=cyclic_request["goal_node_id"],
            path_config=config,
        )
        result = validator.validate()

        assert result.is_valid is False
        assert result.has_blockers is True
        assert any(c.code == "GRAPH_CYCLE_DETECTED" for c in result.critiques)


class TestDetectGraphCycle:
    """Test graph cycle detection."""

    def test_acyclic_graph(self):
        """Test no cycle detected in DAG."""
        edges = [
            {"source_id": "A", "target_id": "B"},
            {"source_id": "B", "target_id": "C"},
            {"source_id": "A", "target_id": "C"},
        ]
        assert detect_graph_cycle(edges) is False

    def test_simple_cycle(self):
        """Test cycle detection for simple cycle."""
        edges = [
            {"source_id": "A", "target_id": "B"},
            {"source_id": "B", "target_id": "A"},  # Cycle
        ]
        assert detect_graph_cycle(edges) is True

    def test_self_loop(self):
        """Test self-loop detection."""
        edges = [
            {"source_id": "A", "target_id": "A"},  # Self-loop
        ]
        assert detect_graph_cycle(edges) is True

    def test_longer_cycle(self):
        """Test detection of longer cycle."""
        edges = [
            {"source_id": "A", "target_id": "B"},
            {"source_id": "B", "target_id": "C"},
            {"source_id": "C", "target_id": "D"},
            {"source_id": "D", "target_id": "B"},  # Cycle: B -> C -> D -> B
        ]
        assert detect_graph_cycle(edges) is True

    def test_empty_graph(self):
        """Test empty graph has no cycle."""
        assert detect_graph_cycle([]) is False


class TestDetectIdenticalOptions:
    """Test identical options detection."""

    def test_different_options(self):
        """Test no detection for different options."""
        options = [
            {"id": "a", "label": "A", "interventions": {"x": 1}},
            {"id": "b", "label": "B", "interventions": {"x": 2}},
        ]
        result = detect_identical_options(options)
        assert result is None

    def test_identical_single_intervention(self):
        """Test detection of identical single intervention."""
        options = [
            {"id": "a", "label": "A", "interventions": {"x": 1}},
            {"id": "b", "label": "B", "interventions": {"x": 1}},
        ]
        result = detect_identical_options(options)
        assert result is not None
        # Returns tuple of (label_a, label_b)
        assert result == ("A", "B")

    def test_identical_multiple_interventions(self):
        """Test detection of identical multiple interventions."""
        options = [
            {"id": "a", "label": "A", "interventions": {"x": 1, "y": 2}},
            {"id": "b", "label": "B", "interventions": {"x": 1, "y": 2}},
        ]
        result = detect_identical_options(options)
        assert result is not None

    def test_identical_within_tolerance(self):
        """Test identical detection within tolerance."""
        options = [
            {"id": "a", "label": "A", "interventions": {"x": 1.0}},
            {"id": "b", "label": "B", "interventions": {"x": 1.0 + 1e-12}},
        ]
        result = detect_identical_options(options)
        assert result is not None


# =============================================================================
# Degenerate Detector Tests
# =============================================================================


class TestDegenerateDetector:
    """Test degenerate outcome detection."""

    def test_distinct_outcomes(self):
        """Test no detection for distinct outcomes."""
        options = [
            OptionResultV2(
                id="a",
                outcome=OutcomeDistributionV2(
                    mean=100, std=10, p10=85, p50=100, p90=115,
                    n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                ),
                status="computed",
            ),
            OptionResultV2(
                id="b",
                outcome=OutcomeDistributionV2(
                    mean=200, std=20, p10=170, p50=200, p90=230,
                    n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                ),
                status="computed",
            ),
        ]
        critique = detect_degenerate_outcomes(options)
        assert critique is None

    def test_degenerate_outcomes(self):
        """Test detection of degenerate outcomes."""
        # Means within 1% relative spread
        options = [
            OptionResultV2(
                id="a",
                outcome=OutcomeDistributionV2(
                    mean=100.0, std=1, p10=99, p50=100, p90=101,
                    n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                ),
                status="computed",
            ),
            OptionResultV2(
                id="b",
                outcome=OutcomeDistributionV2(
                    mean=100.5, std=1, p10=99.5, p50=100.5, p90=101.5,
                    n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                ),
                status="computed",
            ),
        ]
        critique = detect_degenerate_outcomes(options)
        assert critique is not None
        assert critique.code == "DEGENERATE_OUTCOMES"

    def test_empty_options(self):
        """Test no detection for empty options."""
        critique = detect_degenerate_outcomes([])
        assert critique is None

    def test_single_option(self):
        """Test no detection for single option."""
        options = [
            OptionResultV2(
                id="a",
                outcome=OutcomeDistributionV2(
                    mean=100, std=10, p10=85, p50=100, p90=115,
                    n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                ),
                status="computed",
            ),
        ]
        critique = detect_degenerate_outcomes(options)
        assert critique is None

    def test_zero_median_handling(self):
        """Test handling of all-zero p50 values (detector uses p50, not mean)."""
        options = [
            OptionResultV2(
                id="a",
                outcome=OutcomeDistributionV2(
                    mean=0.0, std=0.1, p10=-0.1, p50=0.0, p90=0.1,
                    n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                ),
                status="computed",
            ),
            OptionResultV2(
                id="b",
                outcome=OutcomeDistributionV2(
                    mean=0.0, std=0.1, p10=-0.1, p50=0.0, p90=0.1,
                    n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                ),
                status="computed",
            ),
        ]
        # Should detect degenerate when all p50 values are zero
        critique = detect_degenerate_outcomes(options)
        assert critique is not None
        assert critique.code == "DEGENERATE_OUTCOMES"

    def test_small_nonzero_spread(self):
        """Test that small but significant spread is NOT degenerate."""
        options = [
            OptionResultV2(
                id="a",
                outcome=OutcomeDistributionV2(
                    mean=0.0, std=0.1, p10=-0.1, p50=0.0, p90=0.1,
                    n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                ),
                status="computed",
            ),
            OptionResultV2(
                id="b",
                outcome=OutcomeDistributionV2(
                    mean=1.0, std=0.1, p10=0.9, p50=1.0, p90=1.1,
                    n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                ),
                status="computed",
            ),
        ]
        # p50 spread of 1.0 on max_abs of 1.0 = 100% spread, NOT degenerate
        critique = detect_degenerate_outcomes(options)
        assert critique is None


# =============================================================================
# Response Builder Tests
# =============================================================================


class TestResponseBuilder:
    """Test response builder."""

    @pytest.fixture
    def request_echo(self):
        """Sample request echo."""
        return RequestEchoV2(
            graph_node_count=5,
            graph_edge_count=4,
            options_count=2,
            goal_node_id_hash="abc123def456",
            n_samples=1000,
            response_version_requested=2,
            include_diagnostics=False,
        )

    def test_build_computed_response(self, request_echo):
        """Test building computed response."""
        builder = ResponseBuilder("req_123", request_echo)
        builder.set_results(
            options=[
                OptionResultV2(
                    id="opt1",
                    outcome=OutcomeDistributionV2(
                        mean=100, std=10, p10=85, p50=100, p90=115,
                        n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                    ),
                    status="computed",
                ),
            ],
        )

        response = builder.build()

        assert response.analysis_status == "computed"
        assert response.robustness_status == "skipped"
        assert response.status_reason is None
        assert len(response.options) == 1

    def test_build_failed_response_with_blocker(self, request_echo):
        """Test building failed response with blocker critique."""
        builder = ResponseBuilder("req_123", request_echo)
        builder.add_critique(MISSING_GOAL_NODE.build())

        response = builder.build()

        assert response.analysis_status == "failed"
        assert response.robustness_status == "unavailable"
        assert "Blocked by" in response.status_reason
        assert len(response.critiques) == 1

    def test_build_partial_response(self, request_echo):
        """Test building partial response."""
        builder = ResponseBuilder("req_123", request_echo)
        builder.set_results(
            options=[
                OptionResultV2(
                    id="opt1",
                    outcome=OutcomeDistributionV2(
                        mean=100, std=10, p10=85, p50=100, p90=115,
                        n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                    ),
                    status="computed",
                ),
                OptionResultV2(
                    id="opt2",
                    outcome=OutcomeDistributionV2(
                        mean=0, std=0, p10=0, p50=0, p90=0,
                        n_samples=1000, n_valid_samples=0, validity_ratio=0.0
                    ),
                    status="failed",
                ),
            ],
        )

        response = builder.build()

        assert response.analysis_status == "partial"
        assert "Some options could not be computed" in response.status_reason

    def test_build_with_robustness(self, request_echo):
        """Test building response with robustness."""
        builder = ResponseBuilder("req_123", request_echo)
        builder.set_results(
            options=[
                OptionResultV2(
                    id="opt1",
                    outcome=OutcomeDistributionV2(
                        mean=100, std=10, p10=85, p50=100, p90=115,
                        n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                    ),
                    status="computed",
                ),
            ],
            robustness=RobustnessResultV2(level="high", confidence=0.95),
        )

        response = builder.build()

        assert response.robustness_status == "computed"
        assert response.robustness.level == "high"

    def test_build_error_response(self, request_echo):
        """Test building error response."""
        builder = ResponseBuilder("req_123", request_echo)

        response = builder.build_error_response(ValueError("Test error"))

        assert response.analysis_status == "failed"
        assert response.robustness_status == "error"
        assert response.status_reason == "Internal error occurred"
        assert any(c.code == "INTERNAL_ERROR" for c in response.critiques)

    def test_processing_time_calculated(self, request_echo):
        """Test processing time is calculated."""
        import time
        builder = ResponseBuilder("req_123", request_echo)
        time.sleep(0.01)  # Small delay

        response = builder.build()

        assert response.processing_time_ms >= 10


class TestBuildRequestEcho:
    """Test request echo builder."""

    def test_basic_request_echo(self):
        """Test building basic request echo."""
        echo = build_request_echo(
            graph_node_count=10,
            graph_edge_count=15,
            options_count=3,
            goal_node_id="revenue_node",
            n_samples=1000,
            response_version=2,
            include_diagnostics=True,
        )

        assert echo.graph_node_count == 10
        assert echo.graph_edge_count == 15
        assert echo.options_count == 3
        assert echo.n_samples == 1000
        assert echo.response_version_requested == 2
        assert echo.include_diagnostics is True
        # Goal node ID should be hashed
        assert echo.goal_node_id_hash != "revenue_node"
        assert len(echo.goal_node_id_hash) == 12


class TestDetermineOptionStatus:
    """Test option status determination."""

    def test_computed_status(self):
        """Test computed status when all samples valid."""
        status = determine_option_status(n_valid=1000, n_total=1000)
        assert status == "computed"

    def test_partial_status(self):
        """Test partial status when below threshold."""
        status = determine_option_status(n_valid=700, n_total=1000)
        assert status == "partial"

    def test_failed_status(self):
        """Test failed status when no valid samples."""
        status = determine_option_status(n_valid=0, n_total=1000)
        assert status == "failed"

    def test_threshold_boundary(self):
        """Test status at threshold boundary."""
        # 80% is the threshold
        status = determine_option_status(n_valid=800, n_total=1000)
        assert status == "computed"

        status = determine_option_status(n_valid=799, n_total=1000)
        assert status == "partial"


class TestHashNodeId:
    """Test node ID hashing."""

    def test_consistent_hashing(self):
        """Test consistent hash for same input."""
        hash1 = hash_node_id("revenue_node")
        hash2 = hash_node_id("revenue_node")
        assert hash1 == hash2

    def test_different_hashes(self):
        """Test different hash for different inputs."""
        hash1 = hash_node_id("revenue_node")
        hash2 = hash_node_id("cost_node")
        assert hash1 != hash2

    def test_hash_length(self):
        """Test hash is truncated to 12 characters."""
        hash_val = hash_node_id("any_node_id")
        assert len(hash_val) == 12


# =============================================================================
# Response V2 Model Tests
# =============================================================================


class TestResponseV2Models:
    """Test V2 response model structures."""

    def test_critique_v2_model(self):
        """Test CritiqueV2 model structure."""
        critique = CritiqueV2(
            id="critique_abc123",
            code="TEST_CODE",
            severity="warning",
            message="Test message",
            source="analysis",
            affected_option_ids=["opt1"],
            suggestion="Test suggestion",
        )

        assert critique.id == "critique_abc123"
        assert critique.severity == "warning"
        assert critique.source == "analysis"

    def test_option_diagnostic_v2_model(self):
        """Test OptionDiagnosticV2 model structure."""
        diag = OptionDiagnosticV2(
            option_id="opt1",
            intervention_count=2,
            has_structural_path=True,
            has_effective_path=True,
            targets_with_effective_path_count=2,
            targets_without_effective_path_count=0,
            warnings=[],
        )

        assert diag.has_effective_path is True

    def test_isl_response_v2_full(self):
        """Test full ISLResponseV2 structure."""
        response = ISLResponseV2(
            endpoint_version="analyze/v2",
            engine_version="1.0.0",
            analysis_status="computed",
            robustness_status="computed",
            factor_sensitivity_status="skipped",
            critiques=[],
            request_echo=RequestEchoV2(
                graph_node_count=5,
                graph_edge_count=4,
                options_count=2,
                goal_node_id_hash="abc123",
                n_samples=1000,
                response_version_requested=2,
                include_diagnostics=False,
            ),
            options=[
                OptionResultV2(
                    id="opt1",
                    outcome=OutcomeDistributionV2(
                        mean=100, std=10, p10=85, p50=100, p90=115,
                        n_samples=1000, n_valid_samples=1000, validity_ratio=1.0
                    ),
                    status="computed",
                ),
            ],
            robustness=RobustnessResultV2(level="high", confidence=0.92),
            request_id="req_123",
            processing_time_ms=150,
        )

        assert response.response_schema_version == "2.0"
        assert response.analysis_status == "computed"
        assert response.robustness.level == "high"
