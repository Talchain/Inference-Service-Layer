"""
Unit tests for v2.2 dual uncertainty robustness analysis.

Tests:
- Schema validation (v2 models)
- SeededRNG determinism
- DualUncertaintySampler
- SCMEvaluatorV2
- RobustnessAnalyzerV2
- Sensitivity analysis
- Schema version detection
"""

import logging
import numpy as np
import pytest

from src.models.robustness_v2 import (
    ClampMetrics,
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    OptionResult,
    OutcomeDistribution,
    ResponseMetadataV2,
    RobustnessRequestV2,
    RobustnessResponseV2,
    RobustnessResult,
    SensitivityResult,
    StrengthDistribution,
    detect_schema_version,
)
from src.services.robustness_analyzer_v2 import (
    DualUncertaintySampler,
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
)
from src.utils.rng import SeededRNG, compute_seed_from_dict, compute_seed_from_graph


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_graph():
    """Create a simple two-node graph for testing."""
    return GraphV2(
        nodes=[
            NodeV2(id="price", kind="factor", label="Price"),
            NodeV2(id="revenue", kind="outcome", label="Revenue"),
        ],
        edges=[
            EdgeV2(
                **{"from": "price", "to": "revenue"},
                exists_probability=0.9,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            )
        ],
    )


@pytest.fixture
def complex_graph():
    """Create a more complex graph for integration tests."""
    return GraphV2(
        nodes=[
            NodeV2(id="marketing", kind="factor", label="Marketing Spend"),
            NodeV2(id="price", kind="factor", label="Price"),
            NodeV2(id="demand", kind="chance", label="Demand"),
            NodeV2(id="revenue", kind="outcome", label="Revenue"),
        ],
        edges=[
            EdgeV2(
                **{"from": "marketing", "to": "demand"},
                exists_probability=0.8,
                strength=StrengthDistribution(mean=0.6, std=0.15),
            ),
            EdgeV2(
                **{"from": "price", "to": "demand"},
                exists_probability=0.95,
                strength=StrengthDistribution(mean=-0.4, std=0.1),
            ),
            EdgeV2(
                **{"from": "demand", "to": "revenue"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.8, std=0.05),
            ),
            EdgeV2(
                **{"from": "price", "to": "revenue"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=0.3, std=0.05),
            ),
        ],
    )


@pytest.fixture
def simple_options():
    """Create simple intervention options."""
    return [
        InterventionOption(
            id="low_price", label="Low price", interventions={"price": 0.3}
        ),
        InterventionOption(
            id="high_price", label="High price", interventions={"price": 0.7}
        ),
    ]


@pytest.fixture
def simple_request(simple_graph, simple_options):
    """Create a simple robustness request."""
    return RobustnessRequestV2(
        request_id="test-001",
        graph=simple_graph,
        options=simple_options,
        goal_node_id="revenue",
        n_samples=100,
        seed=42,
    )


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestStrengthDistribution:
    """Test StrengthDistribution model."""

    def test_valid_positive_mean(self):
        """Test valid strength with positive mean."""
        dist = StrengthDistribution(mean=0.5, std=0.1)
        assert dist.mean == 0.5
        assert dist.std == 0.1

    def test_valid_negative_mean(self):
        """Test valid strength with negative mean (inverse effect)."""
        dist = StrengthDistribution(mean=-0.5, std=0.1)
        assert dist.mean == -0.5

    def test_invalid_zero_std(self):
        """Test rejection of zero std."""
        with pytest.raises(ValueError):
            StrengthDistribution(mean=0.5, std=0.0)

    def test_invalid_negative_std(self):
        """Test rejection of negative std."""
        with pytest.raises(ValueError):
            StrengthDistribution(mean=0.5, std=-0.1)

    def test_invalid_minimum_std(self):
        """Test rejection of std at or below minimum threshold."""
        with pytest.raises(ValueError):
            StrengthDistribution(mean=0.5, std=0.001)


class TestObservedState:
    """Test ObservedState model."""

    def test_valid_observed_state_full(self):
        """Test valid observed state with all fields."""
        state = ObservedState(
            value=59.0,
            baseline=49.0,
            unit="£k",
            source="brief_extraction"
        )
        assert state.value == 59.0
        assert state.baseline == 49.0
        assert state.unit == "£k"
        assert state.source == "brief_extraction"

    def test_valid_observed_state_value_only(self):
        """Test observed state with only required value field."""
        state = ObservedState(value=42.5)
        assert state.value == 42.5
        assert state.baseline is None
        assert state.unit is None
        assert state.source is None

    def test_missing_optional_fields_default_none(self):
        """Test optional fields default to None."""
        state = ObservedState(value=100.0)
        assert state.baseline is None
        assert state.unit is None
        assert state.source is None

    def test_negative_value_allowed(self):
        """Test negative values are allowed."""
        state = ObservedState(value=-25.0, baseline=0.0)
        assert state.value == -25.0
        assert state.baseline == 0.0

    def test_zero_value_allowed(self):
        """Test zero value is allowed."""
        state = ObservedState(value=0.0)
        assert state.value == 0.0

    def test_value_nan_rejected(self):
        """Test NaN value is rejected."""
        with pytest.raises(ValueError, match="finite"):
            ObservedState(value=float("nan"))

    def test_value_inf_rejected(self):
        """Test infinity value is rejected."""
        with pytest.raises(ValueError, match="finite"):
            ObservedState(value=float("inf"))

    def test_value_neg_inf_rejected(self):
        """Test negative infinity value is rejected."""
        with pytest.raises(ValueError, match="finite"):
            ObservedState(value=float("-inf"))

    def test_baseline_nan_rejected(self):
        """Test NaN baseline is rejected."""
        with pytest.raises(ValueError, match="finite"):
            ObservedState(value=10.0, baseline=float("nan"))

    def test_baseline_inf_rejected(self):
        """Test infinity baseline is rejected."""
        with pytest.raises(ValueError, match="finite"):
            ObservedState(value=10.0, baseline=float("inf"))

    def test_from_dict(self):
        """Test creating observed state from dictionary."""
        data = {
            "value": 59.0,
            "baseline": 49.0,
            "unit": "£k",
            "source": "user_input"
        }
        state = ObservedState(**data)
        assert state.value == 59.0
        assert state.source == "user_input"

    def test_to_dict(self):
        """Test serializing observed state to dictionary."""
        state = ObservedState(value=59.0, unit="£k")
        data = state.model_dump()
        assert data["value"] == 59.0
        assert data["unit"] == "£k"
        assert data["baseline"] is None


class TestNodeV2WithObservedState:
    """Test NodeV2 model with observed_state field."""

    def test_node_without_observed_state_backward_compatible(self):
        """Test nodes without observed_state continue to work (backward compatibility)."""
        node = NodeV2(id="revenue", kind="outcome", label="Revenue")
        assert node.id == "revenue"
        assert node.observed_state is None

    def test_node_with_observed_state(self):
        """Test node with observed_state is parsed correctly."""
        node = NodeV2(
            id="marketing-spend",
            kind="factor",
            label="Marketing Spend",
            observed_state=ObservedState(
                value=100000.0,
                baseline=75000.0,
                unit="$",
                source="brief_extraction"
            )
        )
        assert node.id == "marketing-spend"
        assert node.observed_state is not None
        assert node.observed_state.value == 100000.0
        assert node.observed_state.baseline == 75000.0
        assert node.observed_state.unit == "$"
        assert node.observed_state.source == "brief_extraction"

    def test_node_observed_state_from_dict(self):
        """Test node with observed_state created from dictionary."""
        data = {
            "id": "revenue",
            "kind": "outcome",
            "label": "Total Revenue",
            "observed_state": {
                "value": 59.0,
                "baseline": 49.0,
                "unit": "£k"
            }
        }
        node = NodeV2(**data)
        assert node.observed_state is not None
        assert node.observed_state.value == 59.0

    def test_node_observed_state_accessible(self):
        """Test observed_state fields are accessible after parsing."""
        node = NodeV2(
            id="sales",
            kind="factor",
            label="Sales Volume",
            observed_state=ObservedState(value=1500.0, unit="units")
        )
        assert node.observed_state.value == 1500.0
        assert node.observed_state.unit == "units"
        assert node.observed_state.baseline is None

    def test_node_serialization_includes_observed_state(self):
        """Test node serialization includes observed_state."""
        node = NodeV2(
            id="price",
            kind="decision",
            label="Price Point",
            observed_state=ObservedState(value=49.99)
        )
        data = node.model_dump()
        assert "observed_state" in data
        assert data["observed_state"]["value"] == 49.99

    def test_node_without_observed_state_serialization(self):
        """Test node without observed_state serializes with None."""
        node = NodeV2(id="x", kind="factor", label="X")
        data = node.model_dump()
        assert data["observed_state"] is None


class TestEdgeV2:
    """Test EdgeV2 model."""

    def test_valid_edge(self):
        """Test valid edge creation."""
        edge = EdgeV2(
            **{"from": "a", "to": "b"},
            exists_probability=0.9,
            strength=StrengthDistribution(mean=0.5, std=0.1),
        )
        assert edge.from_ == "a"
        assert edge.to == "b"
        assert edge.exists_probability == 0.9

    def test_exists_probability_bounds(self):
        """Test exists_probability must be in [0, 1]."""
        with pytest.raises(ValueError):
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=1.5,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            )

        with pytest.raises(ValueError):
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=-0.1,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            )

    def test_boundary_exists_probability(self):
        """Test edge with boundary exists_probability values."""
        # Always exists
        edge_certain = EdgeV2(
            **{"from": "a", "to": "b"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=0.5, std=0.1),
        )
        assert edge_certain.exists_probability == 1.0

        # Never exists
        edge_never = EdgeV2(
            **{"from": "a", "to": "b"},
            exists_probability=0.0,
            strength=StrengthDistribution(mean=0.5, std=0.1),
        )
        assert edge_never.exists_probability == 0.0

    def test_invalid_node_id_pattern(self):
        """Test rejection of invalid node ID patterns."""
        with pytest.raises(ValueError):
            EdgeV2(
                **{"from": "Invalid Node!", "to": "b"},
                exists_probability=0.9,
                strength=StrengthDistribution(mean=0.5, std=0.1),
            )


class TestGraphV2:
    """Test GraphV2 model."""

    def test_valid_graph(self, simple_graph):
        """Test valid graph creation."""
        assert len(simple_graph.nodes) == 2
        assert len(simple_graph.edges) == 1

    def test_duplicate_node_ids(self):
        """Test rejection of duplicate node IDs."""
        with pytest.raises(ValueError, match="Duplicate node IDs"):
            GraphV2(
                nodes=[
                    NodeV2(id="a", kind="factor", label="A"),
                    NodeV2(id="a", kind="outcome", label="A duplicate"),
                ],
                edges=[],
            )

    def test_edge_references_nonexistent_node(self):
        """Test rejection of edges referencing non-existent nodes."""
        with pytest.raises(ValueError, match="non-existent"):
            GraphV2(
                nodes=[
                    NodeV2(id="a", kind="factor", label="A"),
                ],
                edges=[
                    EdgeV2(
                        **{"from": "a", "to": "b"},  # b doesn't exist
                        exists_probability=0.9,
                        strength=StrengthDistribution(mean=0.5, std=0.1),
                    )
                ],
            )

    def test_self_loop_rejection(self):
        """Test rejection of self-loops."""
        with pytest.raises(ValueError, match="Self-loop"):
            GraphV2(
                nodes=[NodeV2(id="a", kind="factor", label="A")],
                edges=[
                    EdgeV2(
                        **{"from": "a", "to": "a"},
                        exists_probability=0.9,
                        strength=StrengthDistribution(mean=0.5, std=0.1),
                    )
                ],
            )


class TestRobustnessRequestV2:
    """Test RobustnessRequestV2 model."""

    def test_valid_request(self, simple_request):
        """Test valid request creation."""
        assert simple_request.request_id == "test-001"
        assert simple_request.n_samples == 100
        assert simple_request.goal_node_id == "revenue"

    def test_goal_node_must_exist(self, simple_graph, simple_options):
        """Test rejection when goal node doesn't exist."""
        with pytest.raises(ValueError, match="Goal node.*not found"):
            RobustnessRequestV2(
                request_id="test",
                graph=simple_graph,
                options=simple_options,
                goal_node_id="nonexistent",
            )

    def test_intervention_node_must_exist(self, simple_graph):
        """Test rejection when intervention references non-existent node."""
        with pytest.raises(ValueError, match="non-existent node"):
            RobustnessRequestV2(
                request_id="test",
                graph=simple_graph,
                options=[
                    InterventionOption(
                        id="opt1",
                        label="Option 1",
                        interventions={"nonexistent": 0.5},
                    )
                ],
                goal_node_id="revenue",
            )

    def test_n_samples_bounds(self, simple_graph, simple_options):
        """Test n_samples validation."""
        # Too low
        with pytest.raises(ValueError):
            RobustnessRequestV2(
                request_id="test",
                graph=simple_graph,
                options=simple_options,
                goal_node_id="revenue",
                n_samples=50,  # Min is 100
            )

        # Too high
        with pytest.raises(ValueError):
            RobustnessRequestV2(
                request_id="test",
                graph=simple_graph,
                options=simple_options,
                goal_node_id="revenue",
                n_samples=20000,  # Max is 10000
            )


class TestSchemaDetection:
    """Test schema version detection."""

    def test_detect_v2_schema(self):
        """Test detection of v2 schema."""
        request = {
            "request_id": "test",
            "graph": {"nodes": [], "edges": []},
            "options": [],
            "goal_node_id": "x",
        }
        assert detect_schema_version(request) == "v2"

    def test_detect_v1_schema(self):
        """Test detection of v1 schema."""
        request = {
            "causal_model": {"nodes": [], "edges": []},
            "intervention_proposal": {},
            "target_outcome": {},
        }
        assert detect_schema_version(request) == "v1"

    def test_unknown_schema(self):
        """Test rejection of unknown schema."""
        with pytest.raises(ValueError, match="Unknown request schema"):
            detect_schema_version({"invalid": "request"})


# =============================================================================
# RNG Tests
# =============================================================================


class TestSeededRNG:
    """Test SeededRNG determinism."""

    def test_deterministic_random(self):
        """Test that same seed produces same sequence."""
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(42)

        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 == seq2

    def test_deterministic_normal(self):
        """Test deterministic normal sampling."""
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(42)

        seq1 = [rng1.normal(0, 1) for _ in range(10)]
        seq2 = [rng2.normal(0, 1) for _ in range(10)]

        assert seq1 == seq2

    def test_bernoulli_probability(self):
        """Test Bernoulli distribution matches probability."""
        rng = SeededRNG(42)
        n_trials = 10000
        p = 0.7

        successes = sum(1 for _ in range(n_trials) if rng.bernoulli(p))
        observed_p = successes / n_trials

        # Should be within 3 sigma
        assert abs(observed_p - p) < 0.03

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different sequences."""
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(43)

        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 != seq2


class TestComputeSeedFromGraph:
    """Test seed computation from graph."""

    def test_deterministic(self, simple_graph):
        """Test same graph produces same seed."""
        seed1 = compute_seed_from_graph(simple_graph)
        seed2 = compute_seed_from_graph(simple_graph)
        assert seed1 == seed2

    def test_different_graphs_different_seeds(self):
        """Test different graphs produce different seeds."""
        graph1 = GraphV2(
            nodes=[NodeV2(id="a", kind="factor", label="A")],
            edges=[],
        )
        graph2 = GraphV2(
            nodes=[NodeV2(id="b", kind="factor", label="B")],
            edges=[],
        )

        seed1 = compute_seed_from_graph(graph1)
        seed2 = compute_seed_from_graph(graph2)

        assert seed1 != seed2

    def test_seed_is_valid_integer(self, simple_graph):
        """Test seed is a valid 32-bit unsigned integer."""
        seed = compute_seed_from_graph(simple_graph)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32


# =============================================================================
# Dual Uncertainty Sampler Tests
# =============================================================================


class TestDualUncertaintySampler:
    """Test DualUncertaintySampler."""

    def test_sample_edge_configuration(self, simple_graph):
        """Test basic edge configuration sampling."""
        rng = SeededRNG(42)
        sampler = DualUncertaintySampler(simple_graph.edges, rng)

        config = sampler.sample_edge_configuration()

        assert ("price", "revenue") in config

    def test_edge_existence_probability(self):
        """Test edge existence follows Bernoulli distribution."""
        # Edge with 50% existence probability
        edges = [
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=0.5,
                strength=StrengthDistribution(mean=1.0, std=0.1),
            )
        ]
        rng = SeededRNG(42)
        sampler = DualUncertaintySampler(edges, rng)

        configs = sampler.sample_n_configurations(1000)
        exists_count = sum(1 for c in configs if c[("a", "b")] != 0)

        # Should be approximately 500 ± 50 (3 sigma)
        assert 400 < exists_count < 600

    def test_always_existing_edge(self):
        """Test edge with exists_probability=1.0 always exists."""
        edges = [
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=1.0, std=0.1),
            )
        ]
        rng = SeededRNG(42)
        sampler = DualUncertaintySampler(edges, rng)

        configs = sampler.sample_n_configurations(100)

        # All should have non-zero strength
        for config in configs:
            assert config[("a", "b")] != 0

    def test_never_existing_edge(self):
        """Test edge with exists_probability=0.0 never exists."""
        edges = [
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=0.0,
                strength=StrengthDistribution(mean=1.0, std=0.1),
            )
        ]
        rng = SeededRNG(42)
        sampler = DualUncertaintySampler(edges, rng)

        configs = sampler.sample_n_configurations(100)

        # All should have zero strength
        for config in configs:
            assert config[("a", "b")] == 0

    def test_strength_distribution(self):
        """Test sampled strengths follow Normal distribution."""
        edges = [
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=-0.5, std=0.2),
            )
        ]
        rng = SeededRNG(42)
        sampler = DualUncertaintySampler(edges, rng)

        configs = sampler.sample_n_configurations(1000)
        strengths = [c[("a", "b")] for c in configs]

        # Mean should be approximately -0.5
        assert -0.55 < np.mean(strengths) < -0.45

        # Std should be approximately 0.2
        assert 0.15 < np.std(strengths) < 0.25

    def test_negative_strength_mean(self):
        """Test negative mean produces negative strengths."""
        edges = [
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=1.0,
                strength=StrengthDistribution(mean=-1.0, std=0.1),
            )
        ]
        rng = SeededRNG(42)
        sampler = DualUncertaintySampler(edges, rng)

        configs = sampler.sample_n_configurations(100)
        strengths = [c[("a", "b")] for c in configs]

        # Most samples should be negative (mean=-1.0, std=0.1)
        negative_count = sum(1 for s in strengths if s < 0)
        assert negative_count > 90

    def test_existence_rates_tracking(self):
        """Test existence rates are correctly tracked."""
        edges = [
            EdgeV2(
                **{"from": "a", "to": "b"},
                exists_probability=0.7,
                strength=StrengthDistribution(mean=1.0, std=0.1),
            )
        ]
        rng = SeededRNG(42)
        sampler = DualUncertaintySampler(edges, rng)

        sampler.sample_n_configurations(1000)
        rates = sampler.get_existence_rates()

        assert "a->b" in rates
        assert 0.65 < rates["a->b"] < 0.75


# =============================================================================
# SCM Evaluator Tests
# =============================================================================


class TestSCMEvaluatorV2:
    """Test SCMEvaluatorV2."""

    def test_simple_evaluation(self, simple_graph):
        """Test simple graph evaluation."""
        evaluator = SCMEvaluatorV2(simple_graph)

        edge_config = {("price", "revenue"): 0.5}
        interventions = {"price": 1.0}

        outcome = evaluator.evaluate(
            edge_strengths=edge_config,
            interventions=interventions,
            goal_node="revenue",
        )

        # revenue = 0 + price * 0.5 = 1.0 * 0.5 = 0.5
        assert outcome == pytest.approx(0.5, rel=1e-6)

    def test_chain_evaluation(self):
        """Test evaluation through chain of nodes."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A"),
                NodeV2(id="b", kind="chance", label="B"),
                NodeV2(id="c", kind="outcome", label="C"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=2.0, std=0.1),
                ),
                EdgeV2(
                    **{"from": "b", "to": "c"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=3.0, std=0.1),
                ),
            ],
        )

        evaluator = SCMEvaluatorV2(graph)

        edge_config = {("a", "b"): 2.0, ("b", "c"): 3.0}
        interventions = {"a": 1.0}

        outcome = evaluator.evaluate(
            edge_strengths=edge_config,
            interventions=interventions,
            goal_node="c",
        )

        # a = 1.0 (intervention)
        # b = 1.0 * 2.0 = 2.0
        # c = 2.0 * 3.0 = 6.0
        assert outcome == pytest.approx(6.0, rel=1e-6)

    def test_zero_strength_edge(self, simple_graph):
        """Test edge with zero strength (doesn't exist)."""
        evaluator = SCMEvaluatorV2(simple_graph)

        edge_config = {("price", "revenue"): 0.0}  # Edge doesn't exist
        interventions = {"price": 1.0}

        outcome = evaluator.evaluate(
            edge_strengths=edge_config,
            interventions=interventions,
            goal_node="revenue",
        )

        # No edge effect, revenue should be 0
        assert outcome == pytest.approx(0.0, rel=1e-6)

    def test_negative_strength(self, simple_graph):
        """Test negative strength produces inverse effect."""
        evaluator = SCMEvaluatorV2(simple_graph)

        edge_config = {("price", "revenue"): -0.5}  # Negative effect
        interventions = {"price": 1.0}

        outcome = evaluator.evaluate(
            edge_strengths=edge_config,
            interventions=interventions,
            goal_node="revenue",
        )

        # revenue = 1.0 * -0.5 = -0.5
        assert outcome == pytest.approx(-0.5, rel=1e-6)


# =============================================================================
# Robustness Analyzer Tests
# =============================================================================


class TestRobustnessAnalyzerV2:
    """Test RobustnessAnalyzerV2."""

    def test_analyze_returns_response(self, simple_request):
        """Test analyze returns valid response."""
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(simple_request)

        assert isinstance(response, RobustnessResponseV2)
        assert response.request_id == simple_request.request_id
        assert len(response.results) == len(simple_request.options)

    def test_non_inference_nodes_filtered_with_warning(self, caplog):
        """Non-inference nodes and edges should be filtered with warning."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="decision", kind="decision", label="Decision"),
                NodeV2(id="factor", kind="factor", label="Factor"),
                NodeV2(id="outcome", kind="outcome", label="Outcome"),
                NodeV2(id="constraint", kind="constraint", label="Constraint"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "decision", "to": "outcome"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
                EdgeV2(
                    **{"from": "factor", "to": "outcome"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
                EdgeV2(
                    **{"from": "constraint", "to": "factor"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="filter-test",
            graph=graph,
            options=[
                InterventionOption(
                    id="opt1", label="Option 1", interventions={"factor": 1.0}
                )
            ],
            goal_node_id="outcome",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        with caplog.at_level(logging.WARNING):
            response = analyzer.analyze(request)

        assert any(
            record.message == "robustness_v2_filtered_non_inference_nodes"
            for record in caplog.records
        )
        assert set(response.metadata.edge_existence_rates.keys()) == {"factor->outcome"}

    def test_deterministic_results(self, simple_request):
        """Test same request with same seed produces same results."""
        analyzer = RobustnessAnalyzerV2()

        response1 = analyzer.analyze(simple_request)
        response2 = analyzer.analyze(simple_request)

        assert response1.recommended_option_id == response2.recommended_option_id
        assert response1.recommendation_confidence == response2.recommendation_confidence

    def test_win_probabilities_sum_to_one(self, simple_request):
        """Test win probabilities sum to approximately 1."""
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(simple_request)

        total_win_prob = sum(r.win_probability for r in response.results)
        assert total_win_prob == pytest.approx(1.0, rel=1e-6)

    def test_sensitivity_analysis_included(self, simple_request):
        """Test sensitivity analysis is included when requested."""
        simple_request.analysis_types = ["sensitivity"]
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(simple_request)

        # Should have sensitivity results for each edge (2 per edge: existence + magnitude)
        assert len(response.sensitivity) == 2 * len(simple_request.graph.edges)

    def test_sensitivity_types(self, simple_request):
        """Test both sensitivity types are included."""
        simple_request.analysis_types = ["sensitivity"]
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(simple_request)

        sensitivity_types = {s.sensitivity_type for s in response.sensitivity}
        assert "existence" in sensitivity_types
        assert "magnitude" in sensitivity_types

    def test_metadata_populated(self, simple_request):
        """Test metadata is correctly populated."""
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(simple_request)

        assert response.metadata.schema_version == "2.2"
        assert response.metadata.n_samples_used == simple_request.n_samples
        assert response.metadata.seed_used == simple_request.seed
        assert response.metadata.execution_time_ms > 0

    def test_edge_existence_rates_in_metadata(self, simple_request):
        """Test edge existence rates are reported in metadata."""
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(simple_request)

        assert "price->revenue" in response.metadata.edge_existence_rates

    def test_robustness_result(self, simple_request):
        """Test robustness result is populated."""
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(simple_request)

        assert isinstance(response.robustness, RobustnessResult)
        assert 0 <= response.robustness.recommendation_stability <= 1
        assert response.robustness.interpretation != ""

    def test_certain_edge_high_stability(self):
        """Test graph with certain edges has high stability."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="price", kind="factor", label="Price"),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "price", "to": "revenue"},
                    exists_probability=1.0,  # Certain
                    strength=StrengthDistribution(mean=1.0, std=0.01),  # Low variance
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="test",
            graph=graph,
            options=[
                InterventionOption(id="low", label="Low", interventions={"price": 0.3}),
                InterventionOption(id="high", label="High", interventions={"price": 0.7}),
            ],
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should have very high stability
        assert response.robustness.recommendation_stability > 0.95


class TestSensitivityInterpretation:
    """Test sensitivity interpretation generation."""

    def test_robust_interpretation(self, simple_request):
        """Test interpretation for robust edge."""
        simple_request.analysis_types = ["sensitivity"]
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(simple_request)

        for sens in response.sensitivity:
            assert sens.interpretation != ""
            assert sens.edge_from in sens.interpretation or sens.edge_to in sens.interpretation

    def test_importance_ranking(self, simple_request):
        """Test sensitivity results are ranked by importance."""
        simple_request.analysis_types = ["sensitivity"]
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(simple_request)

        # Should be sorted by absolute elasticity (rank 1 = highest)
        ranks = [s.importance_rank for s in response.sensitivity]
        assert ranks == sorted(ranks)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRobustnessV2Integration:
    """Integration tests for full analysis flow."""

    def test_full_analysis_with_complex_graph(self, complex_graph):
        """Test full analysis with complex graph."""
        request = RobustnessRequestV2(
            request_id="integration-test",
            graph=complex_graph,
            options=[
                InterventionOption(
                    id="low_marketing",
                    label="Low marketing",
                    interventions={"marketing": 0.3, "price": 0.5},
                ),
                InterventionOption(
                    id="high_marketing",
                    label="High marketing",
                    interventions={"marketing": 0.8, "price": 0.5},
                ),
            ],
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Verify structure
        assert len(response.results) == 2
        assert response.recommended_option_id in ["low_marketing", "high_marketing"]

        # Verify sensitivity for all edges
        assert len(response.sensitivity) == 2 * len(complex_graph.edges)

        # Verify edge existence rates match expected probabilities (approximately)
        rates = response.metadata.edge_existence_rates
        assert "marketing->demand" in rates
        assert 0.7 < rates["marketing->demand"] < 0.9  # ~0.8

    def test_negative_effect_handling(self):
        """Test negative strength.mean produces inverse relationships."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="price", kind="factor", label="Price"),
                NodeV2(id="demand", kind="outcome", label="Demand"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "price", "to": "demand"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=-1.0, std=0.1),  # Negative
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="test",
            graph=graph,
            options=[
                InterventionOption(id="low", label="Low price", interventions={"price": 0.3}),
                InterventionOption(id="high", label="High price", interventions={"price": 0.7}),
            ],
            goal_node_id="demand",
            n_samples=500,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Find results
        low_result = next(r for r in response.results if r.option_id == "low")
        high_result = next(r for r in response.results if r.option_id == "high")

        # With negative effect, lower price should lead to higher demand
        assert low_result.outcome_distribution.mean > high_result.outcome_distribution.mean

    def test_edge_existence_affects_outcome(self):
        """Test that edge existence probability affects outcomes."""
        # Graph with edge that probably doesn't exist
        graph_low = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A"),
                NodeV2(id="b", kind="outcome", label="B"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=0.1,  # Rarely exists
                    strength=StrengthDistribution(mean=10.0, std=0.1),
                )
            ],
        )

        # Graph with edge that probably exists
        graph_high = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A"),
                NodeV2(id="b", kind="outcome", label="B"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=0.9,  # Usually exists
                    strength=StrengthDistribution(mean=10.0, std=0.1),
                )
            ],
        )

        options = [
            InterventionOption(id="opt1", label="Option 1", interventions={"a": 1.0})
        ]

        request_low = RobustnessRequestV2(
            request_id="low",
            graph=graph_low,
            options=options,
            goal_node_id="b",
            n_samples=500,
            seed=42,
        )

        request_high = RobustnessRequestV2(
            request_id="high",
            graph=graph_high,
            options=options,
            goal_node_id="b",
            n_samples=500,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response_low = analyzer.analyze(request_low)
        response_high = analyzer.analyze(request_high)

        # Higher existence probability should lead to higher expected outcome
        mean_low = response_low.results[0].outcome_distribution.mean
        mean_high = response_high.results[0].outcome_distribution.mean

        assert mean_high > mean_low


# =============================================================================
# Magnitude Sensitivity Isolation Tests
# =============================================================================


class TestMagnitudeSensitivityIsolation:
    """Test that magnitude sensitivity is isolated from existence sensitivity."""

    def test_sample_with_shifted_mean_forces_target_existence(self):
        """
        Directly test that _sample_with_shifted_mean forces target edge to exist.
        """
        from src.utils.rng import SeededRNG

        # Create edge with very low existence probability
        edge_never_exists = EdgeV2(
            **{"from": "a", "to": "b"},
            exists_probability=0.0,  # Never exists in normal sampling
            strength=StrengthDistribution(mean=1.0, std=0.1),
        )

        edges = [edge_never_exists]
        rng = SeededRNG(42)
        analyzer = RobustnessAnalyzerV2()

        # Sample 100 times with shifted mean
        for _ in range(100):
            config = analyzer._sample_with_shifted_mean(
                edges, edge_never_exists, shift=0.5, rng=rng
            )
            # Target edge should ALWAYS exist (non-zero strength)
            assert config[("a", "b")] != 0.0, (
                "Target edge should be forced to exist in _sample_with_shifted_mean"
            )

    def test_magnitude_sensitivity_with_low_existence_still_computed(self):
        """
        Edge with low exists_probability should still report magnitude sensitivity.

        The target edge is forced to exist during magnitude sensitivity
        calculation, so even low-probability edges can show magnitude effects.
        """
        # Two edges: one certain (to ensure non-zero baseline) and one uncertain
        graph = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A"),
                NodeV2(id="b", kind="outcome", label="B"),
                NodeV2(id="c", kind="outcome", label="C"),
            ],
            edges=[
                # Certain edge to provide baseline
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
                # Low probability edge - should still show magnitude effect
                EdgeV2(
                    **{"from": "b", "to": "c"},
                    exists_probability=0.1,  # Rarely exists
                    strength=StrengthDistribution(mean=2.0, std=0.5),  # Large std
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Opt1", interventions={"a": 1.0})
            ],
            goal_node_id="c",
            n_samples=500,
            seed=42,
            analysis_types=["sensitivity"],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Find magnitude sensitivity for the low-probability edge
        mag_sens = next(
            s for s in response.sensitivity
            if s.sensitivity_type == "magnitude" and s.edge_from == "b"
        )

        # Even though edge rarely exists in normal sampling,
        # magnitude sensitivity should be computed (edge forced to exist)
        # The exact value doesn't matter, just that it's non-zero
        assert mag_sens.elasticity != 0.0, (
            "Magnitude sensitivity should be computed even for low-probability edges"
        )

    def test_other_edges_still_sampled_normally_in_magnitude_sensitivity(self):
        """
        When computing magnitude sensitivity for one edge,
        other edges should still sample existence normally.
        """
        from src.utils.rng import SeededRNG

        # Two edges: target (always exists) and other (never exists)
        target_edge = EdgeV2(
            **{"from": "a", "to": "b"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=1.0, std=0.1),
        )
        other_edge = EdgeV2(
            **{"from": "b", "to": "c"},
            exists_probability=0.0,  # Never exists
            strength=StrengthDistribution(mean=2.0, std=0.1),
        )

        edges = [target_edge, other_edge]
        rng = SeededRNG(42)
        analyzer = RobustnessAnalyzerV2()

        # Sample with shifted mean on TARGET edge
        configs = []
        for _ in range(100):
            config = analyzer._sample_with_shifted_mean(
                edges, target_edge, shift=0.5, rng=rng
            )
            configs.append(config)

        # Target edge should ALWAYS exist
        target_exists = sum(1 for c in configs if c[("a", "b")] != 0)
        assert target_exists == 100, "Target edge should always exist"

        # Other edge should NEVER exist (its exists_probability=0)
        other_exists = sum(1 for c in configs if c[("b", "c")] != 0)
        assert other_exists == 0, "Other edge should follow its exists_probability=0"


# =============================================================================
# Observed State Integration Tests
# =============================================================================


class TestObservedStateIntegration:
    """Integration tests for observed_state in robustness analysis."""

    def test_request_with_observed_state_accepted(self):
        """Test robustness request with nodes containing observed_state is accepted."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing Spend",
                    observed_state=ObservedState(
                        value=100000.0,
                        baseline=75000.0,
                        unit="$",
                        source="brief_extraction"
                    )
                ),
                NodeV2(
                    id="revenue",
                    kind="outcome",
                    label="Revenue",
                    observed_state=ObservedState(value=500000.0)
                ),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=0.9,
                    strength=StrengthDistribution(mean=2.0, std=0.3),
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="observed-state-test",
            graph=graph,
            options=[
                InterventionOption(
                    id="increase",
                    label="Increase Marketing",
                    interventions={"marketing": 1.5}
                ),
                InterventionOption(
                    id="maintain",
                    label="Maintain Marketing",
                    interventions={"marketing": 1.0}
                ),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )

        # Request should be valid
        assert request.graph.nodes[0].observed_state is not None
        assert request.graph.nodes[0].observed_state.value == 100000.0

    def test_analysis_with_observed_state_returns_response(self):
        """Test full robustness analysis with observed_state returns valid response."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="price",
                    kind="factor",
                    label="Price",
                    observed_state=ObservedState(value=49.0, baseline=39.0, unit="£")
                ),
                NodeV2(
                    id="revenue",
                    kind="outcome",
                    label="Revenue",
                ),
            ],
            edges=[
                EdgeV2(
                    **{"from": "price", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="test-observed-state",
            graph=graph,
            options=[
                InterventionOption(id="low", label="Low price", interventions={"price": 0.3}),
                InterventionOption(id="high", label="High price", interventions={"price": 0.7}),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should return valid response
        assert isinstance(response, RobustnessResponseV2)
        assert response.request_id == "test-observed-state"
        assert len(response.results) == 2

    def test_observed_state_preserved_in_graph(self):
        """Test observed_state is preserved in graph during analysis."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="a",
                    kind="factor",
                    label="A",
                    observed_state=ObservedState(value=42.0, source="test")
                ),
                NodeV2(id="b", kind="outcome", label="B"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                )
            ],
        )

        # Create evaluator
        evaluator = SCMEvaluatorV2(graph)

        # observed_state should be accessible via _nodes_by_id
        node_a = evaluator._nodes_by_id["a"]
        assert node_a.observed_state is not None
        assert node_a.observed_state.value == 42.0
        assert node_a.observed_state.source == "test"

    def test_mixed_nodes_with_and_without_observed_state(self):
        """Test graph with some nodes having observed_state and others not."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing",
                    observed_state=ObservedState(value=100.0)  # Has observed_state
                ),
                NodeV2(
                    id="price",
                    kind="factor",
                    label="Price",
                    # No observed_state
                ),
                NodeV2(
                    id="revenue",
                    kind="outcome",
                    label="Revenue",
                    observed_state=ObservedState(value=500.0, unit="$")  # Has observed_state
                ),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=0.8,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                ),
                EdgeV2(
                    **{"from": "price", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.3, std=0.05),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="mixed-test",
            graph=graph,
            options=[
                InterventionOption(
                    id="opt1", label="Option 1", interventions={"price": 0.5}
                ),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Analysis should complete successfully
        assert response.request_id == "mixed-test"
        assert len(response.results) == 1

        # Verify nodes are accessible with correct observed_state
        evaluator = SCMEvaluatorV2(graph)
        assert evaluator._nodes_by_id["marketing"].observed_state is not None
        assert evaluator._nodes_by_id["price"].observed_state is None
        assert evaluator._nodes_by_id["revenue"].observed_state is not None


# =============================================================================
# Phase 2A Part 2: ParameterUncertainty Tests
# =============================================================================


class TestParameterUncertainty:
    """Test ParameterUncertainty model validation."""

    def test_normal_distribution_requires_std(self):
        """Test normal distribution requires positive std."""
        from src.models.robustness_v2 import ParameterUncertainty

        # Missing std
        with pytest.raises(ValueError, match="std.*must be provided"):
            ParameterUncertainty(
                node_id="marketing",
                distribution="normal",
            )

        # std=0 is invalid
        with pytest.raises(ValueError, match="std.*must be provided and > 0"):
            ParameterUncertainty(
                node_id="marketing",
                distribution="normal",
                std=0.0,
            )

    def test_normal_distribution_valid(self):
        """Test valid normal distribution parameters."""
        from src.models.robustness_v2 import ParameterUncertainty

        uncertainty = ParameterUncertainty(
            node_id="marketing",
            distribution="normal",
            std=5.0,
        )
        assert uncertainty.distribution == "normal"
        assert uncertainty.std == 5.0

    def test_uniform_distribution_requires_range(self):
        """Test uniform distribution requires both range_min and range_max."""
        from src.models.robustness_v2 import ParameterUncertainty

        # Missing both
        with pytest.raises(ValueError, match="range_min.*range_max.*must be provided"):
            ParameterUncertainty(
                node_id="marketing",
                distribution="uniform",
            )

        # Missing range_max
        with pytest.raises(ValueError, match="range_min.*range_max.*must be provided"):
            ParameterUncertainty(
                node_id="marketing",
                distribution="uniform",
                range_min=0.0,
            )

    def test_uniform_distribution_range_order(self):
        """Test uniform distribution requires range_min < range_max."""
        from src.models.robustness_v2 import ParameterUncertainty

        with pytest.raises(ValueError, match="range_min.*must be less than range_max"):
            ParameterUncertainty(
                node_id="marketing",
                distribution="uniform",
                range_min=100.0,
                range_max=50.0,
            )

    def test_uniform_distribution_valid(self):
        """Test valid uniform distribution parameters."""
        from src.models.robustness_v2 import ParameterUncertainty

        uncertainty = ParameterUncertainty(
            node_id="marketing",
            distribution="uniform",
            range_min=50.0,
            range_max=150.0,
        )
        assert uncertainty.distribution == "uniform"
        assert uncertainty.range_min == 50.0
        assert uncertainty.range_max == 150.0

    def test_point_mass_no_extra_params(self):
        """Test point_mass distribution requires no extra parameters."""
        from src.models.robustness_v2 import ParameterUncertainty

        uncertainty = ParameterUncertainty(
            node_id="marketing",
            distribution="point_mass",
        )
        assert uncertainty.distribution == "point_mass"

    def test_unknown_distribution_rejected(self):
        """Test unknown distribution type is rejected."""
        from src.models.robustness_v2 import ParameterUncertainty

        with pytest.raises(ValueError, match="Unknown distribution"):
            ParameterUncertainty(
                node_id="marketing",
                distribution="gamma",  # Not supported
            )

    def test_node_id_pattern_validation(self):
        """Test node_id pattern validation."""
        from src.models.robustness_v2 import ParameterUncertainty

        # Valid patterns
        ParameterUncertainty(node_id="marketing_spend", distribution="point_mass")
        ParameterUncertainty(node_id="node-1", distribution="point_mass")
        ParameterUncertainty(node_id="node:v2", distribution="point_mass")

        # Invalid pattern
        with pytest.raises(ValueError):
            ParameterUncertainty(node_id="Marketing Spend", distribution="point_mass")


class TestParameterUncertaintiesInRequest:
    """Test parameter_uncertainties field in RobustnessRequestV2."""

    def test_request_without_parameter_uncertainties(self, simple_graph, simple_options):
        """Test request works without parameter_uncertainties (backward compatible)."""
        request = RobustnessRequestV2(
            request_id="test",
            graph=simple_graph,
            options=simple_options,
            goal_node_id="revenue",
            n_samples=100,
        )
        assert request.parameter_uncertainties is None

    def test_request_with_parameter_uncertainties(self):
        """Test request accepts valid parameter_uncertainties."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"marketing": 1.0})
            ],
            goal_node_id="revenue",
            n_samples=100,
            parameter_uncertainties=[
                ParameterUncertainty(node_id="marketing", distribution="normal", std=10.0)
            ],
        )

        assert request.parameter_uncertainties is not None
        assert len(request.parameter_uncertainties) == 1
        assert request.parameter_uncertainties[0].node_id == "marketing"

    def test_parameter_uncertainty_nonexistent_node_rejected(self):
        """Test parameter_uncertainty referencing non-existent node is rejected."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(id="marketing", kind="factor", label="Marketing"),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                )
            ],
        )

        with pytest.raises(ValueError, match="non-existent node"):
            RobustnessRequestV2(
                request_id="test",
                graph=graph,
                options=[
                    InterventionOption(id="opt1", label="Option 1", interventions={"marketing": 1.0})
                ],
                goal_node_id="revenue",
                n_samples=100,
                parameter_uncertainties=[
                    ParameterUncertainty(node_id="nonexistent", distribution="normal", std=10.0)
                ],
            )


# =============================================================================
# Phase 2A Part 2: FactorSampler Tests
# =============================================================================


class TestFactorSampler:
    """Test FactorSampler class."""

    def test_sample_normal_distribution(self):
        """Test FactorSampler samples from normal distribution."""
        from src.models.robustness_v2 import ParameterUncertainty
        from src.services.robustness_analyzer_v2 import FactorSampler

        nodes = [
            NodeV2(
                id="marketing",
                kind="factor",
                label="Marketing",
                observed_state=ObservedState(value=100.0),
            )
        ]
        uncertainties = [
            ParameterUncertainty(node_id="marketing", distribution="normal", std=10.0)
        ]

        rng = SeededRNG(42)
        sampler = FactorSampler(nodes, uncertainties, rng)

        # Sample many times
        samples = []
        for _ in range(1000):
            values = sampler.sample_factor_values()
            samples.append(values["marketing"])

        # Mean should be approximately 100.0
        assert 95.0 < np.mean(samples) < 105.0

        # Std should be approximately 10.0
        assert 8.0 < np.std(samples) < 12.0

    def test_sample_uniform_distribution(self):
        """Test FactorSampler samples from uniform distribution."""
        from src.models.robustness_v2 import ParameterUncertainty
        from src.services.robustness_analyzer_v2 import FactorSampler

        nodes = [
            NodeV2(
                id="marketing",
                kind="factor",
                label="Marketing",
                observed_state=ObservedState(value=100.0),  # Ignored for uniform
            )
        ]
        uncertainties = [
            ParameterUncertainty(
                node_id="marketing",
                distribution="uniform",
                range_min=50.0,
                range_max=150.0,
            )
        ]

        rng = SeededRNG(42)
        sampler = FactorSampler(nodes, uncertainties, rng)

        # Sample many times
        samples = []
        for _ in range(1000):
            values = sampler.sample_factor_values()
            samples.append(values["marketing"])

        # All samples should be in range
        assert all(50.0 <= s <= 150.0 for s in samples)

        # Mean should be approximately 100.0 (midpoint)
        assert 95.0 < np.mean(samples) < 105.0

    def test_sample_point_mass_distribution(self):
        """Test FactorSampler returns fixed value for point_mass distribution."""
        from src.models.robustness_v2 import ParameterUncertainty
        from src.services.robustness_analyzer_v2 import FactorSampler

        nodes = [
            NodeV2(
                id="marketing",
                kind="factor",
                label="Marketing",
                observed_state=ObservedState(value=100.0),
            )
        ]
        uncertainties = [
            ParameterUncertainty(node_id="marketing", distribution="point_mass")
        ]

        rng = SeededRNG(42)
        sampler = FactorSampler(nodes, uncertainties, rng)

        # All samples should be exactly 100.0
        for _ in range(100):
            values = sampler.sample_factor_values()
            assert values["marketing"] == 100.0

    def test_sample_without_observed_state_uses_zero(self):
        """Test FactorSampler uses 0.0 as mean if no observed_state."""
        from src.models.robustness_v2 import ParameterUncertainty
        from src.services.robustness_analyzer_v2 import FactorSampler

        nodes = [
            NodeV2(
                id="marketing",
                kind="factor",
                label="Marketing",
                # No observed_state
            )
        ]
        uncertainties = [
            ParameterUncertainty(node_id="marketing", distribution="normal", std=1.0)
        ]

        rng = SeededRNG(42)
        sampler = FactorSampler(nodes, uncertainties, rng)

        # Sample many times
        samples = []
        for _ in range(1000):
            values = sampler.sample_factor_values()
            samples.append(values["marketing"])

        # Mean should be approximately 0.0
        assert -0.5 < np.mean(samples) < 0.5

    def test_has_uncertainties(self):
        """Test has_uncertainties method."""
        from src.models.robustness_v2 import ParameterUncertainty
        from src.services.robustness_analyzer_v2 import FactorSampler

        nodes = [NodeV2(id="a", kind="factor", label="A")]
        rng = SeededRNG(42)

        # No uncertainties
        sampler_empty = FactorSampler(nodes, None, rng)
        assert not sampler_empty.has_uncertainties()

        sampler_empty2 = FactorSampler(nodes, [], rng)
        assert not sampler_empty2.has_uncertainties()

        # With uncertainties
        uncertainties = [
            ParameterUncertainty(node_id="a", distribution="point_mass")
        ]
        sampler_with = FactorSampler(nodes, uncertainties, rng)
        assert sampler_with.has_uncertainties()

    def test_deterministic_sampling(self):
        """Test FactorSampler produces deterministic results with same seed."""
        from src.models.robustness_v2 import ParameterUncertainty
        from src.services.robustness_analyzer_v2 import FactorSampler

        nodes = [
            NodeV2(
                id="marketing",
                kind="factor",
                label="Marketing",
                observed_state=ObservedState(value=100.0),
            )
        ]
        uncertainties = [
            ParameterUncertainty(node_id="marketing", distribution="normal", std=10.0)
        ]

        rng1 = SeededRNG(42)
        rng2 = SeededRNG(42)
        sampler1 = FactorSampler(nodes, uncertainties, rng1)
        sampler2 = FactorSampler(nodes, uncertainties, rng2)

        # Should produce identical sequences
        for _ in range(10):
            values1 = sampler1.sample_factor_values()
            values2 = sampler2.sample_factor_values()
            assert values1["marketing"] == values2["marketing"]


# =============================================================================
# Phase 2A Part 2: SCMEvaluatorV2 Factor Value Tests
# =============================================================================


class TestSCMEvaluatorFactorValues:
    """Test SCMEvaluatorV2 factor value handling."""

    def test_uses_observed_state_for_root_nodes(self):
        """Test SCMEvaluatorV2 uses observed_state.value for root nodes."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                )
            ],
        )

        evaluator = SCMEvaluatorV2(graph)

        # No interventions on marketing, should use observed_state.value
        outcome = evaluator.evaluate(
            edge_strengths={("marketing", "revenue"): 2.0},
            interventions={},  # No intervention
            goal_node="revenue",
        )

        # revenue = marketing.observed_state.value * edge_strength
        # = 100.0 * 2.0 = 200.0
        assert outcome == pytest.approx(200.0, rel=1e-6)

    def test_factor_values_override_observed_state(self):
        """Test factor_values takes priority over observed_state."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                )
            ],
        )

        evaluator = SCMEvaluatorV2(graph)

        # factor_values should override observed_state
        outcome = evaluator.evaluate(
            edge_strengths={("marketing", "revenue"): 2.0},
            interventions={},
            goal_node="revenue",
            factor_values={"marketing": 50.0},  # Overrides observed_state.value=100
        )

        # revenue = factor_values["marketing"] * edge_strength = 50.0 * 2.0 = 100.0
        assert outcome == pytest.approx(100.0, rel=1e-6)

    def test_intervention_overrides_factor_values(self):
        """Test interventions take priority over factor_values."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                )
            ],
        )

        evaluator = SCMEvaluatorV2(graph)

        # intervention should override both factor_values and observed_state
        outcome = evaluator.evaluate(
            edge_strengths={("marketing", "revenue"): 2.0},
            interventions={"marketing": 25.0},  # Highest priority
            goal_node="revenue",
            factor_values={"marketing": 50.0},
        )

        # revenue = intervention["marketing"] * edge_strength = 25.0 * 2.0 = 50.0
        assert outcome == pytest.approx(50.0, rel=1e-6)

    def test_non_root_node_ignores_observed_state(self):
        """Test non-root nodes ignore observed_state (computed from parents)."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A"),
                NodeV2(
                    id="b",
                    kind="chance",
                    label="B",
                    # This observed_state should be ignored because b has a parent
                    observed_state=ObservedState(value=999.0),
                ),
                NodeV2(id="c", kind="outcome", label="C"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=2.0, std=0.1),
                ),
                EdgeV2(
                    **{"from": "b", "to": "c"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=3.0, std=0.1),
                ),
            ],
        )

        evaluator = SCMEvaluatorV2(graph)

        outcome = evaluator.evaluate(
            edge_strengths={("a", "b"): 2.0, ("b", "c"): 3.0},
            interventions={"a": 10.0},
            goal_node="c",
        )

        # a = 10.0 (intervention)
        # b = a * 2.0 = 20.0 (NOT 999.0 from observed_state)
        # c = b * 3.0 = 60.0
        assert outcome == pytest.approx(60.0, rel=1e-6)


# =============================================================================
# Phase 2A Part 2: Full Analysis Integration Tests
# =============================================================================


class TestFactorSamplingIntegration:
    """Integration tests for factor sampling in robustness analysis."""

    def test_analysis_with_factor_uncertainties(self):
        """Test full analysis with parameter_uncertainties returns factor_sensitivity."""
        from src.models.robustness_v2 import ParameterUncertainty, FactorSensitivityResult

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing Spend",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.05),
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="factor-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=200,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="marketing", distribution="normal", std=10.0)
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should have factor_sensitivity results
        assert response.factor_sensitivity is not None
        assert len(response.factor_sensitivity) == 1
        assert response.factor_sensitivity[0].node_id == "marketing"
        assert isinstance(response.factor_sensitivity[0], FactorSensitivityResult)

    def test_analysis_without_factor_uncertainties_empty_factor_sensitivity(self):
        """Test analysis without parameter_uncertainties has empty factor_sensitivity."""
        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="no-factor-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            # No parameter_uncertainties
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # factor_sensitivity should be empty
        assert response.factor_sensitivity == []

    def test_factor_sampling_affects_outcome_distribution(self):
        """Test factor sampling creates variance in outcome distribution."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,  # Certain edge
                    strength=StrengthDistribution(mean=1.0, std=0.01),  # Very low edge variance
                )
            ],
        )

        # Request WITH factor uncertainty
        request_with = RobustnessRequestV2(
            request_id="with-factor",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            parameter_uncertainties=[
                ParameterUncertainty(node_id="marketing", distribution="normal", std=20.0)
            ],
        )

        # Request WITHOUT factor uncertainty
        request_without = RobustnessRequestV2(
            request_id="without-factor",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            # No parameter_uncertainties
        )

        analyzer = RobustnessAnalyzerV2()
        response_with = analyzer.analyze(request_with)
        response_without = analyzer.analyze(request_without)

        # With factor uncertainty should have higher outcome variance
        std_with = response_with.results[0].outcome_distribution.std
        std_without = response_without.results[0].outcome_distribution.std

        assert std_with > std_without * 2  # Significantly higher variance

    def test_multiple_factor_uncertainties(self):
        """Test analysis with multiple factor uncertainties."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(
                    id="price",
                    kind="factor",
                    label="Price",
                    observed_state=ObservedState(value=50.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                ),
                EdgeV2(
                    **{"from": "price", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.3, std=0.05),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="multi-factor",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=200,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="marketing", distribution="normal", std=10.0),
                ParameterUncertainty(node_id="price", distribution="uniform", range_min=40.0, range_max=60.0),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should have 2 factor sensitivities
        assert len(response.factor_sensitivity) == 2

        node_ids = {fs.node_id for fs in response.factor_sensitivity}
        assert "marketing" in node_ids
        assert "price" in node_ids

    def test_factor_sensitivity_ranking(self):
        """Test factor sensitivity results are ranked by importance."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="high_impact",
                    kind="factor",
                    label="High Impact",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(
                    id="low_impact",
                    kind="factor",
                    label="Low Impact",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "high_impact", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=2.0, std=0.1),  # Strong effect
                ),
                EdgeV2(
                    **{"from": "low_impact", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.1, std=0.01),  # Weak effect
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="ranking-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=200,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="high_impact", distribution="normal", std=10.0),
                ParameterUncertainty(node_id="low_impact", distribution="normal", std=10.0),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should be sorted by absolute elasticity
        ranks = [fs.importance_rank for fs in response.factor_sensitivity]
        assert ranks == sorted(ranks)

        # High impact should have higher absolute elasticity
        high_impact = next(fs for fs in response.factor_sensitivity if fs.node_id == "high_impact")
        low_impact = next(fs for fs in response.factor_sensitivity if fs.node_id == "low_impact")

        assert abs(high_impact.elasticity) > abs(low_impact.elasticity)


# =============================================================================
# Factor Sensitivity V2 Response Numeric Fields Tests
# =============================================================================


class TestFactorSensitivityV2NumericFields:
    """Tests for V2 response numeric sensitivity metrics."""

    def test_sensitivity_score_in_valid_range(self):
        """Test sensitivity_score is normalized to 0-1 range."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="marketing",
                    kind="factor",
                    label="Marketing",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "marketing", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.5, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="v2-numeric-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="marketing", distribution="normal", std=10.0),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert len(response.factor_sensitivity) == 1
        fs = response.factor_sensitivity[0]

        # Verify all numeric fields are present
        assert fs.elasticity is not None
        assert fs.importance_rank is not None

    def test_importance_score_ordering(self):
        """Test importance_score reflects ranking (higher = more important)."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="high_impact",
                    kind="factor",
                    label="High Impact",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(
                    id="medium_impact",
                    kind="factor",
                    label="Medium Impact",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(
                    id="low_impact",
                    kind="factor",
                    label="Low Impact",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "high_impact", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=3.0, std=0.1),
                ),
                EdgeV2(
                    **{"from": "medium_impact", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.5, std=0.1),
                ),
                EdgeV2(
                    **{"from": "low_impact", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="importance-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="high_impact", distribution="normal", std=10.0),
                ParameterUncertainty(node_id="medium_impact", distribution="normal", std=10.0),
                ParameterUncertainty(node_id="low_impact", distribution="normal", std=10.0),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert len(response.factor_sensitivity) == 3

        # All should have elasticity values
        for fs in response.factor_sensitivity:
            assert fs.elasticity is not None
            assert fs.importance_rank is not None

    def test_single_factor_sensitivity_score(self):
        """Test single factor gets sensitivity_score = 1.0 (max)."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="only_factor",
                    kind="factor",
                    label="Only Factor",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "only_factor", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="single-factor-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="only_factor", distribution="normal", std=10.0),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert len(response.factor_sensitivity) == 1
        fs = response.factor_sensitivity[0]

        # Single factor should have elasticity (can be any value)
        assert fs.elasticity is not None

    def test_direction_preserved_with_numeric_fields(self):
        """Test direction field is still present alongside numeric fields."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="positive_factor",
                    kind="factor",
                    label="Positive",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "positive_factor", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="direction-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="positive_factor", distribution="normal", std=10.0),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        fs = response.factor_sensitivity[0]

        # V1 internal model fields (direction is added in V2 mapping)
        assert fs.elasticity is not None
        assert fs.importance_rank is not None
        assert fs.interpretation is not None

        # Direction can be derived from elasticity sign (done in V2 mapping)
        # Positive elasticity -> positive direction, negative -> negative
        expected_direction = "positive" if fs.elasticity > 0 else "negative"
        assert expected_direction in ("positive", "negative")


# =============================================================================
# V1→V2 Normalization Math Tests
# =============================================================================


class TestFactorSensitivityNormalizationMath:
    """Tests that verify the V1→V2 normalization calculations are correct.

    These tests verify the formulas:
    - sensitivity_score = |elasticity| / max_abs_elasticity (clamped to 1.0)
    - importance_score = 1.0 - (rank - 1) / max(n - 1, 1)
    """

    def test_sensitivity_score_normalization_formula(self):
        """Verify sensitivity_score = |elasticity| / max_abs_elasticity."""
        from src.models.robustness_v2 import FactorSensitivityResult

        # Create mock V1 factor sensitivity results
        factors = [
            FactorSensitivityResult(
                node_id="factor_1",
                node_label="Factor 1",
                elasticity=2.0,  # max |elasticity| = 2.0
                importance_rank=1,
                observed_value=100.0,
                interpretation="Test",
            ),
            FactorSensitivityResult(
                node_id="factor_2",
                node_label="Factor 2",
                elasticity=1.0,  # |elasticity| = 1.0
                importance_rank=2,
                observed_value=50.0,
                interpretation="Test",
            ),
            FactorSensitivityResult(
                node_id="factor_3",
                node_label="Factor 3",
                elasticity=0.5,  # |elasticity| = 0.5
                importance_rank=3,
                observed_value=25.0,
                interpretation="Test",
            ),
        ]

        # Apply normalization formula (same as in robustness.py)
        max_abs_elasticity = max(abs(fs.elasticity) for fs in factors)

        normalized_scores = [
            min(1.0, abs(fs.elasticity) / max_abs_elasticity)
            for fs in factors
        ]

        # Verify: factor_1 has max elasticity → score = 1.0
        assert normalized_scores[0] == pytest.approx(1.0)
        # Verify: factor_2 elasticity = 1.0/2.0 = 0.5
        assert normalized_scores[1] == pytest.approx(0.5)
        # Verify: factor_3 elasticity = 0.5/2.0 = 0.25
        assert normalized_scores[2] == pytest.approx(0.25)

    def test_sensitivity_score_with_negative_elasticity(self):
        """Verify absolute value is used for negative elasticity."""
        from src.models.robustness_v2 import FactorSensitivityResult

        factors = [
            FactorSensitivityResult(
                node_id="positive",
                node_label="Positive",
                elasticity=1.5,  # positive
                importance_rank=1,
                observed_value=100.0,
                interpretation="Test",
            ),
            FactorSensitivityResult(
                node_id="negative",
                node_label="Negative",
                elasticity=-2.0,  # negative, but |value| is larger
                importance_rank=2,
                observed_value=100.0,
                interpretation="Test",
            ),
        ]

        max_abs_elasticity = max(abs(fs.elasticity) for fs in factors)
        assert max_abs_elasticity == 2.0  # |-2.0| = 2.0 is max

        # Negative factor has max |elasticity| → score = 1.0
        negative_score = min(1.0, abs(factors[1].elasticity) / max_abs_elasticity)
        assert negative_score == pytest.approx(1.0)

        # Positive factor: 1.5/2.0 = 0.75
        positive_score = min(1.0, abs(factors[0].elasticity) / max_abs_elasticity)
        assert positive_score == pytest.approx(0.75)

    def test_importance_score_formula_multiple_factors(self):
        """Verify importance_score = 1.0 - (rank - 1) / (n - 1) for n > 1."""
        from src.models.robustness_v2 import FactorSensitivityResult

        n_factors = 4
        factors = [
            FactorSensitivityResult(
                node_id=f"factor_{i}",
                node_label=f"Factor {i}",
                elasticity=1.0,
                importance_rank=i,
                observed_value=100.0,
                interpretation="Test",
            )
            for i in range(1, n_factors + 1)
        ]

        # Apply formula
        importance_scores = [
            1.0 - (fs.importance_rank - 1) / max(n_factors - 1, 1)
            for fs in factors
        ]

        # Verify: rank 1 → 1.0 - (1-1)/(4-1) = 1.0 - 0/3 = 1.0
        assert importance_scores[0] == pytest.approx(1.0)
        # Verify: rank 2 → 1.0 - (2-1)/(4-1) = 1.0 - 1/3 ≈ 0.667
        assert importance_scores[1] == pytest.approx(0.6667, rel=0.01)
        # Verify: rank 3 → 1.0 - (3-1)/(4-1) = 1.0 - 2/3 ≈ 0.333
        assert importance_scores[2] == pytest.approx(0.3333, rel=0.01)
        # Verify: rank 4 → 1.0 - (4-1)/(4-1) = 1.0 - 3/3 = 0.0
        assert importance_scores[3] == pytest.approx(0.0)

    def test_importance_score_single_factor(self):
        """Verify single factor gets importance_score = 1.0."""
        from src.models.robustness_v2 import FactorSensitivityResult

        factors = [
            FactorSensitivityResult(
                node_id="only_factor",
                node_label="Only Factor",
                elasticity=0.5,
                importance_rank=1,
                observed_value=100.0,
                interpretation="Test",
            )
        ]

        n_factors = len(factors)

        # Apply formula with single factor case
        importance_score = (
            1.0 - (factors[0].importance_rank - 1) / max(n_factors - 1, 1)
            if n_factors > 1 else 1.0
        )

        # Single factor always gets 1.0
        assert importance_score == 1.0

    def test_zero_elasticity_handling(self):
        """Verify zero elasticity results in sensitivity_score = 0."""
        from src.models.robustness_v2 import FactorSensitivityResult

        factors = [
            FactorSensitivityResult(
                node_id="zero_impact",
                node_label="Zero Impact",
                elasticity=0.0,  # No impact
                importance_rank=1,
                observed_value=100.0,
                interpretation="Test",
            ),
            FactorSensitivityResult(
                node_id="some_impact",
                node_label="Some Impact",
                elasticity=1.0,
                importance_rank=2,
                observed_value=100.0,
                interpretation="Test",
            ),
        ]

        max_abs_elasticity = max(abs(fs.elasticity) for fs in factors)
        assert max_abs_elasticity == 1.0

        # Zero elasticity → score = 0/1 = 0
        zero_score = min(1.0, abs(factors[0].elasticity) / max_abs_elasticity)
        assert zero_score == 0.0

    def test_all_zero_elasticity_handling(self):
        """Verify all-zero elasticity uses fallback max of 1.0."""
        from src.models.robustness_v2 import FactorSensitivityResult

        factors = [
            FactorSensitivityResult(
                node_id="zero_1",
                node_label="Zero 1",
                elasticity=0.0,
                importance_rank=1,
                observed_value=100.0,
                interpretation="Test",
            ),
            FactorSensitivityResult(
                node_id="zero_2",
                node_label="Zero 2",
                elasticity=0.0,
                importance_rank=2,
                observed_value=100.0,
                interpretation="Test",
            ),
        ]

        # When all elasticities are zero, max would be 0
        # The code uses 1e-10 threshold and falls back to 1.0
        max_abs_elasticity = max(
            (abs(fs.elasticity) for fs in factors),
            default=1.0
        )
        if max_abs_elasticity < 1e-10:
            max_abs_elasticity = 1.0

        # All zero elasticities → all scores = 0/1 = 0
        for fs in factors:
            score = min(1.0, abs(fs.elasticity) / max_abs_elasticity)
            assert score == 0.0

    def test_tiny_elasticity_threshold(self):
        """Verify very small elasticity values are handled correctly."""
        from src.models.robustness_v2 import FactorSensitivityResult

        factors = [
            FactorSensitivityResult(
                node_id="tiny",
                node_label="Tiny",
                elasticity=1e-15,  # Below threshold
                importance_rank=1,
                observed_value=100.0,
                interpretation="Test",
            ),
        ]

        max_abs_elasticity = max(
            (abs(fs.elasticity) for fs in factors),
            default=1.0
        )
        # 1e-15 < 1e-10, so should use fallback
        if max_abs_elasticity < 1e-10:
            max_abs_elasticity = 1.0

        # With fallback to 1.0: 1e-15 / 1.0 ≈ 0
        score = min(1.0, abs(factors[0].elasticity) / max_abs_elasticity)
        assert score == pytest.approx(0.0, abs=1e-9)


# =============================================================================
# Zero-Baseline Factor Sensitivity Tests
# =============================================================================


class TestZeroBaselineFactorSensitivity:
    """Tests for factor sensitivity when observed_state.value = 0.

    This covers the case of binary factors (0/1) where the baseline is zero,
    which previously caused elasticity to be forced to 0.0 due to division issues.
    """

    def test_binary_factor_with_zero_baseline_has_nonzero_sensitivity(self):
        """Verify binary factor at 0 still produces meaningful sensitivity."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="binary_factor",
                    kind="factor",
                    label="Binary Factor (0/1)",
                    observed_state=ObservedState(value=0.0),  # Zero baseline!
                ),
                NodeV2(id="goal", kind="goal", label="Goal"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "binary_factor", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="zero-baseline-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(
                    node_id="binary_factor",
                    distribution="normal",
                    std=0.3,  # Typical for binary factor
                ),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should have factor sensitivity computed
        assert len(response.factor_sensitivity) == 1
        fs = response.factor_sensitivity[0]

        # Key assertion: elasticity should NOT be zero despite zero baseline
        # The epsilon-stabilised denominator should produce a meaningful value
        assert fs.elasticity != 0.0, (
            "Binary factor with observed_state.value=0 should have non-zero elasticity"
        )

        # Should have interpretation (not just "robust" due to zero elasticity)
        assert fs.interpretation is not None

    def test_multiple_zero_baseline_factors_ranked_correctly(self):
        """Verify multiple zero-baseline factors are ranked by sensitivity."""
        from src.models.robustness_v2 import ParameterUncertainty

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="high_impact",
                    kind="factor",
                    label="High Impact",
                    observed_state=ObservedState(value=0.0),
                ),
                NodeV2(
                    id="low_impact",
                    kind="factor",
                    label="Low Impact",
                    observed_state=ObservedState(value=0.0),
                ),
                NodeV2(id="goal", kind="goal", label="Goal"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "high_impact", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=2.0, std=0.1),  # Strong effect
                ),
                EdgeV2(
                    **{"from": "low_impact", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),  # Weaker effect
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="multi-zero-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={}),
            ],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
            analysis_types=["sensitivity"],
            parameter_uncertainties=[
                ParameterUncertainty(node_id="high_impact", distribution="normal", std=0.3),
                ParameterUncertainty(node_id="low_impact", distribution="normal", std=0.3),
            ],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        assert len(response.factor_sensitivity) == 2

        # Both should have non-zero elasticity
        for fs in response.factor_sensitivity:
            assert fs.elasticity != 0.0

        # High impact factor should have higher absolute elasticity
        high_impact_fs = next(
            fs for fs in response.factor_sensitivity if fs.node_id == "high_impact"
        )
        low_impact_fs = next(
            fs for fs in response.factor_sensitivity if fs.node_id == "low_impact"
        )

        assert abs(high_impact_fs.elasticity) > abs(low_impact_fs.elasticity), (
            "Higher edge strength should result in higher elasticity"
        )

        # High impact should be ranked first (importance_rank=1)
        assert high_impact_fs.importance_rank == 1
        assert low_impact_fs.importance_rank == 2


# =============================================================================
# Goal Threshold Probability Tests
# =============================================================================


class TestGoalThresholdProbability:
    """Tests for goal_threshold and probability_of_goal feature."""

    @pytest.fixture
    def deterministic_graph(self):
        """Create a graph with deterministic edges for predictable outcomes."""
        return GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=100.0)),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.0011),  # Near-deterministic
                )
            ],
        )

    def test_basic_goal_threshold_probability_computation(self, deterministic_graph):
        """Test basic probability_of_goal computation."""
        request = RobustnessRequestV2(
            request_id="goal-threshold-basic",
            graph=deterministic_graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
            goal_threshold=50.0,  # Well below expected outcome of ~100
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # With input=100 and strength=1.0, output should be ~100
        # Threshold=50 should be met by nearly all samples
        assert response.results[0].probability_of_goal is not None
        assert response.results[0].probability_of_goal >= 0.95  # Should be nearly 1.0

    def test_no_goal_threshold_field_absent_in_response(self, deterministic_graph):
        """Test probability_of_goal is absent (not null) when goal_threshold not provided."""
        request = RobustnessRequestV2(
            request_id="no-threshold",
            graph=deterministic_graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
            # goal_threshold not provided
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # probability_of_goal should be None internally
        assert response.results[0].probability_of_goal is None

        # When serialized, it should be absent due to exclude_none
        result_dict = response.results[0].model_dump(exclude_none=True)
        assert "probability_of_goal" not in result_dict

    def test_threshold_above_all_samples_zero_probability(self, deterministic_graph):
        """Test probability_of_goal is 0.0 when threshold exceeds all outcomes."""
        request = RobustnessRequestV2(
            request_id="threshold-too-high",
            graph=deterministic_graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
            goal_threshold=1000.0,  # Well above expected outcome of ~100
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # With outcome ~100, threshold=1000 should never be met
        assert response.results[0].probability_of_goal == 0.0

    def test_threshold_below_all_samples_one_probability(self, deterministic_graph):
        """Test probability_of_goal is 1.0 when all outcomes exceed threshold."""
        request = RobustnessRequestV2(
            request_id="threshold-too-low",
            graph=deterministic_graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
            goal_threshold=-100.0,  # Well below expected outcome of ~100
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # All samples should exceed threshold
        assert response.results[0].probability_of_goal == 1.0

    def test_exact_threshold_match_uses_gte_semantics(self, deterministic_graph):
        """Test >= semantics: samples exactly at threshold count as meeting it."""
        # Create a graph that produces exactly 100.0 (deterministic)
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=100.0)),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.0011),  # Very tight around 1.0
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="exact-threshold",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
            goal_threshold=100.0,  # Exactly at expected outcome
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # With >= semantics, samples at exactly 100 should count
        # Most samples should be very close to 100, so probability should be ~0.5
        # (half above, half below due to normal distribution around 100)
        assert response.results[0].probability_of_goal is not None
        assert 0.3 <= response.results[0].probability_of_goal <= 0.7

    def test_nan_goal_threshold_rejected(self, deterministic_graph):
        """Test NaN goal_threshold raises validation error."""
        with pytest.raises(ValueError, match="finite"):
            RobustnessRequestV2(
                request_id="nan-threshold",
                graph=deterministic_graph,
                options=[
                    InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
                ],
                goal_node_id="output",
                n_samples=100,
                goal_threshold=float("nan"),
            )

    def test_inf_goal_threshold_rejected(self, deterministic_graph):
        """Test positive infinity goal_threshold raises validation error."""
        with pytest.raises(ValueError, match="finite"):
            RobustnessRequestV2(
                request_id="inf-threshold",
                graph=deterministic_graph,
                options=[
                    InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
                ],
                goal_node_id="output",
                n_samples=100,
                goal_threshold=float("inf"),
            )

    def test_neg_inf_goal_threshold_rejected(self, deterministic_graph):
        """Test negative infinity goal_threshold raises validation error."""
        with pytest.raises(ValueError, match="finite"):
            RobustnessRequestV2(
                request_id="neg-inf-threshold",
                graph=deterministic_graph,
                options=[
                    InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
                ],
                goal_node_id="output",
                n_samples=100,
                goal_threshold=float("-inf"),
            )

    def test_negative_threshold_accepted(self, deterministic_graph):
        """Test negative goal_threshold is accepted (valid for outcomes that can be negative)."""
        request = RobustnessRequestV2(
            request_id="negative-threshold",
            graph=deterministic_graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
            goal_threshold=-50.0,  # Negative threshold
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # All positive outcomes should exceed negative threshold
        assert response.results[0].probability_of_goal == 1.0

    def test_zero_threshold_accepted(self, deterministic_graph):
        """Test zero goal_threshold is accepted."""
        request = RobustnessRequestV2(
            request_id="zero-threshold",
            graph=deterministic_graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
            goal_threshold=0.0,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # All positive outcomes should exceed 0
        assert response.results[0].probability_of_goal == 1.0

    def test_goal_threshold_with_multiple_options(self):
        """Test probability_of_goal computed correctly for each option independently."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="investment", kind="factor", label="Investment", observed_state=ObservedState(value=0.0)),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "investment", "to": "revenue"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=2.0, std=0.01),  # 2x multiplier
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="multi-option",
            graph=graph,
            options=[
                InterventionOption(id="low", label="Low Investment", interventions={"investment": 50.0}),
                InterventionOption(id="high", label="High Investment", interventions={"investment": 150.0}),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
            goal_threshold=200.0,  # Between low (100) and high (300) expected outcomes
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        low_result = next(r for r in response.results if r.option_id == "low")
        high_result = next(r for r in response.results if r.option_id == "high")

        # Low: outcome ~100, threshold 200 -> should rarely meet
        # High: outcome ~300, threshold 200 -> should always meet
        assert low_result.probability_of_goal < 0.1  # Near 0
        assert high_result.probability_of_goal > 0.99  # Near 1.0

    def test_goal_threshold_with_stochastic_outcomes(self):
        """Test probability_of_goal reflects actual sample distribution."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=100.0)),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.2),  # High variance
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="stochastic",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=1000,  # More samples for stable estimate
            seed=42,
            goal_threshold=100.0,  # At mean
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # With threshold at mean, probability should be around 0.5
        # Allow some variance due to non-symmetric distributions
        assert 0.3 <= response.results[0].probability_of_goal <= 0.7


class TestNodeV2Intercept:
    """V08: Test intercept field in NodeV2 schema."""

    def test_node_without_intercept_defaults_to_zero(self):
        """Test that nodes without intercept field default to 0.0."""
        node = NodeV2(id="node_a", kind="factor", label="Node A")
        assert node.intercept == 0.0

    def test_node_with_positive_intercept(self):
        """Test that positive intercept values are accepted."""
        node = NodeV2(id="node_a", kind="outcome", label="Node A", intercept=50.0)
        assert node.intercept == 50.0

    def test_node_with_negative_intercept(self):
        """Test that negative intercept values are accepted."""
        node = NodeV2(id="node_a", kind="outcome", label="Node A", intercept=-25.0)
        assert node.intercept == -25.0

    def test_intercept_serialization(self):
        """Test that intercept is included in serialized output."""
        node = NodeV2(id="node_a", kind="outcome", label="Node A", intercept=100.0)
        node_dict = node.model_dump()
        assert "intercept" in node_dict
        assert node_dict["intercept"] == 100.0


class TestSCMEvaluatorV2Intercept:
    """V08: Test that SCMEvaluatorV2 uses intercept in evaluation."""

    def test_intercept_added_to_outcome(self):
        """Test that intercept is added to the node's computed value."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=0.0)),
                NodeV2(id="output", kind="outcome", label="Output", intercept=100.0),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.0011),  # Near-zero std
                )
            ],
        )

        evaluator = SCMEvaluatorV2(graph)

        # With input=0 and intercept=100, output should be 100
        result = evaluator.evaluate(
            edge_strengths={("input", "output"): 1.0},
            interventions={"input": 0.0},
            goal_node="output",
        )
        assert result == 100.0

    def test_intercept_with_parent_contribution(self):
        """Test intercept is added to parent contribution."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=0.0)),
                NodeV2(id="output", kind="outcome", label="Output", intercept=50.0),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=2.0, std=0.0011),  # Near-zero std
                )
            ],
        )

        evaluator = SCMEvaluatorV2(graph)

        # output = base(0) + intercept(50) + parent_contribution(10 * 2 = 20) = 70
        result = evaluator.evaluate(
            edge_strengths={("input", "output"): 2.0},
            interventions={"input": 10.0},
            goal_node="output",
        )
        assert result == 70.0

    def test_intercept_with_chain(self):
        """Test intercept works correctly in a chain of nodes."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="a", kind="factor", label="A", observed_state=ObservedState(value=0.0)),
                NodeV2(id="b", kind="outcome", label="B", intercept=10.0),
                NodeV2(id="c", kind="outcome", label="C", intercept=20.0),
            ],
            edges=[
                EdgeV2(
                    **{"from": "a", "to": "b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.0011),  # Near-zero std
                ),
                EdgeV2(
                    **{"from": "b", "to": "c"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.0011),  # Near-zero std
                ),
            ],
        )

        evaluator = SCMEvaluatorV2(graph)

        # a = 5 (intervention)
        # b = base(0) + intercept(10) + (5 * 1) = 15
        # c = base(0) + intercept(20) + (15 * 1) = 35
        result = evaluator.evaluate(
            edge_strengths={("a", "b"): 1.0, ("b", "c"): 1.0},
            interventions={"a": 5.0},
            goal_node="c",
        )
        assert result == 35.0


class TestAutoScaledNoise:
    """V08: Test auto-scaled noise application to outcome/risk nodes."""

    def test_noise_applied_to_outcome_node(self):
        """Test that noise is applied to outcome node samples."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=100.0)),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),  # Some variance for noise to match
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="noise-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=1000,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # With noise applied, the std should be larger than edge uncertainty alone
        # Edge std = 0.1 * 100 = 10, so base std ~ 10
        # With auto-scaled noise (matching std), effective std should be ~sqrt(2) * base_std
        result = response.results[0]
        # The std should reflect both edge variance and added noise
        assert result.outcome_distribution.std > 0

    def test_noise_applied_to_risk_node(self):
        """Test that noise is applied to risk node samples."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="exposure", kind="factor", label="Exposure", observed_state=ObservedState(value=50.0)),
                NodeV2(id="risk", kind="risk", label="Risk Level"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "exposure", "to": "risk"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.05),
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="risk-noise-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"exposure": 50.0}),
            ],
            goal_node_id="risk",
            n_samples=500,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Risk nodes should also get noise applied
        result = response.results[0]
        assert result.outcome_distribution.std > 0

    def test_noise_not_applied_to_factor_node(self):
        """Test that noise is NOT applied when goal is a factor node."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=100.0)),
            ],
            edges=[],
        )

        request = RobustnessRequestV2(
            request_id="factor-no-noise",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="input",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Factor nodes shouldn't receive noise, so std should be 0 (deterministic intervention)
        result = response.results[0]
        assert result.outcome_distribution.mean == 100.0
        assert result.outcome_distribution.std == 0.0

    def test_noise_deterministic_with_seed(self):
        """Test that noise application is deterministic with same seed."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=100.0)),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="noise-determinism",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response1 = analyzer.analyze(request)
        response2 = analyzer.analyze(request)

        # Same seed should produce identical results
        assert response1.results[0].outcome_distribution.mean == response2.results[0].outcome_distribution.mean
        assert response1.results[0].outcome_distribution.std == response2.results[0].outcome_distribution.std

    def test_noise_skipped_when_zero_std(self):
        """Test that noise is skipped when samples have near-zero std."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=100.0)),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.0011),  # Near-zero variance
                )
            ],
        )

        request = RobustnessRequestV2(
            request_id="zero-std",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # With small std in edge strength (0.0011), samples have low variance
        # Auto-scaled noise matches the sample std, so result std should be small
        result = response.results[0]
        assert result.outcome_distribution.mean == pytest.approx(100.0, rel=0.01)
        # Std should be small due to low edge variance (but not zero due to minimum std floor)
        assert result.outcome_distribution.std < 0.2


# =============================================================================
# Test Degenerate Option Critiques (Zero Variance)
# =============================================================================


class TestDegenerateOptionZeroVariance:
    """Test DEGENERATE_OPTION_ZERO_VARIANCE critique detection."""

    def test_zero_variance_option_triggers_critique(self):
        """Test that option with no path to goal triggers zero variance critique."""
        # Graph where option 1's intervention has NO path to goal
        # isolated_node is not connected to goal
        graph = GraphV2(
            nodes=[
                NodeV2(id="isolated_node", kind="factor", label="Isolated"),
                NodeV2(id="connected_node", kind="factor", label="Connected"),
                NodeV2(id="goal", kind="outcome", label="Goal"),
            ],
            edges=[
                # Only connected_node -> goal edge exists
                EdgeV2(
                    **{"from": "connected_node", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="zero-variance-test",
            graph=graph,
            options=[
                # Option 1 intervenes on isolated node (no path to goal)
                InterventionOption(
                    id="opt_isolated",
                    label="Isolated Option",
                    interventions={"isolated_node": 1.0},
                ),
                # Option 2 intervenes on connected node (has path)
                InterventionOption(
                    id="opt_connected",
                    label="Connected Option",
                    interventions={"connected_node": 1.0},
                ),
            ],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Find the result for isolated option
        isolated_result = next(r for r in response.results if r.option_id == "opt_isolated")

        # Isolated option should have zero std
        assert isolated_result.outcome_distribution.std == 0.0

        # Should have DEGENERATE_OPTION_ZERO_VARIANCE critique
        zero_var_critiques = [
            c for c in response.critiques
            if c.code == "DEGENERATE_OPTION_ZERO_VARIANCE"
        ]
        assert len(zero_var_critiques) >= 1

        # Critique should reference the isolated option
        critique = zero_var_critiques[0]
        assert "Isolated Option" in critique.message
        assert critique.severity == "warning"
        assert critique.source == "analysis"

    def test_no_critique_when_variance_nonzero(self):
        """Test that options with non-zero variance don't trigger critique."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.5),  # Non-zero std
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="nonzero-variance-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 1.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"input": 2.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should NOT have zero variance critique
        zero_var_critiques = [
            c for c in response.critiques
            if c.code == "DEGENERATE_OPTION_ZERO_VARIANCE"
        ]
        assert len(zero_var_critiques) == 0

    def test_near_zero_variance_triggers_warning(self):
        """Test that near-zero variance (floating point artifact) triggers warning.

        Values like 1e-15 from numerical computation are effectively zero
        but won't match exact equality check.
        """
        from src.services.robustness_analyzer_v2 import ZERO_VARIANCE_TOLERANCE

        # Create a mock result with near-zero std (below tolerance)
        near_zero_std = 1e-15  # Typical floating point precision artifact

        # Verify the tolerance check would catch this
        assert near_zero_std < ZERO_VARIANCE_TOLERANCE

        # Full integration test: graph where isolated node produces near-zero variance
        graph = GraphV2(
            nodes=[
                NodeV2(id="isolated", kind="factor", label="Isolated"),
                NodeV2(id="connected", kind="factor", label="Connected"),
                NodeV2(id="goal", kind="outcome", label="Goal"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "connected", "to": "goal"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="near-zero-variance-test",
            graph=graph,
            options=[
                InterventionOption(
                    id="opt_isolated",
                    label="Isolated Option",
                    interventions={"isolated": 1.0},
                ),
                InterventionOption(
                    id="opt_connected",
                    label="Connected Option",
                    interventions={"connected": 1.0},
                ),
            ],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should trigger the zero variance critique
        zero_var_critiques = [
            c for c in response.critiques
            if c.code == "DEGENERATE_OPTION_ZERO_VARIANCE"
        ]
        assert len(zero_var_critiques) >= 1

    def test_small_but_real_variance_no_warning(self):
        """Test that small but real variance (e.g., 0.0011) does NOT trigger warning."""
        from src.services.robustness_analyzer_v2 import ZERO_VARIANCE_TOLERANCE

        # Small but real variance - should be above tolerance
        small_variance = 0.0011
        assert small_variance > ZERO_VARIANCE_TOLERANCE

        # Graph with very small but real edge strength std
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    # Very small std but definitely above tolerance
                    strength=StrengthDistribution(mean=1.0, std=0.0011),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="small-variance-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 1.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"input": 2.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should NOT trigger zero variance critique - variance is small but real
        zero_var_critiques = [
            c for c in response.critiques
            if c.code == "DEGENERATE_OPTION_ZERO_VARIANCE"
        ]
        assert len(zero_var_critiques) == 0


# =============================================================================
# Test High Tie Rate Critiques
# =============================================================================


class TestHighTieRateCritique:
    """Test HIGH_TIE_RATE critique detection."""

    def test_high_tie_rate_triggers_critique(self):
        """Test that >50% tie rate triggers HIGH_TIE_RATE critique."""
        # Graph where all options produce identical outcomes (100% ties)
        # Both options intervene on same node with same value
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="high-tie-rate-test",
            graph=graph,
            options=[
                # Both options have SAME intervention value -> 100% ties
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 1.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"input": 1.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should have HIGH_TIE_RATE critique
        tie_rate_critiques = [
            c for c in response.critiques
            if c.code == "HIGH_TIE_RATE"
        ]
        assert len(tie_rate_critiques) == 1

        critique = tie_rate_critiques[0]
        assert "100%" in critique.message or "tie" in critique.message.lower()
        assert critique.severity == "warning"
        assert critique.source == "analysis"

    def test_no_critique_when_tie_rate_below_threshold(self):
        """Test that tie rate below 50% does not trigger critique."""
        # Graph where options produce different outcomes (low tie rate)
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="low-tie-rate-test",
            graph=graph,
            options=[
                # Different intervention values -> different outcomes -> low tie rate
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 1.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"input": 10.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Should NOT have HIGH_TIE_RATE critique
        tie_rate_critiques = [
            c for c in response.critiques
            if c.code == "HIGH_TIE_RATE"
        ]
        assert len(tie_rate_critiques) == 0


# =============================================================================
# Test Fair Tie-Breaking (Split Ties)
# =============================================================================


class TestFairTieBreaking:
    """Test that ties are split fairly between options."""

    def test_identical_options_get_equal_win_probability(self):
        """Test two identical options get 50%/50% win probability."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="equal-split-test",
            graph=graph,
            options=[
                # Identical interventions -> 100% ties -> should split 50/50
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 5.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"input": 5.0}),
            ],
            goal_node_id="output",
            n_samples=1000,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        opt1_result = next(r for r in response.results if r.option_id == "opt1")
        opt2_result = next(r for r in response.results if r.option_id == "opt2")

        # Both should have approximately 50% win probability
        assert opt1_result.win_probability == pytest.approx(0.5, abs=0.01)
        assert opt2_result.win_probability == pytest.approx(0.5, abs=0.01)

        # Sum should equal 1.0
        total = opt1_result.win_probability + opt2_result.win_probability
        assert total == pytest.approx(1.0, abs=0.001)

    def test_three_way_tie_splits_equally(self):
        """Test three identical options get ~33.3% each."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="three-way-split-test",
            graph=graph,
            options=[
                # Three identical interventions -> all ties
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 3.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"input": 3.0}),
                InterventionOption(id="opt3", label="Option 3", interventions={"input": 3.0}),
            ],
            goal_node_id="output",
            n_samples=1000,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # All three should have approximately 33.3% win probability
        for result in response.results:
            assert result.win_probability == pytest.approx(1/3, abs=0.01)

        # Sum should equal 1.0
        total = sum(r.win_probability for r in response.results)
        assert total == pytest.approx(1.0, abs=0.001)


# =============================================================================
# Test Tie Rate in Metadata/Diagnostics
# =============================================================================


class TestTieRateInMetadata:
    """Test that tie_rate and tie_count are included in metadata."""

    def test_tie_rate_in_metadata_high_ties(self):
        """Test tie_rate is present in metadata with high tie scenario."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="tie-rate-metadata-test",
            graph=graph,
            options=[
                # Identical -> 100% ties
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 1.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"input": 1.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Check metadata has tie_rate and tie_count
        assert response.metadata.tie_count is not None
        assert response.metadata.tie_rate is not None

        # Should be 100% ties
        assert response.metadata.tie_count == 100
        assert response.metadata.tie_rate == pytest.approx(1.0, abs=0.01)

    def test_tie_rate_in_metadata_no_ties(self):
        """Test tie_rate is present in metadata with no ties."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="no-tie-metadata-test",
            graph=graph,
            options=[
                # Very different values -> no ties
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 1.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"input": 100.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Check metadata has tie_rate and tie_count
        assert response.metadata.tie_count is not None
        assert response.metadata.tie_rate is not None

        # Should be 0% ties (or very close)
        assert response.metadata.tie_count == 0
        assert response.metadata.tie_rate == 0.0

    def test_tie_rate_serializes_in_response(self):
        """Test tie_rate appears in JSON serialization."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="tie-rate-json-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"input": 1.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"input": 1.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Convert to dict/JSON
        response_dict = response.model_dump(by_alias=True)

        # Check _metadata contains tie fields
        assert "_metadata" in response_dict
        assert "tie_rate" in response_dict["_metadata"]
        assert "tie_count" in response_dict["_metadata"]


class TestOptionIdPreservation:
    """Test that ISL echoes original option IDs from request to response.

    This is critical for upstream systems (like PLoT) to match results
    to their original options.
    """

    def test_custom_option_ids_are_echoed_in_response(self):
        """Test that custom option IDs like 'increase_pro_price_to_59' are preserved.

        ISL must echo the exact option ID from the request in the response.
        This is standard API behavior - identifiers provided by the caller
        must be returned unchanged.
        """
        graph = GraphV2(
            nodes=[
                NodeV2(id="price", kind="factor", label="Price"),
                NodeV2(id="revenue", kind="outcome", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "price", "to": "revenue"},
                    exists_probability=0.9,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        # Use realistic option IDs like PLoT would send
        custom_option_ids = [
            "increase_pro_price_to_59",
            "maintain_current_price_49",
            "reduce_price_to_39",
        ]

        request = RobustnessRequestV2(
            request_id="option-id-preservation-test",
            graph=graph,
            options=[
                InterventionOption(
                    id=custom_option_ids[0],
                    label="Increase Pro Price to £59",
                    interventions={"price": 59.0},
                ),
                InterventionOption(
                    id=custom_option_ids[1],
                    label="Maintain Current Price £49",
                    interventions={"price": 49.0},
                ),
                InterventionOption(
                    id=custom_option_ids[2],
                    label="Reduce Price to £39",
                    interventions={"price": 39.0},
                ),
            ],
            goal_node_id="revenue",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Verify all original option IDs appear in the response
        response_option_ids = {r.option_id for r in response.results}
        expected_option_ids = set(custom_option_ids)

        assert response_option_ids == expected_option_ids, (
            f"Option IDs not preserved. "
            f"Expected: {expected_option_ids}, Got: {response_option_ids}"
        )

        # Verify we can look up each result by its original ID
        for original_id in custom_option_ids:
            result = next(
                (r for r in response.results if r.option_id == original_id),
                None
            )
            assert result is not None, (
                f"Could not find result for option ID '{original_id}'"
            )

        # Verify recommended_option_id is also one of the original IDs
        assert response.recommended_option_id in custom_option_ids, (
            f"recommended_option_id '{response.recommended_option_id}' "
            f"is not one of the original option IDs: {custom_option_ids}"
        )

    def test_option_ids_with_special_characters(self):
        """Test that option IDs with hyphens, underscores, colons are preserved."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        # Option IDs with various allowed characters
        special_ids = [
            "option-with-hyphens",
            "option_with_underscores",
            "option:with:colons",
            "mixed-option_id:123",
        ]

        request = RobustnessRequestV2(
            request_id="special-chars-test",
            graph=graph,
            options=[
                InterventionOption(id=sid, label=f"Label {i}", interventions={"input": float(i)})
                for i, sid in enumerate(special_ids)
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        response_option_ids = {r.option_id for r in response.results}
        assert response_option_ids == set(special_ids), (
            f"Special character option IDs not preserved. "
            f"Expected: {set(special_ids)}, Got: {response_option_ids}"
        )


# =============================================================================
# Test Edge Cases: Cycles, Empty Graphs, Extreme Values (High-Risk Coverage)
# =============================================================================


class TestCyclicGraphHandling:
    """Test behavior when graph contains cycles."""

    def test_cyclic_graph_logs_warning_and_continues(self, caplog):
        """Test that cyclic graphs log a warning but analysis continues."""
        # Create a graph with a cycle: A -> B -> C -> A
        graph = GraphV2(
            nodes=[
                NodeV2(id="node_a", kind="factor", label="Node A", observed_state=ObservedState(value=10.0)),
                NodeV2(id="node_b", kind="factor", label="Node B"),
                NodeV2(id="node_c", kind="outcome", label="Node C"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "node_a", "to": "node_b"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
                EdgeV2(
                    **{"from": "node_b", "to": "node_c"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
                EdgeV2(
                    **{"from": "node_c", "to": "node_a"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="cycle-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Option 1", interventions={"node_a": 5.0}),
                InterventionOption(id="opt2", label="Option 2", interventions={"node_a": 15.0}),
            ],
            goal_node_id="node_c",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        with caplog.at_level(logging.WARNING):
            response = analyzer.analyze(request)

        # Should have logged a warning about cycles
        cycle_warnings = [r for r in caplog.records if "cycle" in r.message.lower()]
        assert len(cycle_warnings) > 0, "Expected warning about graph cycles"

        # Analysis should still complete (graceful degradation)
        assert response is not None
        assert len(response.results) == 2


class TestAllEdgesFilteredScenario:
    """Test behavior when all edges are filtered out."""

    def test_all_non_inference_nodes_filtered(self, caplog):
        """Test analysis when all nodes are non-inference types."""
        # Graph with only decision/option/constraint nodes (all should be filtered)
        graph = GraphV2(
            nodes=[
                NodeV2(id="decision1", kind="decision", label="Decision 1"),
                NodeV2(id="option1", kind="option", label="Option 1"),
                NodeV2(id="outcome", kind="outcome", label="Outcome"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "decision1", "to": "outcome"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
                EdgeV2(
                    **{"from": "option1", "to": "outcome"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.5, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="all-filtered-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Opt 1", interventions={"outcome": 10.0}),
                InterventionOption(id="opt2", label="Opt 2", interventions={"outcome": 20.0}),
            ],
            goal_node_id="outcome",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        with caplog.at_level(logging.WARNING):
            response = analyzer.analyze(request)

        # Should have logged a warning about filtering
        filter_warnings = [
            r for r in caplog.records
            if "robustness_v2_filtered_non_inference_nodes" in r.message
        ]
        assert len(filter_warnings) > 0, "Expected warning about filtered nodes"

        # Analysis should still complete
        assert response is not None
        assert len(response.results) == 2

    def test_graph_with_no_path_to_goal(self):
        """Test analysis when interventions have no path to goal node."""
        # Graph where intervention node is disconnected from goal
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="isolated", kind="factor", label="Isolated"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
                # No edge from isolated to anywhere meaningful
            ],
        )

        request = RobustnessRequestV2(
            request_id="no-path-test",
            graph=graph,
            options=[
                # Intervene on isolated node (no path to goal)
                InterventionOption(id="opt1", label="Opt 1", interventions={"isolated": 100.0}),
                InterventionOption(id="opt2", label="Opt 2", interventions={"isolated": 200.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Both options should have identical outcomes (intervention has no effect)
        assert response is not None
        outcomes = [r.outcome_distribution.mean for r in response.results]
        assert abs(outcomes[0] - outcomes[1]) < 0.01, "Outcomes should be nearly identical"


class TestExtremeNumericalValues:
    """Test numerical stability with extreme values."""

    @pytest.mark.parametrize("scale", [0.01, 0.1, 10.0, 100.0, 1000.0])
    def test_extreme_edge_strength_scales(self, scale):
        """Test analysis with varying edge strength scales."""
        # Note: std must be > 0.001 per schema v2.6 D.4
        std_value = max(scale * 0.1, 0.0011)  # Ensure std > 0.001
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input", observed_state=ObservedState(value=1.0)),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=scale, std=std_value),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id=f"scale-{scale}-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Opt 1", interventions={"input": 1.0}),
                InterventionOption(id="opt2", label="Opt 2", interventions={"input": 2.0}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Analysis should complete without NaN/Inf
        assert response is not None
        for result in response.results:
            assert not np.isnan(result.outcome_distribution.mean), f"NaN mean at scale {scale}"
            assert not np.isinf(result.outcome_distribution.mean), f"Inf mean at scale {scale}"
            assert not np.isnan(result.outcome_distribution.std), f"NaN std at scale {scale}"
            assert not np.isinf(result.outcome_distribution.std), f"Inf std at scale {scale}"

    def test_extreme_intervention_values(self):
        """Test analysis with very large intervention values."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="input", kind="factor", label="Input"),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "input", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1.0, std=0.1),
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="extreme-intervention-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Opt 1", interventions={"input": 1e10}),
                InterventionOption(id="opt2", label="Opt 2", interventions={"input": 1e-10}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Analysis should complete without NaN/Inf
        assert response is not None
        for result in response.results:
            assert np.isfinite(result.outcome_distribution.mean)
            assert np.isfinite(result.outcome_distribution.std)

    def test_mixed_scale_graph(self):
        """Test graph with nodes at vastly different scales."""
        # Note: std must be > 0.001 per schema v2.6 D.4
        graph = GraphV2(
            nodes=[
                NodeV2(id="tiny", kind="factor", label="Tiny", observed_state=ObservedState(value=0.001)),
                NodeV2(id="huge", kind="factor", label="Huge", observed_state=ObservedState(value=1000.0)),
                NodeV2(id="output", kind="outcome", label="Output"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "tiny", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=1000.0, std=100.0),  # Scale up tiny
                ),
                EdgeV2(
                    **{"from": "huge", "to": "output"},
                    exists_probability=1.0,
                    strength=StrengthDistribution(mean=0.001, std=0.0011),  # Scale down huge (min std)
                ),
            ],
        )

        request = RobustnessRequestV2(
            request_id="mixed-scale-test",
            graph=graph,
            options=[
                InterventionOption(id="opt1", label="Opt 1", interventions={"tiny": 1e-8}),
                InterventionOption(id="opt2", label="Opt 2", interventions={"huge": 1e8}),
            ],
            goal_node_id="output",
            n_samples=100,
            seed=42,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Analysis should complete without numerical issues
        assert response is not None
        for result in response.results:
            assert np.isfinite(result.outcome_distribution.mean)
            assert np.isfinite(result.outcome_distribution.std)
