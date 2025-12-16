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

import numpy as np
import pytest

from src.models.robustness_v2 import (
    ClampMetrics,
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
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
            NodeV2(id="price", kind="decision", label="Price"),
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
            NodeV2(id="price", kind="decision", label="Price"),
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
                NodeV2(id="price", kind="decision", label="Price"),
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
                NodeV2(id="price", kind="decision", label="Price"),
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
