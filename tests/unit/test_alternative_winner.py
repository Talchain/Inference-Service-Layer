"""
Unit tests for alternative winner computation in fragile edges.

Tests the enhanced fragile edge output that identifies which option
would win if a fragile edge's assumption is wrong (weaker than modelled).
"""

import numpy as np
import pytest

from src.models.response_v2 import FragileEdgeV2, RobustnessResultV2
from src.services.robustness_analyzer_v2 import (
    DualUncertaintySampler,
    FactorSampler,
    FragileEdge,
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
)
from src.models.robustness_v2 import (
    EdgeV2,
    FragileEdgeEnhanced,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.utils.rng import SeededRNG


class TestFragileEdgeV2Model:
    """Test the FragileEdgeV2 model structure."""

    def test_fragile_edge_with_alternative_winner(self):
        """FragileEdgeV2 stores alternative winner correctly."""
        fe = FragileEdgeV2(
            edge_id="price->revenue",
            from_id="price",
            to_id="revenue",
            alternative_winner_id="option_economy",
            switch_probability=0.34,
        )
        assert fe.edge_id == "price->revenue"
        assert fe.from_id == "price"
        assert fe.to_id == "revenue"
        assert fe.alternative_winner_id == "option_economy"
        assert fe.switch_probability == 0.34

    def test_fragile_edge_without_alternative(self):
        """FragileEdgeV2 handles no alternative winner."""
        fe = FragileEdgeV2(
            edge_id="demand->sales",
            from_id="demand",
            to_id="sales",
            alternative_winner_id=None,
            switch_probability=None,
        )
        assert fe.alternative_winner_id is None
        assert fe.switch_probability is None

    def test_robustness_result_with_fragile_edges_v2(self):
        """RobustnessResultV2 includes enhanced fragile edges and v1 compat."""
        fragile_edges = [
            FragileEdgeV2(
                edge_id="price->revenue",
                from_id="price",
                to_id="revenue",
                alternative_winner_id="option_b",
                switch_probability=0.42,
            )
        ]
        result = RobustnessResultV2(
            level="low",
            confidence=0.75,
            fragile_edges=fragile_edges,
            fragile_edges_v1=["price->revenue"],
            robust_edges=["demand->revenue"],
            is_robust=False,
            recommendation_stability=0.8,
        )

        # V2 enhanced format
        assert len(result.fragile_edges) == 1
        assert result.fragile_edges[0].alternative_winner_id == "option_b"
        assert result.fragile_edges[0].switch_probability == 0.42

        # V1 compat format
        assert result.fragile_edges_v1 == ["price->revenue"]

    def test_serialization_format(self):
        """Serialized output matches expected format."""
        fe = FragileEdgeV2(
            edge_id="price->revenue",
            from_id="price",
            to_id="revenue",
            alternative_winner_id="option_economy",
            switch_probability=0.34,
        )
        data = fe.model_dump()
        assert data == {
            "edge_id": "price->revenue",
            "from_id": "price",
            "to_id": "revenue",
            "alternative_winner_id": "option_economy",
            "switch_probability": 0.34,
        }


class TestAlternativeWinnerComputation:
    """Test the _compute_alternative_winners function."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return RobustnessAnalyzerV2()

    def test_alternative_winner_binary_decision(self, analyzer):
        """Correctly identifies alternative winner in binary decision."""
        # Simulate fragile edge that flips decision when weak
        fragile_edge_info = {
            "price->revenue": ("price", "revenue"),
        }

        # Create edge configs where weak edge correlates with option_b winning
        n_samples = 100
        edge_configs = []
        winner_per_sample = []

        rng = np.random.RandomState(42)

        for i in range(n_samples):
            # Sample strength from normal distribution
            strength = rng.normal(0.5, 0.2)
            edge_configs.append({("price", "revenue"): strength})

            # When edge is weak (< 25th percentile ~ 0.37), option_b wins
            # When edge is strong, option_a wins
            if strength < 0.37:
                winner_per_sample.append("option_b")
            else:
                winner_per_sample.append("option_a")

        results = analyzer._compute_alternative_winners(
            fragile_edge_info,
            edge_configs,
            winner_per_sample,
            "option_a",  # overall winner
        )

        assert len(results) == 1
        edge_result = results[0]
        assert isinstance(edge_result, FragileEdgeEnhanced)
        assert edge_result.edge_id == "price->revenue"
        assert edge_result.from_id == "price"
        assert edge_result.to_id == "revenue"
        # In weak samples, option_b should win most often
        assert edge_result.alternative_winner_id == "option_b"
        assert edge_result.switch_probability is not None
        assert edge_result.switch_probability > 0.5

    def test_no_alternative_winner(self, analyzer):
        """Handles case where same option wins regardless of edge strength."""
        fragile_edge_info = {
            "price->revenue": ("price", "revenue"),
        }

        n_samples = 100
        edge_configs = []
        winner_per_sample = []

        rng = np.random.RandomState(42)

        for i in range(n_samples):
            strength = rng.normal(0.5, 0.2)
            edge_configs.append({("price", "revenue"): strength})
            # Same option always wins
            winner_per_sample.append("option_a")

        results = analyzer._compute_alternative_winners(
            fragile_edge_info,
            edge_configs,
            winner_per_sample,
            "option_a",
        )

        assert len(results) == 1
        edge_result = results[0]
        assert isinstance(edge_result, FragileEdgeEnhanced)
        # No alternative winner since same option wins
        # switch_probability = 0.0 (not None) to indicate "no switching"
        assert edge_result.alternative_winner_id is None
        assert edge_result.switch_probability == 0.0

    def test_multi_option_decision(self, analyzer):
        """Correctly identifies alternative winner among multiple options."""
        fragile_edge_info = {
            "price->revenue": ("price", "revenue"),
        }

        n_samples = 100
        edge_configs = []
        winner_per_sample = []

        rng = np.random.RandomState(42)
        options = ["premium", "standard", "economy"]

        for i in range(n_samples):
            strength = rng.normal(0.5, 0.2)
            edge_configs.append({("price", "revenue"): strength})

            # When edge weak: economy wins
            # When edge moderate: standard wins
            # When edge strong: premium wins
            if strength < 0.35:
                winner_per_sample.append("economy")
            elif strength < 0.55:
                winner_per_sample.append("standard")
            else:
                winner_per_sample.append("premium")

        # Overall winner is based on most frequent
        overall_winner = max(set(winner_per_sample), key=winner_per_sample.count)

        results = analyzer._compute_alternative_winners(
            fragile_edge_info,
            edge_configs,
            winner_per_sample,
            overall_winner,
        )

        assert len(results) == 1
        edge_result = results[0]
        assert isinstance(edge_result, FragileEdgeEnhanced)
        # In weak edge scenario, economy should win most
        assert edge_result.alternative_winner_id == "economy"

    def test_multiple_fragile_edges(self, analyzer):
        """Handles multiple fragile edges independently."""
        fragile_edge_info = {
            "price->revenue": ("price", "revenue"),
            "demand->sales": ("demand", "sales"),
        }

        n_samples = 100
        edge_configs = []
        winner_per_sample = []

        rng = np.random.RandomState(42)

        for i in range(n_samples):
            price_strength = rng.normal(0.5, 0.2)
            demand_strength = rng.normal(0.3, 0.1)
            edge_configs.append({
                ("price", "revenue"): price_strength,
                ("demand", "sales"): demand_strength,
            })
            # Winner depends on both edges
            if price_strength < 0.35:
                winner_per_sample.append("option_b")
            else:
                winner_per_sample.append("option_a")

        results = analyzer._compute_alternative_winners(
            fragile_edge_info,
            edge_configs,
            winner_per_sample,
            "option_a",
        )

        assert len(results) == 2
        assert all(isinstance(r, FragileEdgeEnhanced) for r in results)
        edge_ids = {r.edge_id for r in results}
        assert "price->revenue" in edge_ids
        assert "demand->sales" in edge_ids

    def test_determinism_with_seed(self, analyzer):
        """Same seed produces identical results."""
        fragile_edge_info = {
            "price->revenue": ("price", "revenue"),
        }

        def run_analysis(seed):
            rng = np.random.RandomState(seed)
            edge_configs = []
            winner_per_sample = []

            for i in range(50):
                strength = rng.normal(0.5, 0.2)
                edge_configs.append({("price", "revenue"): strength})
                if strength < 0.4:
                    winner_per_sample.append("option_b")
                else:
                    winner_per_sample.append("option_a")

            return analyzer._compute_alternative_winners(
                fragile_edge_info,
                edge_configs,
                winner_per_sample,
                "option_a",
            )

        result1 = run_analysis(42)
        result2 = run_analysis(42)

        # Same seed should produce identical results
        assert result1[0].alternative_winner_id == result2[0].alternative_winner_id
        assert result1[0].switch_probability == result2[0].switch_probability


class TestEndToEndAlternativeWinner:
    """Integration tests for alternative winner through full analysis."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
        return GraphV2(
            nodes=[
                NodeV2(
                    id="price",
                    kind="factor",
                    label="Price",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(id="revenue", kind="goal", label="Revenue"),
            ],
            edges=[
                EdgeV2(
                    from_="price",
                    to="revenue",
                    exists_probability=0.9,
                    strength=StrengthDistribution(mean=0.8, std=0.3),  # High variance
                ),
            ],
        )

    @pytest.fixture
    def binary_options(self):
        """Create two options with different interventions."""
        return [
            InterventionOption(
                id="option_premium",
                label="Premium Pricing",
                interventions={"price": 150.0},
            ),
            InterventionOption(
                id="option_economy",
                label="Economy Pricing",
                interventions={"price": 80.0},
            ),
        ]

    def test_full_analysis_returns_fragile_edges_enhanced(
        self, simple_graph, binary_options
    ):
        """Full analysis includes enhanced fragile edge data."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=binary_options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            analysis_types=["comparison", "sensitivity", "robustness"],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Robustness should be computed
        assert response.robustness is not None

        # Check if fragile_edges_enhanced is populated
        if response.robustness.fragile_edges:
            # If there are fragile edges, enhanced data should exist
            assert response.robustness.fragile_edges_enhanced is not None

            for enhanced in response.robustness.fragile_edges_enhanced:
                # Now returns FragileEdgeEnhanced objects, not dicts
                assert isinstance(enhanced, FragileEdgeEnhanced)
                assert enhanced.edge_id is not None
                assert enhanced.from_id is not None
                assert enhanced.to_id is not None
                # alternative_winner_id and switch_probability can be None

    def test_api_response_format(self, simple_graph, binary_options):
        """API response has correct fragile_edges format."""
        request = RobustnessRequestV2(
            graph=simple_graph,
            options=binary_options,
            goal_node_id="revenue",
            n_samples=200,
            seed=42,
            analysis_types=["comparison", "sensitivity", "robustness"],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Serialize to dict (as API would)
        data = response.model_dump()

        if data["robustness"]["fragile_edges_enhanced"]:
            for fe in data["robustness"]["fragile_edges_enhanced"]:
                # Check required fields exist
                assert "edge_id" in fe
                assert "from_id" in fe
                assert "to_id" in fe


class TestSwitchProbabilityCalculation:
    """Test switch_probability calculation accuracy."""

    @pytest.fixture
    def analyzer(self):
        return RobustnessAnalyzerV2()

    def test_switch_probability_bounds(self, analyzer):
        """Switch probability is between 0 and 1."""
        fragile_edge_info = {"a->b": ("a", "b")}

        rng = np.random.RandomState(42)
        edge_configs = [
            {("a", "b"): rng.normal(0.5, 0.2)} for _ in range(100)
        ]
        winner_per_sample = [
            "opt_a" if rng.random() > 0.3 else "opt_b" for _ in range(100)
        ]

        results = analyzer._compute_alternative_winners(
            fragile_edge_info,
            edge_configs,
            winner_per_sample,
            "opt_a",
        )

        if results[0].switch_probability is not None:
            prob = results[0].switch_probability
            assert 0 <= prob <= 1

    def test_switch_probability_100_percent(self, analyzer):
        """Switch probability can be 1.0 when alternative always wins."""
        fragile_edge_info = {"a->b": ("a", "b")}

        # Create scenario where weak edge ALWAYS means option_b wins
        edge_configs = []
        winner_per_sample = []

        for i in range(100):
            strength = 0.1 + (i * 0.01)  # Gradually increasing
            edge_configs.append({("a", "b"): strength})
            # Bottom 25% all go to option_b
            if strength <= 0.35:
                winner_per_sample.append("option_b")
            else:
                winner_per_sample.append("option_a")

        results = analyzer._compute_alternative_winners(
            fragile_edge_info,
            edge_configs,
            winner_per_sample,
            "option_a",
        )

        edge_result = results[0]
        assert isinstance(edge_result, FragileEdgeEnhanced)
        # All weak samples go to option_b
        assert edge_result.alternative_winner_id == "option_b"
        assert edge_result.switch_probability == 1.0


class TestShapeConsistency:
    """Test response shape consistency across V1/V2 formats.

    Ensures:
    - fragile_edges (V2) are typed objects (FragileEdgeV2)
    - fragile_edges_v1 are strings ("from->to")
    - robust_edges are strings ("from->to")
    - fragile and robust edge sets are disjoint
    """

    @pytest.fixture
    def graph_with_edges(self):
        """Create a graph with multiple edges of varying sensitivity."""
        return GraphV2(
            nodes=[
                NodeV2(
                    id="price",
                    kind="factor",
                    label="Price",
                    observed_state=ObservedState(value=100.0),
                ),
                NodeV2(
                    id="quality",
                    kind="factor",
                    label="Quality",
                    observed_state=ObservedState(value=0.8),
                ),
                NodeV2(
                    id="demand",
                    kind="factor",
                    label="Demand",
                    observed_state=ObservedState(value=500.0),
                ),
                NodeV2(id="revenue", kind="goal", label="Revenue"),
            ],
            edges=[
                # High variance edge - likely fragile
                EdgeV2(
                    from_="price",
                    to="revenue",
                    exists_probability=0.95,
                    strength=StrengthDistribution(mean=0.8, std=0.4),
                ),
                # Low variance edge - likely robust
                EdgeV2(
                    from_="quality",
                    to="revenue",
                    exists_probability=0.99,
                    strength=StrengthDistribution(mean=0.5, std=0.05),
                ),
                # Medium variance edge
                EdgeV2(
                    from_="demand",
                    to="revenue",
                    exists_probability=0.9,
                    strength=StrengthDistribution(mean=0.6, std=0.15),
                ),
            ],
        )

    @pytest.fixture
    def options(self):
        """Create test options."""
        return [
            InterventionOption(
                id="option_a",
                label="Option A",
                interventions={"price": 120.0},
            ),
            InterventionOption(
                id="option_b",
                label="Option B",
                interventions={"price": 80.0},
            ),
        ]

    def test_fragile_edges_enhanced_are_typed_objects(self, graph_with_edges, options):
        """Analyzer returns typed FragileEdgeEnhanced objects, not dicts."""
        request = RobustnessRequestV2(
            graph=graph_with_edges,
            options=options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            analysis_types=["comparison", "sensitivity", "robustness"],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        if response.robustness.fragile_edges_enhanced:
            for fe in response.robustness.fragile_edges_enhanced:
                # Must be FragileEdgeEnhanced objects, NOT dicts
                assert isinstance(fe, FragileEdgeEnhanced), (
                    f"Expected FragileEdgeEnhanced, got {type(fe)}"
                )
                # Must have typed attributes
                assert hasattr(fe, "edge_id")
                assert hasattr(fe, "from_id")
                assert hasattr(fe, "to_id")
                assert hasattr(fe, "alternative_winner_id")
                assert hasattr(fe, "switch_probability")

    def test_fragile_edges_v1_are_strings(self, graph_with_edges, options):
        """V1 fragile_edges are strings in 'from->to' format."""
        request = RobustnessRequestV2(
            graph=graph_with_edges,
            options=options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            analysis_types=["comparison", "sensitivity", "robustness"],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # V1 format: List[str]
        for edge in response.robustness.fragile_edges:
            assert isinstance(edge, str), f"Expected str, got {type(edge)}"
            assert "->" in edge, f"Expected 'from->to' format, got {edge}"

    def test_robust_edges_are_strings(self, graph_with_edges, options):
        """robust_edges are always strings in 'from->to' format."""
        request = RobustnessRequestV2(
            graph=graph_with_edges,
            options=options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            analysis_types=["comparison", "sensitivity", "robustness"],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # robust_edges: List[str]
        for edge in response.robustness.robust_edges:
            assert isinstance(edge, str), f"Expected str, got {type(edge)}"
            assert "->" in edge, f"Expected 'from->to' format, got {edge}"

    def test_fragile_robust_sets_are_disjoint(self, graph_with_edges, options):
        """An edge cannot be both fragile AND robust (Bug 2 regression test)."""
        request = RobustnessRequestV2(
            graph=graph_with_edges,
            options=options,
            goal_node_id="revenue",
            n_samples=500,
            seed=42,
            analysis_types=["comparison", "sensitivity", "robustness"],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        fragile_set = set(response.robustness.fragile_edges)
        robust_set = set(response.robustness.robust_edges)

        # Sets must be disjoint
        overlap = fragile_set & robust_set
        assert len(overlap) == 0, (
            f"Edge overlap detected! Edges in both sets: {overlap}"
        )

    def test_serialized_response_shape(self, graph_with_edges, options):
        """Serialized JSON response has correct shape for API consumption."""
        request = RobustnessRequestV2(
            graph=graph_with_edges,
            options=options,
            goal_node_id="revenue",
            n_samples=200,
            seed=42,
            analysis_types=["comparison", "sensitivity", "robustness"],
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Serialize as API would
        data = response.model_dump()

        robustness = data["robustness"]

        # fragile_edges (V1 strings)
        assert isinstance(robustness["fragile_edges"], list)
        for edge in robustness["fragile_edges"]:
            assert isinstance(edge, str)

        # robust_edges (strings)
        assert isinstance(robustness["robust_edges"], list)
        for edge in robustness["robust_edges"]:
            assert isinstance(edge, str)

        # fragile_edges_enhanced (objects when serialized)
        if robustness["fragile_edges_enhanced"]:
            for fe in robustness["fragile_edges_enhanced"]:
                assert isinstance(fe, dict), "Serialized should be dict"
                assert "edge_id" in fe
                assert "from_id" in fe
                assert "to_id" in fe
                assert "alternative_winner_id" in fe
                assert "switch_probability" in fe
