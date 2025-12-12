"""
Unit tests for Decision Robustness Suite.

Tests cover:
- Sensitivity analysis computation
- Robustness bounds calculation
- Value of Information (EVPI/EVSI)
- Pareto frontier analysis
- Narrative generation
- Outcome logging
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models.decision_robustness import (
    AnalysisOptions,
    ConfidenceLevelEnum,
    DecisionOption,
    ImpactDirectionEnum,
    OutcomeLog,
    OutcomeLogRequest,
    OutcomeSummary,
    OutcomeUpdateRequest,
    ParetoPoint,
    ParetoResult,
    RankedOption,
    Recommendation,
    RecommendationStatusEnum,
    RobustnessBound,
    RobustnessLabelEnum,
    RobustnessRequest,
    RobustnessResult,
    SensitiveParameter,
    UtilityDistribution,
    UtilitySpecification,
    ValueOfInformation,
)
from src.models.shared import GraphV1, GraphNodeV1, GraphEdgeV1, NodeKind
from src.services.decision_robustness_analyzer import (
    DecisionRobustnessAnalyzer,
    get_graph_hash,
)
from src.services.outcome_logger import OutcomeLogger, get_outcome_logger


def make_edge(from_node: str, to_node: str, weight: float = 1.0) -> GraphEdgeV1:
    """Helper to create GraphEdgeV1 with proper alias handling."""
    return GraphEdgeV1.model_validate({"from": from_node, "to": to_node, "weight": weight})


def make_test_graph() -> GraphV1:
    """Create a test graph for analysis."""
    return GraphV1(
        nodes=[
            GraphNodeV1(id="marketing_spend", kind=NodeKind.DECISION, label="Marketing Spend", belief=0.5),
            GraphNodeV1(id="price", kind=NodeKind.DECISION, label="Product Price", belief=0.6),
            GraphNodeV1(id="demand", kind=NodeKind.FACTOR, label="Customer Demand", belief=0.5),
            GraphNodeV1(id="revenue", kind=NodeKind.GOAL, label="Revenue", belief=0.5),
        ],
        edges=[
            make_edge("marketing_spend", "demand", 2.0),
            make_edge("price", "demand", -1.5),
            make_edge("demand", "revenue", 2.5),
        ],
    )


def make_test_options() -> list[DecisionOption]:
    """Create test decision options."""
    return [
        DecisionOption(
            id="option_a",
            label="Aggressive Marketing",
            interventions={"marketing_spend": 100000, "price": 49.99},
            is_baseline=False,
        ),
        DecisionOption(
            id="option_b",
            label="Premium Pricing",
            interventions={"marketing_spend": 50000, "price": 79.99},
            is_baseline=True,
        ),
        DecisionOption(
            id="option_c",
            label="Balanced Approach",
            interventions={"marketing_spend": 75000, "price": 59.99},
            is_baseline=False,
        ),
    ]


def make_test_request() -> RobustnessRequest:
    """Create a test robustness request."""
    return RobustnessRequest(
        graph=make_test_graph(),
        options=make_test_options(),
        utility=UtilitySpecification(
            goal_node_id="revenue",
            maximize=True,
        ),
        analysis_options=AnalysisOptions(
            sensitivity_top_n=3,
            perturbation_range=0.5,
            monte_carlo_samples=100,  # Reduced for tests
            include_pareto=False,
            include_voi=False,
        ),
    )


class TestRobustnessSchemas:
    """Tests for Pydantic response schemas."""

    def test_utility_distribution_creation(self):
        """Test UtilityDistribution model."""
        dist = UtilityDistribution(
            p5=85000.0,
            p25=92000.0,
            p50=100000.0,
            p75=108000.0,
            p95=115000.0,
        )
        assert dist.p50 == 100000.0
        assert dist.p5 < dist.p95

    def test_ranked_option_creation(self):
        """Test RankedOption model."""
        option = RankedOption(
            option_id="option_a",
            option_label="Test Option",
            expected_utility=150000.0,
            utility_distribution=UtilityDistribution(
                p5=120000, p25=135000, p50=150000, p75=165000, p95=180000
            ),
            rank=1,
            vs_baseline=25000.0,
            vs_baseline_pct=20.0,
        )
        assert option.rank == 1
        assert option.vs_baseline_pct == 20.0

    def test_recommendation_creation(self):
        """Test Recommendation model."""
        rec = Recommendation(
            option_id="option_a",
            option_label="Best Option",
            confidence=ConfidenceLevelEnum.HIGH,
            recommendation_status=RecommendationStatusEnum.ACTIONABLE,
        )
        assert rec.confidence == ConfidenceLevelEnum.HIGH
        assert rec.recommendation_status == RecommendationStatusEnum.ACTIONABLE

    def test_sensitive_parameter_creation(self):
        """Test SensitiveParameter model."""
        param = SensitiveParameter(
            parameter_id="edge_price_demand",
            parameter_label="Price → Demand weight",
            sensitivity_score=0.85,
            current_value=-1.5,
            impact_direction=ImpactDirectionEnum.NEGATIVE,
            description="Increasing Price → Demand connection decreases outcome",
        )
        assert param.sensitivity_score == 0.85
        assert param.impact_direction == ImpactDirectionEnum.NEGATIVE

    def test_robustness_bound_creation(self):
        """Test RobustnessBound model."""
        bound = RobustnessBound(
            parameter_id="marketing_roi",
            parameter_label="Marketing ROI",
            flip_threshold=0.15,
            flip_threshold_pct=25.0,
            flip_to_option="option_b",
        )
        assert bound.flip_threshold_pct == 25.0

    def test_value_of_information_creation(self):
        """Test ValueOfInformation model."""
        voi = ValueOfInformation(
            parameter_id="churn_rate",
            parameter_label="Customer Churn Rate",
            evpi=15000.0,
            evsi=8500.0,
            current_uncertainty=0.25,
            recommendation="High value - consider gathering data",
            data_collection_suggestion="Survey 50 customers",
        )
        assert voi.evpi > voi.evsi
        assert "gathering data" in voi.recommendation

    def test_pareto_point_creation(self):
        """Test ParetoPoint model."""
        point = ParetoPoint(
            option_id="option_a",
            option_label="Aggressive Marketing",
            goal_values={"revenue": 150000.0, "risk": 0.3},
            is_dominated=False,
            trade_off_description="Highest revenue, moderate risk",
        )
        assert not point.is_dominated
        assert "revenue" in point.goal_values

    def test_robustness_result_creation(self):
        """Test complete RobustnessResult model."""
        result = RobustnessResult(
            option_rankings=[
                RankedOption(
                    option_id="option_a",
                    option_label="Best",
                    expected_utility=100000,
                    utility_distribution=UtilityDistribution(
                        p5=80000, p25=90000, p50=100000, p75=110000, p95=120000
                    ),
                    rank=1,
                    vs_baseline=10000,
                    vs_baseline_pct=10.0,
                )
            ],
            recommendation=Recommendation(
                option_id="option_a",
                option_label="Best",
                confidence=ConfidenceLevelEnum.HIGH,
                recommendation_status=RecommendationStatusEnum.ACTIONABLE,
            ),
            sensitivity=[],
            robustness_label=RobustnessLabelEnum.ROBUST,
            robustness_summary="Your decision is robust.",
            robustness_bounds=[],
            value_of_information=[],
            pareto=None,
            narrative="Your decision is robust.",
        )
        assert result.robustness_label == RobustnessLabelEnum.ROBUST

    def test_robustness_request_validation(self):
        """Test RobustnessRequest validation."""
        request = make_test_request()
        assert len(request.options) == 3
        assert request.utility.goal_node_id == "revenue"

    def test_robustness_request_unique_option_ids(self):
        """Test that duplicate option IDs are rejected."""
        with pytest.raises(ValueError):
            RobustnessRequest(
                graph=make_test_graph(),
                options=[
                    DecisionOption(
                        id="same_id",
                        label="Option A",
                        interventions={"marketing_spend": 100000},
                    ),
                    DecisionOption(
                        id="same_id",  # Duplicate
                        label="Option B",
                        interventions={"marketing_spend": 50000},
                    ),
                ],
                utility=UtilitySpecification(goal_node_id="revenue"),
            )


class TestDecisionRobustnessAnalyzer:
    """Tests for DecisionRobustnessAnalyzer service."""

    def test_analyzer_initialization(self):
        """Test analyzer can be initialized."""
        analyzer = DecisionRobustnessAnalyzer()
        assert analyzer is not None

    def test_basic_analysis(self):
        """Test basic robustness analysis."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()

        result = analyzer.analyze(request, "req_test_001")

        assert result is not None
        assert len(result.option_rankings) == 3
        assert result.recommendation is not None
        assert result.robustness_label in [
            RobustnessLabelEnum.ROBUST,
            RobustnessLabelEnum.MODERATE,
            RobustnessLabelEnum.FRAGILE,
        ]

    def test_option_rankings_sorted(self):
        """Test that options are ranked by utility."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()

        result = analyzer.analyze(request, "req_test_002")

        # Verify ranks are sequential
        ranks = [opt.rank for opt in result.option_rankings]
        assert ranks == [1, 2, 3]

        # Verify utilities are descending
        utilities = [opt.expected_utility for opt in result.option_rankings]
        assert utilities == sorted(utilities, reverse=True)

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis produces results."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()
        request.analysis_options.sensitivity_top_n = 5

        result = analyzer.analyze(request, "req_test_003")

        assert len(result.sensitivity) > 0
        assert all(0 <= p.sensitivity_score <= 1 for p in result.sensitivity)
        # Should be sorted by sensitivity
        scores = [p.sensitivity_score for p in result.sensitivity]
        assert scores == sorted(scores, reverse=True)

    def test_robustness_bounds_computation(self):
        """Test robustness bounds are computed."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()

        result = analyzer.analyze(request, "req_test_004")

        # May or may not have bounds depending on sensitivity
        for bound in result.robustness_bounds:
            assert bound.flip_threshold_pct > 0
            assert bound.flip_to_option in ["option_a", "option_b", "option_c"]

    def test_robustness_classification(self):
        """Test robustness classification logic."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()

        result = analyzer.analyze(request, "req_test_005")

        # Check that summary matches label
        if result.robustness_label == RobustnessLabelEnum.ROBUST:
            assert "robust" in result.robustness_summary.lower()
        elif result.robustness_label == RobustnessLabelEnum.FRAGILE:
            assert "fragile" in result.robustness_summary.lower()

    def test_fragile_sets_exploratory(self):
        """Test that fragile recommendations become exploratory."""
        analyzer = DecisionRobustnessAnalyzer()

        # Create a graph that's likely to be fragile
        # (options with very similar utilities)
        similar_options = [
            DecisionOption(
                id="option_a",
                label="Option A",
                interventions={"marketing_spend": 100000, "price": 50},
                is_baseline=True,
            ),
            DecisionOption(
                id="option_b",
                label="Option B",
                interventions={"marketing_spend": 100001, "price": 50},  # Nearly identical
            ),
        ]

        request = RobustnessRequest(
            graph=make_test_graph(),
            options=similar_options,
            utility=UtilitySpecification(goal_node_id="revenue"),
            analysis_options=AnalysisOptions(monte_carlo_samples=100),
        )

        result = analyzer.analyze(request, "req_test_006")

        # If fragile, should be exploratory
        if result.robustness_label == RobustnessLabelEnum.FRAGILE:
            assert result.recommendation.recommendation_status == RecommendationStatusEnum.EXPLORATORY

    def test_narrative_generation(self):
        """Test narrative is generated and meaningful."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()

        result = analyzer.analyze(request, "req_test_007")

        assert result.narrative is not None
        assert len(result.narrative) > 0
        # Should mention the recommended option
        assert result.recommendation.option_label in result.narrative

    def test_graph_model_building(self):
        """Test internal graph model building."""
        analyzer = DecisionRobustnessAnalyzer()
        graph = make_test_graph()

        model = analyzer._build_graph_model(graph)

        assert "nodes" in model
        assert "edges" in model
        assert "adjacency" in model
        assert len(model["nodes"]) == 4
        assert len(model["edges"]) == 3

    def test_utility_distribution_computation(self):
        """Test utility distributions have valid percentiles."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()

        result = analyzer.analyze(request, "req_test_008")

        for option in result.option_rankings:
            dist = option.utility_distribution
            assert dist.p5 <= dist.p25 <= dist.p50 <= dist.p75 <= dist.p95


class TestValueOfInformation:
    """Tests for Value of Information calculations."""

    def test_voi_with_uncertainties(self):
        """Test VoI is computed when uncertainties provided."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()
        request.analysis_options.include_voi = True
        request.parameter_uncertainties = {
            "churn_rate": {"mean": 0.15, "std": 0.05},
            "price_elasticity": {"mean": -1.2, "std": 0.3},
        }

        result = analyzer.analyze(request, "req_test_voi_001")

        assert len(result.value_of_information) > 0
        for voi in result.value_of_information:
            assert voi.evpi >= 0
            assert voi.evsi >= 0
            assert voi.evsi <= voi.evpi  # EVSI cannot exceed EVPI

    def test_voi_recommendations(self):
        """Test VoI includes actionable recommendations."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()
        request.analysis_options.include_voi = True
        request.parameter_uncertainties = {
            "customer_churn": {"mean": 0.2, "std": 0.1},
        }

        result = analyzer.analyze(request, "req_test_voi_002")

        for voi in result.value_of_information:
            assert len(voi.recommendation) > 0
            assert len(voi.data_collection_suggestion) > 0

    def test_voi_empty_without_uncertainties(self):
        """Test VoI is empty when no uncertainties provided."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()
        request.analysis_options.include_voi = True
        # No parameter_uncertainties

        result = analyzer.analyze(request, "req_test_voi_003")

        assert len(result.value_of_information) == 0


class TestParetoFrontier:
    """Tests for Pareto frontier analysis."""

    def test_pareto_with_multiple_goals(self):
        """Test Pareto frontier with multiple goals."""
        analyzer = DecisionRobustnessAnalyzer()

        # Add a second goal node
        graph = make_test_graph()
        graph.nodes.append(
            GraphNodeV1(id="satisfaction", kind=NodeKind.GOAL, label="Customer Satisfaction")
        )
        graph.edges.append(make_edge("price", "satisfaction", -1.0))

        request = RobustnessRequest(
            graph=graph,
            options=make_test_options(),
            utility=UtilitySpecification(
                goal_node_id="revenue",
                additional_goals=["satisfaction"],
                maximize=True,
            ),
            analysis_options=AnalysisOptions(
                include_pareto=True,
                monte_carlo_samples=100,
            ),
        )

        result = analyzer.analyze(request, "req_test_pareto_001")

        assert result.pareto is not None
        assert len(result.pareto.goals) == 2
        assert len(result.pareto.frontier_options) > 0

    def test_pareto_dominance_detection(self):
        """Test that dominated options are identified."""
        analyzer = DecisionRobustnessAnalyzer()

        graph = make_test_graph()
        graph.nodes.append(
            GraphNodeV1(id="risk", kind=NodeKind.GOAL, label="Risk Level")
        )
        graph.edges.append(make_edge("marketing_spend", "risk", 0.5))

        request = RobustnessRequest(
            graph=graph,
            options=make_test_options(),
            utility=UtilitySpecification(
                goal_node_id="revenue",
                additional_goals=["risk"],
            ),
            analysis_options=AnalysisOptions(include_pareto=True, monte_carlo_samples=100),
        )

        result = analyzer.analyze(request, "req_test_pareto_002")

        assert result.pareto is not None
        # At least one should be non-dominated
        non_dominated = [p for p in result.pareto.frontier_options if not p.is_dominated]
        assert len(non_dominated) > 0

    def test_pareto_none_for_single_goal(self):
        """Test Pareto is None when only one goal."""
        analyzer = DecisionRobustnessAnalyzer()
        request = make_test_request()
        request.analysis_options.include_pareto = True
        # Only one goal (revenue)

        result = analyzer.analyze(request, "req_test_pareto_003")

        assert result.pareto is None


class TestOutcomeLogger:
    """Tests for outcome logging service."""

    def test_logger_initialization(self):
        """Test OutcomeLogger can be initialized."""
        logger = OutcomeLogger()
        assert logger is not None

    def test_log_decision(self):
        """Test logging a decision."""
        logger = OutcomeLogger()
        request = OutcomeLogRequest(
            decision_id="decision_001",
            graph_hash="abc123",
            response_hash="def456",
            chosen_option="option_a",
            recommendation_option="option_a",
            user_id="user_123",
        )

        log = logger.log_decision(request, "req_test_log_001")

        assert log.id is not None
        assert log.decision_id == "decision_001"
        assert log.recommendation_followed is True
        assert log.timestamp is not None

    def test_log_not_followed(self):
        """Test logging when recommendation not followed."""
        logger = OutcomeLogger()
        request = OutcomeLogRequest(
            decision_id="decision_002",
            graph_hash="abc123",
            response_hash="def456",
            chosen_option="option_b",  # Different from recommendation
            recommendation_option="option_a",
        )

        log = logger.log_decision(request, "req_test_log_002")

        assert log.recommendation_followed is False

    def test_update_outcome(self):
        """Test updating outcome with actual values."""
        logger = OutcomeLogger()

        # First log the decision
        log_request = OutcomeLogRequest(
            decision_id="decision_003",
            graph_hash="abc123",
            response_hash="def456",
            chosen_option="option_a",
            recommendation_option="option_a",
        )
        log = logger.log_decision(log_request, "req_test_log_003")

        # Then update with outcome
        update_request = OutcomeUpdateRequest(
            outcome_values={"revenue": 155000.0},
            notes="Q1 results",
        )
        updated = logger.update_outcome(log.id, update_request, "req_test_log_003")

        assert updated is not None
        assert updated.outcome_values == {"revenue": 155000.0}
        assert updated.outcome_timestamp is not None
        assert "Q1 results" in updated.notes

    def test_get_outcome(self):
        """Test retrieving an outcome log."""
        logger = OutcomeLogger()

        log_request = OutcomeLogRequest(
            decision_id="decision_004",
            graph_hash="abc123",
            response_hash="def456",
            chosen_option="option_a",
            recommendation_option="option_a",
        )
        log = logger.log_decision(log_request, "req_test_log_004")

        retrieved = logger.get_outcome(log.id)

        assert retrieved is not None
        assert retrieved.id == log.id
        assert retrieved.decision_id == "decision_004"

    def test_get_nonexistent_outcome(self):
        """Test getting nonexistent outcome returns None."""
        logger = OutcomeLogger()

        result = logger.get_outcome("nonexistent_id")

        assert result is None

    def test_get_summary(self):
        """Test getting outcome summary."""
        logger = OutcomeLogger()

        # Log some decisions
        for i in range(5):
            request = OutcomeLogRequest(
                decision_id=f"decision_{i}",
                graph_hash="abc123",
                response_hash="def456",
                chosen_option="option_a" if i < 3 else "option_b",
                recommendation_option="option_a",
            )
            logger.log_decision(request, f"req_test_summary_{i}")

        summary = logger.get_summary("req_test_summary")

        assert summary.total_logged == 5
        assert summary.recommendations_followed == 3
        assert summary.recommendations_followed_pct == 60.0

    def test_singleton_outcome_logger(self):
        """Test get_outcome_logger returns singleton."""
        logger1 = get_outcome_logger()
        logger2 = get_outcome_logger()

        assert logger1 is logger2


class TestGraphHash:
    """Tests for graph hashing utility."""

    def test_graph_hash_deterministic(self):
        """Test that graph hash is deterministic."""
        graph = make_test_graph()

        hash1 = get_graph_hash(graph)
        hash2 = get_graph_hash(graph)

        assert hash1 == hash2

    def test_graph_hash_different_for_different_graphs(self):
        """Test that different graphs have different hashes."""
        graph1 = make_test_graph()
        graph2 = make_test_graph()
        graph2.nodes.append(
            GraphNodeV1(id="extra_node", kind=NodeKind.FACTOR, label="Extra")
        )

        hash1 = get_graph_hash(graph1)
        hash2 = get_graph_hash(graph2)

        assert hash1 != hash2

    def test_graph_hash_length(self):
        """Test that graph hash has expected length."""
        graph = make_test_graph()

        hash_val = get_graph_hash(graph)

        assert len(hash_val) == 16  # SHA256 truncated to 16 chars


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_two_options_minimum(self):
        """Test analysis works with minimum two options."""
        analyzer = DecisionRobustnessAnalyzer()

        request = RobustnessRequest(
            graph=make_test_graph(),
            options=[
                DecisionOption(
                    id="option_a",
                    label="Option A",
                    interventions={"marketing_spend": 100000},
                    is_baseline=True,
                ),
                DecisionOption(
                    id="option_b",
                    label="Option B",
                    interventions={"marketing_spend": 50000},
                ),
            ],
            utility=UtilitySpecification(goal_node_id="revenue"),
            analysis_options=AnalysisOptions(monte_carlo_samples=100),
        )

        result = analyzer.analyze(request, "req_test_edge_001")

        assert len(result.option_rankings) == 2

    def test_analysis_with_disconnected_nodes(self):
        """Test analysis handles disconnected nodes."""
        analyzer = DecisionRobustnessAnalyzer()

        graph = make_test_graph()
        # Add disconnected node
        graph.nodes.append(
            GraphNodeV1(id="isolated", kind=NodeKind.FACTOR, label="Isolated Node")
        )

        request = RobustnessRequest(
            graph=graph,
            options=make_test_options(),
            utility=UtilitySpecification(goal_node_id="revenue"),
            analysis_options=AnalysisOptions(monte_carlo_samples=100),
        )

        result = analyzer.analyze(request, "req_test_edge_002")

        assert result is not None
        assert result.recommendation is not None

    def test_analysis_with_zero_weights(self):
        """Test analysis handles zero edge weights."""
        analyzer = DecisionRobustnessAnalyzer()

        graph = GraphV1(
            nodes=[
                GraphNodeV1(id="input", kind=NodeKind.DECISION, label="Input"),
                GraphNodeV1(id="output", kind=NodeKind.GOAL, label="Output"),
            ],
            edges=[
                make_edge("input", "output", 0.0),  # Zero weight
            ],
        )

        request = RobustnessRequest(
            graph=graph,
            options=[
                DecisionOption(id="opt_a", label="A", interventions={"input": 100}, is_baseline=True),
                DecisionOption(id="opt_b", label="B", interventions={"input": 200}),
            ],
            utility=UtilitySpecification(goal_node_id="output"),
            analysis_options=AnalysisOptions(monte_carlo_samples=100),
        )

        result = analyzer.analyze(request, "req_test_edge_003")

        assert result is not None

    def test_analysis_with_negative_weights(self):
        """Test analysis handles negative edge weights."""
        analyzer = DecisionRobustnessAnalyzer()

        graph = GraphV1(
            nodes=[
                GraphNodeV1(id="cost", kind=NodeKind.DECISION, label="Cost"),
                GraphNodeV1(id="profit", kind=NodeKind.GOAL, label="Profit"),
            ],
            edges=[
                make_edge("cost", "profit", -2.0),  # Negative weight
            ],
        )

        request = RobustnessRequest(
            graph=graph,
            options=[
                DecisionOption(id="opt_a", label="Low Cost", interventions={"cost": 1000}, is_baseline=True),
                DecisionOption(id="opt_b", label="High Cost", interventions={"cost": 5000}),
            ],
            utility=UtilitySpecification(goal_node_id="profit"),
            analysis_options=AnalysisOptions(monte_carlo_samples=100),
        )

        result = analyzer.analyze(request, "req_test_edge_004")

        assert result is not None
        # Lower cost should rank higher due to negative weight
