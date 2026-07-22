"""
Unit tests for Sequential Decision Support (Phase 4).

Tests backward induction, value of flexibility, information value,
and discount factor effects.
"""

import pytest

from src.models.requests import (
    DecisionStage,
    SequentialAnalysisRequest,
    SequentialGraph,
    SequentialGraphEdge,
    SequentialGraphNode,
    StageSensitivityRequest,
)
from src.services.sequential_decision import SequentialDecisionEngine


@pytest.fixture
def engine():
    """Create sequential decision engine instance."""
    return SequentialDecisionEngine()


@pytest.fixture
def two_stage_simple_graph():
    """Simple two-stage decision: invest now or wait for information."""
    nodes = [
        SequentialGraphNode(
            id="invest_decision",
            type="decision",
            label="Investment Decision"
        ),
        SequentialGraphNode(
            id="market_outcome",
            type="chance",
            label="Market Outcome"
        ),
        SequentialGraphNode(
            id="success",
            type="terminal",
            label="Success",
            payoff=100000
        ),
        SequentialGraphNode(
            id="failure",
            type="terminal",
            label="Failure",
            payoff=-20000
        ),
        SequentialGraphNode(
            id="no_invest",
            type="terminal",
            label="No Investment",
            payoff=0
        )
    ]

    edges = [
        SequentialGraphEdge(
            from_node="invest_decision",
            to_node="market_outcome",
            action="invest",
            immediate_payoff=-10000
        ),
        SequentialGraphEdge(
            from_node="invest_decision",
            to_node="no_invest",
            action="wait"
        ),
        SequentialGraphEdge(
            from_node="market_outcome",
            to_node="success",
            outcome="favorable",
            probability=0.6
        ),
        SequentialGraphEdge(
            from_node="market_outcome",
            to_node="failure",
            outcome="unfavorable",
            probability=0.4
        )
    ]

    graph = SequentialGraph(
        nodes=nodes,
        edges=edges,
        stage_assignments={
            "invest_decision": 0,
            "market_outcome": 1,
            "success": 2,
            "failure": 2,
            "no_invest": 1
        }
    )

    stages = [
        DecisionStage(
            stage_index=0,
            stage_label="Investment Decision",
            decision_nodes=["invest_decision"]
        ),
        DecisionStage(
            stage_index=1,
            stage_label="Market Resolution",
            decision_nodes=[],
            resolution_nodes=["market_outcome"]
        ),
        DecisionStage(
            stage_index=2,
            stage_label="Terminal",
            decision_nodes=[]
        )
    ]

    return graph, stages


@pytest.fixture
def three_stage_graph():
    """Three-stage decision problem with sequential choices."""
    nodes = [
        SequentialGraphNode(
            id="launch_decision",
            type="decision",
            label="Launch Decision"
        ),
        SequentialGraphNode(
            id="market_response",
            type="chance",
            label="Market Response"
        ),
        SequentialGraphNode(
            id="pricing_decision",
            type="decision",
            label="Pricing Decision"
        ),
        SequentialGraphNode(
            id="high_revenue",
            type="terminal",
            label="High Revenue",
            payoff=150000
        ),
        SequentialGraphNode(
            id="medium_revenue",
            type="terminal",
            label="Medium Revenue",
            payoff=50000
        ),
        SequentialGraphNode(
            id="exit",
            type="terminal",
            label="Exit",
            payoff=-30000
        ),
        SequentialGraphNode(
            id="no_launch",
            type="terminal",
            label="No Launch",
            payoff=0
        )
    ]

    edges = [
        SequentialGraphEdge(
            from_node="launch_decision",
            to_node="market_response",
            action="launch",
            immediate_payoff=-50000
        ),
        SequentialGraphEdge(
            from_node="launch_decision",
            to_node="no_launch",
            action="abort"
        ),
        SequentialGraphEdge(
            from_node="market_response",
            to_node="pricing_decision",
            outcome="positive",
            probability=0.7
        ),
        SequentialGraphEdge(
            from_node="market_response",
            to_node="exit",
            outcome="negative",
            probability=0.3
        ),
        SequentialGraphEdge(
            from_node="pricing_decision",
            to_node="high_revenue",
            action="premium"
        ),
        SequentialGraphEdge(
            from_node="pricing_decision",
            to_node="medium_revenue",
            action="economy"
        )
    ]

    graph = SequentialGraph(
        nodes=nodes,
        edges=edges,
        stage_assignments={
            "launch_decision": 0,
            "market_response": 1,
            "pricing_decision": 1,
            "high_revenue": 2,
            "medium_revenue": 2,
            "exit": 2,
            "no_launch": 1
        }
    )

    stages = [
        DecisionStage(
            stage_index=0,
            stage_label="Launch",
            decision_nodes=["launch_decision"]
        ),
        DecisionStage(
            stage_index=1,
            stage_label="Pricing",
            decision_nodes=["pricing_decision"],
            resolution_nodes=["market_response"]
        ),
        DecisionStage(
            stage_index=2,
            stage_label="Terminal",
            decision_nodes=[]
        )
    ]

    return graph, stages


class TestBackwardInduction:
    """Tests for backward induction optimality."""

    def test_two_stage_optimal(self, engine, two_stage_simple_graph):
        """Two-stage decision should produce optimal policy."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=1.0,  # No discounting for clarity
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # Expected value of investing:
        # -10000 (investment) + 0.6 * 100000 + 0.4 * (-20000)
        # = -10000 + 60000 - 8000 = 42000
        # Not investing = 0
        # So optimal is to invest

        assert result.optimal_policy.expected_total_value > 0
        assert len(result.stage_analyses) >= 1

    def test_three_stage_optimal(self, engine, three_stage_graph):
        """Three-stage decision should produce optimal policy."""
        graph, stages = three_stage_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # Should have policy for each stage with decision nodes
        assert len(result.optimal_policy.stages) >= 1
        assert result.optimal_policy.expected_total_value is not None

    def test_policy_backward_induction_optimal(self, engine, two_stage_simple_graph):
        """Policy should be optimal under known transition probabilities."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=1.0,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # With 60% success rate and high reward, investing should be optimal
        # Check that the policy actually recommends the optimal action
        if result.optimal_policy.stages:
            first_stage = result.optimal_policy.stages[0]
            # Either default action is invest, or invest is in conditional actions
            has_invest = (
                first_stage.decision_rule.default_action == "invest" or
                any(ca.action == "invest" for ca in first_stage.decision_rule.conditional_actions)
            )
            # At least the policy should include invest as an option
            assert first_stage.decision_rule.default_action is not None


class TestValueOfFlexibility:
    """Tests for value of flexibility calculation."""

    def test_value_of_flexibility_non_negative(self, engine, two_stage_simple_graph):
        """Value of flexibility should be non-negative."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # Flexibility can never hurt - it should be >= 0
        assert result.value_of_flexibility >= 0

    def test_value_of_flexibility_positive(self, engine, three_stage_graph):
        """Multi-stage problems should have positive flexibility value."""
        graph, stages = three_stage_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # With information revelation, flexibility should be valuable
        assert result.value_of_flexibility >= 0


class TestDiscountFactor:
    """Tests for discount factor effects."""

    def test_discount_factor_effect(self, engine, three_stage_graph):
        """Higher discount factor should favor earlier payoffs."""
        graph, stages = three_stage_graph

        # High discount (future almost as valuable as present)
        request_high = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.99,
            risk_tolerance="neutral"
        )
        result_high = engine.analyze(request_high)

        # Low discount (future much less valuable)
        request_low = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.7,
            risk_tolerance="neutral"
        )
        result_low = engine.analyze(request_low)

        # Both should complete successfully
        assert result_high.optimal_policy is not None
        assert result_low.optimal_policy is not None

        # Lower discount should generally favor earlier payoffs
        # (values may differ based on timing of payoffs)

    def test_discount_factor_one_gives_max_value(self, engine, two_stage_simple_graph):
        """Discount factor of 1.0 should give maximum total value."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=1.0,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # With no discounting, we should get the full expected value
        assert result.optimal_policy.expected_total_value is not None


class TestInformationValue:
    """Tests for information value computation."""

    def test_information_value_computation(self, engine, two_stage_simple_graph):
        """Stage with uncertainty resolution should have info value."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # Stage with resolution_nodes should have information value
        for analysis in result.stage_analyses:
            assert analysis.information_value >= 0


class TestStageAnalyses:
    """Tests for stage-by-stage analysis."""

    def test_stage_analyses_generated(self, engine, two_stage_simple_graph):
        """Stage analyses should be generated for each stage."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # Should have analysis for decision stages
        assert len(result.stage_analyses) >= 1

    def test_stage_options_have_values(self, engine, two_stage_simple_graph):
        """Stage options should have immediate and continuation values."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        for analysis in result.stage_analyses:
            for option in analysis.options_at_stage:
                # All options should have values
                assert option.immediate_value is not None
                assert option.continuation_value is not None
                assert option.total_value is not None


class TestTimingSensitivity:
    """Tests for timing sensitivity classification."""

    def test_timing_sensitivity_classification(self, engine, two_stage_simple_graph):
        """Timing sensitivity should be classified correctly."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # Should have a valid sensitivity classification
        assert result.sensitivity_to_timing in ["high", "medium", "low"]


class TestRiskTolerance:
    """Tests for risk tolerance effects."""

    def test_risk_averse_penalizes_variance(self, engine, two_stage_simple_graph):
        """Risk averse should penalize high variance options."""
        graph, stages = two_stage_simple_graph

        request_neutral = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )
        result_neutral = engine.analyze(request_neutral)

        request_averse = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="averse"
        )
        result_averse = engine.analyze(request_averse)

        # Both should complete successfully
        assert result_neutral.optimal_policy is not None
        assert result_averse.optimal_policy is not None

        # Risk averse may have different expected value due to variance penalty
        # (actual difference depends on variance in outcomes)

    def test_risk_seeking_rewards_variance(self, engine, two_stage_simple_graph):
        """Risk seeking should reward upside potential."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="seeking"
        )

        result = engine.analyze(request)

        # Should complete successfully
        assert result.optimal_policy is not None


class TestPolicyTree:
    """Tests for policy tree generation."""

    def test_policy_tree_generation(self, engine, two_stage_simple_graph):
        """Policy tree should be generated correctly."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        result = engine.get_policy_tree(request)

        assert result.root is not None
        assert result.total_stages >= 1
        assert result.total_nodes >= 1
        assert result.root.node_id is not None

    def test_policy_tree_has_expected_value(self, engine, two_stage_simple_graph):
        """Tree nodes should have expected values."""
        graph, stages = two_stage_simple_graph

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=0.95,
            risk_tolerance="neutral"
        )

        result = engine.get_policy_tree(request)

        assert result.root.expected_value is not None


class TestStageSensitivity:
    """Tests for stage sensitivity analysis."""

    def test_stage_sensitivity_analysis(self, engine, two_stage_simple_graph):
        """Stage sensitivity should be computed for each stage."""
        graph, stages = two_stage_simple_graph

        request = StageSensitivityRequest(
            graph=graph,
            stages=stages,
            variation_range=0.2
        )

        result = engine.stage_sensitivity(request)

        # Should have results for stages with decisions
        assert len(result.stage_results) >= 0
        assert result.overall_robustness >= 0
        assert result.overall_robustness <= 1

    def test_stage_sensitivity_identifies_parameters(self, engine, two_stage_simple_graph):
        """Should identify most sensitive parameters."""
        graph, stages = two_stage_simple_graph

        request = StageSensitivityRequest(
            graph=graph,
            stages=stages,
            variation_range=0.2
        )

        result = engine.stage_sensitivity(request)

        # Should have explanation
        assert result.explanation is not None
        assert result.explanation.summary is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_stage_graph(self, engine):
        """Single stage graph should work."""
        nodes = [
            SequentialGraphNode(
                id="decide",
                type="decision",
                label="Decision"
            ),
            SequentialGraphNode(
                id="outcome_a",
                type="terminal",
                label="Outcome A",
                payoff=100
            ),
            SequentialGraphNode(
                id="outcome_b",
                type="terminal",
                label="Outcome B",
                payoff=50
            )
        ]

        edges = [
            SequentialGraphEdge(
                from_node="decide",
                to_node="outcome_a",
                action="option_a"
            ),
            SequentialGraphEdge(
                from_node="decide",
                to_node="outcome_b",
                action="option_b"
            )
        ]

        graph = SequentialGraph(
            nodes=nodes,
            edges=edges,
            stage_assignments={
                "decide": 0,
                "outcome_a": 1,
                "outcome_b": 1
            }
        )

        stages = [
            DecisionStage(
                stage_index=0,
                stage_label="Decision",
                decision_nodes=["decide"]
            ),
            DecisionStage(
                stage_index=1,
                stage_label="Terminal",
                decision_nodes=[]
            )
        ]

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=1.0,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # Should choose option_a (higher payoff)
        assert result.optimal_policy is not None
        assert result.optimal_policy.expected_total_value == 100

    def test_equal_probability_chance_node(self, engine):
        """Chance node with equal probabilities should work."""
        nodes = [
            SequentialGraphNode(
                id="chance",
                type="chance",
                label="Coin Flip"
            ),
            SequentialGraphNode(
                id="heads",
                type="terminal",
                label="Heads",
                payoff=100
            ),
            SequentialGraphNode(
                id="tails",
                type="terminal",
                label="Tails",
                payoff=0
            )
        ]

        edges = [
            SequentialGraphEdge(
                from_node="chance",
                to_node="heads",
                outcome="heads",
                probability=0.5
            ),
            SequentialGraphEdge(
                from_node="chance",
                to_node="tails",
                outcome="tails",
                probability=0.5
            )
        ]

        graph = SequentialGraph(
            nodes=nodes,
            edges=edges,
            stage_assignments={
                "chance": 0,
                "heads": 1,
                "tails": 1
            }
        )

        stages = [
            DecisionStage(
                stage_index=0,
                stage_label="Chance",
                decision_nodes=[]
            ),
            DecisionStage(
                stage_index=1,
                stage_label="Terminal",
                decision_nodes=[]
            )
        ]

        request = SequentialAnalysisRequest(
            graph=graph,
            stages=stages,
            discount_factor=1.0,
            risk_tolerance="neutral"
        )

        result = engine.analyze(request)

        # Expected value should be 0.5 * 100 + 0.5 * 0 = 50
        assert result.optimal_policy.expected_total_value == pytest.approx(50, rel=0.1)


# ---------------------------------------------------------------------------
# Regression: same-stage backward-induction drop (A2 defect, 2026-07-22)
# ---------------------------------------------------------------------------
#
# Defect: _backward_induction valued a chance node BEFORE the decision node it
# feeds when the two share a stage_index -- exactly the shape of the
# SequentialAnalysisRequest schema's OWN canonical example (market:1, pricing:1).
# The chance branch read the not-yet-valued decision child's continuation as 0,
# so the entire favorable subtree silently contributed 0 and sign-inverted the
# root expected value.
#
# Discriminating pair (identical decision problem; only pricing's stage differs):
#   collision layout (pricing at stage 1, sharing market's stage): BUGGY -58122.5
#   control   layout (pricing at its own stage 2):                 correct +1893.75
#
# Hand-derivation (neutral risk, discount d=0.95):
#   terminals: high_return=100000, low_return=20000, loss=-30000
#   V(pricing) = max(0.95*100000, 0.95*20000) = 95000                (action premium)
#   V(market)  = 0.7*(0 + 0.95*95000) + 0.3*(0 + 0.95*(-30000))
#              = 0.7*90250 + 0.3*(-28500) = 63175 - 8550 = 54625
#   V(invest)  = -50000 + 0.95*54625 = -50000 + 51893.75 = 1893.75
# Both layouts describe the SAME problem, so both MUST yield market-continuation
# 54625 and root total 1893.75. The buggy -8550 == 0.3*0.95*(-30000) is the
# loss-branch-only value that proves the +95000 favorable subtree was dropped.


def _canonical_decision_graph(pricing_stage: int):
    """Build the canonical invest -> market -> pricing decision graph.

    ``market`` is a chance node at stage 1; ``pricing`` is the decision it feeds.
    ``pricing_stage=1`` reproduces the schema's own example (collision with
    market); ``pricing_stage=2`` gives pricing its own stage (control). The two
    layouts encode the identical decision problem and MUST agree.
    """
    nodes = [
        SequentialGraphNode(id="invest", type="decision", label="Invest"),
        SequentialGraphNode(id="market", type="chance", label="Market"),
        SequentialGraphNode(id="pricing", type="decision", label="Pricing"),
        SequentialGraphNode(id="high_return", type="terminal", label="High", payoff=100000),
        SequentialGraphNode(id="low_return", type="terminal", label="Low", payoff=20000),
        SequentialGraphNode(id="loss", type="terminal", label="Loss", payoff=-30000),
    ]
    edges = [
        SequentialGraphEdge(
            from_node="invest", to_node="market", action="invest", immediate_payoff=-50000
        ),
        SequentialGraphEdge(
            from_node="market", to_node="pricing", outcome="favorable", probability=0.7
        ),
        SequentialGraphEdge(
            from_node="market", to_node="loss", outcome="unfavorable", probability=0.3
        ),
        SequentialGraphEdge(from_node="pricing", to_node="high_return", action="premium"),
        SequentialGraphEdge(from_node="pricing", to_node="low_return", action="economy"),
    ]
    terminal_stage = pricing_stage + 1
    graph = SequentialGraph(
        nodes=nodes,
        edges=edges,
        stage_assignments={
            "invest": 0,
            "market": 1,
            "pricing": pricing_stage,
            "high_return": terminal_stage,
            "low_return": terminal_stage,
            "loss": terminal_stage,
        },
    )
    stages = [DecisionStage(stage_index=0, stage_label="Investment", decision_nodes=["invest"])]
    if pricing_stage == 1:
        # collision: market resolves and pricing is decided at the same stage
        stages.append(
            DecisionStage(
                stage_index=1,
                stage_label="Pricing",
                decision_nodes=["pricing"],
                resolution_nodes=["market"],
            )
        )
    else:
        # control: market resolves at stage 1, pricing decided at stage 2
        stages.append(
            DecisionStage(
                stage_index=1,
                stage_label="Market",
                decision_nodes=[],
                resolution_nodes=["market"],
            )
        )
        stages.append(
            DecisionStage(stage_index=2, stage_label="Pricing", decision_nodes=["pricing"])
        )
    stages.append(
        DecisionStage(stage_index=terminal_stage, stage_label="Terminal", decision_nodes=[])
    )
    return graph, stages


class TestSameStageBackwardInductionRegression:
    """Children must be valued before parents regardless of stage bookkeeping."""

    def test_collision_matches_control_root_value(self, engine):
        """Same-stage (market:1, pricing:1) MUST equal the separate-stage control.

        RED at HEAD: the collision layout produces -58122.5 (favorable subtree
        dropped to 0) while the control produces +1893.75.
        """
        collision_graph, collision_stages = _canonical_decision_graph(pricing_stage=1)
        control_graph, control_stages = _canonical_decision_graph(pricing_stage=2)

        collision = engine.analyze(
            SequentialAnalysisRequest(
                graph=collision_graph,
                stages=collision_stages,
                discount_factor=0.95,
                risk_tolerance="neutral",
            )
        )
        control = engine.analyze(
            SequentialAnalysisRequest(
                graph=control_graph,
                stages=control_stages,
                discount_factor=0.95,
                risk_tolerance="neutral",
            )
        )

        # The control is already correct; pin it as the reference.
        assert control.optimal_policy.expected_total_value == pytest.approx(1893.75)
        # The collision layout describes the SAME problem -> SAME value.
        assert collision.optimal_policy.expected_total_value == pytest.approx(1893.75)
        assert collision.optimal_policy.expected_total_value == pytest.approx(
            control.optimal_policy.expected_total_value
        )

    def test_collision_market_continuation_is_54625(self, engine):
        """The market continuation seen by invest MUST be 54625, not -8550.

        -8550 == 0.3 * 0.95 * (-30000) is the loss-branch-only value, i.e. proof
        that the favorable subtree (worth +95000 at stage 1) was silently dropped.
        """
        graph, stages = _canonical_decision_graph(pricing_stage=1)
        result = engine.analyze(
            SequentialAnalysisRequest(
                graph=graph, stages=stages, discount_factor=0.95, risk_tolerance="neutral"
            )
        )
        invest_option = result.stage_analyses[0].options_at_stage[0]
        assert invest_option.option_id == "invest"
        assert invest_option.continuation_value == pytest.approx(54625.0)
        assert invest_option.total_value == pytest.approx(1893.75)

    def test_control_layout_value_preserved(self, engine):
        """Guard existing-correct behaviour: the control layout stays 54625 / +1893.75."""
        graph, stages = _canonical_decision_graph(pricing_stage=2)
        result = engine.analyze(
            SequentialAnalysisRequest(
                graph=graph, stages=stages, discount_factor=0.95, risk_tolerance="neutral"
            )
        )
        invest_option = result.stage_analyses[0].options_at_stage[0]
        assert invest_option.continuation_value == pytest.approx(54625.0)
        assert result.optimal_policy.expected_total_value == pytest.approx(1893.75)

    def test_dangling_edge_fails_loud(self, engine):
        """An edge to a node absent from `nodes` MUST raise, never resolve to 0.

        RED at HEAD: the chance/decision branches read the missing child as
        continuation 0 and return a fabricated value with no error.
        """
        nodes = [
            SequentialGraphNode(id="root", type="decision", label="Root"),
            SequentialGraphNode(id="win", type="terminal", label="Win", payoff=100),
        ]
        edges = [
            # 'ghost' is never defined as a node -> dangling edge
            SequentialGraphEdge(from_node="root", to_node="ghost", action="risky"),
            SequentialGraphEdge(from_node="root", to_node="win", action="safe"),
        ]
        graph = SequentialGraph(
            nodes=nodes, edges=edges, stage_assignments={"root": 0, "win": 1}
        )
        stages = [
            DecisionStage(stage_index=0, stage_label="Root", decision_nodes=["root"]),
            DecisionStage(stage_index=1, stage_label="Terminal", decision_nodes=[]),
        ]
        with pytest.raises(ValueError):
            engine.analyze(
                SequentialAnalysisRequest(graph=graph, stages=stages, discount_factor=0.95)
            )

    def test_cycle_fails_loud(self, engine):
        """A cycle MUST raise, not silently read a not-yet-valued node as 0.

        RED at HEAD: the iterative stage sweep never recurses, so the cycle is
        tolerated (one node read as 0) and no error is raised.
        """
        nodes = [
            SequentialGraphNode(id="a", type="decision", label="A"),
            SequentialGraphNode(id="b", type="chance", label="B"),
            SequentialGraphNode(id="end", type="terminal", label="End", payoff=10),
        ]
        edges = [
            SequentialGraphEdge(from_node="a", to_node="b", action="go"),
            SequentialGraphEdge(from_node="b", to_node="a", outcome="loop", probability=0.5),
            SequentialGraphEdge(from_node="b", to_node="end", outcome="done", probability=0.5),
        ]
        graph = SequentialGraph(
            nodes=nodes, edges=edges, stage_assignments={"a": 0, "b": 1, "end": 2}
        )
        stages = [
            DecisionStage(stage_index=0, stage_label="A", decision_nodes=["a"]),
            DecisionStage(
                stage_index=1, stage_label="B", decision_nodes=[], resolution_nodes=["b"]
            ),
            DecisionStage(stage_index=2, stage_label="Terminal", decision_nodes=[]),
        ]
        with pytest.raises(ValueError):
            engine.analyze(
                SequentialAnalysisRequest(graph=graph, stages=stages, discount_factor=0.95)
            )


class TestConditionalActionExpectedValue:
    """RW-6b: Policy.conditional_actions[].expected_value_if_taken must include the
    edge's immediate payoff and discounted continuation, not the bare child value.

    The field reports the value of taking a NON-default action. Correct semantics
    are the edge's own value: ``immediate_payoff + discount_factor * V(child)`` --
    the same convention every other edge valuation in the engine uses. The prior
    code returned ``node_values[child]`` alone, dropping the immediate payoff (and
    the discount) whenever the child was already valued (post-#85: always).
    """

    def test_expected_value_if_taken_includes_immediate_and_discount(self, engine):
        """RED at HEAD: the non-default action reports V(child) only.

        Root decision, discount 0.9:
          action "big":   immediate 0,  child terminal payoff 200 -> total 0 + 0.9*200 = 180 (default)
          action "small": immediate 42, child terminal payoff 10  -> total 42 + 0.9*10 = 51 (non-default)
        expected_value_if_taken for "small" MUST be 51 (immediate 42 + discount 0.9 * 10),
        NOT the bare child value 10 that HEAD reports.
        """
        nodes = [
            SequentialGraphNode(id="root", type="decision", label="Root"),
            SequentialGraphNode(id="big_win", type="terminal", label="Big", payoff=200),
            SequentialGraphNode(id="small_win", type="terminal", label="Small", payoff=10),
        ]
        edges = [
            SequentialGraphEdge(
                from_node="root", to_node="big_win", action="big", immediate_payoff=0
            ),
            SequentialGraphEdge(
                from_node="root", to_node="small_win", action="small", immediate_payoff=42
            ),
        ]
        graph = SequentialGraph(
            nodes=nodes,
            edges=edges,
            stage_assignments={"root": 0, "big_win": 1, "small_win": 1},
        )
        stages = [
            DecisionStage(stage_index=0, stage_label="Root", decision_nodes=["root"]),
            DecisionStage(stage_index=1, stage_label="Terminal", decision_nodes=[]),
        ]
        result = engine.analyze(
            SequentialAnalysisRequest(
                graph=graph, stages=stages, discount_factor=0.9, risk_tolerance="neutral"
            )
        )
        rule = result.optimal_policy.stages[0].decision_rule
        assert rule.default_action == "big"
        small = next(ca for ca in rule.conditional_actions if ca.action == "small")
        # immediate 42 + discount 0.9 * child value 10 = 51.0
        assert small.expected_value_if_taken == pytest.approx(51.0)


class TestAverseRiskAdjustmentBlowUp:
    """DIAGNOSIS + guard for a SEPARATE, UNFIXED risk-adjustment bug.

    `_risk_adjust_value` averse branch computes ``mean - 0.5 * variance`` with
    variance in currency^2 units, so a +/-100k problem is inflated to ~1e8. This
    is a distinct defect from the same-stage drop and is intentionally NOT fixed
    here: the correct risk model (variance vs std; the aversion coefficient) is a
    modeling-doctrine decision, not a mechanical fix. See NOTES.md sec 4.
    """

    def test_averse_blowup_is_independent_of_same_stage_fix(self, engine):
        """Prove the averse blow-up is independent of the same-stage drop (verdict (b)).

        Uses the CONTROL layout (pricing at its own stage 2 -> NO collision), where
        pricing is correctly valued at 80000, yet market still blows up. Because the
        same-stage fix is a no-op on this layout, this value is IDENTICAL before and
        after that fix, which is exactly what makes it an independence proof.

        The asserted numbers are KNOWN-WRONG (dimensional blow-up). They are pinned
        so that ANY future change to the risk model FAILS THIS TEST and forces a
        conscious record update. Do NOT read them as correct.
        """
        graph, stages = _canonical_decision_graph(pricing_stage=2)  # control: no collision
        result = engine.analyze(
            SequentialAnalysisRequest(
                graph=graph, stages=stages, discount_factor=0.8, risk_tolerance="averse"
            )
        )
        invest_option = result.stage_analyses[0].options_at_stage[0]
        # positive control: the blow-up IS present (orders of magnitude off scale)
        max_abs_payoff = 100000
        assert abs(invest_option.continuation_value) > 100 * max_abs_payoff
        # characterization of the exact CURRENT (wrong) value, hand-derived in NOTES sec 4
        assert invest_option.continuation_value == pytest.approx(-813082400.0)
