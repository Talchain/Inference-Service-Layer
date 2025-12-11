"""
Unit tests for FACTOR node kind in GraphV1 schema.

Tests that 'factor' is a valid NodeKind for representing external
variables/uncertainties (chance nodes) that are not controllable.
"""

import pytest
from pydantic import ValidationError

from src.models.shared import NodeKind, GraphNodeV1, GraphV1, GraphEdgeV1


class TestFactorNodeKind:
    """Test cases for FACTOR node kind."""

    def test_factor_is_valid_node_kind(self):
        """FACTOR should be a valid NodeKind enum value."""
        assert NodeKind.FACTOR.value == "factor"
        assert NodeKind.FACTOR in NodeKind

    def test_all_node_kinds_present(self):
        """All expected node kinds should be present in the enum."""
        expected_kinds = {"goal", "decision", "option", "outcome", "risk", "action", "factor"}
        actual_kinds = {kind.value for kind in NodeKind}
        assert expected_kinds == actual_kinds

    def test_factor_node_creation(self):
        """GraphNodeV1 should accept 'factor' as a valid kind."""
        node = GraphNodeV1(
            id="market_conditions",
            kind=NodeKind.FACTOR,
            label="Market Conditions",
            body="External market factors affecting the decision",
            belief=0.6,
        )
        assert node.kind == NodeKind.FACTOR
        assert node.id == "market_conditions"

    def test_factor_node_from_string(self):
        """GraphNodeV1 should accept 'factor' string for kind field."""
        node = GraphNodeV1(
            id="competitor_response",
            kind="factor",  # type: ignore - testing string coercion
            label="Competitor Response",
        )
        assert node.kind == NodeKind.FACTOR

    def test_graph_with_factor_nodes(self):
        """GraphV1 should accept graphs containing factor nodes."""
        graph = GraphV1(
            nodes=[
                GraphNodeV1(
                    id="decision_1",
                    kind=NodeKind.DECISION,
                    label="Launch Product",
                ),
                GraphNodeV1(
                    id="market_factor",
                    kind=NodeKind.FACTOR,
                    label="Market Demand",
                    belief=0.7,
                ),
                GraphNodeV1(
                    id="outcome_1",
                    kind=NodeKind.OUTCOME,
                    label="Revenue",
                ),
            ],
            edges=[
                GraphEdgeV1(**{"from": "decision_1", "to": "outcome_1", "weight": 2.0}),
                GraphEdgeV1(**{"from": "market_factor", "to": "outcome_1", "weight": 1.5}),
            ],
        )

        # Verify factor node is in graph
        factor_nodes = [n for n in graph.nodes if n.kind == NodeKind.FACTOR]
        assert len(factor_nodes) == 1
        assert factor_nodes[0].id == "market_factor"

    def test_factor_with_belief_probability(self):
        """Factor nodes should support belief probability (0-1)."""
        node = GraphNodeV1(
            id="economic_uncertainty",
            kind=NodeKind.FACTOR,
            label="Economic Uncertainty",
            belief=0.85,
        )
        assert node.belief == 0.85

    def test_factor_without_belief(self):
        """Factor nodes should work without explicit belief."""
        node = GraphNodeV1(
            id="regulatory_change",
            kind=NodeKind.FACTOR,
            label="Regulatory Change",
        )
        assert node.belief is None

    def test_invalid_node_kind_rejected(self):
        """Invalid node kinds should be rejected."""
        with pytest.raises(ValidationError):
            GraphNodeV1(
                id="invalid_node",
                kind="invalid_kind",  # type: ignore
                label="Invalid Node",
            )


class TestFactorNodeSemantics:
    """Test semantic usage of factor nodes."""

    def test_factor_as_chance_node(self):
        """Factor nodes represent uncontrollable chance events."""
        # In decision theory, factor nodes are "chance nodes" - events
        # that happen based on probability, not decision maker's control

        factor = GraphNodeV1(
            id="weather",
            kind=NodeKind.FACTOR,
            label="Weather Conditions",
            body="External weather that affects outdoor event success",
            belief=0.3,  # 30% chance of bad weather
        )

        # Factor should have belief (probability)
        assert factor.belief is not None
        # Factor kind should be explicitly FACTOR
        assert factor.kind == NodeKind.FACTOR
        assert factor.kind.value == "factor"

    def test_factor_vs_action_distinction(self):
        """Factor (uncontrollable) vs Action (controllable) distinction."""
        # Factor: External, uncontrollable
        factor = GraphNodeV1(
            id="competitor_pricing",
            kind=NodeKind.FACTOR,
            label="Competitor Pricing",
            body="What competitors will charge - not in our control",
        )

        # Action: Internal, controllable
        action = GraphNodeV1(
            id="our_pricing",
            kind=NodeKind.ACTION,
            label="Our Pricing Strategy",
            body="Our pricing decision - fully controllable",
        )

        assert factor.kind != action.kind
        assert factor.kind == NodeKind.FACTOR
        assert action.kind == NodeKind.ACTION
