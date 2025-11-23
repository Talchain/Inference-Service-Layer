"""
Unit tests for CausalTransporter service.

Tests transportability analysis including:
- Direct transport (identical domains)
- Selection diagram transport
- Non-transportable scenarios
- Assumption extraction
- Robustness assessment
"""

import pytest

from src.models.requests import DAGStructure, DataSummary, DomainSpec, TransportabilityRequest
from src.models.responses import TransportAssumption, TransportabilityResponse
from src.services.causal_transporter import CausalTransporter


@pytest.fixture
def transporter():
    """Create a CausalTransporter instance."""
    return CausalTransporter()


@pytest.fixture
def simple_dag():
    """Simple causal DAG: Price → Revenue."""
    return DAGStructure(
        nodes=["Price", "Revenue"],
        edges=[["Price", "Revenue"]],
    )


@pytest.fixture
def dag_with_confounder():
    """DAG with confounder: Price ← Brand → Revenue, Price → Revenue."""
    return DAGStructure(
        nodes=["Price", "Revenue", "Brand"],
        edges=[
            ["Brand", "Price"],
            ["Brand", "Revenue"],
            ["Price", "Revenue"],
        ],
    )


@pytest.fixture
def uk_domain(simple_dag):
    """UK domain specification."""
    return DomainSpec(
        name="UK",
        dag=simple_dag,
        data_summary=DataSummary(
            n_samples=1000,
            available_variables=["Price", "Revenue"],
        ),
    )


@pytest.fixture
def germany_domain(simple_dag):
    """Germany domain specification."""
    return DomainSpec(
        name="Germany",
        dag=simple_dag,
        data_summary=DataSummary(
            n_samples=800,
            available_variables=["Price", "Revenue"],
        ),
    )


class TestDirectTransport:
    """Tests for direct transport (identical domains)."""

    def test_direct_transport_simple_dag(self, transporter, uk_domain, germany_domain):
        """Test direct transport with identical simple DAGs."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        assert result.transportable is True
        assert result.method == "direct"
        assert result.formula is not None
        assert "P_target(Revenue|do(Price))" in result.formula
        assert result.robustness in ["robust", "moderate", "fragile"]
        assert len(result.required_assumptions) > 0

    def test_direct_transport_assumptions(self, transporter, uk_domain, germany_domain):
        """Test assumptions for direct transport."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        # Should have same_mechanism assumption
        assumption_types = [a.type for a in result.required_assumptions]
        assert "same_mechanism" in assumption_types

        # Should have common_support assumption
        assert "common_support" in assumption_types

    def test_direct_transport_explanation(self, transporter, uk_domain, germany_domain):
        """Test explanation for direct transport."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        assert result.explanation.summary is not None
        assert "UK" in result.explanation.summary
        assert "Germany" in result.explanation.summary
        assert result.explanation.reasoning is not None
        assert result.explanation.technical_basis is not None


class TestSelectionDiagramTransport:
    """Tests for transport via selection diagrams."""

    def test_selection_diagram_with_explicit_variables(
        self, transporter, uk_domain, germany_domain
    ):
        """Test transport with explicitly specified selection variables."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
            selection_variables=["MarketSize"],
        )

        result = transporter.analyze_transportability(request)

        assert result.transportable is True
        assert result.method == "selection_diagram"
        assert result.formula is not None
        assert "MarketSize" in result.formula

    def test_selection_diagram_assumptions(
        self, transporter, uk_domain, germany_domain
    ):
        """Test assumptions for selection diagram transport."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
            selection_variables=["MarketSize"],
        )

        result = transporter.analyze_transportability(request)

        assumption_types = [a.type for a in result.required_assumptions]
        assert "same_mechanism" in assumption_types
        assert "no_selection_bias" in assumption_types
        assert "measured_selection" in assumption_types
        assert "common_support" in assumption_types

    def test_infer_selection_variables(self, transporter):
        """Test automatic inference of selection variables."""
        # DAG with potential confounders
        dag = DAGStructure(
            nodes=["Price", "Revenue", "Brand", "MarketSize"],
            edges=[
                ["Brand", "Price"],
                ["Brand", "Revenue"],
                ["MarketSize", "Revenue"],
                ["Price", "Revenue"],
            ],
        )

        uk_domain = DomainSpec(name="UK", dag=dag)
        germany_domain = DomainSpec(name="Germany", dag=dag)

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        # Should infer Brand and MarketSize as selection variables
        if result.transportable and result.method == "selection_diagram":
            # Check formula mentions selection variables
            assert "Brand" in result.formula or "MarketSize" in result.formula


class TestNonTransportable:
    """Tests for non-transportable scenarios."""

    def test_different_dag_structures(self, transporter):
        """Test with different DAG structures between domains."""
        uk_dag = DAGStructure(
            nodes=["Price", "Revenue"],
            edges=[["Price", "Revenue"]],
        )

        # Germany has additional edge
        germany_dag = DAGStructure(
            nodes=["Price", "Revenue", "Regulation"],
            edges=[
                ["Price", "Revenue"],
                ["Regulation", "Revenue"],
            ],
        )

        uk_domain = DomainSpec(name="UK", dag=uk_dag)
        germany_domain = DomainSpec(name="Germany", dag=germany_dag)

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        assert result.transportable is False
        assert result.reason is not None
        assert result.suggestions is not None
        assert len(result.suggestions) > 0

    def test_no_causal_path_in_source(self, transporter):
        """Test when there's no causal path in source domain."""
        dag = DAGStructure(
            nodes=["Price", "Revenue", "Quality"],
            edges=[["Quality", "Revenue"]],  # No Price → Revenue
        )

        uk_domain = DomainSpec(name="UK", dag=dag)
        germany_domain = DomainSpec(name="Germany", dag=dag)

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        assert result.transportable is False
        assert result.reason in ["no_source_path", "different_mechanisms"]

    def test_suggestions_for_non_transportable(self, transporter):
        """Test suggestions when effect is not transportable."""
        uk_dag = DAGStructure(
            nodes=["Price", "Revenue"],
            edges=[["Price", "Revenue"]],
        )

        germany_dag = DAGStructure(
            nodes=["Price", "Revenue", "Tax"],
            edges=[
                ["Price", "Revenue"],
                ["Tax", "Price"],
            ],
        )

        uk_domain = DomainSpec(name="UK", dag=uk_dag)
        germany_domain = DomainSpec(name="Germany", dag=germany_dag)

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        if not result.transportable:
            assert result.suggestions is not None
            assert len(result.suggestions) >= 2
            # Should suggest investigating differences
            assert any("differ" in s.lower() for s in result.suggestions)


class TestAssumptions:
    """Tests for assumption extraction and analysis."""

    def test_critical_assumptions_flagged(self, transporter, uk_domain, germany_domain):
        """Test that critical assumptions are properly flagged."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        if result.transportable:
            # Should have at least one critical assumption
            critical_assumptions = [a for a in result.required_assumptions if a.critical]
            assert len(critical_assumptions) > 0

    def test_testable_assumptions_identified(
        self, transporter, uk_domain, germany_domain
    ):
        """Test that testable assumptions are identified."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
            selection_variables=["Brand"],
        )

        result = transporter.analyze_transportability(request)

        if result.transportable:
            testable_assumptions = [a for a in result.required_assumptions if a.testable]
            # Should have at least one testable assumption with selection variables
            assert len(testable_assumptions) > 0

    def test_assumption_descriptions(self, transporter, uk_domain, germany_domain):
        """Test that assumptions have clear descriptions."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        if result.transportable:
            for assumption in result.required_assumptions:
                assert assumption.type is not None
                assert len(assumption.description) > 10
                assert isinstance(assumption.critical, bool)
                assert isinstance(assumption.testable, bool)


class TestRobustness:
    """Tests for robustness assessment."""

    def test_robustness_with_data_summaries(self, transporter):
        """Test robustness assessment when data summaries are provided."""
        dag = DAGStructure(
            nodes=["Price", "Revenue"],
            edges=[["Price", "Revenue"]],
        )

        uk_domain = DomainSpec(
            name="UK",
            dag=dag,
            data_summary=DataSummary(
                n_samples=1000,
                available_variables=["Price", "Revenue"],
            ),
        )

        germany_domain = DomainSpec(
            name="Germany",
            dag=dag,
            data_summary=DataSummary(
                n_samples=800,
                available_variables=["Price", "Revenue"],
            ),
        )

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        assert result.robustness in ["robust", "moderate", "fragile"]

    def test_robustness_levels(self, transporter):
        """Test different robustness levels based on assumptions."""
        dag = DAGStructure(
            nodes=["Price", "Revenue", "Brand"],
            edges=[
                ["Brand", "Price"],
                ["Brand", "Revenue"],
                ["Price", "Revenue"],
            ],
        )

        uk_domain = DomainSpec(name="UK", dag=dag)
        germany_domain = DomainSpec(name="Germany", dag=dag)

        # With selection variables (more assumptions)
        request_with_selection = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
            selection_variables=["Brand"],
        )

        result_with_selection = transporter.analyze_transportability(request_with_selection)

        # Direct transport (fewer assumptions)
        dag_simple = DAGStructure(
            nodes=["Price", "Revenue"],
            edges=[["Price", "Revenue"]],
        )
        uk_simple = DomainSpec(name="UK", dag=dag_simple)
        germany_simple = DomainSpec(name="Germany", dag=dag_simple)

        request_direct = TransportabilityRequest(
            source_domain=uk_simple,
            target_domain=germany_simple,
            treatment="Price",
            outcome="Revenue",
        )

        result_direct = transporter.analyze_transportability(request_direct)

        # Both should succeed but may have different robustness
        if result_with_selection.transportable and result_direct.transportable:
            assert result_with_selection.robustness in ["robust", "moderate", "fragile"]
            assert result_direct.robustness in ["robust", "moderate", "fragile"]


class TestConfidence:
    """Tests for confidence assessment."""

    def test_confidence_with_data(self, transporter):
        """Test confidence when data summaries are available."""
        dag = DAGStructure(
            nodes=["Price", "Revenue"],
            edges=[["Price", "Revenue"]],
        )

        uk_domain = DomainSpec(
            name="UK",
            dag=dag,
            data_summary=DataSummary(
                n_samples=1000,
                available_variables=["Price", "Revenue"],
            ),
        )

        germany_domain = DomainSpec(
            name="Germany",
            dag=dag,
            data_summary=DataSummary(
                n_samples=800,
                available_variables=["Price", "Revenue"],
            ),
        )

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        assert result.confidence in ["high", "medium", "low"]

    def test_confidence_without_data(self, transporter, uk_domain, germany_domain):
        """Test confidence when no data summaries are available."""
        # Remove data summaries
        uk_domain.data_summary = None
        germany_domain.data_summary = None

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        assert result.confidence in ["high", "medium", "low"]


class TestSelectionDiagramConstruction:
    """Tests for selection diagram construction."""

    def test_selection_diagram_nodes(self, transporter):
        """Test that selection diagram includes selection nodes."""
        dag = DAGStructure(
            nodes=["Price", "Revenue", "Brand"],
            edges=[
                ["Brand", "Price"],
                ["Brand", "Revenue"],
                ["Price", "Revenue"],
            ],
        )

        uk_domain = DomainSpec(name="UK", dag=dag)
        germany_domain = DomainSpec(name="Germany", dag=dag)

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
            selection_variables=["Brand"],
        )

        # Build selection diagram (internal method)
        selection_diagram = transporter._build_selection_diagram(request)

        # Should have original nodes plus selection nodes
        assert "Price" in selection_diagram.nodes
        assert "Revenue" in selection_diagram.nodes
        assert "Brand" in selection_diagram.nodes
        assert "S_Brand" in selection_diagram.nodes

    def test_selection_diagram_edges(self, transporter):
        """Test that selection diagram includes correct edges."""
        dag = DAGStructure(
            nodes=["Price", "Revenue", "Brand"],
            edges=[
                ["Brand", "Price"],
                ["Price", "Revenue"],
            ],
        )

        uk_domain = DomainSpec(name="UK", dag=dag)
        germany_domain = DomainSpec(name="Germany", dag=dag)

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
            selection_variables=["Brand"],
        )

        selection_diagram = transporter._build_selection_diagram(request)

        # Should have selection node pointing to selected variable
        assert selection_diagram.has_edge("S_Brand", "Brand")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_selection_variables(self, transporter, uk_domain, germany_domain):
        """Test with empty selection variables list."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
            selection_variables=[],
        )

        result = transporter.analyze_transportability(request)

        # Should still work (falls back to direct transport)
        assert result is not None
        assert isinstance(result.transportable, bool)

    def test_single_node_dag(self, transporter):
        """Test with single-node DAG (pathological case)."""
        dag = DAGStructure(
            nodes=["Price"],
            edges=[],
        )

        uk_domain = DomainSpec(name="UK", dag=dag)
        germany_domain = DomainSpec(name="Germany", dag=dag)

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Price",  # Same as treatment (pathological)
        )

        # Should handle gracefully
        result = transporter.analyze_transportability(request)
        assert result is not None

    def test_large_dag(self, transporter):
        """Test with larger DAG."""
        nodes = [f"Var{i}" for i in range(10)]
        edges = [[f"Var{i}", f"Var{i+1}"] for i in range(9)]

        dag = DAGStructure(nodes=nodes, edges=edges)

        uk_domain = DomainSpec(name="UK", dag=dag)
        germany_domain = DomainSpec(name="Germany", dag=dag)

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Var0",
            outcome="Var9",
        )

        result = transporter.analyze_transportability(request)

        assert result is not None
        assert isinstance(result.transportable, bool)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_deterministic_results(self, transporter, uk_domain, germany_domain):
        """Test that same input produces same output."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result1 = transporter.analyze_transportability(request)
        result2 = transporter.analyze_transportability(request)

        assert result1.transportable == result2.transportable
        assert result1.method == result2.method
        assert result1.formula == result2.formula
        assert result1.robustness == result2.robustness
        assert result1.confidence == result2.confidence


class TestExplanations:
    """Tests for explanation generation."""

    def test_transportable_explanation_completeness(
        self, transporter, uk_domain, germany_domain
    ):
        """Test that transportable case has complete explanation."""
        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        if result.transportable:
            assert result.explanation.summary is not None
            assert len(result.explanation.summary) > 20
            assert result.explanation.reasoning is not None
            assert len(result.explanation.reasoning) > 20
            assert result.explanation.technical_basis is not None
            assert len(result.explanation.assumptions) > 0

    def test_non_transportable_explanation_completeness(self, transporter):
        """Test that non-transportable case has complete explanation."""
        uk_dag = DAGStructure(
            nodes=["Price", "Revenue"],
            edges=[["Price", "Revenue"]],
        )

        germany_dag = DAGStructure(
            nodes=["Price", "Revenue", "Regulation"],
            edges=[
                ["Price", "Revenue"],
                ["Regulation", "Revenue"],
            ],
        )

        uk_domain = DomainSpec(name="UK", dag=uk_dag)
        germany_domain = DomainSpec(name="Germany", dag=germany_dag)

        request = TransportabilityRequest(
            source_domain=uk_domain,
            target_domain=germany_domain,
            treatment="Price",
            outcome="Revenue",
        )

        result = transporter.analyze_transportability(request)

        if not result.transportable:
            assert result.explanation.summary is not None
            assert result.explanation.reasoning is not None
            assert result.explanation.technical_basis is not None
            assert len(result.explanation.assumptions) > 0
