"""
I.5 — exists_probability default warning tests.

When an edge's exists_probability is not provided and defaults to 0.8,
an InferenceWarning with code EXISTS_PROBABILITY_DEFAULT must be emitted.
"""

import pytest

from src.constants import DEFAULT_EXISTS_PROBABILITY
from src.models.response_v2 import InferenceWarning
from src.models.robustness_v2 import EdgeV2, GraphV2, NodeV2, StrengthDistribution


def _make_edge(from_: str, to: str, exists_probability=None):
    """Helper: build an EdgeV2, optionally omitting exists_probability."""
    kwargs = {
        "from": from_,
        "to": to,
        "strength": StrengthDistribution(mean=0.5, std=0.1),
    }
    if exists_probability is not None:
        kwargs["exists_probability"] = exists_probability
    return EdgeV2(**kwargs)


def _make_graph(edges_spec):
    """Helper: build a GraphV2 from (from_, to, exists_prob_or_None) tuples."""
    node_ids = set()
    for from_, to, _ in edges_spec:
        node_ids.add(from_)
        node_ids.add(to)
    nodes = [NodeV2(id=nid, kind="factor", label=nid.title()) for nid in sorted(node_ids)]
    edges = [_make_edge(f, t, ep) for f, t, ep in edges_spec]
    return GraphV2(nodes=nodes, edges=edges)


class TestExistsProbabilityDefaultWarning:
    """Edge without explicit exists_probability emits EXISTS_PROBABILITY_DEFAULT."""

    def test_edge_without_exists_probability_emits_warning(self) -> None:
        """Edge missing exists_probability → warning emitted."""
        edge = _make_edge("a", "b")  # no exists_probability
        assert edge._exists_probability_default_warning is not None
        w = edge._exists_probability_default_warning
        assert w.code == "EXISTS_PROBABILITY_DEFAULT"
        assert "a" in w.field and "b" in w.field
        assert w.detail["default_value"] == DEFAULT_EXISTS_PROBABILITY

    def test_edge_with_explicit_exists_probability_no_warning(self) -> None:
        """Edge with explicit exists_probability → no warning."""
        edge = _make_edge("a", "b", exists_probability=0.9)
        assert edge._exists_probability_default_warning is None

    def test_edge_with_explicit_default_value_no_warning(self) -> None:
        """Edge explicitly setting 0.8 → no warning (it was deliberate)."""
        edge = _make_edge("a", "b", exists_probability=0.8)
        assert edge._exists_probability_default_warning is None

    def test_graph_collects_default_warnings(self) -> None:
        """GraphV2.collect_parse_warnings() includes exists_probability defaults."""
        graph = _make_graph([
            ("a", "b", None),   # default → warning
            ("b", "c", 0.9),   # explicit → no warning
        ])
        warnings = graph.collect_parse_warnings()
        ep_warnings = [w for w in warnings if w.code == "EXISTS_PROBABILITY_DEFAULT"]
        assert len(ep_warnings) == 1
        assert "a" in ep_warnings[0].field
        assert "b" in ep_warnings[0].field

    def test_multiple_default_edges_one_warning_per_edge(self) -> None:
        """Multiple edges with defaults → one warning per edge."""
        graph = _make_graph([
            ("a", "b", None),   # default
            ("b", "c", None),   # default
            ("c", "d", 1.0),   # explicit
        ])
        warnings = graph.collect_parse_warnings()
        ep_warnings = [w for w in warnings if w.code == "EXISTS_PROBABILITY_DEFAULT"]
        assert len(ep_warnings) == 2

    def test_warning_detail_includes_edge_identifiers(self) -> None:
        """Warning detail includes edge_from and edge_to for identification."""
        edge = _make_edge("source_node", "target_node")
        w = edge._exists_probability_default_warning
        assert w is not None
        assert w.detail["edge_from"] == "source_node"
        assert w.detail["edge_to"] == "target_node"

    def test_same_edge_constructed_twice_is_deduplicated(self) -> None:
        """Same edge data parsed twice → each gets its own warning (no cross-contamination)."""
        edge1 = _make_edge("a", "b")
        edge2 = _make_edge("a", "b")
        # Each has its own warning instance (no shared state)
        w1 = edge1._exists_probability_default_warning
        w2 = edge2._exists_probability_default_warning
        assert w1 is not None
        assert w2 is not None
        assert w1 is not w2  # distinct instances

    def test_warning_propagates_through_analyzer(self) -> None:
        """Full analyzer run: edge without exists_probability → warning in response."""
        from src.models.robustness_v2 import (
            InterventionOption,
            ObservedState,
            RobustnessRequestV2,
        )
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        graph = GraphV2(
            nodes=[
                NodeV2(
                    id="factor1",
                    kind="factor",
                    label="Factor 1",
                    observed_state=ObservedState(value=0.5),
                ),
                NodeV2(id="goal", kind="outcome", label="Goal"),
            ],
            edges=[
                EdgeV2(
                    **{"from": "factor1", "to": "goal"},
                    # exists_probability intentionally omitted → should trigger warning
                    strength=StrengthDistribution(mean=0.8, std=0.1),
                ),
            ],
        )
        request = RobustnessRequestV2(
            request_id="exists-prob-warning-test",
            graph=graph,
            options=[InterventionOption(id="opt1", label="Opt 1", interventions={})],
            goal_node_id="goal",
            n_samples=100,
            seed=42,
        )
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        ep_warnings = [
            w for w in response.inference_warnings
            if w.code == "EXISTS_PROBABILITY_DEFAULT"
        ]
        assert len(ep_warnings) == 1
        assert ep_warnings[0].detail["edge_from"] == "factor1"
        assert ep_warnings[0].detail["edge_to"] == "goal"
