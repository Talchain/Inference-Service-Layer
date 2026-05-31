"""
Tests for Track S — Path-Specific Effect Decomposition (Brief A).

Analytic path tracing for the recommended option's retained intervention targets.
Per-edge coefficient and path strength match `_compute_structural_influence`:
`strength.mean * exists_probability` (signed), multiplied along the path; totals stay
SIGNED (no abs, no normalization).  Path effects are NOT scaled by intervention
magnitude — this is a structural decomposition of the modelled effect, not a
real-world causal claim and not an option-level effect-size estimate.

Covers the Brief A test matrix:
  1. single path -> contribution 1.0
  2. two parallel paths -> shares sum to 1.0, proportional to coeff products
  3. opposing-sign paths -> signed shares, do not sum to abs 1.0
  4. near-zero total -> indeterminate, no ratios, total_effect present
  5. shared mediator -> pooled, additive
  6. determinism -> same request twice, identical decomposition
  7. performance -> 10-factor multi-path graph, decomposition < 500ms
  8. decision/option nodes excluded from computed path interiors (post-filter)
  9. bidirected/confounding edges excluded
 10. flag off -> field absent
 11. flag on, no reachable path -> [] paths, container present
 12. zero-coefficient edge
 13. top-3 truncation, total over ALL paths
 14. mechanism wording contract (no "%", no "cause")
 15. option as metadata / retained entry node
 16. structural, NOT intervention-magnitude-weighted (explicit non-scaling contract)
"""

import math
import time

import pytest

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    PathContribution,
    PathDecomposition,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import (
    MAX_DECOMPOSITION_PATHS,
    RobustnessAnalyzerV2,
)


# =============================================================================
# Helpers
# =============================================================================


def _node(node_id: str, kind: str = "factor") -> NodeV2:
    return NodeV2(id=node_id, kind=kind, label=node_id.replace("_", " ").title())


def _edge(
    from_id: str,
    to_id: str,
    mean: float,
    exists_prob: float = 1.0,
    edge_type: str | None = None,
) -> EdgeV2:
    """Edge with exists_probability defaulting to 1.0 so coeff == mean for clear arithmetic."""
    kwargs = {
        "from": from_id,
        "to": to_id,
        "exists_probability": exists_prob,
        "strength": StrengthDistribution(mean=mean, std=0.1),
    }
    if edge_type is not None:
        kwargs["edge_type"] = edge_type
    return EdgeV2(**kwargs)


def _request(
    graph: GraphV2,
    options: list,
    goal_node_id: str,
    *,
    include_path_decomposition: bool = False,
    n_samples: int = 200,
    seed: int = 42,
) -> RobustnessRequestV2:
    return RobustnessRequestV2(
        request_id="test-path-decomp",
        graph=graph,
        options=options,
        goal_node_id=goal_node_id,
        n_samples=n_samples,
        seed=seed,
        include_path_decomposition=include_path_decomposition,
    )


def _analyzer() -> RobustnessAnalyzerV2:
    return RobustnessAnalyzerV2()


def _layered_graph_widths(widths: list) -> GraphV2:
    """Fully-connected layered DAG: src -> L0(widths[0]) -> ... -> goal.

    Number of simple src->goal paths is the product of widths (combinatorial via shared
    mediators), exercising the worst case for path enumeration. Variable per-layer widths
    let a test hit an exact path count (e.g. [5,5,5,5,4,4,2] -> 20,000) within the
    50-node/200-edge schema limits.
    """
    nodes = [_node("src")]
    for i, w in enumerate(widths):
        nodes.extend(_node(f"n{i}_{j}") for j in range(w))
    nodes.append(_node("goal", kind="outcome"))

    edges = [_edge("src", f"n0_{j}", mean=0.5) for j in range(widths[0])]
    for i in range(len(widths) - 1):
        for a in range(widths[i]):
            for b in range(widths[i + 1]):
                edges.append(_edge(f"n{i}_{a}", f"n{i + 1}_{b}", mean=0.5))
    last = len(widths) - 1
    edges.extend(_edge(f"n{last}_{j}", "goal", mean=0.5) for j in range(widths[last]))
    return GraphV2(nodes=nodes, edges=edges)


def _layered_graph(width: int, layers: int) -> GraphV2:
    """Uniform-width fully-connected layered DAG: width**layers simple paths."""
    return _layered_graph_widths([width] * layers)


# =============================================================================
# 1. Single path -> signed_contribution == 1.0
# =============================================================================


def test_single_path_contribution_is_one():
    graph = GraphV2(
        nodes=[_node("price"), _node("revenue", kind="outcome")],
        edges=[_edge("price", "revenue", mean=0.5)],
    )
    options = [InterventionOption(id="opt_a", label="A", interventions={"price": 0.3})]
    request = _request(graph, options, "revenue")

    dec = _analyzer()._compute_path_decomposition(request, "opt_a", graph)

    assert isinstance(dec, PathDecomposition)
    assert dec.recommended_option_id == "opt_a"
    assert dec.entry_nodes == ["price"]
    assert dec.truncated is False
    assert dec.path_count == 1
    assert len(dec.paths) == 1
    p = dec.paths[0]
    assert p.path == ["price", "revenue"]
    assert p.path_effect == pytest.approx(0.5)
    assert p.total_effect == pytest.approx(0.5)
    assert p.signed_contribution == pytest.approx(1.0)
    assert p.status == "computed"


# =============================================================================
# 2. Two parallel paths -> shares sum to 1.0, proportional to coeff products
# =============================================================================


def test_two_parallel_paths_sum_to_one():
    graph = GraphV2(
        nodes=[_node("price"), _node("mid"), _node("revenue", kind="outcome")],
        edges=[
            _edge("price", "mid", mean=0.5),
            _edge("mid", "revenue", mean=0.6),  # path: 0.30
            _edge("price", "revenue", mean=0.1),  # path: 0.10
        ],
    )
    options = [InterventionOption(id="opt_a", label="A", interventions={"price": 0.3})]
    request = _request(graph, options, "revenue")

    dec = _analyzer()._compute_path_decomposition(request, "opt_a", graph)

    assert len(dec.paths) == 2
    effects = {tuple(p.path): p.path_effect for p in dec.paths}
    assert effects[("price", "mid", "revenue")] == pytest.approx(0.30)
    assert effects[("price", "revenue")] == pytest.approx(0.10)
    assert dec.paths[0].total_effect == pytest.approx(0.40)
    # Signed shares sum to 1.0 and are proportional to the path products.
    assert sum(p.signed_contribution for p in dec.paths) == pytest.approx(1.0)
    shares = {tuple(p.path): p.signed_contribution for p in dec.paths}
    assert shares[("price", "mid", "revenue")] == pytest.approx(0.75)
    assert shares[("price", "revenue")] == pytest.approx(0.25)


# =============================================================================
# 3. Opposing-sign paths -> signed shares, do NOT sum to abs 1.0
# =============================================================================


def test_opposing_sign_paths():
    graph = GraphV2(
        nodes=[_node("price"), _node("mid"), _node("revenue", kind="outcome")],
        edges=[
            _edge("price", "revenue", mean=0.6),  # +0.60
            _edge("price", "mid", mean=0.4),
            _edge("mid", "revenue", mean=-0.5),  # 0.4 * -0.5 = -0.20
        ],
    )
    options = [InterventionOption(id="opt_a", label="A", interventions={"price": 0.3})]
    request = _request(graph, options, "revenue")

    dec = _analyzer()._compute_path_decomposition(request, "opt_a", graph)

    assert len(dec.paths) == 2
    effects = {tuple(p.path): p.path_effect for p in dec.paths}
    assert effects[("price", "revenue")] == pytest.approx(0.60)
    assert effects[("price", "mid", "revenue")] == pytest.approx(-0.20)
    assert dec.paths[0].total_effect == pytest.approx(0.40)
    # Signed shares: one positive, one negative; signed sum is 1.0 but abs-sum is not.
    shares = {tuple(p.path): p.signed_contribution for p in dec.paths}
    assert shares[("price", "revenue")] > 0
    assert shares[("price", "mid", "revenue")] < 0
    assert sum(dec.paths[i].signed_contribution for i in range(2)) == pytest.approx(1.0)
    abs_sum = sum(abs(p.signed_contribution) for p in dec.paths)
    assert abs_sum != pytest.approx(1.0)
    assert abs_sum == pytest.approx(2.0)  # |1.5| + |-0.5|


# =============================================================================
# 4. Near-zero total -> indeterminate, no ratios, total_effect present
# =============================================================================


def test_near_zero_total_is_indeterminate():
    graph = GraphV2(
        nodes=[_node("price"), _node("mid"), _node("revenue", kind="outcome")],
        edges=[
            _edge("price", "revenue", mean=0.5),  # +0.5
            _edge("price", "mid", mean=1.0),
            _edge("mid", "revenue", mean=-0.5),  # 1.0 * -0.5 = -0.5  -> total 0.0
        ],
    )
    options = [InterventionOption(id="opt_a", label="A", interventions={"price": 0.3})]
    request = _request(graph, options, "revenue")

    dec = _analyzer()._compute_path_decomposition(request, "opt_a", graph)

    assert len(dec.paths) == 2
    assert all(p.status == "indeterminate" for p in dec.paths)
    assert all(p.signed_contribution is None for p in dec.paths)
    # path_effect and total_effect are still emitted for debugging.
    assert all(p.total_effect == pytest.approx(0.0) for p in dec.paths)
    assert {round(p.path_effect, 6) for p in dec.paths} == {0.5, -0.5}


# =============================================================================
# 5. Shared mediator -> pooled across entry nodes, additive
# =============================================================================


def test_shared_mediator_pooled_additive():
    graph = GraphV2(
        nodes=[_node("a"), _node("b"), _node("m"), _node("goal", kind="outcome")],
        edges=[
            _edge("a", "m", mean=0.5),
            _edge("b", "m", mean=0.3),
            _edge("m", "goal", mean=0.4),
        ],
    )
    options = [InterventionOption(id="opt_a", label="A", interventions={"a": 0.5, "b": 0.5})]
    request = _request(graph, options, "goal")

    dec = _analyzer()._compute_path_decomposition(request, "opt_a", graph)

    assert dec.entry_nodes == ["a", "b"]
    assert len(dec.paths) == 2
    effects = {tuple(p.path): p.path_effect for p in dec.paths}
    assert effects[("a", "m", "goal")] == pytest.approx(0.20)
    assert effects[("b", "m", "goal")] == pytest.approx(0.12)
    assert dec.paths[0].total_effect == pytest.approx(0.32)
    assert sum(p.signed_contribution for p in dec.paths) == pytest.approx(1.0)


# =============================================================================
# 6. Determinism -> same request twice, identical decomposition
# =============================================================================


def test_determinism_same_request_identical_decomposition():
    def build():
        graph = GraphV2(
            nodes=[_node("price"), _node("mid"), _node("revenue", kind="outcome")],
            edges=[
                _edge("price", "mid", mean=0.5),
                _edge("mid", "revenue", mean=0.6),
                _edge("price", "revenue", mean=0.1),
            ],
        )
        options = [
            InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
            InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
        ]
        return _request(graph, options, "revenue", include_path_decomposition=True)

    r1 = _analyzer().analyze(build())
    r2 = _analyzer().analyze(build())

    assert r1.path_decomposition is not None
    assert r2.path_decomposition is not None
    assert r1.path_decomposition == r2.path_decomposition


# =============================================================================
# 7. Performance -> 10-factor multi-path graph, decomposition < 500ms
# =============================================================================


def test_performance_10_factor_multipath_under_500ms():
    # 10 factor nodes (f0..f9) + goal; three disjoint chains from the entry f0.
    nodes = [_node(f"f{i}") for i in range(10)] + [_node("goal", kind="outcome")]
    edges = [
        # chain 1: f0 -> f1 -> f2 -> f3 -> goal
        _edge("f0", "f1", mean=0.5),
        _edge("f1", "f2", mean=0.5),
        _edge("f2", "f3", mean=0.5),
        _edge("f3", "goal", mean=0.5),
        # chain 2: f0 -> f4 -> f5 -> f6 -> goal
        _edge("f0", "f4", mean=0.4),
        _edge("f4", "f5", mean=0.4),
        _edge("f5", "f6", mean=0.4),
        _edge("f6", "goal", mean=0.4),
        # chain 3: f0 -> f7 -> f8 -> f9 -> goal
        _edge("f0", "f7", mean=0.3),
        _edge("f7", "f8", mean=0.3),
        _edge("f8", "f9", mean=0.3),
        _edge("f9", "goal", mean=0.3),
    ]
    graph = GraphV2(nodes=nodes, edges=edges)
    options = [InterventionOption(id="opt_a", label="A", interventions={"f0": 0.5})]
    request = _request(graph, options, "goal")
    analyzer = _analyzer()

    t0 = time.perf_counter()
    dec = analyzer._compute_path_decomposition(request, "opt_a", graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert len(dec.paths) == 3  # three distinct chains, top-3
    assert elapsed_ms < 500, f"path decomposition took {elapsed_ms:.2f}ms (>500ms)"


# =============================================================================
# 7b. Combinatorial layered DAG UNDER the path budget -> exact, fast
# =============================================================================


def test_combinatorial_under_cap_exact_and_fast():
    # width=5, layers=6 -> 5**6 = 15,625 simple paths (combinatorial shared mediators),
    # below MAX_DECOMPOSITION_PATHS. 32 nodes / 135 edges, within schema limits.
    graph = _layered_graph(width=5, layers=6)
    assert len(graph.nodes) <= 50 and len(graph.edges) <= 200
    options = [InterventionOption(id="opt_a", label="A", interventions={"src": 0.5})]
    request = _request(graph, options, "goal")
    analyzer = _analyzer()

    t0 = time.perf_counter()
    dec = analyzer._compute_path_decomposition(request, "opt_a", graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert dec.truncated is False
    assert dec.path_count == 5**6  # all paths enumerated exactly
    assert len(dec.paths) == 3  # top-3 of the combinatorial set
    assert elapsed_ms < 500, f"combinatorial under-cap took {elapsed_ms:.1f}ms (>500ms)"


# =============================================================================
# 7c. Combinatorial layered DAG OVER the path budget -> graceful truncation, fast
# =============================================================================


def test_combinatorial_over_cap_truncates_fast_and_deterministically():
    # width=5, layers=7 -> 5**7 = 78,125 simple paths, exceeding the budget.
    # 37 nodes / 160 edges, within schema limits. Must degrade gracefully AND stay fast.
    graph = _layered_graph(width=5, layers=7)
    assert len(graph.nodes) <= 50 and len(graph.edges) <= 200
    options = [InterventionOption(id="opt_a", label="A", interventions={"src": 0.5})]
    request = _request(graph, options, "goal")
    analyzer = _analyzer()

    t0 = time.perf_counter()
    dec = analyzer._compute_path_decomposition(request, "opt_a", graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert dec.truncated is True
    assert dec.paths == []  # no misleading partial top-3
    assert dec.path_count == MAX_DECOMPOSITION_PATHS  # stopped at the budget
    assert dec.entry_nodes == ["src"]  # option metadata still reported
    assert elapsed_ms < 500, f"combinatorial over-cap took {elapsed_ms:.1f}ms (>500ms)"

    # Truncation is deterministic (count-based, not wall-clock): identical result twice.
    dec2 = analyzer._compute_path_decomposition(request, "opt_a", graph)
    assert dec == dec2


# =============================================================================
# 7d. Boundary: EXACTLY the path budget -> not truncated, exact top-3
# =============================================================================


def test_exact_cap_boundary_is_not_truncated():
    # 5*5*5*5*4*4*2 == 20,000 == MAX_DECOMPOSITION_PATHS exactly. Reaching the budget is
    # not exceeding it: the graph is fully enumerable and must produce an exact top-3,
    # not a truncated/empty result. Regression for the >= vs > off-by-one.
    widths = [5, 5, 5, 5, 4, 4, 2]
    assert math.prod(widths) == MAX_DECOMPOSITION_PATHS
    graph = _layered_graph_widths(widths)
    assert len(graph.nodes) <= 50 and len(graph.edges) <= 200
    options = [InterventionOption(id="opt_a", label="A", interventions={"src": 0.5})]
    request = _request(graph, options, "goal")
    analyzer = _analyzer()

    t0 = time.perf_counter()
    dec = analyzer._compute_path_decomposition(request, "opt_a", graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert dec.truncated is False  # reached the budget but did NOT exceed it
    assert dec.path_count == MAX_DECOMPOSITION_PATHS  # exact count, equal to the cap
    assert len(dec.paths) == 3  # exact top-3 still produced
    assert elapsed_ms < 500, f"exact-cap took {elapsed_ms:.1f}ms (>500ms)"


# =============================================================================
# 8. Decision/option nodes excluded from computed path interiors (post-filter)
# =============================================================================


def test_decision_node_excluded_from_interior_end_to_end():
    # A decision node sits on a would-be path; filter_inference_graph removes it and
    # its edges before analysis, so it must never appear in any computed path.
    graph = GraphV2(
        nodes=[
            _node("price"),
            _node("strategy", kind="decision"),
            _node("revenue", kind="outcome"),
        ],
        edges=[
            _edge("price", "revenue", mean=0.5),
            _edge("strategy", "revenue", mean=0.9),  # removed with the decision node
        ],
    )
    options = [
        InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
        InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
    ]
    request = _request(graph, options, "revenue", include_path_decomposition=True)

    response = _analyzer().analyze(request)

    dec = response.path_decomposition
    assert dec is not None
    assert dec.paths, "expected a path from the retained factor 'price'"
    for p in dec.paths:
        assert "strategy" not in p.path
        assert p.path[0] == "price"  # path starts at the retained intervention target


# =============================================================================
# 9. Bidirected / confounding edges excluded
# =============================================================================


def test_bidirected_edge_excluded():
    graph = GraphV2(
        nodes=[_node("price"), _node("mid"), _node("revenue", kind="outcome")],
        edges=[
            _edge("price", "revenue", mean=0.3),  # directed -> kept
            _edge("price", "mid", mean=0.5),  # directed -> kept
            _edge("mid", "revenue", mean=0.8, edge_type="bidirected"),  # excluded
        ],
    )
    options = [InterventionOption(id="opt_a", label="A", interventions={"price": 0.3})]
    request = _request(graph, options, "revenue")

    dec = _analyzer()._compute_path_decomposition(request, "opt_a", graph)

    # The price->mid->revenue path is broken at the bidirected edge; only price->revenue.
    assert len(dec.paths) == 1
    assert dec.paths[0].path == ["price", "revenue"]
    assert dec.paths[0].path_effect == pytest.approx(0.3)


# =============================================================================
# 10. Flag off -> field absent
# =============================================================================


def test_flag_off_field_absent():
    graph = GraphV2(
        nodes=[_node("price"), _node("revenue", kind="outcome")],
        edges=[_edge("price", "revenue", mean=0.5)],
    )
    options = [
        InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
        InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
    ]
    request = _request(graph, options, "revenue", include_path_decomposition=False)

    response = _analyzer().analyze(request)

    assert response.path_decomposition is None
    dump = response.model_dump(by_alias=True, exclude_none=True)
    assert "path_decomposition" not in dump


# =============================================================================
# 11. Flag on, no reachable path -> [] paths, container present
# =============================================================================


def test_flag_on_no_reachable_path_returns_empty_container():
    # 'price' (intervention) has no path to 'revenue'; revenue is fed by an unrelated root.
    graph = GraphV2(
        nodes=[_node("price"), _node("other"), _node("revenue", kind="outcome")],
        edges=[_edge("other", "revenue", mean=0.5)],
    )
    options = [
        InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
        InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
    ]
    request = _request(graph, options, "revenue", include_path_decomposition=True)

    response = _analyzer().analyze(request)

    dec = response.path_decomposition
    assert dec is not None
    assert dec.paths == []
    assert dec.recommended_option_id == response.recommended_option_id
    dump = response.model_dump(by_alias=True, exclude_none=True)
    assert "path_decomposition" in dump  # container present even with no paths


# =============================================================================
# 12. Zero-coefficient edge
# =============================================================================


def test_zero_coefficient_edge():
    graph = GraphV2(
        nodes=[_node("price"), _node("mid"), _node("revenue", kind="outcome")],
        edges=[
            _edge("price", "revenue", mean=0.0),  # coeff 0.0
            _edge("price", "mid", mean=0.5),
            _edge("mid", "revenue", mean=0.4),  # 0.20  -> total 0.20
        ],
    )
    options = [InterventionOption(id="opt_a", label="A", interventions={"price": 0.3})]
    request = _request(graph, options, "revenue")

    dec = _analyzer()._compute_path_decomposition(request, "opt_a", graph)

    zero_path = next(p for p in dec.paths if p.path == ["price", "revenue"])
    assert zero_path.path_effect == pytest.approx(0.0)
    assert zero_path.status == "computed"  # total is non-zero
    assert zero_path.signed_contribution == pytest.approx(0.0)


# =============================================================================
# 13. Top-3 truncation; total computed over ALL paths
# =============================================================================


def test_top_3_truncation_and_total_over_all_paths():
    # 5 parallel single-mediator paths with distinct effects 0.1..0.5.
    nodes = [_node("price"), _node("revenue", kind="outcome")]
    edges = []
    for i, m in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], start=1):
        nodes.append(_node(f"m{i}"))
        edges.append(_edge("price", f"m{i}", mean=1.0))  # coeff 1.0
        edges.append(_edge(f"m{i}", "revenue", mean=m))  # path effect == m
    graph = GraphV2(nodes=nodes, edges=edges)
    options = [InterventionOption(id="opt_a", label="A", interventions={"price": 0.3})]
    request = _request(graph, options, "revenue")

    dec = _analyzer()._compute_path_decomposition(request, "opt_a", graph)

    assert len(dec.paths) == 3
    # Largest absolute effects first, deterministic.
    assert [round(p.path_effect, 6) for p in dec.paths] == [0.5, 0.4, 0.3]
    # total_effect is the sum over ALL 5 paths, not just the top 3.
    assert dec.paths[0].total_effect == pytest.approx(1.5)


# =============================================================================
# 14. Mechanism wording contract
# =============================================================================


def test_mechanism_wording_contract():
    # Computed case.
    g_computed = GraphV2(
        nodes=[_node("price"), _node("revenue", kind="outcome")],
        edges=[_edge("price", "revenue", mean=0.5)],
    )
    # Indeterminate case (cancelling paths).
    g_indet = GraphV2(
        nodes=[_node("price"), _node("mid"), _node("revenue", kind="outcome")],
        edges=[
            _edge("price", "revenue", mean=0.5),
            _edge("price", "mid", mean=1.0),
            _edge("mid", "revenue", mean=-0.5),
        ],
    )
    options = [InterventionOption(id="opt_a", label="A", interventions={"price": 0.3})]
    analyzer = _analyzer()

    for graph in (g_computed, g_indet):
        dec = analyzer._compute_path_decomposition(
            _request(graph, options, "revenue"), "opt_a", graph
        )
        assert dec.paths
        for p in dec.paths:
            text = p.mechanism.lower()
            assert "modelled pathway contribution" in text
            assert "%" not in p.mechanism
            assert "cause" not in text  # also catches "caused"


# =============================================================================
# 15. Option as metadata / retained entry node
# =============================================================================


def test_option_metadata_and_retained_entry_nodes():
    graph = GraphV2(
        nodes=[
            _node("price"),
            _node("strategy", kind="decision"),
            _node("revenue", kind="outcome"),
        ],
        edges=[
            _edge("price", "revenue", mean=0.5),
            _edge("strategy", "revenue", mean=0.9),
        ],
    )
    options = [
        InterventionOption(id="opt_a", label="A", interventions={"price": 0.3}),
        InterventionOption(id="opt_b", label="B", interventions={"price": 0.7}),
    ]
    request = _request(graph, options, "revenue", include_path_decomposition=True)

    response = _analyzer().analyze(request)
    dec = response.path_decomposition

    assert dec is not None
    # Option referenced as metadata, matching the recomputed-elsewhere winner.
    assert dec.recommended_option_id == response.recommended_option_id
    # Entry nodes are the retained intervention targets and equal each path's first node.
    assert dec.entry_nodes == ["price"]
    assert all(p.path[0] in dec.entry_nodes for p in dec.paths)
    # No decision/option node appears as a path node.
    assert all("strategy" not in p.path for p in dec.paths)


# =============================================================================
# 16. Structural, NOT intervention-magnitude-weighted (explicit non-scaling contract)
# =============================================================================


def test_decomposition_is_structural_not_intervention_magnitude_weighted():
    """Contract: the decomposition is a STRUCTURAL property of the graph edges.

    Two options with the SAME retained intervention target but DIFFERENT intervention
    values must produce the SAME path effects / signed contributions.  path_effect is
    the product of edge coefficients only — it is never multiplied by the do(X=x) value.
    This guards future readers against misreading the metric as an option-level
    effect-size estimate.
    """
    graph = GraphV2(
        nodes=[_node("price"), _node("mid"), _node("revenue", kind="outcome")],
        edges=[
            _edge("price", "mid", mean=0.5),
            _edge("mid", "revenue", mean=0.6),
            _edge("price", "revenue", mean=0.1),
        ],
    )
    options = [
        InterventionOption(id="opt_low", label="Low", interventions={"price": 0.2}),
        InterventionOption(id="opt_high", label="High", interventions={"price": 0.9}),
    ]
    request = _request(graph, options, "revenue")
    analyzer = _analyzer()

    dec_low = analyzer._compute_path_decomposition(request, "opt_low", graph)
    dec_high = analyzer._compute_path_decomposition(request, "opt_high", graph)

    # Only the metadata differs; the structural decomposition is identical.
    assert dec_low.recommended_option_id != dec_high.recommended_option_id
    assert dec_low.entry_nodes == dec_high.entry_nodes
    assert dec_low.paths == dec_high.paths
