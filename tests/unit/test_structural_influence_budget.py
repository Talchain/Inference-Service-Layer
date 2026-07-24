"""UC-2 (D-23.18): the structural-influence path enumerator must be bounded.

Pre-fix, ``_compute_structural_influence`` enumerated ALL simple paths
recursively with no budget — the uncapped twin of the (capped)
``_compute_path_decomposition`` walker. A dense layered DAG legal under the
schema caps admits exponential path counts at near-zero admission charge, so an
authed caller could burn unbounded CPU (guarantee-theatre in the F2 family;
found by the PHASE_COST_ATTRIBUTION guard while it was being built).

The fix: a per-factor walk-CALL budget (bounds dead-branch exploration too, not
just completed paths), deterministic truncation, disclosed per factor via the
STRUCTURAL_INFLUENCE_TRUNCATED critique.

MUTATION ANCHOR: reverting the budget (removing the calls_left check) makes
test_budget_truncates_and_reports flip RED (no truncation reported) and makes
test_adversarial_dense_dag_completes_fast hang far beyond its bound pre-fix.
"""

import time

from src.models.robustness_v2 import GraphV2, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    MAX_INFLUENCE_WALK_CALLS_PER_FACTOR,
    RobustnessAnalyzerV2,
)


def _layered_graph(n_layers: int, width: int) -> dict:
    """Complete layered DAG: width^(n_layers-1) simple paths source->goal.

    Layer 0 holds the factor sources; the single goal node sits after the last
    layer. Every node in layer i connects to every node in layer i+1.
    """
    nodes = []
    edges = []
    for layer in range(n_layers):
        for w in range(width):
            nodes.append({"id": f"l{layer}_{w}", "kind": "factor", "label": f"l{layer}_{w}"})
    nodes.append({"id": "goal", "kind": "outcome", "label": "Goal"})
    for layer in range(n_layers - 1):
        for a in range(width):
            for b in range(width):
                edges.append(
                    {
                        "from": f"l{layer}_{a}",
                        "to": f"l{layer + 1}_{b}",
                        "exists_probability": 0.9,
                        "strength": {"mean": 0.5, "std": 0.1},
                    }
                )
    for w in range(width):
        edges.append(
            {
                "from": f"l{n_layers - 1}_{w}",
                "to": "goal",
                "exists_probability": 0.9,
                "strength": {"mean": 0.5, "std": 0.1},
            }
        )
    return {"nodes": nodes, "edges": edges}


def _graph_v2(raw: dict) -> GraphV2:
    """Validated GraphV2 via the request model (same parse path as production)."""
    req = RobustnessRequestV2(
        graph=raw,
        options=[{"id": "o1", "label": "O1", "interventions": {"l0_0": 0.5}}],
        goal_node_id="goal",
        n_samples=100,
        seed=7,
        analysis_types=["comparison", "robustness"],
    )
    return req.graph


class TestInfluenceWalkBudget:
    def test_small_graph_untruncated_and_normalized(self):
        """Regression guard: small graphs are exact — no truncation, same
        normalized shape as pre-fix (top factor == 1.0)."""
        graph = _graph_v2(_layered_graph(3, 2))  # 2^2 * 2 = 8 paths max
        an = RobustnessAnalyzerV2()
        influences, truncated = an._compute_structural_influence(
            graph, ["l0_0", "l0_1"], "goal"
        )
        assert truncated == []
        assert set(influences) == {"l0_0", "l0_1"}
        assert max(influences.values()) == 1.0

    def test_budget_truncates_and_reports(self):
        """With a tiny budget the enumeration truncates, reports the factor,
        and still returns a finite lower-bound influence.

        MUTATION ANCHOR: without the calls_left check this reports NO
        truncation (and enumerates everything)."""
        graph = _graph_v2(_layered_graph(4, 3))  # 3^3 * 3 = 81+ paths
        an = RobustnessAnalyzerV2()
        influences, truncated = an._compute_structural_influence(
            graph, ["l0_0"], "goal", max_walk_calls_per_factor=10
        )
        assert truncated == ["l0_0"]
        assert influences["l0_0"] >= 0.0  # finite lower bound, never NaN/absent

    def test_truncation_is_deterministic(self):
        """Same graph + same budget -> byte-identical influences and the same
        truncation verdict (count-based budget, insertion-order traversal)."""
        graph = _graph_v2(_layered_graph(4, 3))
        an = RobustnessAnalyzerV2()
        a = an._compute_structural_influence(graph, ["l0_0", "l0_1"], "goal",
                                             max_walk_calls_per_factor=25)
        b = an._compute_structural_influence(graph, ["l0_0", "l0_1"], "goal",
                                             max_walk_calls_per_factor=25)
        assert a == b

    def test_adversarial_dense_dag_completes_fast(self):
        """The UC-2 exposure shape: a schema-legal dense layered DAG whose full
        enumeration is combinatorially explosive must complete within the
        budgeted work envelope.

        10 layers x 4 wide = 49 nodes, 4^9*4 ≈ 1M simple paths per source
        factor — pre-fix this enumerates them all (plus set copies), post-fix
        work is capped at MAX_INFLUENCE_WALK_CALLS_PER_FACTOR calls per factor.
        The generous wall bound is a hang-guard, not a benchmark."""
        raw = _layered_graph(10, 4)
        assert len(raw["nodes"]) <= 50 and len(raw["edges"]) <= 200  # schema-legal
        graph = _graph_v2(raw)
        an = RobustnessAnalyzerV2()
        factor_ids = [f"l0_{w}" for w in range(4)]
        t0 = time.perf_counter()
        influences, truncated = an._compute_structural_influence(graph, factor_ids, "goal")
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"bounded walk took {elapsed:.1f}s — budget not effective"
        assert set(truncated) == set(factor_ids)  # this shape MUST truncate
        assert all(v >= 0.0 for v in influences.values())

    def test_full_budget_constant_sane(self):
        """Silent-revert guard on the budget constant."""
        assert MAX_INFLUENCE_WALK_CALLS_PER_FACTOR == 200_000


class TestTruncationDisclosureOnWire:
    def test_truncated_enumeration_emits_critique_via_factor_sensitivity(self):
        """End-to-end through _compute_factor_sensitivity: a truncating graph
        surfaces STRUCTURAL_INFLUENCE_TRUNCATED into the critiques list.

        Uses a monkeypatched tiny budget so the fixture graph stays small."""
        import src.services.robustness_analyzer_v2 as mod

        raw = _layered_graph(4, 3)
        body = {
            "graph": raw,
            "options": [
                {"id": "o1", "label": "O1", "interventions": {"l0_0": 0.5}},
                {"id": "o2", "label": "O2", "interventions": {"l0_0": 1.0}},
            ],
            "goal_node_id": "goal",
            "n_samples": 200,
            "seed": 7,
            # factor_sensitivity (the influence enumerator's host) gates on BOTH
            # uncertainties AND "sensitivity" in analysis_types (raa_v2 ~L1804).
            "analysis_types": ["comparison", "robustness", "sensitivity"],
            "include_voi": False,
            "parameter_uncertainties": [
                {"node_id": "l0_1", "distribution": "normal", "std": 1.0}
            ],
        }
        req = RobustnessRequestV2(**body)
        original = mod.MAX_INFLUENCE_WALK_CALLS_PER_FACTOR
        try:
            mod.MAX_INFLUENCE_WALK_CALLS_PER_FACTOR = 10
            resp = RobustnessAnalyzerV2().analyze(req)
        finally:
            mod.MAX_INFLUENCE_WALK_CALLS_PER_FACTOR = original
        codes = [c.code for c in (resp.critiques or [])]
        assert "STRUCTURAL_INFLUENCE_TRUNCATED" in codes, (
            f"truncation not disclosed on the wire; critiques = {codes}"
        )
