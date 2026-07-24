"""UC-2 (D-23.18) + Codex re-confirm N1/N2 (D-23.19): the structural-influence
path enumerator must be bounded, PRICED, and exact-or-null.

History:
- Pre-#112, ``_compute_structural_influence`` enumerated ALL simple paths with
  no budget (uncapped twin of the capped ``_compute_path_decomposition``).
- #112 added a PER-FACTOR 200k call budget and published normalized scores of
  truncated cohorts as "lower bounds". Codex's re-confirmation broke both:
  * N1 (P0): raw path sums are lower bounds, NORMALIZED scores are not — the
    data-dependent max denominator can shrink faster than a numerator,
    inflating another factor's score (their repro: exact 0.1 → bounded 1.0)
    and inverting ranks.
  * N2 (P1): the per-factor reset multiplied worst-case work by U
    (13 x 200k measured ~2s) and none of it was priced.
- This revision: ONE request-wide pool (MAX_INFLUENCE_WALK_CALLS_TOTAL),
  charged 1:1 in compute_weighted_cost (`structural_influence` term), and
  EXACT-OR-NULL publication: any truncation withholds every score and rank in
  the cohort, disclosed via STRUCTURAL_INFLUENCE_TRUNCATED.

MUTATION ANCHORS: removing the calls_left check un-truncates the tiny-budget
tests; restoring per-factor reset flips test_pool_is_request_wide; publishing
scores despite truncation flips the N1 tests.
"""

import time

from src.models.robustness_v2 import GraphV2, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    MAX_INFLUENCE_WALK_CALLS_TOTAL,
    RobustnessAnalyzerV2,
    compute_weighted_cost,
)


def _layered_graph(n_layers: int, width: int) -> dict:
    """Complete layered DAG: width^(n_layers-1) simple paths source->goal."""
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


def _graph_v2(raw: dict, lever: str = "l0_0") -> GraphV2:
    """Validated GraphV2 via the request model (same parse path as production)."""
    req = RobustnessRequestV2(
        graph=raw,
        options=[{"id": "o1", "label": "O1", "interventions": {lever: 0.5}}],
        goal_node_id="goal",
        n_samples=100,
        seed=7,
        analysis_types=["comparison", "robustness"],
    )
    return req.graph


def _codex_n1_graph() -> dict:
    """Codex's N1 counterexample shape: factor fa has a strong DIRECT path to
    goal but its adjacency lists dead branches FIRST (insertion order), so a
    tiny budget exhausts on dead exploration before reaching the productive
    edge. Factor fb has one cheap weak path. Exact scores: fa=1.0 (strong),
    fb small. A truncated run finds nothing for fa, something for fb —
    normalizing would invert the ranking (fb=1.0, fa=0.0)."""
    nodes = [
        {"id": "fa", "kind": "factor", "label": "fa"},
        {"id": "fb", "kind": "factor", "label": "fb"},
        # dead-branch chain (no path to goal) wired FIRST from fa
        {"id": "d1", "kind": "factor", "label": "d1"},
        {"id": "d2", "kind": "factor", "label": "d2"},
        {"id": "d3", "kind": "factor", "label": "d3"},
        {"id": "d4", "kind": "factor", "label": "d4"},
        {"id": "goal", "kind": "outcome", "label": "Goal"},
    ]
    def e(a, b, m):
        return {"from": a, "to": b, "exists_probability": 1.0,
                "strength": {"mean": m, "std": 0.05}}
    edges = [
        # fa's dead branches FIRST (insertion order = traversal order)
        e("fa", "d1", 0.9), e("d1", "d2", 0.9), e("d2", "d3", 0.9), e("d3", "d4", 0.9),
        # fa's productive strong edge LAST
        e("fa", "goal", 0.9),
        # fb: one cheap weak path
        e("fb", "goal", 0.09),
    ]
    return {"nodes": nodes, "edges": edges}


class TestExactOrNullSemantics:
    """N1: normalized scores of a truncated cohort must never be published."""

    def test_codex_n1_counterexample_is_not_published_as_bound(self):
        """The exact repro class from the re-confirmation review.

        Exact enumeration: fa=1.0 (0.9 direct), fb=0.1 (0.09/0.9). Under a
        tiny budget fa exhausts on dead branches (raw 0) while fb completes —
        the OLD code published fb=1.0/fa=0.0 as 'lower bounds' (false by 10x).
        NEW contract: the helper still reports raw-normalized values + the
        truncation list; the CALLER must withhold — asserted end-to-end in
        TestWireSemantics below. Here: exact run matches Codex's exact scores.
        """
        graph = _graph_v2(_codex_n1_graph(), lever="fa")
        an = RobustnessAnalyzerV2()
        # Exact (default pool is plenty for 6 edges)
        exact, truncated = an._compute_structural_influence(graph, ["fa", "fb"], "goal")
        assert truncated == []
        assert abs(exact["fa"] - 1.0) < 1e-9
        assert abs(exact["fb"] - 0.1) < 1e-9
        # Truncated: fb (cheap, 2 calls) completes first, then fa exhausts the
        # remaining pool inside its dead branches before reaching its strong
        # direct edge — raw(fa)=0 while raw(fb)>0.
        bounded, truncated_small = an._compute_structural_influence(
            graph, ["fb", "fa"], "goal", max_walk_calls_total=7
        )
        assert truncated_small == ["fa"]
        # The inversion Codex demonstrated is REAL in the raw output (exact
        # fa=1.0/fb=0.1 becomes bounded fa=0.0/fb=1.0) — which is exactly why
        # the caller must never publish a truncated cohort's normalized scores:
        assert bounded["fb"] == 1.0
        assert bounded["fa"] == 0.0

    def test_small_graph_untruncated_and_normalized(self):
        """Exact graphs keep exact normalized scores (top factor == 1.0)."""
        graph = _graph_v2(_layered_graph(3, 2))
        an = RobustnessAnalyzerV2()
        influences, truncated = an._compute_structural_influence(
            graph, ["l0_0", "l0_1"], "goal"
        )
        assert truncated == []
        assert max(influences.values()) == 1.0


class TestRequestWidePool:
    """N2: one shared pool, not a per-factor reset."""

    def test_pool_is_request_wide(self):
        """First factor drains the pool; the second — which would complete
        comfortably under a per-factor reset — must also be truncated.

        MUTATION ANCHOR: restoring `calls_left = budget` inside the factor
        loop makes fb complete and this test flip RED."""
        graph = _graph_v2(_codex_n1_graph(), lever="fa")
        an = RobustnessAnalyzerV2()
        # Budget 5: fa's dead-branch chain (fa,d1..d4 = 5 calls) drains the pool
        # before its direct goal edge -> fa truncates; fb then starts exhausted
        # and truncates too, despite needing only 2 calls of its own.
        _, truncated = an._compute_structural_influence(
            graph, ["fa", "fb"], "goal", max_walk_calls_total=5
        )
        assert truncated == ["fa", "fb"]

    def test_adversarial_dense_dag_completes_fast(self):
        """Schema-legal dense DAG, ALL 40 non-goal nodes as factors — the N2
        measurement shape. Work is bounded by ONE pool regardless of factor
        count; wall must stay near the single-pool envelope (~0.3-0.5s local),
        generous bound as a hang-guard."""
        raw = _layered_graph(10, 4)
        graph = _graph_v2(raw)
        factor_ids = [n["id"] for n in raw["nodes"] if n["id"] != "goal"]
        an = RobustnessAnalyzerV2()
        t0 = time.perf_counter()
        influences, truncated = an._compute_structural_influence(graph, factor_ids, "goal")
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"pooled walk took {elapsed:.1f}s — pool not shared?"
        assert truncated, "this shape must truncate"
        assert set(influences) == set(factor_ids)

    def test_pool_constant_sane(self):
        assert MAX_INFLUENCE_WALK_CALLS_TOTAL == 400_000


class TestAdmissionPricing:
    """N2: the pool ceiling is charged whenever the phase can run."""

    def _body(self, *, sensitivity: bool, uncertainties: bool) -> dict:
        raw = _layered_graph(3, 2)
        types = ["comparison", "robustness"] + (["sensitivity"] if sensitivity else [])
        body = {
            "graph": raw,
            "options": [{"id": "o1", "label": "O1", "interventions": {"l0_0": 0.5}}],
            "goal_node_id": "goal",
            "n_samples": 1000,
            "seed": 7,
            "analysis_types": types,
        }
        if uncertainties:
            body["parameter_uncertainties"] = [
                {"node_id": "l0_1", "distribution": "normal", "std": 1.0}
            ]
        return body

    def test_term_present_when_phase_can_run(self):
        wc = compute_weighted_cost(
            RobustnessRequestV2(**self._body(sensitivity=True, uncertainties=True))
        )
        assert wc.terms["structural_influence"] == MAX_INFLUENCE_WALK_CALLS_TOTAL

    def test_term_absent_when_gated_off(self):
        no_sens = compute_weighted_cost(
            RobustnessRequestV2(**self._body(sensitivity=False, uncertainties=True))
        )
        no_unc = compute_weighted_cost(
            RobustnessRequestV2(**self._body(sensitivity=True, uncertainties=False))
        )
        assert "structural_influence" not in no_sens.terms
        assert "structural_influence" not in no_unc.terms


class TestWireSemantics:
    """End-to-end through analyze(): truncation ⇒ null scores/ranks + critique;
    no false DISCONNECTED labels from withheld influence."""

    def _analyze_truncated(self):
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
            # host phase gates on uncertainties AND "sensitivity" (raa_v2 ~L1810)
            "analysis_types": ["comparison", "robustness", "sensitivity"],
            "parameter_uncertainties": [
                {"node_id": "l0_1", "distribution": "normal", "std": 1.0}
            ],
        }
        req = RobustnessRequestV2(**body)
        original = mod.MAX_INFLUENCE_WALK_CALLS_TOTAL
        try:
            mod.MAX_INFLUENCE_WALK_CALLS_TOTAL = 10
            resp = RobustnessAnalyzerV2().analyze(req)
        finally:
            mod.MAX_INFLUENCE_WALK_CALLS_TOTAL = original
        return resp

    def test_truncation_nulls_scores_and_ranks_and_discloses(self):
        """N1 on the wire: withheld scores/ranks + the critique.

        MUTATION ANCHOR: publishing normalized scores despite truncation
        (dropping the influence_exact gate) flips the None asserts RED."""
        resp = self._analyze_truncated()
        codes = [c.code for c in (resp.critiques or [])]
        assert "STRUCTURAL_INFLUENCE_TRUNCATED" in codes, f"critiques = {codes}"
        rows = resp.factor_sensitivity or []
        assert rows, "factor_sensitivity must still be emitted"
        for row in rows:
            assert row.influence_score is None, (
                f"{row.node_id}: score {row.influence_score} published despite truncation"
            )
            assert row.influence_rank is None, (
                f"{row.node_id}: rank {row.influence_rank} published despite truncation"
            )

    def test_truncation_does_not_fabricate_disconnected(self):
        """The DISCONNECTED zero-reason inference reads the influence score; a
        truncated (withheld) score must not flip a factor to DISCONNECTED —
        a productive path beyond the budget would read as 'no path'."""
        from src.models.response_v2 import ZeroSensitivityReason

        resp = self._analyze_truncated()
        for row in resp.factor_sensitivity or []:
            assert row.zero_reason != ZeroSensitivityReason.DISCONNECTED, (
                f"{row.node_id} labelled DISCONNECTED from a withheld influence score"
            )

    def test_exact_run_still_publishes_scores(self):
        """Regression guard: untruncated analyses keep scores/ranks + no critique."""
        raw = _layered_graph(3, 2)
        body = {
            "graph": raw,
            "options": [
                {"id": "o1", "label": "O1", "interventions": {"l0_0": 0.5}},
                {"id": "o2", "label": "O2", "interventions": {"l0_0": 1.0}},
            ],
            "goal_node_id": "goal",
            "n_samples": 200,
            "seed": 7,
            "analysis_types": ["comparison", "robustness", "sensitivity"],
            "parameter_uncertainties": [
                {"node_id": "l0_1", "distribution": "normal", "std": 1.0}
            ],
        }
        resp = RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**body))
        codes = [c.code for c in (resp.critiques or [])]
        assert "STRUCTURAL_INFLUENCE_TRUNCATED" not in codes
        rows = resp.factor_sensitivity or []
        assert rows
        for row in rows:
            assert row.influence_score is not None
            assert row.influence_rank is not None
