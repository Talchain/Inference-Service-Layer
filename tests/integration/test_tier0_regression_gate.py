"""
Tier 0 scientific regression gate.

Locks in the fixes for the four verified defects from the ISL capability
audit (Track S, staging @ 4f5bf77):

1. Goal-threshold vs goal-constraint probability consistency: a constraint on
   the goal node must be evaluated on the exact same (possibly auto-noised)
   samples as probability_of_goal.
2. Seed reporting: the seed in the V2 envelope must be the seed the analyzer's
   RNG streams actually used, including when organisational nodes are
   filtered out of the graph before sampling.
3. Cyclic graphs fail closed on BOTH the default/V1 and the V2-enhanced
   response paths (422, never plausible-looking results).
4. Duplicate directed edges are rejected at validation instead of silently
   double-counting the same causal effect.

Plus scientific-content assertions the audit found missing: same-seed
full-response reproducibility, and analytic single-edge fixtures asserting
magnitude and sign, not just HTTP 200.

All tests are deterministic, in-process, and network-free.
"""

import copy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.models.robustness_v2 import GraphV2, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    RobustnessAnalyzerV2,
    compute_effective_seed,
    filter_inference_graph,
)
from src.utils.rng import compute_seed_from_graph

V2_URL = "/api/v1/robustness/analyze/v2"


@pytest.fixture
def client():
    """Synchronous in-process test client."""
    return TestClient(app)


def outcome_goal_request(**overrides):
    """Base request: 3-node chain with an outcome-kind goal (auto-noise applies)."""
    request = {
        "graph": {
            "nodes": [
                {
                    "id": "price",
                    "kind": "factor",
                    "label": "Price",
                    "observed_state": {"value": 0.5},
                },
                {"id": "demand", "kind": "chance", "label": "Demand"},
                {"id": "revenue", "kind": "outcome", "label": "Revenue"},
            ],
            "edges": [
                {
                    "from": "price",
                    "to": "demand",
                    "exists_probability": 0.9,
                    "strength": {"mean": -0.6, "std": 0.1},
                },
                {
                    "from": "demand",
                    "to": "revenue",
                    "exists_probability": 0.95,
                    "strength": {"mean": 0.8, "std": 0.1},
                },
                {
                    "from": "price",
                    "to": "revenue",
                    "exists_probability": 0.8,
                    "strength": {"mean": 0.5, "std": 0.15},
                },
            ],
        },
        "options": [
            {"id": "low", "label": "Low price", "interventions": {"price": 0.3}},
            {"id": "high", "label": "High price", "interventions": {"price": 0.7}},
        ],
        "goal_node_id": "revenue",
        "n_samples": 500,
        "seed": 42,
    }
    request.update(overrides)
    return request


def single_edge_request(strength_mean: float):
    """x -> y with near-deterministic strength; analytic expectation is
    intervention_value * strength_mean at the goal."""
    return {
        "graph": {
            "nodes": [
                {"id": "x", "kind": "factor", "label": "X"},
                {"id": "y", "kind": "outcome", "label": "Y"},
            ],
            "edges": [
                {
                    "from": "x",
                    "to": "y",
                    "exists_probability": 1.0,
                    "strength": {"mean": strength_mean, "std": 0.01},
                }
            ],
        },
        "options": [
            {"id": "lo", "label": "lo", "interventions": {"x": 0.2}},
            {"id": "hi", "label": "hi", "interventions": {"x": 0.8}},
        ],
        "goal_node_id": "y",
        "n_samples": 500,
        "seed": 7,
    }


def with_organisational_node(request):
    """Add a decision node (+ incident edge) that the analyzer filters out."""
    request = copy.deepcopy(request)
    request["graph"]["nodes"].append({"id": "dec", "kind": "decision", "label": "Decision"})
    request["graph"]["edges"].append(
        {"from": "dec", "to": "demand", "strength": {"mean": 0.2, "std": 0.05}}
    )
    return request


def with_cycle(request):
    """Add an edge that closes a cycle (revenue -> price)."""
    request = copy.deepcopy(request)
    request["graph"]["edges"].append(
        {"from": "revenue", "to": "price", "strength": {"mean": 0.5, "std": 0.1}}
    )
    return request


# =============================================================================
# 1. Goal-threshold vs goal-constraint consistency
# =============================================================================


class TestGoalConstraintConsistency:
    THRESHOLD = 0.1

    def _request_with_goal_constraint(self, **overrides):
        return outcome_goal_request(
            goal_threshold=self.THRESHOLD,
            goal_constraints=[{"node_id": "revenue", "operator": ">=", "value": self.THRESHOLD}],
            **overrides,
        )

    def test_identical_threshold_and_constraint_agree_with_auto_noise(
        self, client, auto_noise_enabled
    ):
        """Outcome-kind goal (auto-noise active): P(goal >= t) must equal the
        prob_satisfied of a constraint '>= t' on the same node, exactly.

        Arch step 1 (2026-07-26): auto-scaled noise is DEFAULT OFF
        (ENABLE_AUTO_SCALED_NOISE). This test's subject IS the noise-on path, so
        it opts in via the `auto_noise_enabled` fixture; the pinned expectations
        below are unchanged.
        """
        response = client.post(
            f"{V2_URL}?response_version=2", json=self._request_with_goal_constraint()
        )
        assert response.status_code == 200
        body = response.json()
        assert body["auto_noise_applied"] is True  # test premise: noise ran

        for option in body["options"]:
            constraint = option["constraint_analysis"]["constraints"][0]
            assert constraint["node_id"] == "revenue"
            assert option["probability_of_goal"] == pytest.approx(
                constraint["prob_satisfied"], abs=0.0
            )
            # Single constraint: joint probability equals it exactly
            assert option["constraint_analysis"]["joint_probability"] == pytest.approx(
                constraint["prob_satisfied"], abs=0.0
            )

    def test_identical_threshold_and_constraint_agree_without_auto_noise(self, client):
        """Chance-kind goal (no auto-noise): probabilities must also agree."""
        request = self._request_with_goal_constraint()
        request["graph"]["nodes"][2]["kind"] = "chance"
        response = client.post(f"{V2_URL}?response_version=2", json=request)
        assert response.status_code == 200
        body = response.json()
        assert body.get("auto_noise_applied") in (False, None)

        for option in body["options"]:
            constraint = option["constraint_analysis"]["constraints"][0]
            assert option["probability_of_goal"] == pytest.approx(
                constraint["prob_satisfied"], abs=0.0
            )

    def test_multi_constraint_behaviour_preserved_and_mix_disclosed(
        self, client, auto_noise_enabled
    ):
        """A second (non-goal) constraint keeps per/joint/conditional outputs,
        and the noised-goal/un-noised-sibling mix is disclosed, not silent.

        Arch step 1 (2026-07-26): auto-scaled noise is DEFAULT OFF
        (ENABLE_AUTO_SCALED_NOISE). This test's subject IS the noise-on path, so
        it opts in via the `auto_noise_enabled` fixture; the pinned expectations
        below are unchanged.
        """
        request = self._request_with_goal_constraint()
        request["goal_constraints"].append({"node_id": "demand", "operator": "<=", "value": 5.0})
        response = client.post(f"{V2_URL}?response_version=2", json=request)
        assert response.status_code == 200
        body = response.json()
        assert body["auto_noise_applied"] is True

        for option in body["options"]:
            analysis = option["constraint_analysis"]
            assert len(analysis["constraints"]) == 2
            by_node = {c["node_id"]: c for c in analysis["constraints"]}
            # Goal-node constraint still agrees exactly with probability_of_goal
            assert option["probability_of_goal"] == pytest.approx(
                by_node["revenue"]["prob_satisfied"], abs=0.0
            )
            # Joint cannot exceed either individual probability
            joint = analysis["joint_probability"]
            assert joint <= min(c["prob_satisfied"] for c in analysis["constraints"]) + 1e-12
            # Pairwise conditionals still present for 2+ constraints
            assert analysis["conditional_probabilities"] is not None

        warning_codes = [w["code"] for w in body["inference_warnings"]]
        assert "CONSTRAINT_SAMPLES_UNNOISED" in warning_codes


# =============================================================================
# 2. Seed reporting
# =============================================================================


class TestSeedReporting:
    def test_server_computed_seed_matches_analyzer_with_filtered_nodes(self, client):
        """No client seed + organisational nodes: the V2 envelope seed and the
        V1 metadata seed must be the same seed, and it must be the post-filter
        graph hash (the seed the RNGs actually consumed)."""
        request = with_organisational_node(outcome_goal_request())
        del request["seed"]

        v2_body = client.post(f"{V2_URL}?response_version=2", json=request).json()
        v1_body = client.post(f"{V2_URL}?response_version=1", json=request).json()

        assert v2_body["seed_source"] == "server_computed"
        # V1 serialises ResponseMetadataV2 under its "_metadata" alias
        assert int(v2_body["seed_used"]) == v1_body["_metadata"]["seed_used"]

        parsed = RobustnessRequestV2(**request)
        expected_seed = compute_seed_from_graph(filter_inference_graph(parsed.graph))
        assert int(v2_body["seed_used"]) == expected_seed
        # And it must differ from the raw-graph hash (the pre-fix bug value),
        # proving the filter actually changed the derivation input here.
        assert expected_seed != compute_seed_from_graph(parsed.graph)

    def test_explicit_client_seed_wins(self, client):
        """An explicit seed is reported verbatim on both response versions,
        even when organisational nodes are filtered."""
        request = with_organisational_node(outcome_goal_request(seed=12345))

        v2_body = client.post(f"{V2_URL}?response_version=2", json=request).json()
        v1_body = client.post(f"{V2_URL}?response_version=1", json=request).json()

        assert v2_body["seed_used"] == "12345"
        assert v2_body["seed_source"] == "client_provided"
        assert v1_body["_metadata"]["seed_used"] == 12345

    def test_compute_effective_seed_helper_is_shared_derivation(self):
        """The helper is idempotent w.r.t. filtering and honours explicit
        seeds (including seed=0)."""
        parsed = RobustnessRequestV2(**with_organisational_node(outcome_goal_request(seed=0)))
        assert compute_effective_seed(parsed) == (0, "client_provided")

        request = with_organisational_node(outcome_goal_request())
        del request["seed"]
        parsed = RobustnessRequestV2(**request)
        seed, source = compute_effective_seed(parsed)
        assert source == "server_computed"
        filtered = parsed.model_copy(update={"graph": filter_inference_graph(parsed.graph)})
        assert compute_effective_seed(filtered)[0] == seed

    def test_seed_derivation_does_not_duplicate_filter_warning(self, caplog):
        """The helper's auxiliary filter is silent; only the analyzer's
        authoritative filter emits the filtered-nodes warning (once)."""
        import logging

        request = with_organisational_node(outcome_goal_request())
        del request["seed"]
        parsed = RobustnessRequestV2(**request)

        with caplog.at_level(logging.WARNING):
            compute_effective_seed(parsed)
        assert not any(
            "filtered_non_inference" in r.message for r in caplog.records
        ), "Seed derivation must not emit the filter warning"

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            RobustnessAnalyzerV2().analyze(parsed)
        filter_warnings = [r for r in caplog.records if "filtered_non_inference" in r.message]
        assert len(filter_warnings) == 1, "Analyzer must emit the filter warning exactly once"

    def test_all_organisational_graph_does_not_crash_seed_derivation(self, client):
        """A schema-valid graph whose every node is organisational empties the
        inference filter. Seed derivation must fall back to the raw-graph hash
        instead of raising, and the route must return a structured ISLV2
        response (envelope shape, request-ID header), never an unhandled 500
        in the generic error schema."""
        request = {
            "graph": {
                "nodes": [
                    {"id": "g", "kind": "decision", "label": "G"},
                    {"id": "d2", "kind": "option", "label": "D2"},
                ],
                "edges": [{"from": "d2", "to": "g", "strength": {"mean": 0.5, "std": 0.1}}],
            },
            "options": [{"id": "a", "label": "A", "interventions": {"d2": 1.0}}],
            "goal_node_id": "g",
            "n_samples": 100,
        }

        # Helper level: no raise, raw-graph fallback hash
        parsed = RobustnessRequestV2(**request)
        seed, source = compute_effective_seed(parsed)
        assert source == "server_computed"
        assert seed == compute_seed_from_graph(parsed.graph)

        # Route level: structured envelope error, not a bare crash
        response = client.post(f"{V2_URL}?response_version=2", json=request)
        assert response.status_code in (422, 500)
        body = response.json()
        assert "analysis_status" in body, f"Expected ISLV2 envelope shape, got: {sorted(body)}"
        assert response.headers.get("X-Request-Id")


# =============================================================================
# 3. Cyclic graphs fail closed on all live paths
# =============================================================================


class TestCycleFailClosed:
    def test_cycle_rejected_on_v2_enhanced_path(self, client):
        response = client.post(
            f"{V2_URL}?response_version=2", json=with_cycle(outcome_goal_request())
        )
        assert response.status_code == 422
        body = response.json()
        assert body["analysis_status"] == "blocked"
        assert any(c["code"] == "GRAPH_CYCLE_DETECTED" for c in body["critiques"])

    def test_cycle_rejected_on_default_v1_path(self, client):
        """The default path (no response_version param) previously analysed
        cyclic graphs silently; it must now 422."""
        response = client.post(V2_URL, json=with_cycle(outcome_goal_request()))
        assert response.status_code == 422
        assert "GRAPH_CYCLE_DETECTED" in response.text

    def test_cycle_rejected_at_analyzer_level(self):
        """Direct analyzer calls (any future caller) also fail closed."""
        analyzer = RobustnessAnalyzerV2()
        request = RobustnessRequestV2(**with_cycle(outcome_goal_request()))
        with pytest.raises(ValueError, match="GRAPH_CYCLE_DETECTED"):
            analyzer.analyze(request)

    def test_valid_dag_still_passes_both_versions(self, client):
        v1 = client.post(V2_URL, json=outcome_goal_request())
        assert v1.status_code == 200
        assert len(v1.json()["results"]) == 2

        v2 = client.post(f"{V2_URL}?response_version=2", json=outcome_goal_request())
        assert v2.status_code == 200
        assert v2.json()["analysis_status"] == "computed"


# =============================================================================
# 4. Duplicate edges rejected
# =============================================================================


class TestDuplicateEdges:
    def _nodes(self):
        return [
            {"id": "x", "kind": "factor", "label": "X"},
            {"id": "y", "kind": "outcome", "label": "Y"},
        ]

    def test_duplicate_directed_edge_rejected_at_model_level(self):
        with pytest.raises(ValidationError, match="Duplicate edges found"):
            GraphV2(
                nodes=self._nodes(),
                edges=[
                    {"from": "x", "to": "y", "strength": {"mean": 0.5, "std": 0.01}},
                    {"from": "x", "to": "y", "strength": {"mean": 0.5, "std": 0.01}},
                ],
            )

    def test_duplicate_edge_rejected_over_http(self, client):
        request = single_edge_request(0.5)
        request["graph"]["edges"].append(copy.deepcopy(request["graph"]["edges"][0]))
        response = client.post(f"{V2_URL}?response_version=2", json=request)
        assert response.status_code == 422
        assert "Duplicate edges found" in response.text

    def test_distinct_edges_and_mixed_edge_types_still_allowed(self):
        """Reverse-direction pairs and directed+bidirected on the same pair
        are semantically distinct and must not be rejected."""
        graph = GraphV2(
            nodes=[
                {"id": "a", "kind": "factor", "label": "A"},
                {"id": "b", "kind": "factor", "label": "B"},
                {"id": "y", "kind": "outcome", "label": "Y"},
            ],
            edges=[
                {"from": "a", "to": "y", "strength": {"mean": 0.5, "std": 0.01}},
                {"from": "b", "to": "y", "strength": {"mean": 0.3, "std": 0.01}},
                {"from": "a", "to": "b", "strength": {"mean": 0.2, "std": 0.01}},
                # Bidirected confounder sharing endpoints with a directed edge
                {
                    "from": "a",
                    "to": "b",
                    "edge_type": "bidirected",
                    "strength": {"mean": 0.1, "std": 0.01},
                },
            ],
        )
        assert len(graph.edges) == 4


# =============================================================================
# 5. Determinism and analytic-content assertions
# =============================================================================

VOLATILE_RESPONSE_FIELDS = {"processing_time_ms", "timestamp"}


def _stable_body(body: dict) -> dict:
    """Strip per-response volatile fields: timing, and critique correlation
    ids (random UUIDs by design). All scientific content is kept."""
    stable = {k: v for k, v in body.items() if k not in VOLATILE_RESPONSE_FIELDS}
    if "critiques" in stable:
        stable["critiques"] = [
            {k: v for k, v in critique.items() if k != "id"} for critique in stable["critiques"]
        ]
    return stable


class TestDeterminismAndAnalytics:
    def test_same_seed_full_response_equality(self, client):
        """Two identical seeded requests produce byte-identical responses
        modulo timing fields. Runs without parameter_uncertainties so no
        adaptive bootstrap is involved (that path is pinned separately)."""
        request = outcome_goal_request(
            request_id="tier0-determinism",  # pin: generated IDs differ per call
            goal_threshold=0.1,
            goal_constraints=[{"node_id": "revenue", "operator": ">=", "value": 0.1}],
        )
        first = client.post(f"{V2_URL}?response_version=2", json=request).json()
        second = client.post(f"{V2_URL}?response_version=2", json=request).json()
        assert _stable_body(first) == _stable_body(second)

    def test_same_seed_reproducible_with_uncertainties(self):
        """Analyzer-level determinism including factor sensitivity, with the
        bootstrap count pinned to remove the adaptive wall-clock budget."""
        analyzer = RobustnessAnalyzerV2()
        analyzer._n_bootstrap_override = 10
        request = outcome_goal_request(
            request_id="tier0-determinism-analyzer",  # pin: generated IDs differ per call
            parameter_uncertainties=[{"node_id": "price", "distribution": "normal", "std": 0.1}],
        )
        first = analyzer.analyze(RobustnessRequestV2(**request))
        second = analyzer.analyze(RobustnessRequestV2(**request))

        exclude = {"metadata": {"execution_time_ms"}}
        assert first.model_dump(exclude=exclude) == second.model_dump(exclude=exclude)

    def test_single_edge_analytic_magnitude(self):
        """x->y with strength ~0.5: do(x=v) must give mean(y) ~= 0.5*v.
        Auto-noise is mean-zero so option means stay on the analytic value."""
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(RobustnessRequestV2(**single_edge_request(0.5)))
        means = {r.option_id: r.outcome_distribution.mean for r in response.results}
        assert means["lo"] == pytest.approx(0.10, abs=0.02)
        assert means["hi"] == pytest.approx(0.40, abs=0.02)
        assert response.recommended_option_id == "hi"
        wins = {r.option_id: r.win_probability for r in response.results}
        assert wins["hi"] > 0.9

    def test_negative_edge_preserves_sign(self):
        """Negative strength flips the ordering: larger intervention loses."""
        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(RobustnessRequestV2(**single_edge_request(-0.5)))
        means = {r.option_id: r.outcome_distribution.mean for r in response.results}
        assert means["lo"] == pytest.approx(-0.10, abs=0.02)
        assert means["hi"] == pytest.approx(-0.40, abs=0.02)
        assert means["hi"] < means["lo"] < 0
        assert response.recommended_option_id == "lo"
