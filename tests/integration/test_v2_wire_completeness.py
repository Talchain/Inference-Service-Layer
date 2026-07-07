"""
V2 wire-completeness additive-shape tests (lane ISL-R3, audit T1-5 / T1-6).

Asserts that already-computed analysis content actually reaches the V2 HTTP
wire envelope (in-process TestClient, network-free):

A. robustness.edge_sensitivity — edge-level forced-existence + magnitude
   contrasts (V1 content, V2 naming style) that the V2 envelope previously
   dropped (consumers saw EDGE_SENSITIVITY_UNAVAILABLE_V2_WIRE).
B. factor_evpi labelling keys (evpi_status / evpi_noise_floor /
   evpi_noise_floor_method / evpi_labelling_doctrine) serialize end-to-end
   onto the HTTP response, not just onto the internal analyzer dicts.
C. path_decomposition — request-gated (include_path_decomposition) content
   that was previously computed then discarded on the V2 path; plus the
   sensitivity_reference_option_id disclosure (sensitivity analyses run
   against options[0]).

All new fields are ADDITIVE OPTIONAL: absent (exclude_none) when not
computed, never renamed or repurposed. Same-seed determinism of the new
sections is asserted explicitly.
"""

import copy

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

V2_URL = "/api/v1/robustness/analyze/v2"


@pytest.fixture
def client():
    """Synchronous in-process test client."""
    return TestClient(app)


def base_request(**overrides):
    """3-node chain (price -> demand -> revenue, plus price -> revenue) with
    factor uncertainty on price. Small n_samples: shape tests, not science."""
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
        "n_samples": 300,
        "seed": 42,
    }
    request.update(overrides)
    return request


def full_request(**overrides):
    """base_request plus the request-gated enhancements exercised here."""
    return base_request(
        parameter_uncertainties=[{"node_id": "price", "distribution": "normal", "std": 0.1}],
        include_voi=True,
        include_path_decomposition=True,
        **overrides,
    )


def post_v2(client, request):
    """POST to the V2-enhanced envelope path (response_version=2).

    The bare endpoint defaults to the legacy V1-shaped response
    (DEFAULT_RESPONSE_VERSION=1); the wire-completeness fields under test
    live on the V2 envelope, which is what PLoT consumes.
    """
    response = client.post(f"{V2_URL}?response_version=2", json=request)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# A. robustness.edge_sensitivity (T1-6)
# ---------------------------------------------------------------------------


class TestEdgeSensitivityOnWire:
    def test_edge_sensitivity_present_and_shaped(self, client):
        body = post_v2(client, base_request())

        assert "robustness" in body
        edge_sensitivity = body["robustness"].get("edge_sensitivity")
        assert isinstance(edge_sensitivity, list) and edge_sensitivity, (
            "edge_sensitivity must be a non-empty list on the V2 wire when the "
            "default analysis_types (which include 'sensitivity') are used"
        )

        # 3 edges x 2 contrasts (existence + magnitude)
        assert len(edge_sensitivity) == 6

        requested_edge_ids = {"price->demand", "demand->revenue", "price->revenue"}
        seen_edge_ids = set()
        seen_types = set()
        seen_ranks = []
        for entry in edge_sensitivity:
            # V2 naming style, mirroring V1 SensitivityResult content
            assert entry["edge_id"] == f"{entry['from_id']}->{entry['to_id']}"
            assert entry["sensitivity_type"] in ("existence", "magnitude")
            assert 0.0 <= entry["sensitivity_score"] <= 1.0
            assert entry["direction"] in ("positive", "negative")
            assert isinstance(entry["elasticity"], (int, float))
            assert entry["importance_rank"] >= 1
            assert isinstance(entry["interpretation"], str) and entry["interpretation"]
            seen_edge_ids.add(entry["edge_id"])
            seen_types.add(entry["sensitivity_type"])
            seen_ranks.append(entry["importance_rank"])

        assert seen_edge_ids == requested_edge_ids
        assert seen_types == {"existence", "magnitude"}
        # Ranks are 1..N with no gaps (rank assigned after sorting by |elasticity|)
        assert sorted(seen_ranks) == list(range(1, len(edge_sensitivity) + 1))
        # Normalisation: the top-ranked entry scores 1.0
        top = next(e for e in edge_sensitivity if e["importance_rank"] == 1)
        assert top["sensitivity_score"] == pytest.approx(1.0)

    def test_edge_sensitivity_direction_matches_elasticity_sign(self, client):
        body = post_v2(client, base_request())
        for entry in body["robustness"]["edge_sensitivity"]:
            if entry["elasticity"] > 0:
                assert entry["direction"] == "positive"
            else:
                assert entry["direction"] == "negative"

    def test_edge_sensitivity_absent_when_sensitivity_not_requested(self, client):
        body = post_v2(client, base_request(analysis_types=["comparison", "robustness"]))
        # Additive optional + exclude_none: absent, never null/empty-lie
        assert "edge_sensitivity" not in body.get("robustness", {})


# ---------------------------------------------------------------------------
# B. factor_evpi labelling keys serialize end-to-end (lane-4 producer keys)
# ---------------------------------------------------------------------------


class TestFactorEvpiLabellingOnWire:
    def test_evpi_labelling_keys_on_http_response(self, client):
        body = post_v2(client, full_request())

        factor_evpi = body.get("factor_evpi")
        assert isinstance(factor_evpi, list) and factor_evpi, (
            "factor_evpi must be present when include_voi=true and "
            "parameter_uncertainties are supplied"
        )

        for entry in factor_evpi:
            # Base keys
            assert entry["factor_id"] == "price"
            assert isinstance(entry["evpi"], (int, float))
            assert isinstance(entry["evpi_percentage_points"], (int, float))
            assert entry["metric_type"] in ("p_joint_goal", "p_win_recommended")
            assert entry["n_evpi_samples"] == 300
            # The 4 additive labelling keys (lane 4 producer; PLoT forward-tolerates)
            assert entry["evpi_status"] in ("below_resolution", "resolved")
            assert isinstance(entry["evpi_noise_floor"], (int, float))
            assert entry["evpi_noise_floor"] > 0
            assert entry["evpi_noise_floor_method"] == "z95_worst_case_bernoulli_diff"
            assert entry["evpi_labelling_doctrine"] == "provisional_doctrine_v0"

    def test_factor_evpi_absent_without_include_voi(self, client):
        body = post_v2(client, base_request())
        assert "factor_evpi" not in body


# ---------------------------------------------------------------------------
# C. path_decomposition + sensitivity_reference_option_id (T1-6 / T1-5)
# ---------------------------------------------------------------------------


class TestPathDecompositionOnWire:
    def test_path_decomposition_present_when_requested(self, client):
        body = post_v2(client, full_request())

        pd = body.get("path_decomposition")
        assert pd is not None, (
            "path_decomposition must reach the V2 wire when "
            "include_path_decomposition=true (previously computed then discarded)"
        )
        assert pd["recommended_option_id"] in ("low", "high")
        assert pd["entry_nodes"] == ["price"]
        assert pd["truncated"] is False
        # price->demand->revenue and price->revenue
        assert pd["path_count"] == 2
        assert isinstance(pd["paths"], list) and pd["paths"]
        for path_entry in pd["paths"]:
            assert path_entry["path"][0] == "price"
            assert path_entry["path"][-1] == "revenue"
            assert isinstance(path_entry["path_effect"], (int, float))
            assert isinstance(path_entry["total_effect"], (int, float))
            assert path_entry["status"] in ("computed", "indeterminate")
            assert isinstance(path_entry["mechanism"], str) and path_entry["mechanism"]

    def test_path_decomposition_absent_when_not_requested(self, client):
        body = post_v2(client, base_request())
        assert "path_decomposition" not in body


class TestSensitivityReferenceOptionDisclosure:
    def test_reference_option_disclosed(self, client):
        body = post_v2(client, base_request())
        # Sensitivity analyses run against options[0]; the wire now discloses it
        assert body.get("sensitivity_reference_option_id") == "low"

    def test_reference_matches_first_option_not_recommended(self, client):
        # Reorder options: reference must track options[0], not the winner
        request = base_request()
        request["options"] = list(reversed(request["options"]))
        body = post_v2(client, request)
        assert body["sensitivity_reference_option_id"] == "high"

    def test_reference_absent_when_no_sensitivity_computed(self, client):
        body = post_v2(client, base_request(analysis_types=["comparison", "robustness"]))
        assert "sensitivity_reference_option_id" not in body


# ---------------------------------------------------------------------------
# Determinism + additive-envelope invariants
# ---------------------------------------------------------------------------


class TestWireCompletenessInvariants:
    def test_same_seed_new_sections_are_deterministic(self, client):
        request = full_request()
        body_a = post_v2(client, copy.deepcopy(request))
        body_b = post_v2(client, copy.deepcopy(request))

        assert body_a["robustness"]["edge_sensitivity"] == body_b["robustness"]["edge_sensitivity"]
        assert body_a["factor_evpi"] == body_b["factor_evpi"]
        assert body_a["path_decomposition"] == body_b["path_decomposition"]
        assert (
            body_a["sensitivity_reference_option_id"] == body_b["sensitivity_reference_option_id"]
        )

    def test_existing_envelope_fields_unaffected(self, client):
        """Additive-only guard: the pre-existing envelope keys still appear."""
        body = post_v2(client, full_request())
        for key in (
            "analysis_status",
            "robustness_status",
            "factor_sensitivity_status",
            "options",
            "robustness",
            "factor_sensitivity",
            "inference_warnings",
            "request_id",
            "processing_time_ms",
            "seed_used",
            "seed_source",
        ):
            assert key in body, f"pre-existing envelope key {key!r} missing"
        assert body["analysis_status"] == "computed"
        # Pre-existing robustness sub-keys survive alongside edge_sensitivity
        for key in ("level", "confidence", "is_robust", "recommendation_stability"):
            assert key in body["robustness"]
