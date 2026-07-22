"""
S2 — confidence-provenance disclosure marker (endpoint-level emission).

ISL emits a per-factor `confidence` figure derived from PROVISIONAL constants
(STABILITY_CONFIDENCE_MAP etc., "Neil gate 1, NOT research-validated").
Downstream renders it as a precise number with no marker that it is
uncalibrated. S2 rides an additive, honest disclosure marker
`confidence_provenance` = {method_version, calibrated} alongside `confidence`,
so any future recalibration must be a DISCLOSED, versioned change.

Emission rule under test: `confidence_provenance` is populated EXACTLY when
`confidence` is populated, and ABSENT (never a JSON null) when it is not.

All assertions go through the real endpoint so they cover model + serialisation
(by_alias=True, exclude_none=True) end to end — the same path
tests/integration/test_provenance_echo.py exercises for the Track-S echo added
in the same emission block.
"""

import pytest

from fastapi.testclient import TestClient

from src.api.main import app

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}

# The frozen wire contract for S2. Pinned as a literal (not imported) so this
# endpoint test asserts the exact bytes a consumer sees, independent of the
# source constant.
EXPECTED_METHOD_VERSION = "stability-cv-blend-v1"


@pytest.fixture
def client():
    return TestClient(app)


def _build_request():
    """2-factor request where 'marketing' is the uncertain factor driving
    factor_sensitivity (mirrors test_provenance_echo)."""
    return {
        "graph": {
            "nodes": [
                {"id": "price", "kind": "factor", "label": "Price"},
                {"id": "marketing", "kind": "factor", "label": "Marketing"},
                {"id": "revenue", "kind": "goal", "label": "Revenue"},
            ],
            "edges": [
                {"from": "price", "to": "revenue", "strength": {"mean": 0.6, "std": 0.15}},
                {"from": "marketing", "to": "revenue", "strength": {"mean": 0.5, "std": 0.15}},
            ],
        },
        "options": [
            {"id": "opt1", "label": "Raise price", "interventions": {"price": 120}},
            {"id": "opt2", "label": "Lower price", "interventions": {"price": 80}},
        ],
        "goal_node_id": "revenue",
        "seed": 42,
        "n_samples": 200,
        "parameter_uncertainties": [{"node_id": "marketing", "distribution": "normal", "std": 5.0}],
    }


def _factors(client):
    resp = client.post(ENDPOINT, json=_build_request(), headers=V2_HEADERS)
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("factor_sensitivity"), "expected non-empty factor_sensitivity"
    return data["factor_sensitivity"]


class TestConfidenceProvenanceEmission:
    """S2 — the disclosure marker rides the live wire next to confidence."""

    def test_confidence_provenance_present_when_confidence_present(self, client):
        """A factor that carries `confidence` also carries `confidence_provenance`
        = {method_version, calibrated:false}."""
        factors = _factors(client)
        marketing = next((fs for fs in factors if fs["node_id"] == "marketing"), None)
        assert marketing is not None, f"no 'marketing' factor in {factors}"
        # Precondition: confidence IS populated on this factor.
        assert marketing.get("confidence") is not None
        # The additive disclosure marker is present and well-formed.
        assert (
            "confidence_provenance" in marketing
        ), "confidence_provenance must ride the wire whenever confidence does"
        prov = marketing["confidence_provenance"]
        assert prov == {
            "method_version": EXPECTED_METHOD_VERSION,
            "calibrated": False,
        }, f"unexpected confidence_provenance payload: {prov}"

    def test_marker_present_exactly_when_confidence_present(self, client):
        """Across EVERY emitted factor: confidence present <=> provenance present,
        provenance never a JSON null, and it always discloses uncalibrated."""
        factors = _factors(client)
        assert factors, "expected at least one factor"
        for fs in factors:
            has_conf = fs.get("confidence") is not None
            has_prov = "confidence_provenance" in fs
            assert has_prov == has_conf, (
                f"provenance/confidence presence mismatch for {fs.get('node_id')}: "
                f"confidence={fs.get('confidence')}, has_prov={has_prov}"
            )
            if has_prov:
                prov = fs["confidence_provenance"]
                assert prov is not None  # present, not null-injected
                assert prov["method_version"] == EXPECTED_METHOD_VERSION
                # The mapping is provisional (Neil gate 1) — always disclosed uncalibrated.
                assert prov["calibrated"] is False

    def test_marker_deterministic(self, client):
        """Same input + seed → identical disclosure marker across calls."""
        first = _factors(client)
        second = _factors(client)
        fm = next(fs for fs in first if fs["node_id"] == "marketing")
        sm = next(fs for fs in second if fs["node_id"] == "marketing")
        assert fm.get("confidence_provenance") == sm.get("confidence_provenance")


class TestGraphStructuralFallbackVersion:
    """F-2 — the graph_structural fallback stamps its OWN method_version, never the
    bootstrap blend's. The branch is endpoint-unreachable today (bootstrap always
    runs when factors are emitted), so we drive it DIRECTLY: monkeypatching the
    bootstrap confidence to return None forces the emission down the fallback path,
    which is reachable at this level."""

    GRAPH_STRUCTURAL_METHOD_VERSION = "graph-structural-v1"

    def test_fallback_stamps_graph_structural_version(self, client, monkeypatch):
        import src.api.robustness as rob

        # Force the fallback: bootstrap-derived confidence returns None, so the
        # emission takes the graph_structural branch for every factor.
        monkeypatch.setattr(rob, "compute_factor_confidence", lambda *a, **k: None)

        factors = _factors(client)
        marketing = next((fs for fs in factors if fs["node_id"] == "marketing"), None)
        assert marketing is not None, f"no 'marketing' factor in {factors}"
        # Precondition: we really are on the fallback path.
        assert marketing.get("confidence_source") == "graph_structural"
        # The marker must name the FALLBACK method, not the bootstrap blend.
        prov = marketing.get("confidence_provenance")
        assert prov is not None, "fallback confidence must still carry a marker"
        assert prov["method_version"] == self.GRAPH_STRUCTURAL_METHOD_VERSION
        assert prov["method_version"] != EXPECTED_METHOD_VERSION
        assert prov["calibrated"] is False
