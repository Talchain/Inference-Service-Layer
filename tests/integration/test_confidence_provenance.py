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

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}

# The frozen wire contract for S2. Pinned as a literal (not imported) so this
# endpoint test asserts the exact bytes a consumer sees, independent of the
# source constant.
EXPECTED_METHOD_VERSION = "stability-cv-blend-v1"

# The v2_client fixture and the two_factor_request builder (whose defaults
# reproduce this suite's exact 2-factor request) are shared via
# tests/integration/conftest.py.


def _factors(client, two_factor_request):
    resp = client.post(ENDPOINT, json=two_factor_request(), headers=V2_HEADERS)
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("factor_sensitivity"), "expected non-empty factor_sensitivity"
    return data["factor_sensitivity"]


class TestConfidenceProvenanceEmission:
    """S2 — the disclosure marker rides the live wire next to confidence."""

    def test_marker_present_exactly_when_confidence_present(self, v2_client, two_factor_request):
        """Across EVERY emitted factor: confidence present <=> provenance present,
        provenance never a JSON null, and — when present — it is EXACTLY
        {method_version, calibrated:false} with no extra keys. The exact-dict byte
        check (formerly a single-factor test) is folded in here and applied to
        every factor, so it is strictly stronger."""
        factors = _factors(v2_client, two_factor_request)
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
                # Exact wire bytes: EXACTLY {method_version, calibrated:false}, no
                # extra keys. The mapping is provisional (Neil gate 1) — always
                # disclosed uncalibrated (calibrated is False).
                assert prov == {
                    "method_version": EXPECTED_METHOD_VERSION,
                    "calibrated": False,
                }, f"unexpected confidence_provenance payload for {fs.get('node_id')}: {prov}"


class TestGraphStructuralFallbackVersion:
    """F-2 — the graph_structural fallback stamps its OWN method_version, never the
    bootstrap blend's. The branch is endpoint-unreachable today (bootstrap always
    runs when factors are emitted), so we drive it DIRECTLY: monkeypatching the
    bootstrap confidence to return None forces the emission down the fallback path,
    which is reachable at this level."""

    GRAPH_STRUCTURAL_METHOD_VERSION = "graph-structural-v1"

    def test_fallback_stamps_graph_structural_version(
        self, v2_client, monkeypatch, two_factor_request
    ):
        import src.api.robustness as rob

        # Force the fallback: bootstrap-derived confidence returns None, so the
        # emission takes the graph_structural branch for every factor.
        monkeypatch.setattr(rob, "compute_factor_confidence", lambda *a, **k: None)

        factors = _factors(v2_client, two_factor_request)
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
