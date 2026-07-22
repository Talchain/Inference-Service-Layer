"""
Track S — per-factor provenance echo (Brief B).

ISL already *receives* value provenance on each factor's observed_state
(`source`, `extractionType`) but historically dropped it. These tests verify
the enhanced V2 response now echoes that provenance onto each
FactorSensitivityV2, and surfaces `value_defaulted` when a factor's value was
not supplied (so it fell back to 0.0).

Echo-only contract: ISL does NOT consume these fields in inference — it only
passes through what it was given plus the already-known defaulted flag.

All assertions go through the real endpoint so they cover model + serialisation
(by_alias=True, exclude_none=True) end to end.
"""

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}

# The v2_client fixture and the two_factor_request builder are shared via
# tests/integration/conftest.py (see test_confidence_provenance.py, same suite).


def _marketing_factor(client, request):
    """POST the request and return the 'marketing' factor_sensitivity entry (a dict)."""
    resp = client.post(ENDPOINT, json=request, headers=V2_HEADERS)
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("factor_sensitivity"), "expected non-empty factor_sensitivity"
    marketing = next(
        (fs for fs in data["factor_sensitivity"] if fs["node_id"] == "marketing"), None
    )
    assert marketing is not None, f"no 'marketing' factor in {data['factor_sensitivity']}"
    return marketing


class TestProvenanceEcho:
    """Brief B — value-origin passthrough onto FactorSensitivityV2."""

    def test_value_source_echoed(self, v2_client, two_factor_request):
        """1. observed_state.source is echoed to value_source."""
        marketing = _marketing_factor(
            v2_client, two_factor_request({"value": 50.0, "source": "user_input"})
        )
        assert marketing["value_source"] == "user_input"

    def test_extraction_type_echoed(self, v2_client, two_factor_request):
        """2. observed_state.extractionType is echoed to value_extraction_type."""
        marketing = _marketing_factor(
            v2_client,
            two_factor_request(
                {"value": 50.0, "source": "brief_extraction", "extractionType": "inferred"}
            ),
        )
        assert marketing["value_extraction_type"] == "inferred"

    def test_defaulted_value_flagged_true(self, v2_client, two_factor_request):
        """3. Factor with uncertainty but no observed value → value_defaulted: true."""
        marketing = _marketing_factor(v2_client, two_factor_request(marketing_observed_state=None))
        assert marketing["value_defaulted"] is True

    def test_provided_value_not_flagged_defaulted(self, v2_client, two_factor_request):
        """4. Explicitly-provided value → value_defaulted absent (not defaulted)."""
        marketing = _marketing_factor(
            v2_client, two_factor_request({"value": 50.0, "source": "user_input"})
        )
        assert "value_defaulted" not in marketing

    def test_no_provenance_metadata_fields_absent(self, v2_client, two_factor_request):
        """5. Factor with a value but no source/extractionType → echo fields absent,
        not null-injected, and the response is still valid."""
        marketing = _marketing_factor(v2_client, two_factor_request({"value": 50.0}))
        assert "value_source" not in marketing
        assert "value_extraction_type" not in marketing
        # Value WAS provided, so it is not flagged as defaulted either.
        assert "value_defaulted" not in marketing
        # Response remains valid: required factor fields are present.
        assert marketing["node_id"] == "marketing"
        assert "direction" in marketing
        assert "sensitivity_score" in marketing

    def test_provenance_deterministic(self, v2_client, two_factor_request):
        """6. Same input + seed → identical provenance fields across calls."""
        request = two_factor_request(
            {"value": 50.0, "source": "user_input", "extractionType": "explicit"}
        )
        first = _marketing_factor(v2_client, request)
        second = _marketing_factor(v2_client, request)
        for key in ("value_source", "value_extraction_type", "value_defaulted"):
            assert first.get(key) == second.get(key)
        assert first["value_source"] == "user_input"
        assert first["value_extraction_type"] == "explicit"

    def test_present_fields_not_null_injected(self, v2_client, two_factor_request):
        """7. exclude_none contract: supplied provenance is present and non-null;
        the unsupplied value_defaulted is omitted rather than serialised as null."""
        marketing = _marketing_factor(
            v2_client,
            two_factor_request({"value": 50.0, "source": "computed", "extractionType": "explicit"}),
        )
        assert marketing["value_source"] == "computed"
        assert marketing["value_extraction_type"] == "explicit"
        assert marketing["value_source"] is not None
        assert marketing["value_extraction_type"] is not None
        # Value provided → value_defaulted omitted (not present as null).
        assert "value_defaulted" not in marketing
