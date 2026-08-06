"""
ROADMAP 2.720 — S3 transport half of the range→distribution converter.

Spec: parallel-briefs/RANGE-TO-DISTRIBUTION-SPEC-2026-08-08.md §4.2/§4.3/§5
(T8 presence, T9 compute byte-identity), applying ground-truth correction 1:
every ISL request model is extra="ignore", so an undeclared field DIES SILENTLY
at the Pydantic parse with a 200 — which is why T8 goes through the REAL
endpoint and why this file's pristine RED is precisely that silent drop.

DELIBERATE DESIGN: this file asserts on raw request/response JSON only and
imports nothing from the new range-fit modules — at pristine it fails on the
absent wire field (the defect under test), not on an import error.

T8  presence — the declared Optional field round-trips through the real
    /api/v1/robustness/analyze/v2 endpoint and the echo carries the fitted
    disclosure. Positive control FIRST (trap 13): prove the harness can see a
    present field before any absence assertion is trusted.
T9  compute byte-identity — same-seed analysis with and without a carried
    (unapplied) range is byte-identical outside the disclosure block: the S3
    "carried, not applied" claim, proven not asserted. This is also the
    mutant-M6 GREEN witness (deleting the converter must not touch compute).
"""

import copy
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}
SEED = 20260808
N_SAMPLES = 200

client = TestClient(app)


def _base_request() -> Dict[str, Any]:
    """Minimal two-option factor graph with a parameter uncertainty on the
    factor the user states a range for. Values are arbitrary but FIXED — T9
    compares two runs of this exact request, so nothing here is load-bearing
    beyond determinism."""
    return {
        "request_id": "range-fit-transport-test",
        "graph": {
            "nodes": [
                {
                    "id": "fac_a",
                    "kind": "factor",
                    "label": "fac_a",
                    "observed_state": {"value": 0.5},
                },
                {"id": "outcome", "kind": "outcome", "label": "outcome"},
            ],
            "edges": [
                {
                    "from": "fac_a",
                    "to": "outcome",
                    "strength": {"mean": 0.6, "std": 0.05},
                    "exists_probability": 0.9,
                },
            ],
        },
        "options": [
            {"id": "opt_hi", "label": "high", "interventions": {"fac_a": 0.8}},
            {"id": "opt_lo", "label": "low", "interventions": {"fac_a": 0.2}},
        ],
        "goal_node_id": "outcome",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "parameter_uncertainties": [
            {"node_id": "fac_a", "distribution": "normal", "std": 0.1},
        ],
    }


def _stated_range(
    node_id: str = "fac_a",
    lower: Optional[float] = 0.2,
    upper: Optional[float] = 0.6,
    domain: str = "unit_interval",
) -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "lower": lower,
        "upper": upper,
        "domain": domain,
        "source": "user",
        "stated_at": "2026-08-06T00:00:00Z",
    }


def _post(payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
    return client.post(ENDPOINT, json=payload, headers=headers or V2_HEADERS)


_VOLATILE_KEYS = {"execution_time_ms", "processing_time_ms", "timestamp"}


def _normalise(body: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the range-fit DISCLOSURE surface (the disclosure block and its
    RANGE_* refusal warnings) plus volatile timing fields; everything left is
    COMPUTE and must be byte-identical between the with-range and
    without-range runs. Stripping the warnings here (not per-test) keeps the
    byte-identity claim purely about compute — so it stays GREEN under
    converter deletion (mutant M6), exactly as the spec requires of T9.
    Whether disclosures/warnings are RIGHT is T8's job, not T9's."""

    def scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: scrub(v)
                for k, v in obj.items()
                if k not in _VOLATILE_KEYS and k != "range_fit_disclosures"
            }
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        return obj

    scrubbed = scrub(body)
    if isinstance(scrubbed.get("inference_warnings"), list):
        scrubbed["inference_warnings"] = [
            w
            for w in scrubbed["inference_warnings"]
            if not str(w.get("code", "")).startswith("RANGE_")
        ]
    return scrubbed


class TestT8TransportPresence:
    def test_positive_control_declared_field_round_trips(self) -> None:
        """POSITIVE CONTROL (trap 13): the harness can SEE a present field.
        At pristine this is the named RED: extra='ignore' swallows the field
        and no disclosure block exists."""
        payload = _base_request()
        payload["user_stated_ranges"] = [_stated_range()]
        response = _post(payload)
        assert response.status_code == 200, response.text
        body = response.json()

        assert "range_fit_disclosures" in body, (
            "user_stated_ranges was silently dropped at the Pydantic parse — "
            "the exact ground-truth-correction-1 failure this field must not have"
        )
        disclosures = body["range_fit_disclosures"]
        assert isinstance(disclosures, list) and len(disclosures) == 1
        entry = disclosures[0]
        # Identity-bound to the stated node, raw bounds echoed as said.
        assert entry["node_id"] == "fac_a"
        assert entry["lower"] == 0.2
        assert entry["upper"] == 0.6
        assert entry["domain"] == "unit_interval"
        # The fitted disclosure: beta family from the DECLARED domain.
        # (.get: the wire serialises exclude_none, so a None refusal is absent.)
        fitted = entry.get("fitted")
        assert fitted is not None
        assert fitted["family"] == "beta"
        assert fitted["alpha"] > 0
        assert fitted["beta"] > 0
        assert fitted["coverage"] == 0.5
        assert fitted["method_version"] == "range-iq-fit-v1"
        assert entry.get("refusal") is None

    def test_unbounded_domain_fits_normal_through_endpoint(self) -> None:
        payload = _base_request()
        payload["user_stated_ranges"] = [_stated_range(lower=0.2, upper=0.6, domain="unbounded")]
        response = _post(payload)
        assert response.status_code == 200, response.text
        fitted = response.json()["range_fit_disclosures"][0]["fitted"]
        assert fitted["family"] == "normal"
        assert fitted["mu"] == pytest.approx(0.4, abs=1e-12)
        assert fitted["sigma"] == pytest.approx(0.4 / (2 * 0.6744897501960817), rel=1e-9)

    def test_refusal_surfaces_with_named_warning(self) -> None:
        """An inverted range refuses RANGE_INVALID_ORDER through the real
        endpoint: disclosure carries the refusal, compute proceeds (200), and
        the degradation path names the code (spec §4.3 — a refusal the user
        never sees is a silent default with extra steps)."""
        payload = _base_request()
        payload["user_stated_ranges"] = [_stated_range(lower=0.6, upper=0.2)]
        response = _post(payload)
        assert response.status_code == 200, response.text
        body = response.json()

        entry = body["range_fit_disclosures"][0]
        assert entry.get("fitted") is None
        assert entry["refusal"]["code"] == "RANGE_INVALID_ORDER"

        warning_codes = [w["code"] for w in body.get("inference_warnings", [])]
        assert "RANGE_INVALID_ORDER" in warning_codes
        warning = next(w for w in body["inference_warnings"] if w["code"] == "RANGE_INVALID_ORDER")
        assert warning["severity"] == "warning"
        assert warning["field"] == "user_stated_ranges[fac_a]"

    def test_absent_field_absent_disclosures(self) -> None:
        """Inert-when-absent (licensed by the positive control above)."""
        response = _post(_base_request())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body.get("range_fit_disclosures") is None
        for warning in body.get("inference_warnings", []):
            assert not warning["code"].startswith("RANGE_"), warning

    def test_v1_response_format_carries_disclosures_too(self) -> None:
        payload = _base_request()
        payload["user_stated_ranges"] = [_stated_range()]
        response = client.post(ENDPOINT, json=payload)  # default = v1 format
        assert response.status_code == 200, response.text
        body = response.json()
        assert body.get("range_fit_disclosures"), "v1 wire format lost the disclosure"

    def test_unknown_node_id_is_422(self) -> None:
        """Transport validity is the estate's parse-time posture (matches
        parameter_uncertainties): a range for a node not in the graph is a
        malformed request, not a refusal."""
        payload = _base_request()
        payload["user_stated_ranges"] = [_stated_range(node_id="no_such_node")]
        response = _post(payload)
        assert response.status_code == 422, response.text

    def test_duplicate_node_id_is_422(self) -> None:
        payload = _base_request()
        payload["user_stated_ranges"] = [_stated_range(), _stated_range(lower=0.3)]
        response = _post(payload)
        assert response.status_code == 422, response.text

    def test_missing_domain_is_422_never_sniffed(self) -> None:
        """No domain declaration ⇒ parse refusal — family selection MUST come
        from declared metadata, never inferred from whether the values happen
        to lie in [0,1] (spec §2.2)."""
        payload = _base_request()
        stated = _stated_range()
        del stated["domain"]
        payload["user_stated_ranges"] = [stated]
        response = _post(payload)
        assert response.status_code == 422, response.text


class TestT9ComputeByteIdentity:
    def _run_pair(self, ranges: Optional[List[Dict[str, Any]]]) -> Any:
        payload = _base_request()
        if ranges is not None:
            payload["user_stated_ranges"] = ranges
        response = _post(payload)
        assert response.status_code == 200, response.text
        return response.json()

    def test_carried_range_leaves_compute_byte_identical(self) -> None:
        """Same seed, with vs without a carried (unapplied) fitting range:
        byte-identical outside the disclosure block. THE S3 claim (spec §4.3),
        and the R-11-style stream guard — the fit consumes zero RNG draws."""
        without = self._run_pair(None)
        with_range = self._run_pair([_stated_range()])

        # The disclosure must actually be present in the with-range run —
        # otherwise this equality is a control agreeing with itself (trap 13b).
        assert with_range.get("range_fit_disclosures")
        assert without.get("range_fit_disclosures") is None

        assert _normalise(without) == _normalise(with_range)

    def test_refused_range_also_leaves_compute_byte_identical(self) -> None:
        """Refusal is honest absence, never degraded presence (spec §3): the
        compute under a refused range is byte-identical too (the refusal's own
        named warning is part of the disclosure surface _normalise strips)."""
        without = self._run_pair(None)
        refused = self._run_pair([_stated_range(lower=0.6, upper=0.2)])

        assert refused["range_fit_disclosures"][0]["refusal"] is not None

        assert _normalise(without) == _normalise(refused)
