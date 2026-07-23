"""A3 honesty-residuals: per-endpoint config_details honesty.

History: until 23 Jul 2026 `generate_config_details()` emitted
`monte_carlo_samples: settings.MAX_MONTE_CARLO_ITERATIONS` (10000) into every
response's `_metadata.config_details`. #96 briefly gated it behind a `sampling`
flag (emitted only on "sampling" routes); C2 (F-3, 23 Jul) then removed it
OUTRIGHT — both the key AND the `sampling` parameter are gone. NO live compute
path uses that cap as its operative sample budget: the counterfactual engine runs
ADAPTIVE Monte Carlo (convergence-driven n <= the cap), the sequential engine
draws no samples, and robustness draws `request.min_samples`. The V1 robustness
wire had served `monte_carlo_samples: 10000` beside its own `samples_tested: 200`
— a self-contradiction. A config key no code path consults is a fabricated
disclosure, so it was removed. `config_details` now emits exactly the four
determinism-relevant keys {confidence_level, response_timeout, redis_enabled,
deterministic_mode}. (`MAX_MONTE_CARLO_ITERATIONS` still feeds
`config_fingerprint` via generate_config_fingerprint — the fingerprint is
unchanged.)

This suite asserts the ABSENCE of `monte_carlo_samples` across all three live
routes AND the helper, with an exact-key-order/presence positive control (the four
keys ARE emitted — trap #13; re-adding the key makes the order/absence assertions
RED). Wire impact: the cf/sequential `config_details` were byte-unchanged by C2
(they already omitted the key via the old sampling=False); only the robustness V1
wire lost it.

A3 Lane M / C3 (23 Jul): the pre-existing alias-drop bug is FIXED.
RobustnessResponse.metadata is aliased `_metadata`; the V1 /analyze route's
`RobustnessResponse(metadata=create_response_metadata(...))` (passed by FIELD
NAME) was silently dropped by Pydantic v2 and `_metadata` served None on the wire
— isl_version / config_fingerprint / config_details never reached any V1 client.
The route now attribute-assigns the metadata after construction, and C3 added
`populate_by_name: True` to the model, so the wire carries a POPULATED `_metadata`.
`test_robustness_wire_metadata_is_populated` asserts that (and doubles as the
wire-level positive control).
"""

import pytest

from src.models.metadata import create_response_metadata, generate_config_details

# `client` (async httpx) is the shared fixture in tests/conftest.py; the
# sequential decision-tree payload is the shared `sequential_analysis_request`
# fixture in tests/integration/conftest.py (C4 dedup).


def _config_details(response_json: dict) -> dict:
    return response_json["_metadata"]["config_details"]


# ---------------------------------------------------------------------------
# Valid payloads. The sequential route uses the shared `sequential_analysis_request`
# fixture (tests/integration/conftest.py); robustness/counterfactual are local.
# ---------------------------------------------------------------------------
def _robustness_payload() -> dict:
    return {
        "causal_model": {
            "nodes": ["price", "demand", "revenue"],
            "edges": [["price", "demand"], ["demand", "revenue"]],
        },
        "intervention_proposal": {"price": 55.0},
        "target_outcome": {"revenue": (95000.0, 105000.0)},
        "perturbation_radius": 0.1,
        "min_samples": 50,
        "confidence_level": 0.95,
    }


def _counterfactual_payload() -> dict:
    return {
        "model": {
            "variables": ["X", "Y"],
            "equations": {"Y": "2 * X"},
            "distributions": {"X": {"type": "normal", "parameters": {"mean": 5.0, "std": 1.0}}},
        },
        "intervention": {"X": 10.0},
        "outcome": "Y",
    }


# ===========================================================================
# Wire-level: monte_carlo_samples present ONLY where sampling happens
# ===========================================================================
class TestConfigDetailsPerRouteHonesty:
    @pytest.mark.asyncio
    async def test_robustness_wire_metadata_is_populated(self, client):
        """A3 Lane M (2026-07-23): the robustness V1 /analyze wire emits a
        POPULATED `_metadata`.

        Previously the route constructed
        `RobustnessResponse(metadata=create_response_metadata(...))` by FIELD NAME
        against the `_metadata` alias with no populate_by_name, so Pydantic v2
        silently dropped it and the wire served `_metadata: null` —
        isl_version / config_fingerprint / config_details never reached any V1
        client. The route now attribute-assigns the metadata after construction
        (the counterfactual/sequential live routes' working pattern).

        Doubles as the WIRE-level positive control (trap #13): the suite can SEE a
        populated `_metadata` on a live route."""
        resp = await client.post("/api/v1/robustness/analyze", json=_robustness_payload())
        assert resp.status_code == 200, resp.text
        meta = resp.json()["_metadata"]
        assert meta is not None, "V1 _metadata dropped — alias-drop regression"
        assert isinstance(meta["isl_version"], str) and meta["isl_version"]
        assert isinstance(meta["config_fingerprint"], str) and meta["config_fingerprint"]
        assert meta["request_id"]
        cd = meta["config_details"]
        # config_details no longer carries monte_carlo_samples (C2, F-3): robustness
        # draws real Monte Carlo but its budget is request.min_samples-driven, NOT
        # MAX_MONTE_CARLO_ITERATIONS, so the fixed 10000 was a fabricated disclosure.
        assert "monte_carlo_samples" not in cd
        # the four determinism-relevant transparency keys remain
        assert list(cd.keys()) == [
            "confidence_level",
            "response_timeout",
            "redis_enabled",
            "deterministic_mode",
        ]

    @pytest.mark.asyncio
    async def test_counterfactual_omits_monte_carlo_samples(self, client):
        """RED at HEAD: the counterfactual config_details carries
        monte_carlo_samples: 10000. The route declares sampling=False, so the key
        is ABSENT (absent-not-null)."""
        resp = await client.post("/api/v1/causal/counterfactual", json=_counterfactual_payload())
        assert resp.status_code == 200, resp.text
        cd = _config_details(resp.json())
        assert "monte_carlo_samples" not in cd
        # the other transparency keys remain
        assert "confidence_level" in cd
        assert "deterministic_mode" in cd

    @pytest.mark.asyncio
    async def test_sequential_omits_monte_carlo_samples(self, client, sequential_analysis_request):
        """RED at HEAD: the sequential config_details carries
        monte_carlo_samples: 10000 despite the sequential engine drawing NO
        samples. The route declares sampling=False, so the key is ABSENT."""
        resp = await client.post("/api/v1/analysis/sequential", json=sequential_analysis_request)
        assert resp.status_code == 200, resp.text
        cd = _config_details(resp.json())
        assert "monte_carlo_samples" not in cd
        assert "confidence_level" in cd
        assert "deterministic_mode" in cd


# ===========================================================================
# Helper-level: config_details emits exactly four keys, no monte_carlo_samples
# ===========================================================================
class TestGenerateConfigDetails:
    def test_config_details_is_exactly_four_keys_in_order(self):
        """config_details emits EXACTLY the four determinism-relevant keys, in
        order — monte_carlo_samples is gone (C2, F-3). Presence positive control
        AND order/absence pin: re-adding the key makes this RED (C2 mutation)."""
        d = generate_config_details()
        assert list(d.keys()) == [
            "confidence_level",
            "response_timeout",
            "redis_enabled",
            "deterministic_mode",
        ]
        assert "monte_carlo_samples" not in d

    def test_create_response_metadata_config_details_omits_mc(self):
        """create_response_metadata's config_details carries the same four keys and
        never monte_carlo_samples (no route re-introduces the fabricated cap)."""
        cd = create_response_metadata("req_a").config_details
        assert "monte_carlo_samples" not in cd
        assert list(cd.keys()) == [
            "confidence_level",
            "response_timeout",
            "redis_enabled",
            "deterministic_mode",
        ]
