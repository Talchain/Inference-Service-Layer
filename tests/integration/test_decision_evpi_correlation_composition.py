"""COMPOSITION — S1 decision_evpi × B3-S1 correlated factors (D-23.8 × D-23.4).

decision_evpi is a JOINT-population decision-value quantity (min over per-option
expected_regret on whatever joint samples were drawn). It is NOT an
independence-assuming per-factor attribution, so when B3-S1's Gaussian copula is
active it MUST:
  * remain EMITTED (computed from the CORRELATED joint population), and
  * NOT join correlation_model.suppressed_attributions (which suppresses only the
    per-factor attributions: factor_sensitivity, factor_evpi, conditional_winners).

This also proves the reverse interaction: B3-S1's suppression of factor_evpi/
factor_sensitivity does NOT trip the S1 ISLResponseV2 biconditional validator
(different fields — the validator reads options[].downside only), so a correlated
request still returns 200 with decision_evpi present.
"""

import pytest

from fastapi.testclient import TestClient

from src.api.main import app

V2_URL = "/api/v1/robustness/analyze/v2"


@pytest.fixture
def client():
    return TestClient(app)


def _request(correlations):
    """2 correlated normal factors (fa, fb) -> revenue; two non-empty-intervention
    options so there is a real decision (and a regret-bearing population). Mirrors
    B3-S1's activation fixture, adapted to the HTTP wire (non-empty interventions)."""
    return {
        "request_id": "corr-comp",
        "graph": {
            "nodes": [
                {"id": "fa", "kind": "factor", "label": "Fa", "observed_state": {"value": 1.0}},
                {"id": "fb", "kind": "factor", "label": "Fb", "observed_state": {"value": 1.0}},
                {"id": "rev", "kind": "outcome", "label": "Rev"},
            ],
            "edges": [
                {"from": "fa", "to": "rev", "exists_probability": 1.0, "strength": {"mean": 1.0, "std": 0.05}},
                {"from": "fb", "to": "rev", "exists_probability": 1.0, "strength": {"mean": 0.6, "std": 0.05}},
            ],
        },
        "options": [
            {"id": "o1", "label": "O1", "interventions": {"fb": 1.5}},
            {"id": "o2", "label": "O2", "interventions": {"fa": 1.2}},
        ],
        "goal_node_id": "rev",
        "n_samples": 2000,
        "seed": 42,
        "parameter_uncertainties": [
            {"node_id": "fa", "distribution": "normal", "std": 0.5},
            {"node_id": "fb", "distribution": "normal", "std": 0.5},
        ],
        "include_voi": True,
        "factor_correlations": correlations,
    }


def _post(client, correlations):
    resp = client.post(f"{V2_URL}?response_version=2", json=_request(correlations))
    assert resp.status_code == 200, resp.text
    return resp.json()


# Seed-42 deterministic captures (n=2000). Not load-bearing for the composition
# proof (the structural invariants below are), but a strong regression pin.
_DECISION_EVPI_ABSENT = 0.17814637228838173
_DECISION_EVPI_RHO_09 = 0.05684256141885782


class TestDecisionEvpiUnderCorrelation:
    def test_decision_evpi_emitted_under_active_correlation(self, client):
        """Active copula -> correlation_model present AND decision_evpi still emitted
        (== min of the CORRELATED per-option regrets on the wire)."""
        body = _post(client, [{"factor_a": "fa", "factor_b": "fb", "rho": 0.9}])
        assert body.get("correlation_model") is not None, "correlation must be active"
        assert body["correlation_model"]["method"] == "gaussian_copula_v1"
        regrets = [o["downside"]["expected_regret"] for o in body["options"] if o.get("downside")]
        assert regrets, "expected a regret-bearing population under correlation"
        assert body["decision_evpi"] is not None
        assert body["decision_evpi"] == pytest.approx(min(regrets), rel=1e-12)

    def test_decision_evpi_not_in_suppressed_attributions(self, client):
        """decision_evpi is a decision-value quantity, NOT an independence-assuming
        per-factor attribution — it must NOT be in the suppression manifest, which
        lists ONLY the per-factor attributions."""
        body = _post(client, [{"factor_a": "fa", "factor_b": "fb", "rho": 0.9}])
        suppressed = body["correlation_model"]["suppressed_attributions"]
        assert "decision_evpi" not in suppressed
        # positive control: correlation IS genuinely active (the per-factor
        # attributions ARE suppressed), so this is not a vacuous pass.
        assert "factor_evpi" in suppressed
        assert body.get("factor_evpi") is None
        assert body.get("factor_sensitivity") is None

    def test_suppressed_factor_evpi_does_not_trip_decision_evpi_validator(self, client):
        """Reverse interaction: B3-S1 suppresses factor_evpi/factor_sensitivity, yet
        the S1 ISLResponseV2 biconditional validator (reads options[].downside only)
        still admits the response — 200 with decision_evpi present. Different fields,
        proven end-to-end."""
        body = _post(client, [{"factor_a": "fa", "factor_b": "fb", "rho": 0.9}])
        assert body["decision_evpi"] is not None  # validator passed at construction
        assert any(o.get("downside") is not None for o in body["options"])

    def test_correlated_decision_evpi_differs_from_independent(self, client):
        """The number is computed FROM the correlated joint population, not the
        independent one: the rho=0.9 decision_evpi differs materially from the
        absent-correlation decision_evpi (both == their own min-regret)."""
        absent = _post(client, None)
        active = _post(client, [{"factor_a": "fa", "factor_b": "fb", "rho": 0.9}])
        assert absent.get("correlation_model") is None
        assert active.get("correlation_model") is not None
        assert absent["decision_evpi"] != active["decision_evpi"]
        for body in (absent, active):
            regrets = [o["downside"]["expected_regret"] for o in body["options"] if o.get("downside")]
            assert body["decision_evpi"] == pytest.approx(min(regrets), rel=1e-12)

    def test_seed42_value_pins(self, client):
        """Deterministic seed-42 regression pins (n=2000)."""
        absent = _post(client, None)
        active = _post(client, [{"factor_a": "fa", "factor_b": "fb", "rho": 0.9}])
        assert absent["decision_evpi"] == pytest.approx(_DECISION_EVPI_ABSENT, rel=1e-9)
        assert active["decision_evpi"] == pytest.approx(_DECISION_EVPI_RHO_09, rel=1e-9)
