"""
LOW_EFFECTIVE_SAMPLES disclosure on partial-validity options (ROADMAP 2.475).

Drives a run whose Monte Carlo samples are genuinely, partially non-finite
through the REAL analyze/v2 path (no mocking of status or samples): two root
factors carry finite observed values of 1.7e308; when both edges to the goal
exist in a sample the contributions sum past float64 max and the goal outcome
is inf. One edge is a coin flip (exists_probability 0.5), so roughly half of
opt_overflow's samples are non-finite and the rest are finite — validity
ratio ~0.46, below MIN_VALID_RATIO (0.8) and above zero: exactly the
'partial' state. opt_healthy pins both huge factors to modest values, so its
samples are always finite (the in-suite presence control for the absence
assertions in the healthy-run test).

Before the 2.475 fix this request could not even ship: n_valid was counted
AFTER median imputation (always n_total), and the raw-population mean/std
carried inf/nan into the response, which the JSON render rejects — HTTP 500.
The assertions below therefore pin, in one place: honest validity counting,
the LOW_EFFECTIVE_SAMPLES emission (identity-bound to the constructed option
id), and the serializability of the disclosed response.
"""

import json
import os

import pytest

from fastapi.testclient import TestClient

from src.api.main import app

# float64 max is ~1.797e308: one HUGE contribution stays finite, two sum to inf
HUGE_FINITE = 1.7e308

N_SAMPLES = 500
SEED = 42


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    if os.environ.get("ISL_AUTH_DISABLED", "").lower() == "true":
        return {}
    return {"X-API-Key": os.environ.get("ISL_API_KEY", "test_key")}


def _graph(observed_value: float) -> dict:
    """Shared topology; observed_value selects degenerate vs healthy regime."""
    return {
        "nodes": [
            {
                "id": "f_certain",
                "kind": "factor",
                "label": "Certain factor",
                "observed_state": {"value": observed_value},
            },
            {
                "id": "f_coin",
                "kind": "factor",
                "label": "Coinflip factor",
                "observed_state": {"value": observed_value},
            },
            {"id": "d_lever", "kind": "decision", "label": "Small lever"},
            {"id": "goal_out", "kind": "outcome", "label": "Goal"},
        ],
        "edges": [
            {
                "from": "f_certain",
                "to": "goal_out",
                "exists_probability": 1.0,
                "strength": {"mean": 1.0, "std": 0.05},
            },
            {
                "from": "f_coin",
                "to": "goal_out",
                "exists_probability": 0.5,
                "strength": {"mean": 1.0, "std": 0.05},
            },
            {
                "from": "d_lever",
                "to": "goal_out",
                "exists_probability": 1.0,
                "strength": {"mean": 0.9, "std": 0.05},
            },
        ],
    }


def _partial_validity_request() -> dict:
    """opt_overflow hits inf on ~half its samples; opt_healthy pins the huge factors away."""
    return {
        "graph": _graph(HUGE_FINITE),
        "options": [
            {
                "id": "opt_overflow",
                "label": "Overflow option",
                "interventions": {"d_lever": 0.7},
            },
            {
                "id": "opt_healthy",
                "label": "Healthy option",
                "interventions": {"f_certain": 0.5, "f_coin": 0.5},
            },
        ],
        "goal_node_id": "goal_out",
        "n_samples": N_SAMPLES,
        "seed": SEED,
    }


def _healthy_request() -> dict:
    """Same topology, modest magnitudes everywhere: every sample finite."""
    return {
        "graph": _graph(0.6),
        "options": [
            {
                "id": "opt_overflow",
                "label": "Overflow option",
                "interventions": {"d_lever": 0.7},
            },
            {
                "id": "opt_healthy",
                "label": "Healthy option",
                "interventions": {"f_certain": 0.5, "f_coin": 0.5},
            },
        ],
        "goal_node_id": "goal_out",
        "n_samples": N_SAMPLES,
        "seed": SEED,
    }


def _post(client, auth_headers, payload):
    return client.post(
        "/api/v1/robustness/analyze/v2?response_version=2",
        json=payload,
        headers=auth_headers,
    )


def _strict_parse(resp):
    """Parse asserting JSON compliance: Python's json.loads ACCEPTS Infinity/NaN
    tokens by default, so a plain .json() cannot prove the body is compliant.
    parse_constant fires exactly on those tokens."""

    def _reject(token):
        pytest.fail(f"non-JSON-compliant float token in response body: {token}")

    return json.loads(resp.text, parse_constant=_reject)


def _option_by_id(body, option_id):
    matches = [o for o in body["options"] if o["id"] == option_id]
    assert len(matches) == 1, f"expected exactly one option {option_id!r}"
    return matches[0]


class TestLowEffectiveSamplesDisclosure:
    def test_partial_validity_run_emits_low_effective_samples(self, client, auth_headers):
        resp = _post(client, auth_headers, _partial_validity_request())

        # Before the fix this request 500s at JSON render (inf in outcome.mean):
        # the disclosure must ship on a success response.
        assert resp.status_code == 200, resp.text[:500]
        body = _strict_parse(resp)

        # One option partial, one computed -> response-level 'partial' (the
        # builder's existing, honest derivation).
        assert body["analysis_status"] == "partial"
        assert body["status_reason"] == "Some options could not be computed"

        # --- the degraded option: honest validity metrics, driven not mocked ---
        overflow = _option_by_id(body, "opt_overflow")
        assert overflow["status"] == "partial"
        outcome = overflow["outcome"]
        n_valid = outcome["n_valid_samples"]
        assert 0 < n_valid < N_SAMPLES, "fixture must produce a strict mix of valid/invalid"
        assert outcome["n_samples"] == N_SAMPLES
        assert outcome["validity_ratio"] == pytest.approx(n_valid / N_SAMPLES)
        assert 0.0 < outcome["validity_ratio"] < 0.8  # below MIN_VALID_RATIO

        # --- the healthy option is untouched (in-run negative control) ---
        healthy = _option_by_id(body, "opt_healthy")
        assert healthy["status"] == "computed"
        assert healthy["outcome"]["n_valid_samples"] == N_SAMPLES
        assert healthy["outcome"]["validity_ratio"] == 1.0

        # --- the disclosure row, identity-bound to the constructed option ---
        les_rows = [c for c in body["critiques"] if c["code"] == "LOW_EFFECTIVE_SAMPLES"]
        assert len(les_rows) == 1, f"expected exactly one LES row, got {les_rows!r}"
        les = les_rows[0]
        assert les["affected_option_ids"] == ["opt_overflow"]
        # Placeholders populated from the SAME option's wire values, not template braces
        assert les["message"] == (f"Only {n_valid} of {N_SAMPLES} samples were numerically valid")
        assert "{" not in les["message"]
        assert les["severity"] == "warning"
        assert les["source"] == "analysis"
        assert les["suggestion"]  # registry default carried

        # The pre-existing co-emission still stands (CEE drops it; LES is the
        # row that survives to users).
        instability = [c for c in body["critiques"] if c["code"] == "NUMERICAL_INSTABILITY"]
        assert len(instability) == 1
        assert instability[0]["affected_option_ids"] == ["opt_overflow"]

    def test_healthy_run_emits_no_low_effective_samples(self, client, auth_headers):
        """Absence control — its ability to SEE a presence is proven by the
        positive test above running the same assertions path."""
        resp = _post(client, auth_headers, _healthy_request())

        assert resp.status_code == 200, resp.text[:500]
        body = _strict_parse(resp)

        assert body["analysis_status"] == "computed"
        for opt in body["options"]:
            assert opt["status"] == "computed"
            assert opt["outcome"]["n_valid_samples"] == N_SAMPLES
            assert opt["outcome"]["validity_ratio"] == 1.0

        les_rows = [c for c in body["critiques"] if c["code"] == "LOW_EFFECTIVE_SAMPLES"]
        assert les_rows == []
