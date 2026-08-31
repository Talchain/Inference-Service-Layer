"""Carry existing MC statistics through the actual enhanced HTTP converter.

The saved counterpart is the pre-change enhanced response from staging28fe0c9.
The analyzer and its numerical outputs are unchanged: only the two statistics
already in its internal/legacy metadata are added to the enhanced envelope.
"""

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.models.response_v2 import ISLResponseV2

FIXTURE = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/science_transport/base_enhanced_responses.json"
    ).read_text()
)
CASES = FIXTURE["cases"]
ENDPOINT = "/api/v1/robustness/analyze/v2"
TARGET_FIELDS = {"tie_rate", "edge_existence_rates"}


@pytest.fixture
def client():
    return TestClient(app)


def without_volatile_fields(body):
    return {k: v for k, v in body.items() if k not in {"timestamp", "processing_time_ms"}}


@pytest.mark.parametrize(
    "name,expected_tie,expected_existence",
    [("frequent_ties", 0.91, 0.09), ("rare_ties", 0.07, 0.93), ("no_ties", 0.0, 1.0)],
)
def test_computed_statistics_survive_without_other_result_changes(
    client, name, expected_tie, expected_existence
):
    case = CASES[name]
    legacy = client.post(f"{ENDPOINT}?response_version=1", json=case["request"])
    enhanced = client.post(f"{ENDPOINT}?response_version=2", json=case["request"])
    assert legacy.status_code == enhanced.status_code == 200
    body = enhanced.json()
    metadata = legacy.json()["_metadata"]
    assert body["tie_rate"] == metadata["tie_rate"] == expected_tie
    assert (
        body["edge_existence_rates"]
        == metadata["edge_existence_rates"]
        == {"input->goal": expected_existence}
    )
    assert body["tie_rate"] == metadata["tie_count"] / metadata["n_samples_used"]
    # Exact old response counterpart guards against a carriage change silently
    # changing scientific results, recommendation fields or their meaning.
    existing_fields = {
        k: v for k, v in without_volatile_fields(body).items() if k not in TARGET_FIELDS
    }
    assert existing_fields == without_volatile_fields(case["enhanced_response"])


def test_blocked_computation_omits_statistics(client):
    case = CASES["not_computed"]
    response = client.post(f"{ENDPOINT}?response_version=2", json=case["request"])
    assert response.status_code == 422
    assert TARGET_FIELDS.isdisjoint(response.json())
    assert response.json() == case["enhanced_response"]


def test_zero_realised_edge_existence_reaches_wire(client):
    request = copy.deepcopy(CASES["rare_ties"]["request"])
    # Retain the option-to-goal path: making its sole edge impossible correctly
    # triggers the existing NO_EFFECTIVE_PATH_TO_GOAL admission refusal.
    request["graph"]["nodes"].append(
        {"id": "unused", "kind": "factor", "label": "Unused", "observed_state": {"value": 0.4}}
    )
    request["graph"]["edges"].append(
        {
            "from": "unused",
            "to": "goal",
            "exists_probability": 0.0,
            "strength": {"mean": 1, "std": 0.1},
        }
    )
    legacy = client.post(f"{ENDPOINT}?response_version=1", json=request)
    enhanced = client.post(f"{ENDPOINT}?response_version=2", json=request)
    assert legacy.status_code == enhanced.status_code == 200
    assert enhanced.json()["tie_rate"] == legacy.json()["_metadata"]["tie_rate"]
    assert (
        enhanced.json()["edge_existence_rates"]
        == legacy.json()["_metadata"]["edge_existence_rates"]
    )
    assert enhanced.json()["edge_existence_rates"]["unused->goal"] == 0.0


def test_old_successful_producer_stays_absent_under_new_model():
    # This is an actual old producer response, not zero-filled test data.
    old = CASES["rare_ties"]["enhanced_response"]
    parsed = ISLResponseV2.model_validate(old).model_dump(by_alias=True, exclude_none=True)
    assert TARGET_FIELDS.isdisjoint(parsed)


def test_explicit_computed_empty_map_is_preserved():
    body = {**CASES["rare_ties"]["enhanced_response"], "edge_existence_rates": {}}
    parsed = ISLResponseV2.model_validate(body).model_dump(by_alias=True, exclude_none=True)
    assert parsed["edge_existence_rates"] == {}
    assert "tie_rate" not in parsed


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["tie_rate", "edge_existence_rates"])
def test_invalid_statistics_cannot_enter_enhanced_contract(field, invalid):
    body = copy.deepcopy(CASES["rare_ties"]["enhanced_response"])
    body[field] = invalid if field == "tie_rate" else {"input->goal": invalid}
    with pytest.raises(ValidationError) as exc:
        ISLResponseV2.model_validate(body)
    assert all(error["loc"][0] == field for error in exc.value.errors())


def test_unrelated_request_identity_does_not_change_statistics(client):
    request = copy.deepcopy(CASES["rare_ties"]["request"])
    request["request_id"] = "unrelated-request-identity"
    response = client.post(f"{ENDPOINT}?response_version=2", json=request)
    assert response.status_code == 200
    assert response.json()["tie_rate"] == 0.07
    assert response.json()["edge_existence_rates"] == {"input->goal": 0.93}
