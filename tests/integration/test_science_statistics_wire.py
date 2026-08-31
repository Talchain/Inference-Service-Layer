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
from src.models.response_v2 import InferenceWarning, ISLResponseV2
from src.utils.response_builder import ResponseBuilder

FIXTURE = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/science_transport/base_enhanced_responses.json"
    ).read_text()
)
CASES = FIXTURE["cases"]
ENDPOINT = "/api/v1/robustness/analyze/v2"
TARGET_FIELDS = {"tie_rate", "edge_existence_rates"}
MIXED_EDGE_REQUEST = json.loads(
    (Path(__file__).parents[1] / "fixtures/science_transport/mixed_edge_request.json").read_text()
)
MIXED_EDGE_BASE = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/science_transport/mixed_edge_base_response.json"
    ).read_text()
)


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


@pytest.mark.parametrize(
    "mixed,probability",
    [(True, 1.0), (True, 0.1), (False, 1.0), (False, 0.1)],
    ids=["mixed_out_of_range", "mixed_in_range", "directed_valid", "directed_low_probability"],
)
def test_invalid_optional_map_does_not_destroy_completed_computation(client, mixed, probability):
    # Exact independent-review counterexample and its genuine directed counterpart.
    request = copy.deepcopy(MIXED_EDGE_REQUEST)
    if not mixed:
        request["graph"]["edges"] = request["graph"]["edges"][:1]
    for edge in request["graph"]["edges"]:
        edge["exists_probability"] = probability
    legacy = client.post(f"{ENDPOINT}?response_version=1", json=request)
    enhanced = client.post(f"{ENDPOINT}?response_version=2", json=request)
    assert legacy.status_code == enhanced.status_code == 200
    metadata = legacy.json()["_metadata"]
    if probability == 1.0:
        assert metadata["edge_existence_rates"] == {"input->goal": 2.0 if mixed else 1.0}
    elif mixed:
        # Range validity alone cannot legitimise aggregated, ambiguous edge keys.
        assert 0.0 < metadata["edge_existence_rates"]["input->goal"] < 1.0
    else:
        assert metadata["edge_existence_rates"] == {"input->goal": 0.09}
    body = enhanced.json()
    assert body["analysis_status"] == "computed"
    assert body["tie_rate"] == metadata["tie_rate"]
    if probability == 1.0:
        assert body["tie_rate"] == 0.0
    warnings = [
        warning
        for warning in body["inference_warnings"]
        if warning["code"] == "ISL_SAMPLING_DIAGNOSTICS_INVALID"
    ]
    if mixed:
        assert "edge_existence_rates" not in body  # no clamp or partial map salvage
        assert len(warnings) == 1
        assert warnings[0]["field"] == "edge_existence_rates"
        assert warnings[0]["severity"] == "warning"
        assert warnings[0]["detail"]["action"] == "omitted"
        existing_fields = {
            key: value
            for key, value in without_volatile_fields(body).items()
            if key not in TARGET_FIELDS
        }
        existing_fields["inference_warnings"] = [
            warning
            for warning in existing_fields["inference_warnings"]
            if warning["code"] != "ISL_SAMPLING_DIAGNOSTICS_INVALID"
        ]
        if probability == 1.0:
            assert existing_fields == without_volatile_fields(MIXED_EDGE_BASE["response"])
    else:
        assert body["edge_existence_rates"] == metadata["edge_existence_rates"]
        assert warnings == []


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("nan"), float("inf"), True, "0.4"])
@pytest.mark.parametrize("field", ["tie_rate", "edge_existence_rates"])
def test_builder_withholds_only_invalid_diagnostic_and_preserves_source_warnings(field, invalid):
    parsed = ISLResponseV2.model_validate(CASES["rare_ties"]["enhanced_response"])
    builder = ResponseBuilder(request_id=parsed.request_id, request_echo=parsed.request_echo)
    source_warnings = [
        InferenceWarning(
            code="ROOT_NODE_DEFAULT_VALUE", field="nodes[other]", detail={"node_id": "other"}
        )
    ]
    # Match the API ordering: source warnings before result adoption.
    builder.set_inference_warnings(source_warnings)
    builder.set_decision_evpi(parsed.decision_evpi)
    builder.set_results(
        options=parsed.options,
        robustness=parsed.robustness,
        tie_rate=invalid if field == "tie_rate" else 0.0,
        edge_existence_rates=(
            {"valid->goal": 0.4, "invalid->goal": invalid}
            if field == "edge_existence_rates"
            else {"input->goal": 0.6}
        ),
    )
    body = builder.build().model_dump(by_alias=True, exclude_none=True)
    assert body["analysis_status"] == "computed"
    assert field not in body
    if field == "tie_rate":
        assert body["edge_existence_rates"] == {"input->goal": 0.6}
    else:
        assert body["tie_rate"] == 0.0
    assert len(source_warnings) == 1  # disclosure never mutates the producer list
    assert body["inference_warnings"][0] == source_warnings[0].model_dump()
    assert body["inference_warnings"][1]["code"] == "ISL_SAMPLING_DIAGNOSTICS_INVALID"
    assert body["inference_warnings"][1]["field"] == field
    assert body["inference_warnings"][1]["severity"] == "warning"
    assert len(body["inference_warnings"]) == 2


def test_discarded_organisational_edges_do_not_withhold_real_sampled_map(client):
    request = copy.deepcopy(MIXED_EDGE_REQUEST)
    request["graph"]["edges"] = request["graph"]["edges"][:1]
    request["graph"]["nodes"].append({"id": "organising", "kind": "decision", "label": "Group"})
    for edge_type in ("directed", "bidirected"):
        request["graph"]["edges"].append(
            {
                "from": "organising",
                "to": "goal",
                "edge_type": edge_type,
                "exists_probability": 1.0,
                "strength": {"mean": 1.0, "std": 0.1},
            }
        )
    response = client.post(f"{ENDPOINT}?response_version=2", json=request)
    assert response.status_code == 200
    body = response.json()
    assert body["tie_rate"] == 0.0
    assert body["edge_existence_rates"] == {"input->goal": 1.0}
    assert not any(
        warning["code"] == "ISL_SAMPLING_DIAGNOSTICS_INVALID"
        for warning in body["inference_warnings"]
    )


def test_both_invalid_diagnostics_share_one_visible_disclosure():
    parsed = ISLResponseV2.model_validate(CASES["rare_ties"]["enhanced_response"])
    builder = ResponseBuilder(request_id=parsed.request_id, request_echo=parsed.request_echo)
    builder.set_decision_evpi(parsed.decision_evpi)
    builder.set_results(
        options=parsed.options,
        robustness=parsed.robustness,
        tie_rate=float("nan"),
        edge_existence_rates={"input->goal": 0.4},
        ambiguous_sampling_edge_keys=True,
    )
    body = builder.build().model_dump(by_alias=True, exclude_none=True)
    assert TARGET_FIELDS.isdisjoint(body)
    assert len(body["inference_warnings"]) == 1
    warning = body["inference_warnings"][0]
    assert warning["code"] == "ISL_SAMPLING_DIAGNOSTICS_INVALID"
    assert set(warning["detail"]["invalid_fields"]) == TARGET_FIELDS
    assert all(field in warning["detail"]["message"] for field in TARGET_FIELDS)
    json.dumps(body, allow_nan=False)  # invalid raw data cannot poison the warning either
