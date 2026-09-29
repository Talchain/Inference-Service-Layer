from __future__ import annotations

import copy
import json

import pytest

from lab_handoff import EXPECTED_IDENTITY, OUTPUT, SCHEMA, cases
from schema_validation import SchemaError, validate


def test_two_frozen_lab_cases_match_checked_evaluator_and_saved_json():
    actual = cases()
    assert actual == json.loads(OUTPUT.read_text())
    assert len(actual) == 2
    computed, unavailable = actual
    assert computed["flip_thresholds_status"] == "computed"
    assert computed["subject"]["kind"] == "EVIDENCE_VARIABLE"
    assert computed["subject"]["option_id"] == "59_with_feature_release"
    assert computed["current_value"] == 0.5
    assert computed["flip_threshold"] == 1.5
    assert computed["crossing_rule"] == "GREATER_THAN"
    assert computed["flip_kind"] == "HARD_CONSTRAINT_FEASIBILITY"
    assert unavailable["flip_thresholds_status"] == "unavailable"
    assert unavailable["subject"]["kind"] == "OPTION_CONTROLLED_LEVERS"
    assert unavailable["subject"]["option_id"] == "40bb45e7"
    assert unavailable["current_value"] is None and unavailable["flip_threshold"] is None
    assert unavailable["reason"]["code"] == "OPTION_LEVELS_MISSING"
    assert computed["provenance"] == unavailable["provenance"]
    assert all(computed["provenance"][key] == digest for key, digest in EXPECTED_IDENTITY.items())
    assert "not evidence-backed plausibility" in computed["provenance"]["assumption_qualifier"]


def test_schema_refuses_false_numeric_threshold_for_unavailable_control():
    schema = json.loads(SCHEMA.read_text())
    data = cases()
    injected = copy.deepcopy(data)
    injected[1]["flip_threshold"] = 0
    with pytest.raises(SchemaError):
        validate(injected, schema)
    injected = copy.deepcopy(data)
    injected[0]["flip_thresholds_status"] = "unavailable"
    with pytest.raises(SchemaError):
        validate(injected, schema)
    with pytest.raises(SchemaError):
        validate([data[0], copy.deepcopy(data[0])], schema)
    with pytest.raises(SchemaError):
        validate([data[1], copy.deepcopy(data[1])], schema)
