"""Two pinned, read-only threshold cases for the existing AI Experience Lab.

This is a research-to-consumer handoff, not a live analysis endpoint or renderer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from r3b_adapter import GRAPH_ID, PINNED_COMMIT, PricingAdapter, reproduce_stored_thresholds
from schema_validation import validate

ROOT = Path(__file__).resolve().parent
SCHEMA = ROOT / "lab" / "FlipThresholdCaseV1.schema.json"
OUTPUT = ROOT / "lab" / "cases.json"
OPTION = "59_with_feature_release"
UNAVAILABLE_OPTION = "40bb45e7"
MISSING_FACTORS = ("fac_new_pro_customer_price", "fac_grandfathering_existing_customers")
EXPECTED_IDENTITY = {
    "graph_sha256": "6b9a09e5f5da590ab42db6bd0a52d43669cbc5e8c9fe361e744a73e2cae31c5c",
    "mapping_sha256": "235ef1db77ae8cdb816a203d49f06501c1f490d525c6219799c0a0211c96fc21",
    "evaluator_sha256": "7082e94860d80ef4409aa7e4db7ebe9b9dd7ae3660d92db3faef200ff800cd13",
}


def cases() -> list[dict]:
    adapter = PricingAdapter("X_net_reading")
    request = adapter.request(11)
    for key, expected in EXPECTED_IDENTITY.items():
        if adapter.identity[key] != expected:
            raise ValueError(f"pinned {key} changed")
    axis = request["axes"][0]
    if not (axis["binding"] == "X_price_churn_direct" and axis["valid_min"] == 0
            and axis["valid_max"] == 4.25 and axis["joint_support"] is None):
        raise ValueError("frozen exploratory churn axis changed")
    constraint = request["constraints"][0]
    if not (constraint["metric"] == "monthly_churn" and constraint["operator"] == "<="
            and constraint["limit"] == 4 and constraint["enforcement"] == "DETERMINISTIC_HARD"):
        raise ValueError("frozen hard-constraint binding changed")
    constraint_id = constraint["id"]

    def churn(response: float) -> float:
        value = adapter.measurements(response, 0)[OPTION]["constraints"][constraint_id]
        if not math.isfinite(value):
            raise ValueError("non-finite pinned churn measurement")
        return value

    zero, one = churn(0), churn(1)
    slope = one - zero
    if slope <= 0:
        raise ValueError("pinned churn response no longer increases")
    threshold = (constraint["limit"] - zero) / slope
    stored = reproduce_stored_thresholds()["models"]["X_net_reading"]["thresholds"]["churn_limit"]
    if not math.isclose(threshold, stored, rel_tol=0, abs_tol=1e-9):
        raise ValueError("computed churn boundary differs from frozen break-even result")
    if not (churn(0.5) == 3 and churn(threshold) == constraint["limit"]
            and churn(threshold + 1e-6) > constraint["limit"]):
        raise ValueError("inclusive hard-constraint boundary failed evaluator check")

    missing = adapter.graph.nodes[UNAVAILABLE_OPTION]
    if missing.get("interventions") != {} or missing["kind"] != "option":
        raise ValueError("unavailable option now has quantified interventions")
    for factor in MISSING_FACTORS:
        if adapter.graph.nodes[factor].get("category") != "controllable":
            raise ValueError(f"{factor} is no longer an option-controlled factor")
        adapter.graph.edge(f"{UNAVAILABLE_OPTION}->{factor}")
    withheld = adapter.measurements(0.5, 0)[UNAVAILABLE_OPTION]
    if withheld["status"] != "PARTIAL" or "OPTION_LEVELS_MISSING" not in (withheld["reason"] or ""):
        raise ValueError("unavailable control is no longer withheld for missing option levels")

    identity = adapter.identity
    provenance = {
        "source_commit": PINNED_COMMIT,
        "graph_id": GRAPH_ID,
        "graph_sha256": identity["graph_sha256"],
        "mapping_sha256": identity["mapping_sha256"],
        "model_id": adapter.model_id,
        "tier": "X",
        "evaluator_sha256": identity["evaluator_sha256"],
        "source_kind": "PINNED_R3B_RESEARCH",
        "assumption_qualifier": axis["domain_source"] + "; no joint probability interpretation",
    }
    common_fixed = request["fixed_assumptions"] + [
        {"id": "X_feature_competitive_mrr", "value": 0, "source": "frozen one-axis slice"}
    ]
    computed = {
        "case_id": "r3b-A-180910Z-net-churn-limit",
        "flip_thresholds_status": "computed",
        "subject": {
            "id": "X_price_churn_direct",
            "kind": "EVIDENCE_VARIABLE",
            "label": "Direct churn response to a £10 Pro price rise",
            "option_id": OPTION,
            "related_factor_ids": ["monthly_churn"],
        },
        "current_value": 0.5,
        "current_value_source": "PINNED_RESEARCH_REFERENCE",
        "current_unit": "percentage points per +£10",
        "flip_threshold": threshold,
        "threshold_unit": "percentage points per +£10",
        "crossing_rule": "GREATER_THAN",
        "flip_kind": "HARD_CONSTRAINT_FEASIBILITY",
        "flip_meaning": "For £59 with feature release, monthly churn meets the 4% hard limit at the boundary and breaches it above the boundary. Goal attainment and month-12 MRR preference are separate; all six options cannot be compared.",
        "fixed_assumptions": sorted(common_fixed, key=lambda row: row["id"]),
        "provenance": provenance,
        "reason": None,
    }
    unavailable = {
        "case_id": "r3b-A-180910Z-new-pro-grandfathering-unavailable",
        "flip_thresholds_status": "unavailable",
        "subject": {
            "id": UNAVAILABLE_OPTION,
            "kind": "OPTION_CONTROLLED_LEVERS",
            "label": missing["label"],
            "option_id": UNAVAILABLE_OPTION,
            "related_factor_ids": list(MISSING_FACTORS),
        },
        "current_value": None,
        "current_value_source": "NOT_QUANTIFIED",
        "current_unit": None,
        "flip_threshold": None,
        "threshold_unit": None,
        "crossing_rule": None,
        "flip_kind": None,
        "flip_meaning": None,
        "fixed_assumptions": sorted(common_fixed + [
            {"id": "X_price_churn_direct", "value": 0.5, "source": "pinned research reference"}
        ], key=lambda row: row["id"]),
        "provenance": provenance,
        "reason": {
            "code": "OPTION_LEVELS_MISSING",
            "detail": "The option names new-customer pricing and grandfathering, but neither controlled factor has a quantified intervention. A missing effect is not zero effect.",
        },
    }
    result = [computed, unavailable]
    schema = json.loads(SCHEMA.read_text())
    validate(result, schema)
    return result


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(cases(), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(OUTPUT)
