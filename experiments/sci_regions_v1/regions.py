"""Offline SCI-REGIONS v1 request validation and deterministic classifier.

This module contains no R3-B equations. Evaluators provide natural-unit measurements.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from schema_validation import SchemaError, validate as validate_schema

ROOT = Path(__file__).resolve().parent
REQUEST_SCHEMA = json.loads((ROOT / "schema/RegionRequestV1.schema.json").read_text())
RESULT_SCHEMA = json.loads((ROOT / "schema/RegionResultV1.schema.json").read_text())


class Refusal(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise Refusal("NON_FINITE", "non-finite evaluator output")
    return value


def validate_request(request: dict[str, Any], bindings: dict[str, str]) -> None:
    try:
        canonical_bytes(request)
        validate_schema(request, REQUEST_SCHEMA)
    except (ValueError, TypeError, SchemaError) as exc:
        raise Refusal("INVALID_REQUEST", str(exc)) from exc
    if request["baseline_id"] not in request["option_ids"]:
        raise Refusal("BASELINE_MISSING", "baseline is not a declared option")
    if set(request["option_labels"]) != set(request["option_ids"]):
        raise Refusal("OPTION_LABEL_MISMATCH", "option labels must cover every declared option")
    if not set(request["comparison_set"]).issubset(request["option_ids"]):
        raise Refusal("COMPARISON_OPTION_MISSING", "comparison includes an undeclared option")
    for field in ("axes", "fixed_assumptions", "constraints"):
        ids = [entry["id"] for entry in request[field]]
        if len(ids) != len(set(ids)):
            raise Refusal("DUPLICATE_ID", field)
    if len({a["binding"] for a in request["axes"]}) != 2:
        raise Refusal("DUPLICATE_BINDING", "both axes bind the same parameter")
    for axis in request["axes"]:
        if axis["operation"] != "FIX_SCENARIO_PARAMETER" or axis["role"] not in ("SCENARIO_ASSUMPTION", "UNCERTAIN_PARAMETER"):
            raise Refusal("UNSUPPORTED_SEMANTICS", f"{axis['id']} operation/role is outside v1")
        if bindings.get(axis["binding"]) != axis["unit"]:
            raise Refusal("UNRESOLVED_BINDING", f"{axis['binding']} unit or binding differs")
        values = axis["grid_values"]
        if axis["valid_min"] > axis["valid_max"] or any(not math.isfinite(v) for v in values + [axis["valid_min"], axis["valid_max"]]):
            raise Refusal("INVALID_DOMAIN", axis["id"])
        if values != sorted(values) or any(v < axis["valid_min"] or v > axis["valid_max"] for v in values):
            raise Refusal("INVALID_GRID", axis["id"])
        if axis["joint_support"] not in (None, "x+y<=1"):
            raise Refusal("UNSUPPORTED_SEMANTICS", "joint support binding is unknown")
    for item in request["constraints"]:
        if item["enforcement"] != "DETERMINISTIC_HARD":
            raise Refusal("UNSUPPORTED_SEMANTICS", f"{item['id']} enforcement is outside v1")
    objective = request["objective"]
    if objective and objective["functional"] != "POINT_VALUE":
        raise Refusal("UNSUPPORTED_SEMANTICS", "only deterministic point objective is implemented")
    if len(request["axes"][0]["grid_values"]) * len(request["axes"][1]["grid_values"]) > request["execution"]["max_evaluations"]:
        raise Refusal("RESOURCE_LIMIT", "grid exceeds evaluation cap")


def _compare(value: Fraction | float, op: str, limit: Fraction | float) -> bool:
    return {"<=": value <= limit, ">=": value >= limit, "<": value < limit, ">": value > limit}[op]


def _margin(value: Fraction | float, op: str, limit: Fraction | float) -> Fraction | float:
    return limit - value if op in ("<=", "<") else value - limit


def _pair(a: Fraction | float, b: Fraction | float, direction: str, delta: float) -> dict[str, Any]:
    difference = (a - b) if direction == "MAXIMISE" else (b - a)
    if isinstance(difference, Fraction):
        lo = hi = difference
        exact = difference == 0
    else:
        scale = max(1.0, abs(float(a)), abs(float(b)))
        tolerance = 1e-12 * scale
        lo, hi = difference - tolerance, difference + tolerance
        exact = False
    if exact:
        state = "EXACT_TIE"
    elif lo > delta:
        state = "A_PREFERRED" if isinstance(difference, Fraction) else "A_POINT_PREFERRED"
    elif hi < -delta:
        state = "B_PREFERRED" if isinstance(difference, Fraction) else "B_POINT_PREFERRED"
    elif delta > 0 and lo >= -delta and hi <= delta:
        state = "PRACTICALLY_EQUIVALENT"
    else:
        state = "NUMERICALLY_UNRESOLVED"
    return {"state": state, "difference": json_value(difference), "interval": [json_value(lo), json_value(hi)], "numeric_policy": "EXACT_RATIONAL" if isinstance(difference, Fraction) else "FLOAT64_RECORDED", "interval_kind": "EXACT_VALUE" if isinstance(difference, Fraction) else "HEURISTIC_ROUNDOFF_SCREEN"}


def _question(point: dict[str, Any], request: dict[str, Any]) -> dict[str, str] | None:
    if point["state"] != "EVALUATED":
        return None
    for option_id in sorted(request["option_ids"]):
        row = point["options"][option_id]
        if row["feasibility"] in ("FEASIBILITY_UNRESOLVED", "UNSUPPORTED"):
            return {"kind": "MISSING_INFORMATION", "evidence_id": option_id, "text": f"What is the quantified effect of {request['option_labels'][option_id]}?"}
    violated = sorted(c["id"] for row in point["options"].values() for c in row["constraints"] if c["state"] == "VIOLATED")
    if violated:
        return {"kind": "CONSTRAINT", "evidence_id": violated[0], "text": f"What evidence establishes the limit for {violated[0]}?"}
    if point["pairwise"]:
        key = sorted(point["pairwise"])[0]
        return {"kind": "PREFERENCE", "evidence_id": key, "text": f"Which assumption could change the comparison {key}?"}
    return None


def classify_coordinate(request: dict[str, Any], x: float, y: float, measurements: dict[str, dict[str, Any]] | None, *, status: str = "EVALUATED") -> dict[str, Any]:
    point: dict[str, Any] = {"x": x, "y": y, "state": status, "coverage_kind": "EVALUATED_POINTS_ONLY" if status == "EVALUATED" else "NONE", "options": {}, "comparison_completeness": "NOT_EVALUATED", "named_preference": "NOT_EVALUATED", "overall_preference": "NOT_EVALUATED", "pairwise": {}, "question": None}
    if status != "EVALUATED":
        return point
    if measurements is None:
        raise Refusal("EVALUATION_FAILED", "missing evaluator output")
    constraint_specs = {c["id"]: c for c in request["constraints"]}
    for oid in request["option_ids"]:
        raw = measurements.get(oid, {"status": "UNSUPPORTED", "reason": "OPTION_NOT_RETURNED"})
        if raw.get("status") not in ("COMPUTED", "PARTIAL"):
            point["options"][oid] = {"goal": "NOT_APPLICABLE" if request["goal"] is None else "UNSUPPORTED", "goal_value": None, "constraints": [{"id": c["id"], "state": "UNSUPPORTED", "value": None, "margin": None} for c in request["constraints"]], "feasibility": "UNSUPPORTED", "objective_value": None, "reason": raw.get("reason", "UNKNOWN"), "tier": raw.get("tier")}
            continue
        objective_value = raw.get("objective_value")
        if objective_value is not None and not math.isfinite(float(objective_value)):
            raise Refusal("EVALUATION_FAILED", "non-finite objective")
        goal_spec = request["goal"]
        goal_value = raw.get("goal_value")
        if goal_spec is None:
            goal = "NOT_APPLICABLE"
        elif goal_value is None:
            goal = "UNSUPPORTED"
        else:
            goal = "ATTAINED" if _compare(goal_value, goal_spec["operator"], Fraction(str(goal_spec["limit"])) if isinstance(goal_value, Fraction) else goal_spec["limit"]) else "MISSED"
        cs = []
        for cid, spec in constraint_specs.items():
            value = raw.get("constraints", {}).get(cid)
            if value is None:
                cs.append({"id": cid, "state": "UNSUPPORTED", "value": None, "margin": None})
                continue
            lim = Fraction(str(spec["limit"])) if isinstance(value, Fraction) else spec["limit"]
            cs.append({"id": cid, "state": "SATISFIED" if _compare(value, spec["operator"], lim) else "VIOLATED", "value": json_value(value), "margin": json_value(_margin(value, spec["operator"], lim))})
        states = [c["state"] for c in cs]
        feasibility = "INFEASIBLE" if "VIOLATED" in states else ("FEASIBLE" if all(s == "SATISFIED" for s in states) else "FEASIBILITY_UNRESOLVED")
        point["options"][oid] = {"goal": goal, "goal_value": json_value(goal_value), "constraints": cs, "feasibility": feasibility, "objective_value": json_value(objective_value), "reason": raw.get("reason"), "tier": raw.get("tier")}
    rows = point["options"]
    known_feasible = [oid for oid in request["option_ids"] if rows[oid]["feasibility"] == "FEASIBLE"]
    possible_unknown = [oid for oid in request["option_ids"] if rows[oid]["feasibility"] not in ("FEASIBLE", "INFEASIBLE") or (request["objective"] is not None and rows[oid]["feasibility"] == "FEASIBLE" and rows[oid]["objective_value"] is None)]
    complete = not possible_unknown
    point["comparison_completeness"] = "COMPLETE" if complete else "INCOMPLETE_COMPARISON"
    if all(r["feasibility"] == "INFEASIBLE" for r in rows.values()):
        point["overall_preference"] = "NO_FEASIBLE_OPTION"
    elif request["objective"] is None:
        point["overall_preference"] = "OBJECTIVE_UNSPECIFIED"
    elif not complete:
        point["overall_preference"] = "INCOMPLETE_COMPARISON"
    elif not known_feasible:
        point["overall_preference"] = "NO_FEASIBLE_OPTION"
    comparison = [oid for oid in request["comparison_set"] if rows[oid]["feasibility"] == "FEASIBLE" and rows[oid]["objective_value"] is not None]
    unresolved_named = any(rows[oid]["feasibility"] not in ("FEASIBLE", "INFEASIBLE") or (rows[oid]["feasibility"] == "FEASIBLE" and rows[oid]["objective_value"] is None) for oid in request["comparison_set"])
    if request["objective"] is None:
        named = "OBJECTIVE_UNSPECIFIED"
    elif unresolved_named:
        named = "INCOMPLETE_COMPARISON"
    elif not comparison:
        named = "NO_FEASIBLE_OPTION"
    elif len(comparison) == 1:
        named = "SOLE_FEASIBLE:" + comparison[0]
    else:
        from itertools import combinations
        for a, b in combinations(sorted(comparison), 2):
            point["pairwise"][f"{a}|{b}"] = _pair(measurements[a]["objective_value"], measurements[b]["objective_value"], request["objective"]["direction"], request["objective"]["delta"])
        winners = []
        for oid in comparison:
            beats_all = True
            for other in comparison:
                if other == oid:
                    continue
                a, b = sorted((oid, other))
                pair = point["pairwise"][f"{a}|{b}"]["state"]
                accepted = ("A_PREFERRED", "A_POINT_PREFERRED") if oid == a else ("B_PREFERRED", "B_POINT_PREFERRED")
                if pair not in accepted:
                    beats_all = False
                    break
            if beats_all:
                winners.append(oid)
        if len(winners) == 1:
            named = ("PREFERRED:" if all(p["numeric_policy"] == "EXACT_RATIONAL" for p in point["pairwise"].values()) else "POINT_ESTIMATE_PREFERRED:") + winners[0]
        elif all(p["state"] == "EXACT_TIE" for p in point["pairwise"].values()):
            named = "EXACT_TIE"
        elif all(p["state"] in ("EXACT_TIE", "PRACTICALLY_EQUIVALENT") for p in point["pairwise"].values()) and request["objective"]["delta"] > 0:
            named = "PRACTICALLY_EQUIVALENT"
        else:
            named = "NUMERICALLY_UNRESOLVED"
    point["named_preference"] = named
    if point["overall_preference"] == "NOT_EVALUATED":
        point["overall_preference"] = named if set(request["comparison_set"]) == set(request["option_ids"]) else "INCOMPLETE_COMPARISON"
    point["question"] = _question(point, request)
    return point


def transition_brackets(points: list[dict[str, Any]], request: dict[str, Any]) -> list[dict[str, Any]]:
    by_xy = {(p["x"], p["y"]): p for p in points}
    xs, ys = [a["grid_values"] for a in request["axes"]]
    out = []
    for axis, grid, other in ((0, xs, ys), (1, ys, xs)):
        for fixed in other:
            for lo, hi in zip(grid, grid[1:]):
                a = by_xy.get((lo, fixed) if axis == 0 else (fixed, lo))
                b = by_xy.get((hi, fixed) if axis == 0 else (fixed, hi))
                if not a or not b or a["state"] != "EVALUATED" or b["state"] != "EVALUATED":
                    continue
                kinds = []
                if any(a["options"][oid]["feasibility"] != b["options"][oid]["feasibility"] for oid in request["option_ids"]):
                    kinds.append("FEASIBILITY")
                if a["named_preference"] != b["named_preference"]:
                    kinds.append("PREFERENCE")
                if any(a["options"][oid]["goal"] != b["options"][oid]["goal"] for oid in request["option_ids"]):
                    kinds.append("GOAL")
                if kinds:
                    out.append({"axis_id": request["axes"][axis]["id"], "fixed_axis_value": fixed, "low": lo, "high": hi, "width": hi - lo, "kinds": kinds, "coverage_kind": "TRANSITION_BRACKET"})
    return out


def validate_result(result: dict[str, Any]) -> None:
    try:
        canonical_bytes(result)
        validate_request(result["request"], {a["binding"]: a["unit"] for a in result["request"]["axes"]})
        validate_schema(result, RESULT_SCHEMA)
    except (ValueError, TypeError, SchemaError) as exc:
        raise Refusal("INVALID_RESULT", str(exc)) from exc
