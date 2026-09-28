"""Candidate evaluator for the frozen, explicitly synthetic reference cases."""
from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path
import hashlib
import json

from regions import ROOT, Refusal

FROZEN = json.loads((ROOT / "fixtures/FROZEN.json").read_text())
COMMIT = "fdb51e84a673993de7cddb85d5b963741941b80a"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid(case: str, n: int) -> tuple[list[float], list[float]]:
    if n < 2:
        raise ValueError("n must be >=2")
    spec = FROZEN["cases"][case]
    def values(name: str) -> list[float]:
        lo, hi = (F(t) for t in spec[name])
        return [float(lo + (hi - lo) * F(i, n - 1)) for i in range(n)]
    return values("x"), values("y")


def request(case: str, n: int, *, fixed_c: float = 0.0, objective: bool = True) -> dict:
    if case not in FROZEN["cases"]:
        raise Refusal("UNKNOWN_FIXTURE", case)
    xs, ys = grid(case, n)
    spec = FROZEN["cases"][case]
    fixture_file = ROOT / "fixtures/FROZEN.json"
    constraints = [{"id": "option_limit", "metric": "option_exposure", "unit": "unit", "operator": "<=", "limit": .4, "temporal_rule": "POINT", "enforcement": "DETERMINISTIC_HARD"}] if case in ("F3", "F4") else []
    return {
        "schema_version": "RegionRequestV1", "request_id": f"synthetic-{case}-{n}-{fixed_c}-{objective}",
        "source": {"repository": "Talchain/Inference-Service-Layer", "commit": COMMIT, "graph_sha256": _hash(fixture_file), "mapping_sha256": _hash(fixture_file), "tier": "SYNTHETIC", "model_id": case, "evaluator_sha256": _hash(Path(__file__)), "adapter_version": "synthetic-v1"},
        "case": case, "option_ids": ["A", "B"], "option_labels": {"A": "Synthetic option A", "B": "Synthetic option B"}, "baseline_id": "B", "comparison_set": ["A", "B"],
        "axes": [
            {"id": "x", "label": "Synthetic x", "unit": "unit", "role": "SCENARIO_ASSUMPTION", "operation": "FIX_SCENARIO_PARAMETER", "binding": "x", "grid_values": xs, "domain_source": "synthetic frozen fixture", "valid_min": xs[0], "valid_max": xs[-1], "joint_support": "x+y<=1" if case == "F9" else None},
            {"id": "y", "label": "Synthetic y", "unit": "unit", "role": "SCENARIO_ASSUMPTION", "operation": "FIX_SCENARIO_PARAMETER", "binding": "y", "grid_values": ys, "domain_source": "synthetic frozen fixture", "valid_min": ys[0], "valid_max": ys[-1], "joint_support": "x+y<=1" if case == "F9" else None}
        ],
        "fixed_assumptions": [{"id": "c", "value": fixed_c, "source": "synthetic frozen fixture"}] if case == "F13" else [],
        "goal": {"id": "attain", "metric": "outcome", "unit": "unit", "operator": ">=", "limit": .6, "temporal_rule": "POINT"} if case == "F3" else None,
        "objective": {"metric": "utility", "unit": "unit", "direction": "MAXIMISE", "functional": "POINT_VALUE", "temporal_rule": "POINT", "delta": .1 if case == "F2" else 0, "delta_source": "synthetic fixture"} if objective else None,
        "constraints": constraints,
        "execution": {"seed": 20260928, "numeric_policy": "EXACT_RATIONAL", "max_seconds": 600, "max_memory_mib": 1024, "max_evaluations": 1000000, "draw_count": 1, "confidence_scope": "NOT_APPLICABLE"},
        "comparison": {"baseline_status": "PINNED_LOCAL", "reference_id": f"{case}-analytic", "differences": []}
    }


def validate_identity(candidate: dict) -> None:
    case = candidate["case"]
    expected = request(case, len(candidate["axes"][0]["grid_values"]), fixed_c=next((a["value"] for a in candidate["fixed_assumptions"] if a["id"] == "c"), 0.0), objective=candidate["objective"] is not None)
    if candidate["source"] != expected["source"] or candidate["option_ids"] != expected["option_ids"]:
        raise Refusal("MODEL_IDENTITY_MISMATCH", "synthetic source or option set differs")


def measurements(case: str, x: float, y: float, *, fixed_c: float = 0.0) -> dict | None:
    xq, yq = F(str(x)), F(str(y))
    if case == "F9" and xq + yq > 1:
        return None
    if case in ("F1", "F2", "F3", "F4", "F9"):
        difference = xq - yq
    elif case == "F5":
        difference = xq * yq
    elif case == "F6":
        difference = xq * xq - F(1, 4)
    elif case == "F8":
        difference = F(2)
    elif case == "F11":
        difference = F(1, 40000) - (xq - F(1, 80)) ** 2 - (yq - F(1, 80)) ** 2
    elif case == "F13":
        difference = xq - yq + F(str(fixed_c))
    else:
        raise Refusal("UNKNOWN_FIXTURE", case)
    a = {"status": "COMPUTED", "objective_value": difference, "goal_value": xq if case == "F3" else None, "constraints": {}, "tier": "SYNTHETIC", "reason": None}
    b = {"status": "COMPUTED", "objective_value": F(0), "goal_value": yq if case == "F3" else None, "constraints": {}, "tier": "SYNTHETIC", "reason": None}
    if case == "F3":
        a["constraints"] = {"option_limit": yq}
        b["constraints"] = {"option_limit": xq}
    elif case == "F4":
        a["constraints"] = {"option_limit": yq}
        b["status"], b["reason"] = "UNSUPPORTED", "B_FEASIBILITY_UNAVAILABLE"
    return {"A": a, "B": b}
