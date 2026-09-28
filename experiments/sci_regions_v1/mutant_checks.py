"""Known-bad controls: each named mutation must be caught by an unchanged control."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from oracle import truth
from r3b_adapter import PricingAdapter
from regions import Refusal, _pair, classify_coordinate, sha256, transition_brackets, validate_request
from synthetic import measurements, request


def _caught(fn) -> bool:
    try:
        fn()
    except (Refusal, AssertionError):
        return True
    return False


def run() -> dict:
    out = {}
    q = request("F3", 11)
    validate_request(q, {"x": "unit", "y": "unit"})
    control = classify_coordinate(q, 1, 1, measurements("F3", 1, 1))
    assert control["overall_preference"] == truth("F3", 1, 1)["named_preference"] == "NO_FEASIBLE_OPTION"

    bad = copy.deepcopy(q)
    bad["axes"][0]["unit"] = "GBP"
    out["wrong_unit"] = _caught(lambda: validate_request(bad, {"x": "unit", "y": "unit"}))

    bad = copy.deepcopy(q)
    bad["axes"][0]["operation"] = "APPLY_DECLARED_INTERVENTION"
    out["parameter_to_intervention"] = _caught(lambda: validate_request(bad, {"x": "unit", "y": "unit"}))

    bad_values = measurements("F3", 1, 1)
    bad_values["A"]["constraints"]["option_limit"] = 0
    bad_values["B"]["constraints"]["option_limit"] = 0
    out["ignored_constraint"] = classify_coordinate(q, 1, 1, bad_values)["overall_preference"] != control["overall_preference"]

    q4 = request("F4", 11)
    good4 = classify_coordinate(q4, 1, 1, measurements("F4", 1, 1))
    out["dropped_unsupported_option"] = good4["overall_preference"] == "INCOMPLETE_COMPARISON" and good4["overall_preference"] != "NO_FEASIBLE_OPTION"

    # The roundoff screen straddles the declared practical margin: overlap is not equivalence.
    out["overlap_as_tie"] = _pair(1.0, 1.1, "MAXIMISE", .1)["state"] == "NUMERICALLY_UNRESOLVED"

    q9 = request("F9", 11)
    out["filled_forbidden_domain"] = truth("F9", .8, .8)["state"] == "OUTSIDE_DECLARED_DOMAIN" and measurements("F9", .8, .8) is None

    q1 = request("F1", 11)
    pts = [classify_coordinate(q1, x, y, measurements("F1", x, y)) for x in q1["axes"][0]["grid_values"] for y in q1["axes"][1]["grid_values"]]
    brackets = transition_brackets(pts, q1)
    out["interpolated_exact_boundary"] = bool(brackets) and all(b["coverage_kind"] == "TRANSITION_BRACKET" and b["width"] > 0 for b in brackets)

    q13a = request("F13", 11, fixed_c=0)
    q13b = request("F13", 11, fixed_c=.2)
    out["stale_fixed_assumption_cache"] = sha256(q13a) != sha256(q13b) and q13a["source"]["graph_sha256"] == q13b["source"]["graph_sha256"]
    q13c = copy.deepcopy(q13a)
    q13c["objective"]["delta"] = .1
    q13d = copy.deepcopy(q13a)
    q13d["constraints"].append({"id": "new-limit", "metric": "utility", "unit": "unit", "operator": ">=", "limit": 0, "temporal_rule": "POINT", "enforcement": "DETERMINISTIC_HARD"})
    out["stale_criterion_cache"] = len({sha256(q13a), sha256(q13c), sha256(q13d)}) == 3

    adapter = PricingAdapter("X_net_reading")
    qr = adapter.request(11)
    adapter.validate_request_identity(qr)
    corrupt = copy.deepcopy(qr)
    corrupt["source"]["evaluator_sha256"] = "0" * 64
    out["stale_evaluator_identity"] = _caught(lambda: adapter.validate_request_identity(corrupt))

    coords = [(0, 0), (.25, .75), (.8, .1)]
    first = {(x, y): classify_coordinate(q1, x, y, measurements("F1", x, y))["named_preference"] for x, y in coords}
    second = {(x, y): classify_coordinate(q1, x, y, measurements("F1", x, y))["named_preference"] for x, y in reversed(coords)}
    out["execution_order_reproducibility"] = first == second

    if not all(out.values()):
        raise AssertionError(f"known-bad controls missed: {[k for k,v in out.items() if not v]}")
    return {"schema": "SCI-REGIONS-mutant-evidence-v1", "status": "PASS", "controls": out, "not_run": ["random-stream reuse (no Monte Carlo in this slice)", "learned safe box (no learned summary in this slice)"]}


if __name__ == "__main__":
    result = run()
    path = Path(__file__).resolve().parent / "output/mutant-evidence.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, indent=2))
