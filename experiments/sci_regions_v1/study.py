"""Run the bounded, deterministic SCI-REGIONS first checkpoint.

Usage: python3 study.py --output /path/to/research-output
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path

from oracle import truth
from r3b_adapter import PricingAdapter, reproduce_stored_thresholds
from regions import Refusal, classify_coordinate, sha256, transition_brackets, validate_request, validate_result
from synthetic import FROZEN, measurements as synthetic_measurements, request as synthetic_request, validate_identity as validate_synthetic_identity
from visual import render_html
import numpy as np

START = time.monotonic()
MAX_SECONDS = 600
MAX_R3B = 4000
MAX_SYNTHETIC = 1000000
counts = {"r3b": 0, "synthetic": 0}


def rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024


def within_budget(kind: str) -> bool:
    return (time.monotonic() - START < MAX_SECONDS and rss_mib() <= 1024 and counts[kind] < (MAX_R3B if kind == "r3b" else MAX_SYNTHETIC))


def _save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def _result(request: dict, points: list[dict], *, kind: str, t0: float, source_verified: bool, metrics: dict, reference_notes: list[str], errors: list[dict] | None = None, unrun: list[str] | None = None) -> dict:
    completed = sum(p["state"] == "EVALUATED" for p in points)
    processed = sum(p["state"] != "NOT_EVALUATED" for p in points)
    expected = len(request["axes"][0]["grid_values"]) * len(request["axes"][1]["grid_values"])
    result = {
        "schema_version": "RegionResultV1", "request": request, "request_sha256": sha256(request),
        "manifest": {"actual_evaluations": completed, "budget_exhausted": processed < expected, "source_verified": source_verified, "execution_status": "PARTIAL" if processed < expected else "COMPLETE", "python": platform.python_version(), "numpy": np.__version__},
        "points": points, "transition_brackets": transition_brackets(points, request), "metrics": metrics,
        "reference": {"status": request["comparison"]["baseline_status"], "baseline_identity": request["comparison"]["reference_id"], "matched_assumptions": True, "notes": reference_notes},
        "errors": errors or [], "unrun_checks": unrun or []
    }
    validate_result(result)
    return result


def _empty_metrics() -> dict:
    return {"confusion": {}, "false_feasible": 0, "false_confident_preference": 0, "abstentions": 0, "incomplete_whole_comparison": 0, "topology_failures": [], "boundary_error": None}


def run_synthetic(case: str, n: int, out: Path) -> dict:
    q = synthetic_request(case, n)
    validate_request(q, {"x": "unit", "y": "unit"})
    validate_synthetic_identity(q)
    t0 = time.monotonic()
    points = []
    confusion: Counter[str] = Counter()
    metrics = _empty_metrics()
    for y in q["axes"][1]["grid_values"]:
        for x in q["axes"][0]["grid_values"]:
            if not within_budget("synthetic"):
                points.append(classify_coordinate(q, x, y, None, status="NOT_EVALUATED"))
                continue
            raw = synthetic_measurements(case, x, y)
            point = classify_coordinate(q, x, y, raw, status="OUTSIDE_DECLARED_DOMAIN" if raw is None else "EVALUATED")
            reference = truth(case, x, y)
            actual = (point["state"], point["named_preference"] if raw else None)
            expected = (reference["state"], reference.get("named_preference"))
            confusion[f"{expected}->{actual}"] += 1
            if raw is not None:
                counts["synthetic"] += 1
                for oid in q["option_ids"]:
                    got = point["options"][oid]
                    wanted = reference["feasibility"][oid]
                    if got["feasibility"] == "FEASIBLE" and wanted != "FEASIBLE":
                        metrics["false_feasible"] += 1
                    if got["feasibility"] != wanted or got["goal"] != reference["goal"][oid]:
                        raise AssertionError(f"{case} {n} {x},{y} {oid}: {got} != {reference}")
                if point["named_preference"] != reference["named_preference"]:
                    if point["named_preference"].startswith("PREFERRED") and not reference["named_preference"].startswith("PREFERRED"):
                        metrics["false_confident_preference"] += 1
                    raise AssertionError(f"{case} {n} {x},{y}: {actual} != {expected}")
                if point["named_preference"] in ("NUMERICALLY_UNRESOLVED", "INCOMPLETE_COMPARISON"):
                    metrics["abstentions"] += 1
            elif point["state"] != reference["state"]:
                raise AssertionError(f"{case} domain {x},{y}")
            points.append(point)
    metrics["confusion"] = dict(sorted(confusion.items()))
    metrics["incomplete_whole_comparison"] = sum(p["overall_preference"] == "INCOMPLETE_COMPARISON" for p in points)
    result = _result(q, points, kind="synthetic", t0=t0, source_verified=True, metrics=metrics, reference_notes=["Independent exact rational oracle; fixture provenance is synthetic", "Coverage is evaluated coordinates only"], unrun=["Monte Carlo fixture F7", "Objective mismatch F10", "Chance constraint F12", "3-D F14"])
    if case == "F11" and n == 41:
        if any(p["named_preference"] == "PREFERRED:A" for p in points):
            raise AssertionError("F11 coarse grid unexpectedly found island")
        result["metrics"]["topology_failures"].append("A island absent on 41x41; this does not certify B between points")
    if case == "F11" and n == 201 and not any(p["named_preference"] == "PREFERRED:A" for p in points):
        raise AssertionError("F11 fine reference missed island")
    if case in ("F1", "F9", "F13") and result["transition_brackets"]:
        mids = [abs((b["low"] + b["high"]) / 2 - b["fixed_axis_value"]) for b in result["transition_brackets"] if "PREFERENCE" in b["kinds"]]
        result["metrics"]["boundary_error"] = max(mids) if mids else None
    validate_result(result)
    _save(out / f"synthetic-{case}-{n}.json", result)
    return {"case": case, "grid": n, "points": len(points), "evaluated": result["manifest"]["actual_evaluations"], "false_feasible": metrics["false_feasible"], "false_confident_preference": metrics["false_confident_preference"], "topology_failures": metrics["topology_failures"], "elapsed_seconds": round(time.monotonic() - t0, 4)}


def run_r3b(adapter: PricingAdapter, n: int, out: Path) -> dict:
    q = adapter.request(n)
    validate_request(q, adapter.binding_units)
    adapter.validate_request_identity(q)
    t0 = time.monotonic()
    points = []
    errors = []
    for y in q["axes"][1]["grid_values"]:
        for x in q["axes"][0]["grid_values"]:
            if not within_budget("r3b"):
                points.append(classify_coordinate(q, x, y, None, status="NOT_EVALUATED"))
                continue
            try:
                point = classify_coordinate(q, x, y, adapter.measurements(x, y))
            except (Refusal, ValueError, KeyError, FloatingPointError) as exc:
                point = classify_coordinate(q, x, y, None, status="EVALUATION_FAILED")
                errors.append({"code": "EVALUATION_FAILED", "detail": f"{x},{y}: {type(exc).__name__}: {exc}"})
            counts["r3b"] += 1
            points.append(point)
    # The reference point is deliberately evaluated even though it is not on either grid.
    if within_budget("r3b"):
        points.append(classify_coordinate(q, .5, 0, adapter.measurements(.5, 0)))
        counts["r3b"] += 1
    metrics = _empty_metrics()
    metrics["abstentions"] = sum(p["named_preference"] in ("INCOMPLETE_COMPARISON", "NUMERICALLY_UNRESOLVED") for p in points)
    metrics["incomplete_whole_comparison"] = sum(p["overall_preference"] == "INCOMPLETE_COMPARISON" for p in points)
    metrics["false_feasible"] = None
    metrics["false_confident_preference"] = None
    result = _result(q, points, kind="r3b", t0=t0, source_verified=True, metrics=metrics, reference_notes=["Pinned local R3-B model, tier X; no deployed-current comparator", "Whole-decision comparisons include six options; named comparison set is explicit", "Axes are exploratory and area has no probability meaning", "False-classification rates cannot be measured without a real-case truth oracle"], errors=errors, unrun=["Monte Carlo interval/coverage study", "Human comprehension study", "Current deployed-product comparison"])
    _save(out / f"r3b-{adapter.model_id}-{n}.json", result)
    return {"model": adapter.model_id, "grid": n, "evaluated": result["manifest"]["actual_evaluations"], "errors": len(errors), "elapsed_seconds": round(time.monotonic() - t0, 4), "path": f"r3b-{adapter.model_id}-{n}.json"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "SCI-REGIONS-run-manifest-v1", "source_commit": "fdb51e84a673993de7cddb85d5b963741941b80a", "synthetic": [], "r3b": [], "baseline": None, "status": "RUNNING", "unrun": []}
    for case in FROZEN["cases"]:
        for n in (11, 41):
            summary["synthetic"].append(run_synthetic(case, n, out))
            _save(out / "run-manifest.json", summary)
    summary["synthetic"].append(run_synthetic("F11", 201, out))
    _save(out / "run-manifest.json", summary)
    summary["baseline"] = reproduce_stored_thresholds()
    adapters = [PricingAdapter(model) for model in ("X_net_reading", "X_gross_reading")]
    for adapter in adapters:
        adapter.assert_composition(.5, 0)
        adapter.assert_composition(0, -31250)
    for adapter in adapters:
        summary["r3b"].append(run_r3b(adapter, 11, out))
        _save(out / "run-manifest.json", summary)
    pilot_seconds = sum(row["elapsed_seconds"] for row in summary["r3b"])
    forecast = pilot_seconds * (41 * 41 / (11 * 11)) * 1.5
    if forecast < MAX_SECONDS - (time.monotonic() - START) and counts["r3b"] + 2 * (41 * 41 + 1) <= MAX_R3B:
        for adapter in adapters:
            summary["r3b"].append(run_r3b(adapter, 41, out))
            _save(out / "run-manifest.json", summary)
    else:
        summary["unrun"].append("41x41 R3-B grids: conservative projection exceeded resource cap")
    preferred_n = 41 if sum(r["grid"] == 41 for r in summary["r3b"]) == 2 else 11
    data = [json.loads((out / f"r3b-{model}-{preferred_n}.json").read_text()) for model in ("X_net_reading", "X_gross_reading")]
    (out / "comparison.html").write_text(render_html(data), encoding="utf-8")
    summary["status"] = "COMPLETE" if not summary["unrun"] and all(not r["errors"] for r in summary["r3b"]) else "PARTIAL"
    summary["elapsed_seconds"] = round(time.monotonic() - START, 3)
    summary["peak_rss_mib"] = round(rss_mib(), 3)
    summary["evaluations"] = counts
    _save(out / "run-manifest.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "elapsed_seconds", "peak_rss_mib", "evaluations", "unrun")}, indent=2))


if __name__ == "__main__":
    main()
