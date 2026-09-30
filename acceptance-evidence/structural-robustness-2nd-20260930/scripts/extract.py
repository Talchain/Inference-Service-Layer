#!/usr/bin/env python3
"""SCI-STRUCTURAL-ROBUSTNESS (2nd run): read the like-for-like headline fields from ISL V2 responses, and compare two
responses on every science field (volatile fields masked as `benchmarks/science-validation/exp3_determinism.py`).

  python extract.py headline <run.json> [goal_frame]   -> JSON headline
  python extract.py same <a.json> <b.json>             -> exit 0 iff science fields are identical
"""
import json
import sys
from typing import Any, Dict

VOLATILE = {"execution_time_ms", "processing_time_ms", "timestamp", "request_id", "ms", "computed_at",
            "cache_key", "duration_ms", "elapsed_ms", "interpretation"}


def _mask(x: Any) -> Any:
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in VOLATILE or k.endswith("_ms"):
                continue
            if k in ("critique_id", "id") and isinstance(v, str) and v.startswith("critique_"):
                continue
            out[k] = _mask(v)
        return out
    if isinstance(x, list):
        return [_mask(v) for v in x]
    return x


def _resp(path: str) -> Dict[str, Any]:
    d = json.load(open(path))
    return d.get("response", d)


def headline(path: str, goal_frame: float) -> Dict[str, Any]:
    r = _resp(path)
    opts = {}
    for o in r.get("options", []):
        out = o.get("outcome") or {}
        ca = o.get("constraint_analysis") or {}
        opts[o["id"]] = {
            "win_probability": o.get("win_probability"),
            "probability_of_goal": o.get("probability_of_goal"),
            "mrr_mean_gbp": None if out.get("mean") is None else round(out["mean"] * goal_frame, 1),
            "mrr_p10_gbp": None if out.get("p10") is None else round(out["p10"] * goal_frame, 1),
            "mrr_p50_gbp": None if out.get("p50") is None else round(out["p50"] * goal_frame, 1),
            "mrr_p90_gbp": None if out.get("p90") is None else round(out["p90"] * goal_frame, 1),
            "churn_below_limit": [c.get("prob_satisfied") for c in ca.get("constraints", [])],
        }
    wins = {k: v["win_probability"] for k, v in opts.items() if v["win_probability"] is not None}
    rob = r.get("robustness") or {}
    return {
        "analysis_status": r.get("analysis_status"),
        "leader": max(wins, key=wins.get) if wins else None,
        "options": opts,
        "robustness_level": rob.get("level"),
        "recommendation_stability": rob.get("recommendation_stability"),
        "fragile_edges": sorted((e.get("edge_id"), round(e.get("switch_probability") or 0, 4))
                                for e in rob.get("fragile_edges", []) or []),
        "identity_evaluations": [{k: i.get(k) for k in ("node_id", "evaluated", "withheld_reason")}
                                 for i in r.get("identity_evaluations", []) or []],
        "warning_codes": sorted({w.get("code") for w in (r.get("inference_warnings") or []) if isinstance(w, dict)}),
        "critique_codes": sorted({c.get("code") for c in (r.get("critiques") or []) if isinstance(c, dict)}),
    }


def main() -> None:
    mode = sys.argv[1]
    if mode == "headline":
        frame = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        print(json.dumps(headline(sys.argv[2], frame), indent=2, ensure_ascii=False))
    elif mode == "same":
        a, b = _mask(_resp(sys.argv[2])), _mask(_resp(sys.argv[3]))
        if a == b:
            print("IDENTICAL (science fields, volatile masked)")
            return
        diffs = []

        def walk(x: Any, y: Any, p: str) -> None:
            if type(x) is not type(y):
                diffs.append(f"{p}: {x!r:.80} != {y!r:.80}")
            elif isinstance(x, dict):
                for k in sorted(set(x) | set(y)):
                    walk(x.get(k), y.get(k), f"{p}.{k}")
            elif isinstance(x, list):
                if len(x) != len(y):
                    diffs.append(f"{p}: len {len(x)} != {len(y)}")
                for i, (u, v) in enumerate(zip(x, y)):
                    walk(u, v, f"{p}[{i}]")
            elif x != y:
                diffs.append(f"{p}: {x!r:.80} != {y!r:.80}")

        walk(a, b, "$")
        print(f"DIFFERENT: {len(diffs)} paths")
        print("\n".join(diffs[:40]))
        sys.exit(1)


if __name__ == "__main__":
    main()
