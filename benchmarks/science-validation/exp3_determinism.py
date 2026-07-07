"""Experiment 3 — same-seed determinism at scale.

Claim under test: with a pinned seed (and pinned request_id), the full v2
response is byte-identical across repeat runs — in-process, over the wire,
and across OS processes. Volatile fields are catalogued, not assumed.

Method: 50 deterministically generated diverse graphs (2-30 nodes; mixed
kinds; factor uncertainties; goal thresholds; constraints; VoI / e-values /
path decomposition toggles). For each graph:

1. in-process: analyze() twice with fresh analyzer instances; compare raw
   serialised JSON (by_alias) and canonical JSON (execution_time_ms zeroed);
2. wire: POST /api/v1/robustness/analyze/v2 twice via TestClient with
   X-ISL-Response-Version: 2; compare raw bytes and normalised bodies;
3. cross-process: a child Python process regenerates the same graphs from the
   pinned generator seed and reports SHA-256 hashes of canonical responses;
   compared against the parent's hashes.

Any field that differs between same-seed runs is recorded in the volatile
catalogue with its JSON path.

Run:  poetry run python benchmarks/science-validation/exp3_determinism.py [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time

from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("ISL_AUTH_DISABLED", "true")
logging.disable(logging.INFO)  # service INFO logs would swamp harness output

from _lib import (  # noqa: E402
    REPO_ROOT,
    SEEDS,
    build_request,
    canonical_response_json,
    save_result,
)
from graphs import random_graph_payloads  # noqa: E402

from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2  # noqa: E402

N_GRAPHS_FULL = 50
N_GRAPHS_QUICK = 10


def json_diff_paths(a: Any, b: Any, path: str = "$") -> List[str]:
    """Recursive structural diff returning JSON paths that differ."""
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        paths = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                paths.append(f"{path}.{key}")
            else:
                paths.extend(json_diff_paths(a[key], b[key], f"{path}.{key}"))
        return paths
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}.length"]
        paths = []
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            paths.extend(json_diff_paths(x, y, f"{path}[{i}]"))
        return paths
    return [] if a == b else [path]


KNOWN_VOLATILE_KEYS = {"execution_time_ms", "processing_time_ms", "timestamp"}


def mask_known_volatiles(obj: Any) -> Any:
    """Mask the fields known (or found by this experiment) to vary run-to-run:

    - execution_time_ms / processing_time_ms (wall clock)
    - timestamp (envelope)
    - critiques[].id (uuid4 per run — src/models/critique.py)
    """
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in KNOWN_VOLATILE_KEYS and isinstance(value, (int, float, str)):
                out[key] = 0
            elif key == "id" and isinstance(value, str) and value.startswith("critique_"):
                out[key] = "critique_MASKED"
            else:
                out[key] = mask_known_volatiles(value)
        return out
    if isinstance(obj, list):
        return [mask_known_volatiles(v) for v in obj]
    return obj


def stable_json(raw_json: str) -> str:
    return json.dumps(mask_known_volatiles(json.loads(raw_json)), sort_keys=True)


def order_insensitive_json(raw_json: str) -> str:
    """stable_json plus normalisation of the fields whose ORDER (not content)
    depends on the per-process string-hash salt (PYTHONHASHSEED):

    - robustness.fragile_edges / robust_edges are built via list(set(...))
      (robustness_analyzer_v2.py ~2792-2805) => sorted here;
    - robustness.interpretation embeds the first three entries of that
      unordered list => masked here.

    Comparing this form across processes isolates "same numbers, different
    ordering" from any genuine numeric divergence.
    """
    body = mask_known_volatiles(json.loads(raw_json))
    for key in ("robustness",):
        section = body.get(key)
        if isinstance(section, dict):
            for list_key in ("fragile_edges", "robust_edges"):
                if isinstance(section.get(list_key), list):
                    section[list_key] = sorted(section[list_key])
            if "interpretation" in section:
                section["interpretation"] = "MASKED"
    return json.dumps(body, sort_keys=True)


CHILD_SCRIPT = """
import hashlib, json, logging, sys
logging.disable(logging.WARNING)
sys.path.insert(0, {harness_dir!r})
from _lib import build_request, canonical_response_json
from graphs import random_graph_payloads
from exp3_determinism import order_insensitive_json, stable_json
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
payloads = random_graph_payloads({n_graphs}, {gen_seed}, {request_seed})
stable, orderless = [], []
for p in payloads:
    resp = RobustnessAnalyzerV2().analyze(build_request(p))
    raw = canonical_response_json(resp)
    stable.append(hashlib.sha256(stable_json(raw).encode()).hexdigest())
    orderless.append(hashlib.sha256(order_insensitive_json(raw).encode()).hexdigest())
print(json.dumps({{"stable": stable, "order_insensitive": orderless}}))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="10 graphs, skip child process")
    args = parser.parse_args()

    n_graphs = N_GRAPHS_QUICK if args.quick else N_GRAPHS_FULL
    gen_seed = SEEDS["exp3_graphgen_seed"]
    request_seed = SEEDS["exp3_request_seed"]
    payloads = random_graph_payloads(n_graphs, gen_seed, request_seed)

    t0 = time.time()
    volatile_catalogue: Dict[str, int] = {}
    per_graph: List[Dict[str, Any]] = []
    parent_hashes: List[str] = []
    parent_orderless_hashes: List[str] = []

    # Deferred import so ISL_AUTH_DISABLED is set before the app loads
    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)
    url = "/api/v1/robustness/analyze/v2"
    headers = {"X-ISL-Response-Version": "2"}

    for gi, payload in enumerate(payloads):
        entry: Dict[str, Any] = {"graph": gi, "request_id": payload["request_id"]}

        # 1. In-process repeat (fresh analyzer instances)
        request = build_request(payload)
        r1 = RobustnessAnalyzerV2().analyze(request)
        r2 = RobustnessAnalyzerV2().analyze(request)
        raw1 = r1.model_dump_json(by_alias=True)
        raw2 = r2.model_dump_json(by_alias=True)
        entry["inprocess_raw_identical"] = raw1 == raw2
        if raw1 != raw2:
            for p in json_diff_paths(json.loads(raw1), json.loads(raw2)):
                generic = p.replace("$.", "").split("[")[0]
                volatile_catalogue[generic] = volatile_catalogue.get(generic, 0) + 1
        c1, c2 = canonical_response_json(r1), canonical_response_json(r2)
        entry["inprocess_canonical_identical"] = c1 == c2
        s1, s2 = stable_json(c1), stable_json(c2)
        entry["inprocess_stable_identical"] = s1 == s2
        if s1 != s2:
            entry["inprocess_stable_diff"] = json_diff_paths(json.loads(s1), json.loads(s2))[:20]
        parent_hashes.append(hashlib.sha256(s1.encode()).hexdigest())
        parent_orderless_hashes.append(
            hashlib.sha256(order_insensitive_json(c1).encode()).hexdigest()
        )

        # 2. Wire repeat
        w1 = client.post(url, json=payload, headers=headers)
        w2 = client.post(url, json=payload, headers=headers)
        entry["wire_status"] = w1.status_code
        if w1.status_code == 200 and w2.status_code == 200:
            entry["wire_raw_identical"] = w1.content == w2.content
            b1, b2 = w1.json(), w2.json()
            diff = json_diff_paths(b1, b2)
            for p in diff:
                generic = p.replace("$.", "").split("[")[0]
                volatile_catalogue[generic] = volatile_catalogue.get(generic, 0) + 1
            # Mask known volatiles and re-compare
            m1, m2 = mask_known_volatiles(b1), mask_known_volatiles(b2)
            entry["wire_normalised_identical"] = json.dumps(m1, sort_keys=True) == json.dumps(
                m2, sort_keys=True
            )
            if not entry["wire_normalised_identical"]:
                entry["wire_normalised_diff"] = json_diff_paths(m1, m2)[:20]
        per_graph.append(entry)

    # 3. Cross-process: child regenerates everything from pinned seeds
    cross_process: Dict[str, Any] = {"run": False}
    if not args.quick:
        script = CHILD_SCRIPT.format(
            harness_dir=str(Path(__file__).resolve().parent),
            n_graphs=n_graphs,
            gen_seed=gen_seed,
            request_seed=request_seed,
        )
        child = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=1800,
        )
        child_stdout_lines = child.stdout.strip().splitlines()
        if child.returncode != 0 or not child_stdout_lines:
            cross_process = {
                "run": True,
                "error": child.stderr[-2000:] if child.stderr else "empty child stdout",
            }
        else:
            child_hashes = json.loads(child_stdout_lines[-1])
            # strict=True: a truncated child hash list must fail loudly, not
            # silently shorten the comparison.
            mismatches = [
                i
                for i, (a, b) in enumerate(zip(parent_hashes, child_hashes["stable"], strict=True))
                if a != b
            ]
            orderless_mismatches = [
                i
                for i, (a, b) in enumerate(
                    zip(
                        parent_orderless_hashes,
                        child_hashes["order_insensitive"],
                        strict=True,
                    )
                )
                if a != b
            ]
            cross_process = {
                "run": True,
                "all_identical": not mismatches,
                "mismatched_graphs": mismatches,
                "order_insensitive_all_identical": not orderless_mismatches,
                "order_insensitive_mismatched_graphs": orderless_mismatches,
            }

    elapsed = time.time() - t0
    summary = {
        "n_graphs": n_graphs,
        "inprocess_raw_identical": sum(1 for e in per_graph if e["inprocess_raw_identical"]),
        "inprocess_canonical_identical": sum(
            1 for e in per_graph if e["inprocess_canonical_identical"]
        ),
        "inprocess_stable_identical": sum(1 for e in per_graph if e["inprocess_stable_identical"]),
        "wire_ok": sum(1 for e in per_graph if e.get("wire_status") == 200),
        "wire_raw_identical": sum(1 for e in per_graph if e.get("wire_raw_identical")),
        "wire_normalised_identical": sum(
            1 for e in per_graph if e.get("wire_normalised_identical")
        ),
        "volatile_fields": volatile_catalogue,
        "cross_process": cross_process,
    }
    path = save_result(
        "exp3_determinism" + ("_quick" if args.quick else ""),
        {
            "config": {
                "n_graphs": n_graphs,
                "graphgen_seed": gen_seed,
                "request_seed": request_seed,
            },
            "summary": summary,
            "per_graph": per_graph,
            "elapsed_seconds": round(elapsed, 1),
        },
    )
    print(f"exp3 complete in {elapsed:.1f}s -> {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
