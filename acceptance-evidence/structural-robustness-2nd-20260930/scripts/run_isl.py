#!/usr/bin/env python3
"""SCI-STRUCTURAL-ROBUSTNESS (2nd run, 30 Sep 2026): run ISL's OWN v2 route in-process on an ISL request file.

Same route, same engine, same deviation as the first run's `run_isl_direct.py` (ISL `6f7f9b43`): wall-clock budgets
are raised so optional phases complete deterministically on a loaded machine. Budgets gate whether a phase completes,
never its numbers. Only `seed` and `request_id` are substituted.

Usage (from the ISL repo root):
  ISL_AUTH_DISABLED=true poetry run python <this> <request.json> <out.json> [seed ...]
With several seeds, `<out.json>` must contain `{seed}`.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
os.environ.setdefault("ISL_AUTH_DISABLED", "true")

from fastapi.testclient import TestClient  # noqa: E402

import src.middleware.request_limits as request_limits  # noqa: E402
import src.services.analysis_pool as analysis_pool  # noqa: E402
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2  # noqa: E402

BIG_MS = 3_600_000
for attr in ("OVERALL_REQUEST_BUDGET_MS", "EVPI_BUDGET_MS", "PATH_DECOMPOSITION_BUDGET_MS", "E_VALUE_BUDGET_MS",
             "FLIP_STABILITY_BUDGET_MS", "FACTOR_FLIP_BUDGET_MS"):
    setattr(RobustnessAnalyzerV2, attr, BIG_MS)
for k in list(request_limits.ENDPOINT_TIMEOUTS):
    request_limits.ENDPOINT_TIMEOUTS[k] = 3600
analysis_pool.ANALYSIS_HARD_DEADLINE_S = 3600.0

from src.api.main import app  # noqa: E402


def main() -> None:
    req_path, out_tpl, *seeds = sys.argv[1:]
    base = json.load(open(req_path))
    base = base.get("request", base)
    client = TestClient(app)
    for seed in seeds or [str(base.get("seed"))]:
        req = json.loads(json.dumps(base))
        req["seed"] = seed
        req["request_id"] = f"ssr2-{os.path.basename(req_path).split('.')[0]}-s{seed}"
        t0 = time.time()
        r = client.post("/api/v1/robustness/analyze/v2?response_version=2", json=req,
                        headers={"X-ISL-Response-Version": "2"})
        ms = int((time.time() - t0) * 1000)
        out = out_tpl.format(seed=seed)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump({"status": r.status_code, "ms": ms, "request": req, "response": r.json()}, open(out, "w"), indent=2)
        print(f"{out} status={r.status_code} ms={ms}", flush=True)


if __name__ == "__main__":
    main()
