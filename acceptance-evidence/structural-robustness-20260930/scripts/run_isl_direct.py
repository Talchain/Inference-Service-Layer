#!/usr/bin/env python3
"""SCI-STRUCTURAL-ROBUSTNESS 30 Sep 2026: run ISL's OWN v2 route in-process on the exact ISL requests PLoT built.

For each model (A, B, C) x option set (cur = today's CEE submission, served3 = + the served-era "Raise to £54"), the
ISL request is the one PLoT's real /v2/run sent to ISL for seed 1254899477 (captured from fetch). Only `seed` and
`request_id` are substituted per replicate (verified: PLoT's ISL requests for two seeds differ in nothing else).

The ONLY deviation from the served engine: wall-clock budgets are raised, because this machine ran at load average
40-70 and ISL's cooperative 50 s budget would otherwise drop optional phases at random. Budgets gate whether a phase
completes, never its numbers (ISL's own contract: "the RNG/compute is never touched by the guard"). The control
(same request, seed 1254899477) is compared byte-for-byte on the science fields against the PLoT-captured response.

Run from the ISL repo root:  ISL_AUTH_DISABLED=true .venv/bin/python <this> <out_dir>
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

OUT = sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/ssr-20260930/out"
RUNS = f"{OUT}/runs"
DIRECT = f"{OUT}/runs-direct"
os.makedirs(DIRECT, exist_ok=True)
SEEDS = ["1254899477", "1", "20260930"]
MODELS = ["A", "B", "C"]
OPTSETS = ["cur", "served3"]

client = TestClient(app)
for seed in SEEDS:
    for model in MODELS:
        for optset in OPTSETS:
            base = f"{model}-{optset}-s1254899477"
            calls = json.load(open(f"{RUNS}/{base}.isl-calls.json"))
            req = [c for c in calls if "analyze/v2" in c["url"]][0]["request"]
            req = json.loads(json.dumps(req))
            req["seed"] = seed
            req["request_id"] = f"ssr-direct-{model}-{optset}-s{seed}"
            tag = f"{model}-{optset}-s{seed}"
            path = f"{DIRECT}/{tag}.isl-response.json"
            if os.path.exists(path):
                print(f"{tag} exists, skip", flush=True)
                continue
            t0 = time.time()
            r = client.post("/api/v1/robustness/analyze/v2?response_version=2", json=req)
            ms = int((time.time() - t0) * 1000)
            json.dump({"status": r.status_code, "ms": ms, "request": req, "response": r.json()}, open(path, "w"), indent=2)
            print(f"{tag} status={r.status_code} ms={ms}", flush=True)
