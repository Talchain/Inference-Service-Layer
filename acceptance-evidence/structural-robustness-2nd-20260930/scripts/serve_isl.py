#!/usr/bin/env python3
"""SCI-STRUCTURAL-ROBUSTNESS (2nd run): serve ISL locally for PLoT's real ISL client, with the same raised wall-clock
budgets as `run_isl.py` (budgets gate whether a phase completes, never its numbers).

Usage (ISL repo root): ISL_AUTH_DISABLED=true poetry run python <this> [port]
"""
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("ISL_AUTH_DISABLED", "true")

import uvicorn  # noqa: E402

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

uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8931, log_level="warning")
