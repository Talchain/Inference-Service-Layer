"""Shared infrastructure for the science-validation benchmark harness.

This module is benchmark infrastructure only — it is NOT part of the service,
is not collected by pytest (pytest testpaths = tests/), and makes no changes
to src/. All instrumentation (e.g. the marginal-K override) is applied at
harness level via monkeypatching of the in-process analyzer.

Run scripts from the repository root, e.g.::

    poetry run python benchmarks/science-validation/exp1_higher_k.py --quick
"""

from __future__ import annotations

import contextlib
import json
import platform
import subprocess
import sys
import time

from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent.parent
RESULTS_DIR = HARNESS_DIR / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.robustness_v2 import RobustnessRequestV2, RobustnessResponseV2  # noqa: E402
from src.services.robustness_analyzer_v2 import (  # noqa: E402
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
    filter_inference_graph,
)
from src.utils.rng import SeededRNG  # noqa: E402

# ---------------------------------------------------------------------------
# Pinned seeds — the single registry every experiment draws from.
# Changing any value here invalidates committed results/.
# ---------------------------------------------------------------------------
SEEDS: Dict[str, Any] = {
    "exp1_global_seeds": [42, 1042, 2042, 3042, 4042],
    "exp2_policy_seed": 42,
    "exp2_replicate_seeds": [7000 + 13 * i for i in range(20)],
    "exp3_graphgen_seed": 20260707,
    "exp3_request_seed": 42,
    "exp4_request_seed": 42,
}


# ---------------------------------------------------------------------------
# Harness-level K override (no src/ change).
#
# k_samples is a def-time-bound default argument, so monkeypatching the
# MARGINAL_K_SAMPLES constant has no effect. Instead we wrap the method and
# forward k_samples explicitly; the sole production call site passes
# keyword arguments only and omits k_samples, so the wrapper is transparent.
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def override_marginal_k(k: int) -> Iterator[None]:
    """Context manager: force _compute_marginal_switch_probability to use K=k."""
    original = RobustnessAnalyzerV2._compute_marginal_switch_probability

    def wrapper(self: RobustnessAnalyzerV2, **kwargs: Any) -> float:
        kwargs.setdefault("k_samples", k)
        return original(self, **kwargs)

    RobustnessAnalyzerV2._compute_marginal_switch_probability = wrapper  # type: ignore[method-assign]
    try:
        yield
    finally:
        RobustnessAnalyzerV2._compute_marginal_switch_probability = original  # type: ignore[method-assign]


def build_request(payload: Dict[str, Any]) -> RobustnessRequestV2:
    """Parse a wire-shaped dict (edges keyed 'from'/'to') into a validated request."""
    return RobustnessRequestV2.model_validate(payload)


def make_evaluator(request: RobustnessRequestV2) -> Tuple[RobustnessRequestV2, SCMEvaluatorV2]:
    """Mirror RobustnessAnalyzerV2.analyze() setup: filter graph, build evaluator.

    Returns the (possibly graph-filtered) request and an SCMEvaluatorV2 without
    epsilon noise (all harness graphs use epsilon_std = 0).
    """
    filtered = filter_inference_graph(request.graph, log=False)
    if filtered is not request.graph:
        request = request.model_copy(update={"graph": filtered})
    return request, SCMEvaluatorV2(request.graph)


def marginal_switch_probability(
    request: RobustnessRequestV2,
    evaluator: SCMEvaluatorV2,
    edge_key: Tuple[str, str],
    global_seed: int,
    k_samples: int,
) -> float:
    """Direct call to the production marginal-switch estimator with explicit K.

    Uses the exact production code path (per-edge SHA-256 sub-seed, PCG64,
    baseline at expected values) — only the sample count is parameterised.
    """
    analyzer = RobustnessAnalyzerV2()
    return analyzer._compute_marginal_switch_probability(
        edge_key=edge_key,
        request=request,
        evaluator=evaluator,
        global_seed=global_seed,
        k_samples=k_samples,
    )


def canonical_response_json(
    response: RobustnessResponseV2, zero_execution_time: bool = True
) -> str:
    """Serialise a response exactly as the wire does (by_alias), optionally
    zeroing the known-volatile execution_time_ms field."""
    if zero_execution_time:
        response = response.model_copy(deep=True)
        response.metadata.execution_time_ms = 0
    return response.model_dump_json(by_alias=True)


# ---------------------------------------------------------------------------
# Result I/O with provenance
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def save_result(name: str, payload: Dict[str, Any]) -> Path:
    """Write an experiment result JSON with reproduction provenance."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "provenance": {
            "command": f"poetry run python {' '.join(sys.argv)}",
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        **payload,
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")
    return path


def rule_of_three_upper(n_zero_draws: int) -> float:
    """95% upper bound on p when 0 events observed in n draws (rule of three)."""
    return 3.0 / n_zero_draws if n_zero_draws > 0 else float("inf")


__all__ = [
    "HARNESS_DIR",
    "REPO_ROOT",
    "RESULTS_DIR",
    "SEEDS",
    "SeededRNG",
    "build_request",
    "canonical_response_json",
    "make_evaluator",
    "marginal_switch_probability",
    "override_marginal_k",
    "rule_of_three_upper",
    "save_result",
]
