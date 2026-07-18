"""Analysis process-pool lifecycle + self-healing offload (Codex F15).

This is the load-bearing safety layer. Per Paul's 18 Jul ruling there is **no env
kill-switch**: the offload is always active, and robustness is provided by
*self-healing*, not by an operator flip.

Three failure modes, all handled so analysis can **never** brick:

* **Worker crash / ``BrokenProcessPool``** (segfault, OOM-kill, external kill):
  the pool is recreated AND the request is completed **in-process** for correctness
  (``isl_analysis_offload_degraded``). One crash degrades one request; the pool
  recovers for the next. (PC-F.)
* **Hard deadline exceeded**: a ``ProcessPoolExecutor`` cannot cancel a running
  task, so ``asyncio.wait_for`` alone would only *stop awaiting* while the worker
  keeps burning a CPU. We therefore **terminate the worker processes** (SIGKILL)
  to actually free the CPU, recreate the pool, and return a typed 504. This is the
  only real preemption of a GIL-bound loop. (PC-C.)
* **Pool unavailable** (never constructed — e.g. an ASGITransport test with no
  lifespan, or lifespan construction failed): fall straight to the in-process
  path so the endpoint still returns correct science.

Killing worker processes breaks the whole executor, which fails *sibling*
in-flight futures with ``BrokenProcessPool`` — those siblings then self-heal to
the in-process path via the same handler, so terminating the pool is safe even
when a sibling analysis is running.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, List, Tuple, cast

from src.models.robustness_v2 import RobustnessRequestV2, RobustnessResponseV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.services.robustness_worker import run_robustness_v2

logger = logging.getLogger(__name__)

# Hard wall-clock backstop for a single offloaded analysis (seconds). Sits ABOVE
# the analyzer's own cooperative OVERALL_REQUEST_BUDGET_MS (50 s, which makes the
# optional phases degrade-and-disclose) and BELOW the 90 s route-timeout
# middleware for ``/api/v1/robustness/`` — so a genuinely stuck worker is killed
# here (freeing the CPU) rather than the middleware merely returning 504 while the
# worker runs on. A legitimate admitted job returns well under this. Module-level
# so tests can monkeypatch it small; read at call time in ``run_offloaded``.
ANALYSIS_HARD_DEADLINE_S: float = 80.0


class AnalysisDeadlineExceeded(Exception):
    """Raised when an offloaded analysis exceeds :data:`ANALYSIS_HARD_DEADLINE_S`.

    The worker running it has been terminated and the pool recreated by the time
    this propagates. Handlers map it to a typed 504.
    """

    def __init__(self, deadline_s: float):
        super().__init__(f"analysis exceeded hard deadline of {deadline_s}s")
        self.deadline_s = deadline_s


def create_analysis_pool(workers: int) -> ProcessPoolExecutor:
    """Create the analysis pool.

    ``max_tasks_per_child=1`` recycles a worker after every analysis so its memory
    (graph + up to ~100k sample floats) is reclaimed — memory growth is the
    design's top rollback trigger. The cost is a per-request worker (re)spawn:
    negligible under ``fork`` (Linux/Render) and, under ``spawn`` (macOS local),
    dwarfed by the tens-of-seconds compute. Added in CPython 3.11.
    """
    return ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1)


def _terminate_pool_processes(pool: ProcessPoolExecutor) -> List[int]:
    """SIGKILL every worker process of ``pool``; return the pids killed.

    This is what frees the CPU held by a GIL-bound loop — a ``ProcessPoolExecutor``
    exposes no public "terminate now", so we reach into the ``_processes`` map
    (stable in CPython 3.11's ``concurrent.futures.process``). Guarded with
    ``getattr`` so a future CPython shape change degrades to a plain shutdown
    rather than raising.
    """
    killed: List[int] = []
    procs = getattr(pool, "_processes", None) or {}
    for proc in list(procs.values()):
        pid = getattr(proc, "pid", None)
        if pid is None:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except (ProcessLookupError, OSError):
            # Already gone — fine.
            pass
    return killed


def _swap_in_fresh_pool(app: Any, stale_pool: ProcessPoolExecutor) -> ProcessPoolExecutor:
    """Replace ``app.state.analysis_pool`` with a fresh pool, once.

    Identity-guarded so concurrent self-heals (a hard-kill plus a sibling seeing
    ``BrokenProcessPool``) don't churn multiple pools — the event loop is
    single-threaded, so the ``is`` check is race-free. Returns the pool now on
    ``app.state`` (either the one we created, or the fresh one a peer installed).
    """
    current = getattr(app.state, "analysis_pool", None)
    if current is stale_pool or current is None:
        try:
            stale_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # pragma: no cover - shutdown best-effort
            pass
        workers = getattr(app.state, "analysis_workers", None) or 1
        fresh = create_analysis_pool(workers)
        app.state.analysis_pool = fresh
        return fresh
    return cast(ProcessPoolExecutor, current)


def _hard_kill_and_recreate(
    app: Any, pool: ProcessPoolExecutor
) -> Tuple[ProcessPoolExecutor, List[int]]:
    """Terminate ``pool``'s workers (free the CPU) and install a fresh pool."""
    killed = _terminate_pool_processes(pool)
    fresh = _swap_in_fresh_pool(app, pool)
    return fresh, killed


def _run_in_process(
    request: RobustnessRequestV2, request_id: str, *, reason: str, degraded: bool
) -> RobustnessResponseV2:
    """Run the analysis in THIS process (blocks the event loop for its duration).

    ``degraded=True`` logs the self-heal event ``isl_analysis_offload_degraded``
    (paths/reason only — never request values). ``degraded=False`` is the benign
    "no pool configured" path (e.g. a test with no lifespan) and logs at debug.
    """
    if degraded:
        logger.warning(
            "isl_analysis_offload_degraded",
            extra={"request_id": request_id, "reason": reason, "path": "in_process_fallback"},
        )
    else:
        logger.debug(
            "isl_analysis_inprocess_no_pool",
            extra={"request_id": request_id, "reason": reason, "path": "in_process"},
        )
    return RobustnessAnalyzerV2().analyze(request)


async def run_offloaded(
    app: Any, request: RobustnessRequestV2, request_id: str
) -> RobustnessResponseV2:
    """Run one robustness analysis off the event loop via the process pool.

    Returns the same ``RobustnessResponseV2`` the in-process ``analyze()`` would
    (byte-identical for a fixed seed; PC-D). Self-heals on crash/pool-loss;
    enforces the hard deadline by terminating the worker.

    Raises:
        AnalysisDeadlineExceeded: the analysis blew the hard deadline (worker
            killed, pool recreated) → handler returns 504.
        Exception: any application error raised *by the analysis itself* (e.g.
            ``ValueError`` for a cyclic graph) propagates unchanged so the
            handler's existing 422/500 mapping applies — these are NOT pool
            failures and are not retried in-process.
    """
    pool = getattr(app.state, "analysis_pool", None)
    deadline = ANALYSIS_HARD_DEADLINE_S  # read here so monkeypatch/edit takes effect

    if pool is None:
        # Offload subsystem not available — correct science, not a "degraded" event.
        return _run_in_process(request, request_id, reason="pool_unavailable", degraded=False)

    loop = asyncio.get_running_loop()
    request_json = request.model_dump_json()

    try:
        fut = loop.run_in_executor(pool, run_robustness_v2, request_json)
    except RuntimeError as e:
        # e.g. "cannot schedule new futures after shutdown/interpreter shutdown".
        _swap_in_fresh_pool(app, pool)
        return _run_in_process(
            request, request_id, reason=f"submit_failed:{type(e).__name__}", degraded=True
        )

    try:
        result_json = await asyncio.wait_for(fut, timeout=deadline)
    except asyncio.TimeoutError:
        # HARD KILL — terminate the worker to actually free the CPU, recreate.
        _fresh, killed = _hard_kill_and_recreate(app, pool)
        # The orphaned future will resolve to BrokenProcessPool in the background;
        # retrieve+swallow so asyncio does not warn "exception never retrieved".
        fut.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        logger.warning(
            "isl_analysis_offload_deadline",
            extra={
                "request_id": request_id,
                "deadline_s": deadline,
                "killed_pids": killed,
                "reason": "hard_deadline_exceeded",
            },
        )
        raise AnalysisDeadlineExceeded(deadline)
    except BrokenProcessPool:
        # SELF-HEAL — worker died unexpectedly (crash / external kill). Recreate
        # the pool and finish this request in-process with correct science.
        _swap_in_fresh_pool(app, pool)
        return _run_in_process(request, request_id, reason="broken_process_pool", degraded=True)

    return RobustnessResponseV2.model_validate_json(result_json)
