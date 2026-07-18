"""Spawn-safe worker helpers for the F15 offload tests (NOT a test module).

These functions run **inside a ``ProcessPoolExecutor`` worker process**. Under the
``spawn`` start method (macOS local default) the worker re-imports this module and
resolves the function by reference, so every entrypoint here must be module-level,
importable, and depend only on the stdlib. The underscore prefix keeps pytest from
collecting this file.
"""

import os
import time


def sleep_worker(seconds: float) -> str:
    """Block the worker process for ``seconds`` (used by the hard-kill tests)."""
    time.sleep(seconds)
    return "slept"


def crash_worker(request_json: str) -> str:
    """Simulate a hard worker death (segfault / OOM-kill) mid-analysis.

    Signature matches ``run_robustness_v2`` so it can be monkeypatched in for it.
    ``os._exit`` bypasses cleanup so the parent sees the worker vanish and the
    ``ProcessPoolExecutor`` raises ``BrokenProcessPool`` — exactly the failure
    PC-F self-heals from.
    """
    os._exit(70)


def pid_probe(_arg: str = "") -> int:
    """Return the worker's PID (used to snapshot which process ran a task)."""
    return os.getpid()


def slow_analysis_worker(request_json: str) -> str:
    """Sleep far past any test deadline (used by the hard-kill deadline test).

    Signature matches ``run_robustness_v2`` so it can be monkeypatched in for it;
    it never returns because the parent terminates the worker on the deadline.
    """
    time.sleep(120)
    return ""
