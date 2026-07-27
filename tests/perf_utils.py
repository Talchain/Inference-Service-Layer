"""
Shared helpers for hardware-sensitive performance assertions.

CI policy (not an ISL science change): default PR CI proves CORRECTNESS.
Strict wall-clock/throughput budgets are unreliable on shared GitHub-hosted
runners (identical code measured 46-53 RPS against a 100 RPS target, and
3.2-3.3s against a 2.0s budget, purely from runner variability), so strict
timing enforcement runs only when ISL_PERF_STRICT is set — i.e. in the
dedicated perf workflow or on a stable/nightly perf runner.

Two mechanisms, chosen per the audit's boundary rule:

* Tests that exist ONLY to enforce a timing/throughput budget are marked
  ``@pytest.mark.perf`` and excluded from the default PR CI run entirely
  (they still run in the perf workflow).
* MIXED tests (science/functional assertions + a timing budget) stay in the
  default suite so their correctness coverage is never lost; only their
  timing assertion is routed through :func:`assert_time_budget`, which
  always reports the measurement but enforces the budget only under
  ISL_PERF_STRICT.

ROADMAP 1.244 — MEASURE THE QUANTITY YOU ACTUALLY MEAN
------------------------------------------------------
The two mechanisms above make a wall-clock assertion *quieter*; they do not
make it *right*. A loosened wall-clock threshold is still a wall-clock
threshold, so it still flakes under co-tenant load and still fails to say
anything about the code. The rule that follows from that:

* An assertion that bounds **WORK** (how much computation the code performs)
  must measure **CPU time** — :func:`assert_cpu_budget`. CPU time is
  invariant to co-tenant load: a busy runner steals wall-clock, not cycles.
  Because it cannot be moved by scheduler noise, such a budget is safe to
  enforce in the DEFAULT gate — it is a real regression signal, not a
  hardware measurement.
* An assertion that genuinely bounds **LATENCY** (wall-clock as experienced
  by a caller) must derive its tolerance from a calibration run and state
  its load assumption — :func:`assert_time_budget`, strict-gated.

Sibling precedent: CEE fixed the same defect class the same way in PR #714
(``performance.now()`` → ``process.cpuUsage()``).
"""

import os
import time

from contextlib import contextmanager


def is_perf_strict() -> bool:
    """Whether strict timing/throughput budgets are enforced.

    Evaluated at call time (not import time) so tests and CI can toggle
    ISL_PERF_STRICT via monkeypatch.setenv without an import-order trap.
    """
    return os.getenv("ISL_PERF_STRICT", "").strip().lower() in ("1", "true", "yes")


def assert_time_budget(elapsed_ms: float, budget_ms: float, label: str) -> None:
    """Report a wall-clock measurement; enforce the budget only under strict.

    Use inside MIXED tests so the correctness assertions around it keep
    running in default CI while the hardware-sensitive budget is enforced
    only where the runner is trustworthy (ISL_PERF_STRICT=1).
    """
    print(
        f"perf[{label}]: {elapsed_ms:.1f}ms "
        f"(budget {budget_ms:.0f}ms, strict_enforced={is_perf_strict()})"
    )
    if is_perf_strict():
        assert elapsed_ms <= budget_ms, (
            f"{label}: {elapsed_ms:.1f}ms exceeds {budget_ms:.0f}ms budget "
            f"(ISL_PERF_STRICT enforced)"
        )


@contextmanager
def measure_cpu():
    """Measure CPU time (user+system, all threads of this process) over a block.

    Yields a one-key dict that is populated on exit::

        with measure_cpu() as m:
            do_work()
        m["cpu_ms"]  # CPU milliseconds consumed by the block

    ``time.process_time()`` deliberately EXCLUDES time the process spent off-CPU
    — sleeping, blocked on I/O, or descheduled because a co-tenant job had the
    core. That exclusion is the whole point: it is what makes the measurement
    reproducible on a shared runner.
    """
    measurement: dict = {}
    start = time.process_time()
    try:
        yield measurement
    finally:
        measurement["cpu_ms"] = (time.process_time() - start) * 1000.0


def assert_cpu_budget(cpu_ms: float, budget_ms: float, label: str) -> None:
    """Assert a WORK budget in CPU milliseconds. Enforced on EVERY run.

    Unlike :func:`assert_time_budget` this is NOT strict-gated, and that is the
    point of ROADMAP 1.244: CPU time does not move when a shared runner is busy,
    so the budget is a property of the CODE rather than of the hardware weather.
    A breach means the code genuinely started doing more work.

    Scope, stated so nobody over-reads it: this bounds COMPUTATION. It does not
    catch a regression that adds *waiting* (a ``sleep``, a blocking network
    call), because waiting burns no CPU. Latency regressions of that shape are
    the job of :func:`assert_time_budget` and of the event-loop responsiveness
    tests — see ``tests/integration/test_f15_responsiveness.py``.
    """
    print(f"cpu[{label}]: {cpu_ms:.2f}ms CPU (budget {budget_ms:.0f}ms, always enforced)")
    assert cpu_ms <= budget_ms, (
        f"{label}: {cpu_ms:.2f}ms CPU exceeds {budget_ms:.0f}ms work budget. "
        "This is CPU time, not wall-clock — a busy runner cannot cause it, so "
        "the code is doing genuinely more work than the budget allows."
    )
