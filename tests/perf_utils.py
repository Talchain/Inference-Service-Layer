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
"""

import os


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
