"""Compute admission governor for CPU-bound robustness analysis (Codex F15).

The 1000/min rate limiter counts *requests*; it is not a *compute* governor. This
module is the compute governor: it bounds how many analyses may execute (and be
queued) concurrently so a burst cannot pile unbounded work onto the small analysis
process pool. On overload it fails **early and typed** (429/503 + ``Retry-After``)
instead of leaving a caller hanging on a socket.

⚖️ PAUL RULING (18 Jul) — **NO env vars.** All tuning knobs are hardcoded module
constants, changeable by a code edit (not an env gate). Consistent with the
no-env-var stance for F15's offload as a whole.

Gates (checked atomically at admit time — the event loop is single-threaded, so
the check-and-reserve block below runs with no interleaving):

1. **Total admitted bound** (in-flight + queued ≥ ``workers + queue_max``) → 503
   ``service_busy_queue_full``. This is the bound PC-B saturates.
2. **Per-caller concurrency** (one API key already holds ≥ ``workers`` admitted)
   → 429 ``caller_concurrency_exceeded`` — one noisy caller cannot starve others.
3. **Weighted in-flight cost** (Σ admitted ``cost_units`` + new > budget) → 503
   ``service_busy_cost_budget``. The weight **is the F8 ``cost_units``** (via
   ``compute_weighted_cost`` / ``get_max_cost_units`` — one source of truth, no
   second cost model). At the real ceiling the budget is ``workers × max_cost``.
   When gate 3 binds (corrected 19 Jul — the earlier "never binds tighter than
   gate 1" note was WRONG): gate 1 admits up to ``workers + queue_max``
   (= ``3 × workers`` with the defaults here) jobs, and each admitted job is
   ≤ ``max_cost``, so the admitted cost can reach ``3 × workers × max_cost`` —
   THREE times this budget. Gate 3 therefore binds FIRST for expensive jobs: a
   burst of near-``max_cost`` analyses is capped at ≈``workers`` admitted
   (Σ cost ≤ ``workers × max_cost``), well below gate 1's ``3 × workers``. Gate 1
   dominates only for cheap jobs (mean cost ≲ ``max_cost / 3``). The old note
   assumed the budget equalled gate 1's aggregate; it is one-third of it. This is
   real, tested plumbing (tests drive it via the ``max_cost`` override); it also
   binds if the ceiling or worker count changes.

The semaphore (sized to ``workers``) enforces "in-flight never exceeds workers"
(PC-B). Reservations are taken synchronously at admit and released in a ``finally``
so a crash/cancellation can never leak a slot — a leaked slot would silently
shrink capacity forever, so this is load-bearing.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional

from src.services.robustness_analyzer_v2 import get_max_cost_units

# ---------------------------------------------------------------------------
# Hardcoded tuning constants (NO env vars — Paul ruling 18 Jul).
# ---------------------------------------------------------------------------
# Worker processes. Render isl-staging is a 1-2 CPU class, so this may resolve to
# 1 there — still the whole point (CPU off the event loop). Kept low on purpose:
# each worker holds a copy of the graph + samples, and memory growth is the
# design's top rollback trigger.
ANALYSIS_WORKERS: int = min(2, os.cpu_count() or 1)

# Bounded queue depth beyond the executing slots. Total admitted (executing +
# queued) is capped at ANALYSIS_WORKERS + ANALYSIS_QUEUE_MAX.
ANALYSIS_QUEUE_MAX: int = 2 * ANALYSIS_WORKERS

# Retry-After hint (seconds) returned to a rejected caller. Deliberately short —
# analyses are tens of seconds but slots free continuously as jobs complete.
RETRY_AFTER_SECONDS: int = 5


def max_inflight_cost_units() -> int:
    """Weighted in-flight cost budget, derived from F8's single source of truth.

    ``ANALYSIS_WORKERS × get_max_cost_units()`` — the aggregate weight of
    ``workers`` maximally-admitted jobs. Read at call time so an env change to
    ``ISL_MAX_COST_UNITS`` (F8's existing knob) flows through without a second
    mirror.
    """
    return ANALYSIS_WORKERS * get_max_cost_units()


class Overload(Exception):
    """Raised by :meth:`ComputeGovernor.admit` when a gate rejects the request.

    Carries the HTTP status the handler should return (429 for caller-attributable
    concurrency, 503 for global service-busy), a stable machine reason, and the
    ``Retry-After`` seconds hint.
    """

    def __init__(self, status_code: int, reason: str, retry_after: int = RETRY_AFTER_SECONDS):
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason
        self.retry_after = retry_after


class ComputeGovernor:
    """Weighted concurrency governor for the analysis process pool.

    One instance is created in the app lifespan and stored on ``app.state`` (a
    single instance per event loop — the counters are only touched from the loop
    thread, so no locking is needed).
    """

    def __init__(
        self,
        workers: int = ANALYSIS_WORKERS,
        queue_max: int = ANALYSIS_QUEUE_MAX,
        max_cost: Optional[int] = None,
    ) -> None:
        self._workers = workers
        self._queue_max = queue_max
        self._admission_bound = workers + queue_max
        # ``max_cost=None`` → resolve from F8 at call time; an explicit int pins it
        # (tests drive gate 3 by passing a small value).
        self._max_cost = max_cost
        self._sem = asyncio.Semaphore(workers)
        # All three counters track ADMITTED work (queued + executing).
        self._admitted = 0
        self._admitted_cost = 0
        self._per_key_admitted: Dict[str, int] = {}
        # Executing (holding a worker slot) — for the PC-B "in-flight ≤ workers"
        # assertion; bounded by the semaphore.
        self._inflight = 0

    # -- introspection (tests + /metrics could read these) -------------------
    @property
    def workers(self) -> int:
        return self._workers

    @property
    def admission_bound(self) -> int:
        return self._admission_bound

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def admitted(self) -> int:
        return self._admitted

    def _max_cost_budget(self) -> int:
        return self._max_cost if self._max_cost is not None else max_inflight_cost_units()

    @asynccontextmanager
    async def admit(self, cost_units: int, api_key: Optional[str] = None) -> AsyncIterator[None]:
        """Admit one analysis, or raise :class:`Overload`.

        Usage::

            async with governor.admit(cost.total, api_key):
                result = await run_offloaded(...)

        The body runs only once a worker slot is held (``in-flight ≤ workers``).
        """
        key = api_key or "_anon"

        # --- atomic check + reserve (no await in this block) ----------------
        if self._admitted >= self._admission_bound:
            raise Overload(503, "service_busy_queue_full")
        if self._per_key_admitted.get(key, 0) >= self._workers:
            raise Overload(429, "caller_concurrency_exceeded")
        if self._admitted_cost + cost_units > self._max_cost_budget():
            raise Overload(503, "service_busy_cost_budget")

        self._admitted += 1
        self._admitted_cost += cost_units
        self._per_key_admitted[key] = self._per_key_admitted.get(key, 0) + 1
        # --------------------------------------------------------------------

        acquired = False
        try:
            await self._sem.acquire()
            acquired = True
            self._inflight += 1
            yield
        finally:
            if acquired:
                self._inflight -= 1
                self._sem.release()
            self._admitted -= 1
            self._admitted_cost -= cost_units
            self._per_key_admitted[key] = self._per_key_admitted.get(key, 1) - 1
            if self._per_key_admitted.get(key, 0) <= 0:
                self._per_key_admitted.pop(key, None)
