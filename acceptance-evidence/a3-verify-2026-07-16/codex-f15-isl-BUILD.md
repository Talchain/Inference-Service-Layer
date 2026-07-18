# Codex F15 (ISL) — BUILD evidence: offload CPU analysis to a process pool + weighted governor

**Lane:** A3 (ISL). **Base:** `origin/staging` @ `1289365e` (includes the merged+deployed
F8 cost model #80). Built in a **fresh blobless clone** off `origin/staging` (local tree was
stale @ `1df78a5` with a hung fetch — the documented drift). **Scope:** F15 only (F8 already
shipped). **Author:** A3 build agent. **Date:** 2026-07-18.

**⚖️ PAUL RULING obeyed:** NO env kill-switch. The offload is **UNCONDITIONAL** and made safe by
**self-healing** (not a rollback flag). Tuning knobs are **hardcoded constants**, not env vars.

**This is the wave's highest-risk change — it alters the execution model of EVERY analysis
request.** Because it ships unconditional, every failure mode below is tested with a positive
control + mutation-check.

---

## What shipped (F15)

| File | Change |
|---|---|
| `src/services/robustness_worker.py` (new) | Module-level `run_robustness_v2(request_json)->str`: `RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate_json(json)).model_dump_json()`. Picklable, stateless, spawn-safe (imports NOT via `src.api.main`). |
| `src/services/compute_governor.py` (new) | `ComputeGovernor` (asyncio semaphore + bounded queue + weighted tokens) + `Overload`. Hardcoded `ANALYSIS_WORKERS=min(2,cpu)`, `ANALYSIS_QUEUE_MAX=2*W`. Weighted tokens = **F8 `cost_units`** (imports `get_max_cost_units` — one source of truth, no second cost model). |
| `src/services/analysis_pool.py` (new) | Pool lifecycle + `run_offloaded()`: the `run_in_executor` hop, `wait_for` hard deadline, **hard-kill** (SIGKILL workers → free CPU → recreate pool), and **self-healing** (`BrokenProcessPool`/pool-loss → recreate + in-process fallback). |
| `src/api/main.py` | Lifespan creates/shuts down `app.state.analysis_pool` (`ProcessPoolExecutor`) + `app.state.governor`; construction failure → `pool=None` (in-process fallback, never bricks). |
| `src/api/robustness.py` | Both handlers (v2-enhanced ~:606 + v1-legacy ~:410) route the analysis through `governor.admit(cost.total, api_key)` → `run_offloaded(...)`, **unconditionally**. Typed 429/503 + `Retry-After` on overload, 504 on hard deadline (both paths return `JSONResponse` inline so the global `HTTPException` handler cannot strip the `Retry-After` header). `_ensure_governor` lazily provides a governor when the lifespan didn't run. |
| `openapi.json` | Regenerated: adds 429/503/504 + `ErrorResponse` schema to `/analyze/v2` (311 insertions, no deletions; `--check` passes). |

---

## Process model (workers, queue, fork/spawn)

- **Executor:** `ProcessPoolExecutor(max_workers=ANALYSIS_WORKERS, max_tasks_per_child=1)`.
  `max_tasks_per_child=1` recycles a worker after each analysis → bounds worker memory (each
  holds the graph + up to ~100k sample floats), the design's top rollback trigger. Cost: a
  per-request worker (re)spawn — negligible under `fork` (Render), and under `spawn` (macOS
  local) dwarfed by the tens-of-seconds compute.
- **Governor:** `asyncio.Semaphore(W)` caps concurrent execution at W; total **admitted**
  (executing + queued) is capped at `W + QUEUE_MAX`. Three gates, all TESTED:
  1. total-admitted ≥ bound → **503** `service_busy_queue_full`
  2. per-caller admitted ≥ W → **429** `caller_concurrency_exceeded`
  3. Σ admitted `cost_units` + new > budget → **503** `service_busy_cost_budget`
     (budget = `W × get_max_cost_units()`, F8 single source of truth).
- **fork/spawn:** local default is **spawn** (macOS) — all tests pass under it. Render/Linux
  default is **fork**. The worker fn is module-level + JSON-in/JSON-out and its import chain is
  free of `src.api.main`, so it is safe under both. **Caveat:** only spawn is exercised locally;
  fork behaviour (inherited fds/locks) is not — but fork re-uses the same picklable-by-reference
  entrypoint and no locks are held across the boundary, so the risk is low. Worth a staging
  smoke on the offload path (Render is fork).

`ANALYSIS_WORKERS` on this dev box = **2** (cpu_count 10 → min(2,10)). On a 1-CPU Render
instance it resolves to **1** — still the point (CPU off the event loop).

## How the hard-kill backstop actually terminates a running worker

A `ProcessPoolExecutor` **cannot cancel a running task** — `asyncio.wait_for` timing out only
stops *awaiting*; the worker keeps burning a CPU. On the hard deadline
(`ANALYSIS_HARD_DEADLINE_S = 80.0`, above the analyzer's 50s cooperative budget, below the 90s
`/api/v1/robustness/` route-timeout middleware) `run_offloaded` calls
`_terminate_pool_processes(pool)` → **`os.kill(pid, SIGKILL)`** on every worker process (pids read
from `ProcessPoolExecutor._processes`, a private map stable in CPython 3.11, `getattr`-guarded so
a shape change degrades to `shutdown()` rather than raising), then installs a fresh pool. SIGKILL
of the process is the only real preemption of a GIL-bound loop.

Killing the workers breaks the whole executor, so a **sibling** in-flight future fails with
`BrokenProcessPool` and self-heals to the in-process path — terminating the pool is therefore
safe even when a sibling analysis is running.

---

## Per-PC evidence (RED-first positive control + mutation-check)

Run: `ISL_AUTH_DISABLED=true poetry run pytest tests/unit/test_f15_offload_governor.py
tests/integration/test_f15_responsiveness.py` → **14 passed**.

### PC-A — event-loop responsiveness (the headline) + positive control (trap #13)
`tests/integration/test_f15_responsiveness.py`. Same heavy request (20-node DAG, 10k samples,
~2.3M cost units < 24M ceiling, ~1.5s in-process); a 20ms heartbeat measures the max event-loop
stall.
- **POSITIVE CONTROL (offload OFF, `analysis_pool=None`):** max loop stall **1856 ms** — the loop
  is frozen for the whole analysis. This PROVES the blocking F15 removes is visible (an absence
  test that never saw a presence is vacuous). `test_positive_control_in_process_blocks_the_loop`
  asserts `> 400ms`.
- **THE FIX (offload ON, W=1):** max loop stall **36 ms**, and **/health served in 13 ms** (200)
  while the heavy analysis ran in the worker. `test_offload_keeps_event_loop_responsive` asserts
  stall `< 200ms` AND /health `< 100ms`.
- **MUTATION-CHECK (throwaway, reverted):** revert the `run_in_executor` hop to an in-process call
  → offload test RED: *"event loop stalled 1680ms with offload on — expected < 200ms"*. ✅

### PC-B — bounded queue / early-typed 503 (governor + wire, both handler paths)
- Governor unit (`test_pc_b_bounded_queue_last_request_rejected_503`): W=1, QUEUE_MAX=2 (bound 3);
  3 held admits (1 executing + 2 queued), `gov.inflight == 1` (**in-flight never exceeds W**), the
  **4th admit raises `Overload(503, service_busy_queue_full)` immediately** with `retry_after > 0`
  (not a hung connection). Releases cleanly → `admitted==0` (no slot leak).
- Wire, v2-enhanced + v1-legacy (`TestPCBWireOverload`): a saturated governor → real
  `POST /analyze/v2` returns **503 + `Retry-After`** in `< 1s` (not hung); v2 body is the typed
  `{"code":"ISL_SERVICE_UNAVAILABLE","retryable":true}`.
- Also tested: **per-caller 429** (`test_per_caller_concurrency_429`, positive control that a
  different key is not rejected), **weighted cost-budget 503** (`test_weighted_cost_budget_503`,
  positive control that the gate binds and that a fitting job is admitted), and **slot release on
  body exception** (`test_slot_released_on_body_exception`).

### PC-C — hard deadline actually stops the work (worker terminated)
- Mechanism (`test_terminate_pool_processes_kills_running_worker`): a running `sleep_worker`; its
  pids are **proven ALIVE (positive control)**, then `_terminate_pool_processes` SIGKILLs them and
  each pid is **gone within a 4s bound**.
- Wire (`test_run_offloaded_deadline_raises_and_recreates_pool`): a worker that sleeps past a 0.75s
  monkeypatched deadline → `run_offloaded` raises `AnalysisDeadlineExceeded`, the terminated worker
  pids are **gone**, `app.state.analysis_pool` is a **new object** (recreated), and the recreated
  pool serves the next request.

### PC-D — determinism byte-parity (guards the refactor fold, trap #5)
`test_offloaded_response_byte_identical_to_in_process`: fixed seed, all optional phases on;
offloaded vs in-process. A deep-diff proves the **only** pre-normalisation difference is
`metadata.execution_time_ms` (a wall-clock field) — every science byte (`seed_used`,
`config_fingerprint`, `edge_existence_rates`, all `results`) is identical; the diff-finder would
flag a second path if anything drifted (positive control on the comparison itself).

### PC-E — pickle/JSON round-trip, no field loss
`test_every_optional_phase_and_constraints_survive_round_trip`: a request with EVERY optional
phase (e-values, EVPI + parameter_uncertainties, path decomposition) + `goal_constraints` +
`goal_threshold` survives `model_dump_json → model_validate_json` unchanged, and the worker
returns a response carrying every requested phase (`edge_e_values`, `factor_evpi`,
`path_decomposition` all present).

### PC-F — crash resilience / self-healing (replaces the kill-switch — PAUL RULING addition)
`test_worker_crash_falls_back_in_process_with_correct_science`: the offloaded worker `os._exit(70)`
(→ `BrokenProcessPool`). `run_offloaded` **(1)** returns byte-correct science (equal to the
in-process reference), **(2)** logs `isl_analysis_offload_degraded` with `reason=broken_process_pool`
(paths/reason only, no values), **(3)** recreates the pool (new object), **(4)** serves the NEXT
request correctly via the recreated pool.
- Also `test_missing_pool_runs_in_process_not_flagged_degraded`: a never-configured pool runs
  in-process with correct science and is NOT mislabelled a degradation (keeps the alarm meaningful
  — trap #14).
- **MUTATION-CHECK (throwaway, reverted):** break the self-heal (`except BrokenProcessPool: raise`)
  → PC-F RED: `concurrent.futures.process.BrokenProcessPool` propagates instead of the in-process
  fallback. ✅

---

## Gate results

- **mypy** `poetry run mypy src/`: **clean** (137 files).
- **black** `--check src/ tests/...`: **clean** (after formatting the new files, line-length 100).
- **openapi** `generate_openapi.py --check`: **OK, up to date** (base was drift-free; regen adds
  only 429/503/504 + ErrorResponse, 311 insertions, no deletions).
- **F15 suite:** 14 passed.
- **Full suite** (`pytest tests/ --ignore=tests/_archived -m "not perf"`): SEE the appended
  results block below (the pre-existing `test_metadata_populated` timing flake is the only
  tolerated failure).

---

## Memory / robustness caveats (brutally honest)

- **Worker memory:** each worker holds the graph + up to ~100k sample floats (~1-2 MB JSON crosses
  the pipe once/request). `max_tasks_per_child=1` reclaims it per task. On a 1-CPU Render box W=1,
  so at most one worker's footprint at a time.
- **Pool-recreate cost:** a hard-kill or crash recreates the whole pool (all workers). With W≤2
  and `max_tasks_per_child=1` this is cheap-ish, but a crash storm would churn pools; acceptable
  because each event is per-request and the in-process fallback always returns.
- **Whole-pool hard-kill disrupts a sibling:** on a hard deadline ALL workers are killed (not just
  the offender), so a concurrent sibling's future breaks — but it self-heals to in-process, so it
  still returns correct science (slower). Deliberate tradeoff: surgical per-future kill needs
  fragile pid↔future tracking; whole-pool kill + self-heal is simpler and provably safe.
- **In-process fallback blocks the loop** for that one request's duration (the pre-F15 behaviour).
  Rare (only on crash / pool-loss / no-pool), bounded, and the correct-science tradeoff.
- **Private `_processes` dependency** for the hard-kill (guarded, CPython-3.11-stable). If a future
  CPython changes it, hard-kill degrades to `shutdown()` (no CPU-freeing termination) but nothing
  else breaks.
- **`ANALYSIS_HARD_DEADLINE_S = 80s`** is a hardcoded guess: above the 50s cooperative budget,
  below the 90s route middleware. On a very slow Render box a legitimate ~50s job stays well under
  it; a pathological base-MC could in theory approach it — admission (24M ceiling) is the primary
  bound on job size.

## Not covered / residual risk

- **fork path not exercised locally** (only spawn). Production is fork — worth a staging smoke of
  the offload path.
- **No live-staging soak.** This PR is UNCONDITIONAL and active on staging at merge. **Production
  is a SEPARATE, Paul-confirmed `main` push later** (there is no env rollback; prod rollback =
  revert+redeploy, acceptable only because the in-process fallback degrades gracefully and F15
  must prove itself on staging first).
- **Weighted cost-budget gate (3) does not bind at the real ceiling** (`W × max_cost`); it is real,
  tested plumbing that binds only if the ceiling or W changes. This is intentional (binding it
  would spuriously reject legitimately-admitted jobs) and documented in-code.
