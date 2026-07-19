# /simplify — ISL cleanup build (A3 lane)

Behavior-preserving refactors from the /simplify pass (see
`SIMPLIFY-CONSOLIDATED-2026-07-19.md`, ISL cleanup section).

- **Base:** `staging` @ `cdacb2ff` (fresh blobless clone; local tree stale/hung-fetch).
- **Branch:** `refactor/isl-simplify-cleanups-a3`
- **Discipline:** behavior-preserving; the existing #79/#80/#81 tests passing IS the
  behavior-preservation proof for items 1–4. No `--no-verify`. Draft PR only (orchestrator
  reviews + merges).

## Gate result (authoritative)

| Check | Command | Result |
|-------|---------|--------|
| mypy | `poetry run mypy src/` | **Success: no issues found in 137 source files** |
| black | `poetry run black --check src/` (+ edited tests) | **clean (137 unchanged)** |
| pytest (full) | `ISL_AUTH_DISABLED=true poetry run pytest tests/ --ignore=tests/_archived -q -m "not perf"` | **1947 passed, 675 skipped, 4 deselected, 0 failed** (90s) |
| OpenAPI | `poetry run python scripts/generate_openapi.py --check` | **OK: openapi.json is up to date** (no schema drift — items 3/4 changed no API shape) |

The `test_metadata_populated` timing flake did not trip (0 failed).

## Per-item summary

### 1. PhaseDeadline — dedup EVPI/path deadline scaffolding (`robustness_analyzer_v2.py`)
Added a module-level `PhaseDeadline(budget_ms)` value object (`__slots__ = (budget_ms, t0)`):
`exceeded()` = `budget_ms is not None and (monotonic()-t0)*1000.0 > budget_ms`;
`elapsed_ms()` = `round((monotonic()-t0)*1000.0, 1)` — byte-identical to the inline forms.
- `_compute_path_decomposition`: `t0 = time.monotonic()` → `deadline = PhaseDeadline(budget_ms)`;
  walk() predicate, pre-enumeration bail, and 2 elapsed recomputes collapsed. The
  `PATH_DEADLINE_CHECK_INTERVAL` modulo cadence stays LOCAL at the walk() re-check.
- `_compute_evpi`: same; removed the nested `_deadline_passed()` closure; 4 elapsed recomputes →
  `deadline.elapsed_ms()`; threads the SAME `deadline` (shared phase t0) into `_compute_evpi_metric`.
- `_compute_evpi_metric`: signature `deadline_t0/budget_ms` → `deadline: Optional[PhaseDeadline]`;
  predicate → `... and deadline.exceeded()` (EVPI_DEADLINE_CHECK_INTERVAL modulo stays local).
- `analyze()`: 8 `round((monotonic()-budget_start)*1000.0, 1)` disclosure recomputes → a local
  `_elapsed_ms()` closure (same anchor, same rounding). `budget_start` / `_budget_remaining_ms`
  (load-bearing budget math) left untouched.
- The `_optional_phase_unavailable_warning` disclosure was NOT touched. E-value / flip-stability /
  factor-bootstrap deadline blocks (their own `t0`/`budget`, `*1000` not `*1000.0`, some using
  `time.time()`) are DIFFERENT methods, out of the named scope — left as-is.
- Proof: `tests/unit/test_evpi_path_deadline.py` (F7 internal-trip pins) green; full suite green.

### 2. Caps single-source (`request_validators.py` ← `src/constants`)
`RequestSizeLimits` (and the module-level `MAX_OPTIONS`) now import `MAX_GRAPH_NODES` /
`MAX_GRAPH_EDGES` / `MAX_OPTIONS` from `src/constants` (the designated single source — its docstring
says so, and it is already imported by the request model, the analyzer, and /health). Values
unchanged (50 / 200 / 10). `MAX_CRITERIA`/`MAX_PARAMETERS`/`MAX_SWEEP_POINTS` (not in `src/constants`)
stay local. `MAX_PARAMETER_UNCERTAINTIES=50` untouched. No caller changed value: `RequestSizeLimits`
and the module caps are used only internally by this module's validators (grep-confirmed).

### 3. `admit_and_run` — shared helper + normalize legacy 422 (`robustness.py`)
Extracted TWO helpers both v2 handlers call (the cost→422 and the admit block are NOT contiguous in
the enhanced handler — validation runs between them — so folding into one helper would reorder
cost-422-vs-validation-422 precedence; two helpers preserves ordering exactly):
- `_admission_cost_guard(request, request_id) -> (WeightedCost, Optional[JSONResponse])`: computes
  cost, logs `robustness_v2_complexity` (identical extras), returns a 422 JSONResponse (flat
  structured body + `X-Request-Id`) when over the ceiling.
- `_admit_and_run(app, request, request_id, api_key, cost) -> (Optional[RobustnessResponseV2], Optional[JSONResponse])`:
  the identical governor.admit + run_offloaded + `except Overload`(429/503)/`except AnalysisDeadlineExceeded`(504)
  → typed-JSONResponse dance. Exactly one return value is non-None; callers `assert` the response.
- **Normalization (the one intended behavior change):** the legacy (v1) cost-422 previously did
  `raise HTTPException(422, detail=_admission_error_body(...))`, which the app's custom HTTPException
  handler rebuilt into the Olumi Error Schema (the structured body **stringified into `message`**).
  It now returns the SAME flat structured body the enhanced handler serves. Legacy is off the live V5
  path — noted. **Caveat (honest):** `X-Request-Id` was already backstopped on BOTH paths by
  `TracingMiddleware` (tracing.py:177 stamps every response), so the fix's *observable* effect is the
  BODY-SHAPE normalization, not the header presence. The added test asserts both (header + flat body);
  the flat-body assertion is the real discriminator (`data["cost_units"]` KeyErrors on the old wrapped shape).
- Test added: `test_admission_422_preserves_x_request_id_both_handlers` (both handlers: 422 +
  `X-Request-Id` echoed + top-level `cost_units`/`limit`). `test_ten_option_evpi_legacy_endpoint_returns_422`
  docstring + assertions updated to the normalized flat body.

### 4. Precompute static /health block (`health.py`)
`_STATIC_COMPUTE_ADMISSION` computed once at import as `build_compute_admission()` minus
`max_cost_units` (derived from the single source, so it cannot drift). New `_compute_admission_info()`
re-resolves only `max_cost_units` (env) per poll. Response byte-identical. `test_health_endpoint.py` green.

### 5. Governor gate-3 docstring — VERIFIED FALSE, CORRECTED (`compute_governor.py`, docstring only)
**Verdict: the "never binds tighter than gate 1" claim is FALSE. Gate 3 CAN bind first.**
Gate 1 admits up to `workers + queue_max` (= `3 × workers` with the defaults) jobs; each admitted job
is ≤ `max_cost` (F8 endpoint guard), so admitted cost can reach `3 × workers × max_cost` — but gate 3's
real-ceiling budget is only `workers × max_cost` (one third). For a burst of near-`max_cost` jobs,
gate 3 caps admitted at ≈`workers` (the `(workers+1)`-th max-cost job trips
`service_busy_cost_budget`) while gate 1 would still allow `3 × workers`. Gate 1 dominates only for
cheap jobs (mean cost ≲ `max_cost/3`). The test's own positive control
(`test_weighted_cost_budget_503`: with `max_cost=100`, admitted=1, gate-1 bound=8, a 30-unit job
503s) already demonstrates gate 3 binding first. **Gate code NOT changed** (correctness question, out
of /simplify scope). Docstring corrected; the same false claim echoed in the test docstring
(test_f15_offload_governor.py:202) also corrected (docstring only, non-behavioral).

## Not done / flagged
- Items A1/E1/E2 from the consolidated doc (caps-gate handshake, offload cost-floor,
  max_tasks_per_child) are FLAG-to-Paul behavior changes — NOT in this /simplify build, correctly.
- Gate-3 code left unchanged (item 5 was VERIFY-then-fix-docstring only).
- Nothing turned out behavior-changing beyond the intended item-3 legacy-422 body normalization
  (flagged above with the TracingMiddleware caveat).
