# A3 ISL BUILD — Codex F7 (EVPI/path internal deadline) + F4 (typed warning severity + field paths)

Base: `origin/staging` @ `933c3404` (PR #78). Branch: `a3/evpi-path-deadline-warning-severity`.
Repo built in a fresh blobless clone (the local working tree's `origin/staging` cache was
175-ish commits stale and a whole-repo `git fetch` hung on the local pack — clone sidesteps it).
Tests/gate driven by the existing project venv
(`.../inference-service-layer-ypaMgHbQ-py3.11`) invoked directly against the clone (venv is
path-keyed; `cd clone && <venv>/bin/python -m ...` resolves `src` from the clone — verified).

## What was byte-confirmed on staging BEFORE the change

- ONE `InferenceWarning` class only: defined in `src/models/response_v2.py:23-44`
  (`code`, `field`, `detail` — **no `severity`**), re-exported through
  `src/models/robustness_v2.py:21-26` and imported by the analyzer from there. So a single
  model edit covers analyzer + internal `RobustnessResponseV2` + envelope `ISLResponseV2`.
- Envelope serialises via `response.model_dump(by_alias=True, exclude_none=True)`
  (`src/api/robustness.py:1047`). A `severity` default of `"warning"` is non-None so it always
  rides the wire; older consumers ignore it (`extra="ignore"` on every V2 model) — additive.
- `factor_evpi` and `path_decomposition` are **top-level** on `ISLResponseV2`
  (`response_v2.py:905` / `:914`), but the two entry-skip warnings named them
  `robustness.factor_evpi` / `robustness.path_decomposition` (analyzer `:1353` / `:1381`) — F4
  field-path defect.
- `_compute_evpi` (`:3807`), `_compute_evpi_metric` (`:3946`) and `_compute_path_decomposition`
  (`:3128`) took NO deadline and never re-checked `time.monotonic()` in their loops — EVPI is
  entry-gated at `EVPI_MIN_BUDGET_MS=8000` only; path-decomp is entry-gated + 20000-path-count
  capped only. Confirmed the E-value sweep (`_compute_edge_e_values` `:3473`,`:3510`) and band
  sweep (`_attach_flip_stability_bands` `:3689`) DO re-check mid-loop against
  `min(PHASE_CAP, remaining_ms)` and return None / disclose — that is the mechanism reused.
- `FactorSampler.has_uncertainties()` ⟺ `parameter_uncertainties` non-empty
  (`_uncertainty_map = {u.node_id: u for u in (uncertainties or [])}`). The dispatch calls
  `_compute_evpi` only under `has_uncertainties()`, so `_compute_evpi`'s benign
  no-uncertainties `None` is unreachable there — an in-dispatch `None` is unambiguously a
  deadline trip.
- Established internal-trip test pattern (mirrored): `tests/unit/test_request_budget.py`
  trips the E-value / band internal budgets by monkeypatching the phase constant to `-1`
  (`E_VALUE_BUDGET_MS = -1`, `FLIP_STABILITY_BUDGET_MS = -1`) and asserts the `*_UNAVAILABLE`
  code + `elapsed_ms` ride the wire. Positive control = full-budget run where every optional
  phase is present and no `*_UNAVAILABLE` fires.

## Design (reuse, do not invent)

- New class attrs `EVPI_BUDGET_MS = OVERALL_REQUEST_BUDGET_MS` and
  `PATH_DECOMPOSITION_BUDGET_MS = OVERALL_REQUEST_BUDGET_MS` (mirroring
  `E_VALUE_BUDGET_MS`/`FLIP_STABILITY_BUDGET_MS`). Default == governing budget so
  `min(cap, remaining) == remaining`: the phase is bounded ONLY by the governing request
  deadline, never cut tighter than it — no new false cuts of currently-succeeding runs. They
  exist as the phase knob + the `-1` trip pin tests drive.
- Dispatch passes `budget_ms = min(self.<CAP>, remaining_ms)`; each phase anchors its own
  monotonic `t0`, so `(monotonic()-t0)*1000 > budget_ms` == the OVERALL deadline passed
  (identical maths to the E-value sweep).
- Overrun => return `None` (all-or-nothing, discard partial) => dispatch discloses
  `EVPI_UNAVAILABLE` / `PATH_DECOMPOSITION_UNAVAILABLE` via the existing
  `_optional_phase_unavailable_warning()` helper, carrying `elapsed_ms`, reason
  `evpi_budget_exceeded` / `path_decomposition_budget_exceeded`, corrected top-level field.
- F4: `severity: Literal["info","warning","error"]` added to `InferenceWarning`,
  `default="warning"`; helper stamps `severity="warning"` on all four degradation codes;
  the two entry-skip field paths corrected to top-level `factor_evpi` / `path_decomposition`.

## Key hunks

`src/models/response_v2.py` — new typed `severity` on the single shared
`InferenceWarning` (re-exported through `robustness_v2.py`, so this one field covers
analyzer + internal `RobustnessResponseV2` + envelope `ISLResponseV2`):

```python
    severity: Literal["info", "warning", "error"] = Field(
        default="warning",
        description="Severity for downstream routing/display. Defaults to 'warning'.",
    )
```

`src/services/robustness_analyzer_v2.py`:
- `EVPI_BUDGET_MS = OVERALL_REQUEST_BUDGET_MS`, `PATH_DECOMPOSITION_BUDGET_MS =
  OVERALL_REQUEST_BUDGET_MS`, `EVPI_DEADLINE_CHECK_INTERVAL = 64`,
  `PATH_DEADLINE_CHECK_INTERVAL = 512` (new class attrs, near `EVPI_MIN_BUDGET_MS`).
- `_optional_phase_unavailable_warning(..., severity="warning")` stamps severity.
- analyze() dispatch: `budget_ms=min(self.EVPI_BUDGET_MS, remaining_ms)` /
  `budget_ms=min(self.PATH_DECOMPOSITION_BUDGET_MS, remaining_ms)`, plus `if
  factor_evpi is None:` / `if path_decomposition is None:` disclosure branches; the
  two ENTRY-skip field paths corrected `robustness.factor_evpi` -> `factor_evpi`,
  `robustness.path_decomposition` -> `path_decomposition` (F4).
- `_compute_evpi(..., budget_ms=None)`: monotonic `t0`; deadline re-check before the
  baseline, after the baseline metric (None => trip), and at the top of each factor
  loop; returns None all-or-nothing.
- `_compute_evpi_metric(..., deadline_t0=None, budget_ms=None) -> Optional[float]`:
  periodic sample-loop re-check (`i % EVPI_DEADLINE_CHECK_INTERVAL == 0`), return None
  on overrun.
- `_compute_path_decomposition(..., budget_ms=None) -> Optional[PathDecomposition]`:
  monotonic `t0`; walk-internal periodic re-check (`walk_calls %
  PATH_DEADLINE_CHECK_INTERVAL == 0` -> `deadline_hit`), a pre-enumeration check, and
  a post-enumeration `deadline_hit` -> return None. `budget_ms=None` (the direct/legacy
  test callers) disables the guard — those callers still receive a `PathDecomposition`.

The EVPI metric sample-loop guard (representative):

```python
        for i in range(n_samples):
            if (
                budget_ms is not None
                and deadline_t0 is not None
                and i % self.EVPI_DEADLINE_CHECK_INTERVAL == 0
                and (time.monotonic() - deadline_t0) * 1000.0 > budget_ms
            ):
                return None
```

## Baseline (clean staging tip 933c3404)

- `poetry` venv (`inference-service-layer-ypaMgHbQ-py3.11`) driven directly:
  `mypy src/` → **Success: no issues found in 134 source files**.
- `pytest tests/unit/test_request_budget.py test_path_decomposition.py test_response_v2.py`
  → **92 passed**.

## RED-first + positive control (new file `tests/unit/test_evpi_path_deadline.py`, 8 tests)

Run against the **pristine base 933c3404** (throwaway `--detach` worktree, new test file
dropped in):

```
2 passed, 6 failed  (== RED-first)
PASSED  test_ample_budget_phases_present_no_unavailability   <- POSITIVE CONTROL
PASSED  test_deadline_guard_is_inert_when_not_tripped        <- POSITIVE CONTROL
FAILED  test_evpi_internal_trip_all_or_nothing_and_discloses      AttributeError: no attribute 'EVPI_BUDGET_MS'
FAILED  test_evpi_disclosure_severity_and_top_level_field         AttributeError: no attribute 'EVPI_BUDGET_MS'
FAILED  test_path_internal_trip_all_or_nothing_and_discloses      AttributeError: no attribute 'PATH_DECOMPOSITION_BUDGET_MS'
FAILED  test_path_disclosure_severity_and_top_level_field         AttributeError: no attribute 'PATH_DECOMPOSITION_BUDGET_MS'
FAILED  test_entry_skip_fields_top_level_and_severity            assert 'robustness.factor_evpi' == 'factor_evpi'
FAILED  test_severity_serialises_by_alias_exclude_none           AttributeError: 'InferenceWarning' object has no attribute 'severity'
```

The two positive controls PASS on base — the fixture provably produces EVPI rows + a
path decomposition when NOT squeezed, so the "None / no rows" absence assertions can see a
presence (trap-13). On the FIXED tree all 8 PASS.

## Gate

- `mypy src/` (fixed tree) → **Success: no issues found in 134 source files**.
- `black --check` changed files → clean.
- Full suite as the pre-push gate runs it (`ISL_AUTH_DISABLED=true pytest tests/
  --ignore=tests/_archived --ignore=tests/benchmarks -m "not perf"`), coverage dropped
  for speed:
  - **FIXED tree: 1 failed, 1910 passed, 675 skipped** (24s).
  - **PRISTINE base (no change, no new test): 1 failed, 1902 passed, 675 skipped** (22s).
  - The single failure is the SAME test on BOTH —
    `test_robustness_v2.py::TestRobustnessAnalyzerV2::test_metadata_populated` — a
    **pre-existing timing flake**: it asserts `metadata.execution_time_ms > 0` where
    `execution_time_ms = int((time.time()-start)*1000)`, which rounds to 0 when the
    trivial `simple_request` analysis completes in <1 ms under full-suite CPU contention.
    Passes in isolation on both base and fixed. Those timing lines (analyzer :824, :1393)
    are UNTOUCHED by this change. Net effect of the change: **+8 passing tests
    (1910−1902), zero new failures.**

## Mutation-check (throwaway `--detach` worktree off the fix commit)

Reverted ONLY the deadline-threading hunk (the two analyze() dispatch args
`budget_ms=min(self.EVPI_BUDGET_MS, remaining_ms)` and
`budget_ms=min(self.PATH_DECOMPOSITION_BUDGET_MS, remaining_ms)` -> not passed, so
`budget_ms` defaults None and every internal re-check is skipped). Result:

```
4 failed, 4 passed  (DISCRIMINATING)
FAILED  test_evpi_internal_trip_all_or_nothing_and_discloses   <- went RED
FAILED  test_evpi_disclosure_severity_and_top_level_field      <- went RED
FAILED  test_path_internal_trip_all_or_nothing_and_discloses   <- went RED
FAILED  test_path_disclosure_severity_and_top_level_field      <- went RED
PASSED  test_ample_budget_phases_present_no_unavailability      (trip-independent)
PASSED  test_deadline_guard_is_inert_when_not_tripped           (trip-independent)
PASSED  test_entry_skip_fields_top_level_and_severity           (entry-gate path, F4 only)
PASSED  test_severity_serialises_by_alias_exclude_none          (model-only, F4)
```

The 4 internal-trip tests go RED with the threading reverted (mechanism is load-bearing);
the 4 trip-independent tests stay GREEN (the mutation is discriminating, not a blanket
break — the entry-skip F4 field/severity corrections and the additive severity default are
correctly independent of the internal deadline).

## Review delta — F4 severity MAPPING (coordinator, on top of the fix commit)

Review finding: defaulting `InferenceWarning.severity = "warning"` inverted the ~9 benign
input-adjustment/default codes (STRENGTH_MEAN_CLAMPED, CONSTRAINT_NODE_DEFAULT_BASE,
ROOT_NODE_DEFAULT_VALUE, ...) to "warning" — PLoT's `severity==='warning' ? 'warning' :
'info'` would surface them. Intent: ONLY the four degradation codes are "warning".

Surgical change (nothing else in #79 touched):
1. `response_v2.py` — `InferenceWarning.severity` default `"warning"` -> **`"info"`** (quiet
   default; benign diagnostics stay info).
2. `robustness_analyzer_v2.py` `_optional_phase_unavailable_warning()` — dropped the severity
   param and stamp **`severity="warning"` explicitly** in the one place the four degradation
   codes are built (entry-skip AND overrun paths both flow through it).
3. New pin `TestSeverityMappingDegradationVsBenign` — one budget-exhausted run over a graph
   with an out-of-range edge (`strength.mean=1000` -> STRENGTH_MEAN_CLAMPED) so all four
   degradation codes AND a benign code are present in one response; asserts (positive control)
   all five present, then the four are `severity=="warning"` and STRENGTH_MEAN_CLAMPED is
   `severity=="info"`, on both the model and the `model_dump(by_alias, exclude_none)` wire.
   The additive test was adjusted to the new quiet default.

RED-first (delta) — updated tests vs the pre-delta commit `3fcead6` (default still "warning"):

```
2 failed  (== RED-first for the delta)
FAILED  TestSeverityMappingDegradationVsBenign::...   assert 'warning' == 'info'   (STRENGTH_MEAN_CLAMPED)
FAILED  TestSeverityIsAdditiveOnWire::...             assert 'warning' == 'info'   (quiet default)
```
The positive control (all five codes present) and the four `=="warning"` assertions pass on
the pre-delta code — the failing assertion is exactly the benign-stays-info inversion.

Gate (delta, fixed tree): `mypy src/` clean (134 files); `black --check` clean; targeted
regression across every warning-emitting test file + the new file = **484 passed, 1 failed**
— the 1 is the same pre-existing `test_metadata_populated` timing flake (unrelated). New file
now **9 passed** (+1 vs the 8 before the delta).

Mutation-check (delta) — throwaway `--detach` worktree off the delta commit, reverted ONLY
the explicit `severity="warning"` in the helper (-> falls back to the model default "info"):

```
4 failed  (DISCRIMINATING — the explicit "warning" stamp is load-bearing)
FAILED  TestSeverityMappingDegradationVsBenign   assert 'info' == 'warning'   (the 4 codes)
FAILED  test_evpi_disclosure_severity_and_top_level_field   assert 'info' == 'warning'
FAILED  test_path_disclosure_severity_and_top_level_field   assert 'info' == 'warning'
FAILED  test_entry_skip_fields_top_level_and_severity        assert 'info' == 'warning'
```
With the helper's stamp reverted the four degradation codes fall back to the model default
'info' and every "is warning" assertion goes RED; the benign-stays-info half is unaffected
(it never depended on the helper). Load-bearing and correctly scoped.

## OpenAPI spec regeneration (CI "Validate OpenAPI Specification")

CI's `poetry run python scripts/generate_openapi.py --check` was RED (committed `openapi.json`
missing the new field). Regenerated the committed repo-root `openapi.json`.

Drift-isolation before committing (so no pre-existing spec drift is bundled): regenerated on
the **pristine base** `933c3404` first → **zero diff**, `--check` **OK** (base spec was already
in sync). Therefore the branch diff is attributable solely to this PR's only API-surface change,
`InferenceWarning.severity`.

Branch diff = **severity-only, +11 / -0 lines, 8 paths unchanged**: a single new `severity`
property on the `InferenceWarning` component schema —
`{default:"info", enum:["info","warning","error"], type:"string"}`, not in `required`. The only
other schema carrying `severity` is the pre-existing `CritiqueV2`; no other schema/path moved.
`generate_openapi.py --check` now exits **0** ("OK: openapi.json is up to date").

## Scope NOT covered (deliberate)

- PLoT's InferenceWarning mirror + UI severity templates (F4 consumer halves) — separate
  PRs per the brief; untouched here.
- The pre-existing `test_metadata_populated` timing flake is left as-is (out of scope,
  not introduced by this change).

