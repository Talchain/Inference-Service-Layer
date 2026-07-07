# Lane 24 — same-seed response determinism fixes

Branch: `claude-lane24/response-determinism` (from `origin/staging` @ `fea0b3ea`)
Source of the mandate: merged science-validation report,
`docs/science-validation/REPORT.md` §3 (volatile-field catalogue +
cross-process finding) and §5 item 7 (recommended src changes), queued from
ISL PR #66.

## What was broken

The report's exp3 (50 diverse seeded graphs, three comparison modes) found
same-seed responses are **not** byte-identical as shipped:

1. **`critiques[].id` was `uuid.uuid4()` per run** —
   `src/models/critique.py:34` (and a second site for Pydantic 422 errors,
   `src/api/main.py:595`). Any critique-bearing response (34/50 exp3 graphs)
   could never repeat byte-identically, even though the critique *content*
   is fully deterministic.
2. **`robustness.fragile_edges` / `robustness.robust_edges` were
   materialised from sets** (`src/services/robustness_analyzer_v2.py`,
   pre-fix lines 2804–2805: `list(fragile_edge_ids)` /
   `list(robust_edge_ids)` over sets built above). Set iteration order
   follows the per-process string-hash salt, so cross-process same-seed
   responses differed on 44/50 exp3 graphs — and
   **`robustness.interpretation` cites the first three entries** of the
   fragile list ("… sensitive to: …"), i.e. a process-dependent arbitrary
   subset of the fragile edges.

Pre-fix RED evidence reproduced in this lane (unmodified src):

```
# identical build() inputs, two calls:
RED critique ids equal? False critique_2c6dc7f4 critique_fb492e87

# same request/seed, PYTHONHASHSEED=0 vs 1:
fragile: ['n_mkt->n_dem', 'n_base->n_rev', 'n_dem->n_rev']   # HS=0
fragile: ['n_mkt->n_dem', 'n_dem->n_rev', 'n_base->n_rev']   # HS=1
robust:  ['n_qual->n_flat', 'n_churn->n_rev', 'n_qual->n_churn', 'n_mkt->n_rev']  # HS=0
robust:  ['n_mkt->n_rev', 'n_qual->n_churn', 'n_qual->n_flat', 'n_churn->n_rev']  # HS=1
```

The new test module fails against pre-fix src (verified via `git stash` of
the three src files: collection ImportError on the not-yet-existing helper;
the behavioural failures above are the underlying defects).

## What changed

1. `src/models/critique.py`
   - New `deterministic_critique_id(code, message, affected_option_ids,
     affected_node_ids, seed)` — `critique_` + first 12 hex chars of
     SHA-256 over the joined inputs.
   - `CritiqueDefinition.build()` now uses it and gains an optional
     `seed` kwarg (id input only, not part of the payload). `import uuid`
     removed.
   - **Deliberate deviation from §5.7a's literal input list** (hash of
     "(seed, code, affected ids)"): the formatted **message** is included
     in the hash. Reason: `EDGE_STRENGTH_OUT_OF_RANGE` (and others) carry
     their distinguishing detail only in template vars — two such critiques
     in one response share code and (empty) affected-id lists, so the
     literal recipe would give them **colliding ids**. The message is
     itself deterministic (template + deterministic vars), so byte-stability
     is preserved.
2. `src/services/robustness_analyzer_v2.py`
   - `fragile_edges = sorted(fragile_edge_ids)` /
     `robust_edges = sorted(robust_edge_ids)` (§5.7b) — also canonicalises
     the interpretation string's "sensitive to:" head.
   - The three in-analyzer critique sites (`CONSTRAINT_NODE_DEFAULT_BASE`,
     `DEGENERATE_OPTION_ZERO_VARIANCE`, `HIGH_TIE_RATE`) pass `seed=seed`
     per §5.7a. Other build sites (request_validator, degenerate_detector,
     numerical_stability, response_builder) have no seed in scope; their
     ids derive from content alone, which is still fully deterministic.
3. `src/api/main.py` (`_build_v2_pydantic_error_response`)
   - The 422 `VALIDATION_ERROR` critique id is now
     `deterministic_critique_id(code, message)` instead of uuid4 (no seed
     exists for an unparseable request).
4. `tests/unit/test_response_determinism.py` (new)
   - Deterministic-id unit tests (same inputs → same id; content and seed
     both vary the id).
   - Same-seed two-run stability of critique ids, edge lists, and
     interpretation; canonical-sortedness of both edge lists (fixture
     yields 3 fragile + 4 robust edges + 1 critique).
   - **Cross-process byte-stability**: two child interpreters with
     `PYTHONHASHSEED=0` vs `1` analyze the identical request; full JSON
     (only `_metadata.execution_time_ms` zeroed) must be byte-identical.
     This is the exp3 cross-process scenario as a regression test.

## Volatile fields NOT changed (with reasons)

Report §3 catalogued four volatile fields. The other two-plus:

- `metadata.execution_time_ms` / envelope `processing_time_ms` /
  envelope `timestamp` — genuine wall-clock measurements, not determinism
  bugs; zeroing or freezing them would destroy their meaning. Consumers
  comparing responses must mask them (as exp3 did).
- `request_id` when not client-pinned (`robustness-{uuid4}` /
  `isl-{uuid4}`) — volatile **by design**; clients wanting stable ids pin
  them.
- §5.7c (higher-K `k_samples` threading) is a Slice-3 doctrine decision,
  not a determinism fix — untouched.
- `list(set(...))` at `robustness_analyzer_v2.py:1192/3175`
  (`all_target_nodes`) — internal iteration only; exp3 proved the response
  is cross-process identical once the two edge lists are sorted, so these
  do not leak into the wire. Left alone to stay within the report's
  mandate.

## Test proof

- `tests/unit/test_response_determinism.py`: **8 passed** (post-fix);
  pre-fix fails (stash-verified).
- Related modules: `test_response_v2.py`, `test_robustness_v2.py`,
  `test_p2_workstream.py`, `test_cil_phase0_fixes.py`,
  `test_alternative_winner.py` + new module: **381 passed**.
- 422/wire contract modules (`test_observability_middleware.py`,
  `test_v2_wire_completeness.py`, `test_error_schema_contract.py`,
  `test_tier0_regression_gate.py`, `test_seed_used_serialisation.py`,
  `test_p2_verification.py`): **99 passed, 30 skipped**.
- `poetry run mypy src/`: clean (134 files).
- `poetry run black --check`: clean on all touched files.
- `scripts/pre-push-validate.sh`: see PR/commit status (run before push).

## Residual risks

- **Duplicate identical critiques would now share an id** (uuid4 made them
  distinct). No current call site emits two critiques with identical code +
  message + affected ids in one response; if one ever does, the duplicates
  are arguably the same critique. No downstream consumer keys on critique
  id uniqueness (grep: no `critique.id` consumers in src).
- Critique ids are now **stable across responses** for identical content
  (that is the point). Any consumer assuming ids are globally unique
  nonces would be misled — none known.
- The id format widened from 8 to 12 hex chars (`critique_` prefix
  unchanged, matches existing `startswith("critique_")` assertions and the
  free-string `CritiqueV2.id` field).
- The cross-process test spawns two child interpreters (~2 s each); if CI
  sandboxing ever blocks `subprocess`, that one test would need a skip
  marker — the in-process tests still cover the id fix and sortedness.
- Robust/fragile ordering is now canonical (lexicographic), which may
  differ from any (never-guaranteed) order a consumer eyeballed before.
