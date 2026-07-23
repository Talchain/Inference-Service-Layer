# A3 Codex-fix-B — Correlation hard-invalid rejection (F4)

**Lane:** A3 implementation — ISL
**Ruling:** D-23.13 (enforce existing doctrine D-23.4 + PARAMETER-RESEARCH-2026-07-23.md:49-59)
**Doctrine:** "reject hard-invalid, ELSE Higham-project near-PSD with disclosure". The
implementation currently Highams EVERYTHING (no hard-invalid threshold enforced).
**Branch:** `a3-codex-fix-correlation-hard-invalid`
**Clone:** `/private/tmp/.../scratchpad/isl-fixB` (fresh blobless, staging)
**COMMIT LOCALLY, DO NOT PUSH.**

## Setup verification
- Base = staging @ `4675a0a3b9ec1d7c2b75775684908c313c0d25a5` — **VERIFIED, matches, not moved.**
- Branch created off staging tip.
- Sibling lane isl-fixA edits `robustness_v2.py` `control_candidates` validator; my F4 hunk is
  in the correlation validator/matrix path — must keep hunks disjoint. (Overlap note TBD once I
  locate exact lines.)

## The defect (Codex F4 repro)
3 normal factors, rho(a,b)=1, rho(a,c)=1, rho(b,c)=−1 → eigenvalues [−1,2,2]
(logically contradictory: a=b, a=c ⟹ b=c, but rho(b,c)=−1) → currently ACCEPTED,
silently Higham-projected to effective off-diagonals ~±0.5 (Frobenius distance 1.2247,
max adjustment ~0.5); analysis completes under materially different assumptions.
Response discloses aggregate distance only, NOT the effective matrix.

## Fix plan (two parts)
1. Hard-invalid threshold → typed 422 BEFORE projection (versioned constant + fingerprint guard).
2. Disclose the EFFECTIVE adjusted matrix/pairs (additive-optional field on correlation_model).

---

## Progress log

### Setup DONE
- Fresh blobless clone `isl-fixB`, staging tip `4675a0a3b9ec1d7c2b75775684908c313c0d25a5` — matches base, NOT moved.
- Branch `a3-codex-fix-correlation-hard-invalid` created.
- Resolution hazard handled: venv's `inference_service_layer.pth` points at the MAIN repo
  (`/Users/paulslee/Documents/GitHub/Inference-Service-Layer`). `run.sh` pins `PYTHONPATH=<clone>`
  and ASSERTS `src.utils.correlation.__file__` / `robustness_v2.__file__` start with the clone
  before every run. Verified resolving to clone. venv = `inference-service-layer-ypaMgHbQ` (numpy 1.26.4,
  matches the seeded-value pins). Tests need `ISL_API_KEYS=test_key_for_ci`.
- PRISTINE baseline: `tests/unit/test_robustness_correlation.py` = **39 passed** (green).
- Disjointness vs sibling isl-fixA: fixA edits `validate_control_candidates` (robustness_v2.py L1038+);
  my hunks are `validate_factor_correlations` (L958-1035) + `correlation.py` + `response_v2.py`
  CorrelationProjectionV2 + analyzer `_build_correlation_disclosure`. NO overlap.
- `src/services/correlation_validator.py` (CorrelationValidator / CorrelationGroup, `/validate` endpoint)
  is a SEPARATE legacy path, not the factor_correlations copula path. Untouched.

### Calibration (the threshold choice) — computed on the pristine sampler
Frustrated triangle rho/rho/-rho (unit-diagonal 3x3, trace 3):

| case | lambda_min | max off-diag adj | frobenius | verdict |
|------|-----------|------------------|-----------|---------|
| **CONTRADICTION 1/1/-1** (Codex repro) | **-1.000** | **0.500** | 1.2247 | **REJECT (required)** |
| existing test 0.9/0.9/-0.9 (lambda=-0.8) | -0.800 | 0.400 | 0.9798 | REJECT (strongly inconsistent, NOT noise) |
| frustrated 0.7 | -0.400 | 0.200 | 0.490 | REJECT |
| frustrated 0.55 | -0.100 | 0.050 | 0.1225 | REJECT (lambda<-0.05) |
| frustrated **0.51/0.51/-0.51** | **-0.020** | **0.010** | 0.0245 | **PROJECT (near-PSD, my 200 case)** |
| frustrated 0.505 | -0.010 | 0.005 | 0.0122 | PROJECT |
| PSD (rho<=0.5 frustrated, or 0.9 pair, rho=0) | >= -1e-10 | 0 | 0 | PROJECT / no-projection |

**Chosen reject band (Neil-parameter, D-23.13, documented + fingerprint-guarded):**
- `CORRELATION_REJECT_MIN_EIGENVALUE = -0.05`  (reject if lambda_min < this)
- `CORRELATION_REJECT_MAX_ADJUSTMENT = 0.10`   (reject if a stated correlation would be silently moved by more than this)
- Reject iff EITHER fires (fail-closed OR). `CORRELATION_ADMISSION_METHOD_VERSION = "corr_admission_v1"`.

**Research justification (brief):** the Higham (2002) nearest-correlation projection was designed to
repair matrices that are *almost* valid — pairwise/partial-data estimation, differing sample windows,
rounding — where lambda_min is only slightly negative (float-noise scale ~1e-10 up to a few 1e-2). On a
unit-diagonal correlation matrix eigenvalues sum to n and are each in [0,n] when valid; a smallest
eigenvalue below -0.05 is ~5% of a full unit variance "borrowed from nowhere" — a genuine spec
inconsistency, not rounding. The max-off-diagonal-adjustment criterion is the interpretable materiality
gate: if reaching validity requires silently changing a *stated* correlation by more than 0.10, the
projected matrix embodies a materially different dependence assumption than the caller declared, so the
analysis would silently run a different problem. Both fire on the Codex [-1,2,2] case (lambda=-1.0,
adj=0.5). The near-PSD-noise case (0.51 frustrated, lambda=-0.02, adj=0.01) clears both and still projects.
The verdict is PERMUTATION-INVARIANT (eigenvalues + max|off-diag adjustment| are invariant under symmetric
relabeling), so it does not depend on factor draw order.

**Blast radius of the reject band on the copula path = exactly one existing test:**
`test_non_psd_triggers_higham_with_disclosure` (0.9/0.9/-0.9, lambda=-0.8). Per the task's own RED-first
framing ("a genuinely near-PSD matrix (tiny negative eigenvalue) still projects"), lambda=-0.8 is NOT a
tiny/noise eigenvalue; that input is now correctly REJECTED. The test's INTENT (prove the Higham+disclosure
path) is preserved by repointing it at the near-PSD 0.51 case. `test_psd_detection` /
`test_higham_returns_psd_unit_diagonal` exercise the UTILITY functions directly (not the reject band) and
stay green unchanged.

### Design
- `correlation.py`: add Neil constants + version; `AdmissibilityVerdict` + `evaluate_correlation_admissibility(matrix)`;
  extend `ProjectionInfo` with per-pair `effective_pairs`; `build_correlation_plan` computes effective off-diagonals.
- `robustness_v2.py`: shared `_correlation_matrix_inputs()`; `validate_factor_correlations` assembles the matrix
  after the per-pair guards and raises a typed 422 (ValueError) when inadmissible, naming offending pairs + the
  scalar lambda_min/max-adjustment (no raw rho or other request values echoed).
- `response_v2.py`: additive-optional `effective_correlations: List[EffectiveCorrelationV2]` on CorrelationProjectionV2.
- analyzer `_build_correlation_disclosure`: map effective_pairs -> effective_correlations.

### RED-first (pristine) → GREEN (post-fix)
- RED on pristine (`test_correlation_hard_invalid.py`): 5 failed / 2 passed. The 4 reject tests
  ACCEPTED the contradiction (no reject existed); the effective-disclosure test hit
  `AttributeError: 'CorrelationProjectionV2' object has no attribute 'effective_correlations'`.
  The 2 that passed were the near-PSD-projects and valid-PSD-accepted paths (already correct).
- Codex EXACT repro reproduced RED: `rho(fa,fb)=1, rho(fa,fc)=1, rho(fb,fc)=-1` → accepted + silently
  projected on pristine (200); now → typed 422.
- GREEN post-fix: `test_correlation_hard_invalid.py` 14 passed; `test_robustness_correlation.py`
  46 passed (39 orig + updates). Reject message (clean `errors()[0]["msg"]`):
  "...not positive-semidefinite beyond the near-PSD repair band (smallest eigenvalue -1.0000 < -0.05;
  nearest-correlation projection would move a stated correlation by up to 0.5000 > 0.1)... Reconcile the
  offending pairs: ('fb', 'fc'), ('fa', 'fb'), ('fa', 'fc'). (admission method corr_admission_v1)"

### Gates
- **black**: pristine HEAD is ALREADY not black-clean under black 23.12.1 (the gate only runs in
  deploy-production.yml, not staging). Did NOT run `black -w` on whole files (would bundle unrelated
  pre-existing reformats). Verified per-file: my additions add ZERO net new black hunks (every file's
  working-tree hunk count == its pristine hunk count). Manually black-formatted my 2 own flagged lines.
- **mypy src/**: clean on BOTH trees (141 files, "Success: no issues found"). Touched-module mypy clean.
- **openapi regen**: DONE — `generate_openapi.py --check` now OK. Diff = additive `EffectiveCorrelationV2`
  schema + `effective_correlations` property on CorrelationProjectionV2 + my CorrelationProjectionV2
  docstring sentence. SIDE-EFFECT: FastAPI split `CorrelationModelV2` → `CorrelationModelV2-Input`/
  `-Output` (the spec's FIRST such split). PROVEN content-identical (both variants byte-equal; the model's
  own validation==serialization schema). Wire format UNCHANGED (same fields/types/required). It is a
  FastAPI `separate_input_output_schemas` batch-naming artifact tipped by adding a model to the graph, not
  a semantic contract change; only a generated-TS type-name cosmetic (CorrelationModelV2 -> ...Output).
- **contract**: `test_openapi_schema.py` 11 passed/2 skipped; `test_contract_drift.py` 22 passed — NO drift
  refresh needed (drift baseline untouched; the split does not trip it).
- **goldens / valid-path byte-identity**: valid rho=0.9, inert (no correlation), and rho=0 responses are
  ALL byte-identical between my tree and pristine (timing stripped) — PROVEN. flip_stability golden tests
  pass (23 w/ auth disabled).
- Broad regression selection (11 files): 431 passed, 2 skipped, 0 new failures. The 10 initial "failures"
  were a TestClient 401 auth-harness artifact of running with ISL_API_KEYS (enables auth); with
  ISL_AUTH_DISABLED=true (what CI/integration tests use) they pass — identical behavior on pristine, NOT a
  regression.
- FULL `tests/unit`: **2002 passed, 97 skipped, 0 failed**.
- FULL `tests/integration tests/contract tests/contract_drift`: **273 passed, 554 skipped, 0 failed**
  (skips need redis/external services). Total green 2275, 0 failed.

### Commits (LOCAL ONLY — NOT pushed)
- Base: `4675a0a` (staging tip, verified).
- `f9ee416` — fix(correlation): reject hard-invalid matrices (422) + disclose effective adjusted matrix
  (F4). Impl (correlation.py, robustness_v2.py, response_v2.py, robustness_analyzer_v2.py) + openapi.json
  regen + the FORCED existing-test repoint (test_robustness_correlation.py).
- `7d2a44e` — test(correlation): new hard-invalid suite (reject, band, effective disclosure, fingerprint
  guard).
- **Commit-structure note:** task suggested 3 commits (threshold / disclosure / tests). Folded impl parts
  1+2 into one commit because they touch the SAME functions (correlation.py ProjectionInfo/build_plan,
  the analyzer disclosure) and the reject band FORCES the existing-test repoint (0.9->0.51) — separating
  them would need intra-file hunk surgery (no interactive git here) and would leave a RED intermediate
  (the old 0.9/0.9/-0.9 test fails the instant the reject band lands). 2 green, coherent commits instead.
  Full diff fully reviewable.

### Mutation checks (THROWAWAY detached worktree wt-mutation @ 7d2a44e, discarded after)
- **M1** neuter the gate (`admissible=not reasons` -> `admissible=True`): the 4 reject tests go RED
  (contradiction accepted again = the original defect). Reject band is load-bearing.
- **M2** silently retune a constant (MIN_EIGENVALUE -0.05 -> -2.0): the fingerprint guard
  `test_constants_fingerprint_pinned_to_version` goes RED (version unchanged). Silent-retune alarm works.
- **M3** neuter disclosure (`effective_pairs=effective_pairs` -> `effective_pairs=()`): the effective-
  disclosure tests (new + existing test_non_psd) go RED. Disclosure is load-bearing.
- Worktree restored clean after each; removed at the end.

### Contradictions / things to flag for review
1. **openapi CorrelationModelV2 -Input/-Output split** — my additive field made FastAPI emit the spec's
   FIRST -Input/-Output pair (content-identical, proven; wire byte-unchanged). A generated-TS type-name
   cosmetic only, but worth a reviewer glance since it's a new naming pattern. Not semantically avoidable
   without breaking additivity or a global separate_input_output_schemas=False config change (out of scope).
2. **black gate** — pristine HEAD is already not black-clean under the pinned black 23.12.1; the gate only
   runs on deploy-production.yml, not staging. I did NOT bulk-reformat (would bundle unrelated churn); my
   additions are net-zero new black hunks.
3. **Pre-existing test failures under my local env** (test_flip_stability_bands, decision_evpi_correlation_
   composition) were a TestClient 401 auth-harness artifact — pass with ISL_AUTH_DISABLED=true, identical on
   pristine. Not regressions.
4. **robustness_v2.py is co-edited by sibling lane isl-fixA** (validate_control_candidates, L1047+). My
   hunks (validate_factor_correlations + new _correlation_matrix_inputs, ~L969-1044) are strictly ABOVE and
   disjoint; a merge will be clean but note the shared file.

---

## Fable adversarial round (2026-07-24) — VERDICT: SAFE-WITH-FIXES

Adversarial report at `acceptance-evidence/a3-2026-07-24/codex-fix-B/ADVERSARIAL.md` (in-repo).
Both BLOCKING axes passed: threshold bypass proven IMPOSSIBLE (op-norm distortion ≤ ~0.06 for
every admitted matrix — the λ_min≥−0.05 leg spectrally caps every quadratic form and per-entry
move; the max-adjustment leg is a redundant belt, empirically never fires alone), and no
false-reject of genuinely-PSD structure (branch-structural). Admissibility math matches an
independent Higham impl to ~1e-10; disclosure == what the sampler used to 5.4e-9; all 3 original
mutations bite; the adversarial's own full CI = **2297 passed / 669 skipped / exit 0**, mypy clean.

Three required fixes (none touch mechanism / constants / admission version):

### F-1 (MED-HIGH, honesty) — DONE, commit `8c0d517`
Sparse-hub reject misattribution. `rho(fa,fb)=rho(fb,fc)=0.75`, `(fa,fc)` UNSTATED →
assembled [[1,.75,0],[.75,1,.75],[0,.75,1]], λ_min=1−0.75√2=−0.0607 → 422. The two stated pairs
are jointly satisfiable; the inconsistency is the SYSTEM's zero-fill of the unstated (fa,fc)
(default independence). Old message named only stated pairs → steered the user to weaken TRUE
correlations. FIX: message now ranks EVERY pair over the correlated set by projection adjustment,
splits stated vs zero-filled, and when a zero-filled pair is implicated it names the unstated
pair(s), states that unstated pairs default to correlation 0, and offers BOTH remedies (state the
pair, or weaken the stated ones). Fully-specified contradictions keep the standard message.
RED-first: `TestSparseHubHonesty` (2 tests) failed pre-fix (named only stated pairs). Mutation M4
(force `unstated_moved=[]`) → both RED.

### F-2 (LOW-MED, disclosure completeness) — DONE, commit `8c0d517`
Moved UNSTATED pair disclosed nowhere. Hub 0.74/0.74 ADMITTED (λ_min=−0.0465) → sampler used
(fa,fc)=0.0185. FIX (option i, the honest one): `build_correlation_plan` now also emits unstated
pairs the projection moved off 0, flagged `requested_rho=0.0` + new `stated=False` field, so the
effective matrix is fully reconstructable. `EFFECTIVE_DISCLOSURE_MOVE_TOL=1e-9` is the single
"moved" threshold shared by the message and the disclosure (derive-don't-mirror). RED-first:
`TestMovedUnstatedPairDisclosure` failed pre-fix (AttributeError: no `stated`; unstated pair
absent). Mutation M5 (skip unstated emission) → RED.

### F-3 (LOW, process) — this file, committed at the in-repo evidence path
NOTES.md existed at the program-level `/Users/paulslee/Documents/GitHub/acceptance-evidence/...`
path (per dispatch), but this repo's convention is IN-REPO evidence (`acceptance-evidence/` is
tracked — a3-verify-2026-07-16 shipped with the clone; the adversarial wrote ADVERSARIAL.md there).
The adversarial looked in-repo + main repo and reported it absent. Resolution: commit NOTES.md at
`acceptance-evidence/a3-2026-07-24/codex-fix-B/NOTES.md` so the test-docstring reference resolves.

### Fixes verification (post F-1/F-2)
- New SHAs: base `4675a0a`; `f9ee416` (impl) → `7d2a44e` (tests) → **`8c0d517` (F-1/F-2)**.
- mypy src/: clean (141 files). black: net-zero new hunks (F-1 rewrite black-formatted in place;
  only my F-1 region changed vs f9ee416).
- openapi regen: `--check` OK; delta vs prior = ADDITIVE only (`EffectiveCorrelationV2.stated`);
  no schema add/remove, paths unchanged.
- Valid/inert/rho0 responses BYTE-IDENTICAL to base `4675a0a` (re-verified post F-1/F-2).
- Full `tests/unit`: **2005 passed / 97 skipped / 0 failed**. `tests/integration+contract+drift`:
  **273 passed / 554 skipped / 0 failed**. `test_correlation_hard_invalid.py + test_robustness_correlation.py`:
  56 passed.
- Mutations (throwaway worktree, module.__file__ asserted, removed after): M1 gate→4 reject RED;
  M2 retune→fingerprint RED; M4 F-1 honesty→2 hub RED; M5 F-2 disclosure→moved-unstated RED.

### ⚠ DOCTRINE FLAG — for Paul / Neil (NOT decided here)
Should a sparse-but-COMPLETABLE hub pattern (e.g. 0.75/0.75 through a hub, third pair unstated)
be REJECTED (current) or AUTO-COMPLETED to the nearest consistent PSD? D-23.4 says "no invented
default correlations" — but the zero-fill of an unstated pair IS an invented (fa,fc)=0. The safe,
honest containment (reject + name the zero-fill + offer both remedies) SHIPS NOW. Auto-completion
(treat unstated pairs as free and project to nearest PSD instead of forcing 0) is the open doctrine
question. Adversarial row (non-blocking): if CEE ever LLM-drafts DENSE correlation sets over >8
factors, revisit the −0.05 floor (dimension-aware, e.g. −c·√n) under a version bump — the floor
does not scale with n (dense 15-factor coarse-grid input rejects ~21%); not this product's shape today.
