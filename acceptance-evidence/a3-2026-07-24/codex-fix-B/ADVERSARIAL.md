# ADVERSARIAL — Codex-F4 correlation hard-invalid rejection + effective-matrix disclosure

Fable adversarial merge gate. Branch `a3-codex-fix-correlation-hard-invalid`
(f9ee416 + 7d2a44e) vs base 4675a0a3.
Clone: scratchpad/isl-fixB (read-only except tests/evidence).
Mutations: OWN throwaway worktree scratchpad/isl-fixB-adv-mut (created from HEAD,
removed after use — never the shared clone).
Env: poetry venv JTSC4fOU (py3.11, numpy 1.26.4, pydantic 2.6.1), PYTHONPATH pinned
to the tree under test, `module.__file__` asserted at the top of EVERY run (the
venv .pth injects a DIFFERENT clone, isl-fixC-verify — pin verified load-bearing).

## VERDICT: SAFE-WITH-FIXES

No threshold bypass exists (hunt 1: quantified impossibility, not just not-found).
No false-reject of genuinely-PSD structure is possible (branch-structural). The
admissibility math is correct, permutation-invariant, and matches an independent
implementation to 1e-10. Disclosure equals what the sampler actually used to 5e-9.
All 3 mutations bite. Full CI selection green (2297 passed), mypy clean.

Two fixes required before merge (both small, neither touches the mechanism):
F-1 (message truth — the reject message misattributes hub-pattern inconsistency
to the user's stated pairs and cannot name the unstated pair that drives it) and
F-2 (wire-docstring truth — the "reconstruct the effective matrix" claim is false
whenever the projection moves an unstated pair). Finding 0 (missing NOTES.md) needs
the builder to commit the notes or strip the dangling references.

---

## Finding 0 — builder NOTES.md ABSENT (evidence-integrity)

- severity: LOW (process) · tests/unit/test_correlation_hard_invalid.py:10 · claim:
  dispatch and the test-file docstring both reference
  acceptance-evidence/a3-2026-07-24/codex-fix-B/NOTES.md ("Each mechanism carries a
  mutation note in NOTES.md").
- DEMONSTRATED: the path does not exist in the clone at 7d2a44e, nor anywhere under
  the scratchpad, nor in /Users/paulslee/Documents/GitHub/Inference-Service-Layer.
  The builder's claimed CI counts were unavailable; this adversarial derived its own
  (see Hunt 6).
- fix-or-row: commit NOTES.md or strip the dangling references. **F-3.**

## Hunt 1 — threshold BYPASS: NOT-REPRODUCED (quantified impossibility)

Claim hunted: a grossly-contradictory matrix that clears λ_min ≥ −0.05 AND
max-adjustment ≤ 0.10 and is silently projected.

- Block-diagonal k mild frustrated triangles (rho 0.52, λ_min −0.04/block):
  k=1..20 (n up to 60) all ADMITTED; aggregate Frobenius grows √k (0.049 → 0.219)
  but max per-entry adjustment stays 0.020 — many small frustrations do NOT
  compose into any large single distortion.
- Random search, 3000 matrices n∈[3,16) tuned to hover at the boundary: max
  ADMITTED Frobenius distance 0.070; max ADMITTED **operator-norm** distortion
  0.0504.
- Engineered worst case (spectrum forced to k negative eigenvalues at −0.049,
  unit diagonal re-imposed, n=50, k=25): ADMITTED with op-norm distortion 0.0571,
  Frobenius 0.348, sum-of-adjustments 1ᵀΔ1 = −1.13 against an equal-weight
  aggregate variance of ~110 → ~1% distortion of any portfolio-style quantity.
- Why no bypass exists: the λ_min ≥ −0.05 leg bounds the PSD-projection's
  operator norm at ≈|λ_min| (+small diag-restore effects), and |Δᵢⱼ| ≤ ‖Δ‖₂,
  so EVERY quadratic form wᵀΣw and every single entry moves by ≤ ~0.06 across
  everything findable. "Severe aggregate misrepresentation under the per-entry
  radar" is spectrally impossible at these settings.
- Honest labelling note (no action needed): empirically the max_adjustment=0.10
  leg NEVER fires alone — max admitted per-entry adjustment found was 0.023
  (op-norm 0.057). The eigenvalue leg is the binding constraint and enforces an
  EFFECTIVE per-entry materiality cap of ~0.05–0.06, i.e. STRICTER than the
  0.10 the comment advertises. Fail-closed in the safe direction; the OR is a
  redundant belt.
- Boundary semantics: frustrated(0.525) → λ_min = −0.050000000000000044 (float)
  → rejected via strict `<`; constants comment documents strict `<`/`>`.
  Consistent. Exact-boundary behavior is float-representation-dependent but on
  the reject (safe) side.

## Hunt 2 — FALSE-REJECT: one REAL diagnostic defect (F-1), thresholds defensible

- (a) **Genuinely PSD high correlation can never reject — structural.**
  `evaluate_correlation_admissibility` returns trivially-admissible when
  λ_min ≥ −1e-10, before any threshold. 500 random valid Gram matrices n∈[2,20):
  0 rejects. A valid matrix with small λ_min from high real correlation
  (0.9/0.9/0.9, single-factor loadings 0.98) passes untouched.
- (b) **Float-noise: −0.05 is 3.0e13× above the noise floor.** Worst computed
  λ_min over 300 rank-deficient (true λ_min = 0) valid 50×50 matrices:
  −1.66e-15. No realistic n≤50 assembly noise can cross −0.05.
- (c) **Human 0.1-grid triples** (this product's realistic elicitation): 69.7%
  pass / 5.9% project / 24.4% reject over all 6859 triples — but the rejects are
  genuinely inconsistent specs. The boundary sits sensibly for the canonical
  case: 0.9/0.9/0.7 PASS · 0.9/0.9/0.6 PROJECT · 0.9/0.9/0.5 PROJECT
  (λ_min −0.047) · 0.9/0.9/0.4 REJECT (λ_min −0.088; the implied floor for the
  third rho given two 0.9s is 0.62, so 0.4 IS a real contradiction). Defensible.
- (d) Dense fully-stated matrices rounded to 0.1: rejects 0% (n=4) · 0.5% (n=6)
  · 2.5% (n=8) · 4.8% (n=10) · 21.5% (n=15). Rounding to 0.05: ≤0.2% everywhere.
  The floor does not scale with n, so DENSE many-factor coarse-grid input
  degrades — but a dense 15-factor/105-pair hand-stated request is not this
  product's shape. **Row, not blocking: if CEE ever LLM-drafts dense correlation
  sets over >8 factors, revisit the floor (dimension-aware, e.g. −c·√n) under a
  version bump.**
- (e) Pairwise-window-estimated matrices (n=8–12, partially overlapping samples):
  38–90% reject. Out of scope for this product's input path (nobody uploads
  sample-estimated matrices to the canvas); the doctrine explicitly reserves
  "partial-data estimation" slop for the PROJECT band, and these estimates land
  far outside it. Documented limitation only.

### F-1 (fix-before-merge) — hub-pattern reject message misattributes the inconsistency

- severity: MEDIUM-HIGH (message truth / user guidance, not mechanism) ·
  src/models/robustness_v2.py:1088-1119.
- claim: the 422 message asserts "the SUPPLIED pairwise correlations are mutually
  inconsistent" and directs "Reconcile the offending pairs: <stated pairs>" — but
  the offending-pairs list is built from `pairs` (stated only) and CANNOT name an
  unstated pair, while assembly fills unstated pairs with 0 ("Independence is the
  default", field description robustness_v2.py:820).
- DEMONSTRATED repro: graph fa/fb/fc, normal uncertainties, correlations
  (fa,fb,0.75) + (fb,fc,0.75), third pair UNSTATED → assembled
  [[1,.75,0],[.75,1,.75],[0,.75,1]], λ_min = 1−0.75√2 = −0.0607 → 422:
  "the supplied pairwise correlations are mutually inconsistent … Reconcile the
  offending pairs: ('fb','fc'), ('fa','fb')". The two supplied correlations are
  jointly satisfiable (any r_ac ∈ [2·0.5625−1, 1]); the inconsistency is with the
  SYSTEM's default-independence fill of (fa,fc). A user following the message
  would weaken their two true correlations; the correct action is to STATE the
  third (≈0.56+). Hub threshold: two equal stated pairs sharing a factor reject
  at rho ≥ 0.7425 (0.74/0.74 admits, 0.75/0.75 rejects). Stating two strong
  correlations through a hub factor and omitting the third is THE natural sparse
  canvas elicitation (sparse-elicitation simulation: 3–6% of requests at
  n=5–10). Pre-fix these requests ran (silently projected); post-fix they 422
  with a diagnosis that blames the wrong thing.
- The reject itself is doctrinally CORRECT (the request, as contractually
  interpreted, does encode a materially non-PSD matrix; silently projecting it
  moved stated 0.8s to 0.725 and imposed r_ac=0.052 the user never asked for).
  Only the diagnosis is wrong.
- fix-or-row: FIX before merge (small, message-only, no mechanism change):
  compute offenders over ALL moved entries of `verdict.projected − matrix`
  (stated AND unstated); when an unstated pair is involved, name it as
  "unstated pair (fa, fc) — treated as independent by default" and adjust the
  headline to "…are mutually inconsistent (under the default independence of
  unstated pairs)". Keep `corr_admission_v1` (message text is not a
  Neil-parameter; constants unchanged).

## Hunt 3 — admissibility math: CORRECT

- λ_min matches independent `numpy.eigvalsh` on 200 random symmetric unit-diag
  matrices exactly (≤1e-12).
- Higham projection matches an INDEPENDENTLY WRITTEN Higham (paper-derived,
  different convergence criterion) to max-abs 9.8e-11 over 100 non-PSD cases;
  output verified PSD + unit-diagonal; local-optimality cross-check (2000 random
  feasible candidates) never found a closer correlation matrix.
- Permutation invariance: verdict (admissible, λ_min, max-adjustment) identical
  under 240 random row/col permutations incl. reject cases (≤1e-9/1e-7).
  Also verified at the model level: the assembled matrix uses
  first-appearance-in-parameter_uncertainties order — reordering factors
  permutes the matrix symmetrically, which the metrics are invariant to.
- Assembly: symmetric, unit diagonal, reversed pair (b,a) lands symmetric,
  unstated entries 0, self-pair no-op. Disclosed effective_pairs read
  `projected[idx[a], idx[b]]` under the same ordering as assembly — no
  transposition/ordering skew possible (verified by direct comparison).
- Dispatch's exact case: [−1,2,2] spectrum confirmed; λ_min=−1.0,
  max-adjustment 0.5000, both reasons fire → 422. Frustrated 0.51: λ_min −0.0200,
  adjustment 0.0100 → admitted + projected. Codex's [−1,2,2] must-422 and
  near-PSD must-project both hold.

## Hunt 4 — disclosure truth: HONEST to 5e-9; one docstring overclaim (F-2)

- Producer→wire trace: `build_correlation_plan` computes ONE `projected` matrix;
  `plan.cholesky = _safe_cholesky(projected)` (what FactorSampler multiplies) and
  `plan.projection.effective_pairs` (the disclosure) both derive from that same
  object in the same call — the link is by construction. Measured
  max |(L·Lᵀ)ᵢⱼ − disclosed effective_rho| = 5.4e-9 across near-PSD cases —
  the residue is the `_safe_cholesky` eigen-lift (ridge 1e-8 + unit-diag
  renormalisation) applied when the Higham output is PSD-singular. The disclosure
  IS what drove the numbers, to 5e-9.
- Wire model verified end-to-end: analyzer response
  `correlation_model.psd_projection.effective_correlations` == plan values;
  max|pair adjustment| == disclosed `max_abs_off_diagonal_adjustment` exactly.
- Link-break probe: a tampered plan (identity cholesky, unchanged disclosure)
  lies by 0.5 and nothing trips — there is no RUNTIME invariant tying cholesky to
  effective_pairs. Acceptable today (single-producer function); worth remembering
  if anyone ever decouples them.

### F-2 (fix-before-merge) — "reconstruct the effective matrix" wire claim is false for unstated pairs

- severity: LOW-MEDIUM (wire-docstring truth) · src/models/response_v2.py:1038-1046
  (EffectiveCorrelationV2 docstring) and :1101-1108 (effective_correlations field
  description); same text is now baked into openapi.json.
- claim: "Together with the unit diagonal these entries reconstruct the effective
  correlation matrix".
- DEMONSTRATED: hub 0.74/0.74 (ADMITTED, projected): sampler's effective
  (fa,fc) = 0.0185 while the user implied 0.0 and `effective_correlations` lists
  only the 2 stated pairs — reconstruction from the disclosure (unit diagonal +
  disclosed pairs + zeros elsewhere) puts 0 where the copula actually used
  0.0185. 4-factor case: undisclosed effective rho 0.0114. Bound: band caps the
  gap at ≤ ~0.05, but the claim as written is false whenever the projection
  moves an unstated pair.
- fix-or-row: FIX before merge — either (i) also emit pairs whose effective
  off-diagonal moved from the implied 0 (flagged `requested_rho=0.0`,
  unstated), or (ii) weaken the docstring to "…reconstruct the effective values
  of the STATED pairs; unstated pairs may also be adjusted (within the same
  disclosed max_abs_off_diagonal_adjustment) and default to ≈0". (i) is more
  honest and additive-optional on the same field.

## Hunt 5 — composition: CLEAN; openapi split = byte-identical name churn

Ordering verified empirically (each case 422s with the FIRST applicable guard):
dup+contradiction → "duplicate pair" · unknown factor → "non-existent factor
node" · self-pair → "self-correlation" · missing uncertainty → "has no
parameter_uncertainty" · point_mass → "normal or uniform" · duplicate
parameter_uncertainties (+contradiction) → the EARLIER validator's "Duplicate
parameter_uncertainties" message (model validators run in definition order; the
gate never sees a dup-order matrix) · rho=1.2 → field-level bound before any
model validator. No raw exception escapes anywhere (no 500-shaped path found).
Bonus fail-closed behavior: (fa,fb,1.0)+(fa,fc,1.0) with (fb,fc) unstated →
correctly rejected (assembled [[1,1,1],[1,1,0],[1,0,1]] is hard-invalid). Exact
rho=±1.0 single pairs still accepted (PSD-singular, pre-existing lift path) —
gate did not break them.

Derive-don't-mirror: `_build_correlation_plan` now calls the request model's own
`_correlation_matrix_inputs()` — validator and analyzer share ONE derivation
(confirmed by reading both call sites; the old duplicated block is deleted).

openapi `-Input`/`-Output` split (builder-flagged): `CorrelationModelV2` was
replaced by `CorrelationModelV2-Input` + `CorrelationModelV2-Output`; verified the
two are JSON-byte-identical (sorted) and `-Input` == the OLD schema modulo title.
Split is generator bookkeeping triggered by the new defaulted field; HTTP wire
shape of all pre-existing fields unchanged; `effective_correlations` is
additive-optional (absent → old shape). isl-python-client has zero references to
the renamed schema. LOW: any future codegen pinned to the bare name
`#/components/schemas/CorrelationModelV2` would break — schema-NAME churn is
real even though content is identical. Row only.

## Hunt 6 — reproduction, mutations, CI

At HEAD (clone, PYTHONPATH-pinned, `ISL_AUTH_DISABLED=true`):
- tests/unit/test_correlation_hard_invalid.py + test_robustness_correlation.py:
  **53 passed** in 1.98s.
- Byte-identity/valid/inert set named + green:
  TestInertWhenAbsent::test_no_correlation_model_when_absent ·
  test_absent_path_emits_attributions_positive_control ·
  test_factor_sampler_plan_none_is_deterministic_independent ·
  TestCorrelationActivation::test_rho_zero_all_normal_is_bit_identical_to_absent ·
  test_rho_zero_with_uniform_factor_differs_from_absent ·
  TestCorrelationUtility::test_rho_zero_gives_identity_cholesky (6/6) ·
  TestValidPathUnchanged::test_psd_pair_still_accepted (rho=0.9, no projection).
- Mutations (throwaway worktree isl-fixB-adv-mut, module.__file__ asserted per run):
  - **M1** gate neutered (`if False and not verdict.admissible`):
    4 reject tests RED (codex-contradiction, message-names-pairs,
    no-other-values, strongly-inconsistent), 10 others green — the request-level
    tests see the mechanism, not just the pure function.
  - **M2** silent retune (−0.05 → −0.06, version untouched):
    TestAdmissionFingerprintGuard::test_constants_fingerprint_pinned_to_version
    RED (fingerprint literal is hardcoded — cannot self-heal). 13 others green.
  - **M3** disclosure neutered (effective_correlations → None): 2 disclosure
    tests RED across BOTH files, 51 green.
  - Worktree restored clean and removed after use.
- Full CI selection (`pytest tests/ --ignore=tests/_archived
  --ignore=tests/_quarantined -x -q -m "not perf"`): **2297 passed, 669 skipped,
  4 deselected, exit 0** in 69s.
- CI mypy gate (`mypy src/`): **Success: no issues found in 141 source files**.
- Builder counts could not be cross-checked (NOTES.md absent — Finding 0).

## NOT-REPRODUCED (hunted, absent)

- Threshold bypass (contradiction admitted+projected): impossible at these
  settings — op-norm distortion ≤ ~0.06 for every admitted matrix (Hunt 1).
- False-reject of a genuinely PSD matrix: structurally impossible (Hunt 2a).
- Float-noise reject at n≤50: 3e13× margin (Hunt 2b).
- Sign-flip of a stated correlation inside the admitted band: none found
  (grid search; admitted adjustments too small to cross zero from |rho|≥0.01).
- Permutation/ordering dependence of the verdict: none (Hunt 3).
- Disclosure lying about the sampler's matrix: gap 5e-9 = documented lift only
  (Hunt 4).
- KeyError/500 paths in the gate's message builder: none reachable — every
  factor in `pairs` is guaranteed a position by the per-pair guards (Hunt 5).
- Validator-order hazard (dup parameter_uncertainties reaching the gate): the
  earlier validator fires first, empirically (Hunt 5 vi/vi-b).

## Required fixes (recap)

| # | severity | file:line | fix |
|---|----------|-----------|-----|
| F-1 | MEDIUM-HIGH | src/models/robustness_v2.py:1088-1119 | Reject message: include unstated pairs among nameable offenders and attribute the inconsistency to "stated correlations + default independence of unstated pairs", not to the supplied pairs alone |
| F-2 | LOW-MEDIUM | src/models/response_v2.py:1038-1046, 1101-1108 (+openapi regen) | Either disclose moved unstated pairs in effective_correlations or weaken the "reconstruct the effective correlation matrix" claim |
| F-3 | LOW | acceptance-evidence/a3-2026-07-24/codex-fix-B/NOTES.md | Commit the referenced NOTES.md or strip the dangling references from the test docstring |

Rows (no merge gate): dimension-scaling of the −0.05 floor if dense >8-factor
LLM-drafted correlation sets ever become a real input shape; openapi schema-name
churn (`CorrelationModelV2` → `-Input`/`-Output`) if any consumer ever codegens
from openapi.json; max_adjustment leg empirically subsumed by the eigenvalue leg
(comment overstates its role — the effective materiality cap is ~0.06, stricter
than 0.10).

## VERDICT: SAFE-WITH-FIXES (F-1, F-2, F-3 before merge; none touch the mechanism, constants, or version)
