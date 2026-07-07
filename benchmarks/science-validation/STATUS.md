# STATUS — science-validation lane

Updated: 2026-07-07 (session 1 — review pass + handoff)

## Branch note for the orchestrator

The brief named `claude-web/science-validation`; this session's execution
environment designates and only permits `claude/science-validation-higher-k-ptc70x`.
All work is on that branch. Draft PR: **#66**, titled
`[science-validation] harness + report (docs/benchmarks only — no src changes)`
(https://github.com/Talchain/Inference-Service-Layer/pull/66). All CI green.

## Handoff for review (POC orchestrator)

Everything for this lane is in draft PR #66 (base `staging`). It stays draft —
this lane never merges to staging; merge/deploy is the orchestrator's call.
A review-guide comment is pinned on the PR with reproduction commands and the
load-bearing claims to probe. This session is subscribed to PR review events
and will address feedback (validity-assessed first, per CLAUDE.md code-review
rules) on the same branch.

Review checklist:
- Scope: `git diff staging...HEAD --stat` touches only
  `benchmarks/science-validation/**` and `docs/science-validation/**`. No
  `src/` changes (recommended src fixes are written up in REPORT §5.6).
- Reproduce: `poetry run python benchmarks/science-validation/run_all.py`
  (full, ~15 min) or `--quick` (~2 min). Seeds pinned in `_lib.py:SEEDS`;
  `results/*.json` reproduce bit-for-bit apart from `provenance`/`elapsed`.
- Adversarial targets: harness fidelity to production code paths (K override,
  exp2's `_compute_evpi` orchestration vs `_compute_evpi_metric`), the
  closed-form truths in `graphs.py`, the prefix-nested-RNG pooling fix
  (commit `bfa7836`), and the doctrine findings in REPORT §5.
- Two commits: initial delivery, then a self-review correction (`bfa7836`)
  to the pooled statistics — point estimates and classifications unchanged,
  only inferential bounds moved.

## State

- Harness complete and validated: `benchmarks/science-validation/` (exp1-exp4,
  one-command runner, pinned seed registry, results committed with provenance).
- Full runs of all four experiments complete; findings in
  `docs/science-validation/REPORT.md`.
- No `src/` changes made. Recommended src changes are written up as findings
  only (REPORT section "Doctrine-relevant findings").
- Code-review pass (2026-07-07, same session): corrected exp1's pooled
  statistics for overlapping RNG streams (the smaller-K sweeps are prefixes of
  the K=100000 stream, so only the largest-K rows are independent — the
  TRUE-ZERO bound is 6e-6, not the 5.4e-6 first stated), guarded the chi-square
  homogeneity test against sparse counts, and tightened exp2's sign-flip
  reference to the largest-n consensus. All classifications and headline
  conclusions are unchanged; results/ regenerated.

## Headline results (details and reproduction commands in REPORT.md)

1. **Higher-K**: the K=100 default under-resolves. Repo fixture graphs contain
   edges that are uniformly zero at K=100 across 5 seeds but resolve to
   p ~ 6.6e-4 and 8e-5 at K=100000. The estimator itself is unbiased (validated
   against closed-form truth across ten analytic cases; no instability across
   seeds anywhere). TRUE ZERO vs UNDER-RESOLUTION is decidable with a K sweep +
   rule-of-three bound.
2. **EVPI floor**: conservative in comfortable regimes (empirical SD 0.5-0.9x
   the worst-case SE), exactly tight at knife-edge (P(win) ~ 0.5) with ~5%
   false-"resolved" rate on true-zero factors — as a 95% bound should behave.
   Below-floor signs flip across seeds 25-61% of the time.
   Bonus structural finding: under the p_win metric, factor EVPI is exactly
   zero for any factor whose causal path no option intervenes on
   (common-mode cancellation) — such factors' reported EVPI is pure MC noise.
3. **Determinism**: same-seed responses are byte-identical only after masking
   four volatile fields: metadata.execution_time_ms, envelope timestamp,
   envelope processing_time_ms, and critiques[].id (uuid4 per run,
   src/models/critique.py:34 — recommended fix written up, not made).
   With those masked: 50/50 identical in-process and over the wire. **Across
   OS processes, 44/50 graphs differed** (count varies with the process hash
   salt): robustness.fragile_edges /
   robust_edges are built via list(set(...)) so their ORDER follows the
   per-process string-hash salt (PYTHONHASHSEED), and the user-facing
   robustness.interpretation string names the first three entries of that
   unordered list — i.e. WHICH edges the interpretation cites is
   process-dependent. All numeric content is identical once ordering is
   normalised. Recommended one-line fix (sorted()) written up, not made.
4. **Calibration (groundwork)**: without auto-noise, probability_of_goal is
   unbiased against closed-form truth (95.6% of cells within 95% MC bands).
   With auto-noise (outcome/risk goals), reported probabilities deviate from
   the no-noise truth by up to ~0.17; the implementation matches its own
   noisy model, so the distortion is the heuristic itself, quantified.

## Next steps (future sessions)

- Extend calibration grid to 2-3 edge chains (product-of-strengths cases).
- Empirical p99 runtime budget for K=10000 on 50-node/200-edge graphs, to cost
  a production K increase for the Slice-3 decision.
- Waiting on: sight of `docs/slice3-log-and-isl-higherK-spec.md` — not present
  in the plot-lite-service repo (checked main, all remote branches by name, and
  GitHub code search, 2026-07-07); presumed local to the orchestrator.

## Boundaries compliance

- No pushes to staging/main; draft PR only. No deployed-service calls.
- Writes confined to `benchmarks/science-validation/**` and
  `docs/science-validation/**`.
