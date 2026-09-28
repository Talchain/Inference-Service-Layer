# R3-B plan (updated 28 Sep ~11:50Z): ChatGPT feedback incorporated

## Context
There are three inputs:
- the approved 48-hour plan (W1–W5);
- the review response (`experiments/r3b_sim/REVIEW-20260928.md` @ `fdb51e8`);
- ChatGPT's feedback on that review. All 8 points are accepted; amendments are below.

**Since the review:**
- **W1** is ISL #198: REVIEW_READY, AIQ meaning ACK (5868801032), head `c054ce4` after a generated-file fix.
- **The DL assigned R3-B the strength-truncation P0** (5868680004). It is built as `d7ebd53` on `r3b/strength-truncation`, with pre-push running.
- **The next-capability proposal is posted** (5868879851).

## Assessment of ChatGPT's feedback (accept / adjust)
1. **W1 at the served 10,000 draws, with positive controls and "not detected" wording: accept.** This matters because if false positives mostly vanish at 10k draws, the gate's marginal value on the served path is smaller, and that must be said.
2. **Cross-service fields whenever correctness, provenance or freshness requires them: accept.** It corrects my "only when a user-visible output consumes it" rule, which was too narrow. Horizon-in-hash and `spread_source` are examples.
3. **Time semantics as ONE coordinated contract** (Canonical/MG/ISL: horizon, onset, flow meaning, evaluation rule), built in parallel and activated only when invalidation works end to end: **accept.** Adjustment: one written contract with a single drafter, each part behind a flag.
4. **P1 as an independent offline agent: accept.** It combines automatic typing with targeted questions. Metrics:
   - agreement with the frozen hand mapping (class precision/recall, zero invented semantics);
   - manual effort saved;
   - usefulness of the questions.

   The old "more facts than the mapping" stop criterion is dropped.
5. **P2 grid first: accept.** Goal, constraint and preference regions come from a simple grid. PRIM/scenario discovery is added only where it beats the grid.
6. **Product: accept as my recommendation to Paul.**
   - Show T2 results as provisional scenarios with explicit, editable assumptions.
   - Show preference boundaries only with a user-confirmed objective.
   - `spread_source` is provenance, not proof of credibility.
7. **Subscriptions are the first implementation, not the limit: accept.** A known-answer synthetic suite for non-linear, interaction and categorical mechanisms runs in parallel.
8. **Defer integration, not preparation: accept.** Seek DL agreement *now* for P1, P2 and the time-semantics contract design. Integration waits until after the PoC; existing ownership is unchanged.

## Sequence (acceleration; P0 first)
1. **Truncation P0:**
   - pre-push → push `r3b/strength-truncation` → PR → REVIEW_READY on #72;
   - disclose the rows moved to the legacy sampler, the eng-hiring-4 leader change, and the #197 interaction. The trial merge auto-resolves; #197's `test_no_identity_thresholds_are_base_d1cef9a` must run under the served sampler once both land. I offer to do that.
2. **W1 at n = 10,000** (after the pre-push, to avoid CPU contention):
   - rerun the known-answer study at 10k with the #198 code: false-positive and power rates, with CIs;
   - post a PR comment. The head is not moved unless the DL wants a test row;
   - ask AIQ to confirm the wording rule: `below_resolution` means "not detected at this resolution", never "no information value".
3. **One CLAIM on #72** (folded into the truncation REVIEW_READY post to save the DL's reading):
   - an amendment to 5868879851: ask the DL to agree preparation now (P1 offline agent, P2 grid-first, a time-semantics contract draft with Canonical and MG);
   - integration after the PoC.
4. **After DL agreement:**
   - **P1:** a background agent, offline, using the frozen corpus and mapping as the reference;
   - **P2:** the grid on the A-180910Z / C-183807Z X models;
   - **the time-semantics contract draft:** one doc, reviewed by Canonical, MG and AIQ.
5. **The review document gets an addendum** ("ChatGPT feedback, 28 Sep") on `r3b/sim-prototype`, recording the eight amendments. The frozen study is untouched.

## Verification
- **Truncation:** RED at base (0.772; no draw > 1; frame effect −43%), GREEN at head, mutants RED; F on Paul's body is exactly invariant; byte-identity against the old sampler; pre-push green; merge-tree clean against staging, #193 and #195; the #197 trial merge passes except the one pinned row noted above.
- **W1 at 10k:** state the rates with binomial CIs, including the case where the gate adds little at 10k.
- **Posts** stay within CLAIM/REVIEW_READY format, at most 8 lines, with no extra DL review load.
