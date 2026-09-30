# SCI-STRUCTURAL-ROBUSTNESS, independent second run: Paul's MRR brief, subscriber time scope (30 Sep 2026)

**Verdict: KEEP.** This confirms the first run's KEEP of the concept and qualifies one of its user-facing claims.

**Rung: TESTED.** Local and in-process, on the served CEE, PLoT and ISL heads. Not served. No LLM calls.

- **Brief:** Talchain/olumi-programme-docs#75 comment 5907927672.
- **ACK:** #75 comment 5911632589.
- **Scope:** SCIENCE's ruling 5911688274 allows this run one new time-scope ambiguity and then says STOP. This run adds no other ambiguity.
- **Claim boundary:** SCIENCE approved only "structurally robust / structure-sensitive across these explicitly tested plausible representations". The run makes no probability over structures and no "correct model" claim.

## 1. Frozen inputs

| Item | Identity |
|---|---|
| Brief (verbatim; four-cases `paul-mrr`) | "Should we raise our Pro plan price from £49 to £59 a month? We have 1,500 paying subscribers and £75k MRR. Monthly churn must stay below 5%, and we want MRR above £85k within a year." |
| Source draft | R3's served **m2** of that brief (`r3/science-notes` @ `fb3a396c`, `row-b-2339-ef042ce/drafts/m2.json`). Served CEE `ef042ce`, scenario `3eb2e670…`. Copied as `models/m2-served-draft.json` (sha256 `ad0379c4…623f`). |
| Model A | m2, then Olumi's **own** card: `proposeProductIdentity(m2)` returned "Is "MRR" your "Pro plan price" × "Paying subscribers"? £49 × 1,500 = £73,500, close to your £75,000…" Then `applyIdentityConfirmEdit` returned `mutated` (`models/A-card-and-edit.json`). The only change is `stated_in_brief: false → true` (`models/DIFF.json`). No hand edit. |
| Graph hashes (`computeAnalysisAffectingGraphHash`) | m2 `4e611c845a165aab` · A `f136e81e033ca4cd` · A2 `ca913d9bc17348e6` · B `a2596e82a669253a` |
| Heads | CEE `950177e9` (served) · PLoT `af4cd569` (served; same pin as the first run) · ISL `f7f19e31` (served; this branch's base) |
| Chain | CEE's real `run_analysis` handler builds the PLoT `/v2/run` payload (`models/*-plot-payload.json`). PLoT's real `/v2/run` runs in-process against a local ISL (`plot-responses/`, `isl-requests/`). ISL's own V2 route then runs in-process, seeds `1254899477` (the served seed), `1` and `20260930`, n = 10,000 (`runs-direct/`). |

**Why m2 and not the first run's m0:**
- m0 predates CEE #2341, so on today's build its invented "Non-Pro MRR £1,500" is removed (MG 5910471110).
- 7 of the 10 served drafts of this brief read MRR = price × today's subscribers. m2 is one of them, and it carries sized Olumi estimates on its churn pathway.
- The served journey makes m2 analysable with one press: the card's Yes.

## 2. Structural ambiguities the brief and provenance support

| # | Ambiguity | Evidence | Status |
|---|---|---|---|
| 1 | MRR = price × subscribers (product) vs additive | Card; `stated_in_brief:false` (AIQ 5908212412) | Tested by the first run (H2 flips 0 ↔ 1) |
| 2 | Total vs Pro-only MRR | £49 × 1,500 = £73,500 ≠ £75k; four-cases "preserve both and ask about scope" | Tested by the first run |
| 3 | price → churn in or out | Not user-authored (AIQ 5908212412, 5909030106) | **Not selected.** ISL already samples its absence (`exists_probability` 0.8) and reports it in `fragile_edges` and the e-values, so a structural toggle adds nothing parametric analysis can't show. |
| 4 | **Time scope of "paying subscribers" in MRR = price × subscribers** | The brief states today's 1,500 but targets "within a year". Olumi's own 10 served drafts of the unchanged brief split: **m1 and m8** read subscribers at month 12; **m2–m7 and m9** read today's subscribers (`drafts.log`). m2's own reply says: "Does "MRR" get there within 12 months? The model holds the deadline; no result answers that yet." The goal carries `goal_horizon_months: 12` into the PLoT payload, but it never reaches ISL: the ISL request has no horizon. | **Selected** |
| 5 | price → new subscribers | Needs a magnitude the brief lacks (m4, m6 and m7 guess one; m2 has none) | Not constructible without adding a value |

## 3. Model B: exact structural delta (see `models/DIFF.json`)

**Hypothesis B:** the subscriber operand of MRR is the **month-12** count the goal is about ("within a year"). Its level today is still the user's 1,500.

- m2's own churn effect is "−15 subscribers per +1pp of monthly churn". That is one month's extra loss on 1,500 (1% of 1,500).
- m2's new-subscriber effect is "+1 subscriber per new subscriber/month", which is one month's inflow.
- Read at month 12, the same Olumi estimates accumulate over the brief's 12 months, first order (no compounding): **−180 per pp** and **+12 per new subscriber/month**.
- No new fact or value is added. The 12 is the brief's own horizon.

| Change A2 → B (the only differences) | A2 | B |
|---|---|---|
| `paying_subscribers.label` | Paying subscribers | Paying subscribers at 12 months |
| `monthly_churn → paying_subscribers` natural effect | −15 subscribers per pp | −180 per pp (= 12 × −15) |
| its normalised strength (mean, std) | −0.075, 0.0375 | −0.90, 0.45 (same 50% spread) |
| `new_paying_subscribers_per_month → paying_subscribers` natural effect | +1 | +12 (= 12 × 1) |
| its normalised strength (mean, std) | 0.05, 0.025 | 0.60, 0.30 |

**What stays identical:** every node, level, unit, option, other edge, uncertainty, constraint, the goal and its threshold, the identity, and the seeds.

**Why A2 exists:** ISL clamps a normalised strength to [−1, 1]. At the served subscriber frame (cap 5,000), −180 per pp would be −3.6, which is a coercion. So the subscriber frame is re-expressed losslessly to 20,000 in **A2 = A** and in B.

**Frame control:** A2 reproduces A's every headline figure at all three seeds (`results/comparison.json`, rows A vs A2). Some diagnostics are frame-dependent; see finding F1.

**Disclosed limits of B:**
- It is first order. ISL's engine (linear, plus product/sum identities) cannot express monthly compounding. A compounding structure is NOT COMPARABLE here, and it was not built.
- m1's literal drafted shape (unsized default edges into a month-12 outcome with no level) is not analysable. R3's `m1-yes-run-de6c642` run is blocked on the missing unit, and AIQ 5900908629 withholds unsized paths. It was not run.

## 4. Controls

1. **First run reproduced.** Its committed A and B ISL requests (ISL `6f7f9b43`) replay on `f7f19e3` **identically on every science field**, with volatile fields masked (`control-first-run/`).
   - First-run A: £59 leads 0.8605, P(>£85k) 0.
   - First-run B: £59 leads 1.00, P(>£85k) 1.00.
2. **This chain is self-consistent.** For every model, the PLoT-mediated ISL response equals the direct replay of the ISL request PLoT sent (identical science fields).
3. **Frame invariance.** Every A2 headline figure equals A's (above).
4. **Parametric check: does A itself reveal B's result?** No:
   - A's `fragile_edges` is empty.
   - Every edge e-value reads "unflippable", and `edge_sensitivity` reads "Decision is robust to…".
   - `p_win_sensitivity` is below resolution.
   - ISL has no sensitivity output for the goal probability at all.
   - The served model's parametric view therefore cannot show that "reaches £85k" rests on the time scope.

## 5. Like-for-like results (£59 unless stated; [min, max] over 3 seeds; "cur" option set)

| | A: today's subscribers | B: subscribers at month 12 |
|---|---|---|
| Leader | Raise to £59 | Raise to £59 |
| Win probability | 1.00 | 0.835–0.842 (0.809–0.814 with £54) |
| **P(MRR > £85k)** | **1.00** (0.9999–1.00) | **0.53** (0.529–0.533) |
| MRR mean | £89.7k | £83.3k–£83.4k |
| MRR p10 | £88.7k | £71.2k–£71.6k |
| P(£59 ends below today's £75k) | 0.00 | 0.16 (0.158–0.165) |
| P(churn < 5%) | 0.87–0.88 | 0.87–0.88 |
| price → churn e-value (flips the leader?) | unflippable | flips at ≈0.22pp per £1, i.e. 2.2× Olumi's 0.1pp estimate (all 3 seeds; 7–9 of 10 stability seeds) |
| churn → subscribers | not fragile | fragile at 2 of 3 seeds, switch probability 0.43 |
| Robustness label | high (stability 1.00) | high (0.81–0.84) |
| Identity | evaluated | evaluated |
| £54 (served-era set): wins / P(>£85k) | 0 / 0 | 0.05 / ≤0.004 |

**Mechanism.** ISL anchors MRR to the stated £75k, so MRR(£59) ≈ £75,000 + 1.02 × [£59 × (1,500 − L) − £73,500], where L is the extra subscribers lost.
- A reads Olumi's churn effect once: L ≈ 15, giving about £89.7k.
- B builds it up over the year: L ≈ 180 when the chain holds, giving about £83.4k on average.
- £59 stops paying for itself when L > 254. m2's own reply said this: "a loss of at most 254".

## 6. Classification of headline conclusions (across A and B only)

| # | Headline (as the served post-Yes model says it) | Class |
|---|---|---|
| H1 | Raise to £59 comes out ahead | **STRUCTURALLY ROBUST** (margin narrows: 1.00 → ≈0.84) |
| H2 | £59 gets MRR above £85k within a year (≈100%) | **STRUCTURE-SENSITIVE** (1.00 → 0.53) |
| H3 | £59 cannot leave MRR below today's £75k | **STRUCTURE-SENSITIVE** (0.00 → 0.16; p10 £88.7k → £71.4k) |
| H4 | No plausible churn response flips the leader | **STRUCTURE-SENSITIVE** (unflippable → flips at 2.2× Olumi's price → churn estimate) |
| H5 | £59 keeps monthly churn under 5% (0.87) | ROBUST, but **invariant by construction** (the time scope doesn't touch churn's level); not informative here |
| H6 | £54 is never best | **STRUCTURALLY ROBUST** |
| H7 | Robustness "high" | ROBUST label; the stability number moves (1.00 → 0.81–0.84) |
| — | Driver / structural-influence scores | **NOT COMPARABLE** between A and B (they depend on frame; see F1) |

**What this adds to the first run:**
- The first run's line "If MRR is your Pro price × paying subscribers … £59 reaches about £89.8k and clears £85k in every scenario" is reproduced by A: £89.7k, 1.00.
- **That line is itself structure-sensitive to the time scope.** Confirming the product card does not settle the £85k question. The leader survives all five tested structures (three in the first run, two here).

## 7. User-facing explanation

> Raising to £59 comes out ahead whether Olumi counts the extra churn from the price rise once, or builds it up over the year you're aiming at. Whether it gets MRR above £85k within a year depends on that choice. Counted once, £59 reaches about £89.7k and clears £85k in almost every scenario. Built up over 12 months, it averages about £83.4k: roughly an even chance of clearing £85k, and about a 1-in-6 chance of ending below today's £75k. Both versions rest on Olumi's guess that each £1 adds about 0.1 points of monthly churn, which isn't your figure. The thing to check next is how many of your 1,500 Pro subscribers you'd expect to lose over a year at £59. Above about 250, the rise stops paying for itself.

## 8. Verdict: KEEP

**Success criterion met:**
- A real conclusion is classified in a way the served model's parametric sensitivity cannot show: the ≈100% chance of £85k that the served analysis would give after the card's Yes.
- It comes with a concrete user action: estimate the 12-month loss at £59; the break-even is about 254 subscribers.
- The leader is classified robust.
- The alternative is not arbitrary. It is the reading 2 of Olumi's own 10 served drafts chose, it uses only Olumi's own estimates, and its horizon is the brief's.

**Smallest insertion point (for SCIENCE to decide; nothing is integrated):**
- AIQ's proposed route (5911189156 (e)) computes each Run under both readings and names the leader only when they agree.
- Add the time-scope reading to that pair, only when:
  - the goal's product identity has a count operand fed by per-month flow estimates, and
  - `goal_horizon_months > 1`.
- In CEE's `run_analysis` handler, this is one extra PLoT call on the deterministic B transform (§3). Name the leader when both runs agree. Show P(goal) as a single figure only when both runs put it on the same side; otherwise name the assumption and ask for the 12-month loss.

## 9. Findings for owners (not rulings)

- **F1 (R3 / ISL): e-value reach depends on frame.** ISL's e-value search is bounded in normalised strength [−1, 1], so "unflippable" depends on a meaning-free frame choice.
  - The same model reads churn → subscribers as *unflippable* at subscriber cap 5,000 (A).
  - At cap 20,000 (A2), 4 of 10 stability seeds find a winner flip at ≈ −153 to −193 subscribers per pp (median −177).
  - `structural_influence` and the factor confidence scores also move with the frame. The headline figures do not.
- **F2 (PLoT / ISL / AIQ): the goal's horizon is dropped before analysis.** The goal's `goal_horizon_months: 12` is in the CEE graph and the PLoT payload, but not in the ISL request, and nothing in ISL's analysis reads a horizon. Every Olumi per-month flow estimate therefore enters as a one-month effect, which is reading A by default.
- **F3 (AIQ):** "Since #2341, one press on the card lifts Gate 5" (AIQ 5911189156). On m2, that press yields a ≈100% chance of £85k (A), which H2 shows rests on reading A. If Gate 5's lift ships, this assumption should be named, or the figure withheld.
- **Not structural:** the chance churn stays under 5% differs across Olumi's drafts (m0 about 1.00, m2 0.87). The difference comes from values, not structure: churn today 3% vs 3.5%, and 0.05 vs 0.1pp per £.

## 10. Limitations

- n = 1: one brief, one served draft, one alternative.
- B is first order. Compounding cannot be expressed in ISL, and B is not "the true model".
- Olumi's estimates are held fixed. H1's robustness holds only against this choice and the first run's two.
- Local and in-process only. Fixed seeds; optional-phase budgets are raised as in the first run (they gate whether a phase completes, never its numbers).

## Layout

`models/` (m2, A, A2 and B graphs, CEE→PLoT payloads, card/edit, hashes, DIFF) · `isl-requests/` (what PLoT sent to ISL) · `plot-responses/` · `runs-direct/` (18 ISL runs) · `control-first-run/` · `results/comparison.json` · `scripts/`

**Reproduce:**
1. `zz-ssr2-capture.test.ts`, run as an untracked CEE test with `SSR2_OUT` and `SSR2_M2`.
2. `serve_isl.py` together with `ssr-run.ts` (the first run's script, run from a PLoT `af4cd569` checkout).
3. `run_isl.py` (seeds).
4. `compare.py`.
