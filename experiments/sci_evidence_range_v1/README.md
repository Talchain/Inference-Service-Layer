# SCI-EVIDENCE range elicitation (experiment v1)

**NEVER MERGE WITHOUT A SCIENCE RULING. NEVER DEPLOY.** Offline research only. It touches no file outside this
directory, calls no served endpoint and no LLM (provider runs: NOT_RUN).

- **Brief:** `Talchain/olumi-programme-docs#75` comment `5909728656` (Science).
- **Corrected scope:** Science plan review `5911560060` (approve with 3 corrections).
- **Lane:** SCI-EVIDENCE (bounded extension). Not SCI-FERMI, not a new engine.

## Question

When an important assumption has only a point estimate, can Olumi ask for the smallest useful range, re-run
**existing** computation, and show something genuinely new that the point estimate could not support?

## Case and quantity

| Item | Value |
|---|---|
| Case | R3-B graph `pj-20260927T180910Z-A`, model `X_net_reading` (secondary `X_gross_reading`), frozen evaluator at ISL `fdb51e84`, consumed from `sci/regions-contrastive-vulnerability` @ `68e8c887` (the pinned SCI-REGIONS case) |
| Goal | MRR reaches £100,000 at some month by month 12 (first passage), from £75,000 today (user-stated) |
| Hard constraint | Monthly churn ≤ 4 % every month (user-stated) |
| Quantity | **Monthly (base) churn, 3 % per month.** Source `cee_inference`: an Olumi estimate with **no range** |
| Why this one | It sits on both the constraint path and the goal path, and the existing Regions grid and its 1.5 pp churn-feasibility threshold silently condition on it being exactly 3 % |
| Experimental range | **2 % to 4 % per month — HYPOTHETICAL USER-SUPPLIED, experimental control only.** Not evidence, not a distribution, not a claim about the real world |

## Method (existing code only)

1. **Before:** reproduce the frozen reference point with `PricingAdapter.measurements(0.5, 0)` unchanged.
2. **After:** a deterministic sweep of base churn across the hypothetical range, through the same frozen evaluator
   (`sim.static.evaluate_option` → `sim.dynamic.run_model`). No distribution is assumed. Results are **per option**:
   constraint feasibility and goal first passage, with any crossing bracketed on the grid. No option ranking, no
   leader, no EVPPI (Science correction 1 and 2).
3. **Production-path control:** Paul's current MRR graph (A-graph, ISL `sci/structural-robustness-20260930` @
   `6f7f9b43`) on served ISL `f7f19e3`, in-process, before vs after a user spread on churn. This is a limitation test
   of today's product path (Science correction 3). The only distributional assumption in the experiment lives here,
   because the production engine requires a σ.
4. **Evidence card:** before and after bodies through the frozen SCI-EVIDENCE Lab adapter
   (`olumi-programme-docs` `ef108360`, `lab/w1_card.py::card_from_served`), unchanged.

## Verdict rule

KEEP only if the elicited range licenses a genuinely new per-option robustness/threshold claim, or reveals a crossing
inside the range that the point estimate could not support. Otherwise KILL.

## Status

Skeleton only. Runs, results, payload and coaching example follow in later commits on this branch.
