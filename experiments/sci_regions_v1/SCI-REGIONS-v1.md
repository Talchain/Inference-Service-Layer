# SCI-REGIONS v1 — deterministic research checkpoint

**Status:** offline, fixture-first prototype. It is an experimental analysis surface for one decision case, not Olumi's product model or evidence of better human judgement.

## Scientific question and result

Can a two-dimensional map report where an option's goal attainment, hard feasibility and terminal-revenue preference differ, without inventing a winner or claiming a continuous-space guarantee?

The deterministic checkpoint passed its known-answer synthetic controls. On the pinned R3-B pricing case, it produces separate net/gross exploratory panels. At zero extra churn and a −£31,250/month competitive response, the £59-with-feature option **misses the £100k first-passage goal, satisfies the churn limit (2.5% against 4%), and trails keep-current on month-12 MRR**. Calling that option “infeasible” would be wrong. The original [R3-B review](https://github.com/Talchain/Inference-Service-Layer/blob/fdb51e84a673993de7cddb85d5b963741941b80a/experiments/r3b_sim/REVIEW-20260928.md) uses that word for this case. The point calculation here corrects the interpretation without altering its frozen model.

The corpus graph declares six options. Three have no intervention values. The map retains all six and reports `INCOMPLETE_COMPARISON` for the whole decision. Its named, provisional preference compares keep-current, £59-with-feature and £49-with-feature only.

## Contract and computation

`RegionRequestV1` and `RegionResultV1` are strict Draft-07 JSON Schemas in `schema/`. The runtime validates the exact set of schema keywords it uses, rejects unknown fields/enums, non-finite values, duplicate IDs and unresolved bindings, and refuses operations outside `FIX_SCENARIO_PARAMETER`. The goal is independent of the objective. An absent objective returns goal/feasibility records and `OBJECTIVE_UNSPECIFIED`.

At each coordinate, the classifier evaluates hard constraints before comparing feasible options. A known violation excludes an option even when its outcome is unavailable. `NO_FEASIBLE_OPTION` requires every declared option to be known infeasible. It never treats a missed goal as a failed hard constraint unless the request explicitly declares that goal as a constraint.

Exact synthetic values use rational arithmetic. `EXACT_TIE` needs rational equality. A positive practical margin yields equivalence only when every relevant pair is within it. The R3-B float path is labelled `POINT_ESTIMATE_PREFERRED`, and its narrow numerical screen is marked `HEURISTIC_ROUNDOFF_SCREEN`; neither is a certified arithmetic interval or a statistical confidence interval.

The two R3-B axes are direct churn response `[0,4.25]` percentage points per +£10 and competitive response `[−31,250,0]` GBP/month. These are **exploratory ranges**, not validated plausibility or a joint distribution. The source is [commit `fdb51e84a673993de7cddb85d5b963741941b80a`](https://github.com/Talchain/Inference-Service-Layer/tree/fdb51e84a673993de7cddb85d5b963741941b80a/experiments/r3b_sim). The adapter verifies `FROZEN.sha256`, the corpus manifest and content hashes, then composes `model_evals` with the frozen `run_model`; `simulate` is checked for equality at reference coordinates. The stored break-even JSON is reproduced before the grid run. The frozen source files are not edited.

## Reference tests and limits

The frozen synthetic definitions are in `fixtures/FROZEN.json`. F1 explicitly means A for `x>y`, B for `x<y`, exact tie for equality. F11 is the exact circle `1/40000 − (x−1/80)^2 − (y−1/80)^2 > 0`; its A island is missed by the 41×41 grid and found by the 201×201 reference. This is an intentional demonstration that an all-B coarse grid cannot prove B everywhere. `oracle.py` computes independent rational answers and shares no classification helper with the candidate.

Every admitted synthetic coordinate at 11×11 and 41×41 matched the independent reference. F11 also ran at 201×201. The controlled checks found zero false-feasible or falsely confident preference records at evaluated points. The real R3-B case has no external ground-truth oracle, so those error rates are **unmeasured**, represented as `null` rather than zero. Transition brackets and point tiles assert only evaluated-coordinate coverage.

The offline HTML offers a matched local point, a one-axis threshold table and a two-dimensional point map, all reading the same computed JSON. It exposes fixed assumptions, full-versus-named comparison status, both model readings and per-option reasons. It does not compare with a current deployed result; that comparator is `UNAVAILABLE`.

Monte Carlo coverage and interval validity, F7/F10/F12/F14, learned scenario summaries, 3-D and human-comprehension benefits remain `NOT_RUN`. A later study must verify the bounded-IID conditions and interval construction before making confidence claims. Even a valid simultaneous finite-grid interval would not certify unsampled locations. Any product integration needs separate source-to-consumer semantics and a witnessed user journey.

## Adoption decision

Bank the truth contract and research adapter. Use a one-axis threshold plus targeted question when one missing effect dominates. Consider a region interface only if a matched human comparison shows that it helps people identify viable options, the assumption that changes the choice and when no conclusion is justified. This checkpoint establishes computational behaviour under stipulated assumptions; it does not establish user value or production readiness.
