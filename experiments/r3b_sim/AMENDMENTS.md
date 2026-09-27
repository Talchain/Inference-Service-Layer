# Amendments after the freeze

The frozen files are `ANALYSIS-PLAN.md` and `mapping/*.json`, frozen in commit `b1d1907`. **None of them has changed** since the freeze; `FROZEN.sha256` still matches, which is checked on every run and by a test.

The engine code was corrected after its first run. None of the corrections touches a classification, and each is listed here with its effect:

1. **A model's own algebraic operand edges are handled by the model.**
   - When a Mode X model evaluates `mrr = price x subscribers` itself, the operand edges are no longer also treated as behavioural links. Before this fix, they tainted the option.
   - **Effect:** in C-183807Z (`X_inversion`), `feature_59_price` became computable (£79,742-£84,454 at month 6; the goal is not met under any reading). Before, it was withheld. No strict (T0/T1) result changed.
2. **Limit units on a node with no unit.**
   - A limit on a node that carries no unit of its own (E-182848Z `annual_salary_spend`) is now withheld as `UNIT_UNVERIFIABLE` at T0-T2.
   - In Mode X it is evaluated under the named assumption `X_unit_from_constraint`. Before, it was withheld as `UNIT_MISMATCH` at every tier.
   - **Effect:** the Mode X salary verdicts are £240k / £260k ≤ £400k (both hold). No strict result changed.
3. **Reporting only.**
   - A declared identity that the mapping withholds is now reported as withheld for every option, rather than showing the held level.
   - The T2 category counts identity values as well as limit verdicts.

## After the RESULT post (#70 5860374496)
These are additions only. The frozen files are unchanged, and `results.json` is byte-identical (sha256 `967c1f90…`).

4. **MVP-B break-even: `sim/breakeven.py` → `breakeven.json`, with the "MVP B" section in `EVALUATION.md`.**
   - **What was added:** the user challenged the 8–13-week cost of B. This answers that challenge with a reproducible Mode X calculation: the price → churn response at which each £59 verdict flips on A-180910Z. The calculation was first run inline and is now in the repo.
   - **Semantics:** it uses the frozen `attain_by_H` goal semantics and the mapping's own `X_price_churn_direct` effect, and it is labelled `prototype_assumption`.
   - **Engine change:** `sim/run.py` gains `model_evals()`, which is extracted unchanged from `simulate()` so that the churn-limit threshold comes from the engine rather than from hand arithmetic.
   - **Tests:** 3 new tests. Each threshold flips its verdict, the churn threshold matches the hand calculation (4 − 2.5 = 1.5 pp), and the committed file equals a fresh run.
   - **Effect:** no strict result changed, and the answer is unchanged.
