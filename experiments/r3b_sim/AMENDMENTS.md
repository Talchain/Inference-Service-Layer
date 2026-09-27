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
