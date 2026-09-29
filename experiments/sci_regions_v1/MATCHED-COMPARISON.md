# SCI-REGIONS v1: matched reasoning-value check

**Status: THREE-ARM LOCALHOST BROWSER PREFLIGHT PASSED; DIRECT-FILE OFFLINE OPEN AND HUMAN COMPREHENSION NOT RUN.** This is the bounded next test requested by [Science's checkpoint review](https://github.com/Talchain/olumi-programme-docs/issues/72#issuecomment-5878761605). It asks whether additional views help a person understand when the frozen model's implication changes. It does not test a recommendation, probability, area of plausibility, the deployed product, or Olumi's wider proposition.

## Frozen case and access

Use only `output/comparison.html` generated from `r3b-X_net_reading-41.json` in `output/results.zip`. The archived member's SHA-256 is `8c4d8bb7dbcc7777f645ba8149cca807864200e638f4e4998be1492ecaf49bb7`; request SHA-256 is `a6fe73b150e3c2945e86241192c6e2bb1a2f1adcd15753eccd4ad3e7af1eb4e1`. The frozen evaluator source is `fdb51e84a673993de7cddb85d5b963741941b80a`.

Use the same objective and disclosure in every arm: month-12 MRR, δ = 0; first-passage £100k goal by month 12; monthly churn ≤ 4% as a hard constraint; direct churn response and competitive MRR response as exploratory scenario parameters; indirect churn response fixed at zero; all remaining assumptions in the request's `fixed_assumptions`. All six options remain declared, and only keep-current, £49-with-feature, and £59-with-feature have computable interventions. The net reading is Tier X and the current deployed comparator is unavailable.

Open the local HTML file with one of these query suffixes. The suffix fixes the net reading and hides navigation to other views. Give each participant **one** arm before collecting answers; randomise assignment and keep the wording, task order and facilitator help identical. Use the unmodified file without a suffix only for facilitator inspection.

| Arm | Open `output/comparison.html` with | What is exposed |
| --- | --- | --- |
| A | `?arm=point` | Matched point at (0.5 pp per £10, £0/month) |
| B | `?arm=threshold` | Same point detail plus evaluated churn-response sweep at £0/month and transition brackets |
| C | `?arm=map` | Same point detail plus the two-axis evaluated-point map, layer toggle and coordinate table |

The HTML is self-contained and uses no external network resources. On 29 September, the executor opened all three arm URLs in the Codex in-app browser through a local loopback server serving the unchanged HTML file. The point arm showed £117,090, the feasible £59 option and incomplete six-option comparison; the threshold arm showed the exact evaluated bracket [1.4875, 1.59375] with the feasibility/preference change; the map arm showed the exact 0.53125 and −£18,750/−£17,968.75 coordinates, their different named preferences, and the same incomplete comparison. This preflight found and repaired a four-decimal display rounding defect; the computed JSON did not change. The browser's security policy blocked `file:` navigation, so direct-file offline opening remains unverified. Check that mode before recruiting anyone. Do not silently substitute a screenshot or a different model reading.

## Participant tasks

Ask for a short answer and the evidence they used. Record response time per question, whether the participant says the view is insufficient, and a one-question cognitive-load rating after all tasks. Do not show the answer key until all answers are locked.

1. At the marked point, did the £59-with-feature option attain £100k by month 12, satisfy the hard churn limit, and have the highest month-12 MRR among the three computable options? Is a winner over all six established?
2. At competitive response £0/month, compare the **evaluated** churn-response points 1.4875 and 1.59375 pp per £10. What changes for the £59 option and the named preference? Can you name an exact threshold between them?
3. At churn response 0.53125 pp per £10, compare the **evaluated** competitive-response points −£18,750 and −£17,968.75/month. Does the named preference change? At each point, does £59 attain the goal and satisfy the hard churn limit?
4. What would you ask or check next, and which conclusion would you refuse to make from the material shown?

## Locked answer key and scoring

Score **supported correct**, **appropriate abstention**, **unsupported confident claim**, and **incorrect despite available evidence** separately for each task; report time and load separately. A view with less information should not be penalised for a correct abstention. Any claim that coloured area is probability, an all-six winner, or a user action recommendation is an unsupported confident claim.

- Task 1: At (0.5, 0), £59 has £117,090 month-12 MRR, reaches the goal and has 3% monthly churn (within 4%). Its MRR exceeds £49 (£107,337.68) and keep-current (£98,520) among the three computable options. The six-option comparison is incomplete because three intervention values are missing.
- Task 2: At (1.4875, 0), £59 churn is 3.9875% and it is feasible; its month-12 MRR is £107,559.55 and it is the named point-estimate preference. At (1.59375, 0), £59 churn is 4.09375% and it is infeasible despite attaining the goal; £49 is the named point-estimate preference. The observed bracket is [1.4875, 1.59375] pp per £10, width 0.10625. No exact threshold inside it has been evaluated. Arm A should abstain; B and C have the evaluated evidence.
- Task 3: At (0.53125, −£18,750), keep-current is the named point-estimate preference; at (0.53125, −£17,968.75), £59 is. At both points, £59 misses the £100k goal but satisfies the churn limit. The observed preference bracket is [−£18,750, −£17,968.75]/month, width £781.25/month. There is no exact switch location, area probability, or all-six winner. A and B should abstain on this second-axis comparison; C has the evaluated evidence.
- Task 4: The deterministic question candidate first asks for the missing quantified effect of £59 for new Pro customers while grandfathering existing customers (option `40bb45e7`). Constraint-changing churn brackets and then preference-changing brackets follow. Any sensible request to validate these Tier-X assumptions or the missing options is useful; do not assign an information-value ranking or infer which action the team should take.

Report each arm's response matrix and false-certainty count, with the participant's wording preserved. Compare whether B explains Task 2 and C explains Task 3 without degrading Task 1/4 understanding or imposing unacceptable load. Treat a small internal pilot as diagnostic only. Human benefit and promotion remain `NEEDS_NEW_DATA` until the participant population, acceptance threshold and actual responses are recorded.
