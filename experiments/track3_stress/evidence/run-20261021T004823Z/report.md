# Track 3 Milestone 1: G1 and G2 on the local PLoT→ISL path

**Verdict: PASS.** 0 reference failures against the exact oracle; 0 completeness gaps. 4 exact-tie rows are **UNRESOLVED (engine floor)** on the floored path, never counted as met or missed; 4/4 resolve on the core's exact-sum route (below). Negative controls behaved as expected: 12/12.

- Path: service boundary: runner -> local PLoT /v2/run -> capture proxy -> local ISL (auth off, no network beyond loopback).
- PLoT `acf24764e99d913a19c8b6088a381487989f5db9` · ISL `caf38f73099000a7fb799c0bdf8294d18bd798ec` · @talchain/schemas 0.82.0 · ISL analyser `/private/tmp/track3-engine/isl/src/services/robustness_analyzer_v2.py` · seed 20261021 · n = 2000 draws per call · Z = 4.0.
- Units: £000, 12 months, target Y ≥ 90 (inclusive). PLoT scale: goal_threshold_cap = 200 £000 per unit.
- Runtime: 113.2 s wall for 54 PLoT calls (clean run and controls); ISL 121.7 MB, PLoT 158.0 MB resident at the end.

## G1: option A (Growth-first), Y = 120 − 20x − 20y, under M0

| x | y | Exact Y | Exact side | Engine level ± SE | Engine P(goal) | Engine side | Check |
|---:|---:|---:|---|---:|---:|---|---|
| 0 | 0 | 120 | meets | 120.00 ± 0.028 | 1.000 | meets | agrees |
| 0 | 1 | 100 | meets | 99.97 ± 0.037 | 1.000 | meets | agrees |
| 1 | 0 | 100 | meets | 100.01 ± 0.035 | 1.000 | meets | agrees |
| 0.7 | 0.7 | 92 | meets | 91.98 ± 0.042 | 0.852 | meets | agrees |
| 0.75 | 0.75 | 90 | meets | 89.98 ± 0.042 | 0.484 | indeterminate | tie_unresolved_on_engine_path |
| 0.8 | 0.8 | 88 | falls_short | 87.98 ± 0.042 | 0.140 | falls_short | agrees |
| 1 | 1 | 80 | falls_short | 79.98 ± 0.042 | 0.000 | falls_short | agrees |

**Joint failure (engine rows):** A under M0 — meets at (0,0), at x alone and at y alone, falls short at x and y together. An additive threshold, not an interaction term.

## G2: eight settings, three options, two explanations

| Hypothesis | x | y | A exact / engine | B exact / engine | C exact / engine | Highest tested level |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0 | 0 | 120 / 120.00 | 105 / 105.00 | 95 / 95.00 | A |
| M0 | 0 | 1 | 100 / 99.97 | 100 / 99.99 | 93 / 93.00 | A/B |
| M0 | 1 | 0 | 100 / 100.01 | 95 / 95.01 | 93 / 93.00 | A |
| M0 | 1 | 1 | 80 / 79.98 | 90 / 90.00 UNRESOLVED (engine floor) | 91 / 91.00 | C |
| M1 | 0 | 0 | 90 / 90.06 UNRESOLVED (engine floor) | 105 / 105.01 | 95 / 95.00 | B |
| M1 | 0 | 1 | 70 / 70.06 | 100 / 100.01 | 93 / 93.00 | B |
| M1 | 1 | 0 | 70 / 70.02 | 95 / 94.99 | 93 / 93.00 | B |
| M1 | 1 | 1 | 50 / 50.02 | 90 / 89.99 UNRESOLVED (engine floor) | 91 / 91.00 | C |

| Measure over the 8 tested settings | A | B | C |
|---|---:|---:|---:|
| Meets ≥ 90, exact (brief reference) | 4/8 (4/8) | 8/8 (8/8) | 8/8 (8/8) |
| Meets / UNRESOLVED (engine floor) / falls short, engine | 3 / 1 / 4 | 6 / 2 / 0 | 8 / 0 / 0 |
| Worst tested outcome, exact / engine | 50 / 50.02 | 90 / 89.99 | 91 / 91.00 |
| Maximum tested regret, exact / engine | 41 / 40.98 | 15 / 15.00 | 25 / 25.00 |

**Explanations disagree (engine, both rows determinate):** A at (0,1) meets under M0, falls_short under M1; A at (1,0) meets under M0, falls_short under M1.
**Adding the M1 mechanism changes which option holds the highest tested level** at: (0,0) A → B; (0,1) A/B → B; (1,0) A → B.
B has the smallest maximum tested regret; C has the highest worst tested outcome; A has the largest favourable-case outcome but falls short at the joint stress under M0 and at every stressed setting under M1. These are model-relative findings over 8 tested settings: no option is a universal winner, and no probability is placed on M0 or M1.

## Located boundary losses (the engine path, not the fixture)

- **Exact ties are undecidable on this path.** 4 rows sit exactly on the target (A M0:x=0.75,y=0.75, A M1:x=0,y=0, B M0:x=1,y=1, B M1:x=1,y=1). ISL rejects any edge spread ≤ 0.001 and PLoT floors causal edges at min(0.05, max(0.001, |mean|/2)) (`graph-normaliser.ts:1078`), so a declared-exact model reaches ISL with spread; at an exact tie the level estimate cannot certify ≥ 90 and P(goal) is a coin flip. They are reported as unresolved, never as met.
- **Raw £000 graphs are not the supported input.** Sent un-normalised, PLoT capped a strength of 100 to 1, re-scaled interventions, synthesised lever spreads for `std: 0`, and clipped the target 90 to 1 (probe, this session). The runner therefore uses PLoT's own goal ruler (`goal_threshold_cap`).
- **On this path P(goal) is an artefact for an exact model.** exact £92k shows P(goal) 0.85; exact £88k shows P(goal) 0.14. The spread comes only from PLoT's floor, so the runner decides the side from the level and never reports these as chances.
- Engine-only probe: ISL refused edge std 0 for captured request 0001.json: HTTP 422 'graph -> edges -> 0 -> strength -> std: Input should be greater than 0.001'

### The core's exact-sum route on the UNRESOLVED rows (goal = today + change; CEE `mapper.ts` exact sum)

| Row | Exact | ISL level (normalised) | Threshold | Strict sent | P(goal) | Identity evaluated | Resolved |
|---|---:|---:|---:|---|---:|---|---|
| B M0:x=1,y=1 | 90 | 0.4500000000000001 | 0.45 | False | 1.0 | True | yes |
| A M1:x=0,y=0 | 90 | 0.4500000000000001 | 0.45 | False | 1.0 | True | yes |
| B M1:x=1,y=1 | 90 | 0.4500000000000001 | 0.45 | False | 1.0 | True | yes |
| A M0:x=0.75,y=0.75 | 90 | 0.4500000000000001 | 0.45 | False | 1.0 | True | yes |

This route admits no parent of the goal except option-set changes and no negative change, so the runner computes each option's stressed change exactly and the engine only adds and compares: it proves the exact comparison, not stress propagation. The inclusive tie holds by one unit in the last place (level 0.4500000000000001 against 0.45), which comes from the cap round-trip.

## Negative controls (each fault must turn the reference check red)

| Fault | Brief | Expected | Observed | First failure it caused |
|---|---|---|---|---|
| omit_cost_stress | G1: omit one stress | RED | RED | cell M0:x=0,y=1 A: level/side deviates from the exact oracle (oracle 100.0, engine 119.99853503104713) |
| reverse_demand_sign | G1: reverse a stress's sign | RED | RED | cell M0:x=1,y=0 A: level/side deviates from the exact oracle (oracle 100.0, engine 140.0102169577272) |
| strict_comparison | G1: change >= to > | RED | RED | A: oracle meets 3/8 but the brief's reference is 4/8 |
| missing_as_success | G1: replace a missing row with success | RED | RED | A: summary counts 4 meets but its rows show 3 |
| stale_cache | G1: reuse a cached result after a stress changes | RED | RED | cell M0:x=0,y=1 A: level/side deviates from the exact oracle (oracle 100.0, engine 119.99965182303136) |
| remove_m1_path | G2: remove the M1 path | RED | RED | cell M1:x=0,y=0 A: level/side deviates from the exact oracle (oracle 90.0, engine 119.99965182303136) |
| drop_adverse | G2: drop an adverse scenario | RED | RED | requested G2 population is 7 settings, fixture defines 8: missing ["('M1', Fraction(0, 1), Fraction(1, 1))"] |
| average_models | G2: average the models without weights | RED | RED | A: worst tested level 64.9999250457027 is not the minimum of its rows (50.017807748388) |
| oat_only | G2: perturb only one coordinate | RED | RED | requested G2 population is 6 settings, fixture defines 8: missing ["('M0', Fraction(1, 1), Fraction(1, 1))", "('M1', Fraction(1, 1), Fractio |
| unique_winner | G2: select a unique winner at the A/B tie | RED | RED | setting ('M0', Fraction(0, 1), Fraction(1, 1)): highest tested level held by ['B'], exact reference ['A', 'B'] |
| horizon_mismatch | G2: compare different horizons | RED | RED | rows compared across different horizon_months |
| missing_row (contrast, no fault) | G1: same failed row, reported as missing | GREEN, verdict INCOMPLETE | GREEN, verdict INCOMPLETE | row not evaluated: A M0:x=1,y=1 (missing) |

## Scope and what this does not show

- Synthetic fixtures from brief v1.0 §6; not customer evidence. Stresses are explicit coordinates, not probabilities; 'met at 6 of 8 tested settings' is not a 75 % chance.
- Stresses are scenario parameter assumptions (edge strengths −x, −y on controllable exposures); options are interventions. M1's £30k is an illustrative assumption carried on a named intervention → retention → contribution path.
- Engine spread here is PLoT's floor artefact, not uncertainty in the fixture. Monte Carlo precision is not model validity.
- Status: isolated prototype with numerical and service-boundary evidence. Not integrated, not served.
