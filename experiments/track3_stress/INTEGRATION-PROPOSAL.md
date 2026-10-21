# Track 3: minimal integration proposal (draft for the DL)

Status: proposal only. Nothing here is integrated or served. Evidence is in `evidence/` (runs at PLoT acf24764 / ISL 8d862f5a).

## What a user would do
- **The control:** "Stress-test this model". The user picks two or more assumptions already in the living model, gives each an explicit range or setting, and can add one named alternative explanation, such as a proposed retention-loss link. Olumi proposes the settings and the user authors them. Nothing is saved to the model.
- **The words they get back (model-relative):** "Growth-first meets £90k at (0,0), with x alone and with y alone, but falls short when both happen (£80k)." "With the retention-loss explanation, Balanced holds the highest tested level at three of four settings." "Met at 6 of 8 tested settings" is never shown as a chance. Rows the engine cannot decide are shown as UNRESOLVED.
- **Product test:** it shows where the team's own model fails and whether that depends on their explanation. It names no winner and makes no recommendation.

## Interfaces and owners
| Piece | Change | Owner |
|---|---|---|
| CEE core lane | One read-only tool, `stress_test`, beside `run_analysis`. It builds temporary GraphV3 variants from the mapper's output (stresses as declared link parameters; an alternative explanation as one added link) and calls PLoT `/v2/run` once per setting. It writes nothing to the living model. | DL (`sdk-turn.ts`, `path-b/living-model/mapper.ts`) |
| PLoT | **None needed for G1–G4.** All of them are expressible on `/v2/run` today. | — |
| UI | A stress result view (the per-setting table, joint-failure witnesses, hypothesis disagreements, worst level and maximum regret side by side, UNRESOLVED rows). The display fields are as in `results.json` → `analysis`. | Lenses lane |
| Persistence | The stress run is stored as an analysis artefact bound to the model revision, so it survives reopening and never replaces the ordinary Run. | DL |

## Engine findings to allocate (each one isolated patch, with its failing row in `evidence/`)
1. **Exact ties:** ISL refuses an edge spread ≤ 0.001 and PLoT floors causal edges (`graph-normaliser.ts:1078`), so an exact tie is UNRESOLVED. The exact-sum route resolves it only when every goal parent is an option-set change (`mapper.ts:189-195`).
2. **Headline chance distorts a declared law:** on Channel A, PLoT's floor widens a declared Normal sd from 10 to 10.9 (P 0.178 vs 0.159). Channel B is faithful.
3. **The exact-sum route drops a declared spread on today's level** with no warning (sd 0, P 0/1/1 where 0.159/0.5/0.841 is declared). This should either be drawn or be refused with a reason.

## Rollback and budget
- **Rollback:** the tool reads the model and writes nothing, so turning it off is the whole rollback. There is no migration.
- **Measured cost:** about 1–2 s per `/v2/run` locally. G1–G4 with every control is 77 calls in 150 s. A user-facing run is hypotheses × settings × 1 call (8 calls for G2).
- **Proposed ceiling:** 16 calls and 60 s per user request, with cancellation that keeps partial rows marked incomplete.

## The journey that would prove delivery
1. On staging, a user opens a scenario with a supported model.
2. They choose two assumptions and an alternative explanation, and run the stress test.
3. The result shows the joint-failure row and the explanation disagreement.
4. They reopen the scenario: the same result is bound to the same model revision, and the ordinary Run is unchanged.
