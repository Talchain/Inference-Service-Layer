# Track 3 findings 2 and 3 (21 Oct 2026): recorded, not patched

## Finding 2: the exact-sum route drops a spread on today's level

**Verdict:** LATENT in the core today. The core mapper gives `<goal>_today` no std. The smallest correct fix spans
PLoT and ISL and changes a ruled design (AIQ #72: "at the status quo T = o on every draw"), so it is re-briefed to
the DL, not built.

**Failing row:** ISL 8d862f5a, in-process. `f2_probe.py` replays the captured PLoT→ISL requests from run
`run-20261021T012003Z`: `capture/0074.json` (no spread declared) and `capture/0075.json` (Normal(90, 10²) declared on today).
- Both rows give sd ≈ 1e-17 and P 0 / 1 / 1. Drawn exactly, the declared case is 0.159 / 0.5 / 0.841 (`base-8d862f5a.txt`).
- **Mechanism:**
  - `_identity_value` (stated level) computes `o + k·(term − term_sq) + …`.
  - A root participant's draw is in both `term` and `term_sq` (the status quo on the same draw), so it cancels.
- **What PLoT sends:** in BOTH captures PLoT sends a spread on today.
  - Declared: Normal std 0.05 (normalised = 10 £k).
  - Undeclared: a SYNTHESISED Normal std 0.1 (20 £k, the lever rule `max(0.1, 0.15·|v|)`).
  - Neither carries `spread_source`, because the runner's nodes set no `std_source`.

**Why the naive fix is wrong** (`naive-draw-roots.diff`, measured in `naive-draw-roots.txt`). Reading root
participants against their stated level makes the declared case sd 0.0495, which is correct. But it also makes the
**undeclared** case sd 0.099, because the synthesised 20 £k spread now reaches the goal and the route stops being exact.
P(goal) stays 0 / 1 / 1 in both: the headline chance is computed on a second frame that also cancels today's draw.

**Smallest correct change** (for the DL to allocate). The plumbing already exists:
- **Upstream:** CEE marks a declared spread with `observed_state.std_source`. PLoT maps it to ISL's
  `ParameterUncertainty.spread_source` via `translator-v3.ts:1607` `spreadSourceOf`; an absent source means not stated.
- **ISL (two changes):**
  - draw a root identity participant only when `spread_source` is `'user'` or `'template'`, in `_identity_value`;
  - make the same change in the goal-chance frame.

  Both need the evaluator to see that set, which is a constructor argument at about 6 call sites.
- **Row:** 0075 with `spread_source: 'user'` gives P ≈ 0.159 / 0.5 / 0.841. 0074 (synthesised) stays exact (contrast).
- **Ruling:** a science ruling first, because it changes "T = o at the status quo".

## Finding 3: exact ties on the drawn path stay UNRESOLVED

**Verdict:** a capability limit, recorded with no patch.
- **Why no exact drawn link exists:** ISL rejects edge std ≤ 0.001 (`Field gt=0.001`). After finding 1, a stated drawn
  link sits at PLoT's MIN_STD 0.0011.
- **Effect:** a level exactly at the target is within 4 SE, so it is reported UNRESOLVED (engine floor), never met.
- **Where ties do resolve:**
  - The core's exact-sum route resolves them: 4/4 tie rows, P = 1, with the comparison passing by 1 ulp (M1 evidence).
  - The `stress_test` tool compares an sd-0 row exactly.
- **Allowing std 0 on drawn edges** would be an engine change with no failing user row today. Not proposed.
