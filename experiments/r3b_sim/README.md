# experiments/r3b_sim: R3-B offline dynamic-simulation evaluation

**NEVER MERGE. NEVER DEPLOY.** This is an evaluation prototype on branch `r3b/sim-prototype`.

- It touches no file outside `experiments/r3b_sim/`.
- It calls no served endpoint and no LLM.
- The only read-only reuse is ISL's EVPPI estimator (`src/utils/evppi.py`), which is loaded by file path.

| File | What it is |
|---|---|
| `EVALUATION.md` | The deliverable: the answer, the per-graph tables and the decision evidence |
| `MAPPING.md`, `mapping/*.json` | Task 1: temporal-semantics audit of every quantity, with evidence and GAPs |
| `ANALYSIS-PLAN.md`, `FROZEN.sha256` | Pre-specified tiers, rules and decision rule, frozen with the mapping before any run |
| `AMENDMENTS.md` | Every change made after the freeze, with its effect |
| `results.json` | Machine-readable results (deterministic); `runtime.json` holds the timings |
| `breakeven.json` | Mode X price → churn break-even for MVP B (A-180910Z, deterministic), from `sim/breakeven.py` |
| `RESEARCH-BRIEF.md` | Brief for an independent research agent: scientific due diligence on representation, time, elicitation, information value and superseding approaches |
| `evppi_gate/` | Study: ISL's EVPPI `resolved` status on factors with known true EVPPI (ISL's own estimator), and a cross-fitted decision-relevance gate. `results.json` is deterministic |
| `corpus/` | The 12 served graphs, copied verbatim from `olumi-programme-docs@58ca52c`, with sha256 values |
| `sim/` | Engine: `static.py` (tiers and taint), `dynamic.py` (monthly stock-flow), `metrics.py` (deadline metrics and EVPPI), `run.py`, `report.py`, `mapping_report.py`, `breakeven.py` |
| `tests/` | Correctness tests: closed forms, held levels, time coherence, withholding mutants, direction, EVPPI validation, convergence |

Run from `experiments/r3b_sim`:

```bash
poetry run python -m sim.run            # results.json + runtime.json (~15 s)
poetry run python -m sim.report         # refresh the generated tables in EVALUATION.md
poetry run python -m sim.breakeven      # breakeven.json (MVP-B illustration, Mode X)
poetry run python -m sim.mapping_report # refresh MAPPING.md
cd ../.. && poetry run pytest experiments/r3b_sim/tests -q -p no:cacheprovider --no-cov
```
