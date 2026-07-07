# Science-validation harness

Empirical validation of ISL's robustness-v2 science: Higher-K resolution of
`marginal_switch_probability`, the EVPI noise floor, determinism at scale,
and calibration groundwork for `probability_of_goal`.

Findings live in [`docs/science-validation/REPORT.md`](../../docs/science-validation/REPORT.md).
Raw outputs (with provenance: command, git SHA, pinned seeds) are committed
under [`results/`](results/).

## Running

```bash
poetry install                                                     # once
poetry run python benchmarks/science-validation/run_all.py --quick # smoke, ~2 min
poetry run python benchmarks/science-validation/run_all.py         # full,  ~20-30 min
```

Each experiment also runs standalone, e.g.:

```bash
poetry run python benchmarks/science-validation/exp1_higher_k.py
```

All seeds are pinned in the `SEEDS` registry in `_lib.py`. Re-running an
experiment reproduces the committed `results/*.json` bit-for-bit apart from
the `provenance` block (timestamp/SHA) and `elapsed_seconds`.

## Contents

| File | Purpose |
|------|---------|
| `_lib.py` | Seed registry, harness-level K override, request builders, canonical serialisation, result I/O |
| `graphs.py` | Margin family (analytic flip probabilities), EVPI graphs, random DAGs, analytic calibration cases, repo-fixture loader |
| `exp1_higher_k.py` | K sweep {100, 1000, 10000, 100000}; per-edge TRUE ZERO / UNDER-RESOLUTION / UNSTABLE classification |
| `exp2_evpi_floor.py` | Empirical validation of the z95 worst-case EVPI noise floor at n {500, 2000, 10000} |
| `exp3_determinism.py` | Same-seed byte-equality across 50 diverse graphs (in-process, wire, cross-process); volatile-field catalogue |
| `exp4_calibration.py` | probability_of_goal vs closed-form truth; auto-noise distortion quantified |
| `run_all.py` | One-command runner |
| `STATUS.md` | Session status for the orchestrator |

## Design constraints honoured

- **No `src/` changes.** K is instrumented by wrapping
  `_compute_marginal_switch_probability` at harness level and forwarding
  `k_samples` (the production default is bound at function-definition time,
  so patching the `MARGINAL_K_SAMPLES` constant would be a no-op). A guard in
  exp1 verifies the wrapper is transparent at K=100.
- **Production estimators throughout.** exp2 re-implements only the thin
  orchestration of `_compute_evpi` (production caps EVPI at
  n = min(n_samples, 500), making n = 2000/10000 unreachable via `analyze()`);
  the metric itself is the unmodified `_compute_evpi_metric`, with identical
  seed derivations.
- **Local only.** Everything runs in-process or via `TestClient`; no deployed
  service is touched.
- This directory is not collected by pytest (`testpaths = ["tests"]`) and is
  outside the mypy gate (`mypy src/`); pre-commit black/ruff do apply.
