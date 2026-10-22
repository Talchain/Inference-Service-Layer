# Track 3 stress-testing runner (isolated prototype, Milestones 1–3)

Runs brief v1.0 §6 G1–G4 through a **local** PLoT `/v2/run` → capture proxy → local ISL, beside an exact rational
oracle, then runs the negative controls and mutants. Not imported by ISL; changes no product code, schema, lockfile
or default (`src/`, `poetry.lock` and `pyproject.toml` are identical to the ISL pin).

Setup (once): fresh clones at `$T3_ROOT/plot` (PLoT acf24764, `npm ci --ignore-scripts`) and `$T3_ROOT/isl`
(ISL 8d862f5a, or this branch), plus `$T3_ROOT/isl-venv` (Python 3.11, locked main deps from `poetry.lock`).

## The demonstration: one command

Starts and stops the engine pair itself, loopback only, no provider calls:

    cd experiments/track3_stress && T3_ROOT=<root> $T3_ROOT/isl-venv/bin/python -m track3.run --out <fresh dir>

It prints one line (`verdict`, `clean_reference_failures`, `gaps`, `controls_as_expected`, `plot_calls`, `wall_s`) and
writes `report.md` (joint failures, hypothesis disagreement, margins, worst level and regret; no winner) and
`results.json` (every cell, control and per-call timing). Last executed: 22 Oct 2026 09:39Z, `evidence/run-20261022T093910Z-m3/`:
**PASS, 0 clean-reference failures, 0 gaps, 24/24 controls as expected, 86 PLoT calls, 180 s.**

Tests (no engine, under 1 s): `$T3_ROOT/isl-venv/bin/python -m unittest tests.test_track3` (26 tests, OK on 22 Oct).

## Runtime profile (measured, 22 Oct run; Apple silicon laptop, load ~8)

| group | real PLoT calls | engine s | median s per call | non-200 (designed refusals) |
|---|---|---|---|---|
| G1/G2 | 15 | 28.7 | 1.94 | 0 |
| G1/G2 controls and mutants | 50 | 98.0 | 1.90 | 0 |
| G3 | 4 | 6.0 | 1.92 | 1 (invalid ρ 1.2: 422, blocked) |
| G3 controls | 2 | 4.3 | 2.14 | 0 |
| G4 | 13 | 27.5 | 2.04 | 0 |
| G4 controls | 2 | 0.0 | — | 2 (injected faults: unsupported condition → PLoT 422 before ISL; simulated timeout → failed) |
| **total** | **86** | **164.5** | **1.96** | 3 |

- Wall time 180 s; peak resident memory of the runner process 148 MiB (`/usr/bin/time -l`). The PLoT and ISL engine
  processes are separate and their memory was not measured.
- 117 further evaluations were served from the complete-identity cache (not counted above).
- Cost scales as hypotheses × settings × options × 1 call (about 2 s each). Proposed ceiling for one user request:
  16 calls and 60 s, with cancellation that keeps partial rows marked incomplete.

## Explanations

Every vulnerability is a direct evaluated contrast witness (baseline passes, each stress alone passes, both together
fail), linked to its rows. No learned rule was built (optional under brief §5), so the Milestone 2 validation of a rule on
fresh contrast points does not apply.

## Integration

See `INTEGRATION-PROPOSAL.md`: no PLoT extension is required; the exact user control, the display fields, the owners,
the rollback and the journey that would prove delivery.
