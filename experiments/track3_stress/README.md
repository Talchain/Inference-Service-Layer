# Track 3 stress-testing runner (isolated prototype, Milestone 1)

Runs brief v1.0 §6 G1 and the eight G2 settings through a **local** PLoT `/v2/run` →
capture proxy → local ISL, beside an exact rational oracle, then runs the negative controls.
Not imported by ISL; changes no product code, schema, lockfile or default.

Setup (once): fresh clones at `$T3_ROOT/plot` (PLoT acf24764, `npm ci --ignore-scripts`) and
`$T3_ROOT/isl` (ISL 8d862f5a), plus `$T3_ROOT/isl-venv` (Python 3.11, locked main deps from `poetry.lock`).

One command (starts and stops the engine pair itself, loopback only, no provider calls):

    cd experiments/track3_stress && $T3_ROOT/isl-venv/bin/python -m track3.run --out <fresh dir>

Pure tests (no engine): `$T3_ROOT/isl-venv/bin/python -m unittest tests.test_track3`.
Evidence of the executed run is in `evidence/`.
