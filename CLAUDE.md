# ISL — Inference-Service-Layer

The platform's compute layer — the real math. Robustness Monte Carlo is the live
computation; EVPI and the broader science surface are largely dormant (verify a path is
actually wired before relying on it). Python, poetry, Pydantic. Called by PLoT.
Platform map + cross-cutting hazards: repo-parent `CLAUDE.md`. Global git/verify
practice: `~/.claude/CLAUDE.md`.

## Service specifics

- **ISL owns the `from`→`from_` edge-field alias** (Python reserved word). The inbound model
  declares `from_: str = Field(alias="from")` with `populate_by_name=True` (`src/models/robustness_v2.py`),
  so ISL accepts the wire field `from` and maps it. PLoT forwards `from` unchanged — this is
  intentional, do not "fix" it.
- Outbound serialisation uses Pydantic `by_alias` to emit the wire field names — check it
  when an output field name looks wrong.
- **Request IDs are truncated to 128 characters.** If correlation fails with long IDs, check truncation first.

## Commands & gates

```bash
poetry run mypy src/                                  # type check
poetry run black --check src/                         # format
poetry run pytest tests/path/to/changed_test.py -x    # related tests
bash scripts/pre-push-validate.sh                     # full gate (branch guard, mypy, pytest, dep audit, Python version)
```

- **Tier 1 (after every change):** `poetry run mypy src/` + the related `pytest` file. No related tests → mypy alone. (`poetry run pytest --co -q` to confirm collection isn't broken.)
- **Tier 2 (before commit):** `poetry run mypy src/` + `poetry run black --check src/`.
- **Tier 3 (push to staging only):** the pre-push hook runs `scripts/pre-push-validate.sh` automatically. Don't run the full pytest suite after every change — CI is the authoritative gate.

## Deploy

Push to `staging` by default; **never `main` without explicit confirmation.**

## Working rules (shared doctrine, condensed)

- **Data-flow tracing before any cross-boundary fix:** origin (CEE LLM / ISL compute / PLoT assembly) → every transform (with paths) → consumer → alternate shapes (direct error vs PLoT-wrapped error). Fix *all* affected layers, not the first found.
- **Code-review feedback:** read every item first, judge each independently (valid / already-handled / misunderstanding), state reasoning, don't change correct code to appease.
