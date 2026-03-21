# AGENTS.md — Inference Service Layer (ISL)

Instructions for AI code-review agents (Codex, Gemini, etc.).

## Review scope

- **Review the full PR diff against the base branch**, not individual commits.
  Commits may be incremental; only the final merged state matters.
- Do not flag issues that exist only in the base branch (pre-existing problems are out of scope unless the PR makes them worse).

## Architecture context

ISL is a FastAPI service that performs causal inference and robustness analysis.
Key architectural patterns to understand before reviewing:

### Dual model versioning (V1 / V2)

- **V1 models** (`src/models/robustness_v2.py`): internal computation models used by the analyzer.
- **V2 models** (`src/models/response_v2.py`): API response models returned to clients.
- The API layer (`src/api/robustness.py`) converts V1 → V2 inline. This is intentional —
  conversion exists at a single call site, so `from_v1` classmethods are unnecessary abstraction.
- Both model versions must stay in sync. When reviewing new fields, verify they appear in:
  V1 model, V2 model, response builder (`src/utils/response_builder.py`), and API conversion.

### Field naming

- `from_` is used for edge field names (Python reserved word). PLoT translates `from` → `from_` at the boundary. Do **not** flag this as a bug.
- Pydantic models use `by_alias` for outbound serialization.

### Monte Carlo methodology

- `winner_per_sample` uses random tie-breaking. This is statistically valid in MC simulation — by the law of large numbers, probabilities converge across thousands of samples.
- All analysis layers (overall, per-factor conditional, EVPI) must use the **same** `winner_per_sample` for consistency. Do not suggest different tie-handling for subsets.

### OpenAPI specification

- `openapi.json` is auto-generated from the FastAPI app via `scripts/generate_openapi.py`.
- If a PR adds new response models/fields, `openapi.json` must be regenerated.
- The CI workflow `.github/workflows/openapi-validation.yml` enforces this.

## What to flag

- Bugs, logic errors, security issues (OWASP top 10).
- Missing error handling at system boundaries (user input, external APIs).
- Breaking changes to the API contract without version bump.
- Missing test coverage for new logic branches.
- Type errors that mypy would catch.

## What not to flag

- **Style-only changes** (e.g., `dict.get(k, 0) + 1` vs `defaultdict(int)`) — both are idiomatic Python.
- **Local imports in test files** — deliberate pattern for integration-style tests that access internals.
- **Single-use inline code** — do not suggest abstractions (helpers, classmethods, factories) for code used in exactly one place.
- **Pre-existing issues** in unchanged code — out of scope for PR review.
- **Premature generalization** — do not suggest designing for hypothetical future requirements.

## Testing

- Tests live in `tests/unit/` and `tests/integration/`.
- The project uses pytest with `ISL_AUTH_DISABLED=true` for local runs.
- Full suite runs in CI; PR reviews should not suggest running the full suite locally.
