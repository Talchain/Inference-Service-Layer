# CLAUDE.md — Inference Service Layer (ISL)

## Deployment

- Always push to `staging`. Never push to `main` without explicit user confirmation.
- Run `bash scripts/pre-push-validate.sh` before every push. This script covers branch guard, mypy, pytest, dependency audit, and Python version checks.
- Before committing, run `git status` and `git diff --staged` to verify ONLY intended changes are staged.
- If there are uncommitted changes from previous sessions, flag them and get user approval before including.
- Actually execute every git command — do not present commands as a summary without running them.
- After push, verify it succeeded by checking the output.
- Never bundle unrelated uncommitted changes into a deployment commit.

## Git workflow

- No simultaneous Claude Code sessions on this repository.

## Session preamble

At the start of every session, before any other work:

```bash
# 1. Branch and recent history
git branch --show-current && git log --oneline -5 && git status

# 2. Check for uncommitted changes or stash entries
git stash list
```

Report the output. If unexpected uncommitted changes or stash entries exist, flag them before proceeding.

Confirm the branch is correct for the task before starting any work.

## Testing — Three-Tier Process

Testing uses a tiered approach to avoid heavy resource usage on the local machine.
The full suite runs in the pre-push hook and CI — not after every code change.

### Tier 1: Smoke (after every code change)

Run **only** after making changes, before reporting the task as done.
Targets type checking and changed-file tests — fast and light.

```bash
poetry run mypy src/                                  # type checking
poetry run pytest --co -q                             # collect only — verify nothing is broken
poetry run pytest tests/path/to/changed_test.py -x    # only tests related to changes
```

If no test files are directly related to the change, `mypy` alone is sufficient.
Report: "mypy passed. N related tests passed." (or "No related tests for this change.")

### Tier 2: Pre-commit validation

Run before committing. Still lightweight — no full test suite.

```bash
poetry run mypy src/
poetry run black --check src/
```

### Tier 3: Full gate (before pushing to staging only)

Run **only** when the user explicitly says to push to staging.
The pre-push hook (`scripts/pre-push-validate.sh`) handles this automatically.

```bash
git push origin staging    # triggers pre-push hook which runs full suite
```

### Important rules

- **Never run the full pytest suite after every code change** — save it for the pre-push gate.
- The pre-push hook runs mypy, pytest, and all other checks automatically.
- CI is the authoritative gate — local testing is a fast feedback loop, not a replacement.

## Debugging

- ISL uses `from_` for edge field names (Python reserved word avoidance). PLoT translates `from` → `from_` at the boundary. This is intentional — do not "fix" it.
- Request IDs are truncated to 128 characters. If correlation fails with long IDs, check truncation.

### Data flow tracing (mandatory before any fix)

Before implementing any bug fix or feature that touches data flowing between services, trace and document the complete path:

1. Where does the data originate? (CEE LLM response? ISL computation? PLoT assembly?)
2. List every transform/adapter layer it passes through (with file paths)
3. Where is it consumed in the final response?
4. Are there alternate code paths or error shapes? (e.g., direct error vs PLoT-wrapped error, V2 vs V3 adapter)

Only after the trace is documented, implement fixes at ALL affected layers. Do not fix one layer and assume others are correct.

Common multi-layer patterns involving ISL:
- PLoT → ISL request: field name translations (`from` → `from_` for Python reserved words)
- ISL computation → response: Pydantic model serialisation with `by_alias` for outbound field names
- Error responses: ISL error shapes must be consistent whether returned directly or wrapped by PLoT

## Code review analysis

When asked to address code review feedback:

1. Read ALL feedback items first before making any changes
2. For each item, determine independently:
   - Is the feedback valid and does it require a code change?
   - Is it already handled by existing code?
   - Is it incorrect or based on a misunderstanding of the architecture?
3. State your reasoning for each determination before making changes
4. Do not change correct code to appease reviewers
5. Group changes by affected file to minimise unnecessary edits

## Task completion checklist

Before reporting ANY task as complete, run the **Tier 1 smoke checks** (not the full suite):

```bash
git branch --show-current                              # Correct branch?
git status                                             # Clean state?
poetry run mypy src/                                   # Type checking passes?
poetry run pytest tests/path/to/changed_test.py -x     # Related tests pass?
```

If mypy or related tests fail, fix before reporting completion.
Do NOT run the full pytest suite or `pre-push-validate.sh` here — those run in the pre-push
hook when the user decides to push, and again in CI. See "Testing — Three-Tier Process" above.
