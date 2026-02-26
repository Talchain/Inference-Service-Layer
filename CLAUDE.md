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

## Testing

- After code changes, run full test suite before committing. Report pass/fail counts.

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

Before reporting ANY task as complete, run `bash scripts/pre-push-validate.sh` and show its output. Additionally verify:

```bash
# 1. Clean state? (no accidental uncommitted changes)
git status

# 2. Recent commits match the work just done?
git log --oneline -5

# 3. Formatting passes?
poetry run black --check src/
```

If any check fails, fix it before reporting completion. Do not report "done" with failing tests or uncommitted changes unless explicitly discussed with the user.
