# CLAUDE.md — Inference Service Layer (ISL)

## Deployment

- Always push to `staging`. Never push to `main` without explicit user confirmation.
- Run `bash scripts/pre-push-validate.sh` before every push.

## Git workflow

- Run `git status` and `git diff --staged` before committing.
- No simultaneous Claude Code sessions on this repository.

## Session preamble

At the start of every session:

```bash
git branch --show-current && git log --oneline -3 && git status
```

## Testing

- After code changes, run full test suite before committing. Report pass/fail counts.

## Debugging

- ISL uses `from_` for edge field names (Python reserved word avoidance). PLoT translates `from` → `from_` at the boundary. This is intentional — do not "fix" it.
- Request IDs are truncated to 128 characters. If correlation fails with long IDs, check truncation.

## Code review

- Evaluate feedback independently. Do not change correct code to appease reviewers.
