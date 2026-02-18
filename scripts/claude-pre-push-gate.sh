#!/usr/bin/env bash
# Claude Code pre-push gate
# Runs pre-push validation before Claude Code pushes via Bash tool.
# Referenced by .claude/settings.json PreToolUse hook.

set -euo pipefail

exec bash "$(git rev-parse --show-toplevel)/scripts/pre-push-validate.sh"
