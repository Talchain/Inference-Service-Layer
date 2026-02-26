#!/usr/bin/env bash
# Install/verify git hooks for Inference Service Layer
# Usage: bash scripts/install-hooks.sh

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOK_DIR="$REPO_ROOT/.git/hooks"
PRE_PUSH_HOOK="$HOOK_DIR/pre-push"
SCRIPT_PATH="$REPO_ROOT/scripts/pre-push-validate.sh"

header() { printf '\033[1;34m===> %s\033[0m\n' "$1"; }
pass()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$1"; }
fail()   { printf '\033[1;31m  ✗ %s\033[0m\n' "$1"; }

header "Git hook installation"

# Verify the validation script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    fail "scripts/pre-push-validate.sh not found"
    exit 1
fi

# Ensure script is executable
chmod +x "$SCRIPT_PATH"
pass "scripts/pre-push-validate.sh is executable"

# Create or update pre-push hook
# Hook runs in --lenient mode by default (pre-existing mypy/test issues).
# Remove --lenient once pre-existing issues are resolved for strict enforcement.
HOOK_CONTENT='#!/usr/bin/env bash
# Pre-push hook — runs ISL validation gate
# Installed by scripts/install-hooks.sh
# Using --lenient until pre-existing mypy/test issues are resolved.
# Remove --lenient for strict enforcement.
exec bash "$(git rev-parse --show-toplevel)/scripts/pre-push-validate.sh" --lenient'

if [ -f "$PRE_PUSH_HOOK" ]; then
    if grep -q "pre-push-validate.sh" "$PRE_PUSH_HOOK"; then
        # Update in place to pick up any flag changes
        printf '%s\n' "$HOOK_CONTENT" > "$PRE_PUSH_HOOK"
        chmod +x "$PRE_PUSH_HOOK"
        pass "pre-push hook updated"
    else
        # Back up existing hook
        cp "$PRE_PUSH_HOOK" "$PRE_PUSH_HOOK.backup"
        printf '%s\n' "$HOOK_CONTENT" > "$PRE_PUSH_HOOK"
        chmod +x "$PRE_PUSH_HOOK"
        pass "pre-push hook installed (previous hook backed up to pre-push.backup)"
    fi
else
    printf '%s\n' "$HOOK_CONTENT" > "$PRE_PUSH_HOOK"
    chmod +x "$PRE_PUSH_HOOK"
    pass "pre-push hook installed"
fi

# Set up pre-commit framework (existing hook system for pre-commit hooks)
header "Pre-commit framework"
if command -v pre-commit &>/dev/null; then
    pass "pre-commit is installed"
    if [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
        pass ".pre-commit-config.yaml exists"
        # Ensure pre-commit hooks are installed in .git/hooks/pre-commit
        pre-commit install --allow-missing-config 2>/dev/null && \
            pass "pre-commit hooks installed" || \
            printf '  ⊘ pre-commit install skipped (run manually if needed)\n'
    else
        fail ".pre-commit-config.yaml not found"
    fi
else
    printf '  ⊘ pre-commit not installed (optional — pre-push hook works independently)\n'
fi

header "Verification complete"
printf '  Hook path: %s\n' "$PRE_PUSH_HOOK"
printf '  Script:    %s\n' "$SCRIPT_PATH"
printf '  Mode:      --lenient (remove flag in hook for strict enforcement)\n'
