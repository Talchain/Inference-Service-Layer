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
HOOK_CONTENT='#!/usr/bin/env bash
# Pre-push hook — runs ISL validation gate
# Installed by scripts/install-hooks.sh
exec bash "$(git rev-parse --show-toplevel)/scripts/pre-push-validate.sh"'

if [ -f "$PRE_PUSH_HOOK" ]; then
    if grep -q "pre-push-validate.sh" "$PRE_PUSH_HOOK"; then
        pass "pre-push hook already installed and points to validation script"
    else
        # Back up existing hook
        cp "$PRE_PUSH_HOOK" "$PRE_PUSH_HOOK.backup"
        printf '%s\n' "$HOOK_CONTENT" > "$PRE_PUSH_HOOK"
        chmod +x "$PRE_PUSH_HOOK"
        pass "pre-push hook updated (previous hook backed up to pre-push.backup)"
    fi
else
    printf '%s\n' "$HOOK_CONTENT" > "$PRE_PUSH_HOOK"
    chmod +x "$PRE_PUSH_HOOK"
    pass "pre-push hook installed"
fi

# Verify pre-commit framework (existing hook system)
header "Pre-commit framework"
if command -v pre-commit &>/dev/null; then
    pass "pre-commit is installed"
    if [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
        pass ".pre-commit-config.yaml exists"
    else
        fail ".pre-commit-config.yaml not found"
    fi
else
    printf '  ⊘ pre-commit not installed (optional — pre-push hook works independently)\n'
fi

header "Verification complete"
printf '  Hook path: %s\n' "$PRE_PUSH_HOOK"
printf '  Script:    %s\n' "$SCRIPT_PATH"
