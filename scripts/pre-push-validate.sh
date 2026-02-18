#!/usr/bin/env bash
# Pre-push validation gate for Inference Service Layer
# Runs all checks before allowing a push. Exits non-zero on any failure.
#
# Usage:
#   bash scripts/pre-push-validate.sh            # strict (default)
#   bash scripts/pre-push-validate.sh --lenient   # warn on mypy/test failures
#
# As a git hook, receives: <remote-name> <remote-url> on stdin lines of
#   <local-ref> <local-sha> <remote-ref> <remote-sha>

set -euo pipefail

FAILURES=0
CHECKS_RUN=0
BRANCH=$(git branch --show-current)

# Parse flags — --lenient downgrades mypy/test failures to warnings
LENIENT=false
for arg in "$@"; do
    case "$arg" in
        --lenient) LENIENT=true ;;
    esac
done

header() { printf '\n\033[1;34m===> %s\033[0m\n' "$1"; }
pass()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$1"; }
fail()   { printf '\033[1;31m  ✗ %s\033[0m\n' "$1"; FAILURES=$((FAILURES + 1)); }
warn()   { printf '\033[1;33m  ⚠ %s\033[0m\n' "$1"; }
skip()   { printf '\033[1;33m  ⊘ %s\033[0m\n' "$1"; }

# In lenient mode, mypy/test failures are warnings instead of hard failures
check_fail() {
    if [ "$LENIENT" = true ]; then
        warn "$1 (lenient mode — not blocking)"
    else
        fail "$1"
    fi
}

if [ "$LENIENT" = true ]; then
    printf '\033[1;33mRunning in --lenient mode: mypy/test failures are warnings\033[0m\n'
fi

# ---------------------------------------------------------------------------
# Check 1 — Branch guard
# ---------------------------------------------------------------------------
header "Check 1: Branch guard"
CHECKS_RUN=$((CHECKS_RUN + 1))

if [ "$BRANCH" = "main" ]; then
    fail "Direct push from 'main' branch is blocked. Use a feature branch and PR."
else
    pass "Current branch is '$BRANCH' (not main)"
fi

# If invoked as a git pre-push hook, stdin contains ref info.
# Block pushes that target refs/heads/main.
if [ ! -t 0 ]; then
    while read -r local_ref local_sha remote_ref remote_sha; do
        if [ "$remote_ref" = "refs/heads/main" ]; then
            fail "Push targets refs/heads/main — blocked. (local: $local_ref → remote: $remote_ref)"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Check 2 — Python type checking (mypy)
# ---------------------------------------------------------------------------
header "Check 2: Python type checking (mypy)"
CHECKS_RUN=$((CHECKS_RUN + 1))

if grep -q '\[tool\.mypy\]' pyproject.toml 2>/dev/null; then
    MYPY_CMD=""
    if command -v poetry &>/dev/null; then
        MYPY_CMD="poetry run mypy"
    elif command -v mypy &>/dev/null; then
        MYPY_CMD="mypy"
    fi

    if [ -n "$MYPY_CMD" ]; then
        set +e
        MYPY_OUTPUT=$($MYPY_CMD src/ 2>&1)
        MYPY_EXIT=$?
        set -e

        if [ "$MYPY_EXIT" -eq 0 ]; then
            pass "mypy passed"
        else
            MYPY_ERRORS=$(echo "$MYPY_OUTPUT" | grep -cE '^src/' || echo "0")
            echo "$MYPY_OUTPUT" | tail -5
            check_fail "mypy found $MYPY_ERRORS type error(s)"
        fi
    else
        skip "mypy configured but not installed — install with 'poetry install'"
    fi
else
    skip "mypy not configured in pyproject.toml — skipping type check"
fi

# ---------------------------------------------------------------------------
# Check 3 — Full test suite
# ---------------------------------------------------------------------------
header "Check 3: Full test suite"
CHECKS_RUN=$((CHECKS_RUN + 1))

PYTEST_CMD=""
if command -v poetry &>/dev/null; then
    PYTEST_CMD="poetry run pytest"
elif command -v pytest &>/dev/null; then
    PYTEST_CMD="pytest"
elif command -v python &>/dev/null; then
    PYTEST_CMD="python -m pytest"
fi

if [ -n "$PYTEST_CMD" ]; then
    # Run with ISL_AUTH_DISABLED=true (consistent with repo conventions)
    # Exclude _archived tests (dead code with broken imports)
    set +e
    TEST_OUTPUT=$(ISL_AUTH_DISABLED=true $PYTEST_CMD tests/ --ignore=tests/_archived -q 2>&1)
    TEST_EXIT=$?
    set -e

    echo "$TEST_OUTPUT"

    # Extract pass/fail counts from pytest summary line
    SUMMARY_LINE=$(echo "$TEST_OUTPUT" | grep -E '(passed|failed|error)' | tail -1)
    PASSED_COUNT=0
    FAILED_COUNT=0
    ERRORS_COUNT=0
    if [ -n "$SUMMARY_LINE" ]; then
        PASSED_COUNT=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo "0")
        FAILED_COUNT=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo "0")
        ERRORS_COUNT=$(echo "$SUMMARY_LINE" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' || echo "0")
        printf '  Results: %s passed, %s failed, %s errors\n' \
            "${PASSED_COUNT:-0}" "${FAILED_COUNT:-0}" "${ERRORS_COUNT:-0}"
    fi

    if [ "$TEST_EXIT" -eq 0 ]; then
        pass "Test suite passed — all tests green"
    else
        check_fail "Test suite had failures (exit code $TEST_EXIT: ${PASSED_COUNT:-0} passed, ${FAILED_COUNT:-0} failed, ${ERRORS_COUNT:-0} errors)"
    fi
else
    fail "No pytest runner found — install with 'poetry install'"
fi

# ---------------------------------------------------------------------------
# Check 4 — Dependency audit
# ---------------------------------------------------------------------------
header "Check 4: Dependency audit"
CHECKS_RUN=$((CHECKS_RUN + 1))

DEP_FILES_CHECKED=0
LOCAL_REFS_FOUND=0

# Check all relevant dependency files
for DEP_FILE in pyproject.toml requirements.txt requirements-testing.txt; do
    if [ ! -f "$DEP_FILE" ]; then
        continue
    fi
    DEP_FILES_CHECKED=$((DEP_FILES_CHECKED + 1))

    # Check for local path references: -e ., file:, ../ paths
    if grep -nE '(-e \.|file:|\.\./)' "$DEP_FILE" 2>/dev/null; then
        fail "Local path references found in $DEP_FILE — these will break deployment"
        LOCAL_REFS_FOUND=1
    fi
    # Check for {path = ...} in toml files (Poetry local deps)
    if [[ "$DEP_FILE" == *.toml ]]; then
        if grep -nE '\{.*path\s*=' "$DEP_FILE" 2>/dev/null; then
            fail "Local path dependency found in $DEP_FILE — these will break deployment"
            LOCAL_REFS_FOUND=1
        fi
    fi
done

if [ "$DEP_FILES_CHECKED" -eq 0 ]; then
    fail "No dependency files found (expected pyproject.toml or requirements.txt)"
elif [ "$LOCAL_REFS_FOUND" -eq 0 ]; then
    pass "No local path references in $DEP_FILES_CHECKED dependency file(s)"
fi

# ---------------------------------------------------------------------------
# Check 5 — Python version guard
# ---------------------------------------------------------------------------
header "Check 5: Python version guard"
CHECKS_RUN=$((CHECKS_RUN + 1))

if [ -f "runtime.txt" ]; then
    PY_VERSION=$(tr -d '[:space:]' < runtime.txt)
    printf '  runtime.txt specifies: %s\n' "$PY_VERSION"

    # Extract major.minor — handles "3.11.9", "python-3.11.9", "python-3.11" etc.
    PY_MAJOR_MINOR=$(echo "$PY_VERSION" | grep -oE '[0-9]+\.[0-9]+' | head -1)

    if [ -z "$PY_MAJOR_MINOR" ]; then
        fail "Could not parse Python version from runtime.txt: '$PY_VERSION'"
    else
        case "$PY_MAJOR_MINOR" in
            3.11|3.12)
                pass "Python $PY_MAJOR_MINOR is compatible (NumPy 1.26.4 supported)"
                ;;
            3.13)
                fail "Python 3.13 specified — NumPy 1.26.4 build fails on 3.13. Pin to 3.11 or 3.12."
                ;;
            *)
                fail "Unexpected Python version $PY_MAJOR_MINOR — expected 3.11 or 3.12"
                ;;
        esac
    fi
else
    skip "No runtime.txt found — cannot verify Python version"
fi

# ---------------------------------------------------------------------------
# Check 6 — Summary
# ---------------------------------------------------------------------------
header "Summary"

FILES_CHANGED=""
if git rev-parse --verify origin/staging &>/dev/null; then
    FILES_CHANGED=$(git diff --name-only origin/staging 2>/dev/null | wc -l | tr -d ' ')
else
    FILES_CHANGED="(no origin/staging to compare)"
fi

printf '  Branch:         %s\n' "$BRANCH"
printf '  Files changed:  %s (vs origin/staging)\n' "$FILES_CHANGED"
printf '  Checks run:     %d\n' "$CHECKS_RUN"
printf '  Failures:       %d\n' "$FAILURES"
if [ "$LENIENT" = true ]; then
    printf '  Mode:           lenient\n'
fi

if [ "$FAILURES" -gt 0 ]; then
    printf '\n\033[1;31mPre-push validation FAILED (%d failure(s)). Push blocked.\033[0m\n' "$FAILURES"
    exit 1
else
    printf '\n\033[1;32mAll pre-push checks passed. Safe to push.\033[0m\n'
    exit 0
fi
