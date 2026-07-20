#!/usr/bin/env bash
# refresh_contract_schema.sh — the ONLY sanctioned way to (re)produce the
# committed contract artifact, and the CI freshness gate over it.
#
#   --check  (default; CI)  re-derive from the pinned ref and FAIL LOUD on any
#                           byte difference vs the committed artifact. Never
#                           writes into the repo — a gate that "fixes" its own
#                           baseline would be self-healing theatre.
#   --write  (maintainer)   re-derive and overwrite the committed artifact.
#                           Use after bumping tests/fixtures/contract-schema/PIN.json.
#
# The pin (repo, ref, expected package version, pinned zod-to-json-schema) is
# read from PIN.json — the single place the contract ref lives.
set -euo pipefail

ISL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIN_FILE="$ISL_ROOT/tests/fixtures/contract-schema/PIN.json"
ARTIFACT="$ISL_ROOT/tests/fixtures/contract-schema/talchain-schemas.json"
GENERATOR="$ISL_ROOT/scripts/contract_schema/generate_contract_schema.mjs"
MODE="${1:---check}"

[ -f "$PIN_FILE" ] || { echo "FATAL: pin file missing at $PIN_FILE" >&2; exit 1; }
[ -f "$GENERATOR" ] || { echo "FATAL: generator missing at $GENERATOR" >&2; exit 1; }

pin() { node -p "require('$PIN_FILE')['$1']"; }
REPO="$(pin repo)"
REF="$(pin ref)"
EXPECTED_VERSION="$(pin package_version_expected)"
Z2JS_VERSION="$(pin zod_to_json_schema_version)"

echo "Contract pin: $REPO @ $REF (expect $EXPECTED_VERSION, zod-to-json-schema@$Z2JS_VERSION)"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

git clone --quiet "https://github.com/$REPO.git" "$WORKDIR/schemas"
git -C "$WORKDIR/schemas" checkout --quiet "$REF"

(
  cd "$WORKDIR/schemas"
  npm ci --silent
  npm run build >/dev/null
  npm install --no-save --silent "zod-to-json-schema@$Z2JS_VERSION"
)

GENERATED="$WORKDIR/talchain-schemas.json"
node "$GENERATOR" --schemas-dir "$WORKDIR/schemas" --ref "$REF" --out "$GENERATED"

ACTUAL_VERSION="$(node -p "require('$GENERATED')._meta.package_version")"
if [ "$ACTUAL_VERSION" != "$EXPECTED_VERSION" ]; then
  echo "FATAL: package version at pinned ref is $ACTUAL_VERSION but PIN.json expects $EXPECTED_VERSION." >&2
  echo "Update package_version_expected in PIN.json deliberately — do not assume." >&2
  exit 1
fi

case "$MODE" in
  --write)
    cp "$GENERATED" "$ARTIFACT"
    echo "WROTE $ARTIFACT from $REPO@$REF. Commit it together with any PIN.json change."
    ;;
  --check)
    if ! cmp -s "$GENERATED" "$ARTIFACT"; then
      echo "" >&2
      echo "CONTRACT ARTIFACT DRIFT — the committed talchain-schemas.json does NOT match" >&2
      echo "what the pinned ref derives. The committed copy is a hand-maintained mirror" >&2
      echo "until this is fixed. Byte diff (derived vs committed):" >&2
      diff "$GENERATED" "$ARTIFACT" | head -60 >&2 || true
      echo "" >&2
      echo "Fix: run scripts/contract_schema/refresh_contract_schema.sh --write and commit," >&2
      echo "or restore the correct PIN.json. NEVER hand-edit the artifact." >&2
      exit 1
    fi
    echo "Freshness gate PASS: committed artifact is byte-identical to the pinned derivation."
    ;;
  *)
    echo "Unknown mode: $MODE (use --check or --write)" >&2
    exit 2
    ;;
esac
