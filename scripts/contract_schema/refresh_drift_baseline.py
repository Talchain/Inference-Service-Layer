#!/usr/bin/env python
"""(Re)generate the ISL ↔ @talchain/schemas two-way DRIFT BASELINE.

This is the ONLY sanctioned way to (re)produce
``tests/contract_drift/drift_baseline.json`` — the ratchet baseline of KNOWN,
ACCEPTED omissions (contract fields ISL does not emit) and supersets
(ISL-emitted keys the contract pair lacks). Any drift NOT in that baseline
fails CI loud (tests/contract_drift/test_contract_drift.py).

WHY A SCRIPT, NOT SELF-HEAL
---------------------------
The drift TEST must never rewrite the baseline — a check that "repairs" its own
baseline is the F6-class self-healing defect: it would ratify whatever drift
just landed and stay green forever. So regeneration is an explicit, human-run
step (this script), exactly mirroring how
``scripts/contract_schema/refresh_contract_schema.sh --write`` is the only way
to move the contract artifact. The baseline side (ISL Pydantic models) is
derived from the models; the contract side from the pinned artifact — nothing
here is hand-typed.

USAGE
-----
    poetry run python scripts/contract_schema/refresh_drift_baseline.py --refresh-baseline

Run it ONLY when a drift change is intentional and reviewed. The commit that
runs it carries the decision reference (the review record / PR), and each
generated entry gets a derived one-line reason. Refuses to run under pytest so
the gate can never trigger a regeneration.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

from pathlib import Path

# Absolute anchor (trap #15: absolute paths, assert the anchor exists, verify
# the write landed). This file lives at <ISL_ROOT>/scripts/contract_schema/.
ISL_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ISL_ROOT / "tests" / "contract_drift" / "drift_baseline.json"

# Make the ISL package importable when run as a bare script.
if str(ISL_ROOT) not in sys.path:
    sys.path.insert(0, str(ISL_ROOT))


def _omission_reason(contract_location: str, prop: str, required: bool) -> str:
    if required:
        return (
            f"REQUIRED in contract ({contract_location}) — ISL does not emit "
            f"{prop}; confirm PLoT synthesises it, else it is a silent drop at source"
        )
    return (
        f"optional in contract ({contract_location}) — ISL omits {prop} "
        "(PLoT may synthesise it, or the contract has it ahead of ISL)"
    )


def _superset_reason(isl_model: str, prop: str) -> str:
    return (
        f"ISL superset — {isl_model}.{prop} is not present in the paired "
        "contract schema (ISL emits ahead of contract adoption)"
    )


def build_baseline() -> dict:
    from tests.contract_drift.drift_core import run_drift_check

    report = run_drift_check()  # reads the committed artifact + live models
    meta = dict(report.contract_meta)

    omissions = [
        {
            "isl_model": model,
            "contract_location": contract_label,
            "prop": prop,
            "required": bool(required),
            "reason": _omission_reason(contract_label, prop, bool(required)),
        }
        # report.omissions is already deterministically sorted
        for (model, contract_label, prop, required) in report.omissions
    ]
    supersets = [
        {
            "isl_model": model,
            "prop": prop,
            "reason": _superset_reason(model, prop),
        }
        for (model, prop) in report.supersets
    ]

    return {
        "_meta": {
            "purpose": (
                "Ratchet baseline of KNOWN two-way drift (omissions + supersets) "
                "between ISL V2 response models and @talchain/schemas. Any drift "
                "NOT listed here fails CI loud. Add a new entry ONLY with a "
                "decision ref by re-running the refresh script on an intentional "
                "change; otherwise fix the drift."
            ),
            "generated_by": (
                "scripts/contract_schema/refresh_drift_baseline.py --refresh-baseline"
            ),
            "source_repo": meta.get("source_repo"),
            "source_ref": meta.get("source_ref"),
            "package_name": meta.get("package_name"),
            "package_version": meta.get("package_version"),
            "omission_count": len(omissions),
            "superset_count": len(supersets),
        },
        "omissions": omissions,
        "supersets": supersets,
    }


def main(argv: list[str]) -> int:
    if "pytest" in sys.modules:
        print(
            "REFUSING to regenerate the baseline from within pytest — the drift "
            "gate must never self-heal. Run this script by hand.",
            file=sys.stderr,
        )
        return 2
    if "--refresh-baseline" not in argv:
        print(__doc__)
        print("Pass --refresh-baseline to (re)write the committed baseline.", file=sys.stderr)
        return 2

    # Auth env is irrelevant to model/artifact import, but set the same flag CI
    # uses so importing the ISL package never trips the API-key guard.
    os.environ.setdefault("ISL_AUTH_DISABLED", "true")

    assert BASELINE_PATH.parent.is_dir(), f"anchor dir missing: {BASELINE_PATH.parent}"

    payload = build_baseline()
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    BASELINE_PATH.write_text(serialized, encoding="utf-8")

    # Verify the write actually landed (trap #15: a tool that cannot fail is
    # theatre) — read it back and compare digests.
    written = BASELINE_PATH.read_text(encoding="utf-8")
    if (
        hashlib.sha256(written.encode()).hexdigest()
        != hashlib.sha256(serialized.encode()).hexdigest()
    ):
        print("FATAL: baseline write did not land as expected", file=sys.stderr)
        return 1

    print(
        f"Wrote {BASELINE_PATH.relative_to(ISL_ROOT)}: "
        f"{payload['_meta']['omission_count']} omissions, "
        f"{payload['_meta']['superset_count']} supersets "
        f"(contract {payload['_meta'].get('package_version')} @ "
        f"{str(payload['_meta'].get('source_ref'))[:12]})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
