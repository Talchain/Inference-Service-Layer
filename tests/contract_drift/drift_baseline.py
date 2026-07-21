"""Loader for the ISL ↔ @talchain/schemas two-way DRIFT BASELINE.

``drift_baseline.json`` is the ratchet baseline of KNOWN, ACCEPTED omissions
(contract fields ISL does not emit) and supersets (ISL-emitted keys the paired
contract schema lacks). It is the omission/superset analogue of
``allowlist.py`` for collisions: everything is derived (the contract side from
the pinned artifact, the ISL side from the Pydantic models), and any drift NOT
in the baseline fails CI loud.

DERIVE-DON'T-MIRROR: this file only READS the committed JSON. The JSON is
(re)generated exclusively by
``scripts/contract_schema/refresh_drift_baseline.py --refresh-baseline`` — a
human-run step. Nothing in the test path ever writes it (no self-heal), so the
gate cannot ratify drift by rewriting its own baseline.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Tuple

BASELINE_PATH = Path(__file__).resolve().parent / "drift_baseline.json"


@dataclass(frozen=True)
class OmissionEntry:
    """A contract field ISL does not emit, accepted into the baseline."""

    isl_model: str
    contract_location: str
    prop: str
    required: bool
    reason: str

    def key(self) -> Tuple[str, str, str]:
        return (self.isl_model, self.contract_location, self.prop)


@dataclass(frozen=True)
class SupersetEntry:
    """An ISL-emitted key absent from the paired contract schema."""

    isl_model: str
    prop: str
    reason: str

    def key(self) -> Tuple[str, str]:
        return (self.isl_model, self.prop)


def _load() -> Dict[str, Any]:
    with BASELINE_PATH.open() as fh:
        return json.load(fh)


_data = _load()
BASELINE_META: Dict[str, Any] = _data.get("_meta", {})

OMISSION_BASELINE: FrozenSet[OmissionEntry] = frozenset(
    OmissionEntry(
        e["isl_model"], e["contract_location"], e["prop"], bool(e["required"]), e["reason"]
    )
    for e in _data["omissions"]
)
SUPERSET_BASELINE: FrozenSet[SupersetEntry] = frozenset(
    SupersetEntry(e["isl_model"], e["prop"], e["reason"]) for e in _data["supersets"]
)

OMISSION_BASELINE_KEYS: FrozenSet[Tuple[str, str, str]] = frozenset(
    e.key() for e in OMISSION_BASELINE
)
SUPERSET_BASELINE_KEYS: FrozenSet[Tuple[str, str]] = frozenset(e.key() for e in SUPERSET_BASELINE)
