"""CI drift check: ISL V2 response models vs the @talchain/schemas contract.

See tests/contract_drift/drift_core.py for the design and
tests/fixtures/contract-schema/PIN.json for the pinned contract ref.

ORDERING NOTE (trap #13 — a check that cannot fail is theatre): the positive
controls come FIRST in this file. Before any absence assertion ("no
un-allowlisted collisions") is trusted, the same engine must demonstrably SEE
a deliberately mutated schema/model pair. If the positive controls fail,
every green below them is void.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys

from pathlib import Path

import pytest

from tests.contract_drift.allowlist import ALLOWLIST, ALLOWLISTED_KEYS
from tests.contract_drift.drift_core import (
    ARTIFACT_PATH,
    PAIRINGS,
    PIN_PATH,
    load_artifact,
    load_pin,
    pydantic_properties,
    run_drift_check,
)


def _fixture_digests() -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ARTIFACT_PATH, PIN_PATH)}


# ---------------------------------------------------------------------------
# 1. POSITIVE CONTROLS — prove the differ can SEE, before trusting any green.
# ---------------------------------------------------------------------------


class TestPositiveControls:
    """Feed deliberately mutated schema/model pairs; the diff MUST report
    them. Mutated keys are chosen to NOT be in the allowlist, proving the
    allowlist cannot swallow a new defect."""

    def test_control_sees_new_contract_field_as_omission(self) -> None:
        artifact = copy.deepcopy(load_artifact())
        target = artifact["modules"]["index"]["FactorSensitivitySchema"]
        target["properties"]["drift_control_new_contract_field"] = {"type": "string"}
        report = run_drift_check(artifact=artifact)
        assert (
            "FactorSensitivityV2",
            "index.FactorSensitivitySchema",
            "drift_control_new_contract_field",
            False,
        ) in report.omissions, "differ failed to see an added contract field"

    def test_control_sees_removed_contract_field_as_superset(self) -> None:
        artifact = copy.deepcopy(load_artifact())
        target = artifact["modules"]["index"]["FactorSensitivitySchema"]
        removed = target["properties"].pop("elasticity")
        assert removed is not None
        report = run_drift_check(artifact=artifact)
        assert (
            "FactorSensitivityV2",
            "elasticity",
        ) in report.supersets, "differ failed to see an ISL key the contract no longer carries"

    def test_control_sees_pair_enum_domain_change_as_collision(self) -> None:
        """Narrow the contract's inference-warning severity enum out from
        under ISL's — the within-pair rule must fire, and the allowlist must
        NOT swallow it (the mutated domain differs from the allowlisted
        real-world collision only in the contract's live bytes, so we assert
        on the unfiltered report)."""
        artifact = copy.deepcopy(load_artifact())
        target = artifact["modules"]["boundary"]["EnrichmentConstraintResultSchema"]
        # operator is an OPEN string in the live contract; close it to a
        # vocabulary disjoint from ISL's {'>=', '<='}.
        target["properties"]["operator"] = {"type": "string", "enum": ["gt", "lt"]}
        report = run_drift_check(artifact=artifact)
        hits = [
            c
            for c in report.collisions
            if c.isl_model == "ConstraintResultV2" and c.isl_key == "operator"
        ]
        assert hits, "differ failed to see a pair enum-domain change"
        assert all(
            c.key() not in ALLOWLISTED_KEYS for c in hits
        ), "allowlist must not cover a NEW collision"

    def test_control_sees_cross_model_disjoint_enum_as_collision(self) -> None:
        """Plant a same-named key with a disjoint closed enum in an unrelated
        contract model — the confidence_source class. Must fail loud."""
        artifact = copy.deepcopy(load_artifact())
        target = artifact["modules"]["boundary"]["DecisionRecordDecisionSchema"]
        target.setdefault("properties", {})["flip_direction"] = {
            "type": "string",
            "enum": ["up", "down"],
        }
        report = run_drift_check(artifact=artifact)
        hits = [
            c
            for c in report.collisions
            if c.isl_key == "flip_direction"
            and c.contract_location == "boundary.DecisionRecordDecisionSchema.flip_direction"
        ]
        assert hits, "differ failed to see a cross-model disjoint-enum collision"
        assert all(c.key() not in ALLOWLISTED_KEYS for c in hits)

    def test_control_sees_isl_side_mutation(self) -> None:
        """Mutate the ISL side (injected property map): an ISL enum widened
        beyond the contract's closed domain must collide."""
        isl_props = {model.__name__: pydantic_properties(model) for model in PAIRINGS}
        from tests.contract_drift.drift_core import TypeSummary

        isl_props["ISLResponseV2"] = dict(isl_props["ISLResponseV2"])
        isl_props["ISLResponseV2"]["robustness_status"] = TypeSummary(
            kinds=frozenset({"string"}),
            enum=frozenset({"computed", "skipped", "unavailable", "error", "exploded"}),
        )
        report = run_drift_check(isl_properties=isl_props)
        hits = [
            c
            for c in report.collisions
            if c.isl_model == "ISLResponseV2" and c.isl_key == "robustness_status"
        ]
        assert hits, "differ failed to see an ISL-side enum widening"
        assert all(c.key() not in ALLOWLISTED_KEYS for c in hits)

    def test_control_hash_comparison_can_fail(self) -> None:
        """Positive control for the no-self-heal absence assertion below: the
        digest comparison must be able to SEE a byte change."""
        digest_before = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
        mutated = ARTIFACT_PATH.read_bytes() + b"\n"
        assert hashlib.sha256(mutated).hexdigest() != digest_before


# ---------------------------------------------------------------------------
# 2. Built-in RED case: the machinery (not the allowlist) catches the known
#    confidence_source collision.
# ---------------------------------------------------------------------------


def test_builtin_red_case_confidence_source_is_detected() -> None:
    """The historical bite MUST be visible in the UNFILTERED report: ISL
    FactorSensitivityV2.confidence_source ('bootstrap_sampling' |
    'graph_structural') vs the contract's DecisionRecordPrediction
    confidence_source ('model_derived' | 'user_stated'). It is green in CI
    ONLY because of the explicit PENDING-RENAME allowlist entry."""
    report = run_drift_check()
    hits = [
        c
        for c in report.collisions
        if c.isl_model == "FactorSensitivityV2"
        and c.isl_key == "confidence_source"
        and "DecisionRecordPrediction" in c.contract_location
    ]
    assert hits, (
        "the known confidence_source collision is no longer detected — either "
        "it was fixed (then REMOVE its PENDING-RENAME allowlist entries and "
        "this test's expectation) or the check went blind (investigate before "
        "trusting any green)"
    )
    for collision in hits:
        assert collision.key() in ALLOWLISTED_KEYS


# ---------------------------------------------------------------------------
# 3. THE GATE — no un-allowlisted collisions.
# ---------------------------------------------------------------------------


def test_no_unallowlisted_name_collisions() -> None:
    """FAIL LOUD on any same-named key whose type/enum domain differs from
    the contract's and which is not covered by an explicit, decision-cited
    allowlist entry (tests/contract_drift/allowlist.py)."""
    report = run_drift_check()
    new = [c for c in report.collisions if c.key() not in ALLOWLISTED_KEYS]
    assert not new, (
        "NEW name collision(s) between ISL response models and the contract — "
        "a same-named key with a different type/enum domain is how "
        "confidence_source bit us. Fix the field or, with a cited decision, "
        "add an allowlist entry.\n"
        + "\n".join(c.describe() for c in new)
        + "\n\nFull report:\n"
        + report.render()
    )


def test_two_way_diff_report_is_produced() -> None:
    """The two-way report must always be derivable and non-degenerate; print
    it so every CI run carries the current omission/superset state (visible
    with pytest -rA / on failure)."""
    report = run_drift_check()
    rendered = report.render()
    assert "(a) contract fields ISL omits" in rendered
    assert "(b) ISL superset keys" in rendered
    # The seam genuinely has both directions today; if either count drops to
    # zero something upstream changed radically — re-derive, don't assume.
    assert report.omissions, "omission direction unexpectedly empty — verify the artifact"
    assert report.supersets, "superset direction unexpectedly empty — verify the artifact"
    print("\n" + rendered)


# ---------------------------------------------------------------------------
# 4. Allowlist hygiene — the only mutable list cannot rot.
# ---------------------------------------------------------------------------


def test_allowlist_entries_all_cite_decisions() -> None:
    for entry in ALLOWLIST:
        assert entry.status in {
            "PENDING-RENAME",
            "PENDING-ALIGNMENT",
            "ACCEPTED-DIVERGENCE",
            "ACCEPTED-UNRELATED-SEAM",
        }, f"unknown allowlist status: {entry}"
        assert (
            len(entry.decision_ref) > 40
        ), f"allowlist entry lacks a substantive decision reference: {entry}"


def test_allowlist_has_no_stale_entries() -> None:
    """Every allowlist entry must still match a LIVE collision. A stale entry
    means the underlying divergence was fixed — delete the entry (and update
    the built-in RED-case test if it was a confidence_source entry) so the
    list stays minimal and honest."""
    report = run_drift_check()
    live = {c.key() for c in report.collisions}
    stale = sorted(k for k in ALLOWLISTED_KEYS if k not in live)
    assert not stale, f"stale allowlist entries (fix landed? delete them): {stale}"


# ---------------------------------------------------------------------------
# 5. No self-heal + pin integrity.
# ---------------------------------------------------------------------------


def test_no_self_heal_nothing_rewrites_the_baseline() -> None:
    """Running the entire check must leave the committed artifact and pin
    byte-identical, and must not have imported any generation machinery.
    (The regeneration path lives only in scripts/contract_schema/ and is
    exercised by the CI freshness gate, which FAILS on drift rather than
    rewriting.)"""
    before = _fixture_digests()
    run_drift_check()
    after = _fixture_digests()
    assert before == after, "drift check mutated its own baseline fixtures"
    generation_modules = [
        m for m in sys.modules if "contract_schema" in m or "generate_contract" in m
    ]
    assert not generation_modules, (
        f"generation machinery must never be importable from the check: " f"{generation_modules}"
    )


def test_pin_is_the_single_source_and_matches_artifact() -> None:
    """PIN.json is the ONE place the contract ref lives; the committed
    artifact must self-describe as generated from exactly that pin. (Byte
    freshness against the real upstream is enforced by the CI job
    'contract-drift' via scripts/contract_schema/refresh_contract_schema.sh
    --check.)"""
    pin = load_pin()
    meta = load_artifact()["_meta"]
    assert meta["source_repo"] == pin["repo"]
    assert meta["source_ref"] == pin["ref"]
    assert meta["package_version"] == pin["package_version_expected"]
    assert meta["zod-to-json-schema"] == pin["zod_to_json_schema_version"]


def test_workflow_carries_the_freshness_gate() -> None:
    """A committed contract copy WITHOUT a loud freshness gate is exactly the
    hand-maintained mirror this check exists to kill — assert the CI workflow
    still wires the gate."""
    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "pr-ci.yml"
    ).read_text()
    assert "refresh_contract_schema.sh" in workflow and "--check" in workflow, (
        "the contract-drift freshness gate was removed from pr-ci.yml — the "
        "committed artifact is now an unguarded mirror; restore the gate"
    )
