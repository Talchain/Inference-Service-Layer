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

from src.models.response_v2 import ISLResponseV2
from tests.contract_drift.allowlist import ALLOWLIST, ALLOWLISTED_KEYS
from tests.contract_drift.drift_baseline import (
    BASELINE_PATH,
    OMISSION_BASELINE,
    OMISSION_BASELINE_KEYS,
    SUPERSET_BASELINE,
    SUPERSET_BASELINE_KEYS,
)
from tests.contract_drift.drift_core import (
    ARTIFACT_PATH,
    PAIRINGS,
    PIN_PATH,
    DriftReport,
    contract_properties,
    isl_accepted_by_contract,
    load_artifact,
    load_pin,
    pydantic_properties,
    run_drift_check,
)


def _fixture_digests() -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ARTIFACT_PATH, PIN_PATH)}


def _new_omissions(report: DriftReport) -> list:
    """Omissions in the live report whose (model, location, prop) identity is
    NOT accepted in the committed baseline — i.e. a contract field ISL has just
    stopped emitting (rename/removal at source)."""
    return [o for o in report.omissions if (o[0], o[1], o[2]) not in OMISSION_BASELINE_KEYS]


def _new_supersets(report: DriftReport) -> list:
    """Supersets in the live report not accepted in the committed baseline —
    i.e. a key ISL has just started emitting that no contract pair carries
    (the renamed-to name lands here)."""
    return [s for s in report.supersets if s not in SUPERSET_BASELINE_KEYS]


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


# ---------------------------------------------------------------------------
# 6. THE OMISSION / SUPERSET RATCHET (F3 fix).
#
# PR #83 gated ONLY name collisions; omissions and supersets were reported but
# never failed, so a rename/removal of a field ISL emits (Codex's mutation:
# FactorSensitivityV2.elasticity -> elasticity_renamed) moved the counts and
# still passed CI — the silent-alarm class. These tests ratchet BOTH directions
# against tests/contract_drift/drift_baseline.json exactly as allowlist.py
# ratchets collisions: any drift not in the derived-once baseline fails loud.
#
# Positive controls come FIRST (trap #13): a ratchet that cannot SEE a NEW
# omission/superset would pass every green below it vacuously.
# ---------------------------------------------------------------------------


class TestRatchetPositiveControls:
    """Feed a report the engine derived from a deliberately mutated schema/
    model pair; the ratchet MUST classify the planted change as NEW (absent
    from the baseline). The planted keys are chosen to NOT be in the baseline,
    proving the baseline cannot swallow a new defect."""

    def test_control_ratchet_sees_new_omission(self) -> None:
        # Add a field to the contract that ISL cannot emit -> a NEW omission.
        artifact = copy.deepcopy(load_artifact())
        target = artifact["modules"]["index"]["FactorSensitivitySchema"]
        target.setdefault("required", []).append("drift_control_ratchet_omission")
        target["properties"]["drift_control_ratchet_omission"] = {"type": "string"}
        report = run_drift_check(artifact=artifact)
        new = _new_omissions(report)
        assert any(
            o[2] == "drift_control_ratchet_omission" for o in new
        ), "omission ratchet failed to classify a planted new omission as NEW"

    def test_control_ratchet_sees_new_superset(self) -> None:
        # Inject an ISL-side property no contract pair carries -> a NEW superset.
        isl_props = {model.__name__: pydantic_properties(model) for model in PAIRINGS}
        from tests.contract_drift.drift_core import TypeSummary

        isl_props["FactorSensitivityV2"] = dict(isl_props["FactorSensitivityV2"])
        isl_props["FactorSensitivityV2"]["drift_control_ratchet_superset"] = TypeSummary(
            kinds=frozenset({"string"}), enum=None
        )
        report = run_drift_check(isl_properties=isl_props)
        new = _new_supersets(report)
        assert (
            "FactorSensitivityV2",
            "drift_control_ratchet_superset",
        ) in new, "superset ratchet failed to classify a planted new superset as NEW"


def test_no_new_required_omissions() -> None:
    """FAIL LOUD when ISL stops emitting a contract field it used to emit — a
    rename/removal at source. The baseline records EVERY known omission (not
    only required ones), because 'required in ISL' is invisible once the field
    is gone; a required contract field vanishing is the sharpest case and is
    called out explicitly."""
    report = run_drift_check()
    new = _new_omissions(report)
    if new:
        lines = []
        for model, contract_label, prop, required in new:
            sev = "REQUIRED in contract" if required else "optional in contract"
            lines.append(f"    {model} no longer emits {contract_label}.{prop} [{sev}]")
        pytest.fail(
            "NEW omission(s) — ISL stopped emitting field(s) the contract "
            "carries (a rename/removal at source, the exact class that passed "
            "CI before this ratchet). Name(s):\n"
            + "\n".join(lines)
            + "\n\nAdd to tests/contract_drift/drift_baseline.json ONLY with a "
            "decision ref (re-run scripts/contract_schema/refresh_drift_baseline.py "
            "--refresh-baseline on the reviewed, intentional change), or fix the "
            "drift.\n\nFull report:\n" + report.render()
        )


def test_no_new_supersets() -> None:
    """FAIL LOUD when ISL starts emitting a key no contract pair carries — the
    renamed-to name lands here, and an un-tracked superset is drift the
    contract will silently drop."""
    report = run_drift_check()
    new = _new_supersets(report)
    if new:
        lines = [f"    {model}.{prop}" for model, prop in new]
        pytest.fail(
            "NEW superset key(s) — ISL emits field(s) absent from the paired "
            "contract schema (a renamed-to field or an un-adopted addition "
            "lands here). Name(s):\n"
            + "\n".join(lines)
            + "\n\nAdd to tests/contract_drift/drift_baseline.json ONLY with a "
            "decision ref (re-run scripts/contract_schema/refresh_drift_baseline.py "
            "--refresh-baseline on the reviewed, intentional change), or fix the "
            "drift.\n\nFull report:\n" + report.render()
        )


def test_baseline_has_no_stale_entries() -> None:
    """Every baseline entry must still match a LIVE omission/superset. A stale
    entry means the drift was resolved (ISL now emits it, or the contract
    dropped/adopted it) — re-run the refresh script so the baseline stays
    minimal and honest, exactly like the collision allowlist's hygiene gate."""
    report = run_drift_check()
    live_omissions = {(o[0], o[1], o[2]) for o in report.omissions}
    live_supersets = set(report.supersets)
    stale_omissions = sorted(k for k in OMISSION_BASELINE_KEYS if k not in live_omissions)
    stale_supersets = sorted(k for k in SUPERSET_BASELINE_KEYS if k not in live_supersets)
    assert not stale_omissions and not stale_supersets, (
        "stale drift-baseline entries (drift resolved? re-run "
        "scripts/contract_schema/refresh_drift_baseline.py --refresh-baseline):\n"
        f"  omissions: {stale_omissions}\n  supersets: {stale_supersets}"
    )


def test_baseline_matches_current_report_exactly() -> None:
    """On the committed, unmutated tree the baseline must be EXACTLY the live
    two-way drift — no new entries, no stale entries. This is the derive-don't-
    mirror invariant: the committed baseline is the generated truth, not a
    hand-edited approximation of it."""
    report = run_drift_check()
    assert not _new_omissions(report)
    assert not _new_supersets(report)
    assert len(OMISSION_BASELINE) == len(report.omissions)
    assert len(SUPERSET_BASELINE) == len(report.supersets)


def test_ratchet_does_not_self_heal_the_baseline() -> None:
    """Running the whole drift check must leave drift_baseline.json byte-
    identical and must not import the refresh script (F6-class self-heal:
    a gate that rewrites its own baseline ratifies whatever drift just landed).
    Regeneration lives ONLY in scripts/contract_schema/refresh_drift_baseline.py,
    run by a human."""
    before = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
    run_drift_check()
    after = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
    assert before == after, "drift check mutated its own baseline (self-heal)"
    refresh_modules = [m for m in sys.modules if "refresh_drift_baseline" in m]
    assert not refresh_modules, (
        f"baseline-generation machinery must never be importable from the "
        f"check: {refresh_modules}"
    )


def test_baseline_meta_matches_the_pinned_contract() -> None:
    """The baseline self-describes as generated from exactly the pinned ref, so
    a baseline generated against a drifted contract cannot masquerade as fresh."""
    from tests.contract_drift.drift_baseline import BASELINE_META

    pin = load_pin()
    assert BASELINE_META.get("source_ref") == pin["ref"]
    assert BASELINE_META.get("source_repo") == pin["repo"]
    assert BASELINE_META.get("package_version") == pin["package_version_expected"]


# ============================================================================
# ROADMAP 2.160 — the four VOI keys are CONTRACT-ADOPTED, so they are actually
# COMPARED rather than waved through as an accepted superset.
#
# WHY THIS TEST EXISTS AND WHY IT IS NOT REDUNDANT.
# `test_baseline_has_no_stale_entries` above already REDs if these four sit in
# the superset baseline once the contract adopts them — and it is what caught
# this at the 0.20.0 -> 0.30.0 bump. But it is a GENERIC hygiene gate: it says
# "some entry is stale", names a tuple, and is satisfied the moment anyone
# re-runs the refresh script. It cannot say WHY these four keys matter, and it
# would stay just as green if a future pin regression put them BACK into the
# superset baseline — because at that point the entry is live again, not stale.
#
# THE DEFECT BEING PINNED. While an ISL-emitted key sits in the superset
# baseline, the check records only "the contract lacks this key" and NEVER
# compares ISL's emitted domain against the contract's declared domain for it.
# So a genuine type divergence on that key is invisible for exactly as long as
# the exception exists. An exception for an ADOPTED field is therefore a
# hand-maintained mirror that hides real drift (trap 12) — which is why the
# four entries were removed rather than re-blessed.
#
# The VOI family travels together by design: `correlation_model` is the
# DISCRIMINATOR that makes an absent `p_win_sensitivity` readable as
# suppression rather than as "never computed", so a partial adoption is a
# defect in itself and is asserted against below.
# ============================================================================

VOI_FAMILY = ("correlation_model", "decision_evpi", "factor_evppi", "p_win_sensitivity")


def test_voi_family_is_not_carried_as_a_superset_exception() -> None:
    """None of the four may sit in the superset baseline: the contract adopted
    them at 0.30.0, and an exception for an adopted field suppresses the domain
    comparison that is the whole point of pairing the models."""
    carried = sorted(prop for (model, prop) in SUPERSET_BASELINE_KEYS if prop in VOI_FAMILY)
    assert carried == [], (
        "VOI key(s) still carried as ACCEPTED-SUPERSET exceptions, so their "
        "shape is NOT being compared against the contract: "
        f"{carried}. The contract declares all four on "
        "boundary.AnalysisEnrichmentSchema (the schema ISLResponseV2 is paired "
        "with). Re-run scripts/contract_schema/refresh_drift_baseline.py "
        "--refresh-baseline; do not hand-edit."
    )


def test_voi_family_is_declared_by_the_paired_contract_schema() -> None:
    """POSITIVE CONTROL for the assertion above (trap 13 — an absence claim is
    vacuous until it can see a presence).

    Absence from the superset baseline is NOT by itself proof of adoption: it is
    equally consistent with ISL having STOPPED EMITTING the keys, which would
    make the test above pass while the family silently vanished. So assert the
    presence directly, on both sides of the pair."""
    artifact = load_artifact()
    module, export = PAIRINGS[ISLResponseV2]  # type: ignore[index]
    contract_props = contract_properties(artifact, module, export)
    missing_in_contract = sorted(k for k in VOI_FAMILY if k not in contract_props)
    assert missing_in_contract == [], (
        f"the paired contract schema {module}.{export} does not declare "
        f"{missing_in_contract} — the pin has regressed below 0.30.0, or the "
        "contract dropped the VOI family. Either way the keys are no longer "
        "validated at any consumer."
    )

    isl_props = pydantic_properties(ISLResponseV2)
    missing_in_isl = sorted(k for k in VOI_FAMILY if k not in isl_props)
    assert missing_in_isl == [], (
        f"ISLResponseV2 no longer emits {missing_in_isl}. The test above would "
        "have stayed GREEN on this, because a key ISL does not emit cannot be a "
        "superset — that is the vacuity this control exists to catch."
    )


def test_voi_family_domains_are_accepted_by_the_contract() -> None:
    """The consequence of adoption: ISL's emitted domain for each of the four
    must be ACCEPTED by the contract's declared domain. This is the comparison
    the superset exceptions were suppressing, so it is the assertion that makes
    removing them worth something rather than just tidier."""
    artifact = load_artifact()
    module, export = PAIRINGS[ISLResponseV2]  # type: ignore[index]
    contract_props = contract_properties(artifact, module, export)
    isl_props = pydantic_properties(ISLResponseV2)

    rejected = []
    for key in VOI_FAMILY:
        isl_summary = isl_props[key]
        contract_summary = contract_props[key]
        if not isl_accepted_by_contract(isl_summary, contract_summary):
            rejected.append(
                f"{key}: ISL {isl_summary.describe()} vs contract {contract_summary.describe()}"
            )
    assert rejected == [], (
        "ISL emits a VOI shape the contract would REJECT — a consumer "
        "validating the ISL value against the contract fails:\n  " + "\n  ".join(rejected)
    )
