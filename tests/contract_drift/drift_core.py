"""Core engine for the ISL ↔ @talchain/schemas contract drift check.

WHY THIS EXISTS
---------------
ISL's Pydantic V2 response models are a hand-maintained mirror of the
@talchain/schemas contract, with no mechanical drift check — the
derive-don't-mirror defect class. It has already bitten once:
``FactorSensitivityV2.confidence_source`` (``bootstrap_sampling |
graph_structural``) name-collides with the contract's
``DecisionRecordPrediction.confidence_source`` (``model_derived |
user_stated``) — unrelated semantics under one key.

WHAT IT DOES
------------
Per compute-seam response model, a TWO-WAY name+type diff against the
contract JSON-Schema artifact (tests/fixtures/contract-schema/, derived from
Talchain/olumi-schemas at the ref pinned in PIN.json — see that file):

  (a) contract fields ISL omits  — potential silent drop at source; REPORTED,
      never auto-failed (PLoT legitimately synthesises several of them);
  (b) ISL-emitted keys absent from the contract — ISL supersets; REPORTED,
      never auto-failed (ISL legitimately emits fields the contract has not
      adopted yet).

  FAIL-LOUD is reserved for NAME-COLLISIONS: a same-named key whose
  type/enum domain differs from the contract's.
    * WITHIN a paired model: ISL's domain must be ACCEPTED by the contract
      property (kind-compatible; closed-enum ISL ⊆ closed-enum contract;
      open-string contract accepts any ISL enum; an ISL open string against a
      contract closed enum is a violation — ISL could emit out-of-domain
      values). Array items are compared one level deep (kind only).
    * ACROSS the whole contract export surface: any same-named property
      anywhere in the contract where BOTH sides are closed enums with
      DISJOINT domains — unrelated vocabularies under one key, the exact
      ``confidence_source`` class. (Disjointness, not mere non-subset, keeps
      this scan high-precision: overlapping vocabularies across *different*
      models are usually the same concept at different widths and are
      already covered within the pair.)

  Collisions present today are recorded in tests/contract_drift/allowlist.py
  — the check's ONLY mutable list; each entry cites a decision reference.
  Any NEW collision fails CI loud.

NULL-HANDLING NOTE: ISL serialises V2 responses with ``exclude_none=True``
(see e.g. the ``inference_warnings`` field comment in
src/models/response_v2.py), so ``Optional[...]`` fields are OMITTED when
None, never emitted as JSON null. The differ therefore strips ``null`` from
the ISL side of every kind comparison.

NO SELF-HEAL: this module and the tests are strictly read-only over the
fixture — nothing here writes, regenerates, or "repairs" the artifact or the
pin. Regeneration lives only in scripts/contract_schema/ and is invoked by a
human or by the CI freshness gate, which FAILS on any byte difference rather
than rewriting anything. tests/contract_drift/test_contract_drift.py proves
both properties (positive control + fixture-bytes-unchanged).
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Type

from pydantic import BaseModel

from src.models.robustness_v2 import GoalConstraint
from src.models.response_v2 import (
    ConstraintAnalysisV2,
    ConstraintResultV2,
    CritiqueV2,
    DownsideV2,
    EdgeEValueV2,
    EdgeSensitivityV2,
    FactorSensitivityV2,
    FlipStabilityBandV2,
    FragileEdgeV2,
    InferenceWarning,
    ISLResponseV2,
    ObjectiveRankingV2,
    OptionResultV2,
    OutcomeDistributionV2,
    RobustnessResultV2,
    SensitiveFactorV2,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "contract-schema"
ARTIFACT_PATH = FIXTURE_DIR / "talchain-schemas.json"
PIN_PATH = FIXTURE_DIR / "PIN.json"


# ---------------------------------------------------------------------------
# Model ↔ contract pairing
# ---------------------------------------------------------------------------
# The ISL compute-seam response family — plus, since 2.798, the request-side
# ``GoalConstraint`` (see its entry below) — paired with the contract schema
# that corresponds to it on the live wire. ``None`` = the contract has no
# counterpart today (pure ISL superset; still swept by the cross-model
# collision scan). Pairing choices are documented in the draft PR / evidence
# notes; the CONTRACT side of every comparison is derived from the artifact,
# never hand-listed.
#
# NOTE on seam shape: most pairs point at the ``Enrichment*`` schemas, which
# type the PLoT→CEE envelope — the post-PLoT projection of ISL's wire. Where
# PLoT renames (node_id→factor_id) or synthesises fields (from_label,
# confidence_provenance, ...), those surface as reported omissions, which is
# the intended visibility, not a failure.
PAIRINGS: Dict[Type[BaseModel], Optional[Tuple[str, str]]] = {
    ISLResponseV2: ("boundary", "AnalysisEnrichmentSchema"),
    ObjectiveRankingV2: ("boundary", "EnrichmentObjectiveRankingSchema"),
    OptionResultV2: ("boundary", "EnrichmentOptionComparisonEntrySchema"),
    OutcomeDistributionV2: ("boundary", "EnrichmentOutcomeStatsSchema"),
    RobustnessResultV2: ("boundary", "EnrichmentRobustnessSchema"),
    FragileEdgeV2: ("boundary", "EnrichmentRobustnessEdgeSchema"),
    EdgeEValueV2: ("boundary", "EnrichmentEdgeEValueSchema"),
    FlipStabilityBandV2: ("boundary", "EnrichmentEdgeEValueStabilitySchema"),
    FactorSensitivityV2: ("index", "FactorSensitivitySchema"),
    InferenceWarning: ("boundary", "EnrichmentInferenceWarningSchema"),
    CritiqueV2: ("boundary", "EnrichmentCritiqueSchema"),
    ConstraintResultV2: ("boundary", "EnrichmentConstraintResultSchema"),
    # ---------------------------------------------------------------------
    # REQUEST-side pairing (ROADMAP 2.798). The only member of this table that
    # is not a response model, and it is here deliberately.
    #
    # WHY A REQUEST MODEL IS IN A RESPONSE-FAMILY TABLE. `GoalConstraint` is an
    # INGRESS model: it types what CEE/PLoT send ISL, so its mirror-partner is
    # the contract's `DraftGoalConstraintSchema` rather than an `Enrichment*`
    # projection. The drift risk is identical in kind to the egress models' —
    # a hand-maintained Pydantic mirror of a TS source of truth — and the check
    # is direction-agnostic, so it polices ingress exactly as well.
    #
    # WHY IT WAS ADDED. ISL declared `value_frame` (2.798) describing itself as
    # mirroring `@talchain/schemas` 0.38.0 `DraftGoalConstraint.value_frame`,
    # while the pin sat at 0.30.0 — a contract with no such field. The pin bump
    # alone would have been THEATRE: it puts `value_frame` in the artifact, but
    # an unpaired model is never compared, so the field would sit in the fixture
    # unread behind a green gate. This line is what makes the comparison run.
    # It is also what makes a FUTURE divergence bite: widen ISL's Literal beyond
    # the contract's enum and this pairing turns it into a fail-loud collision.
    #
    # `GoalConstraint` reads a strict subset of the contract's declared fields;
    # the six it does not consume ride the omission baseline as OPTIONAL, which
    # is the intended visibility, not a defect.
    GoalConstraint: ("boundary", "DraftGoalConstraintSchema"),
    # No contract counterpart today:
    ConstraintAnalysisV2: None,
    DownsideV2: None,  # B2 downside/tail-risk — ISL emits ahead of contract adoption
    EdgeSensitivityV2: None,
    SensitiveFactorV2: None,
}


# ---------------------------------------------------------------------------
# Type summaries — a common, comparable shape for both schema dialects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeSummary:
    """Kind set + optional closed enum domain for one property."""

    kinds: FrozenSet[str]  # e.g. {"string"}, {"number", "null"}, {"object"}
    enum: Optional[FrozenSet[str]]  # closed value domain, None = open
    item_kinds: Optional[FrozenSet[str]] = None  # array items, one level deep

    def describe(self) -> str:
        parts = ["/".join(sorted(self.kinds)) or "any"]
        if self.enum is not None:
            parts.append("{" + ", ".join(sorted(self.enum)) + "}")
        if self.item_kinds:
            parts.append("items:" + "/".join(sorted(self.item_kinds)))
        return " ".join(parts)


def _resolve_ref(schema: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
    seen = 0
    while "$ref" in schema and seen < 20:
        name = schema["$ref"].split("/")[-1]
        schema = defs.get(name, {})
        seen += 1
    return schema


def summarize(schema: Any, defs: Optional[Dict[str, Any]] = None) -> TypeSummary:
    """Reduce a JSON-Schema fragment (Pydantic or zod-to-json-schema dialect)
    to a comparable TypeSummary. Unwraps $ref / anyOf / allOf / oneOf."""
    defs = defs or {}
    if not isinstance(schema, dict):
        return TypeSummary(frozenset(), None)
    schema = _resolve_ref(schema, defs)

    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in schema:
            kinds: set = set()
            enums: set = set()
            item_kinds: set = set()
            enum_open = False
            for sub in schema[combinator]:
                s = summarize(sub, defs)
                kinds |= s.kinds
                if s.enum is None:
                    # A null-typed arm carries no vocabulary; it does not open
                    # the domain.
                    if s.kinds != {"null"}:
                        enum_open = True
                else:
                    enums |= s.enum
                if s.item_kinds:
                    item_kinds |= s.item_kinds
            return TypeSummary(
                frozenset(kinds),
                None if (enum_open or not enums) else frozenset(enums),
                frozenset(item_kinds) or None,
            )

    kinds_raw = schema.get("type")
    if isinstance(kinds_raw, str):
        kinds = {kinds_raw}
    elif isinstance(kinds_raw, list):
        kinds = set(kinds_raw)
    elif "properties" in schema or "additionalProperties" in schema:
        kinds = {"object"}
    elif "enum" in schema or "const" in schema:
        kinds = set()
    else:
        kinds = set()

    enum: Optional[FrozenSet[str]] = None
    if "const" in schema:
        enum = frozenset({str(schema["const"])})
        kinds = kinds or {_json_kind_of(schema["const"])}
    elif "enum" in schema:
        enum = frozenset(str(v) for v in schema["enum"])
        if not kinds:
            kinds = {_json_kind_of(v) for v in schema["enum"]}

    item_kinds: Optional[FrozenSet[str]] = None
    if "array" in kinds and isinstance(schema.get("items"), dict):
        item_summary = summarize(schema["items"], defs)
        item_kinds = frozenset(item_summary.kinds - {"null"}) or None

    return TypeSummary(frozenset(kinds), enum, item_kinds)


def _json_kind_of(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    return "object"


def _kinds_accepted(isl_kinds: FrozenSet[str], contract_kinds: FrozenSet[str]) -> bool:
    """Every non-null kind ISL can emit must be acceptable to the contract.

    ISL serialises with exclude_none → strip 'null' from the ISL side.
    'integer' is acceptable where the contract says 'number'.
    An unconstrained side (empty kind set = any) is always compatible.
    """
    isl_effective = isl_kinds - {"null"}
    if not isl_effective or not contract_kinds:
        return True
    for kind in isl_effective:
        if kind in contract_kinds:
            continue
        if kind == "integer" and "number" in contract_kinds:
            continue
        return False
    return True


def isl_accepted_by_contract(isl: TypeSummary, contract: TypeSummary) -> bool:
    """Within-pair rule: would everything ISL can emit validate against the
    contract property?"""
    if not _kinds_accepted(isl.kinds, contract.kinds):
        return False
    if contract.enum is not None:
        if isl.enum is None:
            # ISL open where the contract is closed: ISL may emit
            # out-of-domain values.
            return False
        if not isl.enum <= contract.enum:
            return False
    if isl.item_kinds and contract.item_kinds:
        if not _kinds_accepted(isl.item_kinds, contract.item_kinds):
            return False
    return True


def enums_disjoint(isl: TypeSummary, contract: TypeSummary) -> bool:
    """Cross-model rule: both closed enums, zero overlap — unrelated
    vocabularies under one key (the confidence_source class)."""
    return isl.enum is not None and contract.enum is not None and len(isl.enum & contract.enum) == 0


# ---------------------------------------------------------------------------
# Extracting property maps
# ---------------------------------------------------------------------------


def pydantic_properties(model: Type[BaseModel]) -> Dict[str, TypeSummary]:
    """Wire-truth property map for an ISL model: serialization mode (includes
    computed fields), by_alias (wire names, e.g. ISLResponseV2 'version')."""
    schema = model.model_json_schema(by_alias=True, mode="serialization")
    defs = schema.get("$defs", {})
    props = schema.get("properties", {})
    return {name: summarize(sub, defs) for name, sub in props.items()}


def contract_properties(
    artifact: Dict[str, Any], module: str, export: str
) -> Dict[str, TypeSummary]:
    schema = artifact["modules"][module][export]
    props = schema.get("properties", {})
    return {name: summarize(sub) for name, sub in props.items()}


def contract_required(artifact: Dict[str, Any], module: str, export: str) -> FrozenSet[str]:
    return frozenset(artifact["modules"][module][export].get("required", []))


def walk_contract_properties(
    artifact: Dict[str, Any],
) -> List[Tuple[str, str, TypeSummary]]:
    """Every named property ANYWHERE in the contract export surface, as
    (location, property_name, summary). Locations look like
    'boundary.DecisionRecordPredictionSchema.confidence_source' (nested
    objects extend the dotted path). Purely derived from the artifact."""
    found: List[Tuple[str, str, TypeSummary]] = []

    def _walk(node: Any, location: str) -> None:
        if not isinstance(node, dict):
            return
        for combinator in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(combinator, []) or []:
                _walk(sub, location)
        props = node.get("properties")
        if isinstance(props, dict):
            for name, sub in props.items():
                found.append((location, name, summarize(sub)))
                _walk(sub, f"{location}.{name}")
        items = node.get("items")
        if isinstance(items, dict):
            _walk(items, f"{location}[]")
        elif isinstance(items, list):
            for sub in items:
                _walk(sub, f"{location}[]")
        ap = node.get("additionalProperties")
        if isinstance(ap, dict):
            _walk(ap, f"{location}{{}}")

    for module_name, exports in artifact["modules"].items():
        for export_name, schema in exports.items():
            _walk(schema, f"{module_name}.{export_name}")
    return found


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------


@dataclass
class Collision:
    """A same-named key whose type/enum domain differs from the contract's."""

    scope: str  # 'pair' | 'cross-model'
    isl_model: str
    isl_key: str
    isl_summary: str
    contract_location: str
    contract_summary: str

    def key(self) -> Tuple[str, str, str]:
        """Identity used by the allowlist: model, key, contract location."""
        return (self.isl_model, self.isl_key, self.contract_location)

    def describe(self) -> str:
        return (
            f"[{self.scope}] {self.isl_model}.{self.isl_key} "
            f"(ISL: {self.isl_summary}) vs {self.contract_location} "
            f"(contract: {self.contract_summary})"
        )


@dataclass
class DriftReport:
    contract_meta: Dict[str, Any] = field(default_factory=dict)
    # (a) contract fields ISL omits: (isl_model, contract_schema, key, required)
    omissions: List[Tuple[str, str, str, bool]] = field(default_factory=list)
    # (b) ISL-emitted keys absent from the contract pair: (isl_model, key)
    supersets: List[Tuple[str, str]] = field(default_factory=list)
    collisions: List[Collision] = field(default_factory=list)

    def render(self) -> str:
        lines: List[str] = []
        meta = self.contract_meta
        lines.append(
            "ISL ↔ @talchain/schemas drift report — contract "
            f"{meta.get('package_name')}@{meta.get('package_version')} "
            f"(ref {str(meta.get('source_ref'))[:12]})"
        )
        lines.append(
            f"(a) contract fields ISL omits (reported, non-failing): {len(self.omissions)}"
        )
        for model, schema, prop, required in self.omissions:
            req = "REQUIRED" if required else "optional"
            lines.append(f"    {model} omits {schema}.{prop} [{req} in contract]")
        lines.append(
            f"(b) ISL superset keys absent from the contract pair "
            f"(reported, non-failing): {len(self.supersets)}"
        )
        for model, prop in self.supersets:
            lines.append(f"    {model}.{prop}")
        lines.append(f"(c) name collisions (FAIL unless allowlisted): {len(self.collisions)}")
        for collision in self.collisions:
            lines.append("    " + collision.describe())
        return "\n".join(lines)


def load_artifact() -> Dict[str, Any]:
    with ARTIFACT_PATH.open() as fh:
        return json.load(fh)


def load_pin() -> Dict[str, Any]:
    with PIN_PATH.open() as fh:
        return json.load(fh)


def run_drift_check(
    artifact: Optional[Dict[str, Any]] = None,
    pairings: Optional[Dict[Type[BaseModel], Optional[Tuple[str, str]]]] = None,
    isl_properties: Optional[Dict[str, Dict[str, TypeSummary]]] = None,
) -> DriftReport:
    """Produce the full two-way diff. All inputs are injectable so the
    positive-control tests can feed deliberately mutated schemas; production
    use passes nothing and reads the committed artifact."""
    artifact = artifact if artifact is not None else load_artifact()
    pairings = pairings if pairings is not None else PAIRINGS
    if isl_properties is None:
        isl_properties = {model.__name__: pydantic_properties(model) for model in pairings}

    report = DriftReport(contract_meta=artifact.get("_meta", {}))
    paired_locations: Dict[str, set] = {}

    for model, pair in pairings.items():
        model_name = model.__name__
        isl_props = isl_properties[model_name]
        if pair is None:
            for prop in sorted(isl_props):
                report.supersets.append((model_name, prop))
            continue
        module, export = pair
        c_props = contract_properties(artifact, module, export)
        c_required = contract_required(artifact, module, export)
        contract_label = f"{module}.{export}"

        for prop in sorted(set(c_props) - set(isl_props)):
            report.omissions.append((model_name, contract_label, prop, prop in c_required))
        for prop in sorted(set(isl_props) - set(c_props)):
            report.supersets.append((model_name, prop))
        for prop in sorted(set(isl_props) & set(c_props)):
            paired_locations.setdefault(f"{contract_label}.{prop}", set()).add(model_name)
            if not isl_accepted_by_contract(isl_props[prop], c_props[prop]):
                report.collisions.append(
                    Collision(
                        scope="pair",
                        isl_model=model_name,
                        isl_key=prop,
                        isl_summary=isl_props[prop].describe(),
                        contract_location=f"{contract_label}.{prop}",
                        contract_summary=c_props[prop].describe(),
                    )
                )

    # Cross-model scan: ISL keys vs every same-named property anywhere in the
    # contract. Fires only on disjoint closed enums (see module docstring).
    all_contract_props = walk_contract_properties(artifact)
    for model in pairings:
        model_name = model.__name__
        for prop, isl_summary in isl_properties[model_name].items():
            if isl_summary.enum is None:
                continue
            for location, name, contract_summary in all_contract_props:
                if name != prop:
                    continue
                full_location = f"{location}.{name}"
                if model_name in paired_locations.get(full_location, set()):
                    continue  # already adjudicated by the within-pair rule
                if enums_disjoint(isl_summary, contract_summary):
                    report.collisions.append(
                        Collision(
                            scope="cross-model",
                            isl_model=model_name,
                            isl_key=prop,
                            isl_summary=isl_summary.describe(),
                            contract_location=full_location,
                            contract_summary=contract_summary.describe(),
                        )
                    )

    # Deterministic ordering for stable CI output.
    report.collisions.sort(key=lambda c: (c.isl_model, c.isl_key, c.contract_location))
    report.omissions.sort()
    report.supersets.sort()
    return report
