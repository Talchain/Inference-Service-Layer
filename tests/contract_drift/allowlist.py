"""Collision allowlist for the ISL ↔ @talchain/schemas drift check.

THIS IS THE CHECK'S ONLY MUTABLE LIST. Everything else in the check is
derived (the contract side from the pinned artifact, the ISL side from the
Pydantic models themselves). Every entry MUST cite a decision reference; a
stale entry (no longer matching a live collision) FAILS the suite so the
list cannot rot into a zombie registry.

An entry silences exactly one (isl_model, isl_key, contract_location)
collision. Any collision not enumerated here fails CI loud.

Statuses:
  PENDING-RENAME       — known accepted collision awaiting a rename decision.
  PENDING-ALIGNMENT    — same concept, drifted vocabulary; awaiting an
                         alignment decision.
  ACCEPTED-DIVERGENCE  — same seam, divergence understood and normalised by
                         PLoT; documented in the contract itself.
  ACCEPTED-UNRELATED-SEAM — same key name in a contract model on a different
                         seam (UI blocks / orchestrator payloads); no shared
                         consumer conflates them today. Kept visible here —
                         not suppressed in the engine — because the
                         confidence_source bite WAS exactly this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class AllowlistEntry:
    isl_model: str
    isl_key: str
    contract_location: str
    status: str
    decision_ref: str


_REF_CONFIDENCE_SOURCE = (
    "A3 drift-check lane 2026-07-20 — RENAME-DECISION-confidence_source.md "
    "(acceptance-evidence/a3-verify-2026-07-16/drift-check-lane/); ISL emits "
    "'bootstrap_sampling'|'graph_structural' (src/models/response_v2.py "
    "FactorSensitivityV2), contract 0.16.0+ uses the same key for "
    "'model_derived'|'user_stated' on DecisionRecordPrediction. Unrelated "
    "semantics under one key — awaiting Paul/orchestrator rename ruling."
)

_REF_SEED_SOURCE = (
    "A3 drift-check lane 2026-07-20 — NEW finding recorded in "
    "ISL-CHECK-NOTES.md: same concept, drifted vocabulary. ISL emits "
    "'client_provided'|'server_computed' (ISLResponseV2.seed_source), the "
    "contract's ResponseMetaSchema.seed_source is "
    "'client_generated'|'server_generated'. A consumer validating ISL's "
    "value against the contract enum rejects it. Routed to the orchestrator "
    "for an alignment ruling alongside confidence_source."
)

_REF_SEVERITY = (
    "Contract-documented divergence: EnrichmentInferenceWarningSchema "
    '(olumi-schemas src/boundary/enrichment.ts) states "severity is '
    "'info'|'warning' on the V2 wire — PLoT's merge coerces anything else to "
    "'info'\". ISL's InferenceWarning vocabulary reserves 'error' but only "
    "'info'/'warning' are stamped today (src/models/response_v2.py "
    "InferenceWarning.severity comment). A3 drift-check lane 2026-07-20."
)

_REF_ROBUST_EDGES = (
    "Contract-documented normalisation: ISL's robust_edges is the V1-compat "
    "string list ('from->to'); PLoT normalises to NormalizedEdgeInfoV3 "
    "objects before the enrichment envelope (olumi-schemas "
    "src/boundary/enrichment.ts [F2] EnrichmentRobustnessEdgeSchema). The "
    "contract types the post-PLoT shape. A3 drift-check lane 2026-07-20."
)

_REF_UNRELATED = (
    "A3 drift-check lane audit 2026-07-20 (ISL-CHECK-NOTES.md): same key "
    "name in contract models on the UI-blocks/orchestrator seam, which ISL "
    "responses never touch; no shared consumer conflates the two today. "
    "Enumerated exactly so any NEW same-named-key collision still fails."
)

_REF_ANALYSIS_FACT_STATUS = (
    "ROADMAP 2.160 contract-alignment lane 2026-07-30 (contract pin bumped "
    "0.20.0 -> 0.30.0). SAME CLASS as _REF_UNRELATED, on a seam that did not "
    "exist at 0.20.0: the AnalysisFact discriminated union "
    "(olumi-schemas src/contracts/analysis-fact.ts, exported from index) is "
    "new in this span and reuses the key name 'status'. ISL's "
    "OptionResultV2.status is the PER-OPTION COMPUTE RESULT status on the "
    "ISL->PLoT seam ('computed'|'failed'|'partial'). The contract's fact "
    "'status' is the PER-METRIC-PER-SUBJECT fact status on a "
    "status-DISCRIMINATED union ('win_probability' on 'option_a'), where each "
    "branch pins its own single-value domain -- which is why the disjoint "
    "branches (UnavailableFact 'unavailable', SuppressedFact 'suppressed') "
    "surface here while ComputedFact 'computed' does not.\n"
    "WHY NO CONSUMER CAN CONFLATE THEM. The family's own placement note "
    "declares it 'Additive and OPTIONAL, CEE-internal only: "
    "RunAnalysisResult.analysis_facts?', and records that "
    "RunAnalysisResultSchema 'is .strict() but never crosses the UI wire (it is "
    "an ORCHESTRATOR_INTERNAL fixture-coverage exclusion -- the persisted "
    "handler fact payload)'. An ISL response never populates an analysis_fact, "
    "and nothing validates an ISL OptionResultV2 against a fact branch.\n"
    "WHERE THAT QUOTE LIVES -- read it at the source, NOT here. It is a "
    "TypeScript comment in olumi-schemas src/contracts/analysis-fact.ts "
    "(~:47-51) at tag v0.30.0. It is NOT in this repo's committed contract "
    "artifact: comments do not survive the JSON-Schema export, so `rg` over "
    "tests/fixtures/contract-schema/talchain-schemas.json returns ZERO hits for "
    "'Additive and OPTIONAL', 'CEE-internal', 'analysis_facts' and "
    "'RunAnalysisResult' (positive control: 'AnalysisFactSchema' = 1 hit, so the "
    "sweep works). An earlier draft of this note said the quote was 'verified at "
    "the bytes in the 0.30.0 artifact' -- corrected on adversarial review, "
    "because a citation pointing at a file that cannot contain the text is "
    "exactly the label that teaches the next reader to stop looking, and it does "
    "the most damage inside a decision reference. The SUBSTANCE was and remains "
    "verified; only the location was wrong.\n"
    "Kept VISIBLE here rather than suppressed in the engine, for the same reason "
    "as the other unrelated-seam entries: the confidence_source bite WAS exactly "
    "this class, so a same-named key is never silently OK.\n"
    "RE-EXAMINE IF: analysis facts reach the UI wire (the contract says that is "
    "a separate slice at a NEW top-level OlumiResponseSchema key, with a UI "
    "re-vendor in its train), or if ISL ever emits an analysis_fact directly."
)

ALLOWLIST: FrozenSet[AllowlistEntry] = frozenset(
    {
        # ------------------------------------------------------------------
        # THE known bite — the check's built-in RED case, silenced only here.
        # ------------------------------------------------------------------
        AllowlistEntry(
            "FactorSensitivityV2",
            "confidence_source",
            "boundary.DecisionRecordPredictionSchema.confidence_source",
            "PENDING-RENAME",
            _REF_CONFIDENCE_SOURCE,
        ),
        AllowlistEntry(
            "FactorSensitivityV2",
            "confidence_source",
            "boundary.DecisionRecordSchema.prediction.confidence_source",
            "PENDING-RENAME",
            _REF_CONFIDENCE_SOURCE,
        ),
        # ------------------------------------------------------------------
        # New finding surfaced by this check's first run.
        # ------------------------------------------------------------------
        AllowlistEntry(
            "ISLResponseV2",
            "seed_source",
            "index.ResponseMetaSchema.seed_source",
            "PENDING-ALIGNMENT",
            _REF_SEED_SOURCE,
        ),
        # ------------------------------------------------------------------
        # Same-seam divergences the contract itself documents.
        # ------------------------------------------------------------------
        AllowlistEntry(
            "InferenceWarning",
            "severity",
            "boundary.EnrichmentInferenceWarningSchema.severity",
            "ACCEPTED-DIVERGENCE",
            _REF_SEVERITY,
        ),
        AllowlistEntry(
            "RobustnessResultV2",
            "robust_edges",
            "boundary.EnrichmentRobustnessSchema.robust_edges",
            "ACCEPTED-DIVERGENCE",
            _REF_ROBUST_EDGES,
        ),
        # ------------------------------------------------------------------
        # Unrelated-seam name reuse (UI blocks / orchestrator payloads).
        # ------------------------------------------------------------------
        AllowlistEntry(
            "CritiqueV2",
            "source",
            "boundary.BlockSchema.source",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "CritiqueV2",
            "source",
            "boundary.CoachingBlockSchema.source",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "CritiqueV2",
            "source",
            "boundary.MessageTurnPayloadSchema.source",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "CritiqueV2",
            "source",
            "boundary.OlumiResponseSchema.blocks[].source",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "CritiqueV2",
            "source",
            "boundary.OrchestratorTurnPayloadSchema.source",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "EdgeSensitivityV2",
            "direction",
            "boundary.BoundaryErrorSchema.direction",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "FactorSensitivityV2",
            "direction",
            "boundary.BoundaryErrorSchema.direction",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "OptionResultV2",
            "status",
            "boundary.BlockSchema.status",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "OptionResultV2",
            "status",
            "boundary.GraphPatchBlockSchema.status",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "OptionResultV2",
            "status",
            "boundary.OlumiResponseSchema.blocks[].status",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "OptionResultV2",
            "status",
            "index.AnalysisReadyV3Schema.options[].status",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "OptionResultV2",
            "status",
            "index.AnalysisReadyV3Schema.status",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        AllowlistEntry(
            "OptionResultV2",
            "status",
            "index.OptionForAnalysisSchema.status",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_UNRELATED,
        ),
        # ------------------------------------------------------------------
        # Unrelated-seam name reuse on the AnalysisFact union — NEW in the
        # 0.20.0 -> 0.30.0 contract span (ROADMAP 2.160).
        #
        # THREE ENTRIES SILENCE FOUR REPORTED COLLISIONS, and the mismatch is
        # not an omission — it is the counting unit, so state it rather than
        # leave the next reader to wonder which number is wrong:
        #   raw collision RECORDS reported by the engine : 4
        #   unique (isl_model, isl_key, contract_location) keys : 3
        # `index.AnalysisFactSchema.status` is reported TWICE because that one
        # union node is reached via two `anyOf` branches (the 'unavailable' and
        # 'suppressed' arms). ALLOWLISTED_KEYS is a set of those 3-tuples, so a
        # single entry silences both records. Un-allowlisted collisions at
        # 0.30.0: 0.
        # ------------------------------------------------------------------
        AllowlistEntry(
            "OptionResultV2",
            "status",
            "index.AnalysisFactSchema.status",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_ANALYSIS_FACT_STATUS,
        ),
        AllowlistEntry(
            "OptionResultV2",
            "status",
            "index.SuppressedFactSchema.status",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_ANALYSIS_FACT_STATUS,
        ),
        AllowlistEntry(
            "OptionResultV2",
            "status",
            "index.UnavailableFactSchema.status",
            "ACCEPTED-UNRELATED-SEAM",
            _REF_ANALYSIS_FACT_STATUS,
        ),
    }
)

ALLOWLISTED_KEYS = frozenset((e.isl_model, e.isl_key, e.contract_location) for e in ALLOWLIST)
