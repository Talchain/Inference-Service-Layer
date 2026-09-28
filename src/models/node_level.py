"""How ISL evaluated each NON-ROOT node: at an attested held level, or with no level (B1a).

Build train #70 5855068711 row B1a; design MG 5855036638; AIQ ruling 5855046894 (binding).

A non-root node whose ``observed_state`` holds a level stated or estimated by an ATTESTED author is
ANCHORED: every level ISL reports for it (the goal band, a limit on it) is recovered per draw in
anchored-delta form, ``o + (option_sample - status_quo_sample)`` on the SAME draw, so the status quo
reproduces the held level ``o`` on every draw. Every other non-root node is a propagated sum, and nothing
about it is a level. This block says which is which, so a consumer never shows a propagated sum as a level
and always names whose figure a level rests on.

Only reported LEVELS move (round 2; AIQ 5856229075, permanent for additive nodes): the Monte Carlo and
every structural analysis (win share, regret, sensitivity, fragile edges, EVPPI, EVPC, flip thresholds,
conditional winners) run on today's evaluator and today's draws and are byte-identical. Anchoring changes
no sign and does not touch F-01, which lives at the PLoT seam (A3).

Shared by the internal (V1) response and the V2 wire envelope, exactly as ``RangeFitDisclosure`` is. It
lives in its own module because ``response_v2`` cannot import ``robustness_v2`` (circular).
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# WHO authored a held level (AIQ 5855046894 (1)). The raw ``observed_state.source`` literal is echoed
# beside it (``observed_source``) because these three classes are deliberately coarser than the
# estate's source vocabulary (e.g. ``panel_elicited`` and ``user_confirmed`` are both human-ratified).
LevelAnchorSource = Literal["user_stated", "user_ratified", "olumi_estimate"]

NoLevelReason = Literal[
    "no_observed_level", "source_not_attested", "epsilon_breaks_status_quo_reference"
]


class NodeLevelFrame(BaseModel):
    """One non-root node's evaluation frame."""

    node_id: str = Field(..., description="The non-root node this frame describes")
    frame: Literal["anchored_level", "no_level"] = Field(
        ...,
        description=(
            "'anchored_level': the node holds an attested level; every level ISL reports for it (the "
            "goal band, a limit on it) is that level plus the option's change from the same draw's "
            "status quo, so the status quo reproduces the held level. 'no_level': the node is its "
            "parents' propagated sum; nothing reported for it is a level and must never be shown as one."
        ),
    )
    level_anchor_source: Optional[LevelAnchorSource] = Field(
        None,
        description=(
            "Present iff frame is 'anchored_level': who authored the held level. 'olumi_estimate' "
            "means a band or limit on this node rests on Olumi's own estimate of today's level."
        ),
    )
    observed_source: Optional[str] = Field(
        None,
        description="The node's observed_state.source, echoed verbatim when present. Echo only.",
    )
    level: Optional[float] = Field(
        None,
        description=(
            "Present iff frame is 'anchored_level': the held level the status quo reproduces "
            "(observed_state.baseline, else observed_state.value), in the node's own frame."
        ),
    )
    no_level_reason: Optional[NoLevelReason] = Field(
        None,
        description=(
            "Present iff frame is 'no_level': 'no_observed_level' (the node states no level), "
            "'source_not_attested' (it states one, but with no source or a source that is not an "
            "attested author, so it is never used as an anchor), or "
            "'epsilon_breaks_status_quo_reference' (epsilon noise reaches the node, so a draw cannot "
            "be differenced against its status quo without carrying noise no option caused)."
        ),
    )
    level_domain_min: Optional[float] = Field(
        None,
        description=(
            "Anchored nodes only: the lowest level the quantity can take, in its own frame: the "
            "caller's unit meaning when a 'level' limit on the node carries a level_domain; otherwise "
            "0 when the held level is non-negative (AIQ 5855046894 (2)); absent otherwise."
        ),
    )
    level_domain_max: Optional[float] = Field(
        None,
        description=(
            "Anchored nodes only: the highest level, in its own frame, from the UNIT's meaning (AIQ "
            "#72 5866289608): the level_domain.max of a 'level' limit on the node (PLoT sends 1 for a "
            "'%' limit), and ONLY when the node's frame (execution_frame, else observed_state.cap, "
            "else the value/raw_value pair) is 100 points: off a 100-point frame PLoT's {0, 1} means "
            "[0, frame], the frame, not 100%, so no ceiling is stated. Absent otherwise (money, "
            "counts). ISL does not read observed_state.cap as a ceiling."
        ),
    )
    level_out_of_domain_share: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Anchored nodes whose level ISL reports (the goal band, a limit's target) and that have a "
            "domain: option_id -> the share of that option's draws whose level lies outside the domain. "
            "Reported levels (the goal band's p10/p50/p90, the level-frame probability_of_goal, a "
            "level limit's figures) are clamped to the domain; win shares, rankings and differences "
            "are computed on the UNCLAMPED draws. A share above 0 means the model's own changes push "
            "the quantity somewhere it cannot go, in that share of futures."
        ),
    )
    parameter_uncertainty_unused: Optional[bool] = Field(
        None,
        description=(
            "Anchored nodes only: true when the request carried a parameter_uncertainty for this node. "
            "Its draw sits in both an option's sample and the same draw's status quo, so it cancels "
            "from every level reported for the node: those levels rest on the held level, not on that "
            "distribution. The structural analyses still sample it, as before B1a."
        ),
    )

    @model_validator(mode="after")
    def _fields_match_the_frame(self) -> "NodeLevelFrame":
        anchored = self.frame == "anchored_level"
        if anchored != (self.level_anchor_source is not None) or anchored != (
            self.level is not None
        ):
            raise ValueError(
                "level_anchor_source and level are present iff frame is 'anchored_level'"
            )
        if anchored == (self.no_level_reason is not None):
            raise ValueError("no_level_reason is present iff frame is 'no_level'")
        if not anchored and (
            self.level_domain_min is not None
            or self.level_domain_max is not None
            or self.level_out_of_domain_share is not None
            or self.parameter_uncertainty_unused is not None
        ):
            raise ValueError("a 'no_level' frame carries no domain, share or uncertainty flag")
        return self

    model_config = {"extra": "ignore"}
