"""How ISL evaluated each NON-ROOT node: at an attested held level, or with no level (B1a).

Build train #70 5855068711 row B1a; design MG 5855036638; AIQ ruling 5855046894 (binding).

A non-root node whose ``observed_state`` holds a level stated or estimated by an ATTESTED author is
evaluated in anchored-delta form (``SCMEvaluatorV2``): every one of its samples is a LEVEL of the
quantity, and the status quo reproduces the held level on every draw. Every other non-root node keeps the
propagated-sum form, and nothing about it is a level. This block says which is which, so a consumer
never shows a propagated sum as a level and always names whose figure a level rests on.

Only LEVELS move (round 2, #70 5856099271 / 5856103285): edge sensitivity, factor sensitivity and the
fragile-edge gate are computed in the propagated-sum form and are unchanged, and a do(x) on a no-level
node below an anchored one keeps today's effect. Anchoring changes no sign and does not touch F-01, which
lives at the PLoT seam (A3).

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

NoLevelReason = Literal["no_observed_level", "source_not_attested"]


class NodeLevelFrame(BaseModel):
    """One non-root node's evaluation frame."""

    node_id: str = Field(..., description="The non-root node this frame describes")
    frame: Literal["anchored_level", "no_level"] = Field(
        ...,
        description=(
            "'anchored_level': the node holds an attested level; ISL evaluates it as that level plus "
            "the change its parents make from the same draw's status quo, so each of its samples is a "
            "LEVEL and the status quo reproduces the held level. 'no_level': the node is its parents' "
            "propagated sum (today's form); its samples are NOT levels and must never be shown as one."
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
            "Present iff frame is 'no_level': 'no_observed_level' (the node states no level) or "
            "'source_not_attested' (it states one, but with no source or a source that is not an "
            "attested author, so it is never used as an anchor)."
        ),
    )
    level_domain_min: Optional[float] = Field(
        None,
        description=(
            "Anchored nodes only: the lowest level the quantity can take, in its own frame. 0 when "
            "the held level is non-negative (AIQ 5855046894 (2)); absent otherwise."
        ),
    )
    level_domain_max: Optional[float] = Field(
        None,
        description=(
            "Anchored nodes only: the highest level, in its own frame. 1 when the node carries a cap "
            "(its level is a share of that cap); absent otherwise."
        ),
    )
    level_out_of_domain_share: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Anchored nodes whose level ISL reports (the goal band, a limit's target) and that have a "
            "domain: option_id -> the share of that option's draws whose level lies outside the domain. "
            "Reported levels (the goal band, limit probabilities) are clamped to the domain; win shares, "
            "rankings and differences are computed on the UNCLAMPED draws. A share above 0 means the "
            "model's own changes push the quantity somewhere it cannot go, in that share of futures."
        ),
    )
    parameter_uncertainty_unused: Optional[bool] = Field(
        None,
        description=(
            "Anchored nodes only: true when the request carried a parameter_uncertainty for this node. "
            "An anchored node's status quo is its held level on every draw, so that draw is not used for "
            "its level (the band, a limit on it, win shares). Edge and factor sensitivity and the "
            "fragile-edge gate are computed in the propagated-sum form, where it is still used."
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
