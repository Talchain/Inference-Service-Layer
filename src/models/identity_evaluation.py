"""R3 slice 1: how ISL treated each accounting identity CEE declared (AIQ #70 5859633012, 5860087988).

A declared identity is NOT an evaluated one. Only ``evaluated: true`` licenses a numerical claim that
rests on it; a consumer that reads a declaration as if it were used is the R3-4 defect ("declared, not
used in the numbers"). An identity that is not evaluated names why (``withheld_reason``); when the
decision depends on it the whole analysis is withheld as the blocked 422, so this block is what a
consumer reads on a COMPUTED run: the identities that were evaluated, and any that were declared off
every decision path and so left unused.

Shared by the internal (V1) response and the V2 wire envelope, as ``NodeLevelFrame`` is. It lives in its
own module because ``response_v2`` cannot import ``robustness_v2`` (circular).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

IdentityWithheldReason = Literal[
    "identity_frame_missing",
    "identity_operand_missing",
    "identity_zero_level",
    "identity_inconsistent",
    "identity_scale_out_of_range",
]


class IdentityReconciliation(BaseModel):
    """The stated level against the level the identity's own inputs give today, in USER units
    (AIQ 5860087988 item 1). Above 1% a consumer says it; above 5% the identity is withheld."""

    reconstructed: float = Field(..., description="term(operands) + addends at today's levels")
    stated: float = Field(..., description="The node's own stated level today")
    mismatch_share: float = Field(..., ge=0.0, description="|stated - reconstructed| / |stated|")


class IdentityEvaluation(BaseModel):
    """One declared identity and what ISL did with it."""

    node_id: str
    operation: Literal["product", "sum"]
    factor_ids: List[str]
    addends: List[str] = Field(default_factory=list)
    stated_in_brief: bool
    evaluated: bool = Field(..., description="True only when the numbers rest on the identity")
    withheld_reason: Optional[IdentityWithheldReason] = None
    level_source: Optional[Literal["stated_level", "identity_inputs"]] = Field(
        None,
        description=(
            "Evaluated only: anchored at the node's stated level, or (no stated level) the level "
            "its inputs give (AIQ 5860087988 item 4)"
        ),
    )
    reconciliation: Optional[IdentityReconciliation] = None
    # Proposal (3) (AIQ 5876233408): the typed carrier of WHOSE today's level this is. PLoT forwards
    # this block verbatim, so the UI binds it beside a goal's probability_of_goal (Panel 5876811906).
    level_author: Optional[Literal["user", "olumi"]] = Field(
        None,
        description=(
            "identity_inputs only: 'user' when every operand's level today is the user's, else "
            "'olumi' (the weakest operand decides). A goal anchored on it is scored from Olumi's "
            "estimate of today's level when 'olumi'"
        ),
    )
    today_level: Optional[float] = Field(
        None,
        allow_inf_nan=False,
        description="identity_inputs only: the level its inputs give today, in USER units",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _evaluated_xor_withheld(self) -> "IdentityEvaluation":
        if self.evaluated == (self.withheld_reason is not None):
            raise ValueError("an identity is evaluated XOR it names its withheld_reason")
        if self.evaluated != (self.level_source is not None):
            raise ValueError("level_source is stated for an evaluated identity only")
        from_inputs = self.level_source == "identity_inputs"
        if from_inputs != (self.level_author is not None):
            raise ValueError("level_author is stated for identity_inputs, and only there")
        if self.today_level is not None and not from_inputs:
            raise ValueError("today_level is stated for identity_inputs only")
        return self
