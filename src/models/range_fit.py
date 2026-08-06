"""
Range→distribution converter — wire and result models (ROADMAP 2.720).

Spec: parallel-briefs/RANGE-TO-DISTRIBUTION-SPEC-2026-08-08.md, implementing
Neil's 2.521 Q1 ruling: a user-stated range is treated as an approximately 50%
credible interval, and the fitted distribution places its QUARTILES on the
stated bounds — beta for bounded quantities, normal for unbounded scalars
(Quillien, Bramley & Lucas 2025, Cognitive Psychology 159:101745).

This module is pure Pydantic — no scipy, no compute. The converter itself
lives in ``src/services/range_fit.py``.

Postures encoded here (spec §3/§4.2), each deliberate:
- The wire carries the user's RAW statement ``(lower, upper)`` plus provenance;
  fitted parameters are DERIVED at the resolver and appear only in the
  echo/disclosure — never persisted upstream as if elicited.
- ``lower``/``upper`` are Optional so an OPEN-ENDED statement ("at least 5%")
  is expressible and can be refused LOUDLY (``RANGE_OPEN_ENDED``) rather than
  being unrepresentable and silently reshaped by a caller.
- ``domain`` is REQUIRED and declared: family selection never value-sniffs
  (spec §2.2 — a family flip is a semantics change wearing a convenience).
- A refusal is honest absence, never degraded presence: no fallback
  distribution exists anywhere in this type family.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# The quantity's DECLARED domain, carried with the request. ``unit_interval``
# = ISL's normalised factor-value domain [0, 1] (FACTOR_VALUE_MIN/MAX in
# robustness_analyzer_v2), probabilities, percentages → beta. ``unbounded``
# = unbounded scalar → normal.
Domain = Literal["unit_interval", "unbounded"]

# Closed refusal vocabulary (spec §3). Tests DERIVE from this type via
# typing.get_args — never a hand list (trap 12).
RangeFitRefusalCode = Literal[
    "RANGE_ZERO_WIDTH",  # E1: a == b — outside the ruling's domain
    "RANGE_INVALID_ORDER",  # E2: a > b — never silently swapped
    "RANGE_NON_FINITE",  # E3: NaN/±inf — never silently skipped
    "RANGE_OUT_OF_DOMAIN",  # E4: outside the declared support — never clamped
    "RANGE_AT_DOMAIN_EDGE",  # E5: a==0 or b==1 — unsatisfiable ∀(α,β), derived
    "RANGE_OPEN_ENDED",  # E7: one bound absent — under-determined, → Neil
    "RANGE_FIT_NONCONVERGENT",  # E8: both starts failed acceptance — refuse loud
]

# Method provenance for the fitted disclosure — an honest constant that ships
# on the wire (spec §2.1; the design brief §3.3 pattern).
RANGE_FIT_METHOD_VERSION = "range-iq-fit-v1"


class UserStatedRange(BaseModel):
    """One user-stated range for a factor's value — the RAW statement.

    S3 transport (spec §4.2): a DECLARED additive Optional field on the
    analyze request. Declared, never passed through — every ISL request model
    is ``extra="ignore"``, so an undeclared field dies silently at parse with
    a 200 (ground-truth correction 1). The presence test goes through the real
    endpoint for exactly that reason.
    """

    node_id: str = Field(
        ...,
        pattern=r"^[a-z0-9_:-]+$",
        description="ID of the factor node the user stated this range for",
    )
    lower: Optional[float] = Field(
        None,
        description="The user's stated lower bound, raw, as said. Absent = "
        "open-ended below ('no more than X'), which is refused RANGE_OPEN_ENDED "
        "— expressible so the refusal is loud, never silently reshaped.",
    )
    upper: Optional[float] = Field(
        None,
        description="The user's stated upper bound, raw, as said. Absent = "
        "open-ended above ('at least X') — refused RANGE_OPEN_ENDED.",
    )
    domain: Domain = Field(
        ...,
        description="DECLARED domain of the quantity — determines the fitted "
        "family (unit_interval → beta, unbounded → normal). Required: family "
        "selection never infers from whether the values happen to lie in [0,1].",
    )
    # Provenance — who/when/how the statement was elicited (spec §4.2).
    source: Optional[str] = Field(
        None, description="Who stated the range (provenance, e.g. 'user')"
    )
    stated_at: Optional[str] = Field(
        None, description="When the range was stated (ISO 8601, producer-stamped)"
    )
    method_version: Optional[str] = Field(
        None,
        description="Elicitation method version stamped by the producer "
        "(NOT the fit method — that ships in the disclosure echo).",
    )

    # CIL 0.2: accept unknown fields per cross-service contract
    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "node_id": "marketing_spend",
                "lower": 0.2,
                "upper": 0.6,
                "domain": "unit_interval",
                "source": "user",
                "stated_at": "2026-08-06T00:00:00Z",
            }
        },
    }


class RangeFitRefusalPayload(BaseModel):
    """Typed payload of a range-fit refusal (spec §3).

    A refusal always means: the value stays disclosed as confirmed, NO
    distribution is produced, compute is untouched. Never a fallback — a
    Beta(1,1) minted on solver failure is a fabricated value wearing real
    provenance, this estate's dominant defect.
    """

    code: RangeFitRefusalCode = Field(..., description="Closed refusal vocabulary")
    message: str = Field(..., description="Human-readable reason (sanitised)")
    lower: Optional[float] = Field(None, description="The stated lower bound, echoed raw")
    upper: Optional[float] = Field(None, description="The stated upper bound, echoed raw")
    domain: Domain = Field(..., description="The declared domain the fit was asked for")
    starts_tried: Optional[int] = Field(
        None,
        description="For RANGE_FIT_NONCONVERGENT: how many starts of the "
        "deterministic two-start ladder were tried before refusing",
    )

    model_config = {"extra": "ignore"}


class RangeFitRefusal(Exception):
    """Typed exception family for range-fit refusals (422-class shape,
    matching the ParameterUncertainty std<=0 posture: structured, loud, never
    a silent default)."""

    def __init__(self, payload: RangeFitRefusalPayload) -> None:
        self.payload = payload
        super().__init__(payload.message)


class FittedDistribution(BaseModel):
    """The fitted user-reading distribution (spec §2.1).

    Family parameters are the source of truth; ``mean``/``std``/``q25``/``q75``
    are derived read-only disclosure conveniences recomputed from them. The raw
    ``(lower, upper)`` remain the stored source of truth upstream — this object
    is derived at the resolver, never persisted as if elicited.
    """

    family: Literal["normal", "beta"] = Field(..., description="Fitted family")
    # Normal parameters (family == "normal")
    mu: Optional[float] = Field(None, description="Normal location μ")
    sigma: Optional[float] = Field(None, gt=0, description="Normal scale σ > 0")
    # Beta parameters (family == "beta")
    alpha: Optional[float] = Field(None, gt=0, description="Beta shape α > 0")
    beta: Optional[float] = Field(None, gt=0, description="Beta shape β > 0")
    # Derived read-only disclosure fields
    mean: float = Field(..., description="Distribution mean (derived)")
    std: float = Field(..., description="Distribution std (derived)")
    q25: float = Field(..., description="First quartile (derived; = stated lower)")
    q75: float = Field(..., description="Third quartile (derived; = stated upper)")
    coverage: float = Field(
        ...,
        gt=0,
        lt=1,
        description="Credible-interval coverage the fit targeted (ratified 0.5: "
        "the stated bounds are the fitted QUARTILES, equal 25% tails)",
    )
    method_version: str = Field(
        ..., description=f"Fit method provenance, e.g. '{RANGE_FIT_METHOD_VERSION}'"
    )

    @model_validator(mode="after")
    def validate_family_params(self) -> "FittedDistribution":
        """Family ⇒ exactly its own parameter pair (emission-iff, both ways)."""
        if self.family == "normal":
            if self.mu is None or self.sigma is None:
                raise ValueError("family='normal' requires mu and sigma")
            if self.alpha is not None or self.beta is not None:
                raise ValueError("family='normal' must not carry beta parameters")
        else:  # beta
            if self.alpha is None or self.beta is None:
                raise ValueError("family='beta' requires alpha and beta")
            if self.mu is not None or self.sigma is not None:
                raise ValueError("family='beta' must not carry normal parameters")
        return self

    model_config = {"extra": "ignore"}


class RangeFitDisclosure(BaseModel):
    """Echo/disclosure entry for one stated range (spec §4.3): raw bounds as
    said, plus EITHER the fitted distribution OR the typed refusal — exactly
    one, enforced both ways."""

    node_id: str = Field(..., description="Factor node the range was stated for")
    lower: Optional[float] = Field(None, description="Stated lower bound, echoed raw")
    upper: Optional[float] = Field(None, description="Stated upper bound, echoed raw")
    domain: Domain = Field(..., description="Declared domain the family came from")
    fitted: Optional[FittedDistribution] = Field(
        None, description="The fitted distribution (present iff the fit was accepted)"
    )
    refusal: Optional[RangeFitRefusalPayload] = Field(
        None, description="The typed refusal (present iff the fit was refused)"
    )

    @model_validator(mode="after")
    def validate_exactly_one_outcome(self) -> "RangeFitDisclosure":
        """Fitted XOR refusal — a disclosure with neither is a silent drop, one
        with both is incoherent provenance."""
        if (self.fitted is None) == (self.refusal is None):
            raise ValueError("RangeFitDisclosure requires exactly one of 'fitted' or 'refusal'")
        return self

    model_config = {"extra": "ignore"}
