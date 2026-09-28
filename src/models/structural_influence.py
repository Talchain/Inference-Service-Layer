"""R3-5 (DL ruling #72 5872746926, AIQ meaning 5872728325): ONE structural-influence authority over EVERY
factor node.

``factor_sensitivity`` scores structural influence only over the factors that carry a parameter
uncertainty. Structural influence is a property of every factor with a path to the goal, whether or not
its value is uncertain, so a factor with no observed value had no score and a consumer could not show
ISL's influence for every factor (a UI shows producer influence only when EVERY factor carries one).

When an accounting identity is EVALUATED, the envelope carries this list: every factor node of the graph,
one cohort, one normalisation, with the identity walked at its own partials (#195). It is absent
otherwise, so a response without an evaluated identity is byte-identical.

Shared by the internal (V1) response and the V2 wire envelope, as ``IdentityEvaluation`` is. It lives in
its own module because ``response_v2`` cannot import ``robustness_v2`` (circular).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StructuralInfluence(BaseModel):
    """One factor node's structural influence on the goal, normalised over EVERY factor node."""

    node_id: str = Field(..., description="Factor node ID")
    influence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Sum of |path strengths| to the goal, normalised to 0-1 across EVERY factor node "
        "(an evaluated identity's operand and addend edges carry its partial at the centre). "
        "None for every row when the walk truncated (exact-or-null; STRUCTURAL_INFLUENCE_TRUNCATED).",
    )
    influence_rank: Optional[int] = Field(
        None,
        ge=1,
        description="Rank by influence_score, 1 = highest, over every factor node. None when the "
        "walk truncated.",
    )
