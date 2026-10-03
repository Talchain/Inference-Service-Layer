"""R3-5 (DL ruling #72 5872746926, AIQ meaning 5872728325): ONE structural-influence authority over EVERY
factor node.

``factor_sensitivity`` scores structural influence only over the factors that carry a parameter
uncertainty. Structural influence is a property of every factor with a path to the goal, whether or not
its value is uncertain, so a factor with no observed value had no score and a consumer could not show
ISL's influence for every factor (a UI shows producer influence only when EVERY factor carries one).

The envelope carries this list on EVERY graph where the factor phase runs (the ONE influence algorithm,
AIQ #72 5872951506): every factor node of the graph, one cohort, one normalisation, with an evaluated
identity walked at its own partials (#195). ``factor_sensitivity`` stays byte-identical. It is absent
only when the factor phase does not run (no parameter uncertainty).

Shared by the internal (V1) response and the V2 wire envelope, as ``IdentityEvaluation`` is. It lives in
its own module because ``response_v2`` cannot import ``robustness_v2`` (circular).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class StructuralInfluence(BaseModel):
    """One factor node's structural influence on the goal, normalised over EVERY factor node."""

    node_id: str = Field(..., description="Factor node ID")
    influence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Expected NET effect on the goal: |sum of signed path strengths|, each edge at "
        "mean x exists_probability, normalised to 0-1 across EVERY factor node (an evaluated identity's "
        "operand and addend edges carry its partial at the centre). None for every row when the walk "
        "truncated (exact-or-null; STRUCTURAL_INFLUENCE_TRUNCATED), and for a factor whose every path runs "
        "through a product with another input at 0 today (STRUCTURAL_INFLUENCE_GATED).",
    )
    gated_by: Optional[List[str]] = Field(
        None,
        description="Set only when the score is withheld because EVERY path from this factor to the goal "
        "runs through a product with another input at 0 today: those zero inputs. The influence depends on "
        "the option chosen (STRUCTURAL_INFLUENCE_GATED); a consumer shows that, never 0 and never a rank.",
    )
    influence_rank: Optional[int] = Field(
        None,
        ge=1,
        description="Rank by influence_score, 1 = highest, over every factor node with a score. None "
        "when the walk truncated or the factor is gated.",
    )
