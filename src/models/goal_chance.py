"""The precision and the drivers of one option's goal chance (G4 / G5).

Both blocks sit beside ``probability_of_goal`` on the option result, in both response
versions, and are omitted whenever that figure is. ISL emits numbers only: nothing here ranks
the options or words a finding.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GoalChancePrecision(BaseModel):
    """How precisely the simulation estimated ``probability_of_goal``.

    Monte Carlo precision only (``basis``): it narrows with more draws and says nothing about
    how right the model is. The drivers carry that."""

    basis: Literal["simulation_precision"] = Field(
        "simulation_precision", description="What the interval measures"
    )
    method: Literal["wilson_score"] = Field("wilson_score", description="Interval method")
    confidence_level: float = Field(..., gt=0, lt=1, description="Coverage of the interval")
    n_informative: int = Field(
        ..., ge=1, description="Draws the figure was counted over (its denominator)"
    )
    n_met: int = Field(..., ge=0, description="Informative draws that met the goal")
    interval_lower: float = Field(..., ge=0, le=1, description="Lower bound of the interval")
    interval_upper: float = Field(..., ge=0, le=1, description="Upper bound of the interval")


class GoalChanceDriver(BaseModel):
    """The goal chance within two groups of this option's informative draws.

    A continuous quantity (``factor_value``, ``link_strength``) is split at its thirds and
    reports the low and the high third; ``link_existence`` reports the draws where the link
    was absent and where it was present. The fields of the other shape are omitted."""

    model_config = {"populate_by_name": True}

    quantity_id: str = Field(..., description="Node id, or 'from->to' for a link")
    kind: Literal["factor_value", "link_strength", "link_existence"] = Field(
        ..., description="Which sampled quantity the draws were grouped by"
    )
    from_: Optional[str] = Field(None, alias="from", description="Link source node id")
    to: Optional[str] = Field(None, description="Link target node id")
    p_goal_if_low: Optional[float] = Field(
        None, ge=0, le=1, description="Goal chance within the low third of the quantity"
    )
    p_goal_if_high: Optional[float] = Field(
        None, ge=0, le=1, description="Goal chance within the high third of the quantity"
    )
    n_low: Optional[int] = Field(None, ge=1, description="Draws in the low third")
    n_high: Optional[int] = Field(None, ge=1, description="Draws in the high third")
    low_upper_value: Optional[float] = Field(
        None, description="Largest value of the quantity in the low third"
    )
    high_lower_value: Optional[float] = Field(
        None, description="Smallest value of the quantity in the high third"
    )
    p_goal_if_absent: Optional[float] = Field(
        None, ge=0, le=1, description="Goal chance within the draws where the link was absent"
    )
    p_goal_if_present: Optional[float] = Field(
        None, ge=0, le=1, description="Goal chance within the draws where the link was present"
    )
    n_absent: Optional[int] = Field(None, ge=1, description="Draws where the link was absent")
    n_present: Optional[int] = Field(None, ge=1, description="Draws where the link was present")
    spread: float = Field(
        ..., gt=0, le=1, description="Absolute difference between the two groups' goal chances"
    )
    spread_noise_floor: float = Field(
        ...,
        ge=0,
        description=(
            "Largest spread sampling noise alone is expected to produce, adjusted for the "
            "number of quantities compared for this option"
        ),
    )
    status: Literal["resolved", "below_resolution"] = Field(
        ..., description="'resolved' when the spread exceeds spread_noise_floor"
    )
    correlated: Optional[bool] = Field(
        None,
        description=(
            "True when the factor is drawn jointly with others: the two chances are then "
            "conditional on its correlated factors moving with it"
        ),
    )


class GoalChanceDrivers(BaseModel):
    """What moves ``probability_of_goal`` for one option, from the draws already made."""

    method: Literal["tercile_conditional_v1"] = Field(
        "tercile_conditional_v1", description="How the draws were grouped"
    )
    min_group_n: int = Field(..., description="Fewest draws a reported group may hold")
    n_candidates: int = Field(..., ge=0, description="Sampled quantities considered")
    n_compared: int = Field(
        ..., ge=0, description="Quantities evaluated; the family the noise floor adjusts for"
    )
    n_dropped: int = Field(
        ..., ge=0, description="Quantities with no listable row: the sum of dropped_by_reason"
    )
    dropped_by_reason: Dict[str, int] = Field(
        ...,
        description=(
            "n_dropped, by reason. A 'zero_spread' quantity was evaluated (it is also in "
            "n_compared) and its two groups did not differ; every other reason means the "
            "quantity was not evaluated. 'outcome_constant': every informative draw agreed, "
            "so nothing could be compared."
        ),
    )
    drivers: List[GoalChanceDriver] = Field(
        ..., description="The largest non-zero spreads, largest first (at most five)"
    )
