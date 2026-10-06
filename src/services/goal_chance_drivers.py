"""What drives an option's goal chance, from the draws the analysis already made (G4 / G5).

For each sampled quantity the option's informative draws are split into thirds of that
quantity, and the goal chance is counted within the low and the high third. A link whose
existence is sampled is split into the draws where it was absent and where it was present.

Every count reads the ``meets`` and ``informative`` arrays ``probability_of_goal`` was itself
computed from, so the threshold, direction, strictness, tie tolerance and frame cannot differ
from the figure's. Nothing here draws a random number.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import AbstractSet, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.models.goal_chance import GoalChanceDriver, GoalChanceDrivers, GoalChancePrecision

# Fewest draws a reported group may hold.
MIN_GROUP_N = 30
TOP_DRIVERS = 5
CONFIDENCE_LEVEL = 0.95
_NORMAL = NormalDist()
_WILSON_Z = _NORMAL.inv_cdf(0.975)
DROP_REASONS = (
    "outcome_constant",
    "no_variance",
    "group_below_min_n",
    "tied_at_tercile_boundary",
    "set_by_option",
    "non_finite_values",
    "zero_spread",
)

EdgeKey = Tuple[str, str]


@dataclass(frozen=True, eq=False)
class GoalChanceQuantity:
    """One sampled quantity, index-aligned with the option's draws.

    ``values`` holds the quantity on every draw: a number, or for ``link_existence`` whether
    the link was present. ``present`` restricts a sampled-existence link's strength to the
    draws where the link was there, so an absent link is not read as a weak one."""

    quantity_id: str
    kind: Literal["factor_value", "link_strength", "link_existence"]
    values: np.ndarray
    present: Optional[np.ndarray] = None
    from_node: Optional[str] = None
    to_node: Optional[str] = None
    correlated: bool = False


@dataclass(frozen=True)
class GroupCount:
    n: int
    met: int

    @property
    def chance(self) -> float:
        return self.met / self.n


@dataclass(frozen=True)
class TercileCounts:
    low: GroupCount
    mid: GroupCount
    high: GroupCount
    low_upper_value: float
    high_lower_value: float


@dataclass(frozen=True)
class PresenceCounts:
    absent: GroupCount
    present: GroupCount


def wilson_interval(n_met: int, n: int) -> Tuple[float, float]:
    """95% Wilson score interval on ``n_met`` hits in ``n`` draws."""
    p = n_met / n
    z2 = _WILSON_Z**2
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = _WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    # At no hits or all hits the bound on that side IS the figure; rounding must not move it.
    lower = 0.0 if n_met == 0 else max(0.0, min(centre - half, p))
    upper = 1.0 if n_met == n else min(1.0, max(centre + half, p))
    return lower, upper


def goal_chance_precision(n_met: int, n_informative: int) -> GoalChancePrecision:
    lower, upper = wilson_interval(n_met, n_informative)
    return GoalChancePrecision(
        confidence_level=CONFIDENCE_LEVEL,
        n_informative=n_informative,
        n_met=n_met,
        interval_lower=lower,
        interval_upper=upper,
    )


def family_noise_floor(n_a: int, n_b: int, n_compared: int) -> float:
    """Largest spread between two groups that sampling noise alone is expected to produce.

    The top of many noisy spreads is biased upward: at a per-row 95% floor, twenty quantities
    with no effect name a "driver" up to 64% of the time. So the two-sided 5% is shared across
    the ``n_compared`` quantities evaluated for the option, times the worst-case standard
    error of a difference between two proportions."""
    z = _NORMAL.inv_cdf(1 - 0.025 / n_compared)
    return z * math.sqrt(0.25 / n_a + 0.25 / n_b)


def tercile_counts(values: np.ndarray, meets: np.ndarray) -> Union[TercileCounts, str]:
    """Hits within the low, middle and high third of ``values``, or the reason it has none.

    Rank-based: the draws are ordered by value (equal values keep draw order) and cut at
    ``n // 3`` from each end, so the three groups partition the draws exactly."""
    n = int(values.size)
    if n == 0:
        return "group_below_min_n"
    if not np.all(np.isfinite(values)):
        return "non_finite_values"
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    if ranked[0] == ranked[-1]:
        return "no_variance"
    k = n // 3
    if k < MIN_GROUP_N:
        return "group_below_min_n"
    # Equal values either side of a cut would be told apart only by draw order.
    if ranked[k - 1] == ranked[k] or ranked[n - k - 1] == ranked[n - k]:
        return "tied_at_tercile_boundary"
    met = meets[order]
    return TercileCounts(
        low=GroupCount(k, int(np.count_nonzero(met[:k]))),
        mid=GroupCount(n - 2 * k, int(np.count_nonzero(met[k : n - k]))),
        high=GroupCount(k, int(np.count_nonzero(met[n - k :]))),
        low_upper_value=float(ranked[k - 1]),
        high_lower_value=float(ranked[n - k]),
    )


def presence_counts(present: np.ndarray, meets: np.ndarray) -> Union[PresenceCounts, str]:
    """Hits within the draws where a link was absent and where it was present."""
    n_present = int(np.count_nonzero(present))
    n_absent = int(present.size) - n_present
    if n_present == 0 or n_absent == 0:
        return "no_variance"
    if min(n_present, n_absent) < MIN_GROUP_N:
        return "group_below_min_n"
    return PresenceCounts(
        absent=GroupCount(n_absent, int(np.count_nonzero(meets & ~present))),
        present=GroupCount(n_present, int(np.count_nonzero(meets & present))),
    )


def _compared_groups(counts: Union[TercileCounts, PresenceCounts]) -> Tuple[GroupCount, GroupCount]:
    if isinstance(counts, PresenceCounts):
        return counts.absent, counts.present
    return counts.low, counts.high


def _spread(counts: Union[TercileCounts, PresenceCounts]) -> float:
    first, second = _compared_groups(counts)
    return abs(second.chance - first.chance)


def _driver_row(
    quantity: GoalChanceQuantity, counts: Union[TercileCounts, PresenceCounts], n_compared: int
) -> GoalChanceDriver:
    first, second = _compared_groups(counts)
    spread = _spread(counts)
    floor = family_noise_floor(first.n, second.n, n_compared)
    row = GoalChanceDriver(
        quantity_id=quantity.quantity_id,
        kind=quantity.kind,
        from_=quantity.from_node,
        to=quantity.to_node,
        spread=spread,
        spread_noise_floor=floor,
        status="resolved" if spread > floor else "below_resolution",
        correlated=True if quantity.correlated else None,
    )
    if isinstance(counts, PresenceCounts):
        row.p_goal_if_absent, row.p_goal_if_present = first.chance, second.chance
        row.n_absent, row.n_present = first.n, second.n
    else:
        row.p_goal_if_low, row.p_goal_if_high = first.chance, second.chance
        row.n_low, row.n_high = first.n, second.n
        row.low_upper_value = counts.low_upper_value
        row.high_lower_value = counts.high_lower_value
    return row


def goal_chance_drivers(
    quantities: Sequence[GoalChanceQuantity],
    meets: np.ndarray,
    informative: np.ndarray,
    *,
    set_by_option: AbstractSet[str] = frozenset(),
    top_n: Optional[int] = TOP_DRIVERS,
) -> GoalChanceDrivers:
    """The largest goal-chance spreads for one option, with every unlisted quantity counted.

    ``set_by_option``: node ids the option itself sets. Such a factor's draw still moves the
    option's chance in the level frame, but only through the paired status-quo reference, so
    "if it turns out low" would mislead for the option that fixes it. It is dropped.

    A row is listed only when its two groups differ. When every informative draw agrees (the
    figure is 0 or 1) no grouping can differ, so nothing is evaluated and every candidate is
    counted as ``outcome_constant``. A quantity that was evaluated and showed no difference
    is counted as ``zero_spread``; it stays in ``n_compared``, the family the noise floor
    adjusts for, because it was looked at."""
    dropped = dict.fromkeys(DROP_REASONS, 0)
    n_met = int(np.count_nonzero(meets & informative))
    if n_met == 0 or n_met == int(np.count_nonzero(informative)):
        dropped["outcome_constant"] = len(quantities)
        return GoalChanceDrivers(
            min_group_n=MIN_GROUP_N,
            n_candidates=len(quantities),
            n_compared=0,
            n_dropped=len(quantities),
            dropped_by_reason=dropped,
            drivers=[],
        )
    measured: List[Tuple[GoalChanceQuantity, Union[TercileCounts, PresenceCounts]]] = []
    for quantity in quantities:
        if quantity.kind == "factor_value" and quantity.quantity_id in set_by_option:
            dropped["set_by_option"] += 1
            continue
        counts: Union[TercileCounts, PresenceCounts, str]
        if quantity.kind == "link_existence":
            counts = presence_counts(quantity.values[informative].astype(bool), meets[informative])
        else:
            mask = informative if quantity.present is None else informative & quantity.present
            counts = tercile_counts(quantity.values[mask], meets[mask])
        if isinstance(counts, str):
            dropped[counts] += 1
            continue
        measured.append((quantity, counts))

    listed = [
        _driver_row(quantity, counts, len(measured))
        for quantity, counts in measured
        if _spread(counts) > 0
    ]
    dropped["zero_spread"] = len(measured) - len(listed)
    listed.sort(key=lambda row: (-row.spread, row.kind, row.quantity_id))
    return GoalChanceDrivers(
        min_group_n=MIN_GROUP_N,
        n_candidates=len(quantities),
        n_compared=len(measured),
        n_dropped=sum(dropped.values()),
        dropped_by_reason=dropped,
        drivers=listed if top_n is None else listed[:top_n],
    )


def goal_chance_quantities(
    factor_values_per_sample: Sequence[Mapping[str, float]],
    edge_configs_per_sample: Sequence[Mapping[EdgeKey, float]],
    absent_edges_per_sample: Sequence[AbstractSet[EdgeKey]],
    *,
    sampled_existence: AbstractSet[EdgeKey],
    fixed_edges: AbstractSet[EdgeKey],
    correlated_factors: AbstractSet[str],
) -> Optional[List[GoalChanceQuantity]]:
    """Every quantity the analysis sampled per draw, as arrays aligned with the draws.

    Returns None when the three per-draw records are not the same length: they could not be
    aligned with the outcomes, and a driver read off misaligned draws would be a fabrication.

    ``sampled_existence``: links with an existence probability strictly between 0 and 1.
    ``fixed_edges``: definitional links, held at their central strength and never drawn."""
    n_draws = len(edge_configs_per_sample)
    if not (len(factor_values_per_sample) == n_draws == len(absent_edges_per_sample)):
        return None
    if n_draws == 0:
        return []

    quantities: List[GoalChanceQuantity] = []
    for factor_id in factor_values_per_sample[0]:
        quantities.append(
            GoalChanceQuantity(
                quantity_id=factor_id,
                kind="factor_value",
                values=np.array(
                    [draw.get(factor_id, math.nan) for draw in factor_values_per_sample],
                    dtype=float,
                ),
                correlated=factor_id in correlated_factors,
            )
        )
    for edge_key in edge_configs_per_sample[0]:
        if edge_key in fixed_edges:
            continue
        from_node, to_node = edge_key
        quantity_id = f"{from_node}->{to_node}"
        present: Optional[np.ndarray] = None
        if edge_key in sampled_existence:
            present = np.array(
                [edge_key not in absent for absent in absent_edges_per_sample], dtype=bool
            )
            quantities.append(
                GoalChanceQuantity(
                    quantity_id=quantity_id,
                    kind="link_existence",
                    values=present,
                    from_node=from_node,
                    to_node=to_node,
                )
            )
        quantities.append(
            GoalChanceQuantity(
                quantity_id=quantity_id,
                kind="link_strength",
                values=np.array(
                    [config.get(edge_key, math.nan) for config in edge_configs_per_sample],
                    dtype=float,
                ),
                present=present,
                from_node=from_node,
                to_node=to_node,
            )
        )
    return quantities
