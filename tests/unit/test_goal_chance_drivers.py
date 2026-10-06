"""Each option's goal chance says how precise it is and what drives it (G4 / G5).

Beside ``probability_of_goal`` an option carries ``probability_of_goal_precision`` (a Wilson
interval on the informative draws) and ``probability_of_goal_drivers`` (the goal chance within
the low and high third of each sampled quantity, from the draws the analysis already made).

The new module is imported inside each test: at the base commit it does not exist, and a
module-level import would turn every row into one collection error instead of its own RED.
"""

import importlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.robustness_v2 import RobustnessRequestV2
from src.services import robustness_analyzer_v2 as analyser_module
from src.services.robustness_analyzer_v2 import DualUncertaintySampler, RobustnessAnalyzerV2
from src.utils.rng import SeededRNG

ENDPOINT = "/api/v1/robustness/analyze/v2?response_version=2"
SERVED_WIRE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "anchored_delta"
    / "paul_a295e4a1_served_wire_plot_a6da42b.json"
)
DROP_REASONS = {
    "outcome_constant",
    "zero_spread",
    "no_variance",
    "group_below_min_n",
    "tied_at_tercile_boundary",
    "set_by_option",
    "set_by_option_upstream",
    "non_finite_values",
}


def drivers_module():
    return importlib.import_module("src.services.goal_chance_drivers")


def payload_for(
    *,
    direction=None,
    frame="delta",
    threshold=0.7,
    strict=None,
    n_samples=1000,
    seed=7,
    nulls=0,
    lever_exists=1.0,
    options=None,
    correlations=None,
    upstream=None,
):
    """Goal = A + 0.2 x lever. A is uncertain; each ``b..`` factor is uncertain and irrelevant.

    ``upstream`` adds an uncertain factor Z that reaches the goal only through A (Z -> A).
    ``"levelled"``: A keeps its own level for today. ``"unlevelled"``: A states no level, so
    its value is whatever Z gives it."""
    nodes = [
        {"id": "a", "kind": "factor", "label": "A", "observed_state": {"value": 0.5}},
        {"id": "lever", "kind": "factor", "label": "Lever", "observed_state": {"value": 0.0}},
        {
            "id": "goal",
            "kind": "outcome",
            "label": "Goal",
            "observed_state": {"value": 0.5, "baseline": 0.5},
        },
    ]
    edges = [
        {
            "from": "a",
            "to": "goal",
            "exists_probability": 1.0,
            "strength": {"mean": 1.0, "std": 0.01},
        },
        {
            "from": "lever",
            "to": "goal",
            "exists_probability": lever_exists,
            "strength": {"mean": 0.2, "std": 0.0011},
        },
    ]
    uncertainties = [{"node_id": "a", "distribution": "normal", "std": 0.2}]
    if upstream is not None:
        centre = 0.0 if upstream == "levelled" else 0.5
        nodes.append(
            {"id": "z", "kind": "factor", "label": "Z", "observed_state": {"value": centre}}
        )
        edges.append(
            {
                "from": "z",
                "to": "a",
                "exists_probability": 1.0,
                "strength": {"mean": 1.0, "std": 0.0011},
            }
        )
        uncertainties.append({"node_id": "z", "distribution": "normal", "std": 0.2})
    if upstream == "unlevelled":
        nodes[0] = {"id": "a", "kind": "factor", "label": "A"}
        uncertainties = [u for u in uncertainties if u["node_id"] != "a"]
    for index in range(nulls):
        node_id = f"b{index:02d}"
        nodes.append(
            {"id": node_id, "kind": "factor", "label": node_id, "observed_state": {"value": 0.5}}
        )
        edges.append(
            {
                "from": node_id,
                "to": "goal",
                "exists_probability": 1.0,
                "strength": {"mean": 0.0, "std": 0.0011},
            }
        )
        uncertainties.append({"node_id": node_id, "distribution": "normal", "std": 0.2})
    payload = {
        "graph": {"nodes": nodes, "edges": edges},
        "options": options
        or [
            {"id": "act", "label": "Act", "interventions": {"lever": 1.0}},
            {"id": "hold", "label": "Hold", "interventions": {"lever": 0.0}},
        ],
        "goal_node_id": "goal",
        "parameter_uncertainties": uncertainties,
        "n_samples": n_samples,
        "seed": seed,
        "analysis_types": ["comparison"],
    }
    if threshold is not None:
        payload["goal_threshold"] = threshold
        payload["goal_threshold_frame"] = frame
    if direction is not None:
        payload["goal_direction"] = direction
    if strict is not None:
        payload["goal_threshold_strict"] = strict
    if correlations is not None:
        payload["factor_correlations"] = correlations
    return payload


def analyse(**kwargs):
    response = RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**payload_for(**kwargs)))
    return {result.option_id: result for result in response.results}


def row_for(option_result, quantity_id, kind):
    rows = [
        row
        for row in option_result.probability_of_goal_drivers.drivers
        if row.quantity_id == quantity_id and row.kind == kind
    ]
    assert len(rows) == 1, f"expected one {kind} row for {quantity_id}, got {len(rows)}"
    return rows[0]


# ---------------------------------------------------------------- drivers on the wire


def test_main_driver_is_the_uncertain_factor():
    act = analyse()["act"]
    drivers = act.probability_of_goal_drivers
    top = drivers.drivers[0]

    assert (top.quantity_id, top.kind) == ("a", "factor_value")
    assert top.p_goal_if_low < act.probability_of_goal < top.p_goal_if_high
    assert top.spread == abs(top.p_goal_if_high - top.p_goal_if_low)
    assert top.status == "resolved"
    assert top.spread > top.spread_noise_floor
    # The cut values sit either side of A's centre, so the words can name them.
    assert top.low_upper_value < 0.5 < top.high_lower_value
    n_informative = act.probability_of_goal_precision.n_informative
    assert top.n_low == top.n_high == n_informative // 3
    assert drivers.method == "tercile_conditional_v1"
    assert [row.spread for row in drivers.drivers] == sorted(
        (row.spread for row in drivers.drivers), reverse=True
    )


def test_irrelevant_quantity_is_below_resolution():
    act = analyse(nulls=1)["act"]

    null = row_for(act, "b00", "factor_value")
    assert null.status == "below_resolution"
    assert null.spread < null.spread_noise_floor
    # Contrast in the same run: the real driver is resolved.
    assert row_for(act, "a", "factor_value").status == "resolved"


def test_minimise_low_third_raises_the_chance():
    act = analyse(direction="minimise")["act"]

    row = row_for(act, "a", "factor_value")
    assert row.p_goal_if_low > act.probability_of_goal > row.p_goal_if_high


def test_sampled_link_existence_reports_present_and_absent_groups():
    act = analyse(lever_exists=0.5)["act"]
    precision = act.probability_of_goal_precision

    existence = row_for(act, "lever->goal", "link_existence")
    assert (existence.from_, existence.to) == ("lever", "goal")
    assert existence.p_goal_if_present > existence.p_goal_if_absent
    assert existence.status == "resolved"
    # The two groups partition the informative draws, and their hits are the figure's hits.
    assert existence.n_absent + existence.n_present == precision.n_informative
    hits = round(existence.p_goal_if_absent * existence.n_absent) + round(
        existence.p_goal_if_present * existence.n_present
    )
    assert hits == precision.n_met
    # Strength is split over the draws where the link is present: absent is not "weak".
    strength = row_for(act, "lever->goal", "link_strength")
    assert strength.n_low == strength.n_high == existence.n_present // 3
    assert strength.low_upper_value > 0.0


def test_existence_group_under_thirty_draws_is_dropped_and_counted():
    act = analyse(lever_exists=0.99)["act"]
    drivers = act.probability_of_goal_drivers

    assert not [row for row in drivers.drivers if row.kind == "link_existence"]
    assert drivers.dropped_by_reason["group_below_min_n"] == 1
    # Contrast: at an even split the same link is reported (previous test).
    assert set(drivers.dropped_by_reason) == DROP_REASONS
    assert sum(drivers.dropped_by_reason.values()) == drivers.n_dropped
    # A zero-spread quantity was evaluated, so it is in both counts.
    assert drivers.n_candidates == (
        drivers.n_compared + drivers.n_dropped - drivers.dropped_by_reason["zero_spread"]
    )
    assert drivers.min_group_n == 30


def test_twenty_null_quantities_none_resolved():
    """R1: the top of many noisy spreads is biased upward, so the floor is family-adjusted."""
    module = drivers_module()
    act = analyse(nulls=20, seed=7)["act"]
    drivers = act.probability_of_goal_drivers

    assert drivers.n_compared == 43  # 21 factors + 22 link strengths
    assert len(drivers.drivers) == 5
    assert (drivers.drivers[0].quantity_id, drivers.drivers[0].status) == ("a", "resolved")
    nulls = drivers.drivers[1:]
    assert [row.status for row in nulls] == ["below_resolution"] * 4
    # Rows are ranked by spread and every continuous row here shares one floor, so the four
    # largest null spreads being below it means all 42 are.
    assert {row.spread_noise_floor for row in drivers.drivers} == {
        module.family_noise_floor(nulls[0].n_low, nulls[0].n_high, 43)
    }
    # Contrast: the unadjusted per-row floor WOULD have named a null quantity on this seed.
    assert nulls[0].spread > module.family_noise_floor(nulls[0].n_low, nulls[0].n_high, 1)


def test_family_noise_floor_matches_the_ruled_figures():
    module = drivers_module()

    floors = [module.family_noise_floor(333, 333, n) for n in (1, 5, 20)]
    assert floors == pytest.approx([0.076, 0.100, 0.117], abs=5e-4)


def test_quantity_the_option_sets_is_excluded_and_counted(monkeypatch):
    """R3: for the option that sets A, A's draw still moves the chance, but only through the
    paired status-quo reference. Phase 1 drops the row rather than word it."""
    module = drivers_module()
    calls = {}

    def recording(quantities, meets, informative, **kwargs):
        calls[len(calls)] = (quantities, meets, informative, kwargs)
        return module.goal_chance_drivers(quantities, meets, informative, **kwargs)

    monkeypatch.setattr(analyser_module, "goal_chance_drivers", recording)
    options = [
        {"id": "set_a", "label": "Set A", "interventions": {"a": 0.6}},
        {"id": "act", "label": "Act", "interventions": {"lever": 1.0}},
    ]
    results = analyse(frame="level", threshold=0.6, options=options)
    set_a = results["set_a"]

    # The claim, pinned: computed without the exclusion, A is this option's top driver.
    quantities, meets, informative, kwargs = calls[0]
    assert kwargs["set_by_option"] == frozenset({"a"})
    unexcluded = module.goal_chance_drivers(
        quantities, meets, informative, **{**kwargs, "set_by_option": frozenset()}
    )
    assert (unexcluded.drivers[0].quantity_id, unexcluded.drivers[0].kind) == ("a", "factor_value")
    assert unexcluded.drivers[0].spread > 0.9
    assert 0.3 < set_a.probability_of_goal < 0.7

    # The wire: no row for A on that option, and the drop is counted.
    drivers = set_a.probability_of_goal_drivers
    assert not [row for row in drivers.drivers if row.quantity_id == "a"]
    assert drivers.dropped_by_reason["set_by_option"] == 1
    # Contrast: an option that does not set A keeps A's row and drops nothing for that reason.
    plain = analyse()["act"]
    assert plain.probability_of_goal_drivers.dropped_by_reason["set_by_option"] == 0
    assert row_for(plain, "a", "factor_value").status == "resolved"


SET_A_OPTIONS = [
    {"id": "set_a", "label": "Set A", "interventions": {"a": 0.6}},
    {"id": "act", "label": "Act", "interventions": {"lever": 1.0}},
]


def record_driver_calls(monkeypatch):
    module = drivers_module()
    calls = []

    def recording(quantities, meets, informative, **kwargs):
        calls.append((quantities, meets, informative, kwargs))
        return module.goal_chance_drivers(quantities, meets, informative, **kwargs)

    monkeypatch.setattr(analyser_module, "goal_chance_drivers", recording)
    return module, calls


def test_quantity_upstream_of_a_set_node_is_excluded_and_counted(monkeypatch):
    """R3b: Z reaches the goal only through A, and A states no level for today. The option
    that sets A cuts Z off, yet Z's draw still moves that option's chance, because the paired
    status quo carries it. Listed, Z would read as the option's main driver."""
    module, calls = record_driver_calls(monkeypatch)
    set_a = analyse(frame="level", threshold=0.6, options=SET_A_OPTIONS, upstream="unlevelled")[
        "set_a"
    ]

    # The claim, pinned: computed without the exclusion, Z is this option's top driver.
    quantities, meets, informative, kwargs = calls[0]
    assert kwargs["upstream_of_set"] == frozenset({"z", "z->a"})
    unexcluded = module.goal_chance_drivers(
        quantities, meets, informative, **{**kwargs, "upstream_of_set": frozenset()}
    )
    top = unexcluded.drivers[0]
    assert (top.quantity_id, top.kind, top.status) == ("z", "factor_value", "resolved")
    assert top.spread > 0.9
    assert 0.3 < set_a.probability_of_goal < 0.7

    # The wire: no row for Z or its link on that option, and each drop is counted.
    drivers = set_a.probability_of_goal_drivers
    assert not [row for row in drivers.drivers if row.quantity_id in {"z", "z->a"}]
    assert drivers.dropped_by_reason["set_by_option_upstream"] == 2
    assert drivers.n_compared == unexcluded.n_compared - 2


def test_upstream_of_a_levelled_set_node_is_dropped_though_it_cancels(monkeypatch):
    """When A states its level for today, a setting on A is written as a change from the
    status quo, so Z cancels exactly and shows nothing. It is still cut off, so it is still
    dropped: one fewer look for the noise floor to share."""
    module, calls = record_driver_calls(monkeypatch)
    set_a = analyse(frame="level", threshold=0.6, options=SET_A_OPTIONS, upstream="levelled")[
        "set_a"
    ]

    quantities, meets, informative, kwargs = calls[0]
    unexcluded = module.goal_chance_drivers(
        quantities, meets, informative, **{**kwargs, "upstream_of_set": frozenset()}
    )
    z_rows = [row for row in unexcluded.drivers if row.quantity_id == "z"]
    assert [row.status for row in z_rows] == ["below_resolution"]

    drivers = set_a.probability_of_goal_drivers
    assert not [row for row in drivers.drivers if row.quantity_id in {"z", "z->a"}]
    assert drivers.dropped_by_reason["set_by_option_upstream"] == 2
    assert drivers.dropped_by_reason["set_by_option"] == 1
    assert drivers.drivers[0].spread_noise_floor < unexcluded.drivers[0].spread_noise_floor
    # Contrast: for an option that leaves A alone, Z keeps its row and nothing is cut off.
    plain = analyse(upstream="levelled")["act"]
    assert plain.probability_of_goal_drivers.dropped_by_reason["set_by_option_upstream"] == 0
    assert row_for(plain, "z", "factor_value").status == "resolved"


def test_upstream_of_set_keeps_anything_with_another_route_to_the_goal():
    module = drivers_module()
    edges = [
        ("w", "z"),  # W reaches the goal only through Z, then A
        ("z", "a"),
        ("a", "goal"),
        ("y", "a"),  # Y has two routes: through A, and directly
        ("y", "goal"),
        ("lever", "goal"),
        ("q", "r"),  # no route to the goal at all
    ]

    assert module.quantities_upstream_of_set(edges, "goal", frozenset({"a"})) == frozenset(
        {"w", "z", "w->z", "z->a", "y->a"}
    )
    # Contrast: with nothing set, nothing is cut off; a set node elsewhere cuts only its own side.
    assert module.quantities_upstream_of_set(edges, "goal", frozenset()) == frozenset()
    assert module.quantities_upstream_of_set(edges, "goal", frozenset({"z"})) == frozenset(
        {"w", "w->z"}
    )


def test_correlated_factors_are_flagged_not_suppressed():
    act = analyse(nulls=1, correlations=[{"factor_a": "a", "factor_b": "b00", "rho": 0.6}])["act"]

    assert row_for(act, "a", "factor_value").correlated is True
    assert row_for(act, "b00", "factor_value").correlated is True
    link_rows = [
        row for row in act.probability_of_goal_drivers.drivers if row.kind == "link_strength"
    ]
    assert link_rows and all(row.correlated is None for row in link_rows)
    # Contrast: with no correlation plan nothing is flagged.
    plain = analyse(nulls=1)["act"]
    assert row_for(plain, "a", "factor_value").correlated is None


# ---------------------------------------------------------------- same draws, same rule


def test_tercile_hits_sum_to_the_figure_exactly():
    module = drivers_module()
    rng = np.random.default_rng(11)
    values = rng.normal(size=1000)
    meets = (values + rng.normal(scale=0.5, size=1000)) > 0.2

    counts = module.tercile_counts(values, meets)

    assert counts.low.n == counts.high.n == 333 and counts.mid.n == 334
    assert counts.low.met + counts.mid.met + counts.high.met == int(meets.sum())
    assert counts.low.met < counts.mid.met < counts.high.met
    # The cut values are the group boundaries themselves.
    assert counts.low_upper_value == np.sort(values)[332]
    assert counts.high_lower_value == np.sort(values)[667]


def direct_option_result(strict, quantities):
    """Low third of ``q`` sits exactly ON the threshold, the middle below it, and the top
    third alternates above and below, so its chance is one half whatever the strictness."""
    request = RobustnessRequestV2(
        **payload_for(threshold=0.5, strict=strict, n_samples=300, direction="maximise")
    )
    analyser = RobustnessAnalyzerV2()
    plan, warning = analyser._resolve_goal_threshold_in_sample_frame(request)
    assert warning is None
    samples = [0.5] * 100 + [0.25] * 100 + [0.875, 0.25] * 50
    results = analyser._compute_option_results(
        outcomes={option.id: samples for option in request.options},
        wins={option.id: 0 for option in request.options},
        request=request,
        goal_threshold_plan=plan,
        goal_chance_quantities=quantities,
    )
    return results[0]


@pytest.mark.parametrize(
    ("strict", "expected_figure", "expected_if_low"), [(None, 1 / 2, 1.0), (True, 1 / 6, 0.0)]
)
def test_drivers_use_the_figures_own_strictness(strict, expected_figure, expected_if_low):
    module = drivers_module()
    quantity = module.GoalChanceQuantity(
        quantity_id="q", kind="factor_value", values=np.arange(300, dtype=float)
    )

    result = direct_option_result(strict, [quantity])

    assert result.probability_of_goal == expected_figure
    row = row_for(result, "q", "factor_value")
    # A draw ON the threshold is met unless the goal is strict; the drivers must agree.
    assert (row.p_goal_if_low, row.p_goal_if_high) == (expected_if_low, 0.5)
    precision = result.probability_of_goal_precision
    assert precision.n_met / precision.n_informative == result.probability_of_goal


def test_non_informative_draws_are_outside_every_group():
    module = drivers_module()
    values = np.arange(400, dtype=float)
    meets = values >= 200
    informative = np.ones(400, dtype=bool)
    informative[:100] = False  # the lowest hundred draws carry no information

    drivers = module.goal_chance_drivers(
        [module.GoalChanceQuantity(quantity_id="q", kind="factor_value", values=values)],
        meets,
        informative,
    )

    row = drivers.drivers[0]
    assert row.n_low == row.n_high == 100  # thirds of the 300 informative draws
    assert (row.low_upper_value, row.high_lower_value) == (199.0, 300.0)
    assert (row.p_goal_if_low, row.p_goal_if_high) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        (np.full(300, 2.0), "no_variance"),
        (np.arange(60, dtype=float), "group_below_min_n"),
        (np.array([0.0] * 150 + [1.0] * 150), "tied_at_tercile_boundary"),
        (np.array([math.nan] + [float(i) for i in range(299)]), "non_finite_values"),
    ],
)
def test_quantities_that_cannot_be_split_are_dropped_with_a_reason(values, reason):
    module = drivers_module()
    meets = np.arange(values.size) % 2 == 0

    assert module.tercile_counts(values, meets) == reason
    # Contrast: a clean quantity of the same length is split.
    clean = np.arange(max(values.size, 90), dtype=float)
    assert not isinstance(module.tercile_counts(clean, np.arange(clean.size) % 2 == 0), str)


# ---------------------------------------------------------------- nothing to compare


def test_constant_outcome_lists_no_drivers_on_the_served_wire():
    """On Paul's served request no draw of any option reaches the target, so no grouping of
    the draws can differ from the figure: nothing is listed and every candidate is counted.
    (Control: ``test_main_driver_is_the_uncertain_factor`` lists rows when draws differ.)"""
    payload = json.loads(SERVED_WIRE.read_text())
    results = RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**payload)).results

    assert len(results) == 5
    for result in results:
        assert result.probability_of_goal == 0.0
        drivers = result.probability_of_goal_drivers
        assert drivers.drivers == []
        assert drivers.n_candidates == 25
        assert drivers.dropped_by_reason["outcome_constant"] == 25
        assert (drivers.n_compared, drivers.n_dropped) == (0, 25)
        # The precision of that zero is still stated.
        precision = result.probability_of_goal_precision
        assert (precision.n_met, precision.n_informative) == (0, 10000)
        assert precision.interval_lower == 0.0 < precision.interval_upper < 0.001


def test_constant_outcome_at_one_lists_no_drivers():
    # Level frame: A cancels against the paired status quo, so "act" meets the target on every draw.
    act = analyse(frame="level", threshold=0.6)["act"]

    assert act.probability_of_goal == 1.0
    drivers = act.probability_of_goal_drivers
    assert drivers.drivers == []
    assert drivers.dropped_by_reason["outcome_constant"] == drivers.n_candidates == 3
    assert act.probability_of_goal_precision.interval_upper == 1.0


def test_zero_spread_quantity_is_counted_not_listed_and_stays_in_the_family():
    module = drivers_module()
    draw = np.arange(300)
    # Fifty hits in the first hundred draws and fifty in the last hundred.
    meets = (draw < 50) | ((draw >= 200) & (draw < 250))
    flat = module.GoalChanceQuantity(
        quantity_id="flat", kind="factor_value", values=draw.astype(float)
    )
    # Ordered so that every hit sits in its low third.
    real = module.GoalChanceQuantity(
        quantity_id="real", kind="factor_value", values=-meets.astype(float) + draw * 1e-6
    )

    drivers = module.goal_chance_drivers([flat, real], meets, np.ones(300, dtype=bool))

    assert [row.quantity_id for row in drivers.drivers] == ["real"]
    assert (drivers.drivers[0].p_goal_if_low, drivers.drivers[0].p_goal_if_high) == (1.0, 0.0)
    assert drivers.dropped_by_reason["zero_spread"] == 1
    assert (drivers.n_candidates, drivers.n_compared, drivers.n_dropped) == (2, 2, 1)
    # The quantity that showed nothing was still looked at: the floor is shared across both.
    assert drivers.drivers[0].spread_noise_floor == module.family_noise_floor(100, 100, 2)
    assert drivers.drivers[0].spread_noise_floor != module.family_noise_floor(100, 100, 1)


# ---------------------------------------------------------------- precision


def test_wilson_interval_contains_the_figure_and_narrows_with_draws():
    module = drivers_module()

    assert module.wilson_interval(50, 100) == pytest.approx((0.40383, 0.59617), abs=1e-5)
    for met, n in [(0, 100), (1, 100), (37, 100), (99, 100), (100, 100)]:
        lower, upper = module.wilson_interval(met, n)
        assert 0.0 <= lower <= met / n <= upper <= 1.0
    assert module.wilson_interval(0, 100)[0] == 0.0
    assert module.wilson_interval(100, 100)[1] == 1.0
    narrow, wide = module.wilson_interval(960, 2000), module.wilson_interval(96, 200)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


def test_precision_on_the_wire_is_labelled_simulation_precision():
    few, many = analyse(n_samples=200)["act"], analyse(n_samples=2000)["act"]

    for result, draws in ((few, 200), (many, 2000)):
        precision = result.probability_of_goal_precision
        assert (precision.basis, precision.method) == ("simulation_precision", "wilson_score")
        assert precision.confidence_level == 0.95
        assert precision.n_informative == draws
        assert precision.n_met / precision.n_informative == result.probability_of_goal
        assert precision.interval_lower < result.probability_of_goal < precision.interval_upper

    def width(result):
        precision = result.probability_of_goal_precision
        return precision.interval_upper - precision.interval_lower

    assert width(many) < width(few)


# ---------------------------------------------------------------- omitted with the figure


def test_no_threshold_omits_every_new_field():
    act = analyse(threshold=None)["act"]

    assert act.probability_of_goal is None
    assert act.probability_of_goal_precision is None
    assert act.probability_of_goal_drivers is None
    dumped = act.model_dump(exclude_none=True)
    assert not [key for key in dumped if key.startswith("probability_of_goal")]
    # Contrast: the same graph with a threshold carries all three.
    with_threshold = analyse()["act"].model_dump(exclude_none=True)
    assert {key for key in with_threshold if key.startswith("probability_of_goal")} == {
        "probability_of_goal",
        "probability_of_goal_precision",
        "probability_of_goal_drivers",
    }


def test_withheld_partial_extremum_omits_every_new_field():
    module = drivers_module()
    request = RobustnessRequestV2(**payload_for(threshold=0.5, n_samples=100))
    analyser = RobustnessAnalyzerV2()
    plan, _ = analyser._resolve_goal_threshold_in_sample_frame(request)
    quantity = module.GoalChanceQuantity(
        quantity_id="q", kind="factor_value", values=np.arange(100, dtype=float)
    )

    def result_for(samples):
        return analyser._compute_option_results(
            outcomes={option.id: samples for option in request.options},
            wins={option.id: 0 for option in request.options},
            request=request,
            goal_threshold_plan=plan,
            goal_chance_quantities=[quantity],
        )[0]

    # Every informative draw meets the goal, but two draws are not informative: withheld.
    withheld = result_for([0.75] * 98 + [math.inf, -math.inf])
    assert withheld.probability_of_goal is None
    assert withheld.probability_of_goal_precision is None
    assert withheld.probability_of_goal_drivers is None
    # Contrast: a mid-range figure on the same partial population is reported, over 98 draws.
    reported = result_for([0.75] * 49 + [0.25] * 49 + [math.inf, -math.inf])
    assert reported.probability_of_goal == 0.5
    assert reported.probability_of_goal_precision.n_informative == 98
    assert row_for(reported, "q", "factor_value").n_low == 32


# ---------------------------------------------------------------- determinism and bookkeeping


def test_same_seed_gives_identical_output():
    def dumped(seed):
        results = analyse(nulls=3, lever_exists=0.5, seed=seed)
        return json.dumps(
            {
                option_id: result.model_dump(
                    by_alias=True,
                    include={
                        "probability_of_goal",
                        "probability_of_goal_precision",
                        "probability_of_goal_drivers",
                    },
                )
                for option_id, result in results.items()
            },
            sort_keys=True,
        )

    assert dumped(7) == dumped(7)
    assert dumped(7) != dumped(8)
    # The comparison covers the new blocks, not only the figure.
    assert '"kind": "link_existence"' in dumped(7)
    assert '"basis": "simulation_precision"' in dumped(7)


def test_sampler_records_absent_links_without_consuming_draws():
    payload = payload_for(lever_exists=0.5)
    edges = RobustnessRequestV2(**payload).graph.edges
    recorded = DualUncertaintySampler(edges, SeededRNG(3))
    configs = [recorded.sample_edge_configuration() for _ in range(200)]

    assert len(recorded.absent_edges_per_sample) == 200
    for config, absent in zip(configs, recorded.absent_edges_per_sample, strict=True):
        assert absent == frozenset(key for key, strength in config.items() if strength == 0.0)
    assert (
        60 < sum(("lever", "goal") in absent for absent in recorded.absent_edges_per_sample) < 140
    )
    assert not any(("a", "goal") in absent for absent in recorded.absent_edges_per_sample)


# ---------------------------------------------------------------- V2 wire


@pytest.fixture
def auth_headers():
    if os.environ.get("ISL_AUTH_DISABLED", "").lower() == "true":
        return {}
    return {"X-API-Key": os.environ.get("ISL_API_KEY", "test_key")}


def find_option(node, option_id):
    if isinstance(node, dict):
        if node.get("id") == option_id and "probability_of_goal" in node:
            return node
        node = list(node.values())
    if isinstance(node, list):
        for child in node:
            found = find_option(child, option_id)
            if found is not None:
                return found
    return None


def test_v2_response_carries_both_fields_with_no_nulls(auth_headers):
    response = TestClient(app).post(
        ENDPOINT, json=payload_for(lever_exists=0.5), headers=auth_headers
    )
    assert response.status_code == 200, response.text
    act = find_option(response.json(), "act")

    precision = act["probability_of_goal_precision"]
    assert precision["basis"] == "simulation_precision"
    assert precision["n_met"] / precision["n_informative"] == act["probability_of_goal"]
    rows = {
        (row["quantity_id"], row["kind"]): row
        for row in act["probability_of_goal_drivers"]["drivers"]
    }
    existence = rows[("lever->goal", "link_existence")]
    assert (existence["from"], existence["to"]) == ("lever", "goal")
    assert {"p_goal_if_absent", "p_goal_if_present", "n_absent", "n_present"} <= set(existence)
    assert "p_goal_if_low" not in existence
    factor = rows[("a", "factor_value")]
    assert {"p_goal_if_low", "p_goal_if_high", "low_upper_value", "high_lower_value"} <= set(factor)
    assert not {"from", "to", "p_goal_if_absent", "correlated"} & set(factor)
    assert "null" not in json.dumps(act["probability_of_goal_drivers"])
