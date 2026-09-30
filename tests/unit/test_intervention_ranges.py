"""TEMPORAL step 1 — a per-option duration RANGE, sampled for that option's own limit only.

Brief: olumi-programme-docs ``dl/claude-27fbe09b`` → ``dl-claude-27fbe09b/briefs/TEMPORAL.md``.
Science: R3 #75 5909972020, where a stated range is read as the QUARTILES (coverage 0.5, ISL's
``RATIFIED_COVERAGE``), fitted lognormal, and disclosed. Scope (Paul, 30 Sep): only each option's
own chance of meeting a duration limit changes. EVPI, the joint across limits, win share and
cross-option comparisons are held back. Any doubt about the range fails closed, meaning the
limit is refused by name and never scored at the point.

WHY. Today every option's duration is ONE number, so a limit on it scores exactly 1.0 or 0.0
(pristine below: ``plan`` 1.0, ``late`` 0.0). That certainty comes from the input, not from
the analysis.

THE WITNESS GRAPH, in a 0–40 day frame (``normalisation`` raw_at_zero = 0, raw_at_one = 40)::

    d  root factor "migration downtime", observed 0.25 (10 d)
    f  root factor "demand", uniform(0, 1)
    g  goal, g = −0.5 d + 0.5 f
    plan = do(d := 0.25) with the range 5–20 d, meaning "likely_range" (median √(5·20) = 10 d)
    late = do(d := 0.525) (21 d, a point)      hold = no intervention (d stays at 10 d)
    limit "dt": d <= 0.35 (14 d), value_frame 'level'

Hand truth for ``plan``: X ~ lognormal with quartiles (5, 20), so
P(X <= 14) = Φ((ln 14 − ln 10) / σ), with σ = ln(20/5) / (2 · z75) = 0.628323 (R3's 0.628).
With n = 10,000 one standard error is about 0.005, and TOL = 0.02 is about 4 SE. Every row is
located by constraint_id and option id, never by a value predicate.
"""

import copy
import hashlib
import json
import math
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.models.robustness_v2 import (
    EdgeV2,
    GoalConstraint,
    GraphV2,
    InterventionOption,
    LevelDomain,
    NodeV2,
    ObservedState,
    ParameterUncertainty,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

N_SAMPLES = 10_000
SEED = 42
TOL = 0.02
CID = "dt"
FRAME_DAYS = 40.0
Z75 = 0.6744897501960817  # standard-normal quantile at 0.75
NORMALISATION = {"raw_at_zero": 0.0, "raw_at_one": FRAME_DAYS}
NOT_CONVERTIBLE = "CONSTRAINT_NOT_CONVERTIBLE"


def phi(t: float) -> float:
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))


def p_leq(low: float, high: float, limit: float) -> float:
    """Closed form P(X <= limit) for the lognormal whose QUARTILES are (low, high)."""
    mu = 0.5 * (math.log(low) + math.log(high))
    sigma = math.log(high / low) / (2.0 * Z75)
    return phi((math.log(limit) - mu) / sigma)


def days(x: float) -> float:
    return x / FRAME_DAYS


def stated(
    low: float = 5.0, high: float = 20.0, meaning: str = "likely_range", **extra: Any
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "low": low,
        "high": high,
        "meaning": meaning,
        "normalisation": dict(NORMALISATION),
    }
    out.update(extra)
    return out


def edge(src: str, dst: str, mean: float) -> EdgeV2:
    return EdgeV2(
        **{"from": src, "to": dst},
        exists_probability=1.0,
        strength=StrengthDistribution(mean=mean, std=0.0011),
    )


def build(
    *,
    ranges: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    points: Optional[Dict[str, Dict[str, float]]] = None,
    constraints: Optional[List[GoalConstraint]] = None,
    d_parent: bool = False,
    include_voi: bool = False,
    n_samples: int = N_SAMPLES,
) -> RobustnessRequestV2:
    """``ranges`` maps option id → node id → range dict. At pristine the field is not declared,
    and ``extra: ignore`` drops it silently. That silent drop is the RED this file pins."""
    nodes = [
        NodeV2(
            id="d",
            kind="factor",
            label="Migration downtime",
            observed_state=ObservedState(value=0.25),
        ),
        NodeV2(id="f", kind="factor", label="Demand", observed_state=ObservedState(value=0.5)),
        NodeV2(id="g", kind="outcome", label="Goal"),
    ]
    edges = [edge("d", "g", -0.5), edge("f", "g", 0.5)]
    if d_parent:
        nodes.append(
            NodeV2(id="p", kind="factor", label="Vendor", observed_state=ObservedState(value=0.5))
        )
        edges.append(edge("p", "d", 0.5))
    option_points = points or {"plan": {"d": days(10)}, "late": {"d": days(21)}, "hold": {}}
    options = []
    for oid, iv in option_points.items():
        kwargs: Dict[str, Any] = {"id": oid, "label": oid, "interventions": iv}
        if ranges and oid in ranges:
            kwargs["intervention_ranges"] = copy.deepcopy(ranges[oid])
        options.append(InterventionOption(**kwargs))
    if constraints is None:
        constraints = [dt_limit()]
    return RobustnessRequestV2(
        request_id="temporal-step-1",
        graph=GraphV2(nodes=nodes, edges=edges),
        options=options,
        goal_node_id="g",
        n_samples=n_samples,
        seed=SEED,
        goal_constraints=constraints or None,
        parameter_uncertainties=[
            ParameterUncertainty(node_id="f", distribution="uniform", range_min=0.0, range_max=1.0)
        ],
        include_voi=include_voi,
    )


def dt_limit(**overrides: Any) -> GoalConstraint:
    kwargs: Dict[str, Any] = {
        "constraint_id": CID,
        "node_id": "d",
        "operator": "<=",
        "value": days(14),
        "label": "Migration downtime",
        "value_frame": "level",
    }
    kwargs.update(overrides)
    return GoalConstraint(**kwargs)


def run(request: RobustnessRequestV2):
    return RobustnessAnalyzerV2().analyze(request)


def result_of(response, option_id: str):
    (match,) = [r for r in response.results if r.option_id == option_id]
    return match


def row(response, option_id: str, constraint_id: str = CID):
    analysis = result_of(response, option_id).constraint_analysis
    assert analysis is not None, [
        (w.code, w.detail.get("reason")) for w in response.inference_warnings
    ]
    (found,) = [c for c in analysis.constraints if c.constraint_id == constraint_id]
    return found


def no_row(response, option_id: str, constraint_id: str = CID) -> bool:
    analysis = result_of(response, option_id).constraint_analysis
    return analysis is None or not any(
        c.constraint_id == constraint_id for c in analysis.constraints
    )


def refusals(response, reason: str) -> List[Any]:
    return [
        w
        for w in response.inference_warnings
        if w.code == NOT_CONVERTIBLE and w.detail.get("reason") == reason
    ]


def body(response) -> Dict[str, Any]:
    # exclude_none: the wire serialises this way, so an absent optional field is absent here too.
    dumped = response.model_dump(mode="json", by_alias=True, exclude_none=True)
    dumped["_metadata"].pop("execution_time_ms", None)
    return dumped


def digest(response) -> str:
    return hashlib.sha256(json.dumps(body(response), sort_keys=True).encode()).hexdigest()


RANGED = {"plan": {"d": stated()}}


# ---------------------------------------------------------------------------------------------------
# T0 — absent means byte-identical. GREEN at pristine and after: the digest was captured at f7f19e3.
# If an unrelated, reviewed change moves this body, re-capture the constant in that change.
# ---------------------------------------------------------------------------------------------------
PRISTINE_NO_RANGE_DIGEST = "da2b8df040fa7dbe6cb523e51099d29dc7bb737aa69d202a4122219f7c40b394"


def test_t0_a_request_without_ranges_is_byte_identical_to_pristine() -> None:
    assert digest(run(build(n_samples=2_000))) == PRISTINE_NO_RANGE_DIGEST


# ---------------------------------------------------------------------------------------------------
# T1 — the contract parses and validates. The caller states the range and its MEANING, never a family.
# ---------------------------------------------------------------------------------------------------
class TestT1Contract:
    def test_the_field_is_declared_and_kept(self) -> None:
        option = InterventionOption(
            id="plan", label="plan", interventions={"d": 0.25}, intervention_ranges={"d": stated()}
        )
        dumped = option.model_dump()
        assert dumped["intervention_ranges"]["d"]["low"] == 5.0
        assert dumped["intervention_ranges"]["d"]["meaning"] == "likely_range"
        assert dumped["intervention_ranges"]["d"]["normalisation"] == NORMALISATION

    def test_a_caller_cannot_choose_the_family(self) -> None:
        option = InterventionOption(
            id="plan",
            label="plan",
            interventions={"d": 0.25},
            intervention_ranges={"d": stated(family="normal")},
        )
        assert "family" not in option.model_dump()["intervention_ranges"]["d"]

    @pytest.mark.parametrize(
        "bad",
        [
            {"low": 0.0},
            {"low": -1.0},
            {"high": 5.0},
            {"high": 4.0},
            {"low": float("nan")},
            {"high": float("inf")},
            {"meaning": ""},
            {"normalisation": {"raw_at_zero": 0.0}},
            {"normalisation": {"raw_at_zero": 40.0, "raw_at_one": 40.0}},
            {"normalisation": {"raw_at_zero": 40.0, "raw_at_one": 0.0}},
        ],
    )
    def test_malformed_ranges_are_rejected(self, bad: Dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            InterventionOption(
                id="plan",
                label="plan",
                interventions={"d": 0.25},
                intervention_ranges={"d": stated(**bad)},
            )

    def test_a_range_must_sit_on_a_node_the_option_sets(self) -> None:
        with pytest.raises(ValidationError, match="intervention_ranges"):
            build(ranges={"hold": {"d": stated()}})

    def test_a_range_must_name_a_graph_node(self) -> None:
        with pytest.raises(ValidationError, match="intervention_ranges"):
            build(ranges={"plan": {"nope": stated()}}, points={"plan": {"d": 0.25}, "hold": {}})


# ---------------------------------------------------------------------------------------------------
# T2 — the fit: pure, closed form, quartiles on the stated bounds.
# ---------------------------------------------------------------------------------------------------
def test_t2_the_lognormal_fit_puts_the_stated_bounds_on_the_quartiles() -> None:
    from src.services.range_fit import RATIFIED_COVERAGE, fit_lognormal_range

    assert RATIFIED_COVERAGE == 0.5
    mu, sigma = fit_lognormal_range(5.0, 20.0)
    assert mu == pytest.approx(math.log(10.0), abs=1e-12)
    assert math.exp(mu - Z75 * sigma) == pytest.approx(5.0, rel=1e-12)
    assert math.exp(mu + Z75 * sigma) == pytest.approx(20.0, rel=1e-12)
    assert fit_lognormal_range(5.0, 20.0) == (mu, sigma)  # pure: no RNG, same answer twice


def test_t2b_the_sampled_series_has_the_stated_quartiles_and_median() -> None:
    import numpy as np

    series = RobustnessAnalyzerV2._intervention_range_series(build(ranges=RANGED), seed=SEED)
    xs = np.asarray(series["plan"]["d"]) * FRAME_DAYS
    assert len(xs) == N_SAMPLES
    assert np.percentile(xs, 25) == pytest.approx(5.0, rel=0.05)
    assert np.percentile(xs, 50) == pytest.approx(10.0, rel=0.05)
    assert np.percentile(xs, 75) == pytest.approx(20.0, rel=0.05)
    assert xs.min() > 0.0  # positive support: never a negative day


# ---------------------------------------------------------------------------------------------------
# T3 — two options that differ only by their range are not "identical options".
# ---------------------------------------------------------------------------------------------------
def test_t3_options_differing_only_by_range_are_not_identical() -> None:
    from src.validation.request_validator import (
        canonicalise_interventions,
        detect_identical_options,
    )

    same_point = {"d": 0.25}
    plain = {"id": "a", "interventions": same_point}
    ranged = {"id": "b", "interventions": same_point, "intervention_ranges": {"d": stated()}}
    assert detect_identical_options([plain, ranged]) is None
    ranged_twin = {"id": "c", "interventions": same_point, "intervention_ranges": {"d": stated()}}
    assert detect_identical_options([ranged, ranged_twin]) is not None
    assert detect_identical_options([plain, dict(plain, id="z")]) is not None  # control: unchanged
    assert canonicalise_interventions({"d": 0.25}) == canonicalise_interventions({"d": 0.25})


# ---------------------------------------------------------------------------------------------------
# T4 — THE ROW: the option's own chance of meeting the limit, from its range.
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("low,high", [(5.0, 20.0), (1.0, 6.0), (10.0, 30.0)])
def test_t4_a_ranged_option_scores_its_limit_from_the_range(low: float, high: float) -> None:
    median_days = math.sqrt(low * high)
    request = build(
        ranges={"plan": {"d": stated(low, high)}},
        points={"plan": {"d": days(median_days)}, "late": {"d": days(21)}, "hold": {}},
    )
    response = run(request)
    assert row(response, "plan").prob_satisfied == pytest.approx(p_leq(low, high, 14.0), abs=TOL)
    # Controls: a point option keeps its exact answer, and the status quo is a structural level.
    assert row(response, "late").prob_satisfied == 0.0
    assert row(response, "hold").prob_satisfied == 1.0


def test_t4b_r3s_number_for_lift_and_shift() -> None:
    assert p_leq(5.0, 20.0, 14.0) == pytest.approx(0.628323, abs=1e-6)
    assert row(run(build(ranges=RANGED)), "plan").prob_satisfied == pytest.approx(0.628323, abs=TOL)


# ---------------------------------------------------------------------------------------------------
# T5 — the echo: what ISL sampled, and how it read the range. Present only where it sampled.
# ---------------------------------------------------------------------------------------------------
EXPECTED_ECHO = [
    {
        "node_id": "d",
        "meaning": "likely_range",
        "family": "lognormal",
        "coverage": 0.5,
        "low": 5.0,
        "high": 20.0,
    }
]


def test_t5_the_v1_result_echoes_the_sampled_range_on_the_ranged_option_only() -> None:
    response = run(build(ranges=RANGED))
    echoed = result_of(response, "plan").sampled_intervention_ranges
    assert [e.model_dump() for e in echoed] == EXPECTED_ECHO
    assert result_of(response, "late").sampled_intervention_ranges is None
    assert result_of(response, "hold").sampled_intervention_ranges is None


def test_t5b_the_v2_wire_carries_the_echo() -> None:
    # The endpoint blocks an option with no interventions (EMPTY_INTERVENTIONS), so no 'hold' here.
    request = build(
        ranges=RANGED, points={"plan": {"d": days(10)}, "late": {"d": days(21)}}, n_samples=2_000
    )
    payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
    response = TestClient(app).post(
        "/api/v1/robustness/analyze/v2", json=payload, headers={"X-ISL-Response-Version": "2"}
    )
    assert response.status_code == 200, response.text
    options = {o["id"]: o for o in response.json()["options"]}
    assert options["plan"]["sampled_intervention_ranges"] == EXPECTED_ECHO
    assert "sampled_intervention_ranges" not in options["late"]


def test_t5c_a_range_with_no_limit_on_its_node_samples_nothing_and_echoes_nothing() -> None:
    response = run(build(ranges=RANGED, constraints=[]))
    assert result_of(response, "plan").sampled_intervention_ranges is None


# ---------------------------------------------------------------------------------------------------
# T6 — ONLY the limit row moves. Outcome, win share, other options and the envelope are unchanged.
# ---------------------------------------------------------------------------------------------------
def _strip_plan_limit(dumped: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dumped)
    for result in out["results"]:
        if result["option_id"] == "plan":
            result.pop("constraint_analysis", None)
            result.pop("sampled_intervention_ranges", None)
    return out


def test_t6_the_range_changes_only_the_ranged_options_limit_row() -> None:
    without = body(run(build(n_samples=2_000)))
    with_range = body(run(build(ranges=RANGED, n_samples=2_000)))
    # Positive control first: the row really moved, so the equality below is not vacuous.
    plan_without = [r for r in without["results"] if r["option_id"] == "plan"][0]
    plan_with = [r for r in with_range["results"] if r["option_id"] == "plan"][0]
    assert plan_without["constraint_analysis"] != plan_with["constraint_analysis"]
    assert _strip_plan_limit(without) == _strip_plan_limit(with_range)


def test_t6b_one_options_range_never_moves_another_options_draws() -> None:
    both = {"plan": {"d": stated()}, "late": {"d": stated(15.0, 29.4)}}
    alone = run(build(ranges=RANGED))
    together = run(build(ranges=both))
    assert row(alone, "plan").model_dump() == row(together, "plan").model_dump()


def test_t6c_same_seed_same_answer() -> None:
    assert digest(run(build(ranges=RANGED, n_samples=2_000))) == digest(
        run(build(ranges=RANGED, n_samples=2_000))
    )


# ---------------------------------------------------------------------------------------------------
# T7 — FAIL CLOSED. Any doubt about the range refuses the limit by name. It is never scored at the point.
# ---------------------------------------------------------------------------------------------------
class TestT7FailClosed:
    def _assert_refused(self, response, reason: str, **detail: Any) -> None:
        found = refusals(response, reason)
        assert len(found) == 1, [
            (w.code, w.detail.get("reason")) for w in response.inference_warnings
        ]
        assert found[0].severity == "warning"
        assert found[0].detail["constraint_id"] == CID
        for key, value in detail.items():
            assert found[0].detail[key] == value, (key, found[0].detail)
        for option in ("plan", "late", "hold"):
            assert no_row(response, option), option

    def test_a_range_meaning_other_than_likely_range_is_refused(self) -> None:
        response = run(build(ranges={"plan": {"d": stated(meaning="min_max")}}))
        self._assert_refused(
            response, "intervention_range_meaning_unsupported", option_id="plan", meaning="min_max"
        )

    def test_a_point_outside_the_ranges_p40_to_p60_is_refused(self) -> None:
        # R3 5911436566 (1): for 5–20 d the middle band is 7.7–13.0 d. "About 6 days" and
        # "5–20 days" are two different middles: ask, don't analyse around either.
        response = run(
            build(
                ranges=RANGED, points={"plan": {"d": days(6)}, "late": {"d": days(21)}, "hold": {}}
            )
        )
        self._assert_refused(response, "intervention_range_point_conflict", option_id="plan")
        detail = refusals(response, "intervention_range_point_conflict")[0].detail
        assert detail["implied_median"] == pytest.approx(10.0)
        assert detail["point"] == pytest.approx(6.0)
        assert detail["middle_band"] == pytest.approx([7.70, 12.98], abs=0.01)

    def test_a_point_inside_the_middle_band_is_scored_from_the_range(self) -> None:
        # "Likely 9" with 5–20: same side either way (0.666 vs 0.628), so the range is scored.
        response = run(
            build(
                ranges=RANGED, points={"plan": {"d": days(9)}, "late": {"d": days(21)}, "hold": {}}
            )
        )
        assert row(response, "plan").prob_satisfied == pytest.approx(
            p_leq(5.0, 20.0, 14.0), abs=TOL
        )

    def test_a_point_that_moves_the_verdict_to_the_other_side_is_refused(self) -> None:
        # Limit 11 d: median 10 → 0.537 (more likely than not); point 12 d → 0.466 (not).
        response = run(
            build(
                ranges=RANGED,
                points={"plan": {"d": days(12)}, "late": {"d": days(21)}, "hold": {}},
                constraints=[dt_limit(value=days(11))],
            )
        )
        self._assert_refused(response, "intervention_range_point_flips_verdict", option_id="plan")

    def test_a_non_level_limit_on_a_ranged_node_is_refused(self) -> None:
        response = run(
            build(ranges=RANGED, constraints=[dt_limit(value=days(4), value_frame="change_abs")])
        )
        self._assert_refused(response, "intervention_range_frame_unsupported", option_id="plan")

    def test_a_level_domain_never_truncates_the_range_and_the_tail_is_named(self) -> None:
        # R3 5911436566 (3): never truncate. Scored on the untruncated fit; the row names the
        # share of draws beyond the stated bound ("about 1 in 11 runs take more than 40 days").
        capped = dt_limit(level_domain=LevelDomain(min=0.0, max=1.0))
        plan = row(run(build(ranges=RANGED, constraints=[capped])), "plan")
        assert plan.prob_satisfied == pytest.approx(0.628323, abs=TOL)
        tail = 1.0 - phi(math.log(40.0 / 10.0) / (math.log(4.0) / (2.0 * Z75)))
        assert tail == pytest.approx(0.0887, abs=1e-4)
        assert plan.level_out_of_domain_fraction == pytest.approx(tail, abs=TOL)

    def test_truncation_that_would_flip_the_verdict_is_refused(self) -> None:
        # Bound at 8 d, limit ≤ 9 d: clamped → 1.0, untruncated → 0.459. Sides differ: ask.
        capped = dt_limit(value=days(9), level_domain=LevelDomain(min=0.0, max=days(8)))
        response = run(build(ranges=RANGED, constraints=[capped]))
        self._assert_refused(
            response, "intervention_range_truncation_flips_verdict", option_id="plan"
        )

    def test_a_floor_at_zero_days_is_not_crossed(self) -> None:
        floored = dt_limit(level_domain=LevelDomain(min=0.0))
        response = run(build(ranges=RANGED, constraints=[floored]))
        assert row(response, "plan").prob_satisfied == pytest.approx(0.628323, abs=TOL)

    def test_quartiles_outside_the_normalised_domain_are_refused(self) -> None:
        response = run(
            build(
                ranges={"plan": {"d": stated(5.0, 100.0)}},
                points={"plan": {"d": days(math.sqrt(500.0))}, "late": {"d": days(21)}, "hold": {}},
            )
        )
        found = refusals(response, "constraint_values_outside_normalised_domain")
        assert len(found) == 1
        assert "pinned_range_high[plan]" in found[0].detail["out_of_domain"]

    def test_a_limit_downstream_of_a_ranged_node_is_refused_not_scored_at_the_point(self) -> None:
        goal_limit = GoalConstraint(
            constraint_id="goal",
            node_id="g",
            operator=">=",
            value=0.0,
            label="Goal",
            value_frame="delta",
        )
        response = run(build(ranges=RANGED, constraints=[dt_limit(), goal_limit]))
        found = refusals(response, "intervention_range_not_propagated")
        assert len(found) == 1 and found[0].detail["constraint_id"] == "goal"
        assert found[0].detail["ranged_node_id"] == "d"
        # B5 control: the sibling limit on the ranged node itself is still scored.
        assert row(response, "plan").prob_satisfied == pytest.approx(0.628323, abs=TOL)

    def test_a_range_on_a_non_root_node_is_refused(self) -> None:
        response = run(build(ranges=RANGED, d_parent=True))
        self._assert_refused(response, "intervention_range_non_root_target", option_id="plan")

    def test_control_the_same_limit_without_a_range_is_scored(self) -> None:
        response = run(build(points={"plan": {"d": days(9)}, "late": {"d": days(21)}, "hold": {}}))
        assert row(response, "plan").prob_satisfied == 1.0


# ---------------------------------------------------------------------------------------------------
# T8 — held back: the joint across limits, and EVPI, when a row was scored from a range.
# ---------------------------------------------------------------------------------------------------
def test_t8_the_joint_is_withheld_for_an_option_whose_rows_mix_a_range_and_other_limits() -> None:
    demand = GoalConstraint(
        constraint_id="demand",
        node_id="f",
        operator=">=",
        value=0.2,
        label="Demand",
        value_frame="level",
    )
    response = run(build(ranges=RANGED, constraints=[dt_limit(), demand]))
    plan = result_of(response, "plan").constraint_analysis
    assert plan.joint_probability is None and plan.conditional_probabilities is None
    assert row(response, "plan", "demand").prob_satisfied == pytest.approx(0.8, abs=TOL)
    withheld = [w for w in response.inference_warnings if w.code == "CONSTRAINT_JOINT_WITHHELD"]
    assert [w.detail["option_id"] for w in withheld] == ["plan"]
    assert withheld[0].detail["reason"] == "intervention_range_joint_deferred"
    # Control: an option with no range keeps its joint.
    assert result_of(response, "late").constraint_analysis.joint_probability == 0.0


def test_t8b_a_single_ranged_limit_keeps_its_joint_equal_to_its_row() -> None:
    response = run(build(ranges=RANGED))
    plan = result_of(response, "plan").constraint_analysis
    assert plan.joint_probability == plan.constraints[0].prob_satisfied


def test_t8c_evpi_is_held_back_when_a_row_is_scored_from_a_range() -> None:
    held = run(build(ranges=RANGED, include_voi=True, n_samples=2_000))
    assert held.p_win_sensitivity is None
    reasons = [
        w.detail.get("reason") for w in held.inference_warnings if w.code == "EVPI_UNAVAILABLE"
    ]
    assert reasons == ["intervention_ranges_deferred"]
    control = run(build(include_voi=True, n_samples=2_000))
    assert not [w for w in control.inference_warnings if w.code == "EVPI_UNAVAILABLE"]
