"""An anchored level's domain comes from the UNIT's meaning, not from the frame (AIQ #72 5866289608).

THE FINDING (served journey A run 3; CEE a4f4d2b · PLoT c0f0a9a · ISL 9b8aa34). The £59 option's card read
"90th percentile £125,000.00" beside "mean £126,363". ``anchored_level_domain`` gave any node that carried an
``observed_state.cap`` a ceiling of 1.0 ("the cap is read for its PRESENCE only"), so the goal band was clamped
at the cap. For MRR the cap is CEE's normalisation frame (£125,000), not a limit: CEE's factor enricher mints a cap
only for NON-'%' quantities above 1 (money, counts), and a '%' node may carry cap 100 — presence says nothing.

AIQ'S RULING. Percent/share/probability -> [0, 100%]; money/counts -> no ceiling, >= 0 unless signed. ISL cannot
read a unit's meaning off a frame, and does not parse unit strings: the caller owns units. The one carrier of unit
meaning ISL receives is the ``level_domain`` PLoT mints for a '%' LEVEL limit (``levelDomainFor``, the '%' rung),
in the node's own level frame. So a node's ceiling (and floor, where stated) comes from that, and nothing else.

ROWS. (1) the served £59 option: p90 > 1.0 on the 0-1 frame (> £125,000), mean unchanged (1.0109 -> £126,363);
(2) a '%' node (monthly churn) bounded to [0, 1] by its limit's unit meaning, with the out-of-domain share disclosed
(node_levels[churn].level_out_of_domain_share and the limit's level_out_of_domain_fraction); (3) mutant: the cap
read as a ceiling turns row 1 RED; (4) PLoT's '%' {0, 1} is that meaning ONLY on a 100-point frame (DL ISL #196
5869037504): off it, [0, 1] is the frame, so no ceiling (the DL's probe, churn 19.8% on a 20-point frame with
"<= 21%", keeps 0.733 / 0.6905); mutant: the domain applied regardless of frame turns row 4 RED.

FIXTURE: ``journey_a_run3_status_quo_held_plot_c0f0a9a.json`` (``_provenance`` inside), the served ISL request,
unedited. The DERIVED rows (``steep_churn_request``) are labelled as such: churn is held near 100% and one edge
made steep so that churn's levels leave [0, 1] and the ceiling has something to bind on. They are about the clamp's
mechanics, not evidence about the wire.
"""

from __future__ import annotations

import copy
import json

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

from src.models.robustness_v2 import LevelDomain, NodeV2, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    RobustnessAnalyzerV2,
    anchored_level_domain,
    node_level_frame,
    unit_level_domains,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "anchored_delta"
    / "journey_a_run3_status_quo_held_plot_c0f0a9a.json"
)
CAP_GBP = 125_000.0  # the goal's observed_state.cap: CEE's frame for MRR (0.6 = £75,000)

GOAL = "mrr"
CHURN = "monthly_churn"
RISK = "price_related_cancellation_risk"
KEEP = "keep_current_pricing"
P59 = "59_with_feature_release"
P54 = "54_with_feature_release"
CHURN_LIMIT = "agent-lane:monthly_churn:<="
STEEP_HELD = 0.99  # derived rows only: churn held at 99%
STEEP_LIMIT = 0.99  # derived rows only: "churn <= 99%"

# Measured at plain staging a1fa8ae through the V2 route (the served card's "mean £126,363"). RE-PINNED on the
# rebase: it was 1.0109026581600915 at 14f1a3a; ISL #193 (a1fa8ae, one central constant for a product identity)
# moved it by ~£0.33 on plain staging, without this change. This change does not move it (the mean was never
# clamped): the same value at a1fa8ae and on this branch.
P59_MEAN_AT_STAGING = 1.0109000416302791


def served_request() -> Dict[str, Any]:
    payload = json.loads(FIXTURE.read_text())
    assert "_provenance" in payload, "the fixture must say where it came from"
    return copy.deepcopy(payload["request"])


def steep_churn_request(
    *,
    level_domain: bool = True,
    churn_cap: Optional[float] = None,
    churn_raw_value: float = STEEP_HELD * 100,
    limit_value: float = STEEP_LIMIT,
) -> Dict[str, Any]:
    """DERIVED from the served request, so that churn's LEVELS leave [0, 1] and a ceiling has something to bind
    on: churn is held at 99% (was 3%), ``price_related_cancellation_risk -> monthly_churn`` is as steep as an edge
    can be (strength -1, the engine's bound; was 0.0075), and the limit reads "churn <= 99%" so that the draws
    that fail it are the ones pushed toward or past 100%. Optionally the limit loses its unit meaning
    (``level_domain``), or churn carries a cap (the '%'-with-cap-100 shape CEE's graph-data-integrity transform
    allows), or churn's ``raw_value`` puts it on another frame (the pair ``{0.99, raw}`` is ``raw / 0.99`` points)
    with the limit restated in that frame (``limit_value``)."""
    d = served_request()
    (edge,) = [e for e in d["graph"]["edges"] if e["from"] == RISK and e["to"] == CHURN]
    edge["strength"] = {"mean": -1.0, "std": 0.05}
    (churn,) = [n for n in d["graph"]["nodes"] if n["id"] == CHURN]
    churn["observed_state"].update(
        value=STEEP_HELD, baseline=STEEP_HELD, raw_value=churn_raw_value
    )
    if churn_cap is not None:
        churn["observed_state"]["cap"] = churn_cap
    (limit,) = [c for c in d["goal_constraints"] if c["constraint_id"] == CHURN_LIMIT]
    limit["value"] = limit_value
    if not level_domain:
        limit.pop("level_domain")
    d["n_samples"] = 2_000
    d["analysis_types"] = ["comparison"]
    d["include_e_values"] = False
    d["include_voi"] = False
    d["include_factor_flips"] = False
    return d


def post_v2(d: Dict[str, Any]) -> Dict[str, Any]:
    from fastapi.testclient import TestClient

    from src.api.main import app

    response = TestClient(app).post(
        "/api/v1/robustness/analyze/v2", json=d, headers={"X-ISL-Response-Version": "2"}
    )
    assert response.status_code == 200, response.text
    return response.json()


def analyse(d: Dict[str, Any]):
    return RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))


def wire_option(body: Dict[str, Any], option_id: str) -> Dict[str, Any]:
    (option,) = [o for o in body["options"] if o["id"] == option_id]
    return option


def wire_frame(body: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    (frame,) = [f for f in body.get("node_levels") or [] if f["node_id"] == node_id]
    return frame


def frame_of(response, node_id: str):
    (frame,) = [f for f in response.node_levels or [] if f.node_id == node_id]
    return frame


def result(response, option_id: str):
    (row,) = [r for r in response.results if r.option_id == option_id]
    return row


def churn_limit_row(response, option_id: str):
    analysis = result(response, option_id).constraint_analysis
    assert analysis is not None, option_id
    (row,) = [c for c in analysis.constraints if c.constraint_id == CHURN_LIMIT]
    return row


@pytest.fixture(scope="module")
def served_wire() -> Dict[str, Any]:
    return post_v2(served_request())


@pytest.fixture(scope="module")
def served_v1():
    return analyse(served_request())


# ---------------------------------------------------------------------------------------------------------
# Row 1 — the money goal has no ceiling: the served £59 option's p90 is its own, not the cap
# ---------------------------------------------------------------------------------------------------------


class TestRow1TheServedMoneyGoalIsNotClampedAtItsCap:
    def test_the_59_option_p90_is_above_the_cap(self, served_wire):
        """RED at base: p90 == 1.0 exactly (£125,000.00, the cap)."""
        outcome = wire_option(served_wire, P59)["outcome"]
        assert outcome["percentiles_source"] == "samples"
        assert outcome["p90"] > 1.0, outcome
        assert outcome["p90"] * CAP_GBP > CAP_GBP

    def test_the_59_option_mean_is_unchanged_and_below_its_p90(self, served_wire):
        """The mean was never clamped; with the ceiling gone, mean <= p90 holds by itself."""
        outcome = wire_option(served_wire, P59)["outcome"]
        assert outcome["mean"] == pytest.approx(P59_MEAN_AT_STAGING, abs=1e-12)
        assert outcome["mean"] <= outcome["p90"], outcome

    def test_the_goal_band_is_its_own_percentiles_floored_at_zero(self, served_wire, served_v1):
        """Bound by identity: every option's wire band IS np.percentile of the analyzer's own goal levels,
        with only the floor (>= 0) applied."""
        for option_id in (KEEP, P59, P54):
            levels = np.asarray(
                result(served_v1, option_id).outcome_distribution.samples, dtype=float
            )
            expected = [max(float(v), 0.0) for v in np.percentile(levels, [10, 50, 90])]
            outcome = wire_option(served_wire, option_id)["outcome"]
            assert [outcome["p10"], outcome["p50"], outcome["p90"]] == expected, option_id

    def test_the_goal_frame_states_a_floor_and_no_ceiling(self, served_wire, served_v1):
        frame = wire_frame(served_wire, GOAL)
        assert frame["frame"] == "anchored_level"
        assert frame["level_domain_min"] == 0.0
        assert frame.get("level_domain_max") is None, frame
        for option_id in (KEEP, P59, P54):
            levels = np.asarray(
                result(served_v1, option_id).outcome_distribution.samples, dtype=float
            )
            assert frame["level_out_of_domain_share"][option_id] == pytest.approx(
                float(np.mean(levels < 0.0)), abs=1e-12
            ), option_id

    def test_contrast_the_held_status_quo_still_reads_today(self, served_wire):
        """CONTROL: the status quo reproduces £75,000 at every percentile, before and after."""
        outcome = wire_option(served_wire, KEEP)["outcome"]
        for key in ("p10", "p50", "p90"):
            assert outcome[key] == pytest.approx(0.6, abs=1e-12), (key, outcome)

    def test_contrast_probability_of_goal_is_the_unclamped_comparison(self, served_v1):
        """At the served threshold the ceiling never reached the goal verdict: a threshold inside the domain
        is met or missed the same way on clamped and unclamped levels. 'MRR >= £100,000' (0.8) over the
        analyzer's own levels. ABOVE the frame it did (0.0 at base): test_level_domain_probability_impact.py.
        """
        for option_id in (KEEP, P59, P54):
            levels = np.asarray(
                result(served_v1, option_id).outcome_distribution.samples, dtype=float
            )
            assert result(served_v1, option_id).probability_of_goal == pytest.approx(
                float(np.mean(levels[np.isfinite(levels)] >= 0.8)), abs=1e-12
            ), option_id


# ---------------------------------------------------------------------------------------------------------
# Row 2 — a '%' node is bounded to [0, 1] by its unit's meaning, and the out-of-domain share is disclosed
# ---------------------------------------------------------------------------------------------------------


class TestRow2APercentNodeIsBoundedByItsUnitMeaning:
    def test_served_churn_is_bounded_to_zero_and_one(self, served_wire):
        """RED at base: churn's frame had a floor only (no cap on a '%' node), so nothing bounded it above."""
        frame = wire_frame(served_wire, CHURN)
        assert frame["frame"] == "anchored_level"
        assert (frame["level_domain_min"], frame.get("level_domain_max")) == (0.0, 1.0), frame
        assert set(frame["level_out_of_domain_share"]) == {KEEP, P59, P54}

    def test_served_churn_limit_discloses_its_out_of_domain_fraction(self, served_wire):
        for option_id in (KEEP, P59, P54):
            analysis = wire_option(served_wire, option_id)["constraint_analysis"]
            (row,) = [c for c in analysis["constraints"] if c["constraint_id"] == CHURN_LIMIT]
            assert row["level_out_of_domain_fraction"] is not None, option_id

    @pytest.fixture(scope="class")
    def steep(self):
        return analyse(steep_churn_request())

    def test_derived_churn_above_100_percent_is_disclosed(self, steep):
        """DERIVED: the £59 option pushes churn's level past 100% on a large share of draws; the node's share and
        the limit's fraction both say so, and the held status quo stays inside."""
        frame = frame_of(steep, CHURN)
        assert (frame.level_domain_min, frame.level_domain_max) == (0.0, 1.0)
        assert frame.level_out_of_domain_share[P59] > 0.3
        assert frame.level_out_of_domain_share[KEEP] == 0.0
        assert churn_limit_row(steep, P59).level_out_of_domain_fraction == pytest.approx(
            frame.level_out_of_domain_share[P59], abs=1e-12
        )

    def test_derived_churn_limit_reads_the_clamped_level(self, steep):
        """RED at base: 'churn <= 99%' fails by at most 1 point (100% - 99%) — the median failing draw is one
        clamped at 100% — never by an impossible level's margin."""
        assert churn_limit_row(steep, P59).failure_margin_median == pytest.approx(
            1.0 - STEEP_LIMIT, abs=1e-12
        )

    def test_contrast_without_unit_meaning_churn_has_no_ceiling(self):
        """The ceiling comes from the unit meaning ONLY: the same draws, the limit's level_domain removed."""
        response = analyse(steep_churn_request(level_domain=False))
        frame = frame_of(response, CHURN)
        assert (frame.level_domain_min, frame.level_domain_max) == (0.0, None)
        assert churn_limit_row(response, P59).failure_margin_median > (1.0 - STEEP_LIMIT) + 0.005

    def test_contrast_a_cap_is_never_read_as_a_ceiling(self):
        """RED at base: a '%' node carrying cap 100 was bounded by the cap's PRESENCE. The frame is not the
        unit's meaning: with no level_domain, no ceiling."""
        response = analyse(steep_churn_request(level_domain=False, churn_cap=100.0))
        frame = frame_of(response, CHURN)
        assert (frame.level_domain_min, frame.level_domain_max) == (0.0, None)
        assert churn_limit_row(response, P59).failure_margin_median > (1.0 - STEEP_LIMIT) + 0.005


# ---------------------------------------------------------------------------------------------------------
# Row 4 — PLoT's '%' {0, 1} is the unit's meaning ONLY on a 100-point frame (DL ISL #196 5869037504)
# ---------------------------------------------------------------------------------------------------------
#
# PLoT sends level_domain {0, 1} for EVERY '%' level limit (``levelDomainFor``), including the deferred '%' rung
# where the target's own frame is not 100 points. There [0, 1] means [0, frame]: on a 20-point frame it is
# [0%, 20%], the FRAME, not the unit's [0%, 100%]. Read as a ceiling it certified "churn <= 21%" for every draw.
# DERIVED (the DL's probe ``pct_frame20_le21``): churn held at 19.8% on a 20-point pair frame
# ({value 0.99, raw_value 19.8}), the limit "<= 21%" restated in that frame (21 / 20 = 1.05).

FRAME20_RAW = 19.8  # {0.99, 19.8}: a 20-point pair frame (19.8 / 0.99)
FRAME20_LIMIT = 21.0 / 20.0  # "churn <= 21%" on that frame
FRAME100_LIMIT = 1.01  # control: "churn <= 101%" on the 100-point frame ({0.99, 99})
# Measured at plain staging a1fa8ae and at base 14f1a3a (identical) through the V2 route and the analyzer: the
# probe's prob_satisfied with no ceiling. At d6defb4 (this PR before the guard) it was 1.0 for every option.
FRAME20_PROB_WITHOUT_A_CEILING = {KEEP: 1.0, P59: 0.733, P54: 0.6905}
# The control's prob_satisfied at base/staging (no ceiling on a capless '%' node then): 0.546 / 0.509.
FRAME100_PROB_AT_STAGING = {KEEP: 1.0, P59: 0.546, P54: 0.509}


class TestRow4APercentDomainIsTheUnitMeaningOnlyOnA100PointFrame:
    @pytest.fixture(scope="class")
    def frame20(self):
        return analyse(
            steep_churn_request(churn_raw_value=FRAME20_RAW, limit_value=FRAME20_LIMIT)
        )

    @pytest.fixture(scope="class")
    def frame20_without_domain(self):
        return analyse(
            steep_churn_request(
                churn_raw_value=FRAME20_RAW, limit_value=FRAME20_LIMIT, level_domain=False
            )
        )

    @pytest.fixture(scope="class")
    def frame100(self):
        return analyse(steep_churn_request(limit_value=FRAME100_LIMIT))

    def test_the_probe_is_not_a_certified_pass(self, frame20):
        """RED at d6defb4: prob_satisfied 1.0 for every option (churn clamped at 1.0 = 20% <= 21%)."""
        got = {o: churn_limit_row(frame20, o).prob_satisfied for o in (KEEP, P59, P54)}
        assert got == FRAME20_PROB_WITHOUT_A_CEILING

    def test_off_a_100_point_frame_the_limit_reads_as_if_no_domain_were_sent(
        self, frame20, frame20_without_domain
    ):
        """Bound by identity: on a 20-point frame the '%' domain moves nothing the node reports — every option's
        probability and failure margin equal the same request's with the limit's level_domain removed."""
        for option_id in (KEEP, P59, P54):
            with_domain = churn_limit_row(frame20, option_id)
            without = churn_limit_row(frame20_without_domain, option_id)
            assert with_domain.prob_satisfied == without.prob_satisfied, option_id
            assert with_domain.failure_margin_median == without.failure_margin_median, option_id
        assert churn_limit_row(frame20, P59).prob_satisfied < 1.0

    def test_off_a_100_point_frame_the_node_states_a_floor_and_no_ceiling(self, frame20):
        """RED at d6defb4: (0.0, 1.0) — the frame (20%) stated as the ceiling."""
        frame = frame_of(frame20, CHURN)
        assert (frame.level_domain_min, frame.level_domain_max) == (0.0, None)

    def test_control_on_a_100_point_frame_the_domain_is_kept(self, frame100):
        """CONTROL: the same shape on a 100-point pair frame ({0.99, 99}) keeps [0, 1] = [0%, 100%], so
        "churn <= 101%" cannot fail (it read 0.546 / 0.509 at staging, on impossible levels above 100%)."""
        frame = frame_of(frame100, CHURN)
        assert (frame.level_domain_min, frame.level_domain_max) == (0.0, 1.0)
        got = {o: churn_limit_row(frame100, o).prob_satisfied for o in (KEEP, P59, P54)}
        assert got == {KEEP: 1.0, P59: 1.0, P54: 1.0}
        assert got != FRAME100_PROB_AT_STAGING

    def test_unit_meaning_is_withheld_for_a_node_off_a_100_point_frame(self):
        """RED at d6defb4: the 20-point node's limit domain was taken as its unit meaning."""
        d = steep_churn_request(churn_raw_value=FRAME20_RAW, limit_value=FRAME20_LIMIT)
        assert unit_level_domains(RobustnessRequestV2.model_validate(d)) == {}
        control = steep_churn_request(limit_value=FRAME100_LIMIT)
        assert unit_level_domains(RobustnessRequestV2.model_validate(control)) == {
            CHURN: LevelDomain(min=0.0, max=1.0)
        }


# ---------------------------------------------------------------------------------------------------------
# The rule itself, one row per shape
# ---------------------------------------------------------------------------------------------------------


UNIT_PERCENT = LevelDomain(min=0.0, max=1.0)


@pytest.mark.parametrize(
    ("level", "unit_domain", "expected"),
    [
        pytest.param(0.6, None, (0.0, None), id="money-or-count: floor 0, no ceiling"),
        pytest.param(4.2, None, (0.0, None), id="money above its frame: no ceiling"),
        pytest.param(-0.2, None, (None, None), id="signed (held level negative): no floor"),
        pytest.param(0.03, UNIT_PERCENT, (0.0, 1.0), id="percent: [0, 1]"),
        pytest.param(
            1.1,
            UNIT_PERCENT,
            (0.0, None),
            id="percent held above 100% (NRR 110%): not clipped below its own level",
        ),
        pytest.param(
            -0.05,
            UNIT_PERCENT,
            (None, 1.0),
            id="percent held below 0: not floored above its own level",
        ),
        pytest.param(
            0.5,
            LevelDomain(max=1.0),
            (0.0, 1.0),
            id="a stated ceiling only: the sign rule keeps the floor",
        ),
    ],
)
def test_the_domain_is_the_unit_meaning_and_the_sign_rule(level, unit_domain, expected):
    assert anchored_level_domain(level, unit_domain) == expected


def _node(observed: Optional[Dict[str, Any]], execution_frame: Optional[Dict[str, Any]] = None) -> NodeV2:
    d: Dict[str, Any] = {"id": "n", "kind": "factor", "label": "n"}
    if observed is not None:
        d["observed_state"] = observed
    if execution_frame is not None:
        d["execution_frame"] = execution_frame
    return NodeV2.model_validate(d)


@pytest.mark.parametrize(
    ("observed", "execution_frame", "expected"),
    [
        pytest.param({"value": 0.03, "raw_value": 3}, None, 100.0, id="served churn pair {0.03, 3}: 100"),
        pytest.param({"value": 0.07, "raw_value": 7}, None, 100.0, id="float tail 7/0.07: coherent, 100"),
        pytest.param({"value": 0.99, "raw_value": 19.8}, None, 19.8 / 0.99, id="pair {0.99, 19.8}: 20"),
        pytest.param({"value": 0.2, "cap": 100}, None, 100.0, id="cap 100"),
        pytest.param({"value": 0.2, "cap": 20, "raw_value": 20}, None, 20.0, id="cap 20 outranks a 100 pair"),
        pytest.param(
            {"value": 0.03, "raw_value": 3},
            {"frame": 20, "carrier": "scale_frame"},
            20.0,
            id="PLoT's execution_frame (scale_frame 20) outranks a 100 pair",
        ),
        pytest.param({"value": 0.03}, None, None, id="no raw_value, no cap: unresolved"),
        pytest.param({"value": 0.0, "raw_value": 0}, None, None, id="value 0: scale-ambiguous"),
        pytest.param({"value": 0.5, "raw_value": 0.5}, None, None, id="{x, x}: not a frame"),
        pytest.param(None, None, None, id="no observed_state"),
    ],
)
def test_the_node_frame_is_plots_reader_on_the_rungs_isl_can_see(observed, execution_frame, expected):
    assert node_level_frame(_node(observed, execution_frame)) == expected


def test_only_a_level_limit_carries_unit_meaning_for_its_node():
    """The served churn limit ('level', PLoT's '%' domain) is churn's unit meaning; the goal has none. A 'delta'
    limit's domain is not: a change has no physical domain (ISL ignores it there too)."""
    d = served_request()
    assert unit_level_domains(RobustnessRequestV2.model_validate(d)) == {
        CHURN: LevelDomain(min=0.0, max=1.0)
    }
    (limit,) = [c for c in d["goal_constraints"] if c["constraint_id"] == CHURN_LIMIT]
    limit["value_frame"] = "delta"
    assert unit_level_domains(RobustnessRequestV2.model_validate(d)) == {}
