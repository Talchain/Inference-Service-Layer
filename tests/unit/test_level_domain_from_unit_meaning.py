"""An anchored level's domain comes from the UNIT's meaning, never from the frame (AIQ #72 5866289608).

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
read as a ceiling turns row 1 RED.

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

from src.models.robustness_v2 import LevelDomain, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    RobustnessAnalyzerV2,
    anchored_level_domain,
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

# Measured at base 14f1a3a through the V2 route (the served card's "mean £126,363").
P59_MEAN_AT_BASE = 1.0109026581600915


def served_request() -> Dict[str, Any]:
    payload = json.loads(FIXTURE.read_text())
    assert "_provenance" in payload, "the fixture must say where it came from"
    return copy.deepcopy(payload["request"])


def steep_churn_request(
    *, level_domain: bool = True, churn_cap: Optional[float] = None
) -> Dict[str, Any]:
    """DERIVED from the served request, so that churn's LEVELS leave [0, 1] and a ceiling has something to bind
    on: churn is held at 99% (was 3%), ``price_related_cancellation_risk -> monthly_churn`` is as steep as an edge
    can be (strength -1, the engine's bound; was 0.0075), and the limit reads "churn <= 99%" so that the draws
    that fail it are the ones pushed toward or past 100%. Optionally the limit loses its unit meaning
    (``level_domain``), or churn carries a cap (the '%'-with-cap-100 shape CEE's graph-data-integrity transform
    allows)."""
    d = served_request()
    (edge,) = [e for e in d["graph"]["edges"] if e["from"] == RISK and e["to"] == CHURN]
    edge["strength"] = {"mean": -1.0, "std": 0.05}
    (churn,) = [n for n in d["graph"]["nodes"] if n["id"] == CHURN]
    churn["observed_state"].update(
        value=STEEP_HELD, baseline=STEEP_HELD, raw_value=STEEP_HELD * 100
    )
    if churn_cap is not None:
        churn["observed_state"]["cap"] = churn_cap
    (limit,) = [c for c in d["goal_constraints"] if c["constraint_id"] == CHURN_LIMIT]
    limit["value"] = STEEP_LIMIT
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
        assert outcome["mean"] == pytest.approx(P59_MEAN_AT_BASE, abs=1e-12)
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
        """The ceiling never reached the goal verdict: a threshold inside the domain is met or missed the
        same way on clamped and unclamped levels. 'MRR >= £100,000' (0.8) over the analyzer's own levels.
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
