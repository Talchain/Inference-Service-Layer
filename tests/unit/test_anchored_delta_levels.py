"""B1a — ISL evaluates a NON-ROOT node that holds an attested level in ANCHORED-DELTA form.

Build train #70 5855068711 row B1; design MG 5855036638 + 5855037633; AIQ ruling 5855046894 (binding);
MG F-01 result + invariant correction 5855062769.

THE RULE. For a non-root node whose ``observed_state`` carries a held level ``o`` stated or estimated by an
ATTESTED author, on every draw::

    value = o + sum(strength * (parent - parent_sq))

where ``parent_sq`` is that parent's value in the SAME draw's status-quo evaluation (no interventions, no
epsilon; common random numbers). So the status quo reproduces ``o`` on every draw, and every sample of the
node is a LEVEL in its own frame. A non-root with no attested level keeps today's form and is flagged
"no level". A ``do(x)`` still pins the node.

THE FIXTURE is Paul's own persisted graph ``a6ed1bff`` as PLoT actually sent it to ISL, captured at
2026-09-27T10:27:59Z through a local proxy and matched leaf-for-leaf against the staging sha8 capture
(``tests/fixtures/anchored_delta/paul_a6ed1bff_plot_to_isl_request.json``, sha256 2ddc8367…1ca6). Three
documented edits, each because the WIRE disagrees with the persisted graph (a data-contract defect that is
NOT this slice's, reported separately):

* ``6dbac00d`` sets ``monthly_new_pro_subscribers`` to 1 on the wire; the graph holds 0.09 (90/month).
* ``ca47b368`` sets ``monthly_churn`` to 1 on the wire; the graph holds 0.025 (2.5%).
* ``146aa89d`` (£59, grandfather existing customers) is absent from the wire; it is re-added with its
  persisted price lever (``pro_plan_price`` 0.295), plus, for R5 only, its grandfathering switch.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    DualUncertaintySampler,
    FactorSampler,
    RobustnessAnalyzerV2,
    SCMEvaluatorV2,
)
from src.utils.rng import SeededRNG

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "anchored_delta"
    / "paul_a6ed1bff_plot_to_isl_request.json"
)
CAP_GBP = 125_000.0  # observed_state.cap of the goal node ``mrr`` (0.6 = £75,000)
N_SAMPLES = 2_000
SEED = 543524382  # the served request's own seed

GOAL = "mrr"
CHURN = "monthly_churn"
SUBS = "pro_paying_subscribers"
NEW_SUBS = "monthly_new_pro_subscribers"
OTHER_GROWTH = "other_mrr_growth"
PRICE_SENSITIVITY = "price_sensitivity"  # non-root, NO observed level on the wire
GRANDFATHERED = (
    "fac_existing_customers_grandfathered"  # ROOT, no observed state: ROOT_NODE_DEFAULT_VALUE
)
PRICE = "pro_plan_price"  # ROOT lever

KEEP = "keep_current_49_price"
P59 = "increase_price_to_59"
P54 = "increase_price_to_54"
GRANDFATHER = "146aa89d"
CONVERSION = "6dbac00d"
RETENTION = "ca47b368"
HOLD = "hold_no_change"  # a no-intervention option: the pure status quo, for exactness rows


def _wire() -> Dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def paul_request(
    *,
    options: List[str],
    n_samples: int = N_SAMPLES,
    grandfather_switch: bool = False,
) -> Dict[str, Any]:
    """The captured ISL request with the persisted option levels and only the options named."""
    d = _wire()
    persisted = {
        KEEP: {PRICE: 0.245},
        P59: {PRICE: 0.295},
        P54: {PRICE: 0.27},
        GRANDFATHER: ({PRICE: 0.295, GRANDFATHERED: 1.0} if grandfather_switch else {PRICE: 0.295}),
        CONVERSION: {NEW_SUBS: 0.09},
        RETENTION: {CHURN: 0.025},
        HOLD: {},
    }
    wire_by_id = {o["id"]: o for o in d["options"]}
    d["options"] = [
        {
            "id": oid,
            "label": wire_by_id.get(oid, {}).get("label", oid),
            "interventions": persisted[oid],
        }
        for oid in options
    ]
    d["n_samples"] = n_samples
    d["analysis_types"] = ["comparison"]
    d["include_e_values"] = False
    d["include_voi"] = False
    d["include_factor_flips"] = False
    # The persisted limit is "churn <= 4%" (0.04 in the node's frame); the wire carried 1.
    d["goal_constraints"][0]["value"] = 0.04
    return d


def strip_sources(d: Dict[str, Any]) -> Dict[str, Any]:
    """The same request with every observed_state.source removed: nothing is attested, so every node is
    evaluated in today's (pre-B1a) form. The in-process contrast for the byte-identity rows."""
    out = copy.deepcopy(d)
    for node in out["graph"]["nodes"]:
        if node.get("observed_state"):
            node["observed_state"].pop("source", None)
    return out


def analyse(d: Dict[str, Any]):
    return RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))


def result(response, option_id: str):
    matches = [r for r in response.results if r.option_id == option_id]
    assert len(matches) == 1, f"expected exactly one result for {option_id!r}"
    return matches[0]


def frame_of(response, node_id: str):
    frames = [f for f in (response.node_levels or []) if f.node_id == node_id]
    assert (
        len(frames) == 1
    ), f"expected exactly one node_levels entry for {node_id!r}, got {len(frames)}"
    return frames[0]


def samples(response, option_id: str) -> np.ndarray:
    return np.array(result(response, option_id).outcome_distribution.samples)


def draws(request: RobustnessRequestV2, n: int):
    """(edge strengths, factor values) pairs drawn exactly the way the analyzer draws them."""
    sampler = DualUncertaintySampler(request.graph.edges, SeededRNG(SEED))
    factors = FactorSampler(
        request.graph.nodes, request.parameter_uncertainties, SeededRNG(SEED + 1)
    )
    return [(sampler.sample_edge_configuration(), factors.sample_factor_values()) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------------
# R1 — the status quo reproduces the held levels: MRR £75,000, churn 3%, subscribers 1,500
# ---------------------------------------------------------------------------------------------------------


class TestR1StatusQuoReproducesTheHeldLevels:
    def test_every_status_quo_draw_is_the_held_level_exactly(self):
        """Per draw, at the evaluator every analysis reads through: no interventions, same draws."""
        request = RobustnessRequestV2.model_validate(paul_request(options=[HOLD]))
        evaluator = SCMEvaluatorV2(request.graph)
        targets = [GOAL, CHURN, SUBS, NEW_SUBS]
        for edge_config, factor_values in draws(request, 200):
            values = evaluator.evaluate_multi(
                edge_strengths=edge_config,
                interventions={},
                target_nodes=targets,
                factor_values=factor_values,
            )
            assert values == {GOAL: 0.6, CHURN: 0.03, SUBS: 0.15, NEW_SUBS: 0.075}

    def test_the_status_quo_mrr_band_is_75k_on_every_draw(self):
        response = analyse(paul_request(options=[HOLD, KEEP]))
        hold = samples(response, HOLD)
        assert np.all(
            hold == 0.6
        ), f"status-quo MRR draws off 0.6: min {hold.min()}, max {hold.max()}"
        assert float(np.median(hold)) * CAP_GBP == 75_000.0

    def test_keep_current_mrr_median_is_within_1e_6_of_the_held_level(self):
        """Keep-current pins price at 0.245 against a price draw of sd 0.0001, so it is the status quo
        up to that pin. Served n_samples."""
        response = analyse(paul_request(options=[KEEP, P59], n_samples=10_000))
        assert abs(float(np.median(samples(response, KEEP))) - 0.6) <= 1e-6

    def test_the_held_levels_are_disclosed_with_their_author(self):
        response = analyse(paul_request(options=[HOLD, KEEP]))
        goal = frame_of(response, GOAL)
        assert (goal.frame, goal.level, goal.level_anchor_source, goal.observed_source) == (
            "anchored_level",
            0.6,
            "user_stated",
            "brief_extraction",
        )
        for node_id, level in ((CHURN, 0.03), (SUBS, 0.15), (NEW_SUBS, 0.075)):
            f = frame_of(response, node_id)
            assert (f.frame, f.level, f.level_anchor_source, f.observed_source) == (
                "anchored_level",
                level,
                "olumi_estimate",
                "cee_inference",
            ), node_id

    def test_the_churn_limit_is_scored_at_its_level(self):
        """The 'churn <= 4%' limit resolves (no refusal) and the retention option is read at the
        level it sets, 2.5%, on every draw."""
        response = analyse(paul_request(options=[HOLD, RETENTION]))
        for option_id in (HOLD, RETENTION):
            analysis = result(response, option_id).constraint_analysis
            assert analysis is not None, f"constraint_analysis refused for {option_id}"
            (row,) = [c for c in analysis.constraints if c.node_id == CHURN]
            assert row.prob_satisfied == 1.0


# ---------------------------------------------------------------------------------------------------------
# R2 — retention above keep-current; conversion's effect is the anchored ~+£18
# ---------------------------------------------------------------------------------------------------------


class TestR2NonRootSettingsScoreByTheirChangeFromToday:
    """A GUARD row, not a RED-first one: on the persisted option levels it is already green at
    3717e36, because N6 (``_in_model_frame``) reads a setting on a non-root node against the
    same-draw status quo. The served sign flip (F-01, retention -£1,768 against keep-current)
    comes from the WIRE value: PLoT sent ``monthly_churn: 1`` (100%), not 0.025. Measured on the
    captured request: -£1,767.54 at 3717e36 and the same after this change (anchoring changes
    levels, never an input). These rows pin that anchoring keeps the correct effects."""

    def test_retention_mean_mrr_is_above_keep_current(self):
        response = analyse(paul_request(options=[KEEP, RETENTION, CONVERSION]))
        gap = float(np.mean(samples(response, RETENTION) - samples(response, KEEP))) * CAP_GBP
        assert gap > 0.0, f"retention scored below keep-current by £{-gap:.2f}"
        assert 5.0 < gap < 13.0, f"retention effect £{gap:.2f}, expected ~+£9"

    def test_conversion_effect_is_about_18_pounds_not_202(self):
        response = analyse(paul_request(options=[KEEP, RETENTION, CONVERSION]))
        gap = float(np.mean(samples(response, CONVERSION) - samples(response, KEEP))) * CAP_GBP
        assert 15.0 < gap < 21.0, f"conversion effect £{gap:.2f}, expected ~+£18"


# ---------------------------------------------------------------------------------------------------------
# R3 — root-lever options: win shares BYTE-IDENTICAL to today's form on the same seed
# ---------------------------------------------------------------------------------------------------------


class TestR3RootLeverOptionsAreUnchanged:
    OPTIONS = [KEEP, P59, P54, GRANDFATHER]

    def test_win_shares_are_byte_identical_to_the_unanchored_form(self):
        anchored = analyse(paul_request(options=self.OPTIONS))
        today = analyse(strip_sources(paul_request(options=self.OPTIONS)))
        assert {r.option_id: r.win_probability for r in anchored.results} == {
            r.option_id: r.win_probability for r in today.results
        }
        assert anchored.recommended_option_id == today.recommended_option_id

    def test_per_draw_differences_are_unchanged(self):
        anchored = analyse(paul_request(options=self.OPTIONS))
        today = analyse(strip_sources(paul_request(options=self.OPTIONS)))
        for option_id in (P59, P54, GRANDFATHER):
            a = samples(anchored, option_id) - samples(anchored, KEEP)
            t = samples(today, option_id) - samples(today, KEEP)
            assert np.max(np.abs(a - t)) <= 1e-12, option_id

    def test_only_the_level_moves(self):
        """CONTROL: the rows above are not passing because nothing changed. The levels did move."""
        anchored = analyse(paul_request(options=self.OPTIONS))
        today = analyse(strip_sources(paul_request(options=self.OPTIONS)))
        assert abs(float(np.median(samples(anchored, KEEP))) - 0.6) < 1e-4
        assert abs(float(np.median(samples(today, KEEP))) - 0.6) > 0.3


# ---------------------------------------------------------------------------------------------------------
# R4 — an unattested non-root keeps today's form and is flagged
# ---------------------------------------------------------------------------------------------------------


class TestR4UnattestedNonRootsKeepTodaysFormAndAreFlagged:
    def test_a_non_root_with_no_observed_level_is_flagged_no_level(self):
        response = analyse(paul_request(options=[HOLD, KEEP]))
        f = frame_of(response, PRICE_SENSITIVITY)
        assert (f.frame, f.no_level_reason, f.level, f.level_anchor_source) == (
            "no_level",
            "no_observed_level",
            None,
            None,
        )

    def test_a_non_root_with_no_observed_level_is_evaluated_in_the_raw_form(self):
        request = RobustnessRequestV2.model_validate(paul_request(options=[HOLD]))
        evaluator = SCMEvaluatorV2(request.graph)
        for edge_config, factor_values in draws(request, 50):
            values = evaluator.evaluate_multi(
                edge_strengths=edge_config,
                interventions={},
                target_nodes=[PRICE, PRICE_SENSITIVITY],
                factor_values=factor_values,
            )
            raw = 0.0 + 0.0 + values[PRICE] * edge_config[(PRICE, PRICE_SENSITIVITY)]
            assert values[PRICE_SENSITIVITY] == raw

    def test_a_source_less_level_is_not_anchored_and_keeps_the_raw_form(self):
        """AIQ: never anchor on a source-less value. Subscribers WITHOUT a source is flagged and computed
        as today: its (sampled) base plus its parents' raw contributions."""
        d = paul_request(options=[HOLD])
        (subs,) = [n for n in d["graph"]["nodes"] if n["id"] == SUBS]
        subs["observed_state"].pop("source")
        response = analyse(d)
        f = frame_of(response, SUBS)
        assert (f.frame, f.no_level_reason, f.observed_source) == (
            "no_level",
            "source_not_attested",
            None,
        )

        request = RobustnessRequestV2.model_validate(d)
        evaluator = SCMEvaluatorV2(request.graph)
        for edge_config, factor_values in draws(request, 50):
            values = evaluator.evaluate_multi(
                edge_strengths=edge_config,
                interventions={},
                target_nodes=[SUBS, CHURN, NEW_SUBS],
                factor_values=factor_values,
            )
            # today's arithmetic, in its own association order: base + intercept + parents
            parents = 0.0 + values[CHURN] * edge_config[(CHURN, SUBS)]
            parents += values[NEW_SUBS] * edge_config[(NEW_SUBS, SUBS)]
            assert values[SUBS] == factor_values[SUBS] + 0.0 + parents


# ---------------------------------------------------------------------------------------------------------
# R5 — a level pushed out of its domain is disclosed; win shares are untouched
# ---------------------------------------------------------------------------------------------------------


class TestR5OutOfDomainLevelsAreDisclosedNeverAbsorbed:
    OPTIONS = [KEEP, P59, GRANDFATHER, RETENTION]

    def test_grandfathering_pushes_churn_below_zero_and_the_share_says_so(self):
        response = analyse(paul_request(options=self.OPTIONS, grandfather_switch=True))
        churn = frame_of(response, CHURN)
        assert (churn.level_domain_min, churn.level_domain_max) == (0.0, None)
        shares = churn.level_out_of_domain_share
        assert shares is not None
        assert shares[GRANDFATHER] > 0.5, shares
        assert shares[KEEP] == 0.0
        assert shares[RETENTION] == 0.0

    def test_win_shares_are_computed_on_unclamped_draws(self):
        anchored = analyse(paul_request(options=self.OPTIONS, grandfather_switch=True))
        today = analyse(strip_sources(paul_request(options=self.OPTIONS, grandfather_switch=True)))
        assert {r.option_id: r.win_probability for r in anchored.results} == {
            r.option_id: r.win_probability for r in today.results
        }
        gap_a = samples(anchored, GRANDFATHER) - samples(anchored, KEEP)
        gap_t = samples(today, GRANDFATHER) - samples(today, KEEP)
        assert np.max(np.abs(gap_a - gap_t)) <= 1e-12

    def test_the_reported_limit_level_is_clamped_to_the_domain(self):
        """A limit reads the CLAMPED level (a churn below 0% counts as 0%), so every figure in the limit
        block is about levels the quantity can take; the share above says how often that happened.
        Against a floor 'churn >= 1%', grandfathering fails on the draws that push churn below zero,
        and it fails by exactly 1 point (0% vs 1%), not by the ~48 points of an impossible level."""
        d = paul_request(options=self.OPTIONS, grandfather_switch=True)
        d["goal_constraints"].append(
            {
                "constraint_id": "floor-churn",
                "node_id": CHURN,
                "operator": ">=",
                "value": 0.01,
                "label": "Churn floor",
                "value_frame": "level",
            }
        )
        response = analyse(d)
        analysis = result(response, GRANDFATHER).constraint_analysis
        assert analysis is not None
        (floor,) = [c for c in analysis.constraints if c.constraint_id == "floor-churn"]
        assert 0.0 < floor.prob_satisfied < 0.5
        assert floor.failure_margin_median == pytest.approx(0.01, abs=1e-12)


# ---------------------------------------------------------------------------------------------------------
# R6 — an engine-defaulted root, a source-less value or an unknown source is never an anchor
# ---------------------------------------------------------------------------------------------------------


class TestR6OnlyAttestedLevelsAnchor:
    def test_the_engine_defaulted_root_is_not_an_anchor(self):
        response = analyse(paul_request(options=[HOLD, KEEP]))
        assert GRANDFATHERED not in {f.node_id for f in response.node_levels or []}
        assert "ROOT_NODE_DEFAULT_VALUE" in {w.code for w in response.inference_warnings}
        assert all(
            f.frame == "no_level" or f.level_anchor_source is not None for f in response.node_levels
        )

    @pytest.mark.parametrize("source", [None, "computed", "engine_default", ""])
    def test_an_unattested_source_on_a_level_does_not_anchor(self, source):
        d = paul_request(options=[HOLD])
        (churn,) = [n for n in d["graph"]["nodes"] if n["id"] == CHURN]
        if source is None:
            churn["observed_state"].pop("source")
        else:
            churn["observed_state"]["source"] = source
        response = analyse(d)
        f = frame_of(response, CHURN)
        assert (f.frame, f.level_anchor_source, f.no_level_reason) == (
            "no_level",
            None,
            "source_not_attested",
        )
        request = RobustnessRequestV2.model_validate(d)
        evaluator = SCMEvaluatorV2(request.graph)
        for edge_config, factor_values in draws(request, 50):
            values = evaluator.evaluate_multi(
                edge_strengths=edge_config,
                interventions={},
                target_nodes=[CHURN, PRICE_SENSITIVITY, GRANDFATHERED],
                factor_values=factor_values,
            )
            parents = 0.0 + values[PRICE_SENSITIVITY] * edge_config[(PRICE_SENSITIVITY, CHURN)]
            parents += values[GRANDFATHERED] * edge_config[(GRANDFATHERED, CHURN)]
            assert values[CHURN] == factor_values[CHURN] + 0.0 + parents

    @pytest.mark.parametrize(
        "source, author",
        [
            ("brief_extraction", "user_stated"),
            ("explicit", "user_stated"),
            ("user_override", "user_stated"),
            ("user_confirmed", "user_ratified"),
            ("cee_inference", "olumi_estimate"),
            ("cee_repair", "olumi_estimate"),
            ("system_repaired", "olumi_estimate"),
        ],
    )
    def test_each_attested_author_anchors_and_is_named(self, source, author):
        d = paul_request(options=[HOLD])
        (churn,) = [n for n in d["graph"]["nodes"] if n["id"] == CHURN]
        churn["observed_state"]["source"] = source
        response = analyse(d)
        f = frame_of(response, CHURN)
        assert (f.frame, f.level_anchor_source, f.observed_source) == (
            "anchored_level",
            author,
            source,
        )


# ---------------------------------------------------------------------------------------------------------
# The wire: the served V2 envelope carries the frames, and the goal band is a level
# ---------------------------------------------------------------------------------------------------------


class TestTheV2EnvelopeCarriesTheLevelFrames:
    """The served route (the V2 endpoint rejects an option with no interventions, so no HOLD here)."""

    ENDPOINT = "/api/v1/robustness/analyze/v2"
    HEADERS = {"X-ISL-Response-Version": "2"}
    OPTIONS = [KEEP, P59, GRANDFATHER, RETENTION]

    @pytest.fixture(scope="class")
    def request_dict(self) -> Dict[str, Any]:
        return paul_request(options=self.OPTIONS, grandfather_switch=True)

    @pytest.fixture(scope="class")
    def body(self, request_dict) -> Dict[str, Any]:
        from fastapi.testclient import TestClient

        from src.api.main import app

        response = TestClient(app).post(self.ENDPOINT, json=request_dict, headers=self.HEADERS)
        assert response.status_code == 200, response.text
        return response.json()

    def test_node_levels_reach_the_wire(self, body):
        frames = {f["node_id"]: f for f in body.get("node_levels") or []}
        assert frames[GOAL]["frame"] == "anchored_level"
        assert frames[GOAL]["level_anchor_source"] == "user_stated"
        assert frames[CHURN]["level_anchor_source"] == "olumi_estimate"
        assert frames[PRICE_SENSITIVITY]["frame"] == "no_level"
        assert frames[CHURN]["level_out_of_domain_share"][GRANDFATHER] > 0.5
        assert GRANDFATHERED not in frames

    def test_the_keep_current_band_is_at_the_held_level(self, body):
        (keep,) = [o for o in body["options"] if o["id"] == KEEP]
        band = keep["outcome"]
        for key in ("p10", "p50", "p90"):
            assert abs(band[key] - 0.6) < 1e-3, (key, band[key])

    def test_the_goal_band_is_the_unclamped_band_clamped_to_the_goal_domain(
        self, body, request_dict
    ):
        """Grandfathering pushes MRR below £0 in ~13% of draws: the unclamped p10 is negative, the
        reported p10 is exactly 0, and every other percentile is untouched."""
        v1 = analyse(request_dict)
        shares = {f["node_id"]: f for f in body["node_levels"]}[GOAL]["level_out_of_domain_share"]
        for option_id in self.OPTIONS:
            raw = np.percentile(samples(v1, option_id), [10, 50, 90])
            (wire,) = [o for o in body["options"] if o["id"] == option_id]
            band = wire["outcome"]
            assert [band["p10"], band["p50"], band["p90"]] == [
                float(min(max(v, 0.0), 1.0)) for v in raw
            ], option_id
        raw_gf = np.percentile(samples(v1, GRANDFATHER), 10)
        assert raw_gf < 0.0, "precondition: the grandfathering band must leave the domain"
        assert shares[GRANDFATHER] > 0.1
        assert shares[KEEP] == 0.0
