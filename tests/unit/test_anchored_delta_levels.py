"""B1a — ISL evaluates a NON-ROOT node that holds an attested level in ANCHORED-DELTA form.

Build train #70 5855068711 row B1; design MG 5855036638 + 5855037633; AIQ ruling 5855046894 (binding);
MG invariant correction 5855062769. SCOPE CORRECTION (MG 5856099271, DL 5856103285): B1a does not change
F-01 or any sign. F-01 lives at the PLoT seam (A3); N6 already read a setting on a non-root node against
the same draw's status quo. B1a's exits: the status quo reproduces exactly, bands are real levels, and a
limit on a derived node is scored against an anchored level. Nothing that is not a level moves (R7, R11).

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
# R2 — GUARD: B1a leaves option effects alone (retention above keep-current, conversion ~+£18)
# ---------------------------------------------------------------------------------------------------------


class TestR2NonRootSettingsScoreByTheirChangeFromToday:
    """A GUARD row, not a RED-first one: on the persisted option levels both effects are already
    right at 3717e36, because N6 (``_in_model_frame``) reads a setting on a non-root node against
    the same-draw status quo. B1a changes levels, never an effect or a sign, and does not touch
    F-01 (the PLoT seam, A3). These rows pin that anchoring keeps the effects as they were."""

    def test_retention_mean_mrr_is_above_keep_current(self):
        response = analyse(paul_request(options=[KEEP, RETENTION, CONVERSION]))
        gap = float(np.mean(samples(response, RETENTION) - samples(response, KEEP))) * CAP_GBP
        assert gap > 0.0, f"retention scored below keep-current by £{-gap:.2f}"
        assert 5.0 < gap < 13.0, f"retention effect £{gap:.2f}, expected ~+£9"

    def test_conversion_effect_is_about_18_pounds(self):
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


# =========================================================================================================
# ROUND 2 (verifier CHANGES_REQUIRED on 504ffc30; MG #70 5856099271, DL 5856103285)
#
# SCOPE. B1a does NOT change F-01 or any sign: N6 (``_in_model_frame``) already reads a setting on a
# non-root node against the same draw's status quo, and F-01 lives at the PLoT seam (A3). B1a's exits are:
# the status quo reproduces exactly (R1), bands are real levels (R5 + wire), and a limit on a derived node
# is scored against an anchored level (R1, R5, R9). Everything that is not a level stays as it was (R7).
# =========================================================================================================

SERVED_ANALYSES: Dict[str, Any] = {
    "analysis_types": ["comparison", "sensitivity", "robustness"],
    "include_e_values": True,
    "include_voi": True,
    "include_factor_flips": True,
}


# The options PLoT served on a6ed1bff (the wire carried five; 146aa89d was absent), at their persisted
# levels. The V2 route refuses two options with identical interventions, so 146aa89d stays out here too.
SERVED_OPTIONS = [KEEP, P59, P54, CONVERSION, RETENTION]


def served_request(*, options: List[str], n_samples: int = N_SAMPLES, **kwargs) -> Dict[str, Any]:
    """``paul_request`` with every analysis the served request asked for switched back on."""
    d = paul_request(options=options, n_samples=n_samples, **kwargs)
    d.update(copy.deepcopy(SERVED_ANALYSES))
    return d


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def codes(response) -> set:
    return {w.code for w in response.inference_warnings}


# ---------------------------------------------------------------------------------------------------------
# R7 — edge sensitivity, factor sensitivity and the fragile-edge gate are TODAY's, byte for byte
# ---------------------------------------------------------------------------------------------------------


class TestR7SensitivityAndFragileGatesAreTodaysForm:
    """Verifier blocker 1. Anchoring made these phases degenerate: each perturbs an edge or a factor on
    the reference option and reads the goal, and an anchored goal differenced against the SAME draw's
    status quo cancels the perturbation (edge sensitivity 0 everywhere, fragile_edges empty, factor
    sensitivity 0 except an artefact on the price lever). They are not levels, so they are computed in
    today's (propagated-sum) form, against today's baseline mean: byte-identical to the strip_sources run.
    """

    # The served five plus 146aa89d at its persisted price lever (no switch): it ties 59 on every draw,
    # which is what exercises conditional_winners on this graph. EVPC and the path decomposition are
    # switched on so that no output in the list below is vacuously equal (None == None).
    OPTIONS = [KEEP, P59, P54, GRANDFATHER, CONVERSION, RETENTION]

    @pytest.fixture(scope="class")
    def pair(self):
        d = served_request(options=self.OPTIONS)
        d["include_path_decomposition"] = True
        d["control_candidates"] = [{"factor_id": PRICE, "values": [0.245, 0.27, 0.295]}]
        anchored = analyse(d).model_dump(mode="json", by_alias=True)
        today = analyse(strip_sources(d)).model_dump(mode="json", by_alias=True)
        return anchored, today

    def test_edge_sensitivity_is_byte_identical_to_todays_form(self, pair):
        anchored, today = pair
        assert _json(anchored["sensitivity"]) == _json(today["sensitivity"])

    def test_factor_sensitivity_is_byte_identical_to_todays_form(self, pair):
        anchored, today = pair
        assert _json(anchored["factor_sensitivity"]) == _json(today["factor_sensitivity"])

    @pytest.mark.parametrize(
        "key", ["fragile_edges", "fragile_edges_enhanced", "robust_edges", "interpretation"]
    )
    def test_the_fragile_edge_gate_is_byte_identical_to_todays_form(self, pair, key):
        anchored, today = pair
        assert _json(anchored["robustness"][key]) == _json(today["robustness"][key])

    def test_the_rows_above_are_not_vacuous(self, pair):
        """CONTROL: today's phases are live on this graph, and the anchored run's levels did move."""
        anchored, today = pair
        assert max(abs(s["elasticity"]) for s in today["sensitivity"]) > 0.5
        assert len(today["robustness"]["fragile_edges"]) >= 3
        nonzero = [f for f in today["factor_sensitivity"] if abs(f["elasticity"]) > 1e-3]
        assert len(nonzero) >= 3
        keep_a, keep_t = (
            np.median(next(r for r in x["results"] if r["option_id"] == KEEP)["outcome_distribution"]["samples"])
            for x in (anchored, today)
        )
        assert abs(keep_a - 0.6) < 1e-4 and abs(keep_t - 0.6) > 0.3

    def test_no_factor_sensitivity_row_reads_a_parameter_uncertainty_as_unused(self, pair):
        """Deviation 2 of round 1 is gone from factor_sensitivity: no zero_outcome_diff artefacts from
        anchoring, and the price lever is an intervention override again, not -0.162."""
        anchored, _ = pair
        by_id = {f["node_id"]: f for f in anchored["factor_sensitivity"]}
        assert by_id[PRICE]["elasticity"] == 0.0
        assert by_id[PRICE]["zero_reason"] == "intervention_override"
        assert by_id[SUBS]["elasticity"] > 0.1

    # --- every OTHER robustness / sensitivity output: identical, or listed with its reason ---------

    @pytest.mark.parametrize(
        "key",
        [
            "edge_e_values",
            "factor_flip_values",
            "p_win_sensitivity",
            "conditional_winners",
            "path_decomposition",
            "stability_thresholds",
            "objective_ranking",
            "recommended_option_id",
            "recommendation_confidence",
            "critiques",
            "inference_warnings",
        ],
    )
    def test_every_other_decision_output_is_byte_identical(self, pair, key):
        anchored, today = pair
        assert anchored[key] not in (None, []), f"{key} is not exercised on this request"
        assert _json(anchored[key]) == _json(today[key])

    @pytest.mark.parametrize(
        "key", ["is_robust", "confidence", "recommendation_stability", "stability_penalty_factor"]
    )
    def test_the_rest_of_the_robustness_block_is_byte_identical(self, pair, key):
        anchored, today = pair
        assert _json(anchored["robustness"][key]) == _json(today["robustness"][key])

    @pytest.mark.parametrize("key", ["win_probability", "probability_of_goal", "constraint_analysis"])
    def test_per_option_decision_figures_are_byte_identical(self, pair, key):
        anchored, today = pair
        for a, t in zip(anchored["results"], today["results"]):
            assert a["option_id"] == t["option_id"]
            assert _json(a[key]) == _json(t[key]), (a["option_id"], key)

    def test_expected_regret_differs_only_in_rounding(self, pair):
        """NOT byte-identical: regret is E[max - U], computed on the level draws. In exact arithmetic the
        per-draw level offset is common to every option and cancels; in floating point the last bits move
        (measured: <= 1e-16)."""
        anchored, today = pair
        for a, t in zip(anchored["results"], today["results"]):
            assert abs(a["pre_noise_expected_regret"] - t["pre_noise_expected_regret"]) <= 1e-12

    def test_factor_evpc_moves_only_by_the_level(self, pair):
        """NOT byte-identical, for the same reason as EVPPI: ``baseline_max_expected_utility`` and
        ``best_do_expected_utility`` are max E[goal], LEVELS, and move by the same offset; the EVPC (their
        difference) and the best candidate value do not move."""
        anchored, today = pair
        (a,) = anchored["factor_evpc"]
        (t,) = today["factor_evpc"]
        for key in ("factor_id", "evpc", "best_candidate_value", "n_candidate_values", "method"):
            assert a[key] == t[key], key
        assert abs(a["evpc_raw"] - t["evpc_raw"]) <= 1e-9
        shift = a["baseline_max_expected_utility"] - t["baseline_max_expected_utility"]
        assert shift > 0.3
        assert abs((a["best_do_expected_utility"] - t["best_do_expected_utility"]) - shift) <= 2e-6

    def test_factor_evppi_moves_only_by_the_level(self, pair):
        """NOT byte-identical: ``baseline_max_expected_utility`` and ``conditional_max_expected_utility``
        are max E[goal], a LEVEL, so both move by the same level offset; the EVPPI itself (their
        difference) is the same, up to rounding (evppi_raw can land on -0.0 instead of 0.0, which flips
        clamped_low)."""
        anchored, today = pair
        by_a = {r["factor_id"]: r for r in anchored["factor_evppi"]}
        by_t = {r["factor_id"]: r for r in today["factor_evppi"]}
        assert by_a.keys() == by_t.keys() and by_a
        for factor_id, a in by_a.items():
            t = by_t[factor_id]
            assert (a["status"], a["evppi"]) == (t["status"], t["evppi"]), factor_id
            assert abs(a["evppi_raw"] - t["evppi_raw"]) <= 1e-9, factor_id
            shift = a["baseline_max_expected_utility"] - t["baseline_max_expected_utility"]
            assert shift > 0.3, factor_id  # the level offset (MRR ~0.62 vs ~0.14)
            assert (
                abs((a["conditional_max_expected_utility"] - t["conditional_max_expected_utility"]) - shift)
                <= 2e-6
            ), factor_id


class TestR7OnTheV2Wire:
    """The same three surfaces as the verifier read them: the served V2 envelope."""

    ENDPOINT = "/api/v1/robustness/analyze/v2"
    HEADERS = {"X-ISL-Response-Version": "2"}
    OPTIONS = SERVED_OPTIONS

    @pytest.fixture(scope="class")
    def pair(self):
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        d = served_request(options=self.OPTIONS)
        bodies = []
        for request_dict in (d, strip_sources(d)):
            response = client.post(self.ENDPOINT, json=request_dict, headers=self.HEADERS)
            assert response.status_code == 200, response.text
            bodies.append(response.json())
        return bodies

    @pytest.mark.parametrize(
        "path",
        [
            ("robustness", "edge_sensitivity"),
            ("robustness", "fragile_edges"),
            ("robustness", "fragile_edges_v1"),
            ("robustness", "robust_edges"),
            ("factor_sensitivity",),
        ],
    )
    def test_is_byte_identical_to_todays_form(self, pair, path):
        anchored, today = pair
        a, t = anchored, today
        for key in path:
            a, t = a.get(key), t.get(key)
        assert a not in (None, []), f"{'.'.join(path)} absent on the wire"
        if path == ("factor_sensitivity",):
            # ``value_source`` echoes observed_state.source, which the strip_sources contrast
            # removes; it is checked against the request below, and everything else here must match.
            a = [{k: v for k, v in row.items() if k != "value_source"} for row in a]
            t = [{k: v for k, v in row.items() if k != "value_source"} for row in t]
        assert _json(a) == _json(t)

    def test_factor_sensitivity_still_echoes_each_source(self, pair):
        anchored, _ = pair
        sources = {
            n["id"]: (n.get("observed_state") or {}).get("source")
            for n in served_request(options=self.OPTIONS)["graph"]["nodes"]
        }
        rows = anchored["factor_sensitivity"]
        assert rows and all(row["value_source"] == sources[row["node_id"]] for row in rows)


# ---------------------------------------------------------------------------------------------------------
# R8 / R9 / R10 — the three behaviours round 1 claimed without a row that fails on revert
# ---------------------------------------------------------------------------------------------------------


class TestR8AnAnchoredGoalGetsNoFalseBaseDisclosure:
    """M6 (drop the anchored-goal branch of ``_build_goal_node_disclosures``) turns both rows RED."""

    def test_an_anchored_goal_with_a_pu_gets_neither_goal_base_disclosure(self):
        d = paul_request(options=[HOLD, KEEP])
        d["parameter_uncertainties"].append({"node_id": GOAL, "distribution": "normal", "std": 0.05})
        today = codes(analyse(strip_sources(d)))
        assert "GOAL_PU_BASE_ADDITIVE" in today  # CONTRAST: today's form states it, truly
        anchored = analyse(d)
        assert not {"GOAL_PU_BASE_ADDITIVE", "GOAL_OBSERVED_VALUE_UNUSED"} & codes(anchored)
        assert frame_of(anchored, GOAL).parameter_uncertainty_unused is True

    def test_an_anchored_goal_with_no_pu_and_no_threshold_gets_no_observed_value_unused(self):
        d = paul_request(options=[HOLD, KEEP])
        d.pop("goal_threshold", None)
        d.pop("goal_threshold_frame", None)
        today = codes(analyse(strip_sources(d)))
        assert "GOAL_OBSERVED_VALUE_UNUSED" in today  # CONTRAST
        assert not {"GOAL_PU_BASE_ADDITIVE", "GOAL_OBSERVED_VALUE_UNUSED"} & codes(analyse(d))


class TestR9AnAnchoredLimitTargetGetsNoDefaultBase:
    """M7 (drop ``and not is_anchored`` from the constraint default-base check) turns this RED."""

    def test_an_anchored_limit_target_without_a_pu_gets_no_constraint_node_default_base(self):
        d = paul_request(options=[HOLD, KEEP])
        d["parameter_uncertainties"] = [
            u for u in d["parameter_uncertainties"] if u["node_id"] != CHURN
        ]
        today = analyse(strip_sources(d))
        assert "CONSTRAINT_NODE_DEFAULT_BASE" in codes(today)  # CONTRAST
        anchored = analyse(d)
        assert "CONSTRAINT_NODE_DEFAULT_BASE" not in codes(anchored)
        assert frame_of(anchored, CHURN).frame == "anchored_level"
        today_critiques = {c.code for c in today.critiques if CHURN in (c.affected_node_ids or [])}
        anchored_critiques = {
            c.code for c in anchored.critiques if CHURN in (c.affected_node_ids or [])
        }
        assert any(code.startswith("CONSTRAINT_NODE_DEFAULT_BASE") for code in today_critiques)
        assert not any(code.startswith("CONSTRAINT_NODE_DEFAULT_BASE") for code in anchored_critiques)


class TestR10ALevelThresholdOutsideTheDomainGivesTheClampedProbability:
    """M8 (drop the goal-domain clip in the level-frame probability) turns this RED."""

    def test_a_threshold_below_the_floor_is_met_on_every_draw(self):
        """'MRR >= -£6,250' is below MRR's floor of £0: every LEVEL MRR can take meets it, so P = 1, even
        though grandfathering's unclamped draws fall below it."""
        d = paul_request(options=[KEEP, P59, GRANDFATHER], grandfather_switch=True)
        d["goal_threshold"] = -0.05
        response = analyse(d)
        raw = samples(response, GRANDFATHER)
        assert float(np.mean(raw >= -0.05)) < 0.99, "precondition: unclamped draws below threshold"
        assert result(response, GRANDFATHER).probability_of_goal == 1.0
        assert result(response, KEEP).probability_of_goal == 1.0


# ---------------------------------------------------------------------------------------------------------
# R11 — a do() on a NO-LEVEL child of an anchored node has today's effect (verifier: support_load)
# ---------------------------------------------------------------------------------------------------------

SUPPORT = "support_load"
PIN_SUPPORT = "pin_support_load"


def support_load_request(*, n_samples: int = N_SAMPLES) -> Dict[str, Any]:
    """Paul's graph plus a NO-level node under anchored churn that feeds MRR, and an option pinning it."""
    d = paul_request(options=[KEEP, P59, RETENTION], n_samples=n_samples)
    d["graph"]["nodes"].append(
        {"id": SUPPORT, "kind": "factor", "label": "Support load", "intercept": 0, "epsilon_std": 0}
    )
    d["graph"]["edges"] += [
        {"from": CHURN, "to": SUPPORT, "exists_probability": 0.8, "strength": {"mean": 0.5, "std": 0.125}},
        {"from": SUPPORT, "to": GOAL, "exists_probability": 0.8, "strength": {"mean": -0.3, "std": 0.1}},
    ]
    d["options"].append({"id": PIN_SUPPORT, "label": "Pin support load", "interventions": {SUPPORT: 0.01}})
    return d


class TestR11APinnedNoLevelChildOfAnAnchoredNodeKeepsTodaysEffect:
    """A no-level node's samples are a propagated sum: a do(x) on it is written in THAT frame, today's.
    Under anchoring its parents are levels, so its status quo moved; x is translated by exactly that
    move, so the option's effect (and so every win share and tie) is today's."""

    @pytest.fixture(scope="class")
    def pair(self):
        d = support_load_request()
        return analyse(d), analyse(strip_sources(d))

    def test_the_per_draw_effect_is_todays(self, pair):
        anchored, today = pair
        for option_id in (PIN_SUPPORT, P59, RETENTION):
            a = samples(anchored, option_id) - samples(anchored, KEEP)
            t = samples(today, option_id) - samples(today, KEEP)
            assert np.max(np.abs(a - t)) <= 1e-12, option_id

    def test_win_shares_are_byte_identical(self, pair):
        anchored, today = pair
        assert {r.option_id: r.win_probability for r in anchored.results} == {
            r.option_id: r.win_probability for r in today.results
        }

    def test_the_node_is_flagged_no_level_and_the_levels_did_move(self, pair):
        anchored, today = pair
        f = frame_of(anchored, SUPPORT)
        assert (f.frame, f.no_level_reason) == ("no_level", "no_observed_level")
        assert abs(float(np.median(samples(anchored, KEEP))) - float(np.median(samples(today, KEEP)))) > 0.3


# ---------------------------------------------------------------------------------------------------------
# R12 — an unanchored parent with epsilon noise: the anchored child's status quo stays on its level
# ---------------------------------------------------------------------------------------------------------


def epsilon_graph_request() -> Dict[str, Any]:
    """root (0.8) -> noisy no-level node (intercept 0.8, so raw 1.6; epsilon 0.05) -> anchored goal (0.3).
    (An edge strength mean is capped at 1, hence the intercept.)"""
    return {
        "request_id": "b1a-r12-epsilon",
        "graph": {
            "nodes": [
                {"id": "root", "kind": "factor", "label": "Root",
                 "observed_state": {"value": 0.8, "source": "brief_extraction"},
                 "intercept": 0, "epsilon_std": 0},
                {"id": "noisy", "kind": "factor", "label": "Noisy", "intercept": 0.8, "epsilon_std": 0.05},
                {"id": "goal", "kind": "goal", "label": "Goal",
                 "observed_state": {"value": 0.3, "baseline": 0.3, "source": "brief_extraction"},
                 "intercept": 0, "epsilon_std": 0},
            ],
            "edges": [
                {"from": "root", "to": "noisy", "exists_probability": 1.0, "strength": {"mean": 1.0, "std": 0.01}},
                {"from": "noisy", "to": "goal", "exists_probability": 1.0, "strength": {"mean": 0.5, "std": 0.01}},
            ],
        },
        "options": [
            {"id": "hold", "label": "Hold", "interventions": {}},
            {"id": "lift", "label": "Lift root", "interventions": {"root": 0.9}},
        ],
        "goal_node_id": "goal",
        "n_samples": 500,
        "analysis_types": ["comparison"],
        "seed": 7,
    }


class TestR12AnAnchoredChildOfANoisyParentStaysOnItsLevel:
    """Verifier M11. Today's form clamps a noisy node to [0, 1] after its noise; the status-quo reference
    is noise-free. An anchored child differenced against an UNclamped reference drifted by
    strength * (clamp(raw + e) - raw): here 0.5 * (1.0 - 1.6) = -0.3, i.e. the status quo read 0.0, not
    0.3. The reference now carries the same clamp (without the noise) for the anchoring difference only."""

    def test_the_status_quo_reproduces_the_level_on_every_draw(self):
        from src.utils.rng import SeededRNG as _RNG

        request = RobustnessRequestV2.model_validate(epsilon_graph_request())
        evaluator = SCMEvaluatorV2(request.graph, epsilon_rng=_RNG(11))
        for _ in range(200):
            value = evaluator.evaluate(
                edge_strengths={("root", "noisy"): 1.0, ("noisy", "goal"): 0.5},
                interventions={},
                goal_node="goal",
            )
            assert value == 0.3

    def test_the_served_status_quo_band_is_the_level(self):
        response = analyse(epsilon_graph_request())
        hold = samples(response, "hold")
        assert np.all(hold == 0.3), f"status quo off its level: min {hold.min()}, max {hold.max()}"
