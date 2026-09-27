"""B1a — anchored-delta LEVELS for attested non-root nodes (round 2).

Build train #70 5855068711 row B1; design MG 5855036638 + 5855037633; AIQ rulings 5855046894 (who anchors,
clamping) and 5856229075 (R4: structural outputs byte-identical, PERMANENTLY, for additive nodes). Scope
correction MG 5856099271 / DL 5856103285: B1a changes no sign and does not touch F-01 (the PLoT seam, A3).
Acceptance rows: AIQ ``aiq-p2-20260927/ACCEPTANCE-ROWS-R2R3-B5-20260927.md`` @364533c3, section 1 (B1a-1..7).

THE RULE. A NON-ROOT node whose ``observed_state`` holds a level ``o`` from an ATTESTED author, with no
epsilon noise reaching it, is ANCHORED. Every level ISL reports for it is recovered per draw from TODAY's
draws against the same draw's status quo (common random numbers)::

    level_i = o + (option_sample_i - status_quo_sample_i)

so the status quo reproduces ``o`` on every draw. REPORT-ONLY (round 2): the evaluator and the Monte Carlo
are today's, untouched, so win shares, regret, probability_of_goal, the limit block and every structural
analysis are byte-identical to a run in which nothing is anchored. Round 1 anchored inside the evaluator,
which zeroed edge/factor sensitivity and emptied fragile_edges on a6ed1bff (verifier blocker 1); that
design is gone.

THE FIXTURE is Paul's own persisted graph ``a6ed1bff`` as PLoT actually sent it to ISL, captured at
2026-09-27T10:27:59Z through a local proxy and matched leaf-for-leaf against the staging sha8 capture
(``tests/fixtures/anchored_delta/paul_a6ed1bff_plot_to_isl_request.json``, sha256 2ddc8367…1ca6). Three
documented edits, each because the WIRE disagrees with the persisted graph (the A3 seam, not this slice):

* ``6dbac00d`` sets ``monthly_new_pro_subscribers`` to 1 on the wire; the graph holds 0.09 (90/month).
* ``ca47b368`` sets ``monthly_churn`` to 1 on the wire; the graph holds 0.025 (2.5%).
* ``146aa89d`` (£59, grandfather existing customers) is absent from the wire; it is re-added with its
  persisted price lever (``pro_plan_price`` 0.295), plus, where named, its grandfathering switch.

THE CONTRAST ("anchoring off") is the SAME request with ``level_anchor_source`` patched to attest nothing,
so no node is anchored and every figure is today's. It differs from the pre-B1a engine only in the
node_levels disclosure (all 'no_level'). The V2-wire rows run in the offload pool, which a patch cannot
reach, so there the contrast strips every ``observed_state.source`` instead (same effect on anchoring).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from unittest import mock

import numpy as np
import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import RobustnessRequestV2
from src.services import robustness_analyzer_v2 as rav2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "anchored_delta"
    / "paul_a6ed1bff_plot_to_isl_request.json"
)
CAP_GBP = 125_000.0  # observed_state.cap of the goal node ``mrr`` (0.6 = £75,000)
N_SAMPLES = 2_000

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
HOLD = "hold_no_change"  # no interventions: the pure status quo (V1 analyzer only)
HOLD_NON_ROOTS = "hold_non_roots_at_today"  # sets two non-root nodes to exactly their held levels

# The served five plus 146aa89d at its persisted price lever (no switch).
SIX_OPTIONS = [KEEP, P59, P54, GRANDFATHER, CONVERSION, RETENTION]


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
        HOLD_NON_ROOTS: {CHURN: 0.03, NEW_SUBS: 0.075},
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


SERVED_ANALYSES: Dict[str, Any] = {
    "analysis_types": ["comparison", "sensitivity", "robustness"],
    "include_e_values": True,
    "include_voi": True,
    "include_factor_flips": True,
}


def served_request(*, options: List[str], n_samples: int = N_SAMPLES, **kwargs) -> Dict[str, Any]:
    """``paul_request`` with every analysis the served request asked for switched back on."""
    d = paul_request(options=options, n_samples=n_samples, **kwargs)
    d.update(copy.deepcopy(SERVED_ANALYSES))
    return d


def strip_sources(d: Dict[str, Any]) -> Dict[str, Any]:
    """The same request with every observed_state.source removed: nothing is attested, so nothing is
    anchored. The contrast for rows that run through the offload pool."""
    out = copy.deepcopy(d)
    for node in out["graph"]["nodes"]:
        if node.get("observed_state"):
            node["observed_state"].pop("source", None)
    return out


def analyse(d: Dict[str, Any]):
    return RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))


def analyse_unanchored(d: Dict[str, Any]):
    """The SAME request with anchoring off: no author attests any level, so no node is anchored."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rav2, "level_anchor_source", lambda node: None)
        return analyse(d)


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
    """The option's REPORTED goal samples (levels, when the goal is anchored)."""
    return np.array(result(response, option_id).outcome_distribution.samples)


def effect_gbp(response, option_id: str, reference: str = KEEP) -> float:
    return float(np.mean(samples(response, option_id) - samples(response, reference))) * CAP_GBP


def effect_and_se_gbp(response, option_id: str, reference: str = KEEP):
    """The option's mean per-draw effect against ``reference`` in £, and its Monte Carlo standard error."""
    diff = (samples(response, option_id) - samples(response, reference)) * CAP_GBP
    return float(np.mean(diff)), float(np.std(diff, ddof=1) / np.sqrt(len(diff)))


def codes(response) -> set:
    return {w.code for w in response.inference_warnings}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def everything_but_levels(response) -> Dict[str, Any]:
    """The whole V1 response except what B1a is FOR (the reported goal distribution and the
    node_levels disclosure) and the wall-clock timing."""
    d = response.model_dump(mode="json", by_alias=True)
    d.pop("node_levels", None)
    d["_metadata"].pop("execution_time_ms")
    for r in d["results"]:
        r.pop("outcome_distribution")
    return d


# ---------------------------------------------------------------------------------------------------------
# B1a-1 — the status quo reproduces the held levels: MRR £75,000, churn 3%, subscribers 1,500 (exact).
# Mutant: anchoring off -> the ~£15k status quo returns.
# ---------------------------------------------------------------------------------------------------------


class TestB1a1StatusQuoReproducesTheHeldLevels:
    @pytest.fixture(scope="class")
    def hold(self):
        return analyse(paul_request(options=[HOLD, KEEP]))

    def test_the_status_quo_mrr_is_75k_on_every_draw(self, hold):
        draws = samples(hold, HOLD)
        assert np.all(draws == 0.6), f"status-quo MRR draws off 0.6: {draws.min()}..{draws.max()}"
        assert float(np.median(draws)) * CAP_GBP == 75_000.0

    def test_the_held_levels_are_disclosed_with_their_author(self, hold):
        expected = {
            GOAL: (0.6, "user_stated", "brief_extraction"),
            CHURN: (0.03, "olumi_estimate", "cee_inference"),
            SUBS: (0.15, "olumi_estimate", "cee_inference"),
            NEW_SUBS: (0.075, "olumi_estimate", "cee_inference"),
        }
        for node_id, (level, author, source) in expected.items():
            f = frame_of(hold, node_id)
            assert (f.frame, f.level, f.level_anchor_source, f.observed_source) == (
                "anchored_level",
                level,
                author,
                source,
            ), node_id

    def test_the_status_quo_churn_is_exactly_3_percent_on_every_draw(self):
        """Pinned from both sides: 'churn <= 3%' holds on every status-quo draw and 'churn <= 2.99999%'
        on none, so the status-quo churn level is exactly 0.03 on every draw."""
        d = paul_request(options=[HOLD, KEEP])
        for cid, value in (("at-level", 0.03), ("just-below", 0.0299999)):
            d["goal_constraints"].append(
                {
                    "constraint_id": cid,
                    "node_id": CHURN,
                    "operator": "<=",
                    "value": value,
                    "label": cid,
                    "value_frame": "level",
                }
            )
        analysis = result(analyse(d), HOLD).constraint_analysis
        assert analysis is not None
        by_id = {c.constraint_id: c.prob_satisfied for c in analysis.constraints}
        assert (by_id["at-level"], by_id["just-below"]) == (1.0, 0.0)

    def test_keep_current_is_75k_where_it_was_15k(self):
        """Keep-current pins the price at its held £49 (PU sd 0.0001), so its band is the status quo up
        to that pin. Before B1a the same draws read ~£15k (the propagated sum), which the contrast shows.
        """
        d = paul_request(options=[KEEP, P59], n_samples=10_000)
        anchored, today = analyse(d), analyse_unanchored(d)
        assert abs(float(np.median(samples(anchored, KEEP))) - 0.6) <= 1e-6
        assert abs(float(np.median(samples(today, KEEP))) * CAP_GBP - 15_000.0) < 1_000.0

    def test_keep_current_reaches_the_wire_at_75k(self):
        """The served V2 route: keep-current's p50 is £75,000 to within £1.25, and its p10/p90 to within
        £25 (the price pin against a price PU of sd 0.0001, x strength ~0.5: measured 5.5e-5 at p10).
        """
        from fastapi.testclient import TestClient

        from src.api.main import app

        body = TestClient(app).post(
            "/api/v1/robustness/analyze/v2",
            json=paul_request(options=[KEEP, P59, RETENTION]),
            headers={"X-ISL-Response-Version": "2"},
        )
        assert body.status_code == 200, body.text
        (keep,) = [o for o in body.json()["options"] if o["id"] == KEEP]
        band = keep["outcome"]
        assert abs(band["p50"] - 0.6) <= 1e-5, band
        for key in ("p10", "p90"):
            assert abs(band[key] - 0.6) <= 2e-4, (key, band[key])


# ---------------------------------------------------------------------------------------------------------
# B1a-2 — CONTROL: root-lever options are unchanged (£59 / £54; analytic EV +£2,480.6 / +£1,240.3).
# ---------------------------------------------------------------------------------------------------------


class TestB1a2RootLeverOptionsAreUnchanged:
    """The Monte Carlo estimates of the two price effects on this seed are +£2,421.08 / +£1,210.54 at
    n=10,000 (3717e36 and this branch alike); the analytic expected values the acceptance row quotes are
    0.0198447 x £125,000 = +£2,480.6 and half that, +£1,240.3. What B1a must not move is the estimate.
    """

    @pytest.fixture(scope="class")
    def pair(self):
        d = paul_request(options=[KEEP, P59, P54])
        return analyse(d), analyse_unanchored(d)

    def test_win_share_regret_and_goal_probability_are_byte_identical(self, pair):
        anchored, today = pair
        for option_id in (KEEP, P59, P54):
            a, t = result(anchored, option_id), result(today, option_id)
            assert (a.win_probability, a.pre_noise_expected_regret, a.probability_of_goal) == (
                t.win_probability,
                t.pre_noise_expected_regret,
                t.probability_of_goal,
            ), option_id

    def test_the_price_effects_are_unchanged(self, pair):
        anchored, today = pair
        for option_id in (P59, P54):
            assert abs(effect_gbp(anchored, option_id) - effect_gbp(today, option_id)) <= 1e-6

    def test_only_the_level_moved(self, pair):
        """CONTROL for the control: the rows above do not pass because nothing changed."""
        anchored, today = pair
        assert abs(float(np.median(samples(anchored, KEEP))) - 0.6) < 1e-4
        assert abs(float(np.median(samples(today, KEEP))) - 0.6) > 0.3


# ---------------------------------------------------------------------------------------------------------
# B1a-3 — effects at the persisted levels: retention +£8.71, conversion +£17.56 (1e-6), retention >
# keep-current. Mutant: a clamp on the difference path -> an effect moves.
# ---------------------------------------------------------------------------------------------------------

# EXECUTED at ISL 3717e36 (the pre-B1a engine, this session) on the request below: the served five plus
# 146aa89d at its persisted price, persisted levels, n=10,000, the captured seed. MG EXEC 5856099271.
CONVERSION_EFFECT_AT_BASE_GBP = 17.556688738482833
RETENTION_EFFECT_AT_BASE_GBP = 8.707901425264417


class TestB1a3EffectsAtThePersistedLevels:
    @pytest.fixture(scope="class")
    def persisted(self):
        return analyse(paul_request(options=SIX_OPTIONS, n_samples=10_000))

    # AIQ re-statement (#70 5859788040): once the status quo and every option share today's level for a
    # ROOT lever (B1a-5), an effect is no longer byte-identical to the sampled-lever form. It must stay
    # within 3 Monte Carlo standard errors of the reference, with its sign unchanged.

    def test_conversion_is_plus_17_56_within_3_se(self, persisted):
        effect, se = effect_and_se_gbp(persisted, CONVERSION)
        assert abs(effect - CONVERSION_EFFECT_AT_BASE_GBP) <= 3 * se, (effect, se)
        assert effect > 0.0

    def test_retention_is_plus_8_71_within_3_se_and_above_keep_current(self, persisted):
        effect, se = effect_and_se_gbp(persisted, RETENTION)
        assert abs(effect - RETENTION_EFFECT_AT_BASE_GBP) <= 3 * se, (effect, se)
        assert effect > 0.0

    def test_an_effect_whose_levels_leave_the_domain_is_unclamped(self):
        """The discriminating arm. Grandfathering pushes MRR below £0 on some draws; its effect is the
        UNCLAMPED difference, exactly today's. A clamp on the reported draws moves it."""
        d = paul_request(options=[KEEP, P59, GRANDFATHER, RETENTION], grandfather_switch=True)
        anchored, today = analyse(d), analyse_unanchored(d)
        assert float(np.mean(samples(anchored, GRANDFATHER) < 0.0)) > 0.05, "precondition"
        assert abs(effect_gbp(anchored, GRANDFATHER) - effect_gbp(today, GRANDFATHER)) <= 1e-6


# ---------------------------------------------------------------------------------------------------------
# B1a-4 — structural outputs byte-identical (AIQ 5856229075, permanent for additive nodes). Verifier
# blocker 1: edge_sensitivity, factor_sensitivity, fragile_edges. Mutant: a structural consumer reads the
# anchored levels -> RED.
# ---------------------------------------------------------------------------------------------------------


class TestB1a4StructuralOutputsAreByteIdentical:
    """Every analysis the served request asks for, plus EVPC and the path decomposition, on the six
    options (146aa89d ties 59 on every draw, which exercises conditional_winners at this n)."""

    STRUCTURAL = [
        "sensitivity",
        "factor_sensitivity",
        "robustness",
        "conditional_winners",
        "edge_e_values",
        "factor_flip_values",
        "p_win_sensitivity",
        "factor_evppi",
        "factor_evpc",
        "path_decomposition",
        "stability_thresholds",
    ]

    @pytest.fixture(scope="class")
    def pair(self):
        """B1a-4 isolates ANCHORING: the held-root-lever rule (B1a-5) is made inert here, because it
        deliberately moves lever-uncertainty outputs (named in ``TestB1a5HeldRootLeverNamedMoves``),
        and this row must still prove anchoring itself moves nothing structural. Both arms run with the
        same rule, so the comparison is anchored vs unanchored and nothing else."""
        d = served_request(options=SIX_OPTIONS)
        d["include_path_decomposition"] = True
        d["control_candidates"] = [{"factor_id": PRICE, "values": [0.245, 0.27, 0.295]}]
        with mock.patch.object(rav2, "held_root_lever_levels", lambda request: {}):
            return analyse(d), analyse_unanchored(d)

    def test_everything_but_the_reported_levels_is_byte_identical(self, pair):
        anchored, today = pair
        a, t = everything_but_levels(anchored), everything_but_levels(today)
        assert a.keys() == t.keys()
        for key in a:
            assert _json(a[key]) == _json(t[key]), key

    @pytest.mark.parametrize("key", STRUCTURAL)
    def test_each_structural_output_is_exercised(self, pair, key):
        """Non-vacuity: None == None would pass the row above for free."""
        anchored, _ = pair
        assert getattr(anchored, key) not in (None, []), f"{key} is not exercised on this request"

    def test_the_verifier_three_are_live(self, pair):
        """Blocker 1's three outputs are the non-degenerate ones (round 1 read 0 / 0 / empty)."""
        anchored, _ = pair
        assert max(abs(s.elasticity) for s in anchored.sensitivity) > 0.5
        assert len(anchored.robustness.fragile_edges) >= 3
        assert len([f for f in anchored.factor_sensitivity if abs(f.elasticity) > 1e-3]) >= 3

    def test_the_levels_did_move(self, pair):
        """CONTROL: the byte-identity above is not because anchoring did nothing."""
        anchored, today = pair
        assert abs(float(np.median(samples(anchored, KEEP))) - 0.6) < 1e-4
        assert abs(float(np.median(samples(today, KEEP))) - 0.6) > 0.3


class TestB1a4OnTheV2Wire:
    """The same outputs as PLoT reads them: the served V2 envelope."""

    @pytest.fixture(scope="class")
    def pair(self):
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        d = served_request(options=[KEEP, P59, P54, CONVERSION, RETENTION])
        bodies = []
        for request_dict in (d, strip_sources(d)):
            response = client.post(
                "/api/v1/robustness/analyze/v2",
                json=request_dict,
                headers={"X-ISL-Response-Version": "2"},
            )
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
            ("factor_evppi",),
            ("p_win_sensitivity",),
        ],
    )
    def test_is_byte_identical(self, pair, path):
        anchored, today = pair
        a, t = anchored, today
        for key in path:
            a, t = a.get(key), t.get(key)
        assert a not in (None, []), f"{'.'.join(path)} absent on the wire"
        if path == ("factor_sensitivity",):
            # ``value_source`` echoes observed_state.source, which the contrast removes.
            a = [{k: v for k, v in row.items() if k != "value_source"} for row in a]
            t = [{k: v for k, v in row.items() if k != "value_source"} for row in t]
        assert _json(a) == _json(t)

    def test_win_shares_and_goal_probabilities_are_byte_identical(self, pair):
        anchored, today = pair
        for a, t in zip(anchored["options"], today["options"]):
            assert a["id"] == t["id"]
            assert (a.get("win_probability"), a.get("probability_of_goal")) == (
                t.get("win_probability"),
                t.get("probability_of_goal"),
            ), a["id"]


# ---------------------------------------------------------------------------------------------------------
# B1a-5 — carry-on P(goal) = the status quo exactly (0 effect). The NON-ROOT half only; see the report:
# CEE holds a status-quo option at its ROOT factors, which B1a (non-root anchoring) does not touch.
# Mutant: the held-vs-sampled artefact returns (a non-root setting written as given, not against the same
# draw's status quo).
# ---------------------------------------------------------------------------------------------------------


class TestB1a5HoldingNonRootsAtTodayIsTheStatusQuo:
    def test_goal_probability_and_band_equal_the_status_quo_exactly(self):
        """'MRR >= £75,000' is met on every status-quo draw (the level IS £75,000). An option that sets
        churn and new-subscriber intake to exactly their held levels changes nothing, so it meets it on
        every draw too, and its band is £75,000 on every draw. Both nodes carry a PU (sd 0.1), so a
        held-vs-sampled reading would put that spread into the option's 'effect'."""
        d = paul_request(options=[HOLD, HOLD_NON_ROOTS])
        d["goal_threshold"] = 0.6
        response = analyse(d)
        hold, carry_on = result(response, HOLD), result(response, HOLD_NON_ROOTS)
        assert hold.probability_of_goal == 1.0
        assert carry_on.probability_of_goal == hold.probability_of_goal
        assert np.all(samples(response, HOLD_NON_ROOTS) == 0.6)


# ---------------------------------------------------------------------------------------------------------
# B1a-6 — anchor sources: ANY held level anchors (incl. cee_inference) and is stamped; a source-less value
# or an engine-defaulted root never anchors; an unattested non-root is FLAGGED. Mutant: anchoring a
# source-less node -> RED.
# ---------------------------------------------------------------------------------------------------------


class TestB1a6OnlyAttestedLevelsAnchor:
    @pytest.mark.parametrize(
        "source, author",
        [
            ("brief_extraction", "user_stated"),
            ("explicit", "user_stated"),
            ("user_override", "user_stated"),
            ("user_assumption", "user_stated"),
            ("user_confirmed", "user_ratified"),
            ("panel_elicited", "user_ratified"),
            ("cee_inference", "olumi_estimate"),
            ("cee_repair", "olumi_estimate"),
            ("system_repaired", "olumi_estimate"),
        ],
    )
    def test_each_attested_author_anchors_and_is_named(self, source, author):
        d = paul_request(options=[HOLD])
        (goal,) = [n for n in d["graph"]["nodes"] if n["id"] == GOAL]
        goal["observed_state"]["source"] = source
        response = analyse(d)
        f = frame_of(response, GOAL)
        assert (f.frame, f.level_anchor_source, f.observed_source) == (
            "anchored_level",
            author,
            source,
        )
        assert np.all(samples(response, HOLD) == 0.6)

    @pytest.mark.parametrize("source", [None, "computed", "engine_default", ""])
    def test_an_unattested_level_is_flagged_and_never_anchors(self, source):
        """The goal holds 0.6 but nothing attests it: its band stays the propagated sum (~0.12), exactly
        today's, and the frame says why."""
        d = paul_request(options=[HOLD, KEEP])
        (goal,) = [n for n in d["graph"]["nodes"] if n["id"] == GOAL]
        if source is None:
            goal["observed_state"].pop("source")
        else:
            goal["observed_state"]["source"] = source
        response, today = analyse(d), analyse_unanchored(d)
        f = frame_of(response, GOAL)
        assert (f.frame, f.level_anchor_source, f.no_level_reason, f.level) == (
            "no_level",
            None,
            "source_not_attested",
            None,
        )
        assert samples(response, HOLD).tolist() == samples(today, HOLD).tolist()
        assert abs(float(np.median(samples(response, HOLD))) - 0.6) > 0.3

    def test_a_node_with_no_level_is_flagged_no_observed_level(self):
        f = frame_of(analyse(paul_request(options=[HOLD])), PRICE_SENSITIVITY)
        assert (f.frame, f.no_level_reason, f.level, f.level_anchor_source) == (
            "no_level",
            "no_observed_level",
            None,
            None,
        )

    def test_the_engine_defaulted_root_is_not_an_anchor(self):
        response = analyse(paul_request(options=[HOLD, KEEP]))
        assert GRANDFATHERED not in {f.node_id for f in response.node_levels or []}
        assert PRICE not in {f.node_id for f in response.node_levels or []}  # roots: never listed
        assert "ROOT_NODE_DEFAULT_VALUE" in codes(response)

    def test_epsilon_noise_reaching_a_held_level_refuses_the_anchor(self):
        """The status-quo reference is drawn without epsilon, so noise that reaches the node would be
        read as an effect (and its [0, 1] clamp as a change). Refused, with the level plan's own reason.
        """
        d = paul_request(options=[HOLD, KEEP])
        (churn,) = [n for n in d["graph"]["nodes"] if n["id"] == CHURN]
        churn["epsilon_std"] = 0.01
        response = analyse(d)
        for node_id in (CHURN, SUBS, GOAL):  # churn and everything downstream of it
            f = frame_of(response, node_id)
            assert (f.frame, f.no_level_reason) == (
                "no_level",
                "epsilon_breaks_status_quo_reference",
            ), node_id
        assert frame_of(response, NEW_SUBS).frame == "anchored_level"  # not downstream of churn


# ---------------------------------------------------------------------------------------------------------
# B1a-7 — probability_of_goal is NOT double-anchored: the level plan already anchors per draw
# (``baseline + (sample - status_quo_sample)``). Byte-identical. Mutant: double anchor -> RED.
# ---------------------------------------------------------------------------------------------------------


class TestB1a7GoalProbabilityIsNotDoubleAnchored:
    def test_is_byte_identical_where_it_is_not_trivial(self):
        """A threshold between the options' levels (MRR >= £77,000), so the probabilities are not all 0
        or 1 and a second anchoring (which would add ~£60k to every level) cannot hide."""
        d = paul_request(options=[KEEP, P59, P54, RETENTION])
        d["goal_threshold"] = 0.616
        anchored, today = analyse(d), analyse_unanchored(d)
        probabilities = {r.option_id: r.probability_of_goal for r in anchored.results}
        assert probabilities == {r.option_id: r.probability_of_goal for r in today.results}
        assert 0.0 < probabilities[P59] < 1.0, probabilities
        assert probabilities[KEEP] == 0.0


# ---------------------------------------------------------------------------------------------------------
# Verifier blocker 2 — three behaviours round 1 claimed without a row that fails on revert.
# ---------------------------------------------------------------------------------------------------------


class TestAnAnchoredGoalGetsNoFalseBaseDisclosure:
    """(2a). M6 (drop the anchored-goal branch of ``_build_goal_node_disclosures``) turns both RED."""

    def test_an_anchored_goal_with_a_pu_gets_neither_goal_base_disclosure(self):
        d = paul_request(options=[HOLD, KEEP])
        d["parameter_uncertainties"].append(
            {"node_id": GOAL, "distribution": "normal", "std": 0.05}
        )
        assert "GOAL_PU_BASE_ADDITIVE" in codes(analyse_unanchored(d))  # CONTRAST: true of today
        anchored = analyse(d)
        assert not {"GOAL_PU_BASE_ADDITIVE", "GOAL_OBSERVED_VALUE_UNUSED"} & codes(anchored)
        assert frame_of(anchored, GOAL).parameter_uncertainty_unused is True
        assert np.all(samples(anchored, HOLD) == 0.6)  # the PU draw cancels in the level

    def test_an_anchored_goal_with_no_threshold_gets_no_observed_value_unused(self):
        d = paul_request(options=[HOLD, KEEP])
        d.pop("goal_threshold", None)
        d.pop("goal_threshold_frame", None)
        assert "GOAL_OBSERVED_VALUE_UNUSED" in codes(analyse_unanchored(d))  # CONTRAST
        assert not {"GOAL_PU_BASE_ADDITIVE", "GOAL_OBSERVED_VALUE_UNUSED"} & codes(analyse(d))


class TestAnAnchoredLimitTargetGetsNoDefaultBase:
    """(2b). M7 (drop ``and not is_anchored_level_target``) turns the first row RED."""

    @staticmethod
    def _no_churn_pu(frame: str) -> Dict[str, Any]:
        d = paul_request(options=[HOLD, KEEP])
        d["parameter_uncertainties"] = [
            u for u in d["parameter_uncertainties"] if u["node_id"] != CHURN
        ]
        d["goal_constraints"][0]["value_frame"] = frame
        return d

    def test_an_anchored_level_limit_target_without_a_pu_gets_no_default_base(self):
        d = self._no_churn_pu("level")
        today = analyse_unanchored(d)
        assert "CONSTRAINT_NODE_DEFAULT_BASE" in codes(today)  # CONTRAST
        assert any(
            c.code.startswith("CONSTRAINT_NODE_DEFAULT_BASE")
            for c in today.critiques
            if CHURN in (c.affected_node_ids or [])
        )
        anchored = analyse(d)
        assert frame_of(anchored, CHURN).frame == "anchored_level"
        assert "CONSTRAINT_NODE_DEFAULT_BASE" not in codes(anchored)
        assert not any(
            c.code.startswith("CONSTRAINT_NODE_DEFAULT_BASE")
            for c in anchored.critiques
            if CHURN in (c.affected_node_ids or [])
        )

    def test_a_delta_limit_on_the_same_node_keeps_it(self):
        """A 'delta' limit compares the raw samples, where the 0.0 offset is real."""
        anchored = analyse(self._no_churn_pu("delta"))
        assert frame_of(anchored, CHURN).frame == "anchored_level"
        assert "CONSTRAINT_NODE_DEFAULT_BASE" in codes(anchored)


class TestALevelThresholdOutsideTheDomainGivesTheClampedProbability:
    """(2c). M8 (drop the goal-domain clip in the level-frame probability) turns this RED."""

    def test_a_threshold_below_the_floor_is_met_on_every_draw(self):
        """'MRR >= -£6,250' is below MRR's floor of £0: every LEVEL MRR can take meets it, so P = 1, even
        though grandfathering's unclamped levels fall below it."""
        d = paul_request(options=[KEEP, P59, GRANDFATHER], grandfather_switch=True)
        d["goal_threshold"] = -0.05
        response = analyse(d)
        assert float(np.mean(samples(response, GRANDFATHER) >= -0.05)) < 0.99, "precondition"
        assert result(response, GRANDFATHER).probability_of_goal == 1.0
        assert result(response, KEEP).probability_of_goal == 1.0


# ---------------------------------------------------------------------------------------------------------
# Reported levels leave their domain: disclosed and clamped where REPORTED, never where differenced
# ---------------------------------------------------------------------------------------------------------


class TestOutOfDomainLevelsAreDisclosedNeverAbsorbed:
    OPTIONS = [KEEP, P59, GRANDFATHER, RETENTION]

    @pytest.fixture(scope="class")
    def response(self):
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
        return analyse(d)

    def test_the_share_of_levels_outside_the_domain_is_disclosed(self, response):
        churn, goal = frame_of(response, CHURN), frame_of(response, GOAL)
        assert (churn.level_domain_min, churn.level_domain_max) == (0.0, None)
        assert (goal.level_domain_min, goal.level_domain_max) == (0.0, 1.0)
        assert churn.level_out_of_domain_share is not None
        assert churn.level_out_of_domain_share[GRANDFATHER] > 0.5
        assert churn.level_out_of_domain_share[KEEP] == 0.0
        assert churn.level_out_of_domain_share[RETENTION] == 0.0
        assert goal.level_out_of_domain_share is not None
        assert goal.level_out_of_domain_share[GRANDFATHER] > 0.05
        assert goal.level_out_of_domain_share[KEEP] == 0.0
        assert frame_of(response, SUBS).level_out_of_domain_share is None  # no level reported

    def test_a_limit_reads_the_clamped_level(self, response):
        """Against 'churn >= 1%', grandfathering fails on the draws that push churn below zero, by exactly
        1 point (0% vs 1%), not by the ~48 points of an impossible level."""
        analysis = result(response, GRANDFATHER).constraint_analysis
        assert analysis is not None
        (floor,) = [c for c in analysis.constraints if c.constraint_id == "floor-churn"]
        assert 0.0 < floor.prob_satisfied < 0.5
        assert floor.failure_margin_median == pytest.approx(0.01, abs=1e-12)


class TestTheV2EnvelopeCarriesTheLevelFrames:
    """The served route (it rejects an option with no interventions, so no HOLD here)."""

    OPTIONS = [KEEP, P59, GRANDFATHER, RETENTION]

    @pytest.fixture(scope="class")
    def request_dict(self) -> Dict[str, Any]:
        return paul_request(options=self.OPTIONS, grandfather_switch=True)

    @pytest.fixture(scope="class")
    def body(self, request_dict) -> Dict[str, Any]:
        from fastapi.testclient import TestClient

        from src.api.main import app

        response = TestClient(app).post(
            "/api/v1/robustness/analyze/v2",
            json=request_dict,
            headers={"X-ISL-Response-Version": "2"},
        )
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

    def test_the_goal_band_is_the_level_band_clamped_to_the_goal_domain(self, body, request_dict):
        """Grandfathering pushes MRR below £0 on some draws: the unclamped p10 is negative, the reported
        p10 is exactly 0, and every other percentile is untouched."""
        v1 = analyse(request_dict)
        for option_id in self.OPTIONS:
            raw = np.percentile(samples(v1, option_id), [10, 50, 90])
            (wire,) = [o for o in body["options"] if o["id"] == option_id]
            band = wire["outcome"]
            assert [band["p10"], band["p50"], band["p90"]] == [
                float(min(max(v, 0.0), 1.0)) for v in raw
            ], option_id
        assert np.percentile(samples(v1, GRANDFATHER), 10) < 0.0, "precondition"


# ---------------------------------------------------------------------------------------------------------
# The SERVED wire, unedited — B1a-1 / B1a-2 / B1a-3 / B1a-5 (root lever) / B1a-7 moves, per AIQ's
# re-statement #70 5859788040
# ---------------------------------------------------------------------------------------------------------

SERVED_WIRE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "anchored_delta"
    / "paul_a295e4a1_served_wire_plot_a6da42b.json"
)

# AIQ's SERVED reference (#70 5859210779, PLoT 3203991 / ISL 3717e36; reproduced in process to the penny).
SERVED_KEEP_CURRENT_GBP = 14_870.28
SERVED_EFFECTS_GBP = {P59: 2_410.43, P54: 1_205.22, CONVERSION: 17.82, RETENTION: 8.98}


def served_wire(**overrides: Any) -> Dict[str, Any]:
    """The EXACT ISL request PLoT staging ``a6da42b9`` builds for Paul's CEE->PLoT request.

    Produced in process (``POST /v2/run``, ISL stubbed at ``callAnalysisEndpoint``) from PLoT's
    ``tests/fixtures/paul-own-a295e4a1-20260927/cee-to-plot.request.json`` (sha256 ``8bc6f257…``, the
    request AIQ's served acceptance replays). Since PLoT #373 the wire carries conversion 0.09,
    retention 0.025 and the churn limit 0.04 itself, so nothing here is edited."""
    d = json.loads(SERVED_WIRE.read_text())
    d.update(overrides)
    return d


def without_the_lever_rule(fn, *args):
    with mock.patch.object(rav2, "held_root_lever_levels", lambda request: {}):
        return fn(*args)


class TestB1aOnTheServedWire:
    @pytest.fixture(scope="class")
    def served(self):
        return analyse(served_wire())

    def test_b1a_1_keep_current_sits_at_the_held_75k(self, served):
        assert float(np.median(samples(served, KEEP))) * CAP_GBP == pytest.approx(
            75_000.0, abs=1e-6
        )

    def test_b1a_1_control_todays_form_is_the_served_15k(self):
        """Anchoring off (and the B1a-5 lever rule off, as served today) is exactly today's served status
        quo — the RED this row turns GREEN."""
        today = without_the_lever_rule(analyse_unanchored, served_wire())
        assert float(np.mean(samples(today, KEEP))) * CAP_GBP == pytest.approx(
            SERVED_KEEP_CURRENT_GBP, abs=0.005
        )

    @pytest.mark.parametrize("option_id", [P59, P54, CONVERSION, RETENTION])
    def test_b1a_2_3_each_effect_is_within_3_se_of_the_served_effect(self, served, option_id):
        effect, se = effect_and_se_gbp(served, option_id)
        reference = SERVED_EFFECTS_GBP[option_id]
        assert abs(effect - reference) <= 3 * se, (option_id, effect, reference, se)
        assert np.sign(effect) == np.sign(reference)

    def test_b1a_3_retention_still_beats_keep_current(self, served):
        assert effect_and_se_gbp(served, RETENTION)[0] > 0.0

    def test_b1a_2_3_control_without_the_lever_rule_the_effects_are_the_served_pennies(self):
        """CONTROL: with the rule inert, anchoring is report-only and every effect is the served value
        to the penny — so any move above is the lever rule's, not anchoring's."""
        response = without_the_lever_rule(analyse, served_wire())
        for option_id, reference in SERVED_EFFECTS_GBP.items():
            assert effect_gbp(response, option_id) == pytest.approx(reference, abs=0.005), option_id


class TestB1a5CarryOnHoldsARootLeverAtToday:
    """AIQ B1a-5 (#70 5859788040): carry-on IS the status quo. Keep-current holds the ROOT lever price
    at today's £49 (0.245); the status quo used to SAMPLE it (std 1e-4), so keep-current's per-draw
    effect was ±ε around £0 and P(MRR >= £75k) read 0.4985 beside the status quo's 1.0. The principled
    fix, not a carry-on special case: every arm that does not set a root lever shares today's level.
    """

    def request(self) -> Dict[str, Any]:
        d = served_wire(goal_threshold=0.6)  # the goal at exactly today's level (£75,000)
        d["options"] = d["options"] + [{"id": HOLD, "label": "Status quo", "interventions": {}}]
        return d

    def test_precondition_price_is_a_root_lever_with_a_sampled_uncertainty(self):
        d = self.request()
        assert PRICE not in {e["to"] for e in d["graph"]["edges"]}
        assert any(u["node_id"] == PRICE for u in d["parameter_uncertainties"])
        assert rav2.held_root_lever_levels(RobustnessRequestV2.model_validate(d)) == {PRICE: 0.245}

    def test_carry_on_effect_is_exactly_zero_on_every_draw(self):
        response = analyse(self.request())
        assert np.array_equal(samples(response, KEEP), samples(response, HOLD))

    def test_carry_on_goal_probability_equals_the_status_quo(self):
        response = analyse(self.request())
        assert (
            result(response, KEEP).probability_of_goal == result(response, HOLD).probability_of_goal
        )

    def test_the_rows_above_are_red_without_the_rule(self):
        """CONTROL: the sampled-lever artefact is real on this wire (the RED this row fixes)."""
        response = without_the_lever_rule(analyse, self.request())
        assert not np.array_equal(samples(response, KEEP), samples(response, HOLD))
        assert (
            result(response, KEEP).probability_of_goal < result(response, HOLD).probability_of_goal
        )

    def test_a_root_lever_with_no_observed_level_is_not_held(self):
        """No "today's level" to hold it at: inventing 0.0 would be a fabrication."""
        d = self.request()
        for node in d["graph"]["nodes"]:
            if node["id"] == PRICE:
                node["observed_state"] = None
        assert PRICE not in rav2.held_root_lever_levels(RobustnessRequestV2.model_validate(d))

    def test_a_root_that_no_option_sets_keeps_its_sampled_uncertainty(self):
        """``pro_paying_subscribers`` is a root with a PU that no option sets: an uncertainty, not a
        decision. It is not held."""
        d = self.request()
        assert SUBS not in rav2.held_root_lever_levels(RobustnessRequestV2.model_validate(d))


class TestB1a5HeldRootLeverNamedMoves:
    """AIQ B1a-4 / B1a-7 re-stated: every output the lever rule moves is NAMED here, before -> after, on
    the served wire at the £75,000 threshold. An output that moves and is not in ``MOVED`` is RED.

    TWO causes, both named. (1) The rule itself: arms that do not set the price no longer carry its
    sampled spread. (2) A PRE-EXISTING coupling it exposes: tie-breaking draws from the EDGE sampler's
    RNG (``sampler.rng.choice(winners)``), and holding the lever creates exact ties (519 -> 530), so from
    the first new tie (draw 1,107 of 10,000) the edge stream is realised differently. That re-realisation
    moves absolute outcomes by Monte Carlo noise (keep-current's today-form mean £14,870.28 -> £15,047.70,
    1.7 SE of £104) — every effect is a within-run common-random-numbers difference, so effects stay
    valid (``TestB1aOnTheServedWire``: each within 3 SE of the served reference)."""

    MOVED = {
        "_metadata",
        "factor_evppi",
        "factor_sensitivity",
        "recommendation_confidence",
        "results",
        "robustness",
        "sensitivity",
    }
    VOLATILE = {"request_id", "timing", "latency_ms", "computed_at", "metadata"}

    # P(goal >= £75,000) per option: sampled lever -> held lever (measured, 2,000... n = the wire's 10,000).
    GOAL_PROBABILITY = {
        KEEP: (0.5333, 1.0),
        P59: (0.8575, 0.8609),
        P54: (0.8575, 0.8609),
        CONVERSION: (0.9721, 0.9735),
        RETENTION: (0.9707, 0.97),
    }

    @pytest.fixture(scope="class")
    def pair(self):
        d = served_wire(goal_threshold=0.6)
        after = analyse(d).model_dump(mode="json", by_alias=True)
        before = without_the_lever_rule(analyse, d).model_dump(mode="json", by_alias=True)
        return before, after

    def test_only_the_named_outputs_move(self, pair):
        before, after = pair
        moved = {
            k
            for k in set(before) | set(after)
            if k not in self.VOLATILE and _json(before.get(k)) != _json(after.get(k))
        }
        assert moved == self.MOVED

    @pytest.mark.parametrize("option_id", list(GOAL_PROBABILITY))
    def test_each_goal_probability_move_is_the_named_one(self, pair, option_id):
        before, after = pair
        was, now = self.GOAL_PROBABILITY[option_id]
        assert (
            next(r for r in before["results"] if r["option_id"] == option_id)["probability_of_goal"]
            == was
        )
        assert (
            next(r for r in after["results"] if r["option_id"] == option_id)["probability_of_goal"]
            == now
        )

    def test_the_tie_count_move_is_the_named_one(self, pair):
        before, after = pair
        assert (before["_metadata"]["tie_count"], after["_metadata"]["tie_count"]) == (519, 530)

    def test_the_recommendation_and_the_fragile_edge_set_do_not_move(self, pair):
        before, after = pair
        assert before["recommended_option_id"] == after["recommended_option_id"] == P59
        assert before["robustness"]["fragile_edges"] == after["robustness"]["fragile_edges"]
        assert before["robustness"]["is_robust"] == after["robustness"]["is_robust"] is True
