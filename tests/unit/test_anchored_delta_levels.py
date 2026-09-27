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

import numpy as np
import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import RobustnessRequestV2
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


def strip_non_root_sources(d: Dict[str, Any]) -> Dict[str, Any]:
    """Anchoring off through the wire: only NON-root sources removed. Anchoring applies to non-roots
    only, so this switches it off and nothing else; a root's attested level still decides B1a-5's
    "no change" on both sides (``strip_sources`` would switch B1a-5 off too)."""
    out = copy.deepcopy(d)
    targets = {edge["to"] for edge in out["graph"]["edges"]}
    for node in out["graph"]["nodes"]:
        if node["id"] in targets and node.get("observed_state"):
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


def without_the_no_change_rule(fn, *args):
    """B1a-5 switched off: every setting is held, even one equal to today's level (the engine before)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rav2, "is_todays_level", lambda value, level: False)
        return fn(*args)


class _OwnTieBreakStream:
    """The edge RNG, except that a tie-break draws from its OWN stream, so ties consume no edge draw."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._ties = np.random.default_rng(7)

    def choice(self, winners: List[str]) -> str:
        return str(self._ties.choice(winners))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def with_an_isolated_tie_break(fn, *args):
    original = rav2.DualUncertaintySampler.__init__

    def init(self, edges, rng, *a, **k):
        original(self, edges, rng, *a, **k)
        self.rng = _OwnTieBreakStream(self.rng)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rav2.DualUncertaintySampler, "__init__", init)
        return fn(*args)


def effect_se_gbp(response, option_id: str, reference: str = KEEP) -> float:
    """One Monte Carlo standard error of ``effect_gbp`` (per-draw differences, common random numbers)."""
    diff = samples(response, option_id) - samples(response, reference)
    return float(np.std(diff, ddof=1) / np.sqrt(len(diff))) * CAP_GBP


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
    """+£17.56 / +£8.71 exactly as measured before B1a-5 (the rule off). With it on, carry-on's new exact
    ties re-draw the edge stream (TestB1a5NamedMoves), so each effect is within 3 SE, sign unchanged.
    """

    @pytest.fixture(scope="class")
    def persisted(self):
        d = paul_request(options=SIX_OPTIONS, n_samples=10_000)
        return analyse(d), without_the_no_change_rule(analyse, d)

    def test_conversion_is_plus_17_56(self, persisted):
        _, before = persisted
        assert abs(effect_gbp(before, CONVERSION) - CONVERSION_EFFECT_AT_BASE_GBP) <= 1e-6

    def test_retention_is_plus_8_71_and_above_keep_current(self, persisted):
        _, before = persisted
        effect = effect_gbp(before, RETENTION)
        assert abs(effect - RETENTION_EFFECT_AT_BASE_GBP) <= 1e-6
        assert effect > 0.0

    @pytest.mark.parametrize(
        "option_id, at_base",
        [(CONVERSION, CONVERSION_EFFECT_AT_BASE_GBP), (RETENTION, RETENTION_EFFECT_AT_BASE_GBP)],
    )
    def test_with_b1a5_each_effect_is_within_3_se_sign_unchanged(
        self, persisted, option_id, at_base
    ):
        after, _ = persisted
        effect = effect_gbp(after, option_id)
        assert abs(effect - at_base) <= 3.0 * effect_se_gbp(after, option_id)
        assert np.sign(effect) == np.sign(at_base)

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
        d = served_request(options=SIX_OPTIONS)
        d["include_path_decomposition"] = True
        d["control_candidates"] = [{"factor_id": PRICE, "values": [0.245, 0.27, 0.295]}]
        # conditional_winners and fragile_edges are GATED outputs, so whether they appear at n=2000
        # depends on the realisation: at the captured seed they vanish once B1a-5's exact carry-on ties
        # re-draw the edge stream (TestB1a5NamedMoves), and at seed 3 they vanish with B1a-5 OFF. Seed 1
        # exercises every one of them with B1a-5 on AND off, so the non-vacuity rows below guard the
        # byte-identity row rather than one lucky draw.
        d["seed"] = 1
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
        for request_dict in (d, strip_non_root_sources(d)):
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
# The SERVED wire, unedited — B1a-1 / B1a-2 / B1a-3, and B1a-5 (AIQ equal-level rule, #184 5860095249)
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

# The first draw at which carry-on's new exact ties re-draw the edge stream (see
# TestB1a5NamedMoves). Every draw before it is untouched for every option.
FIRST_EXTRA_TIE_DRAW = 1107


def served_wire(**overrides: Any) -> Dict[str, Any]:
    """The EXACT ISL request PLoT staging ``a6da42b9`` builds for Paul's CEE->PLoT request.

    Produced in process (``POST /v2/run``, ISL stubbed at ``callAnalysisEndpoint``) from PLoT's
    ``tests/fixtures/paul-own-a295e4a1-20260927/cee-to-plot.request.json`` (sha256 ``8bc6f257…``, the
    request AIQ's served acceptance replays). Since PLoT #373 the wire carries conversion 0.09,
    retention 0.025 and the churn limit 0.04 itself, so nothing here is edited."""
    d = json.loads(SERVED_WIRE.read_text())
    d.update(overrides)
    return d


class TestB1aOnTheServedWire:
    """B1a measured on the body PLoT sends ISL today. ISL staging reproduces AIQ's SERVED baseline to the
    penny on this body (keep-current £14,870.28; effects +2,410.43 / +1,205.22 / +17.82 / +8.98), so an
    in-process run with B1a-5 off IS the served computation."""

    @pytest.fixture(scope="class")
    def pair(self):
        d = served_wire()
        return analyse(d), analyse_unanchored(d)

    @pytest.fixture(scope="class")
    def served_today(self):
        return without_the_no_change_rule(analyse_unanchored, served_wire())

    def test_b1a_1_keep_current_sits_at_the_held_75k(self, pair):
        anchored, _ = pair
        assert float(np.median(samples(anchored, KEEP))) * CAP_GBP == pytest.approx(
            75_000.0, abs=1e-6
        )

    def test_b1a_1_control_todays_form_is_the_served_15k(self, served_today):
        """Anchoring off and B1a-5 off is exactly today's served status quo — the RED this row turns GREEN."""
        assert float(np.mean(samples(served_today, KEEP))) * CAP_GBP == pytest.approx(
            SERVED_KEEP_CURRENT_GBP, abs=0.005
        )

    @pytest.mark.parametrize("option_id", [P59, P54, CONVERSION, RETENTION])
    def test_control_the_served_effects_reproduce_to_the_penny(self, served_today, option_id):
        assert effect_gbp(served_today, option_id) == pytest.approx(
            SERVED_EFFECTS_GBP[option_id], abs=0.005
        )

    @pytest.mark.parametrize("option_id", [P59, P54, CONVERSION, RETENTION])
    def test_b1a_2_anchoring_moves_no_effect(self, pair, option_id):
        anchored, today = pair
        assert abs(effect_gbp(anchored, option_id) - effect_gbp(today, option_id)) <= 1e-6

    @pytest.mark.parametrize("option_id", [P59, P54, CONVERSION, RETENTION])
    def test_b1a_3_every_effect_is_within_3_se_of_the_served_effect_sign_unchanged(
        self, pair, option_id
    ):
        """B1a-5 moves these only through the tie-break coupling named in TestB1a5NamedMoves."""
        anchored, _ = pair
        effect, served = effect_gbp(anchored, option_id), SERVED_EFFECTS_GBP[option_id]
        assert abs(effect - served) <= 3.0 * effect_se_gbp(anchored, option_id)
        assert np.sign(effect) == np.sign(served)

    def test_b1a_3_retention_still_beats_keep_current(self, pair):
        anchored, _ = pair
        assert effect_gbp(anchored, RETENTION) > 0.0


class TestB1a5CarryOnIsTheStatusQuo:
    """AIQ B1a-5, as ruled on ISL #184 (5860095249): the status quo samples every factor as its
    uncertainty says; an option that sets a factor to its level TODAY changes nothing and takes the status
    quo's own draws of it (``is_todays_level``, S-3: relative, defined once); any other value is held; a
    factor with no level for today cannot be "equal".

    On the served wire keep-current sets price to today's £49 (0.245) while the status quo samples it (PU
    std 1e-4, ±£0.02). Held, that ±ε put carry-on at a coin flip against a goal at exactly today's level.
    """

    def request(self, **keep: float) -> Dict[str, Any]:
        d = served_wire(goal_threshold=0.6)  # the goal at exactly today's level (£75,000)
        if keep:
            (option,) = [o for o in d["options"] if o["id"] == KEEP]
            option["interventions"] = dict(keep)
        d["options"] = d["options"] + [{"id": HOLD, "label": "Status quo", "interventions": {}}]
        return d

    @pytest.fixture(scope="class")
    def response(self):
        return analyse(self.request())

    def test_carry_on_effect_is_exactly_zero_on_every_draw(self, response):
        assert np.array_equal(samples(response, KEEP), samples(response, HOLD))

    def test_carry_on_goal_probability_equals_the_status_quo(self, response):
        assert result(response, HOLD).probability_of_goal == 1.0
        assert (
            result(response, KEEP).probability_of_goal == result(response, HOLD).probability_of_goal
        )

    def test_mutant_the_rule_off_is_the_named_defect(self):
        """B1a-5 off: carry-on's ±ε puts it at about a coin flip (0.5339) beside the status quo's 1.0."""
        response = without_the_no_change_rule(analyse, self.request())
        assert result(response, HOLD).probability_of_goal == 1.0
        assert 0.45 < result(response, KEEP).probability_of_goal < 0.6

    def test_a_setting_one_percent_away_is_held_not_no_change(self):
        """£49.50 is not £49: held at 0.2475, it is not the status quo on every draw. A tolerance wide
        enough to call it equal (S-3 widened to 2%) turns this RED."""
        response = analyse(self.request(**{PRICE: 0.2475}))
        assert not np.array_equal(samples(response, KEEP), samples(response, HOLD))
        assert effect_gbp(response, KEEP, reference=HOLD) > 0.0

    def test_a_factor_with_no_level_for_today_cannot_be_equal(self):
        """Grandfathering has no observed_state: setting it to 0.0 (where its sampled uncertainty is
        centred) is HELD at 0.0, never read as "no change". A missing level defaulted to 0.0 turns this RED.
        """
        d = self.request()
        d["parameter_uncertainties"] = d["parameter_uncertainties"] + [
            {"distribution": "normal", "node_id": GRANDFATHERED, "std": 0.1}
        ]
        (option,) = [o for o in d["options"] if o["id"] == KEEP]
        option["interventions"] = {GRANDFATHERED: 0.0}
        response = analyse(d)
        assert not np.array_equal(samples(response, KEEP), samples(response, HOLD))

    def test_an_unattested_level_is_never_today(self):
        """The same value with no author (price's source removed) is not an attested "today", so keep-current
        HOLDS it — B1a-6's mapping, the rule the uncertain-lever tests rely on (a source-less central value
        pinned against its own wide PU is a real action). Attestation dropped from the rule turns this RED.
        """
        d = self.request()
        (price,) = [n for n in d["graph"]["nodes"] if n["id"] == PRICE]
        price["observed_state"].pop("source")
        response = analyse(d)
        assert not np.array_equal(samples(response, KEEP), samples(response, HOLD))

    def test_s3_equality_is_relative_and_defined_once(self):
        tol = rav2.NO_CHANGE_RELATIVE_TOLERANCE
        assert 0.0 < tol < 0.01  # a 1% move (£49 -> £49.50) is always a change
        assert rav2.is_todays_level(0.245, 0.245)
        assert rav2.is_todays_level(
            0.245 * (1.0 + tol / 10.0), 0.245
        )  # float noise is not a change
        assert not rav2.is_todays_level(0.2475, 0.245)
        assert rav2.is_todays_level(0.0, 0.0)
        assert not rav2.is_todays_level(1e-12, 0.0)  # relative: no absolute floor at a zero level


class TestB1a5NamedMoves:
    """What B1a-5 moves beyond carry-on, and why (AIQ 5860141866: "no other row may move").

    Carry-on is now EXACTLY the status quo, so on a draw where another option's lever is inert it TIES
    with it. The tie-break draws from the EDGE RNG (``sampler.rng.choice``), so the first extra tie
    re-draws every later edge sample for every option. That coupling predates B1a-5. With the tie-break
    on its own stream, B1a-5 moves carry-on and NOTHING else."""

    @pytest.fixture(scope="class")
    def isolated(self):
        d = TestB1a5CarryOnIsTheStatusQuo().request()
        on = with_an_isolated_tie_break(analyse, d)
        off = with_an_isolated_tie_break(without_the_no_change_rule, analyse, d)
        return on, off

    @pytest.fixture(scope="class")
    def coupled(self):
        d = TestB1a5CarryOnIsTheStatusQuo().request()
        return analyse(d), without_the_no_change_rule(analyse, d)

    @pytest.mark.parametrize("option_id", [P59, P54, CONVERSION, RETENTION, HOLD])
    def test_isolated_tie_break_every_other_option_is_byte_identical(self, isolated, option_id):
        on, off = isolated
        assert np.array_equal(samples(on, option_id), samples(off, option_id))

    def test_isolated_tie_break_only_carry_on_moves(self, isolated):
        on, off = isolated
        assert not np.array_equal(samples(on, KEEP), samples(off, KEEP))

    @pytest.mark.parametrize("option_id", [P59, P54, CONVERSION, RETENTION])
    def test_coupled_tie_break_moves_others_only_after_the_first_extra_tie(
        self, coupled, option_id
    ):
        on, off = coupled
        a, b = samples(on, option_id), samples(off, option_id)
        assert np.array_equal(a[:FIRST_EXTRA_TIE_DRAW], b[:FIRST_EXTRA_TIE_DRAW])
        assert not np.array_equal(a, b)

    def test_coupled_the_status_quo_never_moves(self, coupled):
        on, off = coupled
        assert np.array_equal(samples(on, HOLD), samples(off, HOLD))

    def test_coupled_the_recommendation_is_unchanged(self, coupled):
        on, off = coupled
        assert on.recommended_option_id == off.recommended_option_id == P59


class TestS2AStrictLimitIsRefusedAtTheEngine:
    """DL gate (#70 5860001229) / AIQ S-2 (5859984425): once B1a makes today's level an exact atom, a
    strict limit read as non-strict is a wrong pass exactly at today's level. ISL's half of S-2: the
    comparator admits only ``>=`` / ``<=``, so a strict operator is REFUSED at validation — never
    coerced. (Whether CEE/PLoT coerce it before it reaches ISL is Canonical's measurement.)"""

    @pytest.mark.parametrize("operator", ["<", ">"])
    def test_a_strict_operator_is_refused_not_coerced(self, operator):
        d = served_wire()
        d["goal_constraints"][0]["operator"] = operator
        with pytest.raises(Exception) as excinfo:
            RobustnessRequestV2.model_validate(d)
        assert "operator" in str(excinfo.value)
