"""A product identity with a ZERO operand and NO stated level is evaluated, not withheld (AIQ #72 5881596876, hop 2).

The cloud journey declares savings = gcp_workload_share x gcp_unit_cost_saving x aws_workload_spend, with the share 0
today and no stated savings level. Rule 3 (``identity_zero_level``) withheld it because "its ratio and relative check
are undefined there". But only a STATED level needs a ratio: ``k = (o - A - L) / term`` and ``|o - recon| / |o|``.
With no stated level the evaluator reads ``T = term + A + L`` on every draw, with no k and no division, so a zero
operand is an ordinary level. Withholding it walked guessed 0.5 slopes instead (about -£700/month for full migration
where the product gives -£9,900).

Here Paul's served wire stands in for the shape: MRR = price x subscribers (+ other MRR growth), price 0 today, MRR's
level unstated. A stated level with a zero operand stays withheld (the control rows).
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

import src.services.robustness_analyzer_v2 as rav2

from src.services.robustness_analyzer_v2 import identity_partials, resolve_identity_plans
from tests.unit.test_r3_identity_evaluation import (
    FRAMES,
    MRR,
    PRICE,
    SUBS,
    graph_of,
    query_gbp,
    roots_at_today,
    wire,
)


def zero_price(*, stated: bool) -> Dict[str, Any]:
    d = wire()
    nodes = {n["id"]: n for n in d["graph"]["nodes"]}
    nodes[PRICE]["observed_state"][
        "value"
    ] = 0.0  # nothing charged today, as the cloud share is 0 today
    if not stated:
        nodes[MRR]["observed_state"] = None  # no level today; its frame still rides execution_frame
        d["goal_threshold"] = None
        d["goal_threshold_frame"] = None
        d["goal_constraints"] = []
    return d


class TestNoStatedLevelIsEvaluated:
    def test_the_plan_is_evaluated(self):
        (plan,) = resolve_identity_plans(graph_of(zero_price(stated=False))).values()
        assert plan.withheld_reason is None
        assert plan.evaluated and plan.target_level is None
        assert plan.levels[PRICE] == 0.0

    def test_the_option_reads_the_product_exactly(self):
        # Today: 0 x 1,500 + £1,000. Price to £59 with subscribers held: the product term goes 0 -> 59 x 1,500.
        assert query_gbp(zero_price(stated=False), {PRICE: 0.295, SUBS: 0.15}) == pytest.approx(
            88_500.0, abs=1e-6
        )


class TestAStatedLevelKeepsTheWithhold:
    """The ratio IS needed with a stated level: k = (o - A - L) / term divides by the zero term."""

    def test_a_zero_operand_beside_a_stated_level(self):
        (plan,) = resolve_identity_plans(graph_of(zero_price(stated=True))).values()
        assert plan.withheld_reason == rav2.IDENTITY_ZERO_LEVEL

    def test_a_stated_zero_target(self):
        d = wire()
        (mrr,) = [n for n in d["graph"]["nodes"] if n["id"] == MRR]
        mrr["observed_state"].update(value=0.0, baseline=0.0, raw_value=0)
        (plan,) = resolve_identity_plans(graph_of(d)).values()
        assert plan.withheld_reason == rav2.IDENTITY_ZERO_LEVEL


class TestInfluenceStaysAtTodaysCentre:
    """The ruled influence (expected NET effect at today's centre, AIQ 5875853496) is unchanged. With price 0 today,
    moving subscribers moves MRR's product term by nothing (0 x delta): its partial through MRR is exactly 0. Price's
    is subscribers today x the frames. The option comparison above reads the option's own level; influence does not.
    """

    def test_the_partials_at_a_zero_centre(self):
        d = zero_price(stated=False)
        partials = identity_partials(graph_of(d), roots_at_today(d))
        assert partials[(SUBS, MRR)] == 0.0
        assert partials[(PRICE, MRR)] == pytest.approx(
            0.15 * FRAMES[SUBS]["frame"] * FRAMES[PRICE]["frame"] / FRAMES[MRR]["frame"]
        )


CHURN, NEW, OTHER, GRANDFATHERED = (
    "monthly_churn",
    "monthly_new_pro_subscribers",
    "other_mrr_growth",
    "fac_existing_customers_grandfathered",
)
GATED = {
    SUBS,
    CHURN,
    NEW,
}  # every path to MRR runs through subscribers -> MRR, whose other operand (price) is 0


@pytest.fixture(scope="module")
def body() -> Dict[str, Any]:
    from tests.unit.test_r3_identity_evaluation import v2_body

    return v2_body({**zero_price(stated=False), "n_samples": 400})


def by_node(rows, key):
    return {r["node_id"]: r.get(key) for r in rows}


class TestAGatedFactorIsWithheldNotZero:
    """AIQ #72 5881683705 (1b): at today's centre price 0 multiplies subscribers' partial to exactly 0. That 0 is a
    GATE at a point the decision exists to leave, not "on balance", so a factor whose EVERY path to the goal runs
    through a product with another operand at 0 today is WITHHELD (None, with a reason), never 0 and never ranked.
    Every other factor keeps the ruled value."""

    def test_gated_factors_are_withheld(self, body):
        scores = by_node(body["structural_influence"], "influence_score")
        ranks = by_node(body["structural_influence"], "influence_rank")
        for node_id in GATED:
            assert scores[node_id] is None, node_id
            assert ranks[node_id] is None, node_id

    def test_every_other_factor_keeps_its_ruled_value(self, body):
        scores = by_node(body["structural_influence"], "influence_score")
        # Price's own operand edge is not gated (subscribers 0.15 today); grandfathered has a direct belief edge
        # beside its gated path, so NOT every path is gated.
        for node_id in (PRICE, OTHER, GRANDFATHERED):
            assert isinstance(scores[node_id], float) and scores[node_id] > 0.0, node_id
        ranked = sorted(
            (r for r in body["structural_influence"] if r.get("influence_rank") is not None),
            key=lambda r: r["influence_rank"],
        )
        assert [r["influence_rank"] for r in ranked] == [1, 2, 3]
        assert ranked[0]["influence_score"] == 1.0

    def test_the_row_carries_what_gates_it(self, body):
        """The typed carrier (AIQ 5881953818 / R3 SCIENCE 5881691323): PLoT and the UI read ``gated_by``, never a
        critique code. Only a withheld row has it."""
        rows = {r["node_id"]: r for r in body["structural_influence"]}
        for node_id in GATED:
            assert rows[node_id]["gated_by"] == [PRICE], node_id
        for node_id in (PRICE, OTHER, GRANDFATHERED):
            assert "gated_by" not in rows[node_id], node_id

    def test_the_withhold_says_why(self, body):
        (critique,) = [c for c in body["critiques"] if c["code"] == "STRUCTURAL_INFLUENCE_GATED"]
        assert "depends on the option chosen" in critique["message"]
        assert sorted(critique.get("affected_node_ids") or []) == sorted(GATED)

    def test_the_cohort_rows_never_read_zero_for_a_gated_factor(self, body):
        rows = {r["node_id"]: r for r in body.get("factor_sensitivity") or []}
        for node_id in GATED & set(rows):
            assert rows[node_id].get("influence_score") is None, node_id


class TestNoGateNoChange:
    def test_a_stated_level_graph_has_no_gated_critique(self):
        from tests.unit.test_r3_identity_evaluation import v2_body

        body = v2_body({**wire(), "n_samples": 400})
        assert not [c for c in body["critiques"] if c["code"] == "STRUCTURAL_INFLUENCE_GATED"]
        assert all(r["influence_score"] is not None for r in body["structural_influence"])


class TestTheCarrierSurvivesTruncationAndNamesOnlyGoalPaths:
    """PR Review #213 (5882196850): (1) a truncated walk withholds every score, but a gated row keeps its typed
    ``gated_by`` (and the gated reason sits beside the truncation reason), so PLoT never ranks it by the walk;
    (2) ``gated_by`` names only zero inputs on a factor-to-goal path, not a dead-end product's."""

    def test_a_truncated_walk_keeps_the_gate(self, monkeypatch):
        from tests.unit.test_r3_identity_evaluation import v2_body

        monkeypatch.setattr(rav2, "MAX_INFLUENCE_WALK_CALLS_TOTAL", 1)
        body = v2_body({**zero_price(stated=False), "n_samples": 200})
        rows = {r["node_id"]: r for r in body["structural_influence"]}
        assert all(r.get("influence_score") is None for r in rows.values())
        for node_id in GATED:
            assert rows[node_id]["gated_by"] == [PRICE], node_id
        for node_id in (PRICE, OTHER, GRANDFATHERED):
            assert "gated_by" not in rows[node_id], node_id
        codes = {c["code"] for c in body["critiques"]}
        assert {"STRUCTURAL_INFLUENCE_TRUNCATED", "STRUCTURAL_INFLUENCE_GATED"} <= codes

    def test_a_dead_end_products_zero_input_is_not_named(self):
        """Subscribers also feed a dead-end product (side = subscribers x z1, z1 0 today) that never reaches MRR.
        Subscribers is gated by price on its only goal path; z1 gates nothing that reaches the goal.
        """
        from tests.unit.test_r3_identity_evaluation import v2_body

        d = zero_price(stated=False)
        d["graph"]["nodes"] += [
            {
                "id": "z1",
                "kind": "factor",
                "label": "Z1",
                "observed_state": {"value": 0.0},
                "execution_frame": {"frame": 100.0, "carrier": "cap"},
            },
            {
                "id": "side",
                "kind": "factor",
                "label": "Side product",
                "execution_frame": {"frame": 1_000_000.0, "carrier": "cap"},
                "nonlinear_identity": {
                    "operation": "product",
                    "factor_ids": [SUBS, "z1"],
                    "stated_in_brief": False,
                },
            },
        ]
        d["graph"]["edges"] += [
            {"from": SUBS, "to": "side", "strength": {"mean": 1.0, "std": 0.01}},
            {"from": "z1", "to": "side", "strength": {"mean": 1.0, "std": 0.01}},
        ]
        body = v2_body({**d, "n_samples": 200})
        rows = {r["node_id"]: r for r in body["structural_influence"]}
        assert rows[SUBS]["gated_by"] == [PRICE]  # not [PRICE, "z1"]


class TestTheOatRowDoesNotSayNoEffectForAGatedFactor:
    """AIQ #72 5882619314 (2): the one-at-a-time sensitivity reads at today's centre, where price 0 multiplies a
    gated factor's effect to exactly 0 and the row said ``zero_outcome_diff`` ("no effect") for a factor that
    matters once the option moves. ``elasticity`` and ``importance_rank`` are REQUIRED numbers on the wire, so there
    is no field to null: the row is omitted (the 2.514(a) precedent), exactly as a factor ISL never analysed. The
    typed gate rides ``structural_influence[].gated_by`` and the ``STRUCTURAL_INFLUENCE_GATED`` critique."""

    def test_the_gated_factors_were_perturbed(self):
        # Precondition, so the omission below is not vacuous: every gated factor is in the OAT cohort.
        cohort = {u["node_id"] for u in zero_price(stated=False).get("parameter_uncertainties") or []}
        assert GATED <= cohort, GATED - cohort

    def test_no_gated_factor_has_an_oat_row(self, body):
        rows = {r["node_id"]: r for r in body.get("factor_sensitivity") or []}
        assert not GATED & set(rows), sorted(GATED & set(rows))

    def test_the_other_rows_stay_and_rank_contiguously(self, body):
        rows = body.get("factor_sensitivity") or []
        assert {PRICE, OTHER} <= {r["node_id"] for r in rows}  # the ungated half of the cohort
        assert sorted(r["importance_rank"] for r in rows) == list(range(1, len(rows) + 1))

    def test_without_a_gate_the_same_factors_keep_their_rows(self):
        from tests.unit.test_r3_identity_evaluation import v2_body

        rows = {r["node_id"] for r in v2_body({**wire(), "n_samples": 400}).get("factor_sensitivity") or []}
        cohort = {u["node_id"] for u in wire().get("parameter_uncertainties") or []}
        assert GATED & cohort <= rows
        assert GATED & cohort, "control needs a gated factor in the cohort"
