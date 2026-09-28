"""R3-5, ISL half (DL ruling #72 5872746926, AIQ meaning 5872728325): ONE structural-influence authority
over EVERY factor node.

``factor_sensitivity`` scores structural influence only over the factors that carry a parameter
uncertainty. Structural influence is a property of every factor with a path to the goal, whether or not
its value is uncertain. On Paul's served wire (``paul_a295e4a1_served_wire_plot_a6da42b.json``: 6 factor
nodes, 5 parameter uncertainties) ``fac_existing_customers_grandfathered`` has no observed value, so it
had no score, and PLoT could not publish ISL's influence for every row: the UI shows producer influence
only when EVERY factor carries one (DGAI ``useResultsSectionData.ts:2958``).

When an identity is EVALUATED, the V2 envelope carries a top-level ``structural_influence`` list: every
factor node, one cohort, one normalisation, with #195's identity partials. ``factor_sensitivity`` is
byte-identical. Without an evaluated identity the key is absent, so the response is byte-identical.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import RobustnessRequestV2
from tests.unit.test_r3_identity_evaluation import MRR, OTHER, PRICE, PRODUCT, SUBS, v2_body, wire

GRANDFATHERED = "fac_existing_customers_grandfathered"
ALL_FACTORS = [PRICE, SUBS, "monthly_churn", "monthly_new_pro_subscribers", OTHER, GRANDFATHERED]
N_SAMPLES = 400


def envelope(d: Dict[str, Any]) -> Dict[str, Any]:
    """The V2 envelope PLoT reads, through the real route and response builder (#187's ``v2_body``)."""
    return v2_body({**d, "n_samples": N_SAMPLES})


def declared_but_unused() -> Dict[str, Any]:
    """#187's withheld shape (``test_a_declared_but_unused_identity_says_it_was_not_evaluated``): a 200 whose
    only identity is withheld as ``identity_frame_missing``."""
    d = wire(identity=None, frames={})
    d["graph"]["nodes"].append(
        {
            "id": "side_total",
            "kind": "factor",
            "label": "Side total",
            "nonlinear_identity": {"operation": "sum", "factor_ids": [OTHER], "stated_in_brief": False},
        }
    )
    d["graph"]["edges"].append({"from": OTHER, "to": "side_total", "strength": {"mean": 1.0, "std": 0.01}})
    return d


def walk(d: Dict[str, Any], factors: List[str]) -> Dict[str, float]:
    """The one structural walk (#195 partials), over the cohort named."""
    request = RobustnessRequestV2.model_validate(d)
    scores, truncated = rav2.RobustnessAnalyzerV2()._compute_structural_influence(
        request.graph, factors, request.goal_node_id, factor_centres=rav2.factor_centres(request)
    )
    assert truncated == []
    return scores


def by_node(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    return {r["node_id"]: r.get(key) for r in rows}


@pytest.fixture(scope="module")
def p1() -> Dict[str, Any]:
    return envelope(wire(identity=PRODUCT))


@pytest.fixture(scope="module")
def c0() -> Dict[str, Any]:
    return envelope(wire(identity=None))


class TestEveryFactorNodeUnderAnEvaluatedIdentity:
    def test_precondition_this_is_the_envelope_plot_reads_and_the_identity_is_evaluated(self, p1):
        assert [e["evaluated"] for e in p1["identity_evaluations"] if e["node_id"] == MRR] == [True]

    def test_structural_influence_covers_all_six_factor_nodes_including_the_unobserved_one(self, p1):
        assert sorted(r["node_id"] for r in p1["structural_influence"]) == sorted(ALL_FACTORS)
        assert GRANDFATHERED not in {r["node_id"] for r in p1["factor_sensitivity"]}

    def test_each_score_is_the_one_walk_over_the_six_normalised_over_six(self, p1):
        expected = walk(wire(identity=PRODUCT), ALL_FACTORS)
        assert by_node(p1["structural_influence"], "influence_score") == pytest.approx(expected, abs=1e-12)
        assert max(expected.values()) == 1.0

    def test_ranks_follow_the_scores_one_is_highest(self, p1):
        rows = sorted(p1["structural_influence"], key=lambda r: r["influence_rank"])
        assert [r["influence_rank"] for r in rows] == list(range(1, 7))
        scores = [r["influence_score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_price_is_not_the_top_bar_under_mrr_equals_price_times_subscribers(self, p1):
        scores = by_node(p1["structural_influence"], "influence_score")
        assert scores[PRICE] != 1.0
        assert scores[PRICE] < scores[SUBS]

    def test_factor_sensitivity_is_byte_identical_to_walking_its_five_row_cohort_alone(self, p1):
        """The cohort is walked first in the same order and re-normalised over itself: EXACT, not approx."""
        five = [PRICE, SUBS, "monthly_churn", "monthly_new_pro_subscribers", OTHER]
        assert by_node(p1["factor_sensitivity"], "influence_score") == walk(wire(identity=PRODUCT), five)


def unobserved_leads() -> Dict[str, Any]:
    """P1 with FOUR extra unit paths from the unobserved factor into MRR (through chance nodes): its raw
    sum 0.59 + 4 = 4.59 then exceeds subscribers' 3.96 (measured), so the six-factor max is NOT the
    cohort's max. Only here can a six-factor normalisation leaking into factor_sensitivity show."""
    d = wire(identity=PRODUCT)
    for i in range(4):
        d["graph"]["nodes"].append({"id": f"g_path_{i}", "kind": "chance", "label": f"G path {i}"})
        d["graph"]["edges"].append(
            {"from": GRANDFATHERED, "to": f"g_path_{i}", "strength": {"mean": 1.0, "std": 0.01}, "exists_probability": 1.0}
        )
        d["graph"]["edges"].append(
            {"from": f"g_path_{i}", "to": MRR, "strength": {"mean": -1.0, "std": 0.01}, "exists_probability": 1.0}
        )
    return d


class TestWhenTheUnobservedFactorLeads:
    def test_factor_sensitivity_is_still_exactly_its_own_five_row_walk(self):
        d = unobserved_leads()
        body = envelope(d)
        assert [e["evaluated"] for e in body["identity_evaluations"] if e["node_id"] == MRR] == [True]
        scores = by_node(body["structural_influence"], "influence_score")
        assert max(scores, key=scores.get) == GRANDFATHERED
        five = [PRICE, SUBS, "monthly_churn", "monthly_new_pro_subscribers", OTHER]
        assert by_node(body["factor_sensitivity"], "influence_score") == walk(d, five)
        assert by_node(body["factor_sensitivity"], "influence_score")[SUBS] == 1.0


class TestAbsentWithoutAnEvaluatedIdentity:
    def test_c0_has_no_structural_influence_key(self, c0):
        assert "structural_influence" not in c0

    def test_a_withheld_identity_has_no_structural_influence_key(self):
        body = envelope(declared_but_unused())
        assert [e["evaluated"] for e in body["identity_evaluations"]] == [False]
        assert "structural_influence" not in body


class TestOnePoolAndTruncation:
    def test_no_new_cost_term_the_every_factor_walk_rides_the_priced_pool(self):
        with_identity = rav2.compute_weighted_cost(RobustnessRequestV2.model_validate(wire(identity=PRODUCT)))
        without = rav2.compute_weighted_cost(RobustnessRequestV2.model_validate(wire(identity=None)))
        assert with_identity.terms == without.terms
        assert with_identity.terms["structural_influence"] == rav2.MAX_INFLUENCE_WALK_CALLS_TOTAL

    def test_one_pool_the_cohort_first_a_pool_that_fits_five_but_not_six(self, monkeypatch):
        """Measured on this wire: the five-row cohort needs 19 walk calls, all six need 24. At 19 the
        cohort is exact and published; the every-factor list is withheld (exact-or-null), not partial."""
        monkeypatch.setattr(rav2, "MAX_INFLUENCE_WALK_CALLS_TOTAL", 19)
        body = envelope(wire(identity=PRODUCT))
        assert all(r.get("influence_score") is not None for r in body["factor_sensitivity"])
        rows = body["structural_influence"]
        assert sorted(r["node_id"] for r in rows) == sorted(ALL_FACTORS)
        assert all(r.get("influence_score") is None and r.get("influence_rank") is None for r in rows)

    def test_a_truncated_cohort_withholds_every_score_and_rank(self, monkeypatch):
        monkeypatch.setattr(rav2, "MAX_INFLUENCE_WALK_CALLS_TOTAL", 3)
        body = envelope(wire(identity=PRODUCT))
        rows = body["structural_influence"]
        assert sorted(r["node_id"] for r in rows) == sorted(ALL_FACTORS)
        assert all(r.get("influence_score") is None and r.get("influence_rank") is None for r in rows)


def test_c0_serialises_byte_identically_whatever_the_new_model_field():
    """exclude_none: an unset optional is ABSENT on the wire, so C0's JSON carries no new key at all."""
    body = envelope(wire(identity=None))
    assert "structural_influence" not in json.dumps(body)
