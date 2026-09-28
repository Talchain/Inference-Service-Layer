"""S-3 at a limit (AIQ #72 5866734772 root 2): a figure EXACTLY at a limit meets it.

Journey C's shape: a six-month spend tally (``sum`` identity) whose option sets it to exactly the £20,000
limit. The tally is exact in real arithmetic, but floating point leaves it an ulp either side of the limit
on each draw (``(20000 + a) - a`` with a sampled ``a``), so a bare ``value <= threshold`` read about half
the draws as a breach: P(<= £20,000) ~ 0.5 where the answer is 1. "At the limit" is the ONE relative S-3
tolerance (``NO_CHANGE_RELATIVE_TOLERANCE``, the B1a-5 "equal"): a real excess is still a breach.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import GoalConstraint, RobustnessRequestV2

LIMIT = "six_month_budget"


def tally(features_level: float) -> Dict[str, Any]:
    """A spend tally = features + advertising (£, frame 100,000), limit tally <= £20,000, goal revenue.
    Today's advertising spend is uncertain, so it is sampled on every draw; features is set by the option."""
    spend = {"frame": 100_000.0, "carrier": "cap"}

    def lever(node_id: str, label: str) -> Dict[str, Any]:
        return {
            "id": node_id,
            "kind": "factor",
            "label": label,
            "observed_state": {"value": 0.0, "cap": 100_000, "source": "brief_extraction"},
            "execution_frame": spend,
        }

    return {
        "request_id": "s3-limit-atom",
        "graph": {
            "nodes": [
                lever("features_spend", "Features spend"),
                lever("advertising_spend", "Advertising spend"),
                {
                    "id": "total_spend",
                    "kind": "outcome",
                    "label": "Six-month spend",
                    "observed_state": {
                        "value": 0.0,
                        "baseline": 0.0,
                        "cap": 100_000,
                        "source": "brief_extraction",
                    },
                    "execution_frame": spend,
                    "nonlinear_identity": {
                        "operation": "sum",
                        "factor_ids": ["features_spend", "advertising_spend"],
                        "stated_in_brief": True,
                    },
                },
                {"id": "revenue", "kind": "outcome", "label": "Revenue"},
            ],
            "edges": [
                {"from": "features_spend", "to": "total_spend", "strength": {"mean": 0.5, "std": 0.1}},
                {"from": "advertising_spend", "to": "total_spend", "strength": {"mean": 0.5, "std": 0.1}},
                {"from": "features_spend", "to": "revenue", "strength": {"mean": 0.3, "std": 0.1}},
                {"from": "advertising_spend", "to": "revenue", "strength": {"mean": 0.3, "std": 0.1}},
            ],
        },
        "options": [
            {"id": "at_limit", "label": "Spend the budget", "interventions": {"features_spend": features_level}},
            {"id": "carry_on", "label": "Carry on", "interventions": {}},
        ],
        "parameter_uncertainties": [
            {"node_id": "advertising_spend", "distribution": "normal", "std": 0.1},
        ],
        "goal_node_id": "revenue",
        "goal_constraints": [
            {"constraint_id": LIMIT, "node_id": "total_spend", "operator": "<=", "value": 0.2, "value_frame": "level"}
        ],
        "n_samples": 400,
        "seed": 7,
    }


def prob_satisfied(d: Dict[str, Any], option_id: str) -> float:
    response = rav2.RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(copy.deepcopy(d)))
    (result,) = [r for r in response.results if r.option_id == option_id]
    assert result.constraint_analysis is not None, f"{option_id}: constraint_analysis omitted"
    (row,) = [c for c in result.constraint_analysis.constraints if c.constraint_id == LIMIT]
    return float(row.prob_satisfied)


class TestExactlyAtTheLimitMeetsIt:
    def test_spending_exactly_the_limit_meets_a_le_limit_on_every_draw(self):
        """£20,000 against a £20,000 limit: P = 1, not ~0.5."""
        assert prob_satisfied(tally(0.2), "at_limit") == 1.0

    def test_control_a_real_excess_is_still_a_breach(self):
        """£20,000.02 (a millionth over, far above S-3's 1e-9) breaches on every draw."""
        assert prob_satisfied(tally(0.2 * (1 + 1e-6)), "at_limit") == 0.0

    def test_control_below_the_limit_meets_it(self):
        assert prob_satisfied(tally(0.2), "carry_on") == 1.0


def constraint(operator: str, threshold: float) -> GoalConstraint:
    return GoalConstraint(constraint_id=LIMIT, node_id="n", operator=operator, value=threshold, value_frame="level")


class TestTheComparison:
    check = staticmethod(rav2.RobustnessAnalyzerV2()._check_constraint_satisfied)

    @pytest.mark.parametrize("operator, above", [("<=", True), (">=", False)])
    def test_an_ulp_on_the_wrong_side_is_at_the_limit(self, operator: str, above: bool):
        threshold = 0.2
        value = threshold * (1 + 4e-16) if above else threshold * (1 - 4e-16)
        assert value != threshold
        assert self.check(value, constraint(operator, threshold))

    @pytest.mark.parametrize("operator, value", [("<=", 0.2 * (1 + 1e-6)), (">=", 0.2 * (1 - 1e-6))])
    def test_a_real_miss_is_still_a_miss(self, operator: str, value: float):
        assert not self.check(value, constraint(operator, 0.2))

    def test_a_zero_limit_is_met_only_by_zero_or_below(self):
        """Relative, no absolute floor (S-3): at a limit of exactly 0, 1e-300 is over it."""
        assert self.check(0.0, constraint("<=", 0.0))
        assert not self.check(1e-300, constraint("<=", 0.0))

    def test_the_tolerance_is_the_one_s3_constant(self):
        threshold = 0.2
        inside = threshold * (1 + 0.5 * rav2.NO_CHANGE_RELATIVE_TOLERANCE)
        outside = threshold * (1 + 2 * rav2.NO_CHANGE_RELATIVE_TOLERANCE)
        assert self.check(inside, constraint("<=", threshold))
        assert not self.check(outside, constraint("<=", threshold))
