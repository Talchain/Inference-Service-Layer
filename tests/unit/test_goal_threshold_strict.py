"""S4 (B): a STRICT goal ("MRR above £85k") is met only strictly past its threshold (#72 5879133964).

Measured on Paul's pricing brief (live drafter, N=3): the goal comparator is ">" on 3/3, and CEE withholds
the stated current level for any ">" goal because ISL scored ">=" whatever the brief said, so a status quo
held exactly at the target would score 100% on a goal it has not reached. ``goal_threshold_strict`` carries
the comparator to the one place that scores it.

The two comparisons agree on a continuous outcome; they differ only on a draw EXACTLY on the threshold. So
every discriminating row here puts an atom on the threshold (a do-nothing option on a held level, a root
goal with no spread), and the continuous row is the control that the flag changes nothing else.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
import pytest
from pydantic import ValidationError

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from tests.unit.test_goal_threshold_frame import PUSH_DRIVER, build_request
from tests.unit.test_r3_identity_evaluation import v2_body, wire


def p_goal(request: RobustnessRequestV2, **update: Any) -> float:
    response = RobustnessAnalyzerV2().analyze(request.model_copy(update=update))
    (result,) = response.results
    assert result.probability_of_goal is not None
    return result.probability_of_goal


class TestAnAtomOnTheThresholdIsNotPastIt:
    """The witness graph's goal is held at 0.7 today; a do-nothing option leaves it there on every draw."""

    def at_today(self) -> RobustnessRequestV2:
        return build_request(goal_threshold=0.7, goal_threshold_frame="level", baseline=0.7, option_interventions={})

    def test_at_least_is_met_and_above_is_not(self):
        assert p_goal(self.at_today()) == 1.0  # ">=": 0.7 reaches 0.7
        assert p_goal(self.at_today(), goal_threshold_strict=False) == 1.0
        assert p_goal(self.at_today(), goal_threshold_strict=True) == 0.0  # ">": 0.7 is not above 0.7

    def test_minimising_mirrors_it(self):
        assert p_goal(self.at_today(), goal_direction="minimise") == 1.0  # "<=": at most 0.7
        assert p_goal(self.at_today(), goal_direction="minimise", goal_threshold_strict=True) == 0.0  # "<"

    def test_a_root_goal_takes_the_same_rule_through_the_delta_branch(self):
        # A root goal's samples ARE its level (the identity plan, `delta_threshold`); with no spread it is 0.7
        # on every draw.
        root = build_request(goal_threshold=0.7, goal_threshold_frame="level", goal_is_root=True, option_interventions={})
        root = root.model_copy(update={"parameter_uncertainties": None})
        assert p_goal(root) == 1.0
        assert p_goal(root, goal_threshold_strict=True) == 0.0


class TestATieAtTheThresholdIsOnIt:
    """AIQ 5880886200: a held option's compared level (the goal baseline, paired) and the threshold arrive by different
    arithmetic, so an exact comparison lets one ulp decide 0% vs 100% for that option. A draw within a relative 1e-9 of
    the threshold is ON it: not met when strict, met when not. Both sides of the atom, both directions."""

    def held_at(self, level: float) -> RobustnessRequestV2:
        return build_request(
            goal_threshold=0.8, goal_threshold_frame="level", baseline=level, goal_observed_value=level, option_interventions={}
        )

    @pytest.mark.parametrize("side", ["one ulp above", "one ulp below"])
    def test_one_ulp_either_side_is_on_the_threshold(self, side):
        level = float(np.nextafter(0.8, 1.0 if side == "one ulp above" else 0.0))
        assert level != 0.8
        held = self.held_at(level)
        assert p_goal(held) == 1.0  # ">=": on the threshold is met
        assert p_goal(held, goal_threshold_strict=True) == 0.0  # ">": on the threshold is not past it
        assert p_goal(held, goal_direction="minimise") == 1.0  # "<="
        assert p_goal(held, goal_direction="minimise", goal_threshold_strict=True) == 0.0  # "<"

    def test_control_a_real_gap_is_not_a_tie(self):
        above, below = self.held_at(0.8 + 1e-6), self.held_at(0.8 - 1e-6)
        assert p_goal(above) == p_goal(above, goal_threshold_strict=True) == 1.0
        assert p_goal(below) == p_goal(below, goal_threshold_strict=True) == 0.0


class TestItChangesNothingElse:
    def test_a_continuous_outcome_scores_the_same_either_way(self):
        pushed = build_request(goal_threshold=0.8, goal_threshold_frame="level", option_interventions=PUSH_DRIVER)
        loose, strict = p_goal(pushed), p_goal(pushed, goal_threshold_strict=True)
        assert 0.0 < loose < 1.0  # a real spread, so this row can see a difference if one exists
        assert strict == loose

    def test_absent_and_false_are_byte_identical(self):
        pushed = build_request(goal_threshold=0.8, goal_threshold_frame="level", option_interventions=PUSH_DRIVER)
        a = RobustnessAnalyzerV2().analyze(pushed).model_dump(mode="json", exclude={"execution_time_ms", "timestamp"})
        b = RobustnessAnalyzerV2().analyze(pushed.model_copy(update={"goal_threshold_strict": False})).model_dump(
            mode="json", exclude={"execution_time_ms", "timestamp"}
        )
        assert a["results"] == b["results"]


class TestTheEnvelope:
    def base(self) -> Dict[str, Any]:
        return build_request(goal_threshold=0.7, goal_threshold_frame="level").model_dump(mode="json", by_alias=True)

    def test_strict_needs_a_threshold(self):
        d = self.base()
        d["goal_threshold"] = None
        d["goal_threshold_frame"] = None
        d["goal_threshold_strict"] = True
        with pytest.raises(ValidationError, match="requires goal_threshold"):
            RobustnessRequestV2.model_validate(d)

    def test_false_without_a_threshold_is_accepted(self):
        d = self.base()
        d["goal_threshold"] = None
        d["goal_threshold_frame"] = None
        d["goal_threshold_strict"] = False
        RobustnessRequestV2.model_validate(d)


class TestPaulsServedGraphOnTheV2Wire:
    """The served pricing wire (PLoT a6da42b capture of a295e4a1): MRR stated at £75,000 (0.6 of £125,000).
    Keeping the price at £49 changes nothing (B1a-5), so that option's MRR is exactly today's on every draw."""

    def body(self, **extra: Any) -> Dict[str, Any]:
        d = copy.deepcopy(wire())
        d["goal_threshold"] = 0.6  # "above £75,000": today's level itself
        d["n_samples"] = 1000
        d.update(extra)
        return d

    @staticmethod
    def keep(body: Dict[str, Any]) -> float:
        (option,) = [o for o in body["options"] if o["id"] == "keep_current_49_price"]
        return option["probability_of_goal"]

    def test_keeping_the_price_is_at_least_today_but_not_above_it(self):
        assert self.keep(v2_body(self.body())) == 1.0
        assert self.keep(v2_body(self.body(goal_threshold_strict=True))) == 0.0
