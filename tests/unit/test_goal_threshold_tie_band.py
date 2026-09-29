"""AIQ's tie band for the goal comparator (#72 5880886200), the condition before CEE sends ``goal_threshold_strict``.

A held option's compared level (the goal's baseline, paired) and the threshold come by different arithmetic,
so an exact comparison lets ONE ulp decide 0% vs 100% for that option, and the strict comparator (#209) makes
exactly this atom the decisive case. A draw within 1e-9 * max(1, |t|) of the threshold is ON it: not met when
strict ("above"), met when not ("at least"). Every discriminating row puts the held atom one ulp either side of
the threshold; the band-edge rows put it just outside the band, so a band that is too wide is RED too.
"""

from __future__ import annotations

import copy
import math

from typing import Any, Dict

import numpy as np
import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    GOAL_TIE_BAND_REL,
    RobustnessAnalyzerV2,
    meets_goal_threshold,
)
from tests.unit.test_goal_threshold_frame import PUSH_DRIVER, build_request
from tests.unit.test_r3_identity_evaluation import v2_body, wire

T = 0.8
ONE_ULP = {"above": math.nextafter(T, math.inf), "below": math.nextafter(T, -math.inf)}


def p_goal(request: RobustnessRequestV2, **update: Any) -> float:
    response = RobustnessAnalyzerV2().analyze(request.model_copy(update=update))
    (result,) = response.results
    assert result.probability_of_goal is not None
    return result.probability_of_goal


def held_at(level: float, **kwargs: Any) -> RobustnessRequestV2:
    """A do-nothing option on a goal held at ``level`` today, scored against the threshold T."""
    return build_request(
        goal_threshold=T,
        goal_threshold_frame="level",
        baseline=level,
        goal_observed_value=level,
        option_interventions={},
        **kwargs,
    )


def test_the_band_is_the_ruled_width():
    assert GOAL_TIE_BAND_REL == 1e-9


class TestOneUlpIsOnTheThreshold:
    @pytest.mark.parametrize("side", ["above", "below"])
    def test_level_branch_maximise(self, side):
        request = held_at(ONE_ULP[side])
        assert p_goal(request, goal_threshold_strict=True) == 0.0  # "above 0.8": on it, not past it
        assert p_goal(request) == 1.0  # "at least 0.8": on it, so met
        assert p_goal(request, goal_threshold_strict=False) == 1.0

    @pytest.mark.parametrize("side", ["above", "below"])
    def test_level_branch_minimise(self, side):
        request = held_at(ONE_ULP[side])
        assert (
            p_goal(request, goal_direction="minimise", goal_threshold_strict=True) == 0.0
        )  # "below 0.8"
        assert p_goal(request, goal_direction="minimise") == 1.0  # "at most 0.8"

    @pytest.mark.parametrize("side", ["above", "below"])
    def test_delta_branch_root_goal(self, side):
        root = held_at(ONE_ULP[side], goal_is_root=True).model_copy(
            update={"parameter_uncertainties": None}
        )
        assert p_goal(root, goal_threshold_strict=True) == 0.0
        assert p_goal(root) == 1.0


class TestJustOutsideTheBandIsNotATie:
    """The band is 1e-9 * max(1, |t|) = 1e-9 here; 1e-7 away is a real difference and keeps the exact answer."""

    def test_just_above(self):
        request = held_at(T + 1e-7)
        assert p_goal(request, goal_threshold_strict=True) == 1.0
        assert p_goal(request) == 1.0

    def test_just_below(self):
        request = held_at(T - 1e-7)
        assert p_goal(request, goal_threshold_strict=True) == 0.0
        assert p_goal(request) == 0.0


class TestMinimiseIsWiredThrough:
    """Outside the band, the direction decides (R3 SCIENCE 5881297285): the one-ulp rows alone let an analyser that
    ignored ``goal_direction`` survive, because on the tie the band alone decides."""

    @pytest.mark.parametrize("strict", [True, False])
    def test_below_the_threshold_meets_a_minimise_goal(self, strict):
        assert (
            p_goal(held_at(T - 1e-6), goal_direction="minimise", goal_threshold_strict=strict)
            == 1.0
        )

    @pytest.mark.parametrize("strict", [True, False])
    def test_above_the_threshold_does_not(self, strict):
        assert (
            p_goal(held_at(T + 1e-6), goal_direction="minimise", goal_threshold_strict=strict)
            == 0.0
        )


class TestTheComparatorItself:
    """The analyser's one comparator, direct. At |t| > 1 the band is RELATIVE (1e-9 * |t|): an absolute 1e-9 band
    would call 1e-7 away from 1000 "past"; the ruled band calls it ON. NaN is never met; +/-inf is never ON.
    """

    @staticmethod
    def met(value: float, threshold: float, *, strict: bool, minimise: bool = False) -> bool:
        return bool(
            meets_goal_threshold(np.array([value]), threshold, strict=strict, minimise=minimise)[0]
        )

    def test_the_band_scales_with_a_large_threshold(self):
        assert not self.met(1000.0 + 1e-7, 1000.0, strict=True)  # inside 1e-6: on it, not above it
        assert self.met(1000.0 - 1e-7, 1000.0, strict=False)  # inside: on it, so at least it
        assert self.met(1000.0 + 1e-5, 1000.0, strict=True)  # outside: a real difference
        assert not self.met(1000.0 - 1e-5, 1000.0, strict=False)

    def test_minimise_mirrors_the_band(self):
        assert not self.met(
            1000.0 - 1e-7, 1000.0, strict=True, minimise=True
        )  # "below 1000": on it
        assert self.met(1000.0 + 1e-7, 1000.0, strict=False, minimise=True)  # "at most 1000": on it
        assert self.met(1000.0 - 1e-5, 1000.0, strict=True, minimise=True)

    def test_non_finite_draws(self):
        for strict in (True, False):
            assert not self.met(math.nan, 0.8, strict=strict)
            assert self.met(
                math.inf, 0.8, strict=strict
            )  # past, never on (the finiteness gate drops it later)
            assert not self.met(-math.inf, 0.8, strict=strict)


class TestItChangesNothingElse:
    def test_a_continuous_outcome_scores_the_same_either_way(self):
        pushed = build_request(
            goal_threshold=T, goal_threshold_frame="level", option_interventions=PUSH_DRIVER
        )
        loose, strict = p_goal(pushed), p_goal(pushed, goal_threshold_strict=True)
        assert 0.0 < loose < 1.0
        assert strict == loose


class TestPaulsServedGraphOnTheV2Wire:
    """The served pricing wire (as in test_goal_threshold_strict): keeping the price at £49 leaves MRR exactly at
    today's 0.6 on every draw. A threshold one ulp off 0.6, as a different arithmetic path would produce it, must
    not flip that option between 0% and 100%."""

    def body(self, threshold: float, **extra: Any) -> Dict[str, Any]:
        d = copy.deepcopy(wire())
        d["goal_threshold"] = threshold
        d["n_samples"] = 1000
        d.update(extra)
        return d

    @staticmethod
    def keep(body: Dict[str, Any]) -> float:
        (option,) = [o for o in body["options"] if o["id"] == "keep_current_49_price"]
        return option["probability_of_goal"]

    @pytest.mark.parametrize(
        "threshold", [math.nextafter(0.6, math.inf), math.nextafter(0.6, -math.inf)]
    )
    def test_keeping_the_price_is_on_a_threshold_one_ulp_away(self, threshold):
        assert self.keep(v2_body(self.body(threshold, goal_threshold_strict=True))) == 0.0
        assert self.keep(v2_body(self.body(threshold))) == 1.0
