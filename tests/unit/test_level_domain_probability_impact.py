"""The money goal's lost ceiling MOVES probabilities, not only the band (verifier FIX_FIRST on AIQ #72 5866289608).

``test_level_domain_from_unit_meaning.py`` pins the card's p90. The same clamp also fed the goal verdict and the
limit block: ``probability_of_goal`` compares the goal's LEVEL clamped to its domain, and a 'level' limit reads
its target's levels clamped the same way. With MRR's frame (observed_state.cap, £125,000) read as a ceiling, no
draw could exceed 1.0 on the 0-1 frame, so:

* "MRR >= £137,500" (goal_threshold 1.1) was reported as 0% likely for every option (it is ~36% for £59);
* "MRR <= £137,500" (a level limit, no level_domain: money has no unit ceiling) was reported as certain.

Both are corrections: the head's figures ARE the share of the analyzer's own (unclamped) goal levels, which is
what the rows bind to. Where the threshold sits INSIDE [0, 1], or exactly at the frame for a '>=' comparison, a
clamp at 1.0 cannot change a comparison, so the figures are byte-identical at base and head (the controls).

Measured at plain staging a1fa8ae and on this branch rebased onto it, with this file's configuration (seed
972664972, n_samples 2000); the same figures as at base 14f1a3a -> head 4bb0519 before the rebase:
    goal_threshold 1.1   £59 probability_of_goal   0.0 -> 0.358   (£54 0.0 -> 0.232)
    mrr <= 1.1 limit     £59 prob_satisfied        1.0 -> 0.642   (£54 1.0 -> 0.768)
    mrr <= 1.0 limit     £59 prob_satisfied        1.0 -> 0.507   (at the frame: a '<=' moves)
    goal_threshold 1.0   £59 probability_of_goal   0.493 == 0.493 (at the frame: a '>=' does not)

NOT byte-identical, and not claimed: the limit's FAILURE MARGIN. Inside the frame (mrr <= 0.9) the probabilities
are identical, but a failing draw is no longer clipped at 1.0, so failure_margin_median for £59 moves 0.100 ->
0.239 (the median failing level was 1.0, the clip; it is now the draw's own). That is the same correction.

FIXTURE: the served journey A run 3 ISL request (``_provenance`` inside), with only the run's size, analysis set,
threshold and one added limit changed, as each row states.
"""

from __future__ import annotations

import copy
import json

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "anchored_delta"
    / "journey_a_run3_status_quo_held_plot_c0f0a9a.json"
)
SEED = "972664972"  # the served request's own seed, set explicitly
N_SAMPLES = 2_000

GOAL = "mrr"  # 'GBP per month', observed_state.cap 125000 (CEE's frame), no level_domain
KEEP = "keep_current_pricing"
P59 = "59_with_feature_release"
P54 = "54_with_feature_release"
OPTIONS = (KEEP, P59, P54)
MRR_LIMIT = "t5:mrr:level-limit"


def request(
    *, goal_threshold: Optional[float] = None, mrr_limit: Optional[Tuple[str, float]] = None
) -> Dict[str, Any]:
    payload = json.loads(FIXTURE.read_text())
    assert "_provenance" in payload, "the fixture must say where it came from"
    d = copy.deepcopy(payload["request"])
    assert d["seed"] == SEED and d["goal_node_id"] == GOAL
    (goal,) = [n for n in d["graph"]["nodes"] if n["id"] == GOAL]
    assert goal["observed_state"]["cap"] == 125_000 and goal["observed_state"]["value"] == 0.6
    assert not any(c["node_id"] == GOAL for c in d["goal_constraints"]), "no limit on MRR as served"
    d["seed"] = SEED
    d["n_samples"] = N_SAMPLES
    d["analysis_types"] = ["comparison"]
    d["include_e_values"] = False
    d["include_voi"] = False
    d["include_factor_flips"] = False
    if goal_threshold is not None:
        assert d["goal_threshold_frame"] == "level"
        d["goal_threshold"] = goal_threshold
    if mrr_limit is not None:
        operator, value = mrr_limit
        d["goal_constraints"].append(
            {
                "constraint_id": MRR_LIMIT,
                "node_id": GOAL,
                "operator": operator,
                "value": value,
                "value_frame": "level",
                # no level_domain: PLoT mints one only for a '%' limit; money has no unit ceiling
            }
        )
    return d


_CACHE: Dict[str, Any] = {}


def analyse(d: Dict[str, Any]):
    key = json.dumps(d, sort_keys=True)
    if key not in _CACHE:
        _CACHE[key] = RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))
    return _CACHE[key]


def result(response, option_id: str):
    (row,) = [r for r in response.results if r.option_id == option_id]
    return row


def goal_levels(response, option_id: str) -> np.ndarray:
    """The analyzer's own goal LEVELS for the option (unclamped at base and head alike), finite draws only."""
    levels = np.asarray(result(response, option_id).outcome_distribution.samples, dtype=float)
    assert levels.size == N_SAMPLES, levels.size
    return levels[np.isfinite(levels)]


def mrr_limit_row(response, option_id: str):
    analysis = result(response, option_id).constraint_analysis
    assert analysis is not None, option_id
    (row,) = [c for c in analysis.constraints if c.constraint_id == MRR_LIMIT]
    return row


# ---------------------------------------------------------------------------------------------------------
# Row 3a — a goal threshold ABOVE the frame: probability_of_goal was 0 for every option, now it is the share
# ---------------------------------------------------------------------------------------------------------


class TestRow3aAGoalThresholdAboveTheFrameIsNoLongerImpossible:
    @pytest.fixture(scope="class")
    def response(self):
        return analyse(request(goal_threshold=1.1))  # "MRR >= £137,500"

    def test_the_59_option_can_reach_a_goal_above_its_frame(self, response):
        """RED at base: probability_of_goal == 0.0 exactly (every level clamped at 1.0 < 1.1)."""
        p = result(response, P59).probability_of_goal
        assert p is not None
        assert 0.2 < p < 0.5, p

    def test_probability_of_goal_is_the_share_of_its_own_levels(self, response):
        """RED at base (bound by identity): for each option, the share of the analyzer's own levels at or above
        the threshold. The held status quo (0.6 on every draw) stays at 0 — the contrast inside the row."""
        for option_id in OPTIONS:
            levels = goal_levels(response, option_id)
            assert result(response, option_id).probability_of_goal == float(
                np.mean(levels >= 1.1)
            ), option_id
        assert result(response, KEEP).probability_of_goal == 0.0
        assert result(response, P54).probability_of_goal > 0.0


# ---------------------------------------------------------------------------------------------------------
# Row 3b — a money LEVEL limit at or above the frame: prob_satisfied was 1.0, now it is the share
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "low", "high"),
    [
        pytest.param(1.1, 0.5, 0.8, id="mrr <= 1.1 (GBP 137,500): above the frame"),
        pytest.param(1.0, 0.35, 0.65, id="mrr <= 1.0 (GBP 125,000): at the frame"),
    ],
)
def test_row3b_a_money_level_limit_at_or_above_the_frame_is_no_longer_certain(value, low, high):
    """RED at base: prob_satisfied == 1.0 for every option (every clamped level <= 1.0 <= value)."""
    response = analyse(request(mrr_limit=("<=", value)))
    p = mrr_limit_row(response, P59).prob_satisfied
    assert low < p < high, p
    for option_id in OPTIONS:
        levels = goal_levels(response, option_id)
        assert mrr_limit_row(response, option_id).prob_satisfied == float(
            np.mean(levels <= value)
        ), option_id
    assert mrr_limit_row(response, KEEP).prob_satisfied == 1.0  # held at 0.6: the contrast inside the row


# ---------------------------------------------------------------------------------------------------------
# CONTROLS — where a clamp at 1.0 cannot change a comparison, the figures are byte-identical at base and head
# ---------------------------------------------------------------------------------------------------------

# Measured at plain staging a1fa8ae through the analyzer, this file's configuration; identical on this branch.
# RE-PINNED on the rebase (was measured at 14f1a3a): ISL #193 (a1fa8ae, one central constant for a product
# identity) moved three figures by one draw in 2,000 on plain staging, without this change:
#     goal 0.8   £59  0.71   -> 0.7105
#     goal 1.0   £54  0.3695 -> 0.369
#     limit 0.9  £54  0.486  -> 0.4855
GOAL_PROBABILITY_AT_BASE = {
    0.8: {KEEP: 0.0, P59: 0.7105, P54: 0.641},  # inside the domain: "MRR >= £100,000"
    1.0: {KEEP: 0.0, P59: 0.493, P54: 0.369},  # at the frame, '>=': a clamped draw still meets it
}
LIMIT_PROBABILITY_AT_BASE = {
    0.9: {KEEP: 1.0, P59: 0.385, P54: 0.4855},  # inside the domain: "MRR <= £112,500"
}


@pytest.mark.parametrize("threshold", sorted(GOAL_PROBABILITY_AT_BASE))
def test_control_a_goal_threshold_the_clamp_cannot_reach_is_byte_identical(threshold):
    response = analyse(request(goal_threshold=threshold))
    got = {option_id: result(response, option_id).probability_of_goal for option_id in OPTIONS}
    assert got == GOAL_PROBABILITY_AT_BASE[threshold]
    assert {k: repr(v) for k, v in got.items()} == {
        k: repr(v) for k, v in GOAL_PROBABILITY_AT_BASE[threshold].items()
    }


@pytest.mark.parametrize("value", sorted(LIMIT_PROBABILITY_AT_BASE))
def test_control_a_money_limit_inside_the_domain_is_byte_identical(value):
    response = analyse(request(mrr_limit=("<=", value)))
    got = {option_id: mrr_limit_row(response, option_id).prob_satisfied for option_id in OPTIONS}
    assert got == LIMIT_PROBABILITY_AT_BASE[value]
    assert {k: repr(v) for k, v in got.items()} == {
        k: repr(v) for k, v in LIMIT_PROBABILITY_AT_BASE[value].items()
    }
