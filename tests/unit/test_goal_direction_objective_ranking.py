"""ROADMAP 2.1192 — the comparison must evaluate the user's stated objective.

THE DEFECT, measured at staging tip 28fe0c95 with a discriminating contrast
control (supplying a target moved nothing; flipping an edge sign moved
everything)::

    no target                : modest=0.00 | aggressive=1.00  rec=aggressive
    goal_threshold=0.3 delta : modest=0.00 | aggressive=1.00  rec=aggressive
    goal_threshold=0.9 delta : modest=0.00 | aggressive=1.00  rec=aggressive   <- both P(goal)=0.0
    CONTROL flip edge sign   : modest=1.00 | aggressive=0.00  rec=modest       <- instrument discriminates

"Wins" was ``argmax`` over the propagated goal-node scalar and nothing else. The
threshold channel computed ``probability_of_goal`` BESIDE the comparison and
never fed INTO it, so the crowned option could carry a zero percent modelled
chance of meeting the stated goal. And because a linear SCM is monotone in each
intervention, an argmax always lands on a CORNER: an option deliberately placed
between two extremes was structurally incapable of winning, whatever the
evidence said.

THE WITNESS GRAPH is chosen so every expectation is derivable BY HAND rather
than read back off the code::

    driver  root factor, observed_state{value: 0.5, baseline: 0.5}
    goal    non-root outcome, observed_state{value: 0.5, baseline: 0.5}
    driver -> goal   strength mean 1.0, std 0.01, exists_probability 1.0

    sample(goal) ~= driver_value  (single unit-strength parent, near-deterministic)

    modest      do(driver := 0.3)  =>  goal ~= 0.30
    aggressive  do(driver := 0.9)  =>  goal ~= 0.90

The two options differ ONLY in how they relate to a stated target of 0.3: under
``maximise`` the aggressive option is further up, under ``target`` the modest
option is nearer. Nothing else about them differs, which is what makes the
ranking flip attributable to the objective and to nothing else.

EVERY assertion below binds to its option by IDENTITY (``option_id``), never by
a value predicate another option could satisfy.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import ValidationError

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import ObjectivePlan, RobustnessAnalyzerV2

# --------------------------------------------------------------------------
# Instrument binding (Python trap): assert the module under test is THIS tree,
# not an editable .pth rebind to another clone. A mutation measured against the
# wrong tree is indistinguishable from a mutant that does not bite.
# --------------------------------------------------------------------------
import src.services.robustness_analyzer_v2 as _analyzer_module


def test_module_under_test_is_this_worktree() -> None:
    assert Path(_analyzer_module.__file__).resolve() == (
        Path(__file__).resolve().parents[2] / "src/services/robustness_analyzer_v2.py"
    )


N_SAMPLES = 400
SEED = 42


def _request(**overrides: Any) -> RobustnessRequestV2:
    payload: Dict[str, Any] = {
        "graph": {
            "nodes": [
                {
                    "id": "driver",
                    "kind": "factor",
                    "label": "Driver",
                    "observed_state": {"value": 0.5, "baseline": 0.5},
                },
                {
                    "id": "goal",
                    "kind": "outcome",
                    "label": "Goal",
                    "observed_state": {"value": 0.5, "baseline": 0.5},
                },
            ],
            "edges": [
                {
                    "from": "driver",
                    "to": "goal",
                    "exists_probability": 1.0,
                    "strength": {"mean": 1.0, "std": 0.01},
                }
            ],
        },
        "options": [
            {"id": "modest", "label": "Modest move", "interventions": {"driver": 0.3}},
            {
                "id": "aggressive",
                "label": "Aggressive move",
                "interventions": {"driver": 0.9},
            },
        ],
        "goal_node_id": "goal",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "analysis_types": ["comparison"],
    }
    payload.update(overrides)
    return RobustnessRequestV2(**payload)


def _wins(response: Any) -> Dict[str, Optional[float]]:
    """Win probability keyed by option IDENTITY, never by position or value."""
    return {r.option_id: r.win_probability for r in response.results}


def _warning_codes(response: Any) -> List[str]:
    return [w.code for w in response.inference_warnings]


# ==========================================================================
# ACCEPTANCE 1 — the ranking must MOVE with the stated objective
# ==========================================================================


class TestRankingHonoursTheStatedObjective:
    """RED at pristine: at 28fe0c95 every one of these rankings was identical."""

    def test_maximise_crowns_the_option_furthest_up(self) -> None:
        """The historical rule, now stated rather than assumed.

        This is the CONTRAST arm of the pair. It must stay green under every
        change below, or a 'fix' that simply inverted the comparison would look
        indistinguishable from one that honoured the objective.
        """
        response = RobustnessAnalyzerV2().analyze(_request(goal_direction="maximise"))
        wins = _wins(response)
        assert wins["aggressive"] > 0.99
        assert wins["modest"] < 0.01
        assert response.recommended_option_id == "aggressive"

    def test_minimise_crowns_the_option_furthest_down(self) -> None:
        """The case the historical rule got BACKWARDS, not merely missed.

        A goal node that is a cost, a churn rate or a risk was ranked by
        ``max()``: the product crowned whichever option made the outcome WORST,
        with full confidence and no disclosure anywhere on the wire.
        """
        response = RobustnessAnalyzerV2().analyze(_request(goal_direction="minimise"))
        wins = _wins(response)
        assert wins["modest"] > 0.99
        assert wins["aggressive"] < 0.01
        assert response.recommended_option_id == "modest"

    def test_target_crowns_the_option_nearest_the_stated_target(self) -> None:
        """⭐ THE ACCEPTANCE TEST. Two options differing ONLY in how they relate
        to a stated target; the ranking changes accordingly.

        Target 0.3 (delta frame). ``modest`` lands at ~0.30 and ``aggressive``
        at ~0.90, so ``|outcome - target|`` is ~0.00 against ~0.60. Under the
        pristine argmax this ranking was ``aggressive`` 1.00 — supplying the
        target changed it by exactly nothing.
        """
        response = RobustnessAnalyzerV2().analyze(
            _request(
                goal_direction="target",
                goal_threshold=0.3,
                goal_threshold_frame="delta",
            )
        )
        wins = _wins(response)
        assert wins["modest"] > 0.99, wins
        assert wins["aggressive"] < 0.01, wins
        assert response.recommended_option_id == "modest"

    def test_the_target_itself_decides_the_winner_not_merely_its_presence(self) -> None:
        """Move ONLY the target and the crown must move with it.

        Guards the shape where a fix reads 'was a target supplied?' rather than
        'what IS the target?' — which would pass the test above while still not
        evaluating the user's objective. Same graph, same options, same seed:
        the ONLY difference between the two runs is the number.
        """
        near_modest = _wins(
            RobustnessAnalyzerV2().analyze(
                _request(
                    goal_direction="target",
                    goal_threshold=0.3,
                    goal_threshold_frame="delta",
                )
            )
        )
        near_aggressive = _wins(
            RobustnessAnalyzerV2().analyze(
                _request(
                    goal_direction="target",
                    goal_threshold=0.9,
                    goal_threshold_frame="delta",
                )
            )
        )
        assert near_modest["modest"] > 0.99
        assert near_aggressive["aggressive"] > 0.99

    def test_a_moderate_option_can_win(self) -> None:
        """The structural claim: an option BETWEEN the extremes can lead.

        Under a monotone linear SCM an argmax (or argmin) always lands on a
        corner, so a middle option scored ~1.5% in the reproduction purely for
        being in the middle — 'the optimum is in the middle', the correct answer
        to most pricing and capacity questions, was unsayable. Here ``middle``
        is neither the largest nor the smallest outcome, and it wins.
        """
        response = RobustnessAnalyzerV2().analyze(
            _request(
                options=[
                    {"id": "low", "label": "Low", "interventions": {"driver": 0.1}},
                    {"id": "middle", "label": "Middle", "interventions": {"driver": 0.5}},
                    {"id": "high", "label": "High", "interventions": {"driver": 0.9}},
                ],
                goal_direction="target",
                goal_threshold=0.5,
                goal_threshold_frame="delta",
            )
        )
        wins = _wins(response)
        assert wins["middle"] > 0.99, wins
        assert wins["low"] < 0.01, wins
        assert wins["high"] < 0.01, wins
        assert response.recommended_option_id == "middle"


# ==========================================================================
# ACCEPTANCE 2 — an undeterminable direction must WITHHOLD, never guess
# ==========================================================================


class TestUndeterminableDirectionWithholdsTheRanking:
    def test_target_without_a_target_is_refused_at_parse(self) -> None:
        """Envelope incoherence: a 'target' sense with no target.

        Refused before any compute. The alternative — accept it and rank by
        ``max()`` anyway — is the exact substitution this row exists to end.
        """
        with pytest.raises(ValidationError) as excinfo:
            _request(goal_direction="target")
        message = str(excinfo.value)
        assert "goal_threshold" in message
        assert "goal_threshold_frame" in message

    def test_target_without_a_frame_is_refused_at_parse(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            _request(goal_direction="target", goal_threshold=0.3)
        assert "goal_threshold_frame" in str(excinfo.value)

    def test_unresolvable_target_withholds_every_win_probability(self) -> None:
        """⭐ THE SECOND ACCEPTANCE TEST. Direction stated, target unusable.

        The goal here is a ROOT node, which the frame resolver refuses: a root
        goal's samples are seeded from its own observed value, so there is no
        status-quo reference to recover a level against. The objective was
        STATED and cannot be SCORED.

        The product must not answer with a maximiser under the target's label.
        No option carries a win probability, no option is recommended, and the
        refusal is typed.
        """
        response = RobustnessAnalyzerV2().analyze(
            RobustnessRequestV2(
                graph={
                    "nodes": [
                        {
                            "id": "goal",
                            "kind": "outcome",
                            "label": "Goal",
                            "observed_state": {"value": 0.5, "baseline": 0.5},
                        },
                        {
                            "id": "other",
                            "kind": "factor",
                            "label": "Other",
                            "observed_state": {"value": 0.5, "baseline": 0.5},
                        },
                    ],
                    "edges": [
                        {
                            "from": "goal",
                            "to": "other",
                            "exists_probability": 1.0,
                            "strength": {"mean": 1.0, "std": 0.01},
                        }
                    ],
                },
                options=[
                    {"id": "modest", "label": "Modest", "interventions": {"goal": 0.3}},
                    {"id": "aggressive", "label": "Aggressive", "interventions": {"goal": 0.9}},
                ],
                goal_node_id="goal",
                n_samples=N_SAMPLES,
                seed=SEED,
                analysis_types=["comparison"],
                goal_direction="target",
                goal_threshold=0.6,
                goal_threshold_frame="level",
            )
        )

        assert response.objective_ranking.status == "withheld"
        assert response.objective_ranking.withheld_reason == (
            "target_not_resolvable_in_sample_frame"
        )
        assert "OBJECTIVE_RANKING_WITHHELD" in _warning_codes(response)

        # Absence survives internal serialization too; zero would imply a comparison.
        assert _wins(response) == {"modest": None, "aggressive": None}
        assert response.recommendation_confidence is None
        assert response.recommended_option_id is None
        assert response.robustness is None

        # And nothing may restate the ranking through a side channel.
        assert response.conditional_winners is None

    def test_a_withheld_ranking_reports_the_sense_that_was_asked_for(self) -> None:
        """A surface must be able to say WHAT could not be done.

        Reporting 'maximise' here (the sense that structurally ran) would tell
        the user we did the thing we just refused to do.
        """
        response = RobustnessAnalyzerV2().analyze(
            RobustnessRequestV2(
                graph={
                    "nodes": [
                        {
                            "id": "goal",
                            "kind": "outcome",
                            "label": "Goal",
                            "observed_state": {"value": 0.5, "baseline": 0.5},
                        },
                        {
                            "id": "other",
                            "kind": "factor",
                            "label": "Other",
                            "observed_state": {"value": 0.5, "baseline": 0.5},
                        },
                    ],
                    "edges": [
                        {
                            "from": "goal",
                            "to": "other",
                            "exists_probability": 1.0,
                            "strength": {"mean": 1.0, "std": 0.01},
                        }
                    ],
                },
                options=[
                    {"id": "modest", "label": "Modest", "interventions": {"goal": 0.3}},
                    {"id": "aggressive", "label": "Aggressive", "interventions": {"goal": 0.9}},
                ],
                goal_node_id="goal",
                n_samples=N_SAMPLES,
                seed=SEED,
                analysis_types=["comparison"],
                goal_direction="target",
                goal_threshold=0.6,
                goal_threshold_frame="level",
            )
        )
        assert response.objective_ranking.direction == "target"


# ==========================================================================
# The unattested default — pinned so it cannot change SILENTLY
# ==========================================================================


class TestAbsentDirectionIsDisclosedNotAssumedSilently:
    def test_absent_direction_withholds_without_inventing_direction(self) -> None:
        response = RobustnessAnalyzerV2().analyze(_request())
        assert _wins(response) == {"modest": None, "aggressive": None}
        assert response.objective_ranking.direction is None
        assert response.objective_ranking.attested is False
        assert response.objective_ranking.status == "withheld"
        assert response.objective_ranking.ranked_options == []
        assert "GOAL_DIRECTION_UNATTESTED" in _warning_codes(response)

    def test_an_attested_maximise_carries_no_unattested_warning(self) -> None:
        response = RobustnessAnalyzerV2().analyze(_request(goal_direction="maximise"))
        assert response.objective_ranking.attested is True
        assert "GOAL_DIRECTION_UNATTESTED" not in _warning_codes(response)
        assert _wins(response) == {"modest": 0.0, "aggressive": 1.0}

    def test_missing_direction_does_not_discard_marginal_outcomes(self) -> None:
        missing = RobustnessAnalyzerV2().analyze(_request())
        explicit = RobustnessAnalyzerV2().analyze(_request(goal_direction="maximise"))
        assert [r.outcome_distribution for r in missing.results] == [
            r.outcome_distribution for r in explicit.results
        ]


# ==========================================================================
# The canonical owner — one rule, no rival scorer
# ==========================================================================


class TestOneWinnerRule:
    def test_every_sense_routes_through_the_single_owner(self) -> None:
        """``_winners_for_draw`` is the only implementation of 'who wins'.

        Called directly here so a future rival scorer added beside it has to
        break this test to exist.
        """
        outcomes = {"a": 0.1, "b": 0.5, "c": 0.9}
        owner = RobustnessAnalyzerV2._winners_for_draw
        assert owner(outcomes, ObjectivePlan(sense="maximise", attested=True), None) == ["c"]
        assert owner(outcomes, ObjectivePlan(sense="minimise", attested=True), None) == ["a"]
        assert owner(
            outcomes,
            ObjectivePlan(sense="target", attested=True, target_delta=0.5),
            None,
        ) == ["b"]

    def test_a_withheld_plan_awards_no_winner_and_cannot_reach_a_ranking_limb(self) -> None:
        """The refusal is checked FIRST, so no later limb can be reached.

        A withheld plan carries no target, so any limb that fell through to the
        target branch would assert; any limb that fell through to maximise would
        silently rank. Neither may happen.
        """
        assert (
            RobustnessAnalyzerV2._winners_for_draw(
                {"a": 0.1, "b": 0.9},
                ObjectivePlan(sense="withheld", attested=True),
                None,
            )
            == []
        )

    def test_ties_split_identically_under_every_sense(self) -> None:
        """Tie semantics are the rule's, not the sense's."""
        owner = RobustnessAnalyzerV2._winners_for_draw
        tied = {"a": 0.5, "b": 0.5}
        assert sorted(owner(tied, ObjectivePlan(sense="maximise", attested=True), None)) == [
            "a",
            "b",
        ]
        assert sorted(owner(tied, ObjectivePlan(sense="minimise", attested=True), None)) == [
            "a",
            "b",
        ]
        assert sorted(
            owner(tied, ObjectivePlan(sense="target", attested=True, target_delta=0.2), None)
        ) == ["a", "b"]

    def test_a_level_framed_target_without_a_reference_is_uninformative(self) -> None:
        """Never scored against a broken anchor.

        The reference is shared by every option, so its absence poisons the
        whole draw rather than one option — recorded uninformative, exactly as a
        draw with no finite option already is.
        """
        plan = ObjectivePlan(sense="target", attested=True, target_level=0.6, goal_baseline=0.5)
        assert RobustnessAnalyzerV2._winners_for_draw({"a": 0.1, "b": 0.9}, plan, None) == []
        assert (
            RobustnessAnalyzerV2._winners_for_draw({"a": 0.1, "b": 0.9}, plan, float("nan")) == []
        )
