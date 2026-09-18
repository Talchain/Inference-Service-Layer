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
    assert _analyzer_module.__file__.endswith(
        "src/services/robustness_analyzer_v2.py"
    ), _analyzer_module.__file__


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


def _wins(response: Any) -> Dict[str, float]:
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

        # NO ranking: every option's win share is zero because no draw was
        # awarded to anyone, and the wire omits the field entirely (see the
        # api-layer test below). A zero here is the analyzer's internal record
        # of "no draw had a winner", not a measured comparison.
        wins = _wins(response)
        assert wins["modest"] == 0.0, wins
        assert wins["aggressive"] == 0.0, wins
        assert response.recommendation_confidence == 0.0

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
    """A KNOWN, DELIBERATE gap, pinned to EXACTLY its current shape.

    An absent ``goal_direction`` still ranks by ``max()``: ISL declares this
    field before any producer stamps it, and withholding every ranking in the
    window between the two halves would take a journey-witnessed capability dark
    for a contract that has not landed. The cost is stated on the wire rather
    than hidden. These tests RED if the behaviour changes in EITHER direction —
    if the disclosure disappears, and if the default silently becomes something
    other than an unattested maximise.
    """

    def test_absent_direction_ranks_by_maximise_and_says_so(self) -> None:
        response = RobustnessAnalyzerV2().analyze(_request())
        assert _wins(response)["aggressive"] > 0.99
        assert response.objective_ranking.direction == "maximise"
        assert response.objective_ranking.attested is False
        assert response.objective_ranking.status == "computed"
        assert "GOAL_DIRECTION_UNATTESTED" in _warning_codes(response)

    def test_an_attested_maximise_carries_no_unattested_warning(self) -> None:
        """The discrimination: the warning tracks ATTESTATION, not the sense.

        Without this pair, a warning emitted unconditionally would pass the test
        above while telling every consumer that every ranking is an assumption.
        """
        response = RobustnessAnalyzerV2().analyze(_request(goal_direction="maximise"))
        assert response.objective_ranking.attested is True
        assert "GOAL_DIRECTION_UNATTESTED" not in _warning_codes(response)

    def test_absent_direction_is_numerically_identical_to_attested_maximise(self) -> None:
        """No-regression, by execution rather than by assertion.

        Every deployed request today omits ``goal_direction``. If this pair ever
        diverges, the field changed a number it promised not to.
        """
        assert _wins(RobustnessAnalyzerV2().analyze(_request())) == _wins(
            RobustnessAnalyzerV2().analyze(_request(goal_direction="maximise"))
        )


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
        plan = ObjectivePlan(
            sense="target", attested=True, target_level=0.6, goal_baseline=0.5
        )
        assert RobustnessAnalyzerV2._winners_for_draw({"a": 0.1, "b": 0.9}, plan, None) == []
        assert (
            RobustnessAnalyzerV2._winners_for_draw({"a": 0.1, "b": 0.9}, plan, float("nan"))
            == []
        )


# ==========================================================================
# ACCEPTANCE 2b — a withheld ranking may not be restated on ANY side channel,
# and the RECORD of what it suppressed must survive to the wire
#
# Two blocking review findings at 0a570656, both reproduced here before the
# fix, both with a contrast control in the same probe:
#
# P1  `path_decomposition` was NOT withhold-gated. Under a withheld ranking
#     `_winners_for_draw` returns [], so `option_wins` is all-zero and
#     `recommended_option_id = max(option_wins, key=...)` returns the FIRST KEY
#     BY INSERTION ORDER. Measured at pristine on the witness below:
#
#         options [modest, aggressive] -> path_decomposition.recommended_option_id = "modest"
#         options [aggressive, modest] -> path_decomposition.recommended_option_id = "aggressive"
#         path_count = 2 (a full causal breakdown "explaining why it wins")
#         CONTRAST  conditional_winners = None   <- that sibling IS gated
#
#     A field labelled "The recommended option this decomposition explains",
#     populated by array position, on a response that refused to say which
#     option wins.
#
# P2  The suppression markers were DARK whenever correlation was inactive.
#     They were appended to `suppressed_attributions`, whose only carrier is
#     `correlation_model.suppressed_attributions`, and `_build_correlation_
#     disclosure` returns None when no `factor_correlations` were declared —
#     the common case. Measured at pristine: `correlation_model is None` on
#     every request below, so every marker the withhold path recorded was
#     silently discarded and the omission was "merely absent", which is the
#     exact thing the skip-site comment says it is avoiding.
#
# WHY THE SUITE COULD NOT SEE EITHER: `include_path_decomposition` defaults
# OFF, so the phase is absent from every fixture unless a test asks for it;
# and no test referenced the markers at all. Every request in this section
# therefore sets `include_path_decomposition=True` explicitly and declares NO
# `factor_correlations` — without those two clauses these tests pass vacuously,
# which is how both defects got here.
# ==========================================================================


# The witness graph for this section differs from the module-level one in
# exactly one respect: the goal carries NO `observed_state.baseline`, so a
# level-framed target refuses with `missing_goal_baseline` and the ranking is
# withheld. It is a THREE-node chain with a second direct edge so that
# `path_decomposition` has real content (2 paths, entry node `driver`) — the
# leak under test is a populated block, not an empty shell.
#
#     driver --1.0--> mid --1.0--> goal        (path 1, ~0.667 of the effect)
#     driver --0.5------------> goal           (path 2, ~0.333 of the effect)
#
# Options intervene on `driver`, never on the goal, so the goal is not pinned
# and the refusal is attributable to the missing baseline alone.
_WITHHOLD_WITNESS_GRAPH: Dict[str, Any] = {
    "nodes": [
        {
            "id": "driver",
            "kind": "factor",
            "label": "Driver",
            "observed_state": {"value": 0.5, "baseline": 0.5},
        },
        {
            "id": "mid",
            "kind": "factor",
            "label": "Mid",
            "observed_state": {"value": 0.5, "baseline": 0.5},
        },
        {
            "id": "goal",
            "kind": "outcome",
            "label": "Goal",
            "observed_state": {"value": 0.5},  # no baseline -> level frame refuses
        },
    ],
    "edges": [
        {
            "from": "driver",
            "to": "mid",
            "exists_probability": 1.0,
            "strength": {"mean": 1.0, "std": 0.01},
        },
        {
            "from": "mid",
            "to": "goal",
            "exists_probability": 1.0,
            "strength": {"mean": 1.0, "std": 0.01},
        },
        {
            "from": "driver",
            "to": "goal",
            "exists_probability": 1.0,
            "strength": {"mean": 0.5, "std": 0.01},
        },
    ],
}

_WITHHOLD_WITNESS_OPTIONS: List[Dict[str, Any]] = [
    {"id": "modest", "label": "Modest move", "interventions": {"driver": 0.3}},
    {"id": "aggressive", "label": "Aggressive move", "interventions": {"driver": 0.9}},
]

# Two factor uncertainties so the optional phases that the withhold path
# suppresses are genuinely REACHABLE on this request — a skip site that could
# not have run anyway records nothing, and a manifest assertion over it would
# be a tautology.
_WITHHOLD_WITNESS_UNCERTAINTIES: List[Dict[str, Any]] = [
    {"node_id": "mid", "distribution": "normal", "mean": 0.5, "std": 0.2},
    {"node_id": "driver", "distribution": "normal", "mean": 0.5, "std": 0.2},
]


def _withhold_witness_payload(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "graph": _WITHHOLD_WITNESS_GRAPH,
        "options": _WITHHOLD_WITNESS_OPTIONS,
        "goal_node_id": "goal",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "analysis_types": ["comparison"],
        "parameter_uncertainties": _WITHHOLD_WITNESS_UNCERTAINTIES,
        # NOT VACUOUS: the defaults for both of these would hide the defect.
        "include_path_decomposition": True,
        "include_voi": True,
        # NO `factor_correlations` — deliberately absent. This is the clause
        # that makes the P2 assertions non-vacuous.
        # Withhold trigger: a level-framed target the resolver cannot convert.
        "goal_direction": "target",
        "goal_threshold": 0.6,
        "goal_threshold_frame": "level",
    }
    payload.update(overrides)
    return payload


def _withhold_witness(**overrides: Any) -> Any:
    return RobustnessAnalyzerV2().analyze(
        RobustnessRequestV2(**_withhold_witness_payload(**overrides))
    )


def _withheld_warning(response: Any) -> Any:
    """The OBJECTIVE_RANKING_WITHHELD warning, bound by CODE not by position."""
    matches = [w for w in response.inference_warnings if w.code == "OBJECTIVE_RANKING_WITHHELD"]
    assert len(matches) == 1, [w.code for w in response.inference_warnings]
    return matches[0]


class TestWithheldRankingSuppressesPathDecomposition:
    """P1. The sixth field, and it was not gated."""

    def test_the_witness_request_genuinely_withholds(self) -> None:
        """PRECONDITION PINNED IN-TEST (trap 13b).

        Every assertion in this class is about behaviour UNDER withhold. If the
        witness ever stopped withholding — a resolver change, a schema default,
        a graph edit — the absence assertions below would all pass for the wrong
        reason and nothing would go red. This asserts the precondition they rest
        on, so that failure is loud and attributable.
        """
        response = _withhold_witness()
        assert response.objective_ranking.status == "withheld"
        assert response.objective_ranking.withheld_reason == (
            "target_not_resolvable_in_sample_frame"
        )
        assert "OBJECTIVE_RANKING_WITHHELD" in _warning_codes(response)
        # And the phases under test were genuinely ASKED for.
        assert _withhold_witness_payload()["include_path_decomposition"] is True
        assert _withhold_witness_payload()["include_voi"] is True

    def test_a_withheld_ranking_omits_path_decomposition_even_when_requested(self) -> None:
        """⭐ THE P1 ACCEPTANCE TEST.

        RED at 0a570656: `path_decomposition` was PRESENT, naming `modest` as
        the recommended option with a 2-path causal breakdown, on a response
        that had just refused to state a ranking.
        """
        assert _withhold_witness().path_decomposition is None

    def test_the_leak_was_array_order_arbitrary_and_is_gone_in_both_orders(self) -> None:
        """The leaked id was decided by INSERTION ORDER, not by evidence.

        `max()` over an all-zero tally returns the first key. Measured at
        pristine: [modest, aggressive] leaked "modest"; reversing the array
        leaked "aggressive" — the same run, the same evidence, a different
        named winner. Both orders must now be silent.
        """
        forward = _withhold_witness()
        reversed_ = _withhold_witness(options=list(reversed(_WITHHOLD_WITNESS_OPTIONS)))
        assert forward.objective_ranking.status == "withheld"
        assert reversed_.objective_ranking.status == "withheld"
        assert forward.path_decomposition is None
        assert reversed_.path_decomposition is None

    def test_a_computed_ranking_still_receives_its_path_decomposition(self) -> None:
        """⭐ THE DISCRIMINATING TWIN — without it the P1 test above is satisfied
        by deleting the phase outright.

        Same graph, same flag, same uncertainties; only the objective's
        resolvability differs. The block must still be computed, and it must
        name its option BY IDENTITY.
        """
        response = _withhold_witness(
            goal_direction="maximise", goal_threshold=None, goal_threshold_frame=None
        )
        assert response.objective_ranking.status == "computed"
        assert response.path_decomposition is not None
        assert response.path_decomposition.recommended_option_id == "aggressive"
        assert response.path_decomposition.path_count == 2

    def test_the_skipped_phase_is_recorded_not_merely_absent(self) -> None:
        """The siblings record at the skip site; so must this one."""
        manifest = _withheld_warning(_withhold_witness()).detail["suppressed_blocks"]
        assert "path_decomposition" in manifest, manifest


class TestWithholdSuppressionManifestReachesTheWire:
    """P2. The markers were recorded into a list nothing emitted."""

    def test_the_manifest_survives_with_no_declared_correlations(self) -> None:
        """⭐ THE P2 ACCEPTANCE TEST.

        RED at 0a570656: the only carrier was `correlation_model.suppressed_
        attributions`, and `correlation_model` is None on any request that
        declares no `factor_correlations` — so all three markers were dropped.

        The "no correlations" clause is asserted here rather than assumed: a
        version of this test with correlations declared passes at pristine and
        proves nothing.
        """
        response = _withhold_witness()
        assert response.correlation_model is None, (
            "precondition: this request declares no factor_correlations, so the "
            "old carrier is absent — that is the whole point of the test"
        )

        detail = _withheld_warning(response).detail
        manifest = detail["suppressed_blocks"]
        assert set(manifest) == {
            "conditional_winners",
            "p_win_sensitivity",
            "path_decomposition",
        }, manifest
        assert detail["suppression_reason"] == "no_recommendation_to_describe"

    def test_a_computed_ranking_records_no_withhold_suppression(self) -> None:
        """The discriminating twin: the manifest tracks the WITHHOLD, not the
        request shape. Under a computed ranking there is no withhold warning at
        all, so there is nothing to carry a manifest.
        """
        response = _withhold_witness(
            goal_direction="maximise", goal_threshold=None, goal_threshold_frame=None
        )
        assert "OBJECTIVE_RANKING_WITHHELD" not in _warning_codes(response)

    def test_correlation_remains_the_carrier_for_correlation_driven_suppression(
        self,
    ) -> None:
        """⭐ THE ANTI-REGRESSION TWIN. The withhold manifest moved to its own
        channel; the correlation manifest must be UNCHANGED for the suppressions
        that are genuinely correlation's.

        Without this, routing the withhold markers away could have emptied the
        correlation block and nothing would have gone red.
        """
        graph = {
            "nodes": [n.copy() for n in _WITHHOLD_WITNESS_GRAPH["nodes"]],
            "edges": _WITHHOLD_WITNESS_GRAPH["edges"],
        }
        # Restore the goal baseline so the ranking is COMPUTED, not withheld:
        # this arm isolates correlation as the only reason for suppression.
        graph["nodes"][2] = {
            "id": "goal",
            "kind": "outcome",
            "label": "Goal",
            "observed_state": {"value": 0.5, "baseline": 0.5},
        }
        response = RobustnessAnalyzerV2().analyze(
            RobustnessRequestV2(
                **_withhold_witness_payload(
                    graph=graph,
                    goal_direction="maximise",
                    goal_threshold=None,
                    goal_threshold_frame=None,
                    factor_correlations=[{"factor_a": "mid", "factor_b": "driver", "rho": 0.5}],
                )
            )
        )
        assert response.objective_ranking.status == "computed"
        assert response.correlation_model is not None
        assert response.correlation_model.active is True
        assert set(response.correlation_model.suppressed_attributions) == {
            "conditional_winners",
            "p_win_sensitivity",
        }


# ==========================================================================
# THE WIRE. Everything above is measured in-process on the analyzer's V1
# envelope; the leak the review found reaches a USER through the V2 response
# builder (`src/api/robustness.py`, the `PathDecompositionV2` passthrough),
# and that layer had ZERO coverage for any of this — which is precisely how a
# field gated only on "non-null" came to restate a withheld ranking.
#
# These go through the REAL endpoint and assert on raw JSON, so an analyzer
# that suppresses correctly but a builder that re-derives would still be red.
# ==========================================================================


_V2_ENDPOINT = "/api/v1/robustness/analyze/v2"
_V2_HEADERS = {"X-ISL-Response-Version": "2"}


def _post_witness(**overrides: Any) -> Dict[str, Any]:
    """POST the witness request through the REAL endpoint and return raw JSON.

    Built from the SAME payload builder as the in-process tests, so the two
    layers cannot silently diverge on what was asked for.
    """
    from fastapi.testclient import TestClient

    from src.api.main import app

    payload = _withhold_witness_payload(**overrides)
    payload["request_id"] = "withhold-side-channel-wire-test"
    payload = {k: v for k, v in payload.items() if v is not None}
    response = TestClient(app).post(_V2_ENDPOINT, json=payload, headers=_V2_HEADERS)
    assert response.status_code == 200, (response.status_code, response.text[:400])
    return response.json()  # type: ignore[no-any-return]


class TestTheWireCarriesNeitherTheLeakNorASilentOmission:
    def test_the_v2_envelope_omits_path_decomposition_under_a_withheld_ranking(
        self,
    ) -> None:
        body = _post_witness()
        assert body["objective_ranking"]["status"] == "withheld"
        assert body.get("path_decomposition") is None, body.get("path_decomposition")

    def test_the_v2_envelope_still_carries_it_under_a_computed_ranking(self) -> None:
        """Discriminating twin at the wire, bound by option IDENTITY."""
        body = _post_witness(
            goal_direction="maximise", goal_threshold=None, goal_threshold_frame=None
        )
        assert body["objective_ranking"]["status"] == "computed"
        assert body["path_decomposition"]["recommended_option_id"] == "aggressive"

    def test_the_suppression_manifest_is_readable_on_the_wire(self) -> None:
        """⭐ P2 AT THE WIRE — the assertion the old carrier could not satisfy.

        `correlation_model` is absent from this response (no declared
        correlations), which is asserted rather than assumed. The manifest must
        be readable anyway.
        """
        body = _post_witness()
        assert body.get("correlation_model") is None

        withheld = [
            w for w in body["inference_warnings"] if w["code"] == "OBJECTIVE_RANKING_WITHHELD"
        ]
        assert len(withheld) == 1, [w["code"] for w in body["inference_warnings"]]
        detail = withheld[0]["detail"]
        assert set(detail["suppressed_blocks"]) == {
            "conditional_winners",
            "p_win_sensitivity",
            "path_decomposition",
        }, detail["suppressed_blocks"]
        assert detail["suppression_reason"] == "no_recommendation_to_describe"
