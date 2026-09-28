"""Proposal (3) — a goal that states no level today is anchored on its EVALUATED identity.

AIQ #72 5876233408 ruled YES with four conditions and four rows:
  1. evaluated on THIS run only (absent -> ``missing_goal_baseline``, as today);
  2. authorship = the WEAKEST operand's (any Olumi estimate -> ``estimate_only``);
  3. a stated goal level wins, and when both exist and differ both are named;
  4. the anchor is in USER units from the operands' levels, framed by the goal's own frame.

Served shape (AIQ R3-A1b, PLoT 101faf2 capture A1t): Paul's pricing brief gives MRR's target
(£100k of a £125k cap) but not today's MRR, so the goal reaches ISL with no observed_state, its
identity is evaluated from its inputs (``level_source: identity_inputs``) and
``probability_of_goal`` was withheld as ``missing_goal_baseline``. Today from its inputs:
£49 x 1,500 subscribers + £1,000 = £74,500, i.e. 0.596 of the £125,000 frame.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional

import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import RobustnessRequestV2
from tests.unit.test_r3_identity_evaluation import FRAMES, MRR, OTHER, SUBS, v2_body, wire

TODAY_GBP = 49.0 * 1_500.0 + 1_000.0  # £74,500
TODAY = TODAY_GBP / 125_000.0  # 0.596
KEEP = "keep_current_49_price"  # sets price to today's £49: no change, so its level IS today's
CODE = "GOAL_LEVEL_FROM_IDENTITY_INPUTS"
N_SAMPLES = 1_000


def unstated(
    d: Optional[Dict[str, Any]] = None,
    *,
    users: Iterable[str] = (),
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """The served R3-A1b shape: the goal carries no observed_state. ``users`` restates those
    operands' levels as the user's (brief_extraction); served, subscribers and other MRR
    growth are Olumi's (cee_inference)."""
    d = copy.deepcopy(wire() if d is None else d)
    nodes = {n["id"]: n for n in d["graph"]["nodes"]}
    nodes[MRR].pop("observed_state", None)
    for node_id in users:
        nodes[node_id]["observed_state"]["source"] = "brief_extraction"
    if threshold is not None:
        d["goal_threshold"] = threshold
    d["n_samples"] = N_SAMPLES
    return d


def analysed(d: Dict[str, Any]) -> Any:
    return rav2.RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(d))


def p_goal(response: Any) -> Dict[str, Optional[float]]:
    return {r.option_id: r.probability_of_goal for r in response.results}


def coded(response: Any, code: str) -> List[Any]:
    return [w for w in response.inference_warnings if w.code == code]


def refusal_reasons(response: Any) -> List[str]:
    return [
        w.detail.get("reason")
        for w in response.inference_warnings
        if w.code == "GOAL_THRESHOLD_NOT_CONVERTIBLE"
    ]


class TestRowA_EveryOperandTheUsers:
    def test_probability_of_goal_is_present_and_the_base_is_the_users(self):
        response = analysed(unstated(users=(SUBS, OTHER)))
        assert all(p is not None for p in p_goal(response).values()), p_goal(response)
        assert "missing_goal_baseline" not in refusal_reasons(response)
        (warning,) = coded(response, CODE)
        assert warning.severity == "warning"  # PLoT hides 'info'
        assert warning.field == f"nodes[{MRR}].nonlinear_identity"
        detail = warning.detail
        assert detail["goal_node_id"] == detail["node_id"] == MRR
        assert detail["node_label"] == "MRR"
        assert detail["level_source"] == "identity_inputs"
        assert detail["level_author"] == "user"
        assert detail["frame_verdict"] == "scored"
        assert detail["estimated_operand_ids"] == []
        assert detail["today_level"] == pytest.approx(TODAY_GBP, abs=1e-6)
        assert detail["frame"] == FRAMES[MRR]["frame"]
        assert detail["goal_baseline"] == pytest.approx(TODAY, abs=1e-12)
        assert detail["message"] == (
            "MRR has no level stated for today, so the chance of reaching the goal is measured "
            "from the level its inputs give today: 74,500.00 in its own units; every input's "
            "level today is the user's, so this is the user's base."
        )

    def test_the_disclosure_agrees_with_the_identity_disclosure(self):
        response = analysed(unstated(users=(SUBS, OTHER)))
        (entry,) = [e for e in response.identity_evaluations if e.node_id == MRR]
        assert entry.evaluated and entry.level_source == "identity_inputs"


class TestUnitsCondition4:
    """Today is £74,500 exactly: the operands' levels x their own frames, combined, then framed by
    the goal's £125,000. KEEP changes nothing, so its level on every draw is exactly that anchor,
    and a threshold a hair either side of 0.596 flips its P(goal) from 1 to 0. A normalised
    product (0.245 x 0.15 + 0.02 = 0.057), a dropped addend (0.588), or no anchor all give 0 on
    both sides."""

    @pytest.mark.parametrize("threshold, expected", [(TODAY - 1e-4, 1.0), (TODAY + 1e-4, 0.0)])
    def test_the_unchanged_option_sits_exactly_at_todays_level(self, threshold, expected):
        response = analysed(unstated(users=(SUBS, OTHER), threshold=threshold))
        assert p_goal(response)[KEEP] == expected

    def test_the_anchor_itself(self):
        request = RobustnessRequestV2.model_validate(unstated())
        anchor = rav2.identity_level_anchor(request, MRR)
        assert anchor is not None
        assert anchor.level == pytest.approx(TODAY_GBP, abs=1e-6)
        assert anchor.frame == FRAMES[MRR]["frame"]


class TestRowB_TheWeakestOperandDecides:
    def test_served_both_estimated_is_olumis_estimate(self):
        response = analysed(unstated())
        assert all(p is not None for p in p_goal(response).values())
        (warning,) = coded(response, CODE)
        assert warning.detail["level_author"] == "olumi"
        assert warning.detail["frame_verdict"] == "estimate_only"
        assert warning.detail["estimated_operand_ids"] == [SUBS, OTHER]
        assert warning.detail["message"].endswith(
            "are Olumi's estimates, so this is Olumi's estimate of today's MRR, not the user's."
        )

    def test_one_estimated_operand_is_enough(self):
        (warning,) = coded(analysed(unstated(users=(OTHER,))), CODE)
        assert warning.detail["frame_verdict"] == "estimate_only"
        assert warning.detail["estimated_operand_ids"] == [SUBS]
        assert "is Olumi's estimate, so this is Olumi's estimate of today's MRR" in (
            warning.detail["message"]
        )

    def test_the_v2_wire_carries_it(self):
        body = v2_body(unstated())
        (warning,) = [w for w in body["inference_warnings"] if w["code"] == CODE]
        assert warning["detail"]["frame_verdict"] == "estimate_only"
        assert warning["detail"]["today_level"] == pytest.approx(TODAY_GBP, abs=1e-6)
        assert all(o.get("probability_of_goal") is not None for o in body["options"])


class TestRowC_NoEvaluatedIdentityNoAnchor:
    def test_no_identity_is_missing_goal_baseline_as_today(self):
        response = analysed(unstated(wire(identity=None)))
        assert refusal_reasons(response) == ["missing_goal_baseline"]
        assert all(p is None for p in p_goal(response).values())
        assert coded(response, CODE) == []

    def test_a_withheld_identity_gives_no_anchor_and_the_run_is_blocked(self):
        frames = {k: v for k, v in FRAMES.items() if k != SUBS}  # identity_frame_missing
        d = unstated(wire(frames=frames))
        request = RobustnessRequestV2.model_validate(d)
        assert rav2.identity_level_anchor(request, MRR) is None
        with pytest.raises(rav2.IdentityNotEvaluatedError):
            analysed(d)


class TestRowD_AStatedLevelWins:
    """Served, the brief states £75,000 (0.6); the inputs give £74,500 (0.67% apart, inside the
    5% reconciliation). A threshold between the two tells which one the goal is measured from."""

    def test_the_stated_level_anchors_and_both_are_named(self):
        d = wire()
        d["goal_threshold"] = 0.598
        d["n_samples"] = N_SAMPLES
        response = analysed(d)
        assert p_goal(response)[KEEP] == 1.0  # from £75,000; the inputs' £74,500 would give 0
        assert coded(response, CODE) == []
        (entry,) = [e for e in response.identity_evaluations if e.node_id == MRR]
        assert entry.level_source == "stated_level"
        assert entry.reconciliation.stated == pytest.approx(75_000.0)
        assert entry.reconciliation.reconstructed == pytest.approx(TODAY_GBP)

    def test_a_stated_value_without_a_baseline_is_not_replaced_by_the_inputs(self):
        d = unstated()
        mrr = next(n for n in d["graph"]["nodes"] if n["id"] == MRR)
        mrr["observed_state"] = {"value": 0.6, "source": "brief_extraction"}
        response = analysed(d)
        assert refusal_reasons(response) == ["missing_goal_baseline"]
        assert coded(response, CODE) == []

    def test_beyond_the_tolerance_the_run_names_both_and_is_blocked(self):
        d = wire()
        mrr = next(n for n in d["graph"]["nodes"] if n["id"] == MRR)
        mrr["observed_state"]["baseline"] = mrr["observed_state"]["value"] = 0.7  # £87,500
        with pytest.raises(rav2.IdentityNotEvaluatedError, match=r"74,500\.00.*87,500\.00"):
            analysed(d)


class TestTheLimitChannelSharesTheRule:
    """The rules are one implementation for both channels: a LEVEL limit on the same goal is
    scored from the same anchor, and its existing frame_verdict carries whose base it is."""

    @staticmethod
    def limit(d: Dict[str, Any], value: float) -> Dict[str, Any]:
        d["goal_constraints"] = [
            {
                "constraint_id": "mrr-floor",
                "node_id": MRR,
                "operator": ">=",
                "value": value,
                "value_frame": "level",
            }
        ]
        return d

    @pytest.mark.parametrize(
        "users, verdict", [((SUBS, OTHER), "scored"), ((), "estimate_only")]
    )
    def test_a_level_limit_is_scored_from_the_anchor(self, users, verdict):
        for value, expected in ((TODAY - 1e-4, 1.0), (TODAY + 1e-4, 0.0)):
            response = analysed(self.limit(unstated(users=users), value))
            keep = next(r for r in response.results if r.option_id == KEEP)
            (result,) = keep.constraint_analysis.constraints
            assert result.prob_satisfied == expected
            assert result.frame_verdict == verdict
