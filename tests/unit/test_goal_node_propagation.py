"""
Cluster-2 goal-node propagation (Track S Phase 0 credibility floor).

Two test groups:

1. TestPinnedTipBehaviour — characterisation fixtures pinning the behaviour of
   staging tip 209d924 for (a) root goal, (b) non-root goal with parents,
   (c) constraint target with defaulted base. These passed BEFORE the
   Cluster-2 change and must keep passing: the lane changes NO numeric
   output, only disclosure.

2. TestGoalNodeDisclosures — RED-first tests for the honest-disclosure
   warnings introduced by the Cluster-2 lane:
   - GOAL_OBSERVED_VALUE_UNUSED: a non-root goal's observed_state.value is
     not used as a base (doctrine B: the goal's distribution is the
     forward-propagated composition of its parents) — previously dropped
     silently.
   - GOAL_PU_BASE_ADDITIVE: a ParameterUncertainty entry on a non-root goal
     draws a base that is ADDED to parent propagation (it does not pin the
     goal's value) — previously undisclosed double-count semantics.
   - GOAL_ANCESTOR_DATA_GAP (warning + critique): the goal's propagated
     distribution rests partly on root ancestors that defaulted to 0.0 —
     the honest "insufficient data" disclosure at goal level.
   - GOAL_NODE_ROOT_STATIC: a root goal with no ParameterUncertainty and no
     epsilon noise is a constant — options cannot differ through it unless
     they intervene on it.
   - CONSTRAINT_NODE_DEFAULT_BASE detail/message honesty: ancestor-supported
     non-objective constraint targets stop claiming "may be unreliable";
     detail discloses base semantics and any ancestor data gap.
"""

import numpy as np
import pytest

from src.models.robustness_v2 import (
    EdgeV2,
    GoalConstraint,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    ParameterUncertainty,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


# =============================================================================
# Graph builders
# =============================================================================


def _edge(from_id: str, to_id: str, mean: float, std: float = 0.05) -> EdgeV2:
    return EdgeV2(
        **{"from": from_id, "to": to_id},
        exists_probability=1.0,
        strength=StrengthDistribution(mean=mean, std=std),
    )


def _two_options(lever: str = "lever") -> list:
    return [
        InterventionOption(id="o1", label="A", interventions={lever: 0.4}),
        InterventionOption(id="o2", label="B", interventions={lever: 0.8}),
    ]


def make_root_goal_graph() -> GraphV2:
    """Goal node has NO parents; carries an observed value."""
    return GraphV2(
        nodes=[
            NodeV2(id="lever", kind="factor", label="Lever"),
            NodeV2(
                id="goal_mrr",
                kind="outcome",
                label="MRR",
                observed_state=ObservedState(value=0.6),
            ),
        ],
        edges=[],
    )


def make_nonroot_goal_graph(goal_observed=None) -> GraphV2:
    """lever -> mid -> goal_mrr; goal optionally carries an observed value."""
    goal_kwargs = {}
    if goal_observed is not None:
        goal_kwargs["observed_state"] = ObservedState(value=goal_observed)
    return GraphV2(
        nodes=[
            NodeV2(
                id="lever",
                kind="factor",
                label="Lever",
                observed_state=ObservedState(value=0.5),
            ),
            NodeV2(id="mid", kind="chance", label="Mid"),
            NodeV2(id="goal_mrr", kind="outcome", label="MRR", **goal_kwargs),
        ],
        edges=[_edge("lever", "mid", 0.8), _edge("mid", "goal_mrr", 0.7)],
    )


def make_gap_goal_graph() -> GraphV2:
    """lever -> goal, market -> goal; 'market' root carries NO data."""
    return GraphV2(
        nodes=[
            NodeV2(id="lever", kind="factor", label="Lever"),
            NodeV2(id="market", kind="factor", label="Market"),
            NodeV2(id="goal_mrr", kind="outcome", label="MRR"),
        ],
        edges=[_edge("lever", "goal_mrr", 0.6), _edge("market", "goal_mrr", 0.4)],
    )


def make_constraint_graph() -> GraphV2:
    """lever -> cost, lever -> goal_mrr; constraint targets cost + goal."""
    return GraphV2(
        nodes=[
            NodeV2(id="lever", kind="factor", label="Lever"),
            NodeV2(id="cost", kind="outcome", label="Cost"),
            NodeV2(id="goal_mrr", kind="outcome", label="MRR"),
        ],
        edges=[_edge("lever", "cost", 0.4), _edge("lever", "goal_mrr", 0.6)],
    )


def _warnings(response, code):
    return [w for w in response.inference_warnings if w.code == code]


def _critiques(response, code):
    return [c for c in response.critiques if c.code == code]


# =============================================================================
# 1. Pinned tip-209d924 behaviour (characterisation — numeric outputs must
#    NOT change; the Cluster-2 lane is disclosure-only)
# =============================================================================


class TestPinnedTipBehaviour:
    def test_a_root_goal_is_constant_observed_value(self):
        """(a) Root goal: every sample equals observed_state.value; zero
        variance; degenerate critiques fire."""
        request = RobustnessRequestV2(
            graph=make_root_goal_graph(),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=200,
            seed=42,
            goal_threshold=0.5,
        )
        response = RobustnessAnalyzerV2().analyze(request)

        for result in response.results:
            assert result.outcome_distribution.mean == pytest.approx(0.6)
            assert result.outcome_distribution.std == pytest.approx(0.0)
            assert result.probability_of_goal == pytest.approx(1.0)
        assert len(_critiques(response, "DEGENERATE_OPTION_ZERO_VARIANCE")) == 2
        assert len(_critiques(response, "HIGH_TIE_RATE")) == 1

    def test_b_nonroot_goal_is_propagated_composition(self):
        """(b) Non-root goal: distribution IS the forward-propagated
        composition of its parents (lever x 0.8 x 0.7)."""
        request = RobustnessRequestV2(
            graph=make_nonroot_goal_graph(goal_observed=None),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=200,
            seed=42,
            goal_threshold=0.3,
        )
        response = RobustnessAnalyzerV2().analyze(request)

        by_id = {r.option_id: r for r in response.results}
        # o2 intervenes lever=0.8 -> goal mean ~ 0.8*0.8*0.7 = 0.448
        # (auto-scaled noise is zero-mean so the mean is preserved ~sqrt(2) spread)
        assert by_id["o2"].outcome_distribution.mean == pytest.approx(0.448, abs=0.05)
        assert by_id["o1"].outcome_distribution.mean == pytest.approx(0.224, abs=0.05)

    def test_b_nonroot_goal_observed_value_does_not_change_distribution(self):
        """(b) The goal's own observed_state.value is NOT a base for a
        non-root goal (doctrine B) — samples are identical with and without
        it. Pinned so the disclosure lane provably changes no numbers."""
        results = {}
        for tag, obs in (("without", None), ("with", 0.6)):
            request = RobustnessRequestV2(
                graph=make_nonroot_goal_graph(goal_observed=obs),
                options=_two_options(),
                goal_node_id="goal_mrr",
                n_samples=200,
                seed=42,
                goal_threshold=0.3,
            )
            results[tag] = RobustnessAnalyzerV2().analyze(request)

        for r_without, r_with in zip(results["without"].results, results["with"].results):
            assert np.allclose(
                r_without.outcome_distribution.samples,
                r_with.outcome_distribution.samples,
            ), "observed_state.value on a non-root goal must not change samples"

    def test_e_nonroot_goal_pu_base_is_additive(self):
        """(e) A ParameterUncertainty entry on a NON-ROOT goal draws a base
        (mean = observed_state.value) that is ADDED to parent propagation:
        o2 mean ~ 0.6 + 0.448 ~ 1.05, not pinned to 0.6. Pinned so any future
        semantic change is a conscious, RED-first decision."""
        request = RobustnessRequestV2(
            graph=make_nonroot_goal_graph(goal_observed=0.6),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=200,
            seed=42,
            goal_threshold=0.3,
            parameter_uncertainties=[
                ParameterUncertainty(node_id="goal_mrr", distribution="normal", std=0.05)
            ],
        )
        response = RobustnessAnalyzerV2().analyze(request)
        by_id = {r.option_id: r for r in response.results}
        assert by_id["o2"].outcome_distribution.mean == pytest.approx(0.6 + 0.448, abs=0.08)
        assert by_id["o1"].outcome_distribution.mean == pytest.approx(0.6 + 0.224, abs=0.08)

    def test_c_constraint_targets_fire_default_base_variants(self):
        """(c) Both CONSTRAINT_NODE_DEFAULT_BASE variants fire: the
        non-objective target ('cost') and the objective target ('goal_mrr',
        doctrine-B wording, ROADMAP 1.26b)."""
        request = RobustnessRequestV2(
            graph=make_constraint_graph(),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=200,
            seed=42,
            goal_constraints=[
                GoalConstraint(node_id="cost", operator="<=", value=0.5),
                GoalConstraint(node_id="goal_mrr", operator=">=", value=0.3),
            ],
        )
        response = RobustnessAnalyzerV2().analyze(request)

        warnings = _warnings(response, "CONSTRAINT_NODE_DEFAULT_BASE")
        assert {w.detail.get("node_id") for w in warnings} == {"cost", "goal_mrr"}
        by_node = {w.detail["node_id"]: w for w in warnings}
        objective_msg = str(by_node["goal_mrr"].detail.get("message", ""))
        assert "objective node" in objective_msg
        assert "may be unreliable" not in objective_msg
        assert len(_critiques(response, "CONSTRAINT_NODE_DEFAULT_BASE")) == 2


# =============================================================================
# 2. Cluster-2 disclosures (RED before the lane's change)
# =============================================================================


class TestGoalNodeDisclosures:
    def test_nonroot_goal_observed_value_unused_disclosed(self):
        """A non-root goal carrying observed_state.value (and no PU) gets a
        GOAL_OBSERVED_VALUE_UNUSED warning — the silent drop is disclosed."""
        request = RobustnessRequestV2(
            graph=make_nonroot_goal_graph(goal_observed=0.6),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
        )
        response = RobustnessAnalyzerV2().analyze(request)

        warnings = _warnings(response, "GOAL_OBSERVED_VALUE_UNUSED")
        assert len(warnings) == 1
        w = warnings[0]
        assert w.field == "nodes[goal_mrr].observed_state.value"
        assert w.detail["node_id"] == "goal_mrr"
        assert w.detail["observed_value"] == pytest.approx(0.6)
        message = str(w.detail.get("message", ""))
        assert "forward-propagated" in message
        assert "not used" in message.lower()

    def test_nonroot_goal_without_observed_value_no_unused_warning(self):
        """No observed value on the goal -> nothing was dropped -> no warning."""
        request = RobustnessRequestV2(
            graph=make_nonroot_goal_graph(goal_observed=None),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
        )
        response = RobustnessAnalyzerV2().analyze(request)
        assert _warnings(response, "GOAL_OBSERVED_VALUE_UNUSED") == []

    def test_nonroot_goal_pu_additive_semantics_disclosed(self):
        """PU on a non-root goal -> GOAL_PU_BASE_ADDITIVE disclosure (the
        sampled base ADDS to parent propagation; it does not pin the goal)."""
        request = RobustnessRequestV2(
            graph=make_nonroot_goal_graph(goal_observed=0.6),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
            parameter_uncertainties=[
                ParameterUncertainty(node_id="goal_mrr", distribution="normal", std=0.05)
            ],
        )
        response = RobustnessAnalyzerV2().analyze(request)

        warnings = _warnings(response, "GOAL_PU_BASE_ADDITIVE")
        assert len(warnings) == 1
        w = warnings[0]
        assert w.field == "parameter_uncertainties[goal_mrr]"
        assert w.detail["node_id"] == "goal_mrr"
        message = str(w.detail.get("message", ""))
        assert "added" in message.lower()
        assert "pin" in message.lower()
        # With PU the observed value IS consumed (as the PU mean) — the
        # UNUSED warning must not also fire.
        assert _warnings(response, "GOAL_OBSERVED_VALUE_UNUSED") == []

    def test_goal_ancestor_data_gap_disclosed(self):
        """A defaulted root ancestor ('market') with an intervention-free
        path to the goal -> GOAL_ANCESTOR_DATA_GAP warning + critique naming
        the unsupported roots."""
        request = RobustnessRequestV2(
            graph=make_gap_goal_graph(),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
            goal_threshold=0.3,
        )
        response = RobustnessAnalyzerV2().analyze(request)

        warnings = _warnings(response, "GOAL_ANCESTOR_DATA_GAP")
        assert len(warnings) == 1
        w = warnings[0]
        assert w.field == "nodes[goal_mrr]"
        assert w.detail["node_id"] == "goal_mrr"
        assert w.detail["unsupported_root_ancestors"] == ["market"]

        critiques = _critiques(response, "GOAL_ANCESTOR_DATA_GAP")
        assert len(critiques) == 1
        assert critiques[0].severity == "warning"
        assert critiques[0].affected_node_ids == ["goal_mrr"]
        assert "market" in critiques[0].message

    def test_no_ancestor_gap_when_roots_supported(self):
        """All root ancestors carry data (or are intervened by every option)
        -> no gap warning: the propagated composition is data-supported."""
        request = RobustnessRequestV2(
            graph=make_nonroot_goal_graph(goal_observed=None),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
        )
        response = RobustnessAnalyzerV2().analyze(request)
        assert _warnings(response, "GOAL_ANCESTOR_DATA_GAP") == []
        assert _critiques(response, "GOAL_ANCESTOR_DATA_GAP") == []

    def test_no_ancestor_gap_when_path_blocked_by_interventions(self):
        """A defaulted root whose ONLY path to the goal passes through a node
        every option intervenes on cannot influence the goal's samples ->
        no goal-level gap (the generic ROOT_NODE_DEFAULT_VALUE still fires)."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="market", kind="factor", label="Market"),
                NodeV2(id="mid", kind="chance", label="Mid"),
                NodeV2(id="goal_mrr", kind="outcome", label="MRR"),
            ],
            edges=[_edge("market", "mid", 0.8), _edge("mid", "goal_mrr", 0.7)],
        )
        request = RobustnessRequestV2(
            graph=graph,
            options=[
                InterventionOption(id="o1", label="A", interventions={"mid": 0.4}),
                InterventionOption(id="o2", label="B", interventions={"mid": 0.8}),
            ],
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
        )
        response = RobustnessAnalyzerV2().analyze(request)
        assert _warnings(response, "GOAL_ANCESTOR_DATA_GAP") == []
        assert len(_warnings(response, "ROOT_NODE_DEFAULT_VALUE")) == 1

    def test_root_goal_static_disclosed(self):
        """A root goal with no PU and no epsilon noise is a constant — the
        GOAL_NODE_ROOT_STATIC disclosure names the base actually used."""
        request = RobustnessRequestV2(
            graph=make_root_goal_graph(),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
        )
        response = RobustnessAnalyzerV2().analyze(request)

        warnings = _warnings(response, "GOAL_NODE_ROOT_STATIC")
        assert len(warnings) == 1
        w = warnings[0]
        assert w.field == "nodes[goal_mrr]"
        assert w.detail["node_id"] == "goal_mrr"
        assert w.detail["base_value"] == pytest.approx(0.6)
        assert w.detail["value_defaulted"] is False

    def test_nonroot_goal_no_root_static_warning(self):
        """Non-root goals must not get the root-static disclosure."""
        request = RobustnessRequestV2(
            graph=make_nonroot_goal_graph(goal_observed=None),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
        )
        response = RobustnessAnalyzerV2().analyze(request)
        assert _warnings(response, "GOAL_NODE_ROOT_STATIC") == []

    def test_constraint_default_base_supported_message_is_honest(self):
        """Non-objective constraint target whose root ancestors ALL carry
        data (here: 'lever' is intervened by every option): the samples are a
        fully-supported propagated composition — the message must stop
        claiming 'may be unreliable' and the detail must disclose the
        semantics. Code, count, and severity are unchanged (consumers key on
        code — ROADMAP 1.26b precedent)."""
        request = RobustnessRequestV2(
            graph=make_constraint_graph(),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
            goal_constraints=[GoalConstraint(node_id="cost", operator="<=", value=0.5)],
        )
        response = RobustnessAnalyzerV2().analyze(request)

        warnings = _warnings(response, "CONSTRAINT_NODE_DEFAULT_BASE")
        assert len(warnings) == 1
        w = warnings[0]
        assert w.detail["node_id"] == "cost"
        assert w.detail["ancestor_data_gap"] == []
        message = str(w.detail.get("message", ""))
        assert "no ParameterUncertainty" in message
        assert "may be unreliable" not in message
        assert "propagated" in message
        # 1.26b guard: still must not claim the objective/doctrine-B framing
        assert "objective node" not in message
        assert "modelled outcome distribution" not in message

        critiques = _critiques(response, "CONSTRAINT_NODE_DEFAULT_BASE")
        assert len(critiques) == 1
        assert critiques[0].severity == "warning"
        assert "may be unreliable" not in critiques[0].message
        # The old suggestion recommended a point_mass PU "so its
        # observed_state.value is used as the sampling base" — on a non-root
        # node the PU base is ADDED to parent propagation (see
        # test_e_nonroot_goal_pu_base_is_additive), so that suggestion was a
        # double-count trap. It must be gone.
        assert "used as the sampling base" not in (critiques[0].suggestion or "")

    def test_constraint_default_base_gap_message_keeps_caution(self):
        """Non-objective constraint target with a genuinely unsupported root
        ancestor keeps the honest data-gap caution and names the roots."""
        graph = GraphV2(
            nodes=[
                NodeV2(id="lever", kind="factor", label="Lever"),
                NodeV2(id="market", kind="factor", label="Market"),
                NodeV2(id="cost", kind="outcome", label="Cost"),
                NodeV2(id="goal_mrr", kind="outcome", label="MRR"),
            ],
            edges=[
                _edge("lever", "cost", 0.4),
                _edge("market", "cost", 0.5),
                _edge("lever", "goal_mrr", 0.6),
            ],
        )
        request = RobustnessRequestV2(
            graph=graph,
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
            goal_constraints=[GoalConstraint(node_id="cost", operator="<=", value=0.5)],
        )
        response = RobustnessAnalyzerV2().analyze(request)

        warnings = _warnings(response, "CONSTRAINT_NODE_DEFAULT_BASE")
        assert len(warnings) == 1
        w = warnings[0]
        assert w.detail["node_id"] == "cost"
        assert w.detail["ancestor_data_gap"] == ["market"]
        message = str(w.detail.get("message", ""))
        assert "may be unreliable" in message
        assert "market" in message

    def test_objective_constraint_detail_gains_gap_field(self):
        """The doctrine-B objective variant keeps its 1.26b message but the
        detail now discloses the ancestor-gap machine field."""
        request = RobustnessRequestV2(
            graph=make_constraint_graph(),
            options=_two_options(),
            goal_node_id="goal_mrr",
            n_samples=100,
            seed=42,
            goal_constraints=[GoalConstraint(node_id="goal_mrr", operator=">=", value=0.3)],
        )
        response = RobustnessAnalyzerV2().analyze(request)

        warnings = _warnings(response, "CONSTRAINT_NODE_DEFAULT_BASE")
        assert len(warnings) == 1
        w = warnings[0]
        assert w.detail["node_id"] == "goal_mrr"
        assert w.detail["ancestor_data_gap"] == []
        message = str(w.detail.get("message", ""))
        assert "objective node" in message
        assert "may be unreliable" not in message
