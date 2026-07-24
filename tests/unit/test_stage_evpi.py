"""S3 (A3 VOI honesty, D-23.8) — honest per-stage sequential EVPI.

`stage_evpi = E_C[max_a Q(a | C)] − max_a E_C[Q(a)]` (outcome units) REPLACES the
removed `optimal_waiting_value` (a discount × sqrt(Σvar) dispersion heuristic that
was NOT an option value). Both legs are exact reads of the backward-induction tree.

Covers: hand-derived pins on 2 tree shapes (incl. the theorem-0 dominance case),
the degenerate-chance collapse to 0, None for a decision-less stage, an independent
E[max]−max E cross-check, the risk-adjustment posture, and the removal of the old key.
"""

import time

import pytest

from src.models.requests import (
    DecisionStage,
    SequentialAnalysisRequest,
    SequentialGraph,
    SequentialGraphEdge,
    SequentialGraphNode,
)
from pydantic import ValidationError

from src.models.responses import SequentialAnalysisResponse, StageAnalysis
from src.services.sequential_decision import SequentialDecisionEngine


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def two_stage(success=100000, failure=-20000, p_fav=0.6, risk="neutral", df=0.95):
    """invest -> market(chance) -> {success, failure}; wait -> no_invest.
    The A3 risk_neutral honesty fixture (test_a3_seq_honesty._base_seq_request)."""
    g = SequentialGraph(
        nodes=[
            SequentialGraphNode(id="invest", type="decision", label="Invest"),
            SequentialGraphNode(id="market", type="chance", label="Market"),
            SequentialGraphNode(id="success", type="terminal", label="S", payoff=success),
            SequentialGraphNode(id="failure", type="terminal", label="F", payoff=failure),
            SequentialGraphNode(id="no_invest", type="terminal", label="N", payoff=0),
        ],
        edges=[
            SequentialGraphEdge(from_node="invest", to_node="market", action="invest", immediate_payoff=-10000),
            SequentialGraphEdge(from_node="invest", to_node="no_invest", action="wait"),
            SequentialGraphEdge(from_node="market", to_node="success", outcome="fav", probability=p_fav),
            SequentialGraphEdge(from_node="market", to_node="failure", outcome="unf", probability=round(1 - p_fav, 10)),
        ],
        stage_assignments={"invest": 0, "market": 1, "success": 2, "failure": 2, "no_invest": 1},
    )
    stages = [
        DecisionStage(stage_index=0, stage_label="Invest", decision_nodes=["invest"]),
        DecisionStage(stage_index=1, stage_label="Market", decision_nodes=[], resolution_nodes=["market"]),
        DecisionStage(stage_index=2, stage_label="Terminal", decision_nodes=[]),
    ]
    return SequentialAnalysisRequest(graph=g, stages=stages, discount_factor=df, risk_tolerance=risk)


def three_stage(df=0.9, risk="neutral"):
    """launch -> market(chance) -> {positive->pricing(decision)->terminals, negative->exit}.
    The stage-1 pricing decision faces NO downstream chance -> its stage_evpi == 0."""
    g = SequentialGraph(
        nodes=[
            SequentialGraphNode(id="launch", type="decision", label="Launch"),
            SequentialGraphNode(id="market", type="chance", label="Market"),
            SequentialGraphNode(id="pricing", type="decision", label="Pricing"),
            SequentialGraphNode(id="high", type="terminal", label="High", payoff=150000),
            SequentialGraphNode(id="medium", type="terminal", label="Medium", payoff=50000),
            SequentialGraphNode(id="exit", type="terminal", label="Exit", payoff=-30000),
            SequentialGraphNode(id="no_launch", type="terminal", label="None", payoff=0),
        ],
        edges=[
            SequentialGraphEdge(from_node="launch", to_node="market", action="launch", immediate_payoff=-50000),
            SequentialGraphEdge(from_node="launch", to_node="no_launch", action="abort"),
            SequentialGraphEdge(from_node="market", to_node="pricing", outcome="positive", probability=0.7),
            SequentialGraphEdge(from_node="market", to_node="exit", outcome="negative", probability=0.3),
            SequentialGraphEdge(from_node="pricing", to_node="high", action="premium"),
            SequentialGraphEdge(from_node="pricing", to_node="medium", action="economy"),
        ],
        stage_assignments={"launch": 0, "market": 1, "pricing": 1, "high": 2, "medium": 2, "exit": 2, "no_launch": 1},
    )
    stages = [
        DecisionStage(stage_index=0, stage_label="Launch", decision_nodes=["launch"]),
        DecisionStage(stage_index=1, stage_label="Pricing", decision_nodes=["pricing"], resolution_nodes=["market"]),
        DecisionStage(stage_index=2, stage_label="Terminal", decision_nodes=[]),
    ]
    return SequentialAnalysisRequest(graph=g, stages=stages, discount_factor=df, risk_tolerance=risk)


@pytest.fixture
def engine():
    return SequentialDecisionEngine()


def _stage_map(resp):
    return {s.stage_index: s for s in resp.stage_analyses}


# ---------------------------------------------------------------------------
# Hand-derived pins
# ---------------------------------------------------------------------------


def test_stage_evpi_neutral_fixture_exact_pin(engine):
    """Shape 1 (risk_neutral fixture). Hand-derivation (discount 0.95):
    market branch values fav=0.95*100000=95000, unf=0.95*-20000=-19000.
    decide-now  = node_values[invest] = -10000 + 0.95*(0.6*95000+0.4*-19000)
                = -10000 + 0.95*49400 = 36930.
    decide-after = 0.6*max(invest|fav=-10000+0.95*95000=80250, wait=0)
                 + 0.4*max(invest|unf=-10000+0.95*-19000=-28050, wait=0)
                 = 0.6*80250 + 0.4*0 = 48150.
    stage_evpi = 48150 - 36930 = 11220."""
    resp = engine.analyze(two_stage())
    sm = _stage_map(resp)
    assert sm[0].stage_evpi == pytest.approx(11220.0, rel=1e-12)


def test_stage_evpi_positive_control_not_the_dispersion_heuristic(engine):
    """Positive control (trap #13): 11220 is materially different from the OLD
    optimal_waiting_value it replaced (discount * next-stage sqrt(Σvar)).
    old = 0.95 * sqrt(var(market)); var(market)=0.6*(95000-49400)^2 +
    0.4*(-19000-49400)^2 = 3.11904e9 -> sqrt≈55848.4 -> old≈53056 — nowhere near
    the honest 11220. The pin can SEE the heuristic->EVPI change."""
    resp = engine.analyze(two_stage())
    sm = _stage_map(resp)
    old_heuristic = 0.95 * (3119040000 ** 0.5)
    assert abs(sm[0].stage_evpi - old_heuristic) > 1000.0
    assert sm[0].stage_evpi == pytest.approx(11220.0, rel=1e-12)


def test_stage_evpi_zero_when_one_action_dominates_every_branch(engine):
    """THEOREM (brief pin): EVPI is EXACTLY 0 when one action dominates in every
    chance branch — perfect information never changes the choice. failure=+50000
    makes 'invest' beat 'wait' (=0) in BOTH branches, so E_C[max]=E_C[Q(invest)]=
    max_a E_C[Q] and stage_evpi == 0."""
    resp = engine.analyze(two_stage(failure=50000))
    sm = _stage_map(resp)
    assert sm[0].stage_evpi == 0.0


def test_stage_evpi_zero_when_chance_is_degenerate(engine):
    """Degenerate chance (p_fav=1.0): the outcome is certain, so there is no
    uncertainty to resolve and stage_evpi collapses to EXACTLY 0."""
    resp = engine.analyze(two_stage(p_fav=1.0))
    sm = _stage_map(resp)
    assert sm[0].stage_evpi == 0.0


def test_stage_evpi_zero_when_decision_faces_no_chance(engine):
    """3-stage: the stage-1 pricing decision's actions lead straight to terminals
    (the market chance already resolved upstream) — no downstream uncertainty, so
    stage_evpi == 0 for pricing, while the stage-0 launch decision (which faces the
    market chance) has a POSITIVE stage_evpi."""
    resp = engine.analyze(three_stage())
    sm = _stage_map(resp)
    assert sm[1].stage_evpi == 0.0            # pricing: nothing to resolve
    assert sm[0].stage_evpi is not None and sm[0].stage_evpi > 0.0  # launch faces market


def test_stage_evpi_none_for_stage_without_decision_node(engine):
    """A stage with no decision node has no decision to inform -> stage_evpi None,
    NOT a fabricated 0, with status 'no_decision_node' disclosing the cause (F-3)."""
    resp = engine.analyze(two_stage())
    sm = _stage_map(resp)
    assert sm[1].stage_evpi is None and sm[1].stage_evpi_status == "no_decision_node"
    assert sm[2].stage_evpi is None and sm[2].stage_evpi_status == "no_decision_node"


def test_stage_evpi_status_none_when_computed(engine):
    """A computed stage_evpi (incl. a real 0.0) carries status None — the status is
    ONLY for null causes, never overwriting a genuine value."""
    computed = _stage_map(engine.analyze(two_stage()))[0]        # 11220.0
    assert computed.stage_evpi == pytest.approx(11220.0) and computed.stage_evpi_status is None
    zero = _stage_map(engine.analyze(two_stage(failure=50000)))[0]  # real EVPI 0.0
    assert zero.stage_evpi == 0.0 and zero.stage_evpi_status is None


# ---------------------------------------------------------------------------
# F-3 — joint-enumeration safety cap (honest skip, not a DoS, not a 422)
# ---------------------------------------------------------------------------


def _fanout(K, B=2, df=1.0):
    """A decision D with K actions, each -> its OWN B-branch chance node (shared
    terminals). The decide-after leg would enumerate B^K joint cells. Nodes = 1 + K
    + B; edges = K + B·K. resolution_nodes kept empty (its own <=20 cap is unrelated
    to the DoS, which rides the decision's action edges)."""
    nodes = [SequentialGraphNode(id="D", type="decision", label="D")]
    edges = []
    for j in range(B):
        nodes.append(SequentialGraphNode(id=f"t{j}", type="terminal", label=f"t{j}", payoff=100 * (j + 1)))
    sa = {"D": 0}
    for i in range(K):
        cid = f"c{i}"
        nodes.append(SequentialGraphNode(id=cid, type="chance", label=cid))
        sa[cid] = 1
        edges.append(SequentialGraphEdge(from_node="D", to_node=cid, action=f"a{i}"))
        for j in range(B):
            edges.append(SequentialGraphEdge(from_node=cid, to_node=f"t{j}", outcome=f"o{j}", probability=1.0 / B))
    for j in range(B):
        sa[f"t{j}"] = 2
    g = SequentialGraph(nodes=nodes, edges=edges, stage_assignments=sa)
    stages = [
        DecisionStage(stage_index=0, stage_label="s0", decision_nodes=["D"]),
        DecisionStage(stage_index=1, stage_label="s1", decision_nodes=[]),
        DecisionStage(stage_index=2, stage_label="s2", decision_nodes=[]),
    ]
    return SequentialAnalysisRequest(graph=g, stages=stages, discount_factor=df, risk_tolerance="neutral")


def test_stage_evpi_computes_at_cap_boundary(engine):
    """K=12 => 2^12 = 4096 = the cap => COMPUTED (value present). This is the boundary:
    <= cap computes. Note: _fanout(12) gives 12 DISTINCT action-specific chance nodes,
    so it is the UNIDENTIFIED shape (F1, D-23.11) — the value is disclosed with
    stage_evpi_status='assumed_independent_coupling' + coupling_assumption, NOT the old
    exact status None. The cap semantics (computes vs skips at the boundary) are what
    this test pins; the identifiability disclosure is verified below."""
    sm = _stage_map(engine.analyze(_fanout(12)))
    assert sm[0].stage_evpi is not None  # <= cap => COMPUTED (not skipped)
    assert sm[0].stage_evpi_status == "assumed_independent_coupling"
    assert sm[0].coupling_assumption == "independence_across_actions"


def test_stage_evpi_skips_just_over_cap(engine):
    """K=13 => 2^13 = 8192 > 4096 => honest SKIP: stage_evpi null + status. The
    exact analysis still succeeds (optimal_policy / value_of_flexibility present)."""
    resp = engine.analyze(_fanout(13))
    sm = {s.stage_index: s for s in resp.stage_analyses}
    assert sm[0].stage_evpi is None
    assert sm[0].stage_evpi_status == "skipped_joint_space_too_large"
    assert resp.optimal_policy is not None
    assert resp.value_of_flexibility is not None  # exact analysis untouched


def test_stage_evpi_legal_max_fanout_returns_fast_not_hang(engine):
    """F-3 core: a LEGAL request (100 nodes / 291 edges, K=97 => 2^97 joint cells)
    must return quickly with an honest skip — WITHOUT the guard this never
    terminates and freezes the event loop. Time-bound as a safety net; the
    load-bearing assertion is the status (a mechanism pin, no wall-clock race)."""
    req = _fanout(97)
    assert len(req.graph.nodes) == 100 and len(req.graph.edges) == 291  # legal
    t0 = time.monotonic()
    resp = engine.analyze(req)
    elapsed = time.monotonic() - t0
    sm = {s.stage_index: s for s in resp.stage_analyses}
    assert sm[0].stage_evpi is None
    assert sm[0].stage_evpi_status == "skipped_joint_space_too_large"
    assert elapsed < 2.0, f"stage_evpi guard did not bound the enumeration: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# F1 (A3 Codex-fix-C, D-23.11) — identifiability: the joint coupling of outcomes
# across mutually-exclusive actions is NOT in the tree. Independence is ASSUMED
# and must be DISCLOSED, not labelled exact.
# ---------------------------------------------------------------------------


def _two_action_two_chance(ca=(0, 100), cb=(0, 100), p=0.5, df=1.0):
    """Codex's exact counterexample: decision D with two actions A, B; A -> CA
    (chance, two branches ca[0]/ca[1] with prob p / 1-p), B -> CB (chance, two
    branches cb[0]/cb[1]). TWO DISTINCT action-specific chance nodes => the
    decide-after leg products their MARGINALS => independence is assumed."""
    g = SequentialGraph(
        nodes=[
            SequentialGraphNode(id="D", type="decision", label="D"),
            SequentialGraphNode(id="CA", type="chance", label="CA"),
            SequentialGraphNode(id="CB", type="chance", label="CB"),
            SequentialGraphNode(id="a_lo", type="terminal", label="a_lo", payoff=ca[0]),
            SequentialGraphNode(id="a_hi", type="terminal", label="a_hi", payoff=ca[1]),
            SequentialGraphNode(id="b_lo", type="terminal", label="b_lo", payoff=cb[0]),
            SequentialGraphNode(id="b_hi", type="terminal", label="b_hi", payoff=cb[1]),
        ],
        edges=[
            SequentialGraphEdge(from_node="D", to_node="CA", action="A"),
            SequentialGraphEdge(from_node="D", to_node="CB", action="B"),
            SequentialGraphEdge(from_node="CA", to_node="a_lo", outcome="a0", probability=p),
            SequentialGraphEdge(from_node="CA", to_node="a_hi", outcome="a1", probability=round(1 - p, 10)),
            SequentialGraphEdge(from_node="CB", to_node="b_lo", outcome="b0", probability=p),
            SequentialGraphEdge(from_node="CB", to_node="b_hi", outcome="b1", probability=round(1 - p, 10)),
        ],
        stage_assignments={"D": 0, "CA": 1, "CB": 1, "a_lo": 2, "a_hi": 2, "b_lo": 2, "b_hi": 2},
    )
    stages = [
        DecisionStage(stage_index=0, stage_label="s0", decision_nodes=["D"]),
        DecisionStage(stage_index=1, stage_label="s1", decision_nodes=[]),
        DecisionStage(stage_index=2, stage_label="s2", decision_nodes=[]),
    ]
    return SequentialAnalysisRequest(graph=g, stages=stages, discount_factor=df, risk_tolerance="neutral")


def test_stage_evpi_codex_counterexample_discloses_independence(engine):
    """RED-first (Codex F1). D with A->CA(0/100,p=.5), B->CB(0/100,p=.5), both action
    values 50. Hand-derivation (df=1.0): decide_now = max(E[A],E[B]) = max(50,50) = 50.
    decide-after under the INDEPENDENCE product over (CA,CB): outcomes {(0,0):0,
    (0,100):100,(100,0):100,(100,100):100}, each p=0.25 => e_after = 0.25*300 = 75.
    stage_evpi = 75 - 50 = 25.0. The SAME marginals also admit EVPI 0 (same-state
    coupling: 0.5*max(0,0)+0.5*max(100,100)-50 = 0) or 50 (opposite) — so 25 is ONE
    unrequested modelling choice, NOT exact for the supplied tree. The fix must emit
    25.0 WITH the disclosure, never as a bare exact number."""
    sm = _stage_map(engine.analyze(_two_action_two_chance()))
    s0 = sm[0]
    assert s0.stage_evpi == pytest.approx(25.0, rel=1e-12)  # the independence value
    assert s0.stage_evpi_status == "assumed_independent_coupling"
    assert s0.coupling_assumption == "independence_across_actions"


def test_stage_evpi_disclosed_value_is_the_independence_choice_not_forced(engine):
    """Positive control (trap #13): the disclosed 25.0 is specifically the
    INDEPENDENCE-product EVPI, and the identical marginals are consistent with a
    same-state coupling whose EVPI is 0. Both computed by hand here so the pin can
    SEE that 25 is a modelling CHOICE, not the tree's identified answer."""
    s0 = _stage_map(engine.analyze(_two_action_two_chance()))[0]
    # independence product (what the code computes): 25.0
    e_after_indep = 0.25 * (max(0, 0) + max(0, 100) + max(100, 0) + max(100, 100))
    evpi_indep = e_after_indep - 50.0
    # same-state coupling of the IDENTICAL marginals: CA==CB always
    e_after_same = 0.5 * max(0, 0) + 0.5 * max(100, 100)
    evpi_same = e_after_same - 50.0
    assert evpi_indep == pytest.approx(25.0)
    assert evpi_same == pytest.approx(0.0)  # same marginals, different (identified-absent) EVPI
    assert s0.stage_evpi == pytest.approx(evpi_indep, rel=1e-12)  # emitted = independence choice
    assert s0.coupling_assumption == "independence_across_actions"  # disclosed as such


def test_stage_evpi_single_chance_child_is_identified_no_coupling(engine):
    """IDENTIFIED (the other regime): the two_stage decision faces ONE chance node
    ('market') — only one action has a chance child — so a single shared realised
    state is resolved and stage_evpi is EXACT. It carries NO coupling_assumption and
    status None. The pinned 11220.0 must not move (disclosure, not recomputation)."""
    s0 = _stage_map(engine.analyze(two_stage()))[0]
    assert s0.stage_evpi == pytest.approx(11220.0, rel=1e-12)  # identified-path byte identity
    assert s0.stage_evpi_status is None
    assert s0.coupling_assumption is None


def test_stage_evpi_two_actions_share_one_chance_is_skipped_identified(engine):
    """D-23.19 (Codex re-confirm) SUPERSEDED the disclose-shared posture this test
    used to pin: when the SAME chance node id is reachable from >=2 actions, the
    graph IDENTIFIES the coupling (same id = same random variable), so emitting a
    number labelled 'assumed independent' CONTRADICTS the submitted graph. New
    contract: null + 'skipped_shared_chance_nodes_unsupported' until conditional
    subtree re-valuation ships (rowed refinement)."""
    g = SequentialGraph(
        nodes=[
            SequentialGraphNode(id="D", type="decision", label="D"),
            SequentialGraphNode(id="C", type="chance", label="C"),
            SequentialGraphNode(id="lo", type="terminal", label="lo", payoff=0),
            SequentialGraphNode(id="hi", type="terminal", label="hi", payoff=100),
        ],
        edges=[
            SequentialGraphEdge(from_node="D", to_node="C", action="A"),
            SequentialGraphEdge(from_node="D", to_node="C", action="B"),
            SequentialGraphEdge(from_node="C", to_node="lo", outcome="c0", probability=0.5),
            SequentialGraphEdge(from_node="C", to_node="hi", outcome="c1", probability=0.5),
        ],
        stage_assignments={"D": 0, "C": 1, "lo": 2, "hi": 2},
    )
    stages = [
        DecisionStage(stage_index=0, stage_label="s0", decision_nodes=["D"]),
        DecisionStage(stage_index=1, stage_label="s1", decision_nodes=[]),
        DecisionStage(stage_index=2, stage_label="s2", decision_nodes=[]),
    ]
    req = SequentialAnalysisRequest(graph=g, stages=stages, discount_factor=1.0, risk_tolerance="neutral")
    s0 = _stage_map(engine.analyze(req))[0]
    assert s0.stage_evpi is None
    assert s0.stage_evpi_status == "skipped_shared_chance_nodes_unsupported"
    assert s0.coupling_assumption is None


# ---------------------------------------------------------------------------
# F1 completeness — the cross-action joint can enter ONE LEVEL DEEPER, via an
# action whose immediate child is a decision node (Fable adversarial FN-1/FN-2).
# The detector must scan the WHOLE per-action subtree, not just the immediate child.
# ---------------------------------------------------------------------------


def _fn1_deeper_chance():
    """FN-1: D--A-->CA(0/100); D--B-->D2(decision)--go-->CB(0/100). Action B's chance
    (CB) sits one level below the decision D2 — invisible to an immediate-child scan,
    but its cross-action joint with CA is exactly as unidentified as Codex's case."""
    return SequentialAnalysisRequest(
        graph=SequentialGraph(
            nodes=[
                SequentialGraphNode(id="D", type="decision", label="D"),
                SequentialGraphNode(id="CA", type="chance", label="CA"),
                SequentialGraphNode(id="D2", type="decision", label="D2"),
                SequentialGraphNode(id="CB", type="chance", label="CB"),
                SequentialGraphNode(id="a_lo", type="terminal", label="a_lo", payoff=0),
                SequentialGraphNode(id="a_hi", type="terminal", label="a_hi", payoff=100),
                SequentialGraphNode(id="b_lo", type="terminal", label="b_lo", payoff=0),
                SequentialGraphNode(id="b_hi", type="terminal", label="b_hi", payoff=100),
            ],
            edges=[
                SequentialGraphEdge(from_node="D", to_node="CA", action="A"),
                SequentialGraphEdge(from_node="D", to_node="D2", action="B"),
                SequentialGraphEdge(from_node="D2", to_node="CB", action="go"),
                SequentialGraphEdge(from_node="CA", to_node="a_lo", outcome="a0", probability=0.5),
                SequentialGraphEdge(from_node="CA", to_node="a_hi", outcome="a1", probability=0.5),
                SequentialGraphEdge(from_node="CB", to_node="b_lo", outcome="b0", probability=0.5),
                SequentialGraphEdge(from_node="CB", to_node="b_hi", outcome="b1", probability=0.5),
            ],
            stage_assignments={"D": 0, "CA": 1, "D2": 1, "CB": 2, "a_lo": 2, "a_hi": 2, "b_lo": 3, "b_hi": 3},
        ),
        stages=[
            DecisionStage(stage_index=0, stage_label="s0", decision_nodes=["D"]),
            DecisionStage(stage_index=1, stage_label="s1", decision_nodes=["D2"], resolution_nodes=["CA"]),
            DecisionStage(stage_index=2, stage_label="s2", decision_nodes=[], resolution_nodes=["CB"]),
            DecisionStage(stage_index=3, stage_label="s3", decision_nodes=[]),
        ],
        discount_factor=1.0,
        risk_tolerance="neutral",
    )


def _fn2_same_node_deeper():
    """FN-2 == Codex re-confirm F1 repro: D--A-->C(0/100); D--B-->D2(decision)--go-->C
    (the SAME node C reused one level down). The tree IDENTIFIES the joint as
    same-state -> true EVPI 0.0; the old leg emitted the independence value 25.0.
    D-23.19: skipped (null + shared status) — the disclose-with-wrong-value posture
    contradicted the submitted graph. Exact-0 emission = the rowed conditional
    re-valuation refinement."""
    return SequentialAnalysisRequest(
        graph=SequentialGraph(
            nodes=[
                SequentialGraphNode(id="D", type="decision", label="D"),
                SequentialGraphNode(id="C", type="chance", label="C"),
                SequentialGraphNode(id="D2", type="decision", label="D2"),
                SequentialGraphNode(id="c_lo", type="terminal", label="c_lo", payoff=0),
                SequentialGraphNode(id="c_hi", type="terminal", label="c_hi", payoff=100),
            ],
            edges=[
                SequentialGraphEdge(from_node="D", to_node="C", action="A"),
                SequentialGraphEdge(from_node="D", to_node="D2", action="B"),
                SequentialGraphEdge(from_node="D2", to_node="C", action="go"),
                SequentialGraphEdge(from_node="C", to_node="c_lo", outcome="c0", probability=0.5),
                SequentialGraphEdge(from_node="C", to_node="c_hi", outcome="c1", probability=0.5),
            ],
            stage_assignments={"D": 0, "C": 1, "D2": 1, "c_lo": 2, "c_hi": 2},
        ),
        stages=[
            DecisionStage(stage_index=0, stage_label="s0", decision_nodes=["D"]),
            DecisionStage(stage_index=1, stage_label="s1", decision_nodes=["D2"], resolution_nodes=["C"]),
            DecisionStage(stage_index=2, stage_label="s2", decision_nodes=[]),
        ],
        discount_factor=1.0,
        risk_tolerance="neutral",
    )


def _fn0_deterministic_alt():
    """FN-0 control: D--A-->CA(0/100); D--B-->t50(terminal 50). Only ONE action faces
    chance; the alternative is deterministic under EVERY coupling, so E[Q_B|C]==Q_B and
    the stage is genuinely IDENTIFIED — must stay EXACT (status None)."""
    return SequentialAnalysisRequest(
        graph=SequentialGraph(
            nodes=[
                SequentialGraphNode(id="D", type="decision", label="D"),
                SequentialGraphNode(id="CA", type="chance", label="CA"),
                SequentialGraphNode(id="t50", type="terminal", label="t50", payoff=50),
                SequentialGraphNode(id="a_lo", type="terminal", label="a_lo", payoff=0),
                SequentialGraphNode(id="a_hi", type="terminal", label="a_hi", payoff=100),
            ],
            edges=[
                SequentialGraphEdge(from_node="D", to_node="CA", action="A"),
                SequentialGraphEdge(from_node="D", to_node="t50", action="B"),
                SequentialGraphEdge(from_node="CA", to_node="a_lo", outcome="a0", probability=0.5),
                SequentialGraphEdge(from_node="CA", to_node="a_hi", outcome="a1", probability=0.5),
            ],
            stage_assignments={"D": 0, "CA": 1, "t50": 1, "a_lo": 2, "a_hi": 2},
        ),
        stages=[
            DecisionStage(stage_index=0, stage_label="s0", decision_nodes=["D"]),
            DecisionStage(stage_index=1, stage_label="s1", decision_nodes=[], resolution_nodes=["CA"]),
            DecisionStage(stage_index=2, stage_label="s2", decision_nodes=[]),
        ],
        discount_factor=1.0,
        risk_tolerance="neutral",
    )


def test_stage_evpi_fn1_deeper_chance_is_disclosed(engine):
    """RED-first (adversarial FN-1). Deeper cross-action chance (via a decision node)
    must be DISCLOSED, not labelled exact. Was: status None (immediate-child detector
    blind to it). Same {0,25,50} unidentified spread as Codex; emitted value 25.0."""
    s0 = _stage_map(engine.analyze(_fn1_deeper_chance()))[0]
    assert s0.stage_evpi == pytest.approx(25.0, rel=1e-12)
    assert s0.stage_evpi_status == "assumed_independent_coupling"
    assert s0.coupling_assumption == "independence_across_actions"


def test_stage_evpi_fn2_same_node_deeper_is_skipped_identified(engine):
    """FN-2 CLOSED at D-23.19 (was disclose-with-wrong-value, the 'value-refinement
    flag'). The SAME chance reused one level deeper: the tree identifies same-state
    (true EVPI 0.0); the old leg emitted 25.0 labelled 'assumed independent' — a
    label the graph contradicts (Codex re-confirm F1 PARTIAL). New contract: null +
    'skipped_shared_chance_nodes_unsupported'. Was pre-D-23.11: 25.0/None labelled
    exact; was pre-D-23.19: 25.0 disclosed.

    MUTATION ANCHOR: removing the shared_chance guard re-emits 25.0+disclosed."""
    s0 = _stage_map(engine.analyze(_fn2_same_node_deeper()))[0]
    assert s0.stage_evpi is None
    assert s0.stage_evpi_status == "skipped_shared_chance_nodes_unsupported"
    assert s0.coupling_assumption is None


def test_stage_evpi_fn0_deterministic_alternative_stays_exact(engine):
    """Control (adversarial FN-0): one action faces chance, the other is deterministic
    -> genuinely IDENTIFIED. Must stay EXACT (status None, no coupling). Guards against
    the broadened detector over-disclosing a real single-uncertainty stage."""
    s0 = _stage_map(engine.analyze(_fn0_deterministic_alt()))[0]
    assert s0.stage_evpi == pytest.approx(25.0, rel=1e-12)
    assert s0.stage_evpi_status is None
    assert s0.coupling_assumption is None


def test_stage_evpi_shared_chance_not_falsely_skipped_by_cap(engine):
    """F-A3: a chance node shared by 2 action edges must be counted ONCE against the
    joint-cell cap (deduped by cid), not B^2. A 65-branch node shared by 2 actions is
    65 cells (< 4096), not 65^2=4225 (> 4096) — so the CAP must not fire. D-23.19:
    the shared-identity guard then fires INSTEAD (null + shared status) — the pin
    here is that the skip reason is the IDENTIFIED-coupling one, never the cap
    (a cap-skip would mean the dedupe regressed to B^k counting)."""
    B = 65
    nodes = [SequentialGraphNode(id="D", type="decision", label="D"),
             SequentialGraphNode(id="C", type="chance", label="C")]
    edges = [SequentialGraphEdge(from_node="D", to_node="C", action="A"),
             SequentialGraphEdge(from_node="D", to_node="C", action="B")]
    sa = {"D": 0, "C": 1}
    for j in range(B):
        nodes.append(SequentialGraphNode(id=f"t{j}", type="terminal", label=f"t{j}", payoff=float(j)))
        sa[f"t{j}"] = 2
        edges.append(SequentialGraphEdge(from_node="C", to_node=f"t{j}", outcome=f"o{j}", probability=1.0 / B))
    g = SequentialGraph(nodes=nodes, edges=edges, stage_assignments=sa)
    stages = [DecisionStage(stage_index=0, stage_label="s0", decision_nodes=["D"]),
              DecisionStage(stage_index=1, stage_label="s1", decision_nodes=[]),
              DecisionStage(stage_index=2, stage_label="s2", decision_nodes=[])]
    req = SequentialAnalysisRequest(graph=g, stages=stages, discount_factor=1.0, risk_tolerance="neutral")
    s0 = _stage_map(engine.analyze(req))[0]
    # D-23.19: shared-identity guard fires (C read by both actions) — but NEVER the
    # cap: the dedupe keeps the joint count at 65 < 4096, so a cap-skip here would
    # mean the F-A3 dedupe regressed.
    assert s0.stage_evpi_status == "skipped_shared_chance_nodes_unsupported"
    assert s0.stage_evpi_status != "skipped_joint_space_too_large"


def test_stage_evpi_zero_immediate_chance_but_deeper_is_exact_immediate_scope(engine):
    """R2-F1 pin (round-2 metric-scope). T4: D--A-->D_a(decision)--ga-->CA(0/100);
    D--B-->D_b(decision)--gb-->CB(0/100). The stage-0 decision D faces NO immediate
    chance (both children are decisions), so its stage_evpi is EXACTLY 0.0 with status
    None — the INTENDED per-immediate-stage semantics, NOT a missed disclosure. Even
    though >=2 actions have chance reachable DEEPER, that deeper chance is a LATER
    stage's EVPI (surfaced when D_a / D_b are analysed), and no independence product
    runs at stage 0 (there is no immediate chance to product over). The value 0.0 is
    coupling-invariant-exact. Guards the description's immediate-scope claim: status
    None here does NOT mean 'no chance anywhere in the subtree'."""
    g = SequentialGraph(
        nodes=[
            SequentialGraphNode(id="D", type="decision", label="D"),
            SequentialGraphNode(id="D_a", type="decision", label="D_a"),
            SequentialGraphNode(id="D_b", type="decision", label="D_b"),
            SequentialGraphNode(id="CA", type="chance", label="CA"),
            SequentialGraphNode(id="CB", type="chance", label="CB"),
            SequentialGraphNode(id="a_lo", type="terminal", label="a_lo", payoff=0),
            SequentialGraphNode(id="a_hi", type="terminal", label="a_hi", payoff=100),
            SequentialGraphNode(id="b_lo", type="terminal", label="b_lo", payoff=0),
            SequentialGraphNode(id="b_hi", type="terminal", label="b_hi", payoff=100),
        ],
        edges=[
            SequentialGraphEdge(from_node="D", to_node="D_a", action="A"),
            SequentialGraphEdge(from_node="D", to_node="D_b", action="B"),
            SequentialGraphEdge(from_node="D_a", to_node="CA", action="ga"),
            SequentialGraphEdge(from_node="D_b", to_node="CB", action="gb"),
            SequentialGraphEdge(from_node="CA", to_node="a_lo", outcome="a0", probability=0.5),
            SequentialGraphEdge(from_node="CA", to_node="a_hi", outcome="a1", probability=0.5),
            SequentialGraphEdge(from_node="CB", to_node="b_lo", outcome="b0", probability=0.5),
            SequentialGraphEdge(from_node="CB", to_node="b_hi", outcome="b1", probability=0.5),
        ],
        stage_assignments={"D": 0, "D_a": 1, "D_b": 1, "CA": 2, "CB": 2, "a_lo": 3, "a_hi": 3, "b_lo": 3, "b_hi": 3},
    )
    stages = [
        DecisionStage(stage_index=0, stage_label="s0", decision_nodes=["D"]),
        DecisionStage(stage_index=1, stage_label="s1", decision_nodes=["D_a", "D_b"], resolution_nodes=[]),
        DecisionStage(stage_index=2, stage_label="s2", decision_nodes=[], resolution_nodes=["CA", "CB"]),
        DecisionStage(stage_index=3, stage_label="s3", decision_nodes=[]),
    ]
    req = SequentialAnalysisRequest(graph=g, stages=stages, discount_factor=1.0, risk_tolerance="neutral")
    s0 = _stage_map(engine.analyze(req))[0]
    assert s0.stage_evpi == 0.0
    assert s0.stage_evpi_status is None
    assert s0.coupling_assumption is None


# ---------------------------------------------------------------------------
# Independent identity cross-check (two implementations must agree)
# ---------------------------------------------------------------------------


def test_stage_evpi_matches_independent_e_max_minus_max_e(engine):
    """Cross-check the production helper against an independent E[max]−max E
    computed straight from the backward-induction node_values — non-vacuous, and
    catches a subtle discount/probability bug the single value pin might miss."""
    req = two_stage(success=120000, failure=-5000, p_fav=0.55)
    graph_data = engine._build_graph_data(req.graph)
    node_values, _ = engine._backward_induction(
        graph_data, req.stages, req.discount_factor, req.risk_tolerance or "neutral"
    )
    df = req.discount_factor
    nv = node_values
    # Independent legs for the 'invest' decision facing 'market'.
    p_fav, p_unf = 0.55, 0.45
    fav_branch = 0.0 + df * nv["success"]     # edge_value(market->success)
    unf_branch = 0.0 + df * nv["failure"]     # edge_value(market->failure)
    q_invest_fav = -10000 + df * fav_branch
    q_invest_unf = -10000 + df * unf_branch
    q_wait = 0.0 + df * nv["no_invest"]
    e_after = p_fav * max(q_invest_fav, q_wait) + p_unf * max(q_invest_unf, q_wait)
    decide_now = nv["invest"]
    expected = max(0.0, e_after - decide_now)

    got, status, coupling = engine._compute_stage_evpi(
        "invest", graph_data, node_values, df
    )
    assert status is None  # computed, identified (single chance child), under cap
    assert coupling is None  # identified -> no coupling assumption disclosed
    assert got == pytest.approx(expected, rel=1e-12)
    assert got >= 0.0


def test_stage_evpi_nonnegative_across_shapes(engine):
    for req in (two_stage(), two_stage(failure=50000), two_stage(p_fav=0.3),
                two_stage(success=1000, failure=-500), three_stage()):
        for sa in engine.analyze(req).stage_analyses:
            if sa.stage_evpi is not None:
                assert sa.stage_evpi >= 0.0, (req, sa)


# ---------------------------------------------------------------------------
# Risk-adjustment posture (documented choice)
# ---------------------------------------------------------------------------


def test_stage_evpi_risk_posture_uses_adjusted_values(engine):
    """Documented posture: stage_evpi is computed on the engine's risk-ADJUSTED
    node values (consistent with how the policy chooses actions). Under risk
    aversion, resolving the chance ALSO removes its variance penalty, so the
    averse stage_evpi is strictly LARGER than the neutral one (the variance-removal
    premium) — a value change, not a no-op — while staying >= 0."""
    neutral = _stage_map(engine.analyze(two_stage(risk="neutral")))[0].stage_evpi
    averse = _stage_map(engine.analyze(two_stage(risk="averse")))[0].stage_evpi
    assert neutral == pytest.approx(11220.0, rel=1e-12)
    assert averse > neutral
    assert averse >= 0.0


# ---------------------------------------------------------------------------
# Removal of the sqrt heuristic (replacement, not accretion)
# ---------------------------------------------------------------------------


def test_optimal_waiting_value_removed_from_stage_analysis_model():
    """The heuristic key is GONE from the model; the honest key is present."""
    assert "optimal_waiting_value" not in StageAnalysis.model_fields
    assert "stage_evpi" in StageAnalysis.model_fields


def test_optimal_waiting_value_absent_stage_evpi_present_on_response(engine):
    """End-to-end (in-process): the serialised stage_analyses carry stage_evpi and
    NOT optimal_waiting_value."""
    resp = engine.analyze(two_stage())
    dumped = resp.model_dump(by_alias=True, exclude_none=True)
    for sa in dumped["stage_analyses"]:
        assert "optimal_waiting_value" not in sa
    # at least the decision stage carries stage_evpi
    assert any("stage_evpi" in sa for sa in dumped["stage_analyses"])


class TestStageEvpiStatusEmissionIff:
    """Altitude Q1 (C3): sibling-presence emission-iff on StageAnalysis, fail-loud."""

    def _base(self, **over):
        kw = dict(
            stage_index=0,
            stage_label="S",
            options_at_stage=[],
            resolved_uncertainty=0.0,
            stage_evpi=11220.0,
            stage_evpi_status=None,
        )
        kw.update(over)
        return kw

    def test_computed_value_no_status_ok(self):
        StageAnalysis(**self._base(stage_evpi=0.0, stage_evpi_status=None))  # incl 0.0
        StageAnalysis(**self._base(stage_evpi=11220.0, stage_evpi_status=None))

    def test_null_value_with_status_ok(self):
        StageAnalysis(**self._base(stage_evpi=None, stage_evpi_status="no_decision_node"))
        StageAnalysis(
            **self._base(stage_evpi=None, stage_evpi_status="skipped_joint_space_too_large")
        )

    def test_value_with_status_rejected(self):
        # A real value must NOT carry a skip reason (fabricated status).
        with pytest.raises(ValidationError):
            StageAnalysis(**self._base(stage_evpi=5.0, stage_evpi_status="no_decision_node"))

    def test_null_value_without_status_rejected(self):
        # A null value must disclose WHY (no silent null).
        with pytest.raises(ValidationError):
            StageAnalysis(**self._base(stage_evpi=None, stage_evpi_status=None))

    # F1 (D-23.11) — the assumption-laden status rides WITH a computed value.
    def test_assumed_independent_coupling_with_value_and_coupling_ok(self):
        StageAnalysis(
            **self._base(
                stage_evpi=25.0,
                stage_evpi_status="assumed_independent_coupling",
                coupling_assumption="independence_across_actions",
            )
        )

    def test_assumed_independent_coupling_without_coupling_rejected(self):
        # The disclosed assumption MUST name the coupling.
        with pytest.raises(ValidationError):
            StageAnalysis(
                **self._base(
                    stage_evpi=25.0,
                    stage_evpi_status="assumed_independent_coupling",
                    coupling_assumption=None,
                )
            )

    def test_assumed_independent_coupling_with_null_value_rejected(self):
        # An independence assumption is meaningless without a computed value to ride.
        with pytest.raises(ValidationError):
            StageAnalysis(
                **self._base(
                    stage_evpi=None,
                    stage_evpi_status="assumed_independent_coupling",
                    coupling_assumption="independence_across_actions",
                )
            )

    def test_coupling_without_its_status_rejected(self):
        # coupling_assumption may appear ONLY under 'assumed_independent_coupling'.
        with pytest.raises(ValidationError):
            StageAnalysis(
                **self._base(
                    stage_evpi=25.0,
                    stage_evpi_status=None,
                    coupling_assumption="independence_across_actions",
                )
            )
