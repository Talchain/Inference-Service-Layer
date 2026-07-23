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
    """K=12 => 2^12 = 4096 = the cap => COMPUTED (status None). This is the boundary:
    <= cap computes."""
    sm = _stage_map(engine.analyze(_fanout(12)))
    assert sm[0].stage_evpi is not None
    assert sm[0].stage_evpi_status is None


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

    got, status = engine._compute_stage_evpi("invest", graph_data, node_values, df)
    assert status is None  # computed, under the joint-cell cap
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
