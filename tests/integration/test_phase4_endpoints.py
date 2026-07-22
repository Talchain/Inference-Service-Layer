"""
Integration tests for Phase 4 API endpoints.

Tests the conditional recommendation and sequential decision endpoints
through the FastAPI application.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.api.main import app


@pytest_asyncio.fixture
async def client():
    """Async HTTP client for FastAPI testing."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def conditional_recommend_request():
    """Sample conditional recommendation request."""
    return {
        "run_id": "test_integration_001",
        "ranked_options": [
            {
                "option_id": "aggressive",
                "label": "Aggressive Expansion",
                "expected_value": 75000.0,
                "distribution": {
                    "type": "normal",
                    "parameters": {"mean": 75000, "std": 20000}
                },
                "risk_metrics": {
                    "variance": 400000000,
                    "downside_risk": 50000,
                    "probability_of_loss": 0.1
                }
            },
            {
                "option_id": "conservative",
                "label": "Conservative Growth",
                "expected_value": 50000.0,
                "distribution": {
                    "type": "normal",
                    "parameters": {"mean": 50000, "std": 5000}
                }
            }
        ],
        "condition_types": ["threshold", "risk_profile"],
        "max_conditions": 5
    }


@pytest.fixture
def sequential_analysis_request():
    """Sample sequential analysis request."""
    return {
        "graph": {
            "nodes": [
                {"id": "invest", "type": "decision", "label": "Investment Decision"},
                {"id": "market", "type": "chance", "label": "Market Outcome"},
                {"id": "success", "type": "terminal", "label": "Success", "payoff": 100000},
                {"id": "failure", "type": "terminal", "label": "Failure", "payoff": -20000},
                {"id": "no_invest", "type": "terminal", "label": "No Investment", "payoff": 0}
            ],
            "edges": [
                {"from": "invest", "to": "market", "action": "invest", "immediate_payoff": -10000},
                {"from": "invest", "to": "no_invest", "action": "wait"},
                {"from": "market", "to": "success", "outcome": "favorable", "probability": 0.6},
                {"from": "market", "to": "failure", "outcome": "unfavorable", "probability": 0.4}
            ],
            "stage_assignments": {
                "invest": 0,
                "market": 1,
                "success": 2,
                "failure": 2,
                "no_invest": 1
            }
        },
        "stages": [
            {"stage_index": 0, "stage_label": "Investment", "decision_nodes": ["invest"]},
            {"stage_index": 1, "stage_label": "Market", "decision_nodes": [], "resolution_nodes": ["market"]},
            {"stage_index": 2, "stage_label": "Terminal", "decision_nodes": []}
        ],
        "discount_factor": 0.95,
        "risk_tolerance": "neutral",
        "monte_carlo_samples": 100
    }


@pytest.fixture
def stage_sensitivity_request():
    """Sample stage sensitivity request."""
    return {
        "graph": {
            "nodes": [
                {"id": "decide", "type": "decision", "label": "Decision"},
                {"id": "chance", "type": "chance", "label": "Chance Event"},
                {"id": "good", "type": "terminal", "label": "Good Outcome", "payoff": 100},
                {"id": "bad", "type": "terminal", "label": "Bad Outcome", "payoff": -50}
            ],
            "edges": [
                {"from": "decide", "to": "chance", "action": "proceed"},
                {"from": "chance", "to": "good", "outcome": "good", "probability": 0.7},
                {"from": "chance", "to": "bad", "outcome": "bad", "probability": 0.3}
            ],
            "stage_assignments": {
                "decide": 0,
                "chance": 1,
                "good": 2,
                "bad": 2
            }
        },
        "stages": [
            {"stage_index": 0, "stage_label": "Decision", "decision_nodes": ["decide"]},
            {"stage_index": 1, "stage_label": "Chance", "decision_nodes": []},
            {"stage_index": 2, "stage_label": "Terminal", "decision_nodes": []}
        ],
        "variation_range": 0.2
    }


class TestConditionalRecommendEndpoint:
    """Tests for POST /api/v1/analysis/conditional-recommend"""

    @pytest.mark.asyncio
    async def test_conditional_recommend_success(self, client, conditional_recommend_request):
        """Should successfully generate conditional recommendations."""
        response = await client.post(
            "/api/v1/analysis/conditional-recommend",
            json=conditional_recommend_request
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "schema_version" in data
        assert data["schema_version"] == "conditional_recommend.v1"
        assert "primary_recommendation" in data
        assert "conditional_recommendations" in data
        assert "robustness_summary" in data

        # Check primary recommendation
        primary = data["primary_recommendation"]
        assert primary["option_id"] == "aggressive"  # Higher expected value
        assert primary["confidence"] in ["high", "medium", "low"]
        assert primary["expected_value"] == 75000.0

        # Check robustness summary
        robustness = data["robustness_summary"]
        assert robustness["recommendation_stability"] in ["robust", "moderate", "fragile"]
        assert "conditions_count" in robustness

    @pytest.mark.asyncio
    async def test_conditional_recommend_with_threshold_only(self, client, conditional_recommend_request):
        """Should generate only threshold conditions when specified."""
        conditional_recommend_request["condition_types"] = ["threshold"]

        response = await client.post(
            "/api/v1/analysis/conditional-recommend",
            json=conditional_recommend_request
        )

        assert response.status_code == 200
        data = response.json()

        # All conditions should be threshold type
        for cond in data["conditional_recommendations"]:
            assert cond["condition_type"] == "threshold"

    @pytest.mark.asyncio
    async def test_conditional_recommend_with_risk_profile(self, client, conditional_recommend_request):
        """Should generate risk profile conditions."""
        conditional_recommend_request["condition_types"] = ["risk_profile"]

        response = await client.post(
            "/api/v1/analysis/conditional-recommend",
            json=conditional_recommend_request
        )

        assert response.status_code == 200
        data = response.json()

        # Should have risk profile conditions or empty list
        for cond in data["conditional_recommendations"]:
            assert cond["condition_type"] == "risk_profile"

    @pytest.mark.asyncio
    async def test_conditional_recommend_respects_max_conditions(self, client, conditional_recommend_request):
        """Should respect max_conditions limit."""
        conditional_recommend_request["max_conditions"] = 2
        conditional_recommend_request["condition_types"] = ["threshold", "risk_profile", "scenario"]

        response = await client.post(
            "/api/v1/analysis/conditional-recommend",
            json=conditional_recommend_request
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["conditional_recommendations"]) <= 2

    @pytest.mark.asyncio
    async def test_conditional_recommend_invalid_condition_type(self, client, conditional_recommend_request):
        """Should reject invalid condition types."""
        conditional_recommend_request["condition_types"] = ["invalid_type"]

        response = await client.post(
            "/api/v1/analysis/conditional-recommend",
            json=conditional_recommend_request
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_conditional_recommend_single_option_fails(self, client):
        """Should reject requests with fewer than 2 options."""
        request = {
            "run_id": "test",
            "ranked_options": [
                {
                    "option_id": "only_one",
                    "label": "Only Option",
                    "expected_value": 100.0,
                    "distribution": {"type": "normal", "parameters": {"mean": 100, "std": 10}}
                }
            ],
            "condition_types": ["threshold"],
            "max_conditions": 5
        }

        response = await client.post(
            "/api/v1/analysis/conditional-recommend",
            json=request
        )

        assert response.status_code == 422  # Validation error


class TestSequentialAnalysisEndpoint:
    """Tests for POST /api/v1/analysis/sequential"""

    @pytest.mark.asyncio
    async def test_sequential_analysis_success(self, client, sequential_analysis_request):
        """Should successfully analyze sequential decision."""
        response = await client.post(
            "/api/v1/analysis/sequential",
            json=sequential_analysis_request
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "schema_version" in data
        assert data["schema_version"] == "sequential.v1"
        assert "optimal_policy" in data
        assert "stage_analyses" in data
        assert "value_of_flexibility" in data
        assert "sensitivity_to_timing" in data

        # Check optimal policy
        policy = data["optimal_policy"]
        assert "stages" in policy
        assert "expected_total_value" in policy
        assert "value_distribution" in policy

        # Value-correct (hand-derived, neutral, d=0.95):
        #   V(market)  = 0.6*0.95*100000 + 0.4*0.95*(-20000) = 57000 - 7600 = 49400
        #   V(invest)  = max(-10000 + 0.95*49400, wait=0)    = 36930
        # (This fixture graph has no chance/decision stage collision, so 36930 holds
        # before and after the same-stage fix; it anchors correct behaviour.)
        assert policy["expected_total_value"] == pytest.approx(36930.0)

        # Check value of flexibility is non-negative
        assert data["value_of_flexibility"] >= 0

        # Check timing sensitivity
        assert data["sensitivity_to_timing"] in ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_sequential_analysis_with_discount(self, client, sequential_analysis_request):
        """Should respect discount factor."""
        sequential_analysis_request["discount_factor"] = 0.8

        response = await client.post(
            "/api/v1/analysis/sequential",
            json=sequential_analysis_request
        )

        assert response.status_code == 200
        data = response.json()

        # Should complete successfully with different discount
        assert data["optimal_policy"] is not None

    @pytest.mark.asyncio
    async def test_sequential_analysis_risk_averse(self, client, sequential_analysis_request):
        """Should handle risk averse tolerance."""
        sequential_analysis_request["risk_tolerance"] = "averse"

        response = await client.post(
            "/api/v1/analysis/sequential",
            json=sequential_analysis_request
        )

        assert response.status_code == 200
        data = response.json()

        assert data["optimal_policy"] is not None

    @pytest.mark.asyncio
    async def test_sequential_analysis_invalid_risk_tolerance(self, client, sequential_analysis_request):
        """Should reject invalid risk tolerance."""
        sequential_analysis_request["risk_tolerance"] = "invalid"

        response = await client.post(
            "/api/v1/analysis/sequential",
            json=sequential_analysis_request
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_sequential_same_stage_collision_matches_control(self, client):
        """Served-path guard for the A2 same-stage backward-induction drop.

        A chance node and the decision it feeds sharing a stage_index (the request
        schema's OWN canonical example: market:1, pricing:1) must NOT collapse the
        favorable subtree to 0. This posts the identical decision problem in two
        layouts -- collision (pricing at stage 1) and control (pricing at its own
        stage 2) -- and asserts both yield the correct +1893.75 with market
        continuation 54625. If the drop ever returns, the collision layout goes RED
        (-58122.5 / continuation -8550). Hand-derivation: NOTES.md sec 2 and
        tests/unit/test_sequential_decision.py.
        """

        def payload(pricing_stage):
            terminal_stage = pricing_stage + 1
            stages = [
                {"stage_index": 0, "stage_label": "Investment", "decision_nodes": ["invest"]}
            ]
            if pricing_stage == 1:
                stages.append({
                    "stage_index": 1, "stage_label": "Pricing",
                    "decision_nodes": ["pricing"], "resolution_nodes": ["market"],
                })
            else:
                stages.append({
                    "stage_index": 1, "stage_label": "Market",
                    "decision_nodes": [], "resolution_nodes": ["market"],
                })
                stages.append({
                    "stage_index": 2, "stage_label": "Pricing", "decision_nodes": ["pricing"],
                })
            stages.append({
                "stage_index": terminal_stage, "stage_label": "Terminal", "decision_nodes": [],
            })
            return {
                "graph": {
                    "nodes": [
                        {"id": "invest", "type": "decision", "label": "Invest"},
                        {"id": "market", "type": "chance", "label": "Market"},
                        {"id": "pricing", "type": "decision", "label": "Pricing"},
                        {"id": "high_return", "type": "terminal", "label": "High", "payoff": 100000},
                        {"id": "low_return", "type": "terminal", "label": "Low", "payoff": 20000},
                        {"id": "loss", "type": "terminal", "label": "Loss", "payoff": -30000},
                    ],
                    "edges": [
                        {"from": "invest", "to": "market", "action": "invest", "immediate_payoff": -50000},
                        {"from": "market", "to": "pricing", "outcome": "favorable", "probability": 0.7},
                        {"from": "market", "to": "loss", "outcome": "unfavorable", "probability": 0.3},
                        {"from": "pricing", "to": "high_return", "action": "premium"},
                        {"from": "pricing", "to": "low_return", "action": "economy"},
                    ],
                    "stage_assignments": {
                        "invest": 0, "market": 1, "pricing": pricing_stage,
                        "high_return": terminal_stage, "low_return": terminal_stage,
                        "loss": terminal_stage,
                    },
                },
                "stages": stages,
                "discount_factor": 0.95,
                "risk_tolerance": "neutral",
            }

        collision = await client.post("/api/v1/analysis/sequential", json=payload(1))
        control = await client.post("/api/v1/analysis/sequential", json=payload(2))
        assert collision.status_code == 200
        assert control.status_code == 200

        control_val = control.json()["optimal_policy"]["expected_total_value"]
        collision_val = collision.json()["optimal_policy"]["expected_total_value"]
        assert control_val == pytest.approx(1893.75)
        assert collision_val == pytest.approx(1893.75)

        # market continuation seen by invest must be 54625, never -8550
        invest_option = collision.json()["stage_analyses"][0]["options_at_stage"][0]
        assert invest_option["continuation_value"] == pytest.approx(54625.0)

    @pytest.mark.asyncio
    async def test_sequential_omitted_probability_equal_split(self, client):
        """F-3: chance edges with OMITTED probability default to an equal split.

        The probability field's Pydantic default is None; omitting it must mean
        'unspecified' -> equal split across the node's outgoing edges, NOT a
        TypeError-500. RED at HEAD: `edge.get('probability', 1/len)` returns the
        present-but-None value -> None*float -> 500.

        Hand-derivation (d=1.0, neutral, coin has 2 outgoing edges -> 0.5 each):
          V(coin)  = 0.5*100 + 0.5*0 = 50
          V(decide)= max(play=0 + 1.0*50, skip=0) = 50
        """
        request = {
            "graph": {
                "nodes": [
                    {"id": "decide", "type": "decision", "label": "Decide"},
                    {"id": "coin", "type": "chance", "label": "Coin"},
                    {"id": "heads", "type": "terminal", "label": "Heads", "payoff": 100},
                    {"id": "tails", "type": "terminal", "label": "Tails", "payoff": 0},
                    {"id": "skip", "type": "terminal", "label": "Skip", "payoff": 0},
                ],
                "edges": [
                    {"from": "decide", "to": "coin", "action": "play"},
                    {"from": "decide", "to": "skip", "action": "skip"},
                    # probability OMITTED on both -> equal split expected
                    {"from": "coin", "to": "heads", "outcome": "h"},
                    {"from": "coin", "to": "tails", "outcome": "t"},
                ],
                "stage_assignments": {
                    "decide": 0, "coin": 1, "heads": 2, "tails": 2, "skip": 1,
                },
            },
            "stages": [
                {"stage_index": 0, "stage_label": "Decide", "decision_nodes": ["decide"]},
                {
                    "stage_index": 1, "stage_label": "Coin",
                    "decision_nodes": [], "resolution_nodes": ["coin"],
                },
                {"stage_index": 2, "stage_label": "Terminal", "decision_nodes": []},
            ],
            "discount_factor": 1.0,
            "risk_tolerance": "neutral",
        }
        response = await client.post("/api/v1/analysis/sequential", json=request)
        assert response.status_code == 200
        assert response.json()["optimal_policy"]["expected_total_value"] == pytest.approx(50.0)


class TestPolicyTreeEndpoint:
    """Tests for POST /api/v1/analysis/policy-tree"""

    @pytest.mark.asyncio
    async def test_policy_tree_success(self, client, sequential_analysis_request):
        """Should successfully generate policy tree."""
        response = await client.post(
            "/api/v1/analysis/policy-tree",
            json=sequential_analysis_request
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "schema_version" in data
        assert data["schema_version"] == "policy_tree.v1"
        assert "root" in data
        assert "total_stages" in data
        assert "total_nodes" in data

        # Check root node
        root = data["root"]
        assert "node_id" in root
        assert "stage" in root
        assert "node_type" in root
        assert "expected_value" in root

    @pytest.mark.asyncio
    async def test_policy_tree_has_children(self, client, sequential_analysis_request):
        """Policy tree root should have children."""
        response = await client.post(
            "/api/v1/analysis/policy-tree",
            json=sequential_analysis_request
        )

        assert response.status_code == 200
        data = response.json()

        root = data["root"]
        assert "children" in root
        # Root is a decision node, should have children (actions)


class TestStageSensitivityEndpoint:
    """Tests for POST /api/v1/analysis/stage-sensitivity"""

    @pytest.mark.asyncio
    async def test_stage_sensitivity_success(self, client, stage_sensitivity_request):
        """Should successfully analyze stage sensitivity."""
        response = await client.post(
            "/api/v1/analysis/stage-sensitivity",
            json=stage_sensitivity_request
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "schema_version" in data
        assert data["schema_version"] == "stage_sensitivity.v1"
        assert "stage_results" in data
        assert "overall_robustness" in data
        assert "most_sensitive_parameters" in data
        assert "explanation" in data

        # Check robustness bounds
        assert 0 <= data["overall_robustness"] <= 1

        # Check explanation
        assert "summary" in data["explanation"]

    @pytest.mark.asyncio
    async def test_stage_sensitivity_with_parameters(self, client, stage_sensitivity_request):
        """Should work with specified parameters."""
        stage_sensitivity_request["parameters_to_vary"] = ["probability", "payoff"]

        response = await client.post(
            "/api/v1/analysis/stage-sensitivity",
            json=stage_sensitivity_request
        )

        assert response.status_code == 200
        data = response.json()

        assert data["overall_robustness"] >= 0


class TestPhase4Integration:
    """End-to-end integration tests for Phase 4."""

    @pytest.mark.asyncio
    async def test_full_decision_workflow(self, client, sequential_analysis_request):
        """Test complete workflow: analyze, get tree, check sensitivity."""
        # Step 1: Analyze sequential decision
        response1 = await client.post(
            "/api/v1/analysis/sequential",
            json=sequential_analysis_request
        )
        assert response1.status_code == 200
        analysis = response1.json()

        # Step 2: Get policy tree
        response2 = await client.post(
            "/api/v1/analysis/policy-tree",
            json=sequential_analysis_request
        )
        assert response2.status_code == 200
        tree = response2.json()

        # Step 3: Check sensitivity
        sensitivity_request = {
            "graph": sequential_analysis_request["graph"],
            "stages": sequential_analysis_request["stages"],
            "variation_range": 0.2
        }
        response3 = await client.post(
            "/api/v1/analysis/stage-sensitivity",
            json=sensitivity_request
        )
        assert response3.status_code == 200
        sensitivity = response3.json()

        # Verify consistency (value-correct, not just presence: same fixture as
        # test_sequential_analysis_success, hand-derived expected_total_value 36930).
        assert analysis["optimal_policy"]["expected_total_value"] == pytest.approx(36930.0)
        assert tree["root"]["expected_value"] is not None
        assert sensitivity["overall_robustness"] >= 0

    @pytest.mark.asyncio
    async def test_conditional_to_sequential_workflow(self, client, conditional_recommend_request):
        """Test workflow from conditional recommendations."""
        # Get conditional recommendations
        response = await client.post(
            "/api/v1/analysis/conditional-recommend",
            json=conditional_recommend_request
        )
        assert response.status_code == 200
        data = response.json()

        # Verify we got actionable conditions
        assert data["primary_recommendation"] is not None
        assert data["robustness_summary"]["recommendation_stability"] in [
            "robust", "moderate", "fragile"
        ]


class TestPhase4ErrorHandling:
    """Tests for error handling in Phase 4 endpoints."""

    @pytest.mark.asyncio
    async def test_invalid_graph_structure(self, client):
        """Should handle invalid graph structure."""
        request = {
            "graph": {
                "nodes": [],  # Empty nodes
                "edges": [],
                "stage_assignments": {}
            },
            "stages": [],
            "discount_factor": 0.95
        }

        response = await client.post(
            "/api/v1/analysis/sequential",
            json=request
        )

        # Should fail validation
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_duplicate_node_ids(self, client):
        """Should reject duplicate node IDs."""
        request = {
            "graph": {
                "nodes": [
                    {"id": "duplicate", "type": "decision", "label": "Node 1"},
                    {"id": "duplicate", "type": "terminal", "label": "Node 2", "payoff": 100}
                ],
                "edges": [],
                "stage_assignments": {"duplicate": 0}
            },
            "stages": [
                {"stage_index": 0, "stage_label": "Stage 0", "decision_nodes": ["duplicate"]}
            ],
            "discount_factor": 0.95
        }

        response = await client.post(
            "/api/v1/analysis/sequential",
            json=request
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_discount_factor(self, client, sequential_analysis_request):
        """Should reject discount factor outside valid range."""
        sequential_analysis_request["discount_factor"] = 1.5  # Invalid (> 1)

        response = await client.post(
            "/api/v1/analysis/sequential",
            json=sequential_analysis_request
        )

        assert response.status_code == 422


class TestPhase4ResponseTimes:
    """Performance tests for Phase 4 endpoints."""

    @pytest.mark.asyncio
    async def test_conditional_recommend_response_time(self, client, conditional_recommend_request):
        """Conditional recommend should respond within 500ms for typical graphs."""
        import time

        start = time.time()
        response = await client.post(
            "/api/v1/analysis/conditional-recommend",
            json=conditional_recommend_request
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should be under 500ms for typical graph
        assert elapsed < 2.0  # Allow 2s for CI environments

    @pytest.mark.asyncio
    async def test_sequential_analysis_response_time(self, client, sequential_analysis_request):
        """Sequential analysis should respond within 2s for 3-stage problems."""
        import time

        start = time.time()
        response = await client.post(
            "/api/v1/analysis/sequential",
            json=sequential_analysis_request
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should be under 2s for 3-stage problem
        assert elapsed < 5.0  # Allow 5s for CI environments


# ---------------------------------------------------------------------------
# D-12: engine input-rejection ValueErrors must surface as 422, never 500.
# ---------------------------------------------------------------------------
#
# The sequential engine fails loud (ValueError) on dangling edges and cycles
# (A2 fix). The router previously mapped any Exception -> 500; D-12 maps these
# client-input rejections to the repo's 422 validation-error envelope, matching
# the robustness v2 handler (`except ValueError: HTTPException(422, str(e))`).
#
# This fixture mounts the phase4 router with the PRODUCTION exception handlers on
# a local app so the ValueError->422 mapping is exercised at the router level
# regardless of whether the global app has mounted the route yet (C4 precedes the
# C5 selective mount). It asserts the real Olumi ErrorResponse envelope.


@pytest_asyncio.fixture
async def sequential_error_client():
    """Local app: phase4 sequential route + production HTTPException/Exception handlers."""
    from fastapi import FastAPI, HTTPException

    from src.api import main as isl_main
    from src.api.phase4 import sequential_router as phase4_sequential_router

    test_app = FastAPI()
    test_app.include_router(phase4_sequential_router, prefix="/api/v1/analysis")
    test_app.add_exception_handler(HTTPException, isl_main.http_exception_handler)
    test_app.add_exception_handler(Exception, isl_main.global_exception_handler)

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as ac:
        yield ac


class TestSequentialEngineErrorMapping:
    """D-12: engine ValueError (dangling edge / cycle) -> 422, not 500."""

    @pytest.mark.asyncio
    async def test_dangling_edge_returns_422_envelope(self, sequential_error_client):
        """RED at HEAD: dangling edge -> ValueError -> generic 500.

        `root -> ghost` where `ghost` is never defined as a node. Pydantic does not
        validate edge endpoints against `nodes`, so the request reaches the engine,
        which fails loud. That is a client-input rejection -> 422 (Olumi envelope),
        never a 500.
        """
        request = {
            "graph": {
                "nodes": [
                    {"id": "root", "type": "decision", "label": "Root"},
                    {"id": "win", "type": "terminal", "label": "Win", "payoff": 100},
                ],
                "edges": [
                    {"from": "root", "to": "ghost", "action": "risky"},
                    {"from": "root", "to": "win", "action": "safe"},
                ],
                "stage_assignments": {"root": 0, "win": 1},
            },
            "stages": [
                {"stage_index": 0, "stage_label": "Root", "decision_nodes": ["root"]},
                {"stage_index": 1, "stage_label": "Terminal", "decision_nodes": []},
            ],
            "discount_factor": 0.95,
        }
        response = await sequential_error_client.post(
            "/api/v1/analysis/sequential", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert body["code"] == "ISL_VALIDATION_ERROR"
        # message carries the engine's diagnostic naming the dangling node
        assert "ghost" in body["message"]

    @pytest.mark.asyncio
    async def test_cycle_returns_422_envelope(self, sequential_error_client):
        """RED at HEAD: a cycle -> ValueError -> generic 500.

        a -> b -> a is a directed cycle; backward induction requires a DAG and
        fails loud. -> 422, not 500.
        """
        request = {
            "graph": {
                "nodes": [
                    {"id": "a", "type": "decision", "label": "A"},
                    {"id": "b", "type": "chance", "label": "B"},
                    {"id": "end", "type": "terminal", "label": "End", "payoff": 10},
                ],
                "edges": [
                    {"from": "a", "to": "b", "action": "go"},
                    {"from": "b", "to": "a", "outcome": "loop", "probability": 0.5},
                    {"from": "b", "to": "end", "outcome": "done", "probability": 0.5},
                ],
                "stage_assignments": {"a": 0, "b": 1, "end": 2},
            },
            "stages": [
                {"stage_index": 0, "stage_label": "A", "decision_nodes": ["a"]},
                {
                    "stage_index": 1,
                    "stage_label": "B",
                    "decision_nodes": [],
                    "resolution_nodes": ["b"],
                },
                {"stage_index": 2, "stage_label": "Terminal", "decision_nodes": []},
            ],
            "discount_factor": 0.95,
        }
        response = await sequential_error_client.post(
            "/api/v1/analysis/sequential", json=request
        )
        assert response.status_code == 422
        body = response.json()
        assert body["reason"] == "validation_failed"
        assert body["source"] == "isl"
        assert "cycle" in body["message"].lower()
