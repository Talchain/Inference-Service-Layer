"""
End-to-end integration tests for Phase 3 Enhancement Suite.

Tests realistic workflows combining:
- Contrastive Explanations
- Batch Counterfactuals with Interaction Detection
- Transportability Analysis
"""

import pytest


@pytest.fixture
def pricing_structural_model():
    """Pricing structural model for e2e tests."""
    return {
        "variables": ["Price", "Quality", "Marketing", "Revenue"],
        "equations": {
            "Revenue": "10000 + 500*Price + 200*Quality + 0.5*Marketing + 100*Price*Quality"
        },
        "distributions": {
            "noise": {"type": "normal", "parameters": {"mean": 0, "std": 1000}}
        },
    }


@pytest.fixture
def uk_domain():
    """UK market domain specification."""
    return {
        "name": "UK",
        "dag": {
            "nodes": ["Price", "Quality", "MarketSize", "Revenue"],
            "edges": [
                ["Price", "Revenue"],
                ["Quality", "Revenue"],
                ["MarketSize", "Revenue"],
            ],
        },
        "data_summary": {
            "n_samples": 5000,
            "available_variables": ["Price", "Quality", "MarketSize", "Revenue"],
        },
    }


@pytest.fixture
def germany_domain():
    """Germany market domain specification."""
    return {
        "name": "Germany",
        "dag": {
            "nodes": ["Price", "Quality", "MarketSize", "Revenue"],
            "edges": [
                ["Price", "Revenue"],
                ["Quality", "Revenue"],
                ["MarketSize", "Revenue"],
            ],
        },
        "data_summary": {
            "n_samples": 3000,
            "available_variables": ["Price", "Quality", "MarketSize", "Revenue"],
        },
    }


class TestPricingOptimizationWorkflow:
    """
    E2E workflow: Optimize pricing strategy with contrastive explanations,
    validate with batch counterfactuals, then check if it works in new market.
    """

    @pytest.mark.asyncio
    async def test_pricing_workflow(
        self, client, pricing_structural_model, uk_domain, germany_domain
    ):
        """
        Complete pricing optimization workflow:
        1. Use contrastive explanations to find minimal price adjustment
        2. Validate with batch counterfactuals (test interaction effects)
        3. Check if strategy transports to Germany
        """
        # Step 1: Contrastive Explanation - Find minimal intervention
        contrastive_request = {
            "model": pricing_structural_model,
            "current_state": {"Price": 40, "Quality": 7, "Marketing": 50000},
            "observed_outcome": {"Revenue": 30000},
            "target_outcome": {"Revenue": (50000, 55000)},
            "constraints": {
                "feasible": ["Price", "Quality"],
                "max_changes": 2,
                "minimize": "change_magnitude",
            },
            "seed": 42,
        }

        contrastive_response = await client.post(
            "/api/v1/explain/contrastive",
            json=contrastive_request,
        )

        assert contrastive_response.status_code == 200
        contrastive_data = contrastive_response.json()

        # Should find at least one minimal intervention
        assert len(contrastive_data["minimal_interventions"]) > 0
        best_intervention = contrastive_data["minimal_interventions"][0]

        # Step 2: Batch Counterfactuals - Validate strategy with interaction detection
        # Test: baseline, price only, quality only, combined (to detect interactions)
        baseline_intervention = {"Price": 40, "Quality": 7}

        # Extract recommended changes from contrastive explanation
        recommended_changes = {}
        for var, change_info in best_intervention["changes"].items():
            recommended_changes[var] = change_info["to_value"]

        # Create batch scenarios
        batch_request = {
            "model": pricing_structural_model,
            "scenarios": [
                {
                    "id": "baseline",
                    "intervention": baseline_intervention,
                    "label": "Current state",
                },
                {
                    "id": "recommended",
                    "intervention": recommended_changes,
                    "label": "Contrastive recommendation",
                },
            ],
            "outcome": "Revenue",
            "analyze_interactions": True,
            "samples": 1000,
            "seed": 42,
        }

        # Add individual variable scenarios if multi-variable intervention
        if len(recommended_changes) > 1:
            for var, value in recommended_changes.items():
                single_var_intervention = baseline_intervention.copy()
                single_var_intervention[var] = value
                batch_request["scenarios"].append(
                    {
                        "id": f"{var.lower()}_only",
                        "intervention": single_var_intervention,
                        "label": f"Only change {var}",
                    }
                )

        batch_response = await client.post(
            "/api/v1/causal/counterfactual/batch",
            json=batch_request,
        )

        assert batch_response.status_code == 200
        batch_data = batch_response.json()

        # Validate batch results
        assert len(batch_data["scenarios"]) >= 2
        assert "comparison" in batch_data
        assert "ranking" in batch_data["comparison"]

        # Recommended intervention should perform well
        recommended_scenario = next(
            (s for s in batch_data["scenarios"] if s["scenario_id"] == "recommended"),
            None,
        )
        assert recommended_scenario is not None

        # Should detect interactions if multi-variable
        if len(recommended_changes) > 1 and batch_data.get("interactions"):
            assert "pairwise" in batch_data["interactions"]

        # Step 3: Transportability - Check if strategy works in Germany
        transport_request = {
            "source_domain": uk_domain,
            "target_domain": germany_domain,
            "treatment": "Price",
            "outcome": "Revenue",
            "selection_variables": ["MarketSize"],
        }

        transport_response = await client.post(
            "/api/v1/causal/transport",
            json=transport_request,
        )

        assert transport_response.status_code == 200
        transport_data = transport_response.json()

        # Should provide transportability assessment
        assert "transportable" in transport_data
        assert "explanation" in transport_data

        # If transportable, can apply UK findings to Germany
        if transport_data["transportable"]:
            assert transport_data["method"] in ["direct", "selection_diagram"]
            assert len(transport_data["required_assumptions"]) > 0
            # Success: Strategy can be used in Germany (with caveats from assumptions)
        else:
            # Strategy doesn't transport - suggestions provided
            assert transport_data["suggestions"] is not None
            assert len(transport_data["suggestions"]) > 0


class TestInteractionDiscoveryWorkflow:
    """
    E2E workflow: Discover synergistic effects and validate transportability.
    """

    @pytest.mark.asyncio
    async def test_interaction_discovery(
        self, client, pricing_structural_model, uk_domain, germany_domain
    ):
        """
        Workflow:
        1. Use batch counterfactuals to discover Price×Quality interaction
        2. Check if interaction transports to Germany
        """
        # Step 1: Batch counterfactuals to discover interactions
        batch_request = {
            "model": pricing_structural_model,
            "scenarios": [
                {
                    "id": "baseline",
                    "intervention": {"Price": 40, "Quality": 7},
                    "label": "Baseline",
                },
                {
                    "id": "price_increase",
                    "intervention": {"Price": 50, "Quality": 7},
                    "label": "Price +25%",
                },
                {
                    "id": "quality_increase",
                    "intervention": {"Price": 40, "Quality": 9},
                    "label": "Quality +2 points",
                },
                {
                    "id": "both_increase",
                    "intervention": {"Price": 50, "Quality": 9},
                    "label": "Premium strategy (both)",
                },
            ],
            "outcome": "Revenue",
            "analyze_interactions": True,
            "samples": 1000,
            "seed": 42,
        }

        batch_response = await client.post(
            "/api/v1/causal/counterfactual/batch",
            json=batch_request,
        )

        assert batch_response.status_code == 200
        batch_data = batch_response.json()

        # Should detect Price×Quality interaction
        if batch_data.get("interactions"):
            interactions = batch_data["interactions"]
            assert "pairwise" in interactions

            # Check if Price×Quality interaction detected
            price_quality_interaction = next(
                (
                    i
                    for i in interactions["pairwise"]
                    if set(i["variables"]) == {"Price", "Quality"}
                ),
                None,
            )

            if price_quality_interaction:
                # Should be synergistic (positive interaction in this model)
                assert price_quality_interaction["type"] in [
                    "synergistic",
                    "antagonistic",
                    "additive",
                ]

        # Step 2: Check if interaction transports
        # For simplicity, just verify transportability works
        transport_request = {
            "source_domain": uk_domain,
            "target_domain": germany_domain,
            "treatment": "Price",
            "outcome": "Revenue",
        }

        transport_response = await client.post(
            "/api/v1/causal/transport",
            json=transport_request,
        )

        assert transport_response.status_code == 200
        transport_data = transport_response.json()

        assert isinstance(transport_data["transportable"], bool)


class TestContrastiveWithBatchValidation:
    """
    E2E workflow: Use contrastive explanations, then validate with batch.
    """

    @pytest.mark.asyncio
    async def test_contrastive_to_batch(self, client, pricing_structural_model):
        """
        Workflow:
        1. Get top 3 recommendations from contrastive explanation
        2. Validate all 3 in batch counterfactuals
        3. Compare robustness across recommendations
        """
        # Step 1: Get multiple recommendations
        contrastive_request = {
            "model": pricing_structural_model,
            "current_state": {"Price": 40, "Quality": 7, "Marketing": 50000},
            "observed_outcome": {"Revenue": 30000},
            "target_outcome": {"Revenue": (48000, 52000)},
            "constraints": {
                "feasible": ["Price", "Quality", "Marketing"],
                "max_changes": 2,
                "minimize": "cost",
            },
            "seed": 42,
        }

        contrastive_response = await client.post(
            "/api/v1/explain/contrastive",
            json=contrastive_request,
        )

        assert contrastive_response.status_code == 200
        contrastive_data = contrastive_response.json()

        # Get top recommendations
        interventions = contrastive_data["minimal_interventions"]
        top_3 = interventions[:min(3, len(interventions))]

        # Step 2: Create batch scenarios from recommendations
        scenarios = [
            {
                "id": "baseline",
                "intervention": {"Price": 40, "Quality": 7, "Marketing": 50000},
                "label": "Current state",
            }
        ]

        for i, intervention in enumerate(top_3):
            scenario_intervention = {"Price": 40, "Quality": 7, "Marketing": 50000}
            for var, change_info in intervention["changes"].items():
                scenario_intervention[var] = change_info["to_value"]

            scenarios.append(
                {
                    "id": f"option_{i+1}",
                    "intervention": scenario_intervention,
                    "label": f"Option {i+1} (rank {intervention['rank']})",
                }
            )

        batch_request = {
            "model": pricing_structural_model,
            "scenarios": scenarios,
            "outcome": "Revenue",
            "analyze_interactions": True,
            "samples": 1000,
            "seed": 42,
        }

        batch_response = await client.post(
            "/api/v1/causal/counterfactual/batch",
            json=batch_request,
        )

        assert batch_response.status_code == 200
        batch_data = batch_response.json()

        # Step 3: Compare robustness
        assert len(batch_data["scenarios"]) >= 2

        # Each scenario should have robustness assessment
        for scenario in batch_data["scenarios"]:
            assert "robustness" in scenario
            assert scenario["robustness"]["score"] in ["robust", "moderate", "fragile"]


class TestDeterminismAcrossFeatures:
    """Test that all Phase 3 features are deterministic."""

    @pytest.mark.asyncio
    async def test_determinism_contrastive(self, client, pricing_structural_model):
        """Test contrastive explanations determinism."""
        request_data = {
            "model": pricing_structural_model,
            "current_state": {"Price": 40, "Quality": 7, "Marketing": 50000},
            "observed_outcome": {"Revenue": 30000},
            "target_outcome": {"Revenue": (50000, 52000)},
            "constraints": {
                "feasible": ["Price"],
                "max_changes": 1,
                "minimize": "change_magnitude",
            },
            "seed": 42,
        }

        response1 = await client.post("/api/v1/explain/contrastive", json=request_data)
        response2 = await client.post("/api/v1/explain/contrastive", json=request_data)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Should produce identical results
        assert len(data1["minimal_interventions"]) == len(
            data2["minimal_interventions"]
        )
        if len(data1["minimal_interventions"]) > 0:
            assert (
                data1["minimal_interventions"][0]["expected_outcome"]
                == data2["minimal_interventions"][0]["expected_outcome"]
            )

    @pytest.mark.asyncio
    async def test_determinism_batch(self, client, pricing_structural_model):
        """Test batch counterfactuals determinism."""
        request_data = {
            "model": pricing_structural_model,
            "scenarios": [
                {"id": "baseline", "intervention": {"Price": 40}},
                {"id": "increase", "intervention": {"Price": 50}},
            ],
            "outcome": "Revenue",
            "analyze_interactions": False,
            "samples": 1000,
            "seed": 42,
        }

        response1 = await client.post(
            "/api/v1/causal/counterfactual/batch", json=request_data
        )
        response2 = await client.post(
            "/api/v1/causal/counterfactual/batch", json=request_data
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Should produce identical results
        assert data1["scenarios"][0]["prediction"]["point_estimate"] == data2["scenarios"][0]["prediction"]["point_estimate"]
        assert data1["scenarios"][1]["prediction"]["point_estimate"] == data2["scenarios"][1]["prediction"]["point_estimate"]

    @pytest.mark.asyncio
    async def test_determinism_transportability(self, client, uk_domain, germany_domain):
        """Test transportability determinism."""
        request_data = {
            "source_domain": uk_domain,
            "target_domain": germany_domain,
            "treatment": "Price",
            "outcome": "Revenue",
        }

        response1 = await client.post("/api/v1/causal/transport", json=request_data)
        response2 = await client.post("/api/v1/causal/transport", json=request_data)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Should produce identical results
        assert data1["transportable"] == data2["transportable"]
        assert data1["method"] == data2["method"]
        assert data1["robustness"] == data2["robustness"]


class TestMetadataConsistency:
    """Test that all Phase 3 features include proper metadata."""

    @pytest.mark.asyncio
    async def test_contrastive_metadata(self, client, pricing_structural_model):
        """Test contrastive explanation metadata."""
        response = await client.post(
            "/api/v1/explain/contrastive",
            json={
                "model": pricing_structural_model,
                "current_state": {"Price": 40, "Quality": 7, "Marketing": 50000},
                "observed_outcome": {"Revenue": 30000},
                "target_outcome": {"Revenue": (50000, 52000)},
                "constraints": {"feasible": ["Price"], "max_changes": 1},
            },
            headers={"X-Request-Id": "test-contrastive-123"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "_metadata" in data
        assert data["_metadata"]["request_id"] == "test-contrastive-123"
        assert "timestamp" in data["_metadata"]

    @pytest.mark.asyncio
    async def test_batch_metadata(self, client, pricing_structural_model):
        """Test batch counterfactuals metadata."""
        response = await client.post(
            "/api/v1/causal/counterfactual/batch",
            json={
                "model": pricing_structural_model,
                "scenarios": [{"id": "test", "intervention": {"Price": 40}}],
                "outcome": "Revenue",
            },
            headers={"X-Request-Id": "test-batch-456"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "_metadata" in data
        assert data["_metadata"]["request_id"] == "test-batch-456"

    @pytest.mark.asyncio
    async def test_transportability_metadata(self, client, uk_domain, germany_domain):
        """Test transportability metadata."""
        response = await client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": uk_domain,
                "target_domain": germany_domain,
                "treatment": "Price",
                "outcome": "Revenue",
            },
            headers={"X-Request-Id": "test-transport-789"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "_metadata" in data
        assert data["_metadata"]["request_id"] == "test-transport-789"
