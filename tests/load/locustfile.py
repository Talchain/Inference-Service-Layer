"""
Load testing with Locust.

Simulates pilot-scale load: 50 concurrent users making realistic requests.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8000
"""

import json
import random
from datetime import datetime

from locust import HttpUser, between, task


class ISLUser(HttpUser):
    """
    Simulated ISL user making realistic requests.

    Models typical usage patterns:
    - Frequent: Validation (quick)
    - Common: Counterfactual (moderate)
    - Occasional: Robustness, Deliberation (expensive)
    """

    wait_time = between(1, 3)  # 1-3 seconds between requests

    def on_start(self):
        """Initialize user session."""
        self.user_id = f"load_test_user_{random.randint(1, 1000)}"
        self.session_id = None

    @task(10)  # Most common: validation (quick, cheap)
    def validate_causal(self):
        """Test causal validation."""
        models = [
            {
                "nodes": ["X", "Y", "Z"],
                "edges": [["X", "Y"], ["Z", "Y"]],
                "treatment": "X",
                "outcome": "Y",
            },
            {
                "nodes": ["price", "demand", "revenue"],
                "edges": [["price", "demand"], ["demand", "revenue"]],
                "treatment": "price",
                "outcome": "revenue",
            },
            {
                "nodes": ["A", "B", "C", "D"],
                "edges": [["A", "B"], ["B", "C"], ["A", "D"]],
                "treatment": "A",
                "outcome": "C",
            },
        ]

        model = random.choice(models)

        with self.client.post(
            "/api/v1/causal/validate",
            json={"dag": {"nodes": model["nodes"], "edges": model["edges"]}, "treatment": model["treatment"], "outcome": model["outcome"]},
            catch_response=True,
            name="/causal/validate",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "status" in result:
                    response.success()
                else:
                    response.failure("Missing status in response")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(5)  # Second most common: counterfactual
    def generate_counterfactual(self):
        """Test counterfactual generation."""
        price = random.uniform(40, 60)

        with self.client.post(
            "/api/v1/causal/counterfactual",
            json={
                "model": {
                    "variables": ["price", "revenue"],
                    "equations": {"revenue": "100000 - 1000 * price"},
                    "distributions": {
                        "price": {
                            "type": "normal",
                            "parameters": {"mean": 50, "std": 5},
                        }
                    },
                },
                "intervention": {"price": price},
                "outcome": "revenue",
                "samples": 500,  # Smaller for load test
            },
            catch_response=True,
            name="/causal/counterfactual",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "counterfactual_outcome" in result:
                    response.success()
                else:
                    response.failure("Missing counterfactual_outcome")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(3)  # Less common: robustness (expensive)
    def analyze_robustness(self):
        """Test robustness analysis."""
        price = random.uniform(50, 60)

        with self.client.post(
            "/api/v1/robustness/analyze",
            json={
                "structural_model": {
                    "variables": ["price", "revenue"],
                    "equations": {"revenue": "100000 - 1000 * price"},
                    "distributions": {
                        "price": {
                            "type": "normal",
                            "parameters": {"mean": 50, "std": 5},
                        }
                    },
                },
                "intervention_proposal": {"price": price},
                "target_outcome": {"revenue": (90000, 100000)},
                "robustness_config": {
                    "perturbation_radius": 0.1,
                    "min_samples_per_region": 30,  # Smaller for load test
                },
            },
            catch_response=True,
            name="/robustness/analyze",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "analysis" in result:
                    response.success()
                else:
                    response.failure("Missing analysis in response")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(5)  # Preferences (moderate)
    def elicit_preferences(self):
        """Test preference elicitation."""
        with self.client.post(
            "/api/v1/preferences/elicit",
            json={
                "user_id": self.user_id,
                "causal_model": {
                    "nodes": ["X", "Y"],
                    "edges": [["X", "Y"]],
                },
                "context": {"Y": 100},
                "feature_names": ["Y"],
            },
            catch_response=True,
            name="/preferences/elicit",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "status" in result or "scenario_pair" in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(2)  # Occasional: deliberation
    def conduct_deliberation(self):
        """Test deliberation."""
        scenarios = [
            {
                "context": "Choose API design approach",
                "positions": [
                    {
                        "member_id": "eng_001",
                        "position_statement": "REST is simpler and more widely understood.",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    {
                        "member_id": "eng_002",
                        "position_statement": "GraphQL provides better flexibility for clients.",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                ],
            },
            {
                "context": "Select deployment strategy",
                "positions": [
                    {
                        "member_id": "devops_001",
                        "position_statement": "Kubernetes offers better scalability.",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    {
                        "member_id": "devops_002",
                        "position_statement": "Serverless reduces operational overhead.",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                ],
            },
        ]

        scenario = random.choice(scenarios)

        with self.client.post(
            "/api/v1/deliberation/deliberate",
            json={
                "decision_context": scenario["context"],
                "positions": scenario["positions"],
            },
            catch_response=True,
            name="/deliberation/deliberate",
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "consensus_statement" in result and "session_id" in result:
                    self.session_id = result["session_id"]
                    response.success()
                else:
                    response.failure("Missing required fields in response")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)  # Rare: health check
    def health_check(self):
        """Test health endpoint."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class StressTestUser(HttpUser):
    """
    Heavy stress testing user.

    Makes expensive requests rapidly to test system limits.
    """

    wait_time = between(0.5, 1.5)  # Faster requests

    @task
    def stress_robustness(self):
        """Stress test robustness analysis."""
        with self.client.post(
            "/api/v1/robustness/analyze",
            json={
                "structural_model": {
                    "variables": ["X", "Y", "Z"],
                    "equations": {"Y": "2*X + 3*Z", "Z": "X + 1"},
                    "distributions": {
                        "X": {
                            "type": "normal",
                            "parameters": {"mean": 0, "std": 1},
                        }
                    },
                },
                "intervention_proposal": {"X": random.uniform(-2, 2)},
                "target_outcome": {"Y": (0, 10)},
                "robustness_config": {
                    "perturbation_radius": 0.2,
                    "min_samples_per_region": 50,
                },
            },
            catch_response=True,
            name="/robustness/stress",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:  # Service overloaded
                response.failure("Service overloaded")
            else:
                response.failure(f"Stress test error: {response.status_code}")
