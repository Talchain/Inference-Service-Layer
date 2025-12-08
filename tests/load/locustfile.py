"""
Load testing for ISL using Locust.

Run with:
    # Ramp test
    locust -f tests/load/locustfile.py --users 100 --spawn-rate 10 --host http://localhost:8000

    # Sustained load
    locust -f tests/load/locustfile.py --users 50 --spawn-rate 5 --run-time 30m --host http://localhost:8000

    # Spike test
    locust -f tests/load/locustfile.py --users 200 --spawn-rate 50 --run-time 5m --host http://localhost:8000

    # Web UI
    locust -f tests/load/locustfile.py --host http://localhost:8000
    # Then open http://localhost:8089
"""

from locust import HttpUser, TaskSet, task, between, events
import json
import random
import logging

logger = logging.getLogger(__name__)


class CausalInferenceTasks(TaskSet):
    """Task set for causal inference endpoints."""

    def on_start(self):
        """Initialize test data."""
        self.sample_dag = {
            "nodes": ["Price", "Quality", "Revenue", "CustomerSatisfaction"],
            "edges": [
                ["Price", "Revenue"],
                ["Quality", "Revenue"],
                ["Quality", "CustomerSatisfaction"],
                ["Revenue", "CustomerSatisfaction"],
            ],
        }

        self.sample_model = {
            "equations": {
                "Revenue": {
                    "formula": "2.5 * Price + 1.2 * Quality + noise",
                    "noise_dist": "normal(0, 50)",
                }
            },
            "variables": ["Price", "Quality", "Revenue"],
        }

        self.sample_calibration_data = [
            {"Price": 40 + i, "Quality": 7 + (i % 3), "Revenue": 1200 + i * 25}
            for i in range(20)
        ]

        self.sample_discovery_data = [
            {
                "Price": random.uniform(30, 60),
                "Quality": random.uniform(5, 10),
                "Revenue": random.uniform(1000, 1500),
            }
            for _ in range(50)
        ]

    @task(5)
    def validate_causal(self):
        """Test causal validation (most common)."""
        payload = {
            "dag_structure": self.sample_dag,
            "treatment": "Price",
            "outcome": "Revenue",
        }

        with self.client.post(
            "/api/v1/causal/validate",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")

    @task(3)
    def counterfactual_prediction(self):
        """Test counterfactual prediction."""
        payload = {
            "model": self.sample_model,
            "intervention": {"Price": random.uniform(35, 55)},
            "seed": random.randint(1, 1000),
        }

        with self.client.post(
            "/api/v1/causal/counterfactual",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")

    @task(2)
    def conformal_prediction(self):
        """Test conformal prediction."""
        payload = {
            "model": self.sample_model,
            "intervention": {"Price": random.uniform(35, 55)},
            "calibration_data": self.sample_calibration_data,
            "alpha": 0.05,
            "seed": random.randint(1, 1000),
        }

        with self.client.post(
            "/api/v1/causal/conformal",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")

    @task(2)
    def validation_strategies(self):
        """Test validation with strategies."""
        payload = {
            "dag_structure": self.sample_dag,
            "treatment": "Price",
            "outcome": "Revenue",
        }

        with self.client.post(
            "/api/v1/validation/strategies",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")

    @task(1)
    def batch_counterfactuals(self):
        """Test batch counterfactuals."""
        payload = {
            "model": self.sample_model,
            "scenarios": [
                {"id": f"scenario{i}", "intervention": {"Price": 40 + i * 5}}
                for i in range(3)
            ],
            "analyze_interactions": True,
            "seed": random.randint(1, 1000),
        }

        with self.client.post(
            "/api/v1/batch/counterfactuals",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")

    @task(1)
    def causal_discovery(self):
        """Test causal discovery (expensive operation)."""
        payload = {
            "data": self.sample_discovery_data[:30],
            "variable_names": ["Price", "Quality", "Revenue"],
            "algorithm": "notears",
            "threshold": 0.3,
            "seed": random.randint(1, 1000),
        }

        with self.client.post(
            "/api/v1/causal/discover",
            json=payload,
            catch_response=True,
            timeout=30,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")


class HealthCheckTasks(TaskSet):
    """Task set for health monitoring endpoints."""

    @task(10)
    def health_check(self):
        """Test basic health check."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")

    @task(1)
    def service_health(self):
        """Test service health monitoring."""
        with self.client.get("/health/services", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")

    @task(1)
    def circuit_breakers(self):
        """Test circuit breaker status."""
        with self.client.get("/health/circuit-breakers", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status {response.status_code}")


class ISLUser(HttpUser):
    """Simulated ISL user."""

    wait_time = between(1, 3)
    tasks = {CausalInferenceTasks: 8, HealthCheckTasks: 2}

    def on_start(self):
        """Called when user starts."""
        logger.info(f"User {self.environment.runner.user_count} started")

    def on_stop(self):
        """Called when user stops."""
        logger.info(f"User stopped")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    logger.info("=" * 60)
    logger.info("ISL LOAD TEST STARTING")
    logger.info("=" * 60)
    logger.info(f"Host: {environment.host}")
    logger.info(f"Users: {environment.runner.target_user_count}")
    logger.info("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    stats = environment.runner.stats

    logger.info("\n" + "=" * 60)
    logger.info("LOAD TEST RESULTS")
    logger.info("=" * 60)

    total_rps = stats.total.total_rps
    total_fail_ratio = stats.total.fail_ratio

    logger.info(f"Total RPS: {total_rps:.2f}")
    logger.info(f"Failure rate: {total_fail_ratio * 100:.2f}%")
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")

    logger.info(f"\nResponse times:")
    logger.info(f"  Median (P50): {stats.total.get_response_time_percentile(0.5):.0f}ms")
    logger.info(f"  P95: {stats.total.get_response_time_percentile(0.95):.0f}ms")
    logger.info(f"  P99: {stats.total.get_response_time_percentile(0.99):.0f}ms")
    logger.info(f"  Average: {stats.total.avg_response_time:.0f}ms")
    logger.info(f"  Max: {stats.total.max_response_time:.0f}ms")

    p50 = stats.total.get_response_time_percentile(0.5)
    meets_target = p50 < 500

    logger.info(f"\nP50 <500ms target: {'✅ PASS' if meets_target else '❌ FAIL'}")
    logger.info("=" * 60)
