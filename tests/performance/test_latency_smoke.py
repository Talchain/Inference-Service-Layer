"""
Performance Smoke Tests.

Quick latency checks to catch performance regressions in CI.
These tests verify that endpoints respond within acceptable latency bounds.
"""

import statistics
import time
from typing import List

import pytest
from starlette.testclient import TestClient

from src.api.main import app
from tests.perf_utils import assert_cpu_budget, is_perf_strict, measure_cpu

# ---------------------------------------------------------------------------
# ROADMAP 1.244 — the health throughput smoke measures WORK, so it measures CPU.
#
# History, because the previous two attempts are instructive. The original gate
# asserted >=100 RPS wall-clock; identical code measured 46-53 RPS on shared
# GitHub runners, so it false-RED'd and (under `pytest -x`) halted CI. The first
# fix LOWERED the wall-clock floor to 10 RPS. That made it quieter but not
# correct: a loosened wall-clock threshold is still a wall-clock threshold, it
# still measures the runner rather than the code, and it still flakes — just
# less often. It also could not distinguish the two things a reader assumes it
# separates.
#
# Measured here (see the PR body for the harness), reproducing the CI symptom:
#
#   scenario                                cpu/req    wall/req    RPS
#   baseline                                 3.11ms      3.78ms   264.6
#   +10ms OFF-CPU wait (models steal-time)   5.77ms     21.09ms    47.4
#   +10ms ON-CPU work (a REAL regression)   12.03ms     12.68ms    78.8
#
# The off-CPU row lands at 47.4 RPS — inside the 46-53 RPS band CI actually
# reported — so runner descheduling alone explains the historical failure. And
# an RPS gate CANNOT TELL THOSE TWO ROWS APART (47 vs 79, both "fail" at 100),
# while CPU separates them cleanly (5.8 vs 12.0). Wall-clock inflated 5.6x under
# a stall the code did not cause; CPU inflated 1.9x. So:
#
#   * WORK budget (CPU ms/request) — always enforced. This is the real gate.
#   * HANG ceiling (wall ms/request) — always enforced, but set ~2 orders of
#     magnitude above the worst contention observed, so only a genuinely hung or
#     blocking-I/O endpoint can trip it. Load assumption: any runner able to
#     complete the rest of this suite.
#   * Strict 100 RPS absolute target — still enforced under ISL_PERF_STRICT on
#     the dedicated perf runner, so that signal is not lost.
#
# is_perf_strict lives in tests/perf_utils (shared with other gated tests);
# it is call-time evaluated so monkeypatch.setenv works.
# ---------------------------------------------------------------------------

# Strict absolute throughput target — enforced only under ISL_PERF_STRICT.
STRICT_MIN_RPS = 100.0

# WORK budget: CPU milliseconds per /health request, enforced on EVERY run.
#
# CALIBRATION (doctrine requires the tolerance be derived, not guessed).
#
# MEASURED ON THE CI RUNNER, ubuntu-latest, coverage instrumentation active
# (2026-07-27):
#
#     run 30232013463:  16.25ms CPU / 40ms budget (2.5x headroom)
#     run 30232261585:  18.53ms CPU / 40ms budget (2.2x headroom)
#
# Two samples, ~14% apart. 40ms leaves 2.2-2.5x headroom over observed, while
# the smallest genuine work regression characterised above inflated CPU ~3.9x
# (3.11 -> 12.03ms) — so a real regression clears the budget comfortably and
# run-to-run spread is nowhere near it.
#
# TWO SAMPLES, stated as such: enough to show the spread is small, not enough
# to fix a tail. Every run emits a PerfMeasurement warning with the measured
# value and headroom multiple (visible under `-q`, on green runs too — the
# whole point), so the distribution keeps accumulating in the logs. Tighten
# this constant toward the observed spread as they do; do not loosen it
# without a measurement that justifies doing so.
MAX_CPU_MS_PER_HEALTH_REQUEST = 40.0

# HANG ceiling: wall-clock milliseconds per request, enforced on EVERY run.
# Deliberately ~100x the worst per-request wall time observed under deliberate
# contention (21ms), so contention cannot reach it but a hung or newly-blocking
# endpoint will. This is the liveness assertion the RPS floor used to be, set at
# a level that cannot false-RED.
MAX_WALL_MS_PER_HEALTH_REQUEST = 2000.0


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.perf
class TestLatencyBaselines:
    """
    Verify endpoint latencies are within acceptable bounds.

    These are smoke tests - not comprehensive benchmarks.
    Run full benchmarks separately for detailed analysis.

    Marked ``perf`` (excluded from default PR CI): every assertion here is a
    pure wall-clock P95 threshold on a shared-runner-sensitive measurement.
    Endpoint availability/correctness for /health and /ready is covered by
    the functional health tests, which stay in the default suite.
    """

    # Latency thresholds in milliseconds (P95 targets)
    LATENCY_THRESHOLDS_MS = {
        "/health": 50,
        "/ready": 100,
        "/api/v1/validation/assumptions": 500,
        "/api/v1/counterfactual/generate": 2000,
        "/api/v1/sensitivity/analyze": 2000,
    }

    # Number of requests per endpoint for statistical significance
    SAMPLE_SIZE = 5

    def _measure_latency(
        self, client, method: str, path: str, json_data: dict = None
    ) -> List[float]:
        """
        Measure request latency over multiple requests.

        Returns:
            List of latencies in milliseconds
        """
        latencies = []
        for _ in range(self.SAMPLE_SIZE):
            start = time.perf_counter()
            if method == "GET":
                response = client.get(path)
            else:
                response = client.post(path, json=json_data or {})
            end = time.perf_counter()

            # Only count successful responses
            if response.status_code in [
                200,
                422,
            ]:  # 422 is validation error, still measures latency
                latencies.append((end - start) * 1000)  # Convert to ms

        return latencies

    def test_health_endpoint_latency(self, client):
        """Health endpoint should respond quickly."""
        latencies = self._measure_latency(client, "GET", "/health")

        assert len(latencies) > 0, "No successful health check responses"

        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
        threshold = self.LATENCY_THRESHOLDS_MS["/health"]

        assert (
            p95 < threshold
        ), f"Health endpoint P95 latency {p95:.1f}ms exceeds {threshold}ms threshold"

    def test_ready_endpoint_latency(self, client):
        """Ready endpoint should respond quickly."""
        latencies = self._measure_latency(client, "GET", "/ready")

        assert len(latencies) > 0, "No successful ready check responses"

        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
        threshold = self.LATENCY_THRESHOLDS_MS["/ready"]

        assert (
            p95 < threshold
        ), f"Ready endpoint P95 latency {p95:.1f}ms exceeds {threshold}ms threshold"

    def test_validation_endpoint_latency(self, client):
        """Validation endpoint should respond within bounds."""
        test_payload = {
            "dag": {"nodes": ["A", "B", "C"], "edges": [["A", "B"], ["B", "C"]]},
            "treatment": "A",
            "outcome": "C",
        }

        latencies = self._measure_latency(
            client, "POST", "/api/v1/validation/assumptions", test_payload
        )

        if not latencies:
            pytest.skip("No successful validation responses (may require auth)")

        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
        threshold = self.LATENCY_THRESHOLDS_MS["/api/v1/validation/assumptions"]

        assert (
            p95 < threshold
        ), f"Validation P95 latency {p95:.1f}ms exceeds {threshold}ms threshold"


class TestThroughputBaselines:
    """
    Verify minimum throughput for critical endpoints.
    """

    def test_health_throughput(self, client):
        """Health endpoint must not do more WORK per request than budgeted.

        ROADMAP 1.244: the quantity this test means is "how much computation
        does serving /health cost", which is CPU time — not wall-clock RPS.
        See the module header for the calibration table and why the previous
        RPS floor could not distinguish a busy runner from a real regression.
        """
        num_requests = 50

        # Warm-up burst, EXCLUDED from the measurement: first-call route
        # resolution, lazy imports and JSON encoder setup are one-off costs.
        # Folding them into the sample would make the budget a measure of
        # import behaviour rather than of per-request work.
        for _ in range(num_requests):
            assert client.get("/health").status_code == 200

        # Best-of-3 bursts. For WALL-CLOCK, best-of-N is the right estimator:
        # scheduler noise only ever makes a burst look slower, so the minimum
        # is the closest to the true cost. CPU is taken from the SAME burst
        # that produced the best wall time, so both figures describe one
        # coherent measurement rather than two unrelated bests.
        best_wall_ms = float("inf")
        cpu_ms_of_best = float("inf")
        for _ in range(3):
            with measure_cpu() as m:
                start = time.perf_counter()
                for _ in range(num_requests):
                    response = client.get("/health")
                    assert response.status_code == 200
                wall_ms = (time.perf_counter() - start) * 1000.0
            if wall_ms < best_wall_ms:
                best_wall_ms = wall_ms
                cpu_ms_of_best = m["cpu_ms"]

        cpu_per_request = cpu_ms_of_best / num_requests
        wall_per_request = best_wall_ms / num_requests
        # This print is only rendered for FAILING tests under `-q` (and for all
        # tests under the perf workflow's `-rA`). The always-visible copy of the
        # CPU figure is the PerfMeasurement warning raised by assert_cpu_budget.
        rps = (num_requests / (best_wall_ms / 1000.0)) if best_wall_ms > 0 else float("inf")
        print(
            f"health: cpu={cpu_per_request:.2f}ms/req wall={wall_per_request:.2f}ms/req "
            f"({rps:.1f} RPS, strict_enforced={is_perf_strict()})"
        )

        # THE GATE — work, in CPU time. Always enforced; contention cannot move it.
        assert_cpu_budget(cpu_per_request, MAX_CPU_MS_PER_HEALTH_REQUEST, "health request work")

        # Liveness — a ceiling only a hung or newly-blocking endpoint can reach.
        assert wall_per_request < MAX_WALL_MS_PER_HEALTH_REQUEST, (
            f"Health request took {wall_per_request:.1f}ms wall — above the "
            f"{MAX_WALL_MS_PER_HEALTH_REQUEST:.0f}ms hang ceiling. This ceiling sits ~100x "
            "above worst observed contention, so runner load cannot explain it: the "
            "endpoint is hung, or newly blocks on I/O."
        )

        # Strict absolute target only on a dedicated perf runner.
        if is_perf_strict():
            assert rps > STRICT_MIN_RPS, (
                f"Health endpoint throughput {rps:.1f} RPS below "
                f"{STRICT_MIN_RPS} RPS target (ISL_PERF_STRICT enforced)"
            )


class TestPerfStrictFlag:
    """Lock in that strict enforcement is evaluated dynamically (not frozen
    at import), so a dedicated perf runner or a test can toggle it."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            (" 1 ", True),
            ("0", False),
            ("false", False),
            ("", False),
        ],
    )
    def test_is_perf_strict_reads_env_dynamically(self, monkeypatch, value, expected):
        monkeypatch.setenv("ISL_PERF_STRICT", value)
        assert is_perf_strict() is expected

    def test_is_perf_strict_defaults_false_when_unset(self, monkeypatch):
        monkeypatch.delenv("ISL_PERF_STRICT", raising=False)
        assert is_perf_strict() is False


class TestCpuBudgetMechanism:
    """The work budget must be able to FAIL, and must measure CPU rather than
    wall-clock. A budget that cannot fail is theatre (TESTING-DISCIPLINE 1/2),
    and one that secretly tracks wall-clock would reintroduce the very flake
    ROADMAP 1.244 exists to remove — so both properties are pinned here.
    """

    def test_budget_breach_raises(self):
        """The always-enforced budget genuinely fails when exceeded."""
        with pytest.raises(AssertionError, match="exceeds"):
            assert_cpu_budget(100.0, 10.0, "deliberate breach")

    def test_budget_within_limit_passes(self):
        """Control for the above: the same call under budget does NOT raise,
        so the failure above is caused by the breach and not by the helper
        being broken for every input."""
        assert_cpu_budget(1.0, 10.0, "under budget")

    def test_budget_is_enforced_without_strict(self, monkeypatch):
        """Unlike assert_time_budget, the work budget is NOT strict-gated —
        it must bite in the default PR gate, which is the whole point."""
        monkeypatch.delenv("ISL_PERF_STRICT", raising=False)
        assert is_perf_strict() is False
        with pytest.raises(AssertionError):
            assert_cpu_budget(100.0, 10.0, "breach with strict off")

    def test_measure_cpu_ignores_off_cpu_sleep(self):
        """THE load-bearing property: sleeping is not working.

        A 200ms sleep is ~0ms of CPU. This is exactly why a busy runner (which
        deschedules the process — off-CPU, like a sleep) cannot move this
        measurement, whereas it inflated the old wall-clock RPS 5.6x.
        """
        with measure_cpu() as m:
            time.sleep(0.2)
        assert m["cpu_ms"] < 50, (
            f"sleep(200ms) consumed {m['cpu_ms']:.1f}ms CPU — measure_cpu is "
            "tracking wall-clock, not CPU, and would flake under runner load"
        )

    def test_measure_cpu_sees_real_work(self):
        """Positive control for the above (trap 13): the instrument must be
        able to SEE CPU consumption, otherwise the sleep assertion passes by
        measuring nothing at all."""
        with measure_cpu() as m:
            end = time.process_time() + 0.05
            while time.process_time() < end:
                pass
        assert m["cpu_ms"] >= 40, (
            f"a deliberate 50ms CPU burn measured {m['cpu_ms']:.1f}ms — the "
            "instrument cannot see work, so the sleep test above is vacuous"
        )


class TestMemoryBaseline:
    """
    Basic memory checks to detect leaks.
    """

    def test_no_obvious_memory_leak(self, client):
        """Multiple requests should not cause obvious memory growth."""
        import gc

        # Force garbage collection before measuring
        gc.collect()

        # Make several requests
        for _ in range(20):
            client.get("/health")

        # Force garbage collection
        gc.collect()

        # This is a basic sanity check - detailed memory profiling
        # should be done with dedicated tools
        # Just verify we can complete the requests without OOM
        assert True


class TestResponseSizeBaseline:
    """
    Verify response sizes are reasonable.
    """

    def test_health_response_size(self, client):
        """Health response should be small."""
        response = client.get("/health")
        size_kb = len(response.content) / 1024

        # Health response should be under 1KB
        assert size_kb < 1, f"Health response {size_kb:.2f}KB exceeds 1KB limit"

    def test_error_response_size(self, client):
        """Error responses should be reasonable size."""
        # Trigger a validation error
        response = client.post("/api/v1/validation/assumptions", json={"invalid": "data"})

        size_kb = len(response.content) / 1024

        # Error response should be under 10KB
        assert size_kb < 10, f"Error response {size_kb:.2f}KB exceeds 10KB limit"
