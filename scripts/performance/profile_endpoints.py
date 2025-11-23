"""
Performance profiling for ISL endpoints.

Profiles CPU usage, memory consumption, and execution time
for all major ISL endpoints.
"""

import asyncio
import cProfile
import io
import json
import logging
import pstats
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def profile_cpu():
    """Context manager for CPU profiling."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()


class EndpointProfiler:
    """Profile ISL API endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self.results: List[Dict[str, Any]] = []

    async def profile_endpoint(
        self,
        name: str,
        method: str,
        path: str,
        payload: Dict[str, Any],
        iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Profile a single endpoint.

        Args:
            name: Endpoint name for reporting
            method: HTTP method
            path: API path
            payload: Request payload
            iterations: Number of iterations

        Returns:
            Profile results dict
        """
        logger.info(f"Profiling {name}...")

        timings = []
        memory_usage = []
        status_codes = []

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with profile_cpu() as profiler:
            for i in range(iterations):
                start_time = time.perf_counter()

                try:
                    response = await self.client.request(
                        method=method, url=path, json=payload
                    )
                    status_codes.append(response.status_code)

                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    status_codes.append(0)

                end_time = time.perf_counter()
                duration = (end_time - start_time) * 1000  # ms

                timings.append(duration)

                # Memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage.append(current_memory - initial_memory)

                # Small delay between iterations
                await asyncio.sleep(0.1)

        # Analyze profiler stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        # Calculate statistics
        timings_sorted = sorted(timings)
        n = len(timings)

        result = {
            "endpoint": name,
            "method": method,
            "path": path,
            "iterations": iterations,
            "success_rate": sum(1 for c in status_codes if c == 200) / iterations,
            "latency": {
                "mean": sum(timings) / n,
                "median": timings_sorted[n // 2],
                "p95": timings_sorted[int(n * 0.95)],
                "p99": timings_sorted[int(n * 0.99)],
                "min": min(timings),
                "max": max(timings),
            },
            "memory": {
                "mean_mb": sum(memory_usage) / len(memory_usage),
                "max_mb": max(memory_usage),
            },
            "cpu_profile_top_20": stats_stream.getvalue(),
        }

        self.results.append(result)
        logger.info(
            f"  {name}: P50={result['latency']['median']:.1f}ms, "
            f"P95={result['latency']['p95']:.1f}ms, "
            f"Memory={result['memory']['max_mb']:.1f}MB"
        )

        return result

    async def profile_all(self) -> List[Dict[str, Any]]:
        """Profile all major endpoints."""
        logger.info("Starting endpoint profiling...")

        # Sample DAG for testing
        sample_dag = {
            "nodes": ["X", "Y", "Z", "W"],
            "edges": [["X", "Y"], ["Z", "X"], ["Z", "Y"], ["W", "Z"]],
        }

        sample_model = {
            "equations": {
                "Y": {"formula": "2*X + noise", "noise_dist": "normal(0, 1)"}
            },
            "variables": ["X", "Y"],
        }

        sample_data = [
            {"X": float(i), "Y": float(i * 2 + 0.5)} for i in range(50)
        ]

        # Profile causal validation
        await self.profile_endpoint(
            name="Causal Validation",
            method="POST",
            path="/api/v1/causal/validate",
            payload={
                "dag_structure": sample_dag,
                "treatment": "X",
                "outcome": "Y",
            },
        )

        # Profile validation with strategies
        await self.profile_endpoint(
            name="Validation Strategies",
            method="POST",
            path="/api/v1/validation/strategies",
            payload={
                "dag_structure": sample_dag,
                "treatment": "X",
                "outcome": "Y",
            },
        )

        # Profile counterfactual
        await self.profile_endpoint(
            name="Counterfactual",
            method="POST",
            path="/api/v1/causal/counterfactual",
            payload={
                "model": sample_model,
                "intervention": {"X": 5.0},
                "seed": 42,
            },
        )

        # Profile conformal prediction
        await self.profile_endpoint(
            name="Conformal Prediction",
            method="POST",
            path="/api/v1/causal/conformal",
            payload={
                "model": sample_model,
                "intervention": {"X": 5.0},
                "calibration_data": sample_data,
                "alpha": 0.05,
                "seed": 42,
            },
        )

        # Profile batch counterfactuals
        await self.profile_endpoint(
            name="Batch Counterfactuals",
            method="POST",
            path="/api/v1/batch/counterfactuals",
            payload={
                "model": sample_model,
                "scenarios": [
                    {"id": "scenario1", "intervention": {"X": 3.0}},
                    {"id": "scenario2", "intervention": {"X": 5.0}},
                    {"id": "scenario3", "intervention": {"X": 7.0}},
                ],
                "analyze_interactions": True,
                "seed": 42,
            },
        )

        # Profile causal discovery
        await self.profile_endpoint(
            name="Causal Discovery",
            method="POST",
            path="/api/v1/causal/discover",
            payload={
                "data": sample_data,
                "variable_names": ["X", "Y"],
                "algorithm": "notears",
                "threshold": 0.3,
                "seed": 42,
            },
            iterations=5,  # Fewer iterations (expensive)
        )

        logger.info(f"\nProfiled {len(self.results)} endpoints")
        return self.results

    def generate_report(self, output_path: str = "performance_report.json") -> None:
        """Generate performance report."""
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "base_url": self.base_url,
            "endpoints": self.results,
            "summary": {
                "total_endpoints": len(self.results),
                "average_p50": sum(r["latency"]["median"] for r in self.results)
                / len(self.results),
                "average_p95": sum(r["latency"]["p95"] for r in self.results)
                / len(self.results),
                "slowest_endpoint": max(
                    self.results, key=lambda r: r["latency"]["p95"]
                )["endpoint"],
                "meets_p50_target": all(
                    r["latency"]["median"] < 500 for r in self.results
                ),
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n{'=' * 60}")
        logger.info("PERFORMANCE SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Endpoints profiled: {report['summary']['total_endpoints']}")
        logger.info(
            f"Average P50 latency: {report['summary']['average_p50']:.1f}ms"
        )
        logger.info(
            f"Average P95 latency: {report['summary']['average_p95']:.1f}ms"
        )
        logger.info(f"Slowest endpoint: {report['summary']['slowest_endpoint']}")
        logger.info(
            f"Meets P50 <500ms target: {'✅ YES' if report['summary']['meets_p50_target'] else '❌ NO'}"
        )
        logger.info(f"\nFull report saved to: {output_path}")

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


async def main():
    """Run profiling."""
    profiler = EndpointProfiler()

    try:
        await profiler.profile_all()
        profiler.generate_report("scripts/performance/performance_report.json")

    finally:
        await profiler.close()


if __name__ == "__main__":
    asyncio.run(main())
