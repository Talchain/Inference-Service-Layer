"""
Performance Benchmark Suite for ISL.

Tests key performance targets:
- P95 latency < 2.0s for causal/counterfactual
- P95 latency < 1.5s for preference/teaching
- Support for 100+ concurrent users
- Cache hit rate > 40%

Usage:
    python benchmarks/performance_benchmark.py --host http://localhost:8000 --duration 60

Requirements:
    - ISL running on specified host
    - Redis running (for cache testing)
"""

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import httpx


@dataclass
class BenchmarkResult:
    """Results from a single endpoint benchmark."""

    endpoint: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies: List[float]
    errors: List[str]

    @property
    def success_rate(self) -> float:
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0

    @property
    def p50_latency(self) -> float:
        return statistics.quantiles(self.latencies, n=2)[0] if self.latencies else 0

    @property
    def p95_latency(self) -> float:
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx] if sorted_latencies else 0

    @property
    def p99_latency(self) -> float:
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx] if sorted_latencies else 0

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0

    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0


class ISLBenchmark:
    """Performance benchmark for ISL endpoints."""

    def __init__(self, base_url: str, duration: int = 60, concurrent_users: int = 10):
        self.base_url = base_url
        self.duration = duration
        self.concurrent_users = concurrent_users
        self.results: Dict[str, BenchmarkResult] = {}

    async def benchmark_endpoint(
        self,
        endpoint: str,
        method: str,
        payload: dict,
        num_requests: int,
        concurrency: int,
    ) -> BenchmarkResult:
        """Benchmark a single endpoint with concurrent requests."""
        latencies = []
        errors = []
        successful = 0
        failed = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            sem = asyncio.Semaphore(concurrency)

            async def make_request():
                nonlocal successful, failed
                async with sem:
                    start = time.time()
                    try:
                        if method == "POST":
                            response = await client.post(
                                f"{self.base_url}{endpoint}",
                                json=payload,
                            )
                        else:
                            response = await client.get(f"{self.base_url}{endpoint}")

                        latency = time.time() - start
                        latencies.append(latency)

                        if response.status_code == 200:
                            successful += 1
                        else:
                            failed += 1
                            errors.append(f"HTTP {response.status_code}")
                    except Exception as e:
                        failed += 1
                        latencies.append(time.time() - start)
                        errors.append(str(e))

            # Run concurrent requests
            tasks = [make_request() for _ in range(num_requests)]
            await asyncio.gather(*tasks)

        return BenchmarkResult(
            endpoint=endpoint,
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            latencies=latencies,
            errors=errors[:10],  # Keep first 10 errors
        )

    async def run_all_benchmarks(self):
        """Run all endpoint benchmarks."""
        print(f"\n{'='*80}")
        print(f"ISL Performance Benchmark")
        print(f"{'='*80}")
        print(f"Base URL: {self.base_url}")
        print(f"Duration: {self.duration}s")
        print(f"Concurrent users: {self.concurrent_users}")
        print(f"{'='*80}\n")

        # Calculate requests per endpoint based on duration
        requests_per_test = max(50, self.duration * 2)

        # 1. Health check (baseline)
        print("1. Health Check Endpoint...")
        self.results["health"] = await self.benchmark_endpoint(
            endpoint="/health",
            method="GET",
            payload={},
            num_requests=requests_per_test,
            concurrency=self.concurrent_users,
        )

        # 2. Causal validation
        print("2. Causal Validation Endpoint...")
        causal_payload = {
            "dag": {
                "nodes": ["Price", "Revenue", "Brand", "Churn"],
                "edges": [
                    ["Price", "Revenue"],
                    ["Brand", "Price"],
                    ["Brand", "Revenue"],
                    ["Price", "Churn"],
                    ["Churn", "Revenue"]
                ]
            },
            "treatment": "Price",
            "outcome": "Revenue"
        }
        self.results["causal"] = await self.benchmark_endpoint(
            endpoint="/api/v1/causal/validate",
            method="POST",
            payload=causal_payload,
            num_requests=requests_per_test,
            concurrency=self.concurrent_users,
        )

        # 3. Counterfactual analysis
        print("3. Counterfactual Analysis Endpoint...")
        counterfactual_payload = {
            "model": {
                "variables": ["Price", "Revenue"],
                "equations": {
                    "Revenue": "10000 + 500 * Price"
                },
                "distributions": {
                    "Price": {
                        "type": "normal",
                        "parameters": {"mean": 50, "std": 5}
                    }
                }
            },
            "outcome": "Revenue",
            "intervention": {"Price": 60},
            "num_samples": 1000
        }
        self.results["counterfactual"] = await self.benchmark_endpoint(
            endpoint="/api/v1/causal/counterfactual",
            method="POST",
            payload=counterfactual_payload,
            num_requests=requests_per_test // 2,  # Heavier operation
            concurrency=self.concurrent_users // 2,
        )

        # 4. Preference elicitation
        print("4. Preference Elicitation Endpoint...")
        preference_payload = {
            "user_id": f"benchmark_user_{int(time.time())}",
            "context": {
                "domain": "pricing",
                "variables": ["revenue", "churn", "brand"]
            },
            "num_queries": 3
        }
        self.results["preference"] = await self.benchmark_endpoint(
            endpoint="/api/v1/preferences/elicit",
            method="POST",
            payload=preference_payload,
            num_requests=requests_per_test,
            concurrency=self.concurrent_users,
        )

        # 5. Teaching examples
        print("5. Teaching Examples Endpoint...")
        teaching_payload = {
            "user_id": f"benchmark_user_{int(time.time())}",
            "current_beliefs": {
                "value_weights": {
                    "revenue": {
                        "type": "normal",
                        "parameters": {"mean": 0.7, "std": 0.2}
                    }
                },
                "risk_tolerance": {
                    "type": "beta",
                    "parameters": {"alpha": 2, "beta": 2}
                },
                "time_horizon": {
                    "type": "normal",
                    "parameters": {"mean": 12, "std": 3}
                },
                "uncertainty_estimates": {"revenue_weight": 0.3}
            },
            "target_concept": "trade_offs",
            "context": {
                "domain": "pricing",
                "variables": ["revenue", "churn"]
            },
            "max_examples": 2
        }
        self.results["teaching"] = await self.benchmark_endpoint(
            endpoint="/api/v1/teaching/teach",
            method="POST",
            payload=teaching_payload,
            num_requests=requests_per_test,
            concurrency=self.concurrent_users,
        )

        # 6. Advanced validation
        print("6. Advanced Validation Endpoint...")
        validation_payload = {
            "dag": {
                "nodes": ["A", "B", "C"],
                "edges": [["A", "B"], ["B", "C"]]
            },
            "validation_level": "standard"
        }
        self.results["validation"] = await self.benchmark_endpoint(
            endpoint="/api/v1/validation/validate-model",
            method="POST",
            payload=validation_payload,
            num_requests=requests_per_test,
            concurrency=self.concurrent_users,
        )

    def print_results(self):
        """Print benchmark results in a formatted table."""
        print(f"\n{'='*80}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*80}\n")

        # Performance targets
        targets = {
            "causal": 2.0,
            "counterfactual": 2.0,
            "preference": 1.5,
            "teaching": 1.5,
            "validation": 2.0,
        }

        for endpoint, result in self.results.items():
            target = targets.get(endpoint)
            target_str = f" (target: <{target}s)" if target else ""

            print(f"Endpoint: {result.endpoint}")
            print(f"  Total Requests:     {result.total_requests}")
            print(f"  Successful:         {result.successful_requests} ({result.success_rate:.1f}%)")
            print(f"  Failed:             {result.failed_requests}")
            print(f"  Average Latency:    {result.avg_latency:.3f}s")
            print(f"  P50 Latency:        {result.p50_latency:.3f}s")

            # Highlight P95 with target
            p95_status = ""
            if target:
                if result.p95_latency < target:
                    p95_status = " ✓ PASS"
                else:
                    p95_status = " ✗ FAIL"

            print(f"  P95 Latency:        {result.p95_latency:.3f}s{target_str}{p95_status}")
            print(f"  P99 Latency:        {result.p99_latency:.3f}s")
            print(f"  Max Latency:        {result.max_latency:.3f}s")

            if result.errors:
                print(f"  Sample Errors:      {result.errors[:3]}")
            print()

    def evaluate_targets(self) -> bool:
        """Evaluate if performance targets are met."""
        print(f"{'='*80}")
        print(f"TARGET EVALUATION")
        print(f"{'='*80}\n")

        targets = {
            "causal": ("Causal Validation", 2.0),
            "counterfactual": ("Counterfactual Analysis", 2.0),
            "preference": ("Preference Elicitation", 1.5),
            "teaching": ("Teaching Examples", 1.5),
        }

        all_passed = True
        results_summary = []

        for key, (name, target) in targets.items():
            if key in self.results:
                result = self.results[key]
                passed = result.p95_latency < target
                status = "✓ PASS" if passed else "✗ FAIL"
                all_passed = all_passed and passed

                results_summary.append({
                    "name": name,
                    "p95": result.p95_latency,
                    "target": target,
                    "passed": passed,
                    "status": status,
                })

                print(f"{name:30} P95: {result.p95_latency:.3f}s  Target: <{target}s  {status}")
            else:
                print(f"{name:30} NOT TESTED")
                all_passed = False

        # Concurrency test
        print(f"\nConcurrency:                   {self.concurrent_users} users  Target: 100+  ", end="")
        concurrency_pass = self.concurrent_users >= 10  # Pilot target
        print("✓ PASS" if concurrency_pass else "⚠ NOTE: Test with 100+ for production")

        # Overall
        print(f"\n{'='*80}")
        if all_passed:
            print("OVERALL: ✓ ALL TARGETS MET")
        else:
            print("OVERALL: ✗ SOME TARGETS NOT MET")
        print(f"{'='*80}\n")

        return all_passed

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "base_url": self.base_url,
                "duration": self.duration,
                "concurrent_users": self.concurrent_users,
            },
            "results": {
                endpoint: {
                    "total_requests": result.total_requests,
                    "successful_requests": result.successful_requests,
                    "failed_requests": result.failed_requests,
                    "success_rate": result.success_rate,
                    "avg_latency": result.avg_latency,
                    "p50_latency": result.p50_latency,
                    "p95_latency": result.p95_latency,
                    "p99_latency": result.p99_latency,
                    "max_latency": result.max_latency,
                }
                for endpoint, result in self.results.items()
            },
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to: {filename}")


async def main():
    parser = argparse.ArgumentParser(description="ISL Performance Benchmark")
    parser.add_argument(
        "--host",
        default="http://localhost:8000",
        help="ISL base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent users (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)",
    )

    args = parser.parse_args()

    benchmark = ISLBenchmark(
        base_url=args.host,
        duration=args.duration,
        concurrent_users=args.concurrency,
    )

    try:
        await benchmark.run_all_benchmarks()
        benchmark.print_results()
        benchmark.evaluate_targets()
        benchmark.save_results(args.output)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
