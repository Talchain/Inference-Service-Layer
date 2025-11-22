"""
Comprehensive performance benchmarks for ISL.

Tests:
- Batch vs sequential processing (validates 5-10x speedup)
- Adaptive sampling improvements
- Topological sort caching benefits
- P95 latency targets
"""

import time
import statistics
from typing import List, Dict, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.services.counterfactual_engine import CounterfactualEngine
from src.services.causal_validator import CausalValidator
from src.models.requests import CounterfactualRequest, CausalValidationRequest
from src.models.shared import DAGStructure, StructuralModel, Distribution, DistributionType


class BenchmarkSuite:
    """Comprehensive performance benchmarks."""

    def __init__(self):
        self.results = {}

    def benchmark_batch_validation(self) -> Dict[str, Any]:
        """
        Benchmark batch validation vs sequential.

        Expected: 5-10x speedup for batch processing.
        """
        print("\n🔬 Benchmarking batch validation...")

        validator = CausalValidator()

        # Create 30 validation requests
        requests = []
        for i in range(30):
            request = CausalValidationRequest(
                dag=DAGStructure(
                    nodes=["A", "B", "C", "D", "E"],
                    edges=[
                        ("A", "B"), ("B", "C"), ("C", "D"),
                        ("A", "E"), ("E", "D")
                    ]
                ),
                treatment="A",
                outcome="D"
            )
            requests.append(request)

        # Sequential processing
        print("  Sequential processing...")
        start = time.perf_counter()
        for request in requests:
            validator.validate(request)
        time_sequential = time.perf_counter() - start

        # Batch processing (simulated parallel)
        print("  Parallel processing...")
        from concurrent.futures import ThreadPoolExecutor
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(validator.validate, requests))
        time_parallel = time.perf_counter() - start

        # Calculate speedup
        speedup = time_sequential / time_parallel
        print(f"  Sequential: {time_sequential*1000:.1f}ms")
        print(f"  Parallel: {time_parallel*1000:.1f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        if speedup >= 5:
            print(f"  ✅ Batch speedup exceeds 5x target!")
        elif speedup >= 3:
            print(f"  ⚠️  Batch speedup is {speedup:.1f}x (target: 5x+)")
        else:
            print(f"  ❌ Batch speedup is only {speedup:.1f}x (target: 5x+)")

        return {
            "sequential_ms": time_sequential * 1000,
            "parallel_ms": time_parallel * 1000,
            "speedup": speedup,
            "batch_size": len(requests),
            "target_met": speedup >= 5
        }

    def benchmark_batch_counterfactual(self) -> Dict[str, Any]:
        """
        Benchmark batch counterfactual vs sequential.

        Expected: 5-10x speedup for batch processing.
        """
        print("\n🔬 Benchmarking batch counterfactual...")

        engine = CounterfactualEngine()

        # Create 20 counterfactual requests
        requests = []
        for i in range(20):
            model = StructuralModel(
                variables=["X", "Y", "Z"],
                equations={"Y": "2 * X + 5", "Z": "Y + 3"},
                distributions={
                    "X": Distribution(
                        type=DistributionType.NORMAL,
                        parameters={"mean": 10.0 + i, "std": 2.0}
                    )
                }
            )
            request = CounterfactualRequest(
                model=model,
                intervention={"X": 15.0 + i},
                outcome="Z",
                context={}
            )
            requests.append(request)

        # Sequential processing
        print("  Sequential processing...")
        start = time.perf_counter()
        for request in requests:
            engine.analyze(request)
        time_sequential = time.perf_counter() - start

        # Batch processing (simulated parallel)
        print("  Parallel processing...")
        from concurrent.futures import ThreadPoolExecutor
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(engine.analyze, requests))
        time_parallel = time.perf_counter() - start

        # Calculate speedup
        speedup = time_sequential / time_parallel
        print(f"  Sequential: {time_sequential*1000:.1f}ms")
        print(f"  Parallel: {time_parallel*1000:.1f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        if speedup >= 5:
            print(f"  ✅ Batch speedup exceeds 5x target!")
        elif speedup >= 3:
            print(f"  ⚠️  Batch speedup is {speedup:.1f}x (target: 5x+)")
        else:
            print(f"  ❌ Batch speedup is only {speedup:.1f}x (target: 5x+)")

        return {
            "sequential_ms": time_sequential * 1000,
            "parallel_ms": time_parallel * 1000,
            "speedup": speedup,
            "batch_size": len(requests),
            "target_met": speedup >= 5
        }

    def benchmark_p95_latencies(self) -> Dict[str, Any]:
        """
        Verify P95 latency targets are met.

        Targets:
        - Validation: P95 < 5000ms
        - Counterfactual: P95 < 5000ms
        """
        print("\n🔬 Benchmarking P95 latencies...")

        validator = CausalValidator()
        engine = CounterfactualEngine()

        # Validation latencies
        print("  Measuring validation latencies...")
        val_latencies = []
        for _ in range(50):
            request = CausalValidationRequest(
                dag=DAGStructure(
                    nodes=["A", "B", "C", "D"],
                    edges=[("A", "B"), ("B", "C"), ("C", "D")]
                ),
                treatment="A",
                outcome="D"
            )
            start = time.perf_counter()
            validator.validate(request)
            latency = (time.perf_counter() - start) * 1000
            val_latencies.append(latency)

        val_latencies.sort()
        val_p50 = val_latencies[len(val_latencies) // 2]
        val_p95 = val_latencies[int(len(val_latencies) * 0.95)]
        val_p99 = val_latencies[int(len(val_latencies) * 0.99)]

        print(f"  Validation - P50: {val_p50:.1f}ms, P95: {val_p95:.1f}ms, P99: {val_p99:.1f}ms")

        # Counterfactual latencies
        print("  Measuring counterfactual latencies...")
        cf_latencies = []
        for _ in range(50):
            model = StructuralModel(
                variables=["X", "Y", "Z"],
                equations={"Y": "2 * X", "Z": "Y + 5"},
                distributions={
                    "X": Distribution(
                        type=DistributionType.NORMAL,
                        parameters={"mean": 10.0, "std": 2.0}
                    )
                }
            )
            request = CounterfactualRequest(
                model=model,
                intervention={"X": 15.0},
                outcome="Z",
                context={}
            )
            start = time.perf_counter()
            engine.analyze(request)
            latency = (time.perf_counter() - start) * 1000
            cf_latencies.append(latency)

        cf_latencies.sort()
        cf_p50 = cf_latencies[len(cf_latencies) // 2]
        cf_p95 = cf_latencies[int(len(cf_latencies) * 0.95)]
        cf_p99 = cf_latencies[int(len(cf_latencies) * 0.99)]

        print(f"  Counterfactual - P50: {cf_p50:.1f}ms, P95: {cf_p95:.1f}ms, P99: {cf_p99:.1f}ms")

        # Check targets
        val_target_met = val_p95 < 5000
        cf_target_met = cf_p95 < 5000

        if val_target_met and cf_target_met:
            print(f"  ✅ All P95 targets met (<5000ms)")
        else:
            print(f"  ⚠️  Some P95 targets exceeded")

        return {
            "validation": {
                "p50_ms": val_p50,
                "p95_ms": val_p95,
                "p99_ms": val_p99,
                "target_met": val_target_met
            },
            "counterfactual": {
                "p50_ms": cf_p50,
                "p95_ms": cf_p95,
                "p99_ms": cf_p99,
                "target_met": cf_target_met
            }
        }

    def benchmark_adaptive_sampling_benefit(self) -> Dict[str, Any]:
        """
        Measure adaptive sampling benefit on various models.

        Tests both low-variance and high-variance models.
        """
        print("\n🔬 Benchmarking adaptive sampling benefit...")

        # Low variance model (should converge fast)
        print("  Low-variance model...")
        model_low = StructuralModel(
            variables=["X", "Y"],
            equations={"Y": "3 * X + 10"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 20.0, "std": 0.5}  # Low std
                )
            }
        )
        request_low = CounterfactualRequest(
            model=model_low,
            intervention={"X": 25.0},
            outcome="Y",
            context={}
        )

        # Fixed sampling
        engine_fixed = CounterfactualEngine(enable_adaptive_sampling=False)
        start = time.perf_counter()
        for _ in range(10):
            engine_fixed.analyze(request_low)
        time_fixed_low = (time.perf_counter() - start) * 1000

        # Adaptive sampling
        engine_adaptive = CounterfactualEngine(enable_adaptive_sampling=True)
        start = time.perf_counter()
        for _ in range(10):
            engine_adaptive.analyze(request_low)
        time_adaptive_low = (time.perf_counter() - start) * 1000

        speedup_low = time_fixed_low / time_adaptive_low
        print(f"  Low-variance speedup: {speedup_low:.2f}x")

        # High variance model (may not converge as fast)
        print("  High-variance model...")
        model_high = StructuralModel(
            variables=["X", "Y"],
            equations={"Y": "3 * X + 10"},
            distributions={
                "X": Distribution(
                    type=DistributionType.NORMAL,
                    parameters={"mean": 20.0, "std": 10.0}  # High std
                )
            }
        )
        request_high = CounterfactualRequest(
            model=model_high,
            intervention={"X": 25.0},
            outcome="Y",
            context={}
        )

        # Fixed sampling
        start = time.perf_counter()
        for _ in range(10):
            engine_fixed.analyze(request_high)
        time_fixed_high = (time.perf_counter() - start) * 1000

        # Adaptive sampling
        start = time.perf_counter()
        for _ in range(10):
            engine_adaptive.analyze(request_high)
        time_adaptive_high = (time.perf_counter() - start) * 1000

        speedup_high = time_fixed_high / time_adaptive_high
        print(f"  High-variance speedup: {speedup_high:.2f}x")

        return {
            "low_variance": {
                "fixed_ms": time_fixed_low,
                "adaptive_ms": time_adaptive_low,
                "speedup": speedup_low
            },
            "high_variance": {
                "fixed_ms": time_fixed_high,
                "adaptive_ms": time_adaptive_high,
                "speedup": speedup_high
            }
        }

    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        print("=" * 60)
        print("ISL Performance Benchmark Suite")
        print("=" * 60)

        results = {}

        # Run benchmarks
        results["batch_validation"] = self.benchmark_batch_validation()
        results["batch_counterfactual"] = self.benchmark_batch_counterfactual()
        results["p95_latencies"] = self.benchmark_p95_latencies()
        results["adaptive_sampling"] = self.benchmark_adaptive_sampling_benefit()

        # Summary
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)

        # Batch speedup
        val_speedup = results["batch_validation"]["speedup"]
        cf_speedup = results["batch_counterfactual"]["speedup"]
        print(f"\n📦 Batch Processing:")
        print(f"  Validation speedup: {val_speedup:.2f}x")
        print(f"  Counterfactual speedup: {cf_speedup:.2f}x")

        # P95 latencies
        val_p95 = results["p95_latencies"]["validation"]["p95_ms"]
        cf_p95 = results["p95_latencies"]["counterfactual"]["p95_ms"]
        print(f"\n⏱️  P95 Latencies:")
        print(f"  Validation: {val_p95:.1f}ms (target: <5000ms)")
        print(f"  Counterfactual: {cf_p95:.1f}ms (target: <5000ms)")

        # Adaptive sampling
        low_speedup = results["adaptive_sampling"]["low_variance"]["speedup"]
        high_speedup = results["adaptive_sampling"]["high_variance"]["speedup"]
        print(f"\n🎯 Adaptive Sampling:")
        print(f"  Low-variance speedup: {low_speedup:.2f}x")
        print(f"  High-variance speedup: {high_speedup:.2f}x")

        # Overall assessment
        print(f"\n✅ Acceptance Criteria:")
        batch_ok = val_speedup >= 5 and cf_speedup >= 5
        latency_ok = val_p95 < 5000 and cf_p95 < 5000
        adaptive_ok = low_speedup > 1.0

        print(f"  {'✅' if batch_ok else '❌'} Batch speedup ≥ 5x")
        print(f"  {'✅' if latency_ok else '❌'} P95 latency < 5s")
        print(f"  {'✅' if adaptive_ok else '❌'} Adaptive sampling benefit > 1x")

        print("=" * 60)

        return results


if __name__ == "__main__":
    suite = BenchmarkSuite()
    results = suite.run_all()

    # Save results
    import json
    output_path = os.path.join(os.path.dirname(__file__), "BENCHMARK_RESULTS.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")
