"""
Profile ISL endpoints to identify bottlenecks.

Uses cProfile and time measurements for detailed analysis.
"""

import cProfile
import pstats
import io
import time
import statistics
from typing import Dict, List, Callable
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.services.counterfactual_engine import CounterfactualEngine
from src.services.causal_validator import CausalValidator
from src.models.requests import CounterfactualRequest, CausalValidationRequest
from src.models.shared import DAGStructure, StructuralModel, Distribution, DistributionType


class PerformanceProfiler:
    """Profile endpoint performance and identify bottlenecks."""

    def __init__(self):
        self.results = {}

    def profile_endpoint(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict:
        """Profile endpoint with cProfile."""

        # Warmup
        print(f"  Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            try:
                func()
            except Exception as e:
                print(f"  Warmup error: {e}")
                return None

        # Profile
        print(f"  Profiling ({iterations} iterations)...")
        profiler = cProfile.Profile()
        latencies = []

        for i in range(iterations):
            start = time.perf_counter()
            profiler.enable()
            try:
                func()
            except Exception as e:
                print(f"  Error on iteration {i}: {e}")
                profiler.disable()
                continue
            profiler.disable()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        if not latencies:
            print(f"  ERROR: No successful iterations for {name}")
            return None

        # Analyze
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        return {
            "name": name,
            "iterations": len(latencies),
            "p50_ms": latencies_sorted[n // 2],
            "p95_ms": latencies_sorted[int(n * 0.95)],
            "p99_ms": latencies_sorted[int(n * 0.99)],
            "mean_ms": statistics.mean(latencies),
            "max_ms": max(latencies),
            "min_ms": min(latencies),
            "hotspots": s.getvalue()
        }

    def profile_causal_validation(self):
        """Profile causal validation."""
        print("\n🔍 Profiling causal validation...")

        validator = CausalValidator()

        # Simple model
        def run_simple():
            request = CausalValidationRequest(
                dag=DAGStructure(
                    nodes=["X", "Y", "Z"],
                    edges=[("X", "Y"), ("Y", "Z")]
                ),
                treatment="X",
                outcome="Z"
            )
            validator.validate(request)

        result_simple = self.profile_endpoint("causal_validation_simple", run_simple, iterations=50)

        # Complex model
        def run_complex():
            request = CausalValidationRequest(
                dag=DAGStructure(
                    nodes=["A", "B", "C", "D", "E", "F", "G", "H"],
                    edges=[
                        ("A", "B"), ("B", "C"), ("C", "D"),
                        ("A", "E"), ("E", "F"), ("F", "D"),
                        ("G", "B"), ("G", "E"), ("H", "F"), ("H", "C")
                    ]
                ),
                treatment="A",
                outcome="D"
            )
            validator.validate(request)

        result_complex = self.profile_endpoint("causal_validation_complex", run_complex, iterations=30)

        return {
            "simple": result_simple,
            "complex": result_complex
        }

    def profile_counterfactual(self):
        """Profile counterfactual generation."""
        print("\n🔍 Profiling counterfactual generation...")

        engine = CounterfactualEngine()

        # Small model
        def run_small():
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
            engine.analyze(request)

        result_small = self.profile_endpoint("counterfactual_small", run_small, iterations=30)

        # Large model (more complex equations)
        def run_large():
            model = StructuralModel(
                variables=["A", "B", "C", "D", "E"],
                equations={
                    "B": "2 * A + 5",
                    "C": "B * 1.5 + 3",
                    "D": "B + C",
                    "E": "D * 0.8 + C * 0.2"
                },
                distributions={
                    "A": Distribution(
                        type=DistributionType.NORMAL,
                        parameters={"mean": 50.0, "std": 10.0}
                    )
                }
            )
            request = CounterfactualRequest(
                model=model,
                intervention={"A": 75.0},
                outcome="E",
                context={}
            )
            engine.analyze(request)

        result_large = self.profile_endpoint("counterfactual_large", run_large, iterations=20)

        return {
            "small": result_small,
            "large": result_large
        }

    def generate_report(self) -> str:
        """Generate performance report."""
        report = ["# ISL Performance Profile Baseline\n"]
        report.append("_Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "_\n")

        for category, results in self.results.items():
            report.append(f"\n## {category.replace('_', ' ').title()}\n")

            if isinstance(results, dict):
                # Check if nested results (like simple/large)
                for subcategory, result in results.items():
                    if result is None:
                        continue
                    report.append(f"\n### {subcategory.title()}\n")
                    report.append(f"- Iterations: {result['iterations']}")
                    report.append(f"- P50: {result['p50_ms']:.1f}ms")
                    report.append(f"- P95: {result['p95_ms']:.1f}ms")
                    report.append(f"- P99: {result['p99_ms']:.1f}ms")
                    report.append(f"- Mean: {result['mean_ms']:.1f}ms")
                    report.append(f"- Min: {result['min_ms']:.1f}ms")
                    report.append(f"- Max: {result['max_ms']:.1f}ms\n")

                    # Extract top hotspots
                    hotspot_lines = result['hotspots'].split('\n')
                    report.append("#### Top Hotspots\n```")
                    # Get first 25 lines (header + top functions)
                    report.append('\n'.join(hotspot_lines[:25]))
                    report.append("```\n")

        return "\n".join(report)

    def identify_bottlenecks(self) -> List[str]:
        """Identify top bottlenecks from profiling data."""
        bottlenecks = []

        # Analyze counterfactual performance
        if "counterfactual" in self.results:
            cf_results = self.results["counterfactual"]
            if cf_results and "small" in cf_results and cf_results["small"]:
                small_p95 = cf_results["small"]["p95_ms"]
                if small_p95 > 1000:  # >1s for small model
                    bottlenecks.append(
                        f"⚠️  Counterfactual (small model): P95 = {small_p95:.0f}ms (target: <1000ms)"
                    )

            if cf_results and "large" in cf_results and cf_results["large"]:
                large_p95 = cf_results["large"]["p95_ms"]
                if large_p95 > 3000:  # >3s for large model
                    bottlenecks.append(
                        f"⚠️  Counterfactual (large model): P95 = {large_p95:.0f}ms (target: <3000ms)"
                    )

        # Analyze validation performance
        if "causal_validation" in self.results:
            val_results = self.results["causal_validation"]
            if val_results and "simple" in val_results and val_results["simple"]:
                simple_p95 = val_results["simple"]["p95_ms"]
                if simple_p95 > 500:  # >500ms for simple validation
                    bottlenecks.append(
                        f"⚠️  Causal validation (simple): P95 = {simple_p95:.0f}ms (target: <500ms)"
                    )

        if not bottlenecks:
            bottlenecks.append("✅ No major bottlenecks identified - all endpoints within targets")

        return bottlenecks


# Run profiling
if __name__ == "__main__":
    print("=" * 60)
    print("ISL Performance Profiling")
    print("=" * 60)

    profiler = PerformanceProfiler()

    # Profile causal validation
    profiler.results['causal_validation'] = profiler.profile_causal_validation()

    # Profile counterfactual
    profiler.results['counterfactual'] = profiler.profile_counterfactual()

    # Generate report
    print("\n📊 Generating report...")
    report = profiler.generate_report()

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "PROFILE_REPORT_BASELINE.md")
    with open(report_path, "w") as f:
        f.write(report)

    # Identify bottlenecks
    print("\n🎯 Bottleneck Analysis:")
    bottlenecks = profiler.identify_bottlenecks()
    for bottleneck in bottlenecks:
        print(f"  {bottleneck}")

    # Add bottlenecks to report
    with open(report_path, "a") as f:
        f.write("\n\n## Bottleneck Analysis\n\n")
        for bottleneck in bottlenecks:
            f.write(f"{bottleneck}\n")

    print(f"\n✅ Profile complete: {report_path}")
    print("=" * 60)
