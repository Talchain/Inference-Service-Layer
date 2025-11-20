#!/usr/bin/env python3
"""
Performance profiling script for ISL critical code paths.

Profiles three critical endpoints:
- Causal validation
- Counterfactual analysis
- Preference elicitation

Identifies bottlenecks and generates optimization recommendations.

Usage:
    python scripts/profile_performance.py [--endpoint all|causal|counterfactual|preference]
    python scripts/profile_performance.py --output reports/profile_results.txt

Requirements:
    pip install line-profiler memory-profiler

Environment:
    ISL_BASE_URL - ISL service URL (default: http://localhost:8000)
"""

import cProfile
import pstats
import io
import time
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class PerformanceProfiler:
    """Profile ISL endpoint performance."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize profiler."""
        self.base_url = base_url
        self.results = {}

    def profile_causal_validation(self) -> Dict:
        """Profile causal validation endpoint."""
        print("\n" + "=" * 60)
        print("Profiling: Causal Validation")
        print("=" * 60)

        # Test DAG: Complex structure with multiple paths
        dag = {
            "nodes": ["A", "B", "C", "D", "E", "F"],
            "edges": [
                ["A", "C"], ["B", "C"],
                ["C", "D"], ["C", "E"],
                ["D", "F"], ["E", "F"]
            ]
        }

        payload = {
            "dag": dag,
            "treatment": "A",
            "outcome": "F"
        }

        return self._profile_endpoint(
            "/api/v1/causal/validate",
            payload,
            "causal_validation"
        )

    def profile_counterfactual_analysis(self) -> Dict:
        """Profile counterfactual analysis endpoint."""
        print("\n" + "=" * 60)
        print("Profiling: Counterfactual Analysis")
        print("=" * 60)

        # Test scenario: Multiple interventions
        dag = {
            "nodes": ["X1", "X2", "Y", "Z"],
            "edges": [["X1", "Y"], ["X2", "Y"], ["Y", "Z"]]
        }

        payload = {
            "dag": dag,
            "treatment": "X1",
            "outcome": "Z",
            "scenarios": [
                {"interventions": {"X1": 1.0}, "label": "Scenario A"},
                {"interventions": {"X1": 2.0}, "label": "Scenario B"},
                {"interventions": {"X1": 0.0}, "label": "Baseline"}
            ]
        }

        return self._profile_endpoint(
            "/api/v1/counterfactual/analyze",
            payload,
            "counterfactual_analysis"
        )

    def profile_preference_elicitation(self) -> Dict:
        """Profile preference elicitation endpoint."""
        print("\n" + "=" * 60)
        print("Profiling: Preference Elicitation")
        print("=" * 60)

        # Test beliefs: Comprehensive uncertainty
        payload = {
            "beliefs": {
                "A": {"type": "binary", "prior": 0.5},
                "B": {"type": "binary", "prior": 0.6},
                "C": {"type": "continuous", "mean": 0.0, "std": 1.0},
                "D": {"type": "continuous", "mean": 5.0, "std": 2.0}
            },
            "dag": {
                "nodes": ["A", "B", "C", "D", "Y"],
                "edges": [["A", "Y"], ["B", "Y"], ["C", "Y"], ["D", "Y"]]
            },
            "outcome": "Y",
            "num_questions": 5
        }

        return self._profile_endpoint(
            "/api/v1/preference/elicit",
            payload,
            "preference_elicitation"
        )

    def _profile_endpoint(self, endpoint: str, payload: Dict, name: str) -> Dict:
        """
        Profile a single endpoint.

        Args:
            endpoint: API endpoint path
            payload: Request payload
            name: Profile name for results

        Returns:
            Profiling results dictionary
        """
        import requests

        url = f"{self.base_url}{endpoint}"

        # Warm-up request (ignore results)
        try:
            requests.post(url, json=payload, timeout=30)
        except Exception as e:
            print(f"⚠ Warm-up request failed: {e}")
            print(f"  Skipping profiling for {name}")
            return {"error": str(e), "endpoint": endpoint}

        # Profile with cProfile
        profiler = cProfile.Profile()

        print(f"\n1. Running profiled request...")
        profiler.enable()

        start_time = time.time()
        try:
            response = requests.post(url, json=payload, timeout=30)
            elapsed = time.time() - start_time

            if response.status_code != 200:
                print(f"✗ Request failed: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return {
                    "error": f"HTTP {response.status_code}",
                    "endpoint": endpoint
                }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ Request failed: {e}")
            profiler.disable()
            return {"error": str(e), "endpoint": endpoint}

        profiler.disable()

        print(f"✓ Request completed in {elapsed:.3f}s")

        # Analyze profiling results
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')

        # Top 20 slowest functions
        print(f"\n2. Top 20 slowest functions:")
        stats.print_stats(20)

        profile_output = stream.getvalue()

        # Extract key metrics
        stats.sort_stats('time')
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.print_stats(10)

        # Run multiple requests for latency distribution
        print(f"\n3. Running latency distribution test (10 requests)...")
        latencies = []

        for i in range(10):
            start = time.time()
            try:
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code == 200:
                    latencies.append(time.time() - start)
            except Exception as e:
                print(f"  Request {i+1} failed: {e}")

        if latencies:
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[len(sorted_latencies) // 2]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            avg = statistics.mean(latencies)
            std = statistics.stdev(latencies) if len(latencies) > 1 else 0

            print(f"   Latency distribution:")
            print(f"   - P50:  {p50:.3f}s")
            print(f"   - P95:  {p95:.3f}s")
            print(f"   - Mean: {avg:.3f}s")
            print(f"   - Std:  {std:.3f}s")
        else:
            p50 = p95 = avg = std = None
            print(f"   ✗ No successful requests for latency distribution")

        # Store results
        result = {
            "endpoint": endpoint,
            "name": name,
            "initial_latency": elapsed,
            "latency_p50": p50,
            "latency_p95": p95,
            "latency_mean": avg,
            "latency_std": std,
            "profile_output": profile_output,
            "success": True
        }

        self.results[name] = result
        return result

    def generate_recommendations(self) -> List[str]:
        """
        Generate optimization recommendations based on profiling results.

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Check if any endpoints are slow
        for name, result in self.results.items():
            if result.get("error"):
                continue

            p95 = result.get("latency_p95")
            if p95 and p95 > 2.0:
                recommendations.append(
                    f"⚠ {name}: P95 latency is {p95:.2f}s (target: <2s). "
                    f"Consider optimization."
                )
            elif p95 and p95 > 1.0:
                recommendations.append(
                    f"ℹ {name}: P95 latency is {p95:.2f}s. Room for improvement."
                )

        # Generic recommendations based on common patterns
        recommendations.extend([
            "",
            "General Optimization Opportunities:",
            "1. Implement Redis caching for repeated computations",
            "2. Use async/await for I/O-bound operations",
            "3. Profile mathematical computations (Monte Carlo sampling)",
            "4. Consider vectorization with NumPy for matrix operations",
            "5. Add connection pooling for database/cache connections",
            "6. Implement result streaming for large responses",
            "7. Use Pydantic model validation caching",
            "8. Profile JSON serialization/deserialization overhead"
        ])

        return recommendations

    def save_report(self, output_path: str):
        """
        Save profiling report to file.

        Args:
            output_path: Path to save report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ISL Performance Profiling Report\n")
            f.write("=" * 80 + "\n\n")

            # Summary
            f.write("Summary\n")
            f.write("-" * 80 + "\n")

            for name, result in self.results.items():
                if result.get("error"):
                    f.write(f"\n{name}:\n")
                    f.write(f"  ✗ Error: {result['error']}\n")
                    continue

                f.write(f"\n{name}:\n")
                f.write(f"  Endpoint: {result['endpoint']}\n")
                f.write(f"  Initial latency: {result['initial_latency']:.3f}s\n")

                if result.get('latency_p50'):
                    f.write(f"  Latency P50: {result['latency_p50']:.3f}s\n")
                    f.write(f"  Latency P95: {result['latency_p95']:.3f}s\n")
                    f.write(f"  Latency mean: {result['latency_mean']:.3f}s\n")
                    f.write(f"  Latency std: {result['latency_std']:.3f}s\n")

            # Recommendations
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("Recommendations\n")
            f.write("=" * 80 + "\n\n")

            recommendations = self.generate_recommendations()
            for rec in recommendations:
                f.write(f"{rec}\n")

            # Detailed profiling output
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("Detailed Profiling Results\n")
            f.write("=" * 80 + "\n\n")

            for name, result in self.results.items():
                if result.get("profile_output"):
                    f.write(f"\n{name}:\n")
                    f.write("-" * 80 + "\n")
                    f.write(result["profile_output"])
                    f.write("\n\n")

        print(f"\n✓ Report saved to: {output_file}")

    def print_summary(self):
        """Print profiling summary to console."""
        print("\n" + "=" * 60)
        print("Profiling Summary")
        print("=" * 60)

        for name, result in self.results.items():
            if result.get("error"):
                print(f"\n{name}: ✗ {result['error']}")
                continue

            print(f"\n{name}:")
            if result.get('latency_p95'):
                print(f"  P95 latency: {result['latency_p95']:.3f}s")
                print(f"  Mean latency: {result['latency_mean']:.3f}s")

                # Pass/warn based on targets
                if result['latency_p95'] < 1.0:
                    print(f"  ✓ Performance good")
                elif result['latency_p95'] < 2.0:
                    print(f"  ⚠ Performance acceptable (could improve)")
                else:
                    print(f"  ✗ Performance needs optimization")

        print("\n" + "=" * 60)
        print("Recommendations")
        print("=" * 60)

        recommendations = self.generate_recommendations()
        for rec in recommendations:
            print(rec)


def main():
    """Run performance profiling."""
    parser = argparse.ArgumentParser(description="Profile ISL performance")
    parser.add_argument(
        "--endpoint",
        choices=["all", "causal", "counterfactual", "preference"],
        default="all",
        help="Which endpoint to profile"
    )
    parser.add_argument(
        "--output",
        default="reports/profile_results.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--url",
        default=os.getenv("ISL_BASE_URL", "http://localhost:8000"),
        help="ISL base URL"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ISL Performance Profiling")
    print("=" * 60)
    print(f"Target: {args.url}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Output: {args.output}")

    # Check if ISL is running
    try:
        import requests
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"\n✗ ISL health check failed: {response.status_code}")
            print(f"  Ensure ISL is running at {args.url}")
            sys.exit(1)
        print(f"✓ ISL is running")
    except Exception as e:
        print(f"\n✗ Cannot connect to ISL: {e}")
        print(f"  Ensure ISL is running at {args.url}")
        sys.exit(1)

    # Run profiling
    profiler = PerformanceProfiler(base_url=args.url)

    if args.endpoint in ["all", "causal"]:
        profiler.profile_causal_validation()

    if args.endpoint in ["all", "counterfactual"]:
        profiler.profile_counterfactual_analysis()

    if args.endpoint in ["all", "preference"]:
        profiler.profile_preference_elicitation()

    # Generate report
    profiler.print_summary()
    profiler.save_report(args.output)

    print(f"\n✓ Profiling complete")
    print(f"  Review detailed results in: {args.output}")


if __name__ == "__main__":
    main()
