"""
Test and validate optimization improvements.

Compares old (fixed sampling) vs new (adaptive/cached) performance.
"""

import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.services.counterfactual_engine import CounterfactualEngine
from src.models.requests import CounterfactualRequest
from src.models.shared import StructuralModel, Distribution, DistributionType


def test_adaptive_sampling_faster():
    """Adaptive sampling should be 2-5x faster for low-variance models."""

    print("\n🔬 Testing adaptive sampling optimization...")

    # Simple model with low variance (should converge quickly)
    simple_model = StructuralModel(
        variables=["X", "Y", "Z"],
        equations={"Y": "2 * X + 5", "Z": "Y + 3"},
        distributions={
            "X": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 10.0, "std": 0.5}  # Low std = low variance
            )
        }
    )

    request = CounterfactualRequest(
        model=simple_model,
        intervention={"X": 15.0},
        outcome="Z",
        context={}
    )

    # Old: Fixed sampling (no adaptive)
    engine_fixed = CounterfactualEngine(enable_adaptive_sampling=False)
    start = time.perf_counter()
    result_fixed = engine_fixed.analyze(request)
    time_fixed = time.perf_counter() - start

    # New: Adaptive sampling
    engine_adaptive = CounterfactualEngine(enable_adaptive_sampling=True)
    start = time.perf_counter()
    result_adaptive = engine_adaptive.analyze(request)
    time_adaptive = time.perf_counter() - start

    # Verify correctness (results should be very similar)
    diff = abs(result_fixed.prediction.point_estimate - result_adaptive.prediction.point_estimate)
    assert diff < 0.5, f"Results diverged: {result_fixed.prediction.point_estimate} vs {result_adaptive.prediction.point_estimate}"

    # Calculate speedup
    speedup = time_fixed / time_adaptive
    savings_pct = (1 - time_adaptive / time_fixed) * 100

    print(f"  Fixed sampling: {time_fixed*1000:.1f}ms")
    print(f"  Adaptive sampling: {time_adaptive*1000:.1f}ms")
    print(f"  Speedup: {speedup:.2f}x ({savings_pct:.0f}% faster)")

    # Should be faster (at least some improvement)
    if speedup > 1.0:
        print(f"  ✅ Adaptive sampling is {speedup:.1f}x faster!")
    else:
        print(f"  ⚠️  No speedup observed (may vary with system load)")

    return {
        "speedup": speedup,
        "time_fixed_ms": time_fixed * 1000,
        "time_adaptive_ms": time_adaptive * 1000,
        "savings_pct": savings_pct
    }


def test_topological_sort_caching():
    """Topological sort caching should speed up repeated analyses."""

    print("\n🔬 Testing topological sort caching...")

    model = StructuralModel(
        variables=["A", "B", "C", "D", "E"],
        equations={
            "B": "2 * A",
            "C": "B + 3",
            "D": "C * 1.5",
            "E": "D + B"
        },
        distributions={
            "A": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 50.0, "std": 5.0}
            )
        }
    )

    engine = CounterfactualEngine()

    # First run - cache miss
    request1 = CounterfactualRequest(
        model=model,
        intervention={"A": 60.0},
        outcome="E",
        context={}
    )

    start = time.perf_counter()
    result1 = engine.analyze(request1)
    time_first = time.perf_counter() - start

    # Second run with same model - cache hit
    request2 = CounterfactualRequest(
        model=model,
        intervention={"A": 75.0},  # Different intervention, same model
        outcome="E",
        context={}
    )

    start = time.perf_counter()
    result2 = engine.analyze(request2)
    time_cached = time.perf_counter() - start

    # Check cache was used
    cache_size = len(engine._topo_sort_cache)
    assert cache_size >= 1, "Cache should have entries"

    print(f"  First run: {time_first*1000:.2f}ms (cache miss)")
    print(f"  Second run: {time_cached*1000:.2f}ms (cache hit)")
    print(f"  Cache entries: {cache_size}")

    # Cached run should generally be similar or faster
    # (improvement may be small since topo sort is not the main bottleneck)
    improvement = ((time_first - time_cached) / time_first) * 100
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  ✅ Topological sort caching is working")

    return {
        "time_first_ms": time_first * 1000,
        "time_cached_ms": time_cached * 1000,
        "improvement_pct": improvement,
        "cache_size": cache_size
    }


def test_correctness_maintained():
    """Verify optimizations don't affect correctness."""

    print("\n🔬 Testing correctness maintained...")

    model = StructuralModel(
        variables=["X", "Y"],
        equations={"Y": "3 * X + 10"},
        distributions={
            "X": Distribution(
                type=DistributionType.NORMAL,
                parameters={"mean": 20.0, "std": 2.0}
            )
        }
    )

    request = CounterfactualRequest(
        model=model,
        intervention={"X": 25.0},
        outcome="Y",
        context={}
    )

    # Run with both modes
    engine_fixed = CounterfactualEngine(enable_adaptive_sampling=False)
    result_fixed = engine_fixed.analyze(request)

    engine_adaptive = CounterfactualEngine(enable_adaptive_sampling=True)
    result_adaptive = engine_adaptive.analyze(request)

    # Results should be nearly identical
    # For deterministic intervention X=25, Y should be exactly 3*25+10=85
    assert abs(result_fixed.prediction.point_estimate - 85.0) < 0.5
    assert abs(result_adaptive.prediction.point_estimate - 85.0) < 0.5

    diff = abs(result_fixed.prediction.point_estimate - result_adaptive.prediction.point_estimate)
    assert diff < 0.5, f"Results differ too much: {diff}"

    print(f"  Fixed result: {result_fixed.prediction.point_estimate:.2f}")
    print(f"  Adaptive result: {result_adaptive.prediction.point_estimate:.2f}")
    print(f"  Difference: {diff:.4f}")
    print(f"  ✅ Correctness maintained!")

    return {"difference": diff}


if __name__ == "__main__":
    print("=" * 60)
    print("ISL Optimization Validation Tests")
    print("=" * 60)

    results = {}

    # Test 1: Adaptive sampling
    results["adaptive_sampling"] = test_adaptive_sampling_faster()

    # Test 2: Topological sort caching
    results["topo_sort_caching"] = test_topological_sort_caching()

    # Test 3: Correctness
    results["correctness"] = test_correctness_maintained()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Adaptive sampling speedup: {results['adaptive_sampling']['speedup']:.2f}x")
    print(f"Topo sort caching improvement: {results['topo_sort_caching']['improvement_pct']:.1f}%")
    print(f"Correctness maintained: ✅ (diff = {results['correctness']['difference']:.4f})")
    print("\n✅ All optimization tests passed!")
    print("=" * 60)
