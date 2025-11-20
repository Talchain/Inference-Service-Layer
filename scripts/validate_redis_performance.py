#!/usr/bin/env python3
"""
Validate Redis performance meets requirements.

Measures:
- Latency (P50, P95, P99) for set/get operations
- Throughput (operations per second)
- Memory configuration
- Configuration validation

Usage:
    python scripts/validate_redis_performance.py

Environment:
    REDIS_URL - Redis connection URL (default: redis://localhost:6379/0)
"""

import redis
import time
import statistics
import os
import sys
from typing import Dict, List


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Performance targets
TARGET_P95_LATENCY_MS = 10  # <10ms for cache operations
TARGET_THROUGHPUT_OPS = 5000  # >5000 ops/sec


def measure_latency(redis_client: redis.Redis, operations: int = 1000) -> Dict[str, float]:
    """
    Measure Redis operation latency.

    Args:
        redis_client: Redis client instance
        operations: Number of operations to measure

    Returns:
        Dictionary with p50, p95, p99 latencies in milliseconds
    """
    latencies = []

    for i in range(operations):
        key = f"perf:test:{i}"
        value = f"value_{i}" * 10  # ~100 byte value

        start = time.perf_counter()
        redis_client.setex(key, 60, value)
        redis_client.get(key)
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

    # Cleanup
    redis_client.delete(*[f"perf:test:{i}" for i in range(operations)])

    sorted_latencies = sorted(latencies)

    return {
        "p50": sorted_latencies[len(sorted_latencies) // 2],
        "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
        "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)],
        "mean": statistics.mean(latencies),
        "max": max(latencies)
    }


def measure_throughput(redis_client: redis.Redis, duration: int = 10) -> int:
    """
    Measure Redis throughput (ops/second).

    Args:
        redis_client: Redis client instance
        duration: Test duration in seconds

    Returns:
        Operations per second
    """
    operations = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        key = f"throughput:test:{operations}"
        redis_client.setex(key, 60, "value")
        redis_client.get(key)
        operations += 1

    # Cleanup
    for key in redis_client.scan_iter("throughput:*", count=1000):
        redis_client.delete(key)

    ops_per_second = operations / duration
    return int(ops_per_second)


def validate_configuration(redis_client: redis.Redis) -> Dict[str, any]:
    """
    Validate Redis configuration.

    Args:
        redis_client: Redis client instance

    Returns:
        Dictionary with configuration validation results
    """
    results = {}

    # Check eviction policy
    try:
        config = redis_client.config_get("maxmemory-policy")
        policy = config.get("maxmemory-policy", "unknown")
        results["eviction_policy"] = {
            "value": policy,
            "valid": policy == "volatile-lru",
            "expected": "volatile-lru"
        }
    except Exception as e:
        results["eviction_policy"] = {
            "value": "error",
            "valid": False,
            "error": str(e)
        }

    # Check maxmemory
    try:
        config = redis_client.config_get("maxmemory")
        maxmemory = int(config.get("maxmemory", 0))
        maxmemory_gb = maxmemory / (1024 ** 3) if maxmemory > 0 else 0

        results["maxmemory"] = {
            "value_bytes": maxmemory,
            "value_gb": maxmemory_gb,
            "valid": maxmemory > 0,
            "expected": ">0 (configured)"
        }
    except Exception as e:
        results["maxmemory"] = {
            "value": "error",
            "valid": False,
            "error": str(e)
        }

    # Check persistence
    try:
        config = redis_client.config_get("save")
        save_config = config.get("save", "")
        results["persistence"] = {
            "value": save_config,
            "valid": save_config != "",
            "expected": "RDB snapshots configured"
        }
    except Exception as e:
        results["persistence"] = {
            "value": "error",
            "valid": False,
            "error": str(e)
        }

    return results


def check_memory_usage(redis_client: redis.Redis) -> Dict[str, any]:
    """
    Check current memory usage.

    Args:
        redis_client: Redis client instance

    Returns:
        Dictionary with memory usage stats
    """
    try:
        info = redis_client.info("memory")

        used_memory_mb = info["used_memory"] / (1024 ** 2)
        maxmemory = info.get("maxmemory", 0)

        result = {
            "used_memory_mb": used_memory_mb,
            "maxmemory_mb": maxmemory / (1024 ** 2) if maxmemory > 0 else 0,
            "usage_pct": 0
        }

        if maxmemory > 0:
            result["usage_pct"] = (used_memory_mb / result["maxmemory_mb"]) * 100

        return result

    except Exception as e:
        return {"error": str(e)}


def main():
    """Run Redis performance validation."""
    print("=" * 60)
    print("Redis Performance Validation")
    print("=" * 60)

    # Connect to Redis
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        print(f"✗ Failed to connect to Redis: {e}")
        print(f"  URL: {REDIS_URL}")
        sys.exit(1)

    # Test 1: Connectivity
    print("\n1. Testing connectivity...")
    try:
        redis_client.ping()
        print("   ✓ Connected to Redis")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        sys.exit(1)

    # Test 2: Latency
    print("\n2. Measuring latency (1000 operations)...")
    try:
        latencies = measure_latency(redis_client, operations=1000)

        print(f"   P50:  {latencies['p50']:.2f} ms")
        print(f"   P95:  {latencies['p95']:.2f} ms")
        print(f"   P99:  {latencies['p99']:.2f} ms")
        print(f"   Max:  {latencies['max']:.2f} ms")
        print(f"   Mean: {latencies['mean']:.2f} ms")

        # Validate against targets
        if latencies['p95'] < TARGET_P95_LATENCY_MS:
            print(f"   ✓ Latency meets target (<{TARGET_P95_LATENCY_MS}ms P95)")
        else:
            print(f"   ⚠ Latency high (target: <{TARGET_P95_LATENCY_MS}ms P95)")

    except Exception as e:
        print(f"   ✗ Latency measurement failed: {e}")

    # Test 3: Throughput
    print("\n3. Measuring throughput (10 second test)...")
    try:
        ops_per_sec = measure_throughput(redis_client, duration=10)

        print(f"   {ops_per_sec:,} ops/second")

        # Validate against targets
        if ops_per_sec > TARGET_THROUGHPUT_OPS:
            print(f"   ✓ Throughput meets target (>{TARGET_THROUGHPUT_OPS:,} ops/sec)")
        else:
            print(f"   ⚠ Throughput low (target: >{TARGET_THROUGHPUT_OPS:,} ops/sec)")

    except Exception as e:
        print(f"   ✗ Throughput measurement failed: {e}")

    # Test 4: Memory Info
    print("\n4. Checking memory configuration...")
    try:
        memory_info = check_memory_usage(redis_client)

        if "error" in memory_info:
            print(f"   ✗ Failed to read memory info: {memory_info['error']}")
        else:
            print(f"   Used:  {memory_info['used_memory_mb']:.2f} MB")
            if memory_info['maxmemory_mb'] > 0:
                print(f"   Max:   {memory_info['maxmemory_mb']:.2f} MB")
                print(f"   Usage: {memory_info['usage_pct']:.1f}%")

                if memory_info['usage_pct'] < 50:
                    print("   ✓ Memory usage healthy")
                elif memory_info['usage_pct'] < 80:
                    print("   ⚠ Memory usage moderate")
                else:
                    print("   ✗ Memory usage high")
            else:
                print("   Max:   Unlimited")
                print("   ⚠ Maxmemory not configured")

    except Exception as e:
        print(f"   ✗ Memory check failed: {e}")

    # Test 5: Configuration Validation
    print("\n5. Validating configuration...")
    try:
        config_results = validate_configuration(redis_client)

        for config_name, result in config_results.items():
            if result.get("valid"):
                print(f"   ✓ {config_name}: {result.get('value')}")
            else:
                print(f"   ✗ {config_name}: {result.get('value')} (expected: {result.get('expected')})")
                if "error" in result:
                    print(f"      Error: {result['error']}")

    except Exception as e:
        print(f"   ✗ Configuration validation failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Redis validation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
