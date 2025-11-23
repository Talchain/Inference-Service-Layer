"""
Unit tests for caching infrastructure.

Tests TTLCache, cache decorators, and service-level caching.
"""

import time
import numpy as np
import pytest

from src.utils.cache import (
    TTLCache,
    cached,
    get_cache,
    clear_all_caches,
    get_all_cache_stats,
)


class TestTTLCache:
    """Test TTLCache basic operations."""

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = TTLCache(max_size=10, ttl=60, name="test")

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.put("key2", {"nested": "value"})
        assert cache.get("key2") == {"nested": "value"}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = TTLCache(max_size=10, ttl=60, name="test")

        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = TTLCache(max_size=10, ttl=1, name="test")  # 1 second TTL

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        assert cache.get("key1") is None

    def test_lru_eviction(self):
        """Test LRU eviction when max size reached."""
        cache = TTLCache(max_size=3, ttl=60, name="test")

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # All should be present
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Add 4th item - should evict key1 (least recently used)
        cache.put("key4", "value4")

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = TTLCache(max_size=10, ttl=60, name="test")

        cache.put("key1", "value1")

        # Miss
        cache.get("nonexistent")

        # Hit
        cache.get("key1")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["cache_size"] == 1

    def test_invalidate(self):
        """Test cache invalidation."""
        cache = TTLCache(max_size=10, ttl=60, name="test")

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        """Test cache clearing."""
        cache = TTLCache(max_size=10, ttl=60, name="test")

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.size() == 0

    def test_complex_keys(self):
        """Test caching with complex keys."""
        cache = TTLCache(max_size=10, ttl=60, name="test")

        # Dict key
        key1 = {"nodes": ["A", "B"], "edges": [("A", "B")]}
        cache.put(key1, "dag_result")
        assert cache.get(key1) == "dag_result"

        # Tuple key
        key2 = ("operation", "param1", 123)
        cache.put(key2, "result2")
        assert cache.get(key2) == "result2"


class TestCachedDecorator:
    """Test cached decorator."""

    def test_function_caching(self):
        """Test function result caching."""
        cache = TTLCache(max_size=10, ttl=60, name="test")
        call_count = [0]

        @cached(cache)
        def expensive_function(x, y):
            call_count[0] += 1
            return x + y

        # First call - should execute
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count[0] == 1

        # Second call with same args - should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count[0] == 1  # Not incremented

        # Call with different args - should execute
        result3 = expensive_function(3, 4)
        assert result3 == 7
        assert call_count[0] == 2

    def test_custom_key_function(self):
        """Test cached decorator with custom key function."""
        cache = TTLCache(max_size=10, ttl=60, name="test")

        def key_fn(x, y):
            # Only use x for caching (ignore y)
            return x

        @cached(cache, key_fn=key_fn)
        def func(x, y):
            return x + y

        # First call
        result1 = func(1, 2)
        assert result1 == 3

        # Second call with different y but same x - should use cache
        result2 = func(1, 999)
        assert result2 == 3  # Cached result from (1, 2)


class TestGlobalCaches:
    """Test global cache management."""

    def test_get_cache(self):
        """Test get_cache creates and reuses caches."""
        cache1 = get_cache("test_cache", max_size=10, ttl=60)
        cache2 = get_cache("test_cache")

        # Should return same instance
        assert cache1 is cache2

        cache1.put("key", "value")
        assert cache2.get("key") == "value"

    def test_clear_all_caches(self):
        """Test clearing all global caches."""
        cache1 = get_cache("cache1")
        cache2 = get_cache("cache2")

        cache1.put("key1", "value1")
        cache2.put("key2", "value2")

        clear_all_caches()

        assert cache1.get("key1") is None
        assert cache2.get("key2") is None

    def test_get_all_cache_stats(self):
        """Test getting stats for all caches."""
        cache1 = get_cache("cache1")
        cache2 = get_cache("cache2")

        cache1.put("key1", "value1")
        cache1.get("key1")

        cache2.put("key2", "value2")
        cache2.get("key2")

        stats = get_all_cache_stats()

        assert "cache1" in stats
        assert "cache2" in stats
        assert stats["cache1"]["hits"] >= 1
        assert stats["cache2"]["hits"] >= 1


class TestServiceCaching:
    """Test caching integration with services."""

    def test_discovery_engine_caching(self):
        """Test CausalDiscoveryEngine caching."""
        from src.services.causal_discovery_engine import CausalDiscoveryEngine

        # Create engine with caching enabled
        engine = CausalDiscoveryEngine(enable_caching=True)

        # Create test data
        np.random.seed(42)
        data = np.random.randn(100, 3)
        variable_names = ["X", "Y", "Z"]

        # First discovery - should execute
        dag1, conf1 = engine.discover_from_data(
            data, variable_names, threshold=0.3, seed=42
        )

        # Second discovery with same data - should use cache
        dag2, conf2 = engine.discover_from_data(
            data, variable_names, threshold=0.3, seed=42
        )

        # Should return same DAG structure
        assert set(dag1.nodes()) == set(dag2.nodes())
        assert set(dag1.edges()) == set(dag2.edges())
        assert conf1 == conf2

    def test_validation_suggester_caching(self):
        """Test AdvancedValidationSuggester caching."""
        from src.services.advanced_validation_suggester import (
            AdvancedValidationSuggester,
        )
        import networkx as nx

        # Create suggester with caching
        suggester = AdvancedValidationSuggester(enable_caching=True)

        # Create test DAG
        dag = nx.DiGraph()
        dag.add_edges_from([("Z", "X"), ("Z", "Y"), ("X", "Y")])

        # First call - should execute
        strategies1 = suggester.suggest_adjustment_strategies(dag, "X", "Y")

        # Second call with same DAG - should use cache
        strategies2 = suggester.suggest_adjustment_strategies(dag, "X", "Y")

        # Should return same number of strategies
        assert len(strategies1) == len(strategies2)

    def test_caching_disabled(self):
        """Test services work correctly with caching disabled."""
        from src.services.causal_discovery_engine import CausalDiscoveryEngine

        # Create engine with caching disabled
        engine = CausalDiscoveryEngine(enable_caching=False)

        np.random.seed(42)
        data = np.random.randn(50, 2)
        variable_names = ["X", "Y"]

        # Should work without errors
        dag, conf = engine.discover_from_data(data, variable_names)

        assert len(dag.nodes()) == 2


class TestCachePerformance:
    """Test cache performance benefits."""

    def test_cache_speedup(self):
        """Test that caching provides speedup for repeated queries."""
        cache = TTLCache(max_size=10, ttl=60, name="perf_test")
        execution_times = []

        @cached(cache)
        def slow_function(n):
            # Simulate expensive operation
            total = 0
            for i in range(n):
                total += i
            return total

        # First call (uncached)
        start = time.time()
        result1 = slow_function(1000000)
        time1 = time.time() - start
        execution_times.append(time1)

        # Second call (cached)
        start = time.time()
        result2 = slow_function(1000000)
        time2 = time.time() - start
        execution_times.append(time2)

        # Cached call should be much faster
        assert result1 == result2
        assert time2 < time1  # Cache should be faster
        # Typically 100-1000x faster, but at least faster


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
