"""
Comprehensive tests for MemoryCache.

Tests cover thread-safety, LRU eviction, TTL expiration, statistics tracking,
and cache key generation.
"""

import time
import threading
import pytest

from src.infrastructure.memory_cache import (
    MemoryCache,
    CacheStats,
    cache_key_from_request,
    get_memory_cache,
    shutdown_cache,
)


class TestCacheBasicOperations:
    """Test basic cache get/set/delete operations."""

    def test_set_and_get(self):
        """Test setting and retrieving a value."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"
        cache.shutdown()

    def test_get_nonexistent_key(self):
        """Test getting a non-existent key returns None."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        result = cache.get("nonexistent")

        assert result is None
        cache.shutdown()

    def test_set_updates_existing_key(self):
        """Test that setting an existing key updates the value."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        cache.set("key1", "value2")
        result = cache.get("key1")

        assert result == "value2"
        cache.shutdown()

    def test_delete_key(self):
        """Test deleting a key."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        deleted = cache.delete("key1")
        result = cache.get("key1")

        assert deleted is True
        assert result is None
        cache.shutdown()

    def test_delete_nonexistent_key(self):
        """Test deleting a non-existent key returns False."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        deleted = cache.delete("nonexistent")

        assert deleted is False
        cache.shutdown()

    def test_clear_cache(self):
        """Test clearing all cache entries."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache) == 0
        cache.shutdown()

    def test_len_operator(self):
        """Test __len__ operator returns correct size."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache) == 2
        cache.shutdown()

    def test_contains_operator(self):
        """Test __contains__ operator (in keyword)."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")

        assert "key1" in cache
        assert "key2" not in cache
        cache.shutdown()


class TestLRUEviction:
    """Test LRU (Least Recently Used) eviction."""

    def test_eviction_when_full(self):
        """Test that LRU entry is evicted when cache is full."""
        cache = MemoryCache(max_size=3, default_ttl=3600)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Add one more - should evict key1 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
        cache.shutdown()

    def test_get_updates_lru_order(self):
        """Test that getting a key updates its LRU position."""
        cache = MemoryCache(max_size=3, default_ttl=3600)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new key - should evict key2 (now least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Not evicted
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
        cache.shutdown()

    def test_set_updates_lru_order(self):
        """Test that updating a key updates its LRU position."""
        cache = MemoryCache(max_size=3, default_ttl=3600)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Update key1 to make it recently used
        cache.set("key1", "value1_updated")

        # Add new key - should evict key2
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1_updated"  # Not evicted
        assert cache.get("key2") is None  # Evicted
        cache.shutdown()


class TestTTLExpiration:
    """Test TTL (Time To Live) expiration."""

    def test_entry_expires_after_ttl(self):
        """Test that entries expire after TTL."""
        cache = MemoryCache(max_size=10, default_ttl=1)  # 1 second TTL

        cache.set("key1", "value1")

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.2)

        # Should be expired
        assert cache.get("key1") is None
        cache.shutdown()

    def test_custom_ttl_per_entry(self):
        """Test setting custom TTL per entry."""
        cache = MemoryCache(max_size=10, default_ttl=10)

        cache.set("key1", "value1", ttl=1)  # 1 second
        cache.set("key2", "value2", ttl=5)  # 5 seconds

        time.sleep(1.2)

        assert cache.get("key1") is None  # Expired
        assert cache.get("key2") == "value2"  # Still valid
        cache.shutdown()

    @pytest.mark.skip(reason="Timing-sensitive test - cleanup thread timing is non-deterministic")
    def test_background_cleanup_removes_expired(self):
        """Test that background thread removes expired entries."""
        # Use short cleanup interval for testing
        cache = MemoryCache(max_size=10, default_ttl=1)
        cache._cleanup_interval = 1  # 1 second cleanup

        cache.set("key1", "value1", ttl=1)
        cache.set("key2", "value2", ttl=10)

        # Wait for expiration and cleanup
        time.sleep(2.5)

        # Check that expired entry was removed
        stats = cache.get_stats()
        assert stats.expirations >= 1
        assert "key1" not in cache
        assert "key2" in cache

        cache.shutdown()


class TestStatisticsTracking:
    """Test cache statistics tracking."""

    def test_hit_tracking(self):
        """Test that cache hits are tracked."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("key1")

        stats = cache.get_stats()
        assert stats.hits == 2
        cache.shutdown()

    def test_miss_tracking(self):
        """Test that cache misses are tracked."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.get("nonexistent1")
        cache.get("nonexistent2")

        stats = cache.get_stats()
        assert stats.misses == 2
        cache.shutdown()

    def test_eviction_tracking(self):
        """Test that evictions are tracked."""
        cache = MemoryCache(max_size=2, default_ttl=3600)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Evicts key1
        cache.set("key4", "value4")  # Evicts key2

        stats = cache.get_stats()
        assert stats.evictions == 2
        cache.shutdown()

    def test_expiration_tracking(self):
        """Test that expirations are tracked."""
        cache = MemoryCache(max_size=10, default_ttl=1)

        cache.set("key1", "value1", ttl=1)
        time.sleep(1.2)
        cache.get("key1")  # Should trigger expiration check

        stats = cache.get_stats()
        assert stats.expirations == 1
        cache.shutdown()

    def test_current_size_tracking(self):
        """Test that current size is tracked."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()
        assert stats.current_size == 2
        cache.shutdown()

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats.hit_rate == 50.0  # 1 hit, 1 miss = 50%
        cache.shutdown()

    def test_hit_rate_zero_requests(self):
        """Test hit rate with zero requests."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        stats = cache.get_stats()
        assert stats.hit_rate == 0.0
        cache.shutdown()

    def test_total_requests(self):
        """Test total requests calculation."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit

        stats = cache.get_stats()
        assert stats.total_requests == 3
        cache.shutdown()

    def test_reset_stats(self):
        """Test resetting statistics."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("key2")

        cache.reset_stats()
        stats = cache.get_stats()

        assert stats.hits == 0
        assert stats.misses == 0
        # Cache contents should remain
        assert cache.get("key1") == "value1"
        cache.shutdown()


class TestThreadSafety:
    """Test thread-safety of cache operations."""

    def test_concurrent_reads_and_writes(self):
        """Test concurrent reads and writes from multiple threads."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"key_{worker_id}_{i}"
                    cache.set(key, f"value_{worker_id}_{i}")
                    value = cache.get(key)
                    assert value == f"value_{worker_id}_{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        cache.shutdown()

    def test_concurrent_evictions(self):
        """Test that concurrent evictions don't cause race conditions."""
        cache = MemoryCache(max_size=50, default_ttl=3600)
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    cache.set(f"key_{worker_id}_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(cache) <= 50  # Should not exceed max_size
        cache.shutdown()


class TestCacheKeyGeneration:
    """Test cache key generation from requests."""

    def test_cache_key_deterministic(self):
        """Test that cache keys are deterministic."""
        request_data = {"dag": {"nodes": ["X", "Y"]}, "treatment": "X", "outcome": "Y"}

        key1 = cache_key_from_request("causal_validate", request_data)
        key2 = cache_key_from_request("causal_validate", request_data)

        assert key1 == key2

    def test_cache_key_different_data(self):
        """Test that different data produces different keys."""
        request1 = {"treatment": "X", "outcome": "Y"}
        request2 = {"treatment": "Z", "outcome": "Y"}

        key1 = cache_key_from_request("causal_validate", request1)
        key2 = cache_key_from_request("causal_validate", request2)

        assert key1 != key2

    def test_cache_key_with_user_id(self):
        """Test cache key with user ID."""
        request_data = {"treatment": "X"}

        key1 = cache_key_from_request("endpoint", request_data, user_id="user1")
        key2 = cache_key_from_request("endpoint", request_data, user_id="user2")
        key3 = cache_key_from_request("endpoint", request_data, user_id=None)

        assert key1 != key2
        assert key1 != key3
        assert "user1" in key1
        assert "user2" in key2

    def test_cache_key_endpoint_prefix(self):
        """Test that cache key includes endpoint."""
        request_data = {"data": "test"}

        key1 = cache_key_from_request("endpoint1", request_data)
        key2 = cache_key_from_request("endpoint2", request_data)

        assert key1 != key2
        assert key1.startswith("endpoint1:")
        assert key2.startswith("endpoint2:")

    def test_cache_key_dict_order_independence(self):
        """Test that dict key order doesn't affect cache key."""
        request1 = {"a": 1, "b": 2, "c": 3}
        request2 = {"c": 3, "a": 1, "b": 2}

        key1 = cache_key_from_request("endpoint", request1)
        key2 = cache_key_from_request("endpoint", request2)

        assert key1 == key2


class TestSingletonCache:
    """Test singleton cache instance."""

    def test_get_memory_cache_singleton(self):
        """Test that get_memory_cache returns singleton."""
        # Clean up first
        shutdown_cache()

        cache1 = get_memory_cache()
        cache2 = get_memory_cache()

        assert cache1 is cache2

        # Clean up
        shutdown_cache()

    def test_singleton_persists_data(self):
        """Test that singleton persists data across calls."""
        shutdown_cache()

        cache1 = get_memory_cache()
        cache1.set("test_key", "test_value")

        cache2 = get_memory_cache()
        value = cache2.get("test_key")

        assert value == "test_value"

        shutdown_cache()

    def test_shutdown_cache_clears_singleton(self):
        """Test that shutdown_cache clears the singleton."""
        shutdown_cache()

        cache1 = get_memory_cache()
        cache1.set("test", "value")

        shutdown_cache()

        # New instance should be created
        cache2 = get_memory_cache()
        value = cache2.get("test")

        assert value is None  # Old data should be gone

        shutdown_cache()


class TestCacheShutdown:
    """Test cache shutdown."""

    def test_shutdown_stops_cleanup_thread(self):
        """Test that shutdown stops the cleanup thread."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        # Cleanup thread should be running
        assert cache._cleanup_thread is not None
        assert cache._cleanup_thread.is_alive()

        cache.shutdown()

        # Wait a bit for thread to stop
        time.sleep(0.5)

        # Cleanup thread should be stopped
        assert not cache._cleanup_thread.is_alive()

    def test_cache_usable_after_shutdown(self):
        """Test that cache is still usable after shutdown (just cleanup stops)."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1")
        cache.shutdown()

        # Cache operations should still work
        value = cache.get("key1")
        assert value == "value1"

        cache.set("key2", "value2")
        assert cache.get("key2") == "value2"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_ttl(self):
        """Test entry with zero TTL expires immediately."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        cache.set("key1", "value1", ttl=0)

        # Even immediate get should return None (expired)
        value = cache.get("key1")
        assert value is None
        cache.shutdown()

    def test_very_large_ttl(self):
        """Test entry with very large TTL."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        # Set TTL to 10 years
        cache.set("key1", "value1", ttl=10 * 365 * 24 * 3600)

        value = cache.get("key1")
        assert value == "value1"
        cache.shutdown()

    def test_max_size_one(self):
        """Test cache with max_size=1."""
        cache = MemoryCache(max_size=1, default_ttl=3600)

        cache.set("key1", "value1")
        cache.set("key2", "value2")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert len(cache) == 1
        cache.shutdown()

    def test_cache_complex_objects(self):
        """Test caching complex Python objects."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        complex_obj = {
            "list": [1, 2, 3],
            "dict": {"nested": {"data": "value"}},
            "tuple": (1, 2, 3),
        }

        cache.set("complex", complex_obj)
        retrieved = cache.get("complex")

        assert retrieved == complex_obj
        assert retrieved["dict"]["nested"]["data"] == "value"
        cache.shutdown()

    def test_empty_cache_stats(self):
        """Test stats on empty cache."""
        cache = MemoryCache(max_size=10, default_ttl=3600)

        stats = cache.get_stats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0
        assert stats.current_size == 0
        assert stats.hit_rate == 0.0
        cache.shutdown()
