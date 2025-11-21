"""
Tests for production-grade in-memory cache.

Tests cover:
- Basic get/set operations
- TTL expiration
- LRU eviction
- Thread safety
- Cache statistics
- Edge cases
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.memory_cache import (
    MemoryCache,
    CacheEntry,
    CacheStats,
    cache_key_from_request,
    get_memory_cache,
    shutdown_cache,
)


class TestMemoryCache:
    """Tests for MemoryCache class."""

    def test_basic_get_set(self):
        """Test basic cache get/set operations."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        # Set and get value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Get non-existent key
        assert cache.get("nonexistent") == None

        # Stats updated correctly
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.current_size == 1

        cache.shutdown()

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = MemoryCache(max_size=100, default_ttl=1)  # 1 second TTL

        # Set value with short TTL
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)

        # Value should be expired
        assert cache.get("key1") == None

        # Stats should show expiration
        stats = cache.get_stats()
        assert stats.expirations == 1
        assert stats.current_size == 0

        cache.shutdown()

    def test_lru_eviction(self):
        """Test LRU eviction when max_size reached."""
        cache = MemoryCache(max_size=3, default_ttl=3600)

        # Fill cache to max
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4 - should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still present (recently used)
        assert cache.get("key2") == None  # Evicted
        assert cache.get("key3") == "value3"  # Still present
        assert cache.get("key4") == "value4"  # New entry

        # Stats should show eviction
        stats = cache.get_stats()
        assert stats.evictions == 1
        assert stats.current_size == 3

        cache.shutdown()

    def test_update_existing_key(self):
        """Test updating existing key value."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        # Set initial value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Update value
        cache.set("key1", "value2")
        assert cache.get("key1") == "value2"

        # Size should not increase
        stats = cache.get_stats()
        assert stats.current_size == 1

        cache.shutdown()

    def test_delete_key(self):
        """Test deleting keys from cache."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        # Set and delete
        cache.set("key1", "value1")
        assert cache.delete("key1") == True
        assert cache.get("key1") == None

        # Delete non-existent key
        assert cache.delete("nonexistent") == False

        # Size updated correctly
        stats = cache.get_stats()
        assert stats.current_size == 0

        cache.shutdown()

    def test_clear_cache(self):
        """Test clearing all cache entries."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        # Add multiple entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get_stats().current_size == 3

        # Clear cache
        cache.clear()

        assert cache.get("key1") == None
        assert cache.get("key2") == None
        assert cache.get("key3") == None
        assert cache.get_stats().current_size == 0

        cache.shutdown()

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        # Generate hits and misses
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        cache.get("another_miss")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.hit_rate == 50.0
        assert stats.total_requests == 4
        assert stats.current_size == 1

        cache.shutdown()

    def test_reset_stats(self):
        """Test resetting statistics."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        # Generate some activity
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("nonexistent")

        # Reset stats
        cache.reset_stats()

        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0
        # current_size should NOT reset
        assert stats.current_size == 1

        cache.shutdown()

    def test_thread_safety(self):
        """Test thread-safe operations."""
        cache = MemoryCache(max_size=1000, default_ttl=3600)
        num_threads = 10
        ops_per_thread = 100

        def worker(thread_id):
            for i in range(ops_per_thread):
                key = f"thread{thread_id}_key{i}"
                cache.set(key, f"value{i}")
                cache.get(key)

        # Run concurrent operations
        threads = []
        for t in range(num_threads):
            thread = threading.Thread(target=worker, args=(t,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all operations completed
        stats = cache.get_stats()
        assert stats.current_size == num_threads * ops_per_thread
        assert stats.hits == num_threads * ops_per_thread

        cache.shutdown()

    def test_background_cleanup(self):
        """Test background cleanup of expired entries."""
        cache = MemoryCache(max_size=100, default_ttl=1)

        # Set entries with short TTL
        for i in range(5):
            cache.set(f"key{i}", f"value{i}", ttl=1)

        assert cache.get_stats().current_size == 5

        # Wait for expiration + cleanup cycle
        time.sleep(2)

        # Entries should be cleaned up (but we need to access them to trigger removal)
        for i in range(5):
            assert cache.get(f"key{i}") == None

        stats = cache.get_stats()
        assert stats.current_size == 0
        assert stats.expirations == 5

        cache.shutdown()

    def test_contains_operator(self):
        """Test __contains__ operator."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        cache.set("key1", "value1")

        assert "key1" in cache
        assert "nonexistent" not in cache

        cache.shutdown()

    def test_len_operator(self):
        """Test __len__ operator."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        assert len(cache) == 0

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache) == 2

        cache.shutdown()

    def test_hit_rate_calculation(self):
        """Test hit rate percentage calculation."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 75.0

        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 100.0

    def test_cache_with_complex_values(self):
        """Test caching complex data structures."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        # Test with dict
        cache.set("dict_key", {"nested": {"data": [1, 2, 3]}})
        assert cache.get("dict_key") == {"nested": {"data": [1, 2, 3]}}

        # Test with list
        cache.set("list_key", [1, 2, {"key": "value"}])
        assert cache.get("list_key") == [1, 2, {"key": "value"}]

        cache.shutdown()


class TestCacheKeyGeneration:
    """Tests for cache_key_from_request helper."""

    def test_deterministic_keys(self):
        """Test that identical requests generate identical keys."""
        request1 = {"dag": {"nodes": ["X", "Y"]}, "treatment": "X"}
        request2 = {"dag": {"nodes": ["X", "Y"]}, "treatment": "X"}

        key1 = cache_key_from_request("causal_validate", request1)
        key2 = cache_key_from_request("causal_validate", request2)

        assert key1 == key2

    def test_different_data_different_keys(self):
        """Test that different requests generate different keys."""
        request1 = {"dag": {"nodes": ["X", "Y"]}, "treatment": "X"}
        request2 = {"dag": {"nodes": ["X", "Z"]}, "treatment": "X"}

        key1 = cache_key_from_request("causal_validate", request1)
        key2 = cache_key_from_request("causal_validate", request2)

        assert key1 != key2

    def test_key_includes_endpoint(self):
        """Test that endpoint name is included in key."""
        request = {"data": "test"}

        key1 = cache_key_from_request("endpoint1", request)
        key2 = cache_key_from_request("endpoint2", request)

        assert key1 != key2
        assert "endpoint1" in key1
        assert "endpoint2" in key2

    def test_key_includes_user_id(self):
        """Test that user ID is included when provided."""
        request = {"data": "test"}

        key1 = cache_key_from_request("endpoint", request, user_id="user1")
        key2 = cache_key_from_request("endpoint", request, user_id="user2")
        key3 = cache_key_from_request("endpoint", request)

        assert key1 != key2
        assert key1 != key3
        assert "user1" in key1
        assert "user2" in key2

    def test_key_order_independence(self):
        """Test that dict key order doesn't affect cache key."""
        request1 = {"a": 1, "b": 2, "c": 3}
        request2 = {"c": 3, "a": 1, "b": 2}

        key1 = cache_key_from_request("endpoint", request1)
        key2 = cache_key_from_request("endpoint", request2)

        assert key1 == key2

    def test_key_format(self):
        """Test cache key format."""
        request = {"data": "test"}

        key = cache_key_from_request("endpoint", request, user_id="user1")

        # Should be: endpoint:user1:hash
        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "endpoint"
        assert parts[1] == "user1"
        assert len(parts[2]) == 16  # Hash is 16 chars


class TestCacheSingleton:
    """Tests for get_memory_cache singleton."""

    def test_singleton_returns_same_instance(self):
        """Test that get_memory_cache returns same instance."""
        # Reset singleton
        shutdown_cache()

        cache1 = get_memory_cache()
        cache2 = get_memory_cache()

        assert cache1 is cache2

    def test_singleton_configuration(self):
        """Test that singleton uses settings configuration."""
        # Reset singleton
        shutdown_cache()

        with patch("src.infrastructure.memory_cache.get_settings") as mock_settings:
            settings = MagicMock()
            settings.CACHE_MAX_SIZE = 500
            settings.CACHE_DEFAULT_TTL = 7200
            mock_settings.return_value = settings

            cache = get_memory_cache()

            assert cache.max_size == 500
            assert cache.default_ttl == 7200

        shutdown_cache()

    def test_shutdown_cache(self):
        """Test that shutdown_cache cleans up properly."""
        cache = get_memory_cache()
        cache.set("key1", "value1")

        shutdown_cache()

        # New instance should be created
        new_cache = get_memory_cache()
        assert new_cache.get("key1") == None

        shutdown_cache()


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test CacheEntry creation."""
        entry = CacheEntry(
            value="test_value", expires_at=time.time() + 3600, created_at=time.time()
        )

        assert entry.value == "test_value"
        assert entry.hit_count == 0

    def test_cache_entry_hit_count(self):
        """Test hit count tracking."""
        entry = CacheEntry(
            value="test_value", expires_at=time.time() + 3600, created_at=time.time()
        )

        entry.hit_count += 1
        assert entry.hit_count == 1


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_hit_rate_property(self):
        """Test hit_rate property calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 80.0

    def test_total_requests_property(self):
        """Test total_requests property."""
        stats = CacheStats(hits=100, misses=50)
        assert stats.total_requests == 150

    def test_zero_division_handling(self):
        """Test hit_rate with zero requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
        assert stats.total_requests == 0
