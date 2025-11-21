"""
Production-grade in-memory cache with LRU eviction and TTL support.

This module provides an in-memory cache optimized for ISL's caching needs,
particularly for LLM responses. It supports:
- Thread-safe operations
- LRU (Least Recently Used) eviction
- TTL (Time To Live) expiration
- Cache statistics tracking
- Optional Redis fallback

Design decisions:
- Uses OrderedDict for O(1) LRU operations
- Thread locks prevent race conditions
- Background cleanup for expired entries
- Singleton pattern for shared cache instance
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.config import get_settings


@dataclass
class CacheEntry:
    """Single cache entry with value and metadata."""

    value: Any
    expires_at: float
    created_at: float
    hit_count: int = 0


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    @property
    def total_requests(self) -> int:
        """Total cache requests (hits + misses)."""
        return self.hits + self.misses


class MemoryCache:
    """
    Thread-safe in-memory cache with LRU eviction and TTL support.

    Features:
    - O(1) get/set operations via OrderedDict
    - Automatic LRU eviction when max_size reached
    - TTL-based expiration with background cleanup
    - Thread-safe via locks
    - Detailed statistics tracking

    Usage:
        cache = MemoryCache(max_size=1000, default_ttl=3600)
        cache.set("key", "value", ttl=7200)
        value = cache.get("key")
        stats = cache.get_stats()
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries (LRU eviction triggers)
            default_ttl: Default TTL in seconds for entries
        """
        self.max_size = max_size
        self.default_ttl = default_ttl

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()  # Reentrant lock for nested calls

        # Statistics
        self._stats = CacheStats(max_size=max_size)

        # Background cleanup
        self._cleanup_interval = 60  # Check every 60 seconds
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

        # Start background cleanup
        self._start_cleanup_thread()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            # Check if expired
            if time.time() > entry.expires_at:
                # Remove expired entry
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                self._stats.current_size = len(self._cache)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1

            self._stats.hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default_ttl if None)
        """
        with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl
            expires_at = time.time() + ttl
            now = time.time()

            # Create or update entry
            entry = CacheEntry(
                value=value, expires_at=expires_at, created_at=now, hit_count=0
            )

            # If key exists, update it (moves to end automatically)
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # New entry - check if we need to evict
                if len(self._cache) >= self.max_size:
                    # Evict least recently used (first item)
                    self._cache.popitem(last=False)
                    self._stats.evictions += 1

                self._cache[key] = entry

            self._stats.current_size = len(self._cache)

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.current_size = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._stats.current_size = 0

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats object with current statistics
        """
        with self._lock:
            # Return copy to prevent external modification
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                current_size=self._stats.current_size,
                max_size=self._stats.max_size,
            )

    def reset_stats(self) -> None:
        """Reset statistics counters (not cache contents)."""
        with self._lock:
            self._stats.hits = 0
            self._stats.misses = 0
            self._stats.evictions = 0
            self._stats.expirations = 0

    def _cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self._cache.items() if now > entry.expires_at
            ]

            for key in expired_keys:
                del self._cache[key]
                self._stats.expirations += 1

            if expired_keys:
                self._stats.current_size = len(self._cache)

            return len(expired_keys)

    def _start_cleanup_thread(self) -> None:
        """Start background thread for periodic cleanup."""

        def cleanup_loop():
            while not self._stop_cleanup.is_set():
                # Wait for cleanup interval or stop signal
                if self._stop_cleanup.wait(timeout=self._cleanup_interval):
                    break  # Stop signal received

                # Run cleanup
                self._cleanup_expired()

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop, daemon=True, name="cache-cleanup"
        )
        self._cleanup_thread.start()

    def shutdown(self) -> None:
        """Shutdown cache and cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

    def __len__(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (doesn't check expiration)."""
        with self._lock:
            return key in self._cache


# Global cache instance
_cache_instance: Optional[MemoryCache] = None
_cache_lock = threading.Lock()


def get_memory_cache() -> MemoryCache:
    """
    Get singleton memory cache instance.

    Returns:
        Shared MemoryCache instance configured from settings
    """
    global _cache_instance

    if _cache_instance is None:
        with _cache_lock:
            # Double-check pattern
            if _cache_instance is None:
                settings = get_settings()

                # Get cache config from settings or use defaults
                max_size = getattr(settings, "CACHE_MAX_SIZE", 1000)
                default_ttl = getattr(settings, "CACHE_DEFAULT_TTL", 3600)

                _cache_instance = MemoryCache(max_size=max_size, default_ttl=default_ttl)

    return _cache_instance


def cache_key_from_request(
    endpoint: str, request_data: Dict[str, Any], user_id: Optional[str] = None
) -> str:
    """
    Generate deterministic cache key from request parameters.

    Args:
        endpoint: API endpoint name (e.g., "causal_validate")
        request_data: Request parameters as dict
        user_id: Optional user ID for per-user caching

    Returns:
        Deterministic cache key string

    Example:
        key = cache_key_from_request(
            "causal_validate",
            {"dag": {...}, "treatment": "X", "outcome": "Y"},
            user_id="user_123"
        )
        # Returns: "causal_validate:user_123:a1b2c3d4..."
    """
    # Sort dict for deterministic serialization
    sorted_data = json.dumps(request_data, sort_keys=True)

    # Hash for compact key
    data_hash = hashlib.sha256(sorted_data.encode()).hexdigest()[:16]

    # Build key
    parts = [endpoint]
    if user_id:
        parts.append(user_id)
    parts.append(data_hash)

    return ":".join(parts)


def shutdown_cache() -> None:
    """Shutdown global cache instance (for testing/cleanup)."""
    global _cache_instance

    if _cache_instance is not None:
        _cache_instance.shutdown()
        _cache_instance = None
