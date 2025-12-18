"""
Caching utilities for causal inference operations.

Provides thread-safe, TTL-based caching for expensive operations
like DAG path analysis and strategy generation.

NOTE: For new code, prefer using `get_named_cache` from
`src/infrastructure/memory_cache` which provides the same functionality
with the production-grade MemoryCache implementation:

    from src.infrastructure.memory_cache import get_named_cache
    cache = get_named_cache("my_domain", max_size=500, ttl=3600)

This module's `get_cache` function remains available for backward compatibility.
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Cache configuration constants
DEFAULT_TTL = 3600  # 1 hour in seconds
DEFAULT_MAX_SIZE = 1000  # Maximum cache entries
CACHE_STATS_LOG_INTERVAL = 100  # Log stats every N operations


class CacheStats:
    """Track cache performance statistics."""

    def __init__(self):
        """Initialize cache statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_requests = 0
        self.lock = Lock()

    def record_hit(self):
        """Record a cache hit."""
        with self.lock:
            self.hits += 1
            self.total_requests += 1

    def record_miss(self):
        """Record a cache miss."""
        with self.lock:
            self.misses += 1
            self.total_requests += 1

    def record_eviction(self):
        """Record a cache eviction."""
        with self.lock:
            self.evictions += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self.lock:
            hit_rate = (
                self.hits / self.total_requests if self.total_requests > 0 else 0.0
            )
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "total_requests": self.total_requests,
                "hit_rate": hit_rate,
            }

    def reset(self):
        """Reset statistics."""
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.total_requests = 0


class TTLCache:
    """
    Thread-safe TTL (Time-To-Live) cache with LRU eviction.

    Features:
    - TTL-based expiration
    - LRU eviction when max size reached
    - Thread-safe operations
    - Cache statistics tracking
    - Automatic hash generation for complex keys
    """

    def __init__(
        self, max_size: int = DEFAULT_MAX_SIZE, ttl: int = DEFAULT_TTL, name: str = "cache"
    ):
        """
        Initialize TTL cache.

        Args:
            max_size: Maximum number of entries
            ttl: Time-to-live in seconds
            name: Cache name for logging
        """
        self.max_size = max_size
        self.ttl = ttl
        self.name = name
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = Lock()
        self.stats = CacheStats()
        logger.info(
            f"cache_initialized",
            extra={
                "cache_name": name,
                "max_size": max_size,
                "ttl": ttl,
            }
        )

    def _compute_key_hash(self, key: Any) -> str:
        """
        Compute stable hash for cache key.

        Args:
            key: Cache key (can be dict, tuple, str, etc.)

        Returns:
            Hash string
        """
        if isinstance(key, str):
            return key

        # For complex objects, use JSON serialization for stable hashing
        try:
            # Sort dict keys for consistent hashing
            if isinstance(key, dict):
                key_str = json.dumps(key, sort_keys=True)
            elif isinstance(key, (list, tuple)):
                key_str = json.dumps(key)
            else:
                key_str = str(key)

            return hashlib.sha256(key_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(
                f"Failed to hash key, using str(): {e}",
                extra={"cache_name": self.name}
            )
            return hashlib.sha256(str(key).encode()).hexdigest()

    def _is_expired(self, key_hash: str) -> bool:
        """
        Check if entry is expired.

        Args:
            key_hash: Cache key hash

        Returns:
            True if expired
        """
        if key_hash not in self._timestamps:
            return True

        age = time.time() - self._timestamps[key_hash]
        return age > self.ttl

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return

        # OrderedDict maintains insertion order
        # popitem(last=False) removes oldest item
        key_hash, _ = self._cache.popitem(last=False)
        if key_hash in self._timestamps:
            del self._timestamps[key_hash]
        self.stats.record_eviction()

        logger.debug(
            f"cache_eviction",
            extra={
                "cache_name": self.name,
                "evicted_key": key_hash[:16],
            }
        )

    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        key_hash = self._compute_key_hash(key)

        with self._lock:
            # Check if key exists and is not expired
            if key_hash in self._cache and not self._is_expired(key_hash):
                # Move to end (most recently used)
                self._cache.move_to_end(key_hash)
                self.stats.record_hit()

                # Log stats periodically
                if self.stats.total_requests % CACHE_STATS_LOG_INTERVAL == 0:
                    logger.info(
                        f"cache_stats",
                        extra={
                            "cache_name": self.name,
                            **self.stats.get_stats(),
                        }
                    )

                return self._cache[key_hash]
            else:
                # Remove expired entry
                if key_hash in self._cache:
                    del self._cache[key_hash]
                    if key_hash in self._timestamps:
                        del self._timestamps[key_hash]

                self.stats.record_miss()
                return None

    def put(self, key: Any, value: Any):
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        key_hash = self._compute_key_hash(key)

        with self._lock:
            # Remove if already exists (to update timestamp)
            if key_hash in self._cache:
                del self._cache[key_hash]

            # Evict if at max size
            if len(self._cache) >= self.max_size:
                self._evict_lru()

            # Add new entry
            self._cache[key_hash] = value
            self._timestamps[key_hash] = time.time()

            logger.debug(
                f"cache_put",
                extra={
                    "cache_name": self.name,
                    "key": key_hash[:16],
                    "cache_size": len(self._cache),
                }
            )

    def invalidate(self, key: Any):
        """
        Invalidate cache entry.

        Args:
            key: Cache key
        """
        key_hash = self._compute_key_hash(key)

        with self._lock:
            if key_hash in self._cache:
                del self._cache[key_hash]
            if key_hash in self._timestamps:
                del self._timestamps[key_hash]

            logger.debug(
                f"cache_invalidate",
                extra={
                    "cache_name": self.name,
                    "key": key_hash[:16],
                }
            )

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            logger.info(
                f"cache_cleared",
                extra={
                    "cache_name": self.name,
                }
            )

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats.get_stats(),
            "cache_size": self.size(),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }


def cached(cache: TTLCache, key_fn: Optional[Callable] = None):
    """
    Decorator for caching function results.

    Args:
        cache: TTLCache instance
        key_fn: Optional function to compute cache key from args/kwargs
                If None, uses all args/kwargs

    Example:
        >>> cache = TTLCache(max_size=100, ttl=300)
        >>> @cached(cache)
        ... def expensive_function(x, y):
        ...     return x + y
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Compute cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Use function name + args + kwargs as key
                cache_key = {
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs,
                }

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(
                    f"cache_hit",
                    extra={
                        "function": func.__name__,
                        "cache_name": cache.name,
                    }
                )
                return result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache.put(cache_key, result)

            return result

        return wrapper

    return decorator


# Global caches for different operations
_global_caches: Dict[str, TTLCache] = {}
_global_cache_lock = Lock()


def get_cache(
    name: str, max_size: int = DEFAULT_MAX_SIZE, ttl: int = DEFAULT_TTL
) -> TTLCache:
    """
    Get or create a named global cache.

    Args:
        name: Cache name
        max_size: Maximum cache size
        ttl: Time-to-live in seconds

    Returns:
        TTLCache instance
    """
    with _global_cache_lock:
        if name not in _global_caches:
            _global_caches[name] = TTLCache(max_size=max_size, ttl=ttl, name=name)
        return _global_caches[name]


def clear_all_caches():
    """Clear all global caches."""
    with _global_cache_lock:
        for cache in _global_caches.values():
            cache.clear()
        logger.info("all_caches_cleared")


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all global caches."""
    with _global_cache_lock:
        return {name: cache.get_stats() for name, cache in _global_caches.items()}
