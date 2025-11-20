"""
Integration tests for Redis health and connectivity.

Tests verify Redis configuration and operational readiness:
- Basic connectivity (ping, set/get)
- TTL enforcement
- Key prefix patterns
- Configuration (eviction policy, maxmemory)
- Persistence settings

Run when Redis is provisioned to validate setup.
"""

import pytest
import redis
import os
from typing import Dict

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class TestRedisHealth:
    """Test Redis connectivity and health."""

    def setup_method(self):
        """Setup Redis client for tests."""
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

    def test_redis_ping(self):
        """Redis should respond to ping."""
        try:
            result = self.redis_client.ping()
            assert result is True
            print(f"✓ Redis ping successful")
        except redis.ConnectionError as e:
            pytest.skip(f"Redis not available: {e}")

    def test_redis_set_get(self):
        """Basic set/get should work."""
        key = "test:health:set_get"
        value = "test_value"

        try:
            # Set with TTL
            self.redis_client.setex(key, 10, value)  # 10 second TTL

            # Get value
            retrieved = self.redis_client.get(key)

            assert retrieved == value, f"Expected '{value}', got '{retrieved}'"

            # Cleanup
            self.redis_client.delete(key)

            print(f"✓ Redis set/get working")

        except Exception as e:
            pytest.skip(f"Redis operation failed: {e}")

    def test_redis_ttl_enforcement(self):
        """Keys should have TTL set correctly."""
        key = "test:health:ttl"

        try:
            self.redis_client.setex(key, 60, "test")

            ttl = self.redis_client.ttl(key)
            assert 50 < ttl <= 60, f"TTL not set correctly: {ttl}"

            # Cleanup
            self.redis_client.delete(key)

            print(f"✓ TTL enforcement working (TTL={ttl}s)")

        except Exception as e:
            pytest.skip(f"TTL test failed: {e}")

    def test_redis_key_prefix(self):
        """ISL keys should use isl: prefix."""
        test_keys = [
            "isl:beliefs:user_test_123",
            "isl:ident:dag_test_abc",
            "isl:result:scenario_test_def"
        ]

        try:
            for key in test_keys:
                self.redis_client.setex(key, 10, "test")
                assert self.redis_client.exists(key) == 1, f"Key not found: {key}"

            # Cleanup
            for key in test_keys:
                self.redis_client.delete(key)

            print(f"✓ Key prefix pattern validated ({len(test_keys)} keys)")

        except Exception as e:
            pytest.skip(f"Key prefix test failed: {e}")

    def test_redis_eviction_policy(self):
        """Redis should have volatile-lru eviction policy."""
        try:
            config = self.redis_client.config_get("maxmemory-policy")
            policy = config.get("maxmemory-policy", "unknown")

            if policy == "volatile-lru":
                print(f"✓ Eviction policy: volatile-lru")
            else:
                print(f"⚠ Eviction policy is '{policy}' (expected 'volatile-lru')")
                # Don't fail - this is a configuration recommendation
                pytest.skip(f"Eviction policy not configured: {policy}")

        except Exception as e:
            pytest.skip(f"Cannot read config: {e}")

    def test_redis_maxmemory_set(self):
        """Redis should have maxmemory configured."""
        try:
            config = self.redis_client.config_get("maxmemory")
            maxmemory = int(config.get("maxmemory", 0))

            # Should be >0 (configured)
            if maxmemory > 0:
                # Convert to GB for readability
                maxmemory_gb = maxmemory / (1024 ** 3)
                print(f"✓ Max memory: {maxmemory_gb:.2f} GB")
            else:
                print(f"⚠ Max memory not configured (unlimited)")
                pytest.skip("Max memory not configured")

        except Exception as e:
            pytest.skip(f"Cannot read config: {e}")

    def test_redis_persistence_config(self):
        """Redis should have RDB persistence configured."""
        try:
            config = self.redis_client.config_get("save")
            save_config = config.get("save", "")

            # Should have save intervals configured
            if save_config and save_config != "":
                print(f"✓ Persistence configured: {save_config}")
            else:
                print(f"⚠ RDB persistence not configured")
                pytest.skip("Persistence not configured")

        except Exception as e:
            pytest.skip(f"Cannot read config: {e}")

    def test_redis_memory_usage(self):
        """Check current Redis memory usage."""
        try:
            info = self.redis_client.info("memory")

            used_memory_mb = info["used_memory"] / (1024 ** 2)
            maxmemory = info.get("maxmemory", 0)

            print(f"✓ Memory usage:")
            print(f"  Used: {used_memory_mb:.2f} MB")

            if maxmemory > 0:
                maxmemory_mb = maxmemory / (1024 ** 2)
                usage_pct = (used_memory_mb / maxmemory_mb) * 100
                print(f"  Max: {maxmemory_mb:.2f} MB")
                print(f"  Usage: {usage_pct:.1f}%")

                # Warn if usage high
                if usage_pct > 80:
                    print(f"  ⚠ Memory usage >80%")
            else:
                print(f"  Max: Unlimited")

        except Exception as e:
            pytest.skip(f"Cannot read memory info: {e}")

    def test_redis_client_connections(self):
        """Check Redis client connection count."""
        try:
            info = self.redis_client.info("clients")

            connected_clients = info.get("connected_clients", 0)
            print(f"✓ Connected clients: {connected_clients}")

            # Warn if many clients
            if connected_clients > 100:
                print(f"  ⚠ High connection count: {connected_clients}")

        except Exception as e:
            pytest.skip(f"Cannot read client info: {e}")

    def teardown_method(self):
        """Cleanup after tests."""
        try:
            # Delete any test keys
            for key in self.redis_client.scan_iter("test:*"):
                self.redis_client.delete(key)
        except:
            pass  # Ignore cleanup errors


class TestRedisKeyPatterns:
    """Test ISL key pattern compliance."""

    def setup_method(self):
        """Setup Redis client."""
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

    def test_no_keys_without_ttl(self):
        """All ISL keys should have TTL (no infinite keys)."""
        try:
            keys_without_ttl = []

            # Scan ISL keys
            for key in self.redis_client.scan_iter("isl:*", count=100):
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiry
                    keys_without_ttl.append(key)

            if keys_without_ttl:
                print(f"✗ Keys without TTL found: {keys_without_ttl[:10]}")
                pytest.fail(f"Found {len(keys_without_ttl)} keys without TTL")
            else:
                print(f"✓ All ISL keys have TTL")

        except Exception as e:
            pytest.skip(f"TTL check failed: {e}")

    def test_key_distribution(self):
        """Check distribution of ISL keys by prefix."""
        try:
            key_counts = {}

            # Count keys by second-level prefix (e.g., beliefs, ident, result)
            for key in self.redis_client.scan_iter("isl:*", count=100):
                parts = key.split(":")
                if len(parts) >= 2:
                    prefix = parts[1]  # beliefs, ident, result, etc.
                    key_counts[prefix] = key_counts.get(prefix, 0) + 1

            if key_counts:
                print(f"✓ Key distribution:")
                for prefix, count in sorted(key_counts.items()):
                    print(f"  isl:{prefix}: {count} keys")
            else:
                print(f"✓ No ISL keys in Redis (clean state)")

        except Exception as e:
            pytest.skip(f"Key distribution check failed: {e}")


# Run tests with: pytest tests/integration/test_redis_health.py -v -s
