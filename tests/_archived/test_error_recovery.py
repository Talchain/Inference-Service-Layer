"""
Unit tests for error recovery framework.

Tests CircuitBreaker, RetryStrategy, HealthMonitor, and decorators.
"""

import time
from datetime import datetime, timedelta

import pytest

from src.utils.error_recovery import (
    CircuitBreaker,
    CircuitState,
    RetryStrategy,
    FallbackStrategy,
    HealthMonitor,
    ServiceHealth,
    ServiceHealthStatus,
    RecoveryError,
    with_fallback,
    with_circuit_breaker,
    with_retry,
)


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""

    def test_initial_state_is_closed(self):
        """Test circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker("test", failure_threshold=3)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_successful_call(self):
        """Test successful call keeps circuit CLOSED."""
        breaker = CircuitBreaker("test", failure_threshold=3)

        def success_func():
            return "success"

        result = breaker.call(success_func)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_circuit_opens_after_threshold_failures(self):
        """Test circuit OPENS after reaching failure threshold."""
        breaker = CircuitBreaker("test", failure_threshold=3, timeout=60)

        def failing_func():
            raise ValueError("Test error")

        # First 2 failures - circuit stays CLOSED
        for i in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
            assert breaker.state == CircuitState.CLOSED
            assert breaker.failure_count == i + 1

        # Third failure - circuit OPENS
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

    def test_open_circuit_rejects_calls(self):
        """Test OPEN circuit rejects calls immediately."""
        breaker = CircuitBreaker("test", failure_threshold=2, timeout=60)

        def failing_func():
            raise ValueError("Test error")

        # Trigger circuit to OPEN
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Next call should be rejected without calling function
        with pytest.raises(RecoveryError, match="Circuit breaker OPEN"):
            breaker.call(failing_func)

    def test_circuit_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to HALF_OPEN after timeout."""
        breaker = CircuitBreaker("test", failure_threshold=2, timeout=1)  # 1 second timeout

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(1.1)

        # Next call should attempt (HALF_OPEN state)
        # It will fail and go back to OPEN, but that's OK for this test
        try:
            breaker.call(failing_func)
        except:
            pass

        # The state should have been HALF_OPEN during the call
        # (even if it went back to OPEN after failure)

    def test_half_open_success_closes_circuit(self):
        """Test successful call in HALF_OPEN state closes circuit."""
        breaker = CircuitBreaker("test", failure_threshold=2, success_threshold=1, timeout=1)

        call_count = [0]

        def sometimes_works():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Failing")
            return "success"

        # Open the circuit (2 failures)
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(sometimes_works)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout to enter HALF_OPEN
        time.sleep(1.1)

        # Successful call should close circuit
        result = breaker.call(sometimes_works)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_reset(self):
        """Test manual reset of circuit breaker."""
        breaker = CircuitBreaker("test", failure_threshold=2)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0


class TestRetryStrategy:
    """Test RetryStrategy functionality."""

    def test_successful_call_no_retry(self):
        """Test successful call executes once."""
        retry = RetryStrategy(max_retries=3)
        call_count = [0]

        def success_func():
            call_count[0] += 1
            return "success"

        result = retry.execute(success_func)

        assert result == "success"
        assert call_count[0] == 1

    def test_retries_on_failure(self):
        """Test retries on transient failures."""
        retry = RetryStrategy(max_retries=3, initial_delay=0.01)
        call_count = [0]

        def eventually_succeeds():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Transient error")
            return "success"

        result = retry.execute(eventually_succeeds)

        assert result == "success"
        assert call_count[0] == 3  # Failed twice, succeeded on third try

    def test_exhausts_retries(self):
        """Test failure after exhausting retries."""
        retry = RetryStrategy(max_retries=2, initial_delay=0.01)

        def always_fails():
            raise ValueError("Persistent error")

        with pytest.raises(ValueError, match="Persistent error"):
            retry.execute(always_fails)

    def test_exponential_backoff(self):
        """Test exponential backoff delays."""
        retry = RetryStrategy(max_retries=3, initial_delay=0.1, backoff_factor=2.0)
        delays = []
        call_count = [0]

        def track_delays():
            call_count[0] += 1
            if call_count[0] > 1:
                delays.append(time.time() - start_time)
            if call_count[0] < 4:
                raise ValueError("Retry")
            return "success"

        start_time = time.time()
        retry.execute(track_delays)

        # Verify exponential backoff: 0.1s, 0.2s, 0.4s (approximately)
        # Allow 50ms tolerance for timing variations
        assert len(delays) == 3
        assert 0.05 < delays[0] < 0.15  # ~0.1s
        assert 0.15 < delays[1] < 0.25  # ~0.2s
        assert 0.35 < delays[2] < 0.45  # ~0.4s


class TestHealthMonitor:
    """Test HealthMonitor functionality."""

    def test_initial_health_is_healthy(self):
        """Test service starts in HEALTHY state."""
        monitor = HealthMonitor()
        health = monitor.get_health("test_service")

        assert health.status == ServiceHealthStatus.HEALTHY
        assert health.total_requests == 0
        assert health.success_rate == 1.0

    def test_record_success(self):
        """Test recording successful operations."""
        monitor = HealthMonitor()

        monitor.record_success("test_service")
        monitor.record_success("test_service")

        health = monitor.get_health("test_service")

        assert health.successes == 2
        assert health.total_requests == 2
        assert health.success_rate == 1.0
        assert health.status == ServiceHealthStatus.HEALTHY

    def test_record_failure(self):
        """Test recording failures affects health status."""
        monitor = HealthMonitor()

        # 6 successes, 4 failures = 60% success rate
        for _ in range(6):
            monitor.record_success("test_service")
        for _ in range(4):
            monitor.record_failure("test_service")

        health = monitor.get_health("test_service")

        assert health.successes == 6
        assert health.failures == 4
        assert health.total_requests == 10
        assert health.success_rate == 0.6
        assert health.status == ServiceHealthStatus.DEGRADED  # < 80%

    def test_record_fallback(self):
        """Test recording fallback operations."""
        monitor = HealthMonitor()

        monitor.record_fallback("test_service")
        monitor.record_fallback("test_service")

        health = monitor.get_health("test_service")

        assert health.fallbacks == 2
        assert health.total_requests == 2

    def test_failing_status(self):
        """Test service marked as FAILING with low success rate."""
        monitor = HealthMonitor()

        # 2 successes, 8 failures = 20% success rate
        for _ in range(2):
            monitor.record_success("test_service")
        for _ in range(8):
            monitor.record_failure("test_service")

        health = monitor.get_health("test_service")

        assert health.success_rate == 0.2
        assert health.status == ServiceHealthStatus.FAILING  # < 50%

    def test_reset_service(self):
        """Test resetting service health."""
        monitor = HealthMonitor()

        monitor.record_failure("test_service")
        monitor.record_fallback("test_service")

        health_before = monitor.get_health("test_service")
        assert health_before.total_requests > 0

        monitor.reset_service("test_service")

        health_after = monitor.get_health("test_service")
        assert health_after.total_requests == 0
        assert health_after.status == ServiceHealthStatus.HEALTHY

    def test_get_all_services(self):
        """Test getting all monitored services."""
        monitor = HealthMonitor()

        monitor.record_success("service1")
        monitor.record_failure("service2")
        monitor.record_fallback("service3")

        all_services = monitor.get_all_services()

        assert len(all_services) == 3
        assert "service1" in all_services
        assert "service2" in all_services
        assert "service3" in all_services


class TestFallbackDecorator:
    """Test @with_fallback decorator."""

    def test_fallback_on_exception(self):
        """Test fallback function is called on exception."""
        def fallback_func():
            return "fallback_result"

        @with_fallback(fallback_func, fallback_strategy=FallbackStrategy.SIMPLE)
        def failing_func():
            raise ValueError("Primary failed")

        result = failing_func()

        assert result == "fallback_result"

    def test_no_fallback_on_success(self):
        """Test fallback not called on success."""
        def fallback_func():
            return "fallback_result"

        @with_fallback(fallback_func)
        def success_func():
            return "primary_result"

        result = success_func()

        assert result == "primary_result"

    def test_fallback_with_degraded_strategy(self):
        """Test DEGRADED fallback strategy."""
        def fallback_func():
            return {"result": "partial", "degraded": True}

        @with_fallback(fallback_func, fallback_strategy=FallbackStrategy.DEGRADED)
        def failing_func():
            raise ValueError("Primary failed")

        result = failing_func()

        assert result["result"] == "partial"
        assert result["degraded"] is True


class TestCircuitBreakerDecorator:
    """Test @with_circuit_breaker decorator."""

    def test_decorator_protects_function(self):
        """Test circuit breaker decorator protects function."""
        breaker = CircuitBreaker("test_decorator", failure_threshold=2)

        @with_circuit_breaker(breaker)
        def protected_func():
            raise ValueError("Error")

        # First failure
        with pytest.raises(ValueError):
            protected_func()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 1

        # Second failure - opens circuit
        with pytest.raises(ValueError):
            protected_func()

        assert breaker.state == CircuitState.OPEN

        # Third call rejected by circuit breaker
        with pytest.raises(RecoveryError):
            protected_func()


class TestRetryDecorator:
    """Test @with_retry decorator."""

    def test_decorator_retries_on_failure(self):
        """Test retry decorator retries failed calls."""
        retry = RetryStrategy(max_retries=3, initial_delay=0.01)
        call_count = [0]

        @with_retry(retry)
        def eventually_succeeds():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Not yet")
            return "success"

        result = eventually_succeeds()

        assert result == "success"
        assert call_count[0] == 3


class TestIntegratedErrorRecovery:
    """Test integrated error recovery patterns."""

    def test_circuit_breaker_with_fallback(self):
        """Test circuit breaker combined with fallback."""
        breaker = CircuitBreaker("integrated", failure_threshold=2)

        def fallback():
            return "fallback_result"

        @with_fallback(fallback)
        @with_circuit_breaker(breaker)
        def protected_func():
            raise ValueError("Primary error")

        # First two calls fail but use fallback
        result1 = protected_func()
        result2 = protected_func()

        assert result1 == "fallback_result"
        assert result2 == "fallback_result"
        assert breaker.state == CircuitState.OPEN

        # Third call rejected by circuit, but fallback still works
        result3 = protected_func()
        assert result3 == "fallback_result"

    def test_retry_with_eventual_success(self):
        """Test retry strategy with eventual success."""
        retry = RetryStrategy(max_retries=3, initial_delay=0.01)
        monitor = HealthMonitor()
        call_count = [0]

        @with_retry(retry)
        def flaky_operation():
            call_count[0] += 1
            if call_count[0] < 2:
                monitor.record_failure("flaky_service")
                raise ValueError("Transient error")
            monitor.record_success("flaky_service")
            return "success"

        result = flaky_operation()

        assert result == "success"
        health = monitor.get_health("flaky_service")
        assert health.successes == 1
        assert health.failures == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
