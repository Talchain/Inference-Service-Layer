"""
Enterprise-grade error recovery patterns for production resilience.

Implements:
- Graceful degradation with fallback strategies
- Circuit breaker pattern for fault isolation
- Retry logic with exponential backoff
- Partial result returns (never return 500 errors)
- Comprehensive error logging and monitoring

Design principles:
- Always return a result, even if degraded
- Log all fallbacks for monitoring
- Maintain service availability over feature completeness
- Provide clear user feedback about degraded functionality
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Circuit breaker configuration
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Failures before opening circuit
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2  # Successes before closing circuit
CIRCUIT_BREAKER_TIMEOUT = 60  # Seconds before trying half-open
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = 3  # Max calls in half-open state

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 0.1  # 100ms
DEFAULT_MAX_DELAY = 5.0  # 5 seconds
DEFAULT_BACKOFF_MULTIPLIER = 2.0


class ServiceHealth(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class FallbackStrategy(Enum):
    """Fallback strategy types."""
    SIMPLE = "simple"  # Return simplified result
    CACHED = "cached"  # Return cached result
    DEFAULT = "default"  # Return safe default
    PARTIAL = "partial"  # Return partial result with warnings


class RecoveryError(Exception):
    """Base exception for error recovery."""
    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        fallback_used: Optional[FallbackStrategy] = None,
        degraded_result: Optional[Any] = None
    ):
        super().__init__(message)
        self.original_error = original_error
        self.fallback_used = fallback_used
        self.degraded_result = degraded_result


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by:
    - Opening circuit after threshold failures
    - Rejecting requests while open (fail fast)
    - Periodically testing if service recovered (half-open)
    - Closing circuit after successful recoveryPrevents repeated calls to failing services.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        success_threshold: int = CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
        timeout: int = CIRCUIT_BREAKER_TIMEOUT
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name for logging
            failure_threshold: Failures before opening circuit
            success_threshold: Successes before closing circuit
            timeout: Seconds before attempting half-open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0

        logger.info(
            f"circuit_breaker_initialized",
            extra={
                "name": name,
                "failure_threshold": failure_threshold,
                "timeout": timeout
            }
        )

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            RecoveryError: If circuit is open
        """
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                logger.warning(
                    f"circuit_breaker_open",
                    extra={"name": self.name, "state": self.state.value}
                )
                raise RecoveryError(
                    f"Circuit breaker '{self.name}' is OPEN - service unavailable",
                    fallback_used=FallbackStrategy.DEFAULT
                )

        # Check half-open limit
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS:
                logger.warning(
                    f"circuit_breaker_half_open_limit",
                    extra={"name": self.name}
                )
                raise RecoveryError(
                    f"Circuit breaker '{self.name}' half-open limit reached",
                    fallback_used=FallbackStrategy.DEFAULT
                )
            self.half_open_calls += 1

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self.last_failure_time is None:
            return False

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        logger.info(
            f"circuit_breaker_half_open",
            extra={"name": self.name}
        )

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.success_threshold:
                self._close_circuit()

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def _on_failure(self, error: Exception):
        """Handle failed call."""
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during testing - go back to open
            self._open_circuit()

        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1

            if self.failure_count >= self.failure_threshold:
                self._open_circuit()

        logger.warning(
            f"circuit_breaker_failure",
            extra={
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "error": str(error)
            }
        )

    def _open_circuit(self):
        """Open the circuit."""
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.error(
            f"circuit_breaker_opened",
            extra={
                "name": self.name,
                "failure_count": self.failure_count
            }
        )

    def _close_circuit(self):
        """Close the circuit (recovered)."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        logger.info(
            f"circuit_breaker_closed",
            extra={"name": self.name}
        )


class RetryStrategy:
    """
    Retry logic with exponential backoff.

    For transient failures that may succeed on retry.
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
        retryable_exceptions: Optional[List[type]] = None
    ):
        """
        Initialize retry strategy.

        Args:
            max_retries: Maximum retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            backoff_multiplier: Delay multiplier for each retry
            retryable_exceptions: Exceptions that should trigger retry
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retryable_exceptions = retryable_exceptions or [Exception]

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Function to call
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None
        delay = self.initial_delay

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)

                if attempt > 0:
                    logger.info(
                        f"retry_succeeded",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "total_attempts": self.max_retries + 1
                        }
                    )

                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not any(isinstance(e, exc_type) for exc_type in self.retryable_exceptions):
                    logger.debug(f"Non-retryable exception: {type(e).__name__}")
                    raise

                # Check if we have retries left
                if attempt >= self.max_retries:
                    logger.error(
                        f"retry_exhausted",
                        extra={
                            "function": func.__name__,
                            "attempts": attempt + 1,
                            "error": str(e)
                        }
                    )
                    raise

                # Log and wait before retry
                logger.warning(
                    f"retry_attempt",
                    extra={
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "delay": delay,
                        "error": str(e)
                    }
                )

                time.sleep(delay)
                delay = min(delay * self.backoff_multiplier, self.max_delay)

        # Should never reach here, but just in case
        raise last_exception or Exception("Retry failed with unknown error")


def with_fallback(
    fallback_func: Callable[..., T],
    fallback_strategy: FallbackStrategy = FallbackStrategy.SIMPLE,
    log_fallback: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for graceful degradation with fallback.

    Usage:
        @with_fallback(lambda *args, **kwargs: default_value)
        def risky_function(x):
            # might fail
            return complex_computation(x)

    Args:
        fallback_func: Function to call if primary fails
        fallback_strategy: Type of fallback for logging
        log_fallback: Whether to log fallback usage

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)

            except Exception as e:
                if log_fallback:
                    logger.warning(
                        f"fallback_triggered",
                        extra={
                            "function": func.__name__,
                            "fallback_strategy": fallback_strategy.value,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )

                try:
                    result = fallback_func(*args, **kwargs)

                    if log_fallback:
                        logger.info(
                            f"fallback_succeeded",
                            extra={
                                "function": func.__name__,
                                "fallback_strategy": fallback_strategy.value
                            }
                        )

                    return result

                except Exception as fallback_error:
                    logger.error(
                        f"fallback_failed",
                        extra={
                            "function": func.__name__,
                            "primary_error": str(e),
                            "fallback_error": str(fallback_error)
                        }
                    )
                    raise RecoveryError(
                        f"Both primary and fallback failed for {func.__name__}",
                        original_error=e,
                        fallback_used=fallback_strategy
                    )

        return wrapper
    return decorator


def with_circuit_breaker(
    breaker: CircuitBreaker
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for circuit breaker pattern.

    Usage:
        breaker = CircuitBreaker("external_service")

        @with_circuit_breaker(breaker)
        def call_external_service():
            # might fail
            return requests.get(...)

    Args:
        breaker: CircuitBreaker instance

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    retryable_exceptions: Optional[List[type]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retry logic.

    Usage:
        @with_retry(max_retries=3, initial_delay=0.1)
        def flaky_function():
            # might fail transiently
            return result

    Args:
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
        retryable_exceptions: Exceptions that should trigger retry

    Returns:
        Decorated function
    """
    strategy = RetryStrategy(
        max_retries=max_retries,
        initial_delay=initial_delay,
        retryable_exceptions=retryable_exceptions
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return strategy.execute(func, *args, **kwargs)
        return wrapper
    return decorator


class HealthMonitor:
    """
    Monitor service health and degradation.

    Tracks:
    - Success rates
    - Error rates
    - Fallback usage
    - Circuit breaker states
    """

    def __init__(self):
        """Initialize health monitor."""
        self.metrics: Dict[str, Dict[str, int]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def record_success(self, service: str):
        """Record successful operation."""
        if service not in self.metrics:
            self.metrics[service] = {"success": 0, "failure": 0, "fallback": 0}
        self.metrics[service]["success"] += 1

    def record_failure(self, service: str):
        """Record failed operation."""
        if service not in self.metrics:
            self.metrics[service] = {"success": 0, "failure": 0, "fallback": 0}
        self.metrics[service]["failure"] += 1

    def record_fallback(self, service: str):
        """Record fallback usage."""
        if service not in self.metrics:
            self.metrics[service] = {"success": 0, "failure": 0, "fallback": 0}
        self.metrics[service]["fallback"] += 1

    def get_health(self, service: str) -> ServiceHealth:
        """
        Get service health status.

        Args:
            service: Service name

        Returns:
            ServiceHealth status
        """
        if service not in self.metrics:
            return ServiceHealth.HEALTHY

        metrics = self.metrics[service]
        total = sum(metrics.values())

        if total == 0:
            return ServiceHealth.HEALTHY

        success_rate = metrics["success"] / total
        fallback_rate = metrics["fallback"] / total

        if success_rate >= 0.95 and fallback_rate < 0.05:
            return ServiceHealth.HEALTHY
        elif success_rate >= 0.80 or fallback_rate < 0.20:
            return ServiceHealth.DEGRADED
        else:
            return ServiceHealth.FAILING

    def get_metrics(self, service: Optional[str] = None) -> Dict:
        """
        Get metrics for service or all services.

        Args:
            service: Optional service name

        Returns:
            Metrics dictionary
        """
        if service:
            return self.metrics.get(service, {})
        return self.metrics

    def register_circuit_breaker(self, name: str, breaker: CircuitBreaker):
        """Register circuit breaker for monitoring."""
        self.circuit_breakers[name] = breaker

    def get_circuit_states(self) -> Dict[str, str]:
        """Get all circuit breaker states."""
        return {
            name: breaker.state.value
            for name, breaker in self.circuit_breakers.items()
        }


# Global health monitor instance
health_monitor = HealthMonitor()
