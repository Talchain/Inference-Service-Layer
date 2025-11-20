# PLoT Error Handling Guide

**Robust Error Management for ISL Integration**

**Version**: 1.0.0
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Error Response Format](#error-response-format)
2. [HTTP Status Codes](#http-status-codes)
3. [Error Code Reference](#error-code-reference)
4. [Retry Strategies](#retry-strategies)
5. [Circuit Breaker Pattern](#circuit-breaker-pattern)
6. [Timeout Handling](#timeout-handling)
7. [Error Logging](#error-logging)
8. [Common Error Scenarios](#common-error-scenarios)
9. [Production-Ready Error Handler](#production-ready-error-handler)

---

## Error Response Format

All ISL error responses follow this structure:

```json
{
  "detail": "Human-readable error message",
  "error_code": "ERROR_CODE_CONSTANT",
  "request_id": "req_abc123def456",
  "timestamp": "2025-11-20T16:30:00Z",
  "path": "/api/v1/causal/validate",
  "validation_errors": [  // Only for 422 errors
    {
      "field": "dag.nodes",
      "message": "Field required",
      "type": "missing"
    }
  ]
}
```

---

## HTTP Status Codes

### 2xx - Success

| Code | Meaning | Action |
|------|---------|--------|
| 200 OK | Request successful | Process response normally |

### 4xx - Client Errors

| Code | Meaning | Retry? | Action |
|------|---------|--------|--------|
| 400 Bad Request | Invalid input | ❌ No | Fix request payload |
| 422 Unprocessable Entity | Validation error | ❌ No | Fix validation errors |
| 429 Too Many Requests | Rate limit exceeded | ✅ Yes | Backoff and retry |

### 5xx - Server Errors

| Code | Meaning | Retry? | Action |
|------|---------|--------|--------|
| 500 Internal Server Error | Server-side error | ✅ Yes | Retry with backoff |
| 502 Bad Gateway | Upstream service failed | ✅ Yes | Retry with backoff |
| 503 Service Unavailable | Service overloaded | ✅ Yes | Retry after longer delay |
| 504 Gateway Timeout | Request timeout | ✅ Yes | Increase timeout, retry |

---

## Error Code Reference

### Causal Inference Errors

| Error Code | HTTP Status | Description | Solution |
|------------|-------------|-------------|----------|
| `INVALID_DAG` | 400 | DAG is malformed or invalid | Verify nodes/edges format |
| `CYCLIC_DAG` | 400 | DAG contains cycles | Remove cycles from graph |
| `NODE_NOT_FOUND` | 400 | Treatment/outcome not in DAG | Check variable names |
| `EMPTY_DAG` | 400 | No nodes or edges provided | Provide valid DAG |
| `DISCONNECTED_GRAPH` | 400 | Treatment and outcome not connected | Add connecting edges |
| `UNIDENTIFIABLE` | 200 | Effect cannot be identified | Add confounders or change model |
| `STRUCTURAL_MODEL_INVALID` | 400 | Equations are malformed | Fix equation syntax |
| `DISTRIBUTION_INVALID` | 400 | Invalid distribution parameters | Check distribution format |
| `TOPOLOGICAL_SORT_FAILED` | 500 | Circular dependencies in equations | Remove circular references |

### Preference Learning Errors

| Error Code | HTTP Status | Description | Solution |
|------------|-------------|-------------|----------|
| `USER_NOT_FOUND` | 400 | User ID not in session | Initialize with `/elicit` first |
| `QUERY_NOT_FOUND` | 400 | Query ID invalid | Use query ID from `/elicit` |
| `INVALID_CONTEXT` | 400 | Context missing required fields | Provide domain and variables |
| `BELIEF_INITIALIZATION_FAILED` | 500 | Cannot initialize user beliefs | Check context format |
| `INVALID_RESPONSE` | 400 | Query response malformed | Use valid option (A/B) |

### Teaching Errors

| Error Code | HTTP Status | Description | Solution |
|------------|-------------|-------------|----------|
| `CONCEPT_NOT_SUPPORTED` | 400 | Unknown teaching concept | Use valid concept name |
| `BELIEFS_INCOMPLETE` | 400 | Current beliefs missing fields | Provide complete belief structure |
| `EXAMPLE_GENERATION_FAILED` | 500 | Cannot generate examples | Simplify context or reduce max_examples |

### Validation Errors

| Error Code | HTTP Status | Description | Solution |
|------------|-------------|-------------|----------|
| `VALIDATION_LEVEL_INVALID` | 400 | Unknown validation level | Use basic/standard/comprehensive |
| `MODEL_TOO_COMPLEX` | 400 | Model exceeds complexity limits | Simplify DAG (max 50 nodes) |

### System Errors

| Error Code | HTTP Status | Description | Solution |
|------------|-------------|-------------|----------|
| `REDIS_CONNECTION_FAILED` | 503 | Cannot connect to Redis | Check Redis availability |
| `REDIS_TIMEOUT` | 504 | Redis operation timed out | Retry request |
| `INTERNAL_ERROR` | 500 | Unexpected server error | Check logs with request_id |
| `COMPUTATION_TIMEOUT` | 504 | Computation exceeded timeout | Reduce num_samples or complexity |

---

## Retry Strategies

### Exponential Backoff

**Use for**: 5xx errors, 429 rate limiting, network errors

```python
import time
import httpx
from typing import Optional, Callable, Any


def exponential_backoff_retry(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_status_codes: set = {500, 502, 503, 504, 429}
) -> Any:
    """
    Retry function with exponential backoff.

    Args:
        func: Function to retry (must return httpx.Response)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Delay multiplier for each retry
        retryable_status_codes: HTTP codes that should trigger retry

    Returns:
        Function result if successful

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = func()

            # Check if response is retryable
            if hasattr(response, 'status_code'):
                if response.status_code in retryable_status_codes:
                    if attempt < max_retries:
                        delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                        print(f"⚠ HTTP {response.status_code}, retrying in {delay:.1f}s "
                              f"(attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        response.raise_for_status()

            return response

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_exception = e

            if attempt < max_retries:
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                print(f"⚠ Network error: {e}, retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise

        except httpx.HTTPStatusError as e:
            # Don't retry 4xx errors (except 429)
            if e.response.status_code < 500 and e.response.status_code != 429:
                raise
            last_exception = e

            if attempt < max_retries:
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                print(f"⚠ HTTP {e.response.status_code}, retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise

    if last_exception:
        raise last_exception


# Example usage
def make_request_with_retry():
    """Make ISL request with automatic retry."""

    def request_func():
        with httpx.Client(timeout=10.0) as client:
            return client.post(
                f"{ISL_BASE_URL}/api/v1/causal/validate",
                json={
                    "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
                    "treatment": "A",
                    "outcome": "B"
                },
                headers={"X-Request-Id": generate_request_id()}
            )

    response = exponential_backoff_retry(
        request_func,
        max_retries=3,
        base_delay=1.0
    )

    return response.json()
```

### Jittered Exponential Backoff

**Use for**: High-concurrency scenarios to avoid thundering herd

```python
import random


def jittered_exponential_backoff_retry(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Any:
    """Retry with jittered exponential backoff to avoid thundering herd."""

    for attempt in range(max_retries + 1):
        try:
            return func()

        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                # Add jitter: random value between 0 and delay
                delay = min(base_delay * (2 ** attempt), max_delay)
                jittered_delay = delay * (0.5 + random.random() * 0.5)

                print(f"⚠ Retrying in {jittered_delay:.2f}s (attempt {attempt + 1})")
                time.sleep(jittered_delay)
            else:
                raise
```

---

## Circuit Breaker Pattern

**Use for**: Preventing cascading failures when ISL is consistently failing

```python
import time
from enum import Enum
from typing import Callable, Any


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for ISL requests."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = httpx.HTTPError
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                print("🔄 Circuit breaker: Attempting recovery (HALF_OPEN)")
            else:
                raise Exception(
                    f"Circuit breaker OPEN: ISL failing consistently. "
                    f"Retry after {self.recovery_timeout}s"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        return (
            self.last_failure_time is not None
            and time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        """Handle successful request."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            print("✅ Circuit breaker: Service recovered (CLOSED)")

    def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"🔴 Circuit breaker: OPEN ({self.failure_count} failures)")


# Example usage
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0
)


def make_protected_request():
    """Make ISL request protected by circuit breaker."""

    def request_func():
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{ISL_BASE_URL}/api/v1/causal/validate",
                json={...},
                headers={"X-Request-Id": generate_request_id()}
            )
            response.raise_for_status()
            return response.json()

    try:
        return circuit_breaker.call(request_func)
    except Exception as e:
        print(f"✗ Request failed: {e}")
        # Fall back to cached result or default behavior
        return {"status": "error", "cached": True}
```

---

## Timeout Handling

### Request-Level Timeouts

```python
import httpx


# Standard timeout configuration
timeout_config = httpx.Timeout(
    connect=5.0,  # Connection timeout
    read=10.0,    # Read timeout
    write=5.0,    # Write timeout
    pool=5.0      # Pool timeout
)

with httpx.Client(timeout=timeout_config) as client:
    try:
        response = client.post(
            f"{ISL_BASE_URL}/api/v1/causal/counterfactual",
            json={...}
        )
    except httpx.TimeoutException as e:
        print(f"⚠ Request timed out: {e}")
        # Retry with increased timeout or reduce complexity
```

### Endpoint-Specific Timeouts

```python
# Different timeouts for different endpoints
ENDPOINT_TIMEOUTS = {
    "/health": 2.0,
    "/api/v1/causal/validate": 5.0,
    "/api/v1/causal/counterfactual": 15.0,  # Longer for Monte Carlo
    "/api/v1/preferences/elicit": 8.0,
    "/api/v1/teaching/teach": 10.0,
}


def get_timeout_for_endpoint(endpoint: str) -> float:
    """Get recommended timeout for endpoint."""
    return ENDPOINT_TIMEOUTS.get(endpoint, 10.0)  # Default 10s
```

---

## Error Logging

### Structured Error Logging

```python
import logging
import json
from datetime import datetime


def log_isl_error(
    request_id: str,
    endpoint: str,
    error: Exception,
    request_payload: dict,
    response: httpx.Response = None
):
    """
    Log ISL error with structured data.

    Args:
        request_id: Request ID for tracing
        endpoint: ISL endpoint called
        error: Exception raised
        request_payload: Request body sent
        response: HTTP response (if available)
    """
    error_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": request_id,
        "endpoint": endpoint,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "request_payload": request_payload,
    }

    if response is not None:
        error_data["http_status"] = response.status_code
        try:
            error_data["response_body"] = response.json()
        except:
            error_data["response_body"] = response.text[:500]

    if isinstance(error, httpx.HTTPStatusError):
        error_data["http_status"] = error.response.status_code
        error_data["response_body"] = error.response.text[:500]

    logging.error(
        f"ISL request failed: {endpoint}",
        extra=error_data
    )

    # Also log to application monitoring (e.g., Sentry, DataDog)
    # sentry_sdk.capture_exception(error, extra=error_data)


# Example usage
try:
    response = client.post(...)
    response.raise_for_status()
except httpx.HTTPStatusError as e:
    log_isl_error(
        request_id=request_id,
        endpoint="/api/v1/causal/validate",
        error=e,
        request_payload=payload,
        response=e.response
    )
    raise
```

---

## Common Error Scenarios

### Scenario 1: Invalid DAG Structure

**Error**:
```json
{
  "detail": "Node 'InvalidNode' not found in DAG",
  "error_code": "NODE_NOT_FOUND"
}
```

**Solution**:
```python
def validate_dag_locally(dag: dict, treatment: str, outcome: str) -> bool:
    """Validate DAG structure before sending to ISL."""
    nodes = set(dag.get("nodes", []))

    # Check treatment and outcome exist
    if treatment not in nodes:
        raise ValueError(f"Treatment '{treatment}' not in DAG nodes")
    if outcome not in nodes:
        raise ValueError(f"Outcome '{outcome}' not in DAG nodes")

    # Check edges reference valid nodes
    for edge in dag.get("edges", []):
        if len(edge) != 2:
            raise ValueError(f"Invalid edge format: {edge}")
        if edge[0] not in nodes or edge[1] not in nodes:
            raise ValueError(f"Edge {edge} references unknown node")

    return True


# Use before ISL call
try:
    validate_dag_locally(dag, "Price", "Revenue")
except ValueError as e:
    print(f"✗ Local validation failed: {e}")
    return  # Don't call ISL
```

### Scenario 2: User Session Not Found

**Error**:
```json
{
  "detail": "User 'alice' not found in session store",
  "error_code": "USER_NOT_FOUND"
}
```

**Solution**:
```python
def update_beliefs_safe(user_id: str, query_id: str, response: dict):
    """Update beliefs with automatic session initialization."""

    try:
        # Try update
        result = client.post(
            f"{ISL_BASE_URL}/api/v1/preferences/update",
            json={"user_id": user_id, "query_id": query_id, "response": response}
        )
        result.raise_for_status()
        return result.json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            error_data = e.response.json()

            if error_data.get("error_code") == "USER_NOT_FOUND":
                print(f"⚠ User session expired, re-initializing...")

                # Re-initialize session
                init_result = client.post(
                    f"{ISL_BASE_URL}/api/v1/preferences/elicit",
                    json={
                        "user_id": user_id,
                        "context": {...},  # Restore context
                        "num_queries": 1
                    }
                )
                init_result.raise_for_status()

                # Retry update
                retry_result = client.post(
                    f"{ISL_BASE_URL}/api/v1/preferences/update",
                    json={"user_id": user_id, "query_id": query_id, "response": response}
                )
                retry_result.raise_for_status()
                return retry_result.json()

        raise
```

### Scenario 3: Computation Timeout

**Error**:
```json
{
  "detail": "Counterfactual computation exceeded timeout",
  "error_code": "COMPUTATION_TIMEOUT"
}
```

**Solution**:
```python
def run_counterfactual_adaptive(
    model: dict,
    outcome: str,
    intervention: dict,
    initial_samples: int = 1000
):
    """Run counterfactual with adaptive sample reduction on timeout."""

    num_samples = initial_samples

    for attempt in range(3):
        try:
            result = client.post(
                f"{ISL_BASE_URL}/api/v1/causal/counterfactual",
                json={
                    "model": model,
                    "outcome": outcome,
                    "intervention": intervention,
                    "num_samples": num_samples
                },
                timeout=15.0  # Longer timeout
            )
            result.raise_for_status()
            return result.json()

        except httpx.TimeoutException:
            # Reduce samples and retry
            num_samples = num_samples // 2
            print(f"⚠ Timeout, reducing samples to {num_samples} and retrying...")

            if num_samples < 100:
                raise ValueError("Cannot reduce samples further")

    raise Exception("Counterfactual computation failed after 3 attempts")
```

### Scenario 4: Rate Limiting

**Error**:
```json
{
  "detail": "Rate limit exceeded: 100 requests/minute",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 30
}
```

**Solution**:
```python
def handle_rate_limiting(response: httpx.Response):
    """Handle 429 rate limit response."""

    if response.status_code == 429:
        error_data = response.json()
        retry_after = error_data.get("retry_after", 60)

        print(f"⚠ Rate limited, waiting {retry_after}s...")
        time.sleep(retry_after)

        # Retry request
        # ... (use retry logic from earlier)
```

---

## Production-Ready Error Handler

### Complete Error Handling Wrapper

```python
import httpx
import time
import logging
from typing import Any, Dict, Optional, Callable
from enum import Enum


class ISLClient:
    """Production-ready ISL client with comprehensive error handling."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        max_retries: int = 3,
        enable_circuit_breaker: bool = True
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None

        self.logger = logging.getLogger(__name__)

    def request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make ISL request with full error handling.

        Args:
            method: HTTP method (GET, POST)
            endpoint: ISL endpoint path
            payload: Request body (for POST)
            request_id: Optional request ID

        Returns:
            Response JSON

        Raises:
            ISLError: On unrecoverable errors
        """
        request_id = request_id or generate_request_id()
        url = f"{self.base_url}{endpoint}"

        def make_request():
            with httpx.Client(timeout=self.timeout) as client:
                if method == "GET":
                    response = client.get(
                        url,
                        headers={"X-Request-Id": request_id}
                    )
                else:  # POST
                    response = client.post(
                        url,
                        json=payload,
                        headers={"X-Request-Id": request_id}
                    )

                response.raise_for_status()
                return response

        try:
            # Use circuit breaker if enabled
            if self.circuit_breaker:
                response = self.circuit_breaker.call(
                    exponential_backoff_retry,
                    make_request,
                    max_retries=self.max_retries
                )
            else:
                response = exponential_backoff_retry(
                    make_request,
                    max_retries=self.max_retries
                )

            return response.json()

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e, request_id, endpoint, payload)

        except httpx.TimeoutException as e:
            self._handle_timeout(e, request_id, endpoint)

        except Exception as e:
            self._handle_unexpected_error(e, request_id, endpoint)

    def _handle_http_error(
        self,
        error: httpx.HTTPStatusError,
        request_id: str,
        endpoint: str,
        payload: Optional[dict]
    ):
        """Handle HTTP status errors."""
        status_code = error.response.status_code

        try:
            error_data = error.response.json()
            error_code = error_data.get("error_code", "UNKNOWN")
            detail = error_data.get("detail", str(error))
        except:
            error_code = "UNKNOWN"
            detail = error.response.text

        self.logger.error(
            f"ISL HTTP error: {status_code}",
            extra={
                "request_id": request_id,
                "endpoint": endpoint,
                "status_code": status_code,
                "error_code": error_code,
                "detail": detail
            }
        )

        raise ISLError(
            message=detail,
            error_code=error_code,
            status_code=status_code,
            request_id=request_id
        )

    def _handle_timeout(self, error: Exception, request_id: str, endpoint: str):
        """Handle timeout errors."""
        self.logger.error(
            f"ISL timeout: {endpoint}",
            extra={"request_id": request_id, "endpoint": endpoint}
        )

        raise ISLError(
            message=f"Request timed out after {self.timeout}s",
            error_code="TIMEOUT",
            status_code=504,
            request_id=request_id
        )

    def _handle_unexpected_error(self, error: Exception, request_id: str, endpoint: str):
        """Handle unexpected errors."""
        self.logger.exception(
            f"Unexpected ISL error: {endpoint}",
            extra={"request_id": request_id, "endpoint": endpoint}
        )

        raise ISLError(
            message=str(error),
            error_code="UNEXPECTED_ERROR",
            status_code=500,
            request_id=request_id
        )


class ISLError(Exception):
    """Custom exception for ISL errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int,
        request_id: str
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.request_id = request_id

        super().__init__(
            f"[{error_code}] {message} (request_id: {request_id})"
        )


# Example usage
if __name__ == "__main__":
    client = ISLClient(
        base_url="http://localhost:8000",
        timeout=10.0,
        max_retries=3,
        enable_circuit_breaker=True
    )

    try:
        result = client.request(
            method="POST",
            endpoint="/api/v1/causal/validate",
            payload={
                "dag": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
                "treatment": "A",
                "outcome": "B"
            }
        )

        print(f"✓ Success: {result['status']}")

    except ISLError as e:
        print(f"✗ ISL Error: {e.error_code}")
        print(f"  Message: {e.message}")
        print(f"  Request ID: {e.request_id}")

        # Handle specific errors
        if e.error_code == "NODE_NOT_FOUND":
            print("  → Fix DAG node references")
        elif e.error_code == "TIMEOUT":
            print("  → Reduce complexity or increase timeout")
```

---

## Error Handling Checklist

- [ ] **Implement retry logic** with exponential backoff
- [ ] **Add circuit breaker** for cascading failure prevention
- [ ] **Set appropriate timeouts** per endpoint
- [ ] **Log errors with request IDs** for tracing
- [ ] **Validate inputs locally** before sending to ISL
- [ ] **Handle 4xx errors** gracefully (don't retry)
- [ ] **Retry 5xx errors** with backoff
- [ ] **Monitor error rates** and alert on spikes
- [ ] **Cache results** to reduce load during failures
- [ ] **Provide fallback behavior** when ISL unavailable

---

## Next Steps

- Test error scenarios in development environment
- Set up error monitoring and alerting
- Review [Performance Guide](PLOT_PERFORMANCE_GUIDE.md) for optimization

---

**Last Updated**: 2025-11-20
**Document Version**: 1.0.0
**ISL Version**: 1.0.0
