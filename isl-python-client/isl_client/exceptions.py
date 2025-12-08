"""Custom exceptions for ISL client."""


class ISLException(Exception):
    """Base exception for ISL client errors."""

    pass


class ValidationError(ISLException):
    """Invalid request parameters or validation failed."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ServiceUnavailable(ISLException):
    """ISL service is unavailable or unreachable."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitError(ISLException):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(ISLException):
    """Authentication failed (invalid API key)."""

    pass


class NotFoundError(ISLException):
    """Requested resource not found."""

    pass


class TimeoutError(ISLException):
    """Request timed out."""

    pass


class CircuitBreakerError(ISLException):
    """Circuit breaker is open (service experiencing failures)."""

    def __init__(self, message: str, circuit_name: str):
        super().__init__(message)
        self.circuit_name = circuit_name
