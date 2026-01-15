"""
Main FastAPI application for Inference Service Layer.

This module sets up the FastAPI app with all routers, middleware,
and exception handlers.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from src.config import get_settings, setup_logging
from src.middleware.auth import APIKeyAuthMiddleware
from src.middleware.circuit_breaker import MemoryCircuitBreaker
from src.middleware.observability import ObservabilityMiddleware
from src.middleware.rate_limiting import RateLimitMiddleware
from src.middleware.request_limits import RequestSizeLimitMiddleware, RequestTimeoutMiddleware
from src.models.responses import ErrorCode, ErrorResponse
from src.utils.ip_extraction import get_client_ip
from src.utils.tracing import TracingMiddleware, get_trace_id

from .aggregation import router as aggregation_router
from .analysis import router as analysis_router
from .batch import router as batch_router
from .causal import router as causal_router
from .cee import router as cee_router
# ARCHIVED: Deliberation deferred to TAE PoC v02
# from .deliberation import router as deliberation_router
from .dominance import router as dominance_router
from .explain import router as explain_router
from .health import router as health_router
from .metrics import router as metrics_router
from .phase4 import router as phase4_router
# ARCHIVED: Preferences deferred to TAE PoC v02
# from .preferences import router as preferences_router
from .risk import router as risk_router
from .robustness import router as robustness_router
from .teaching import router as teaching_router
from .team import router as team_router
from .threshold import router as threshold_router
from .utility import router as utility_router
from .validation import router as validation_router
from .identifiability import router as identifiability_router
from .decision_robustness import router as decision_robustness_router
from .outcomes import router as outcomes_router

# Setup logging
logger = setup_logging()
settings = get_settings()


# =============================================================================
# Sentry Error Tracking
# =============================================================================
def _init_sentry() -> None:
    """
    Initialize Sentry error tracking if enabled.

    Configures:
    - FastAPI and Starlette integrations
    - Performance tracing
    - Request ID correlation
    - Release tracking
    """
    if not settings.SENTRY_ENABLED:
        logger.info("Sentry error tracking disabled")
        return

    if not settings.SENTRY_DSN:
        logger.warning("SENTRY_ENABLED=true but SENTRY_DSN not configured - Sentry disabled")
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        def before_send_filter(event, hint):
            """Add request_id to Sentry events and filter sensitive data."""
            from src.utils.tracing import get_trace_id

            # Add request_id for correlation
            request_id = get_trace_id()
            if request_id:
                event.setdefault("tags", {})["request_id"] = request_id
                event.setdefault("extra", {})["request_id"] = request_id

            # Filter sensitive data from request headers
            if "request" in event:
                headers = event["request"].get("headers", {})
                if isinstance(headers, dict):
                    headers.pop("Authorization", None)
                    headers.pop("X-API-Key", None)
                    headers.pop("x-api-key", None)

            return event

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.SENTRY_ENVIRONMENT or settings.ENVIRONMENT,
            traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
            profiles_sample_rate=settings.SENTRY_PROFILES_SAMPLE_RATE,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                StarletteIntegration(transaction_style="endpoint"),
            ],
            before_send=before_send_filter,
            release=f"isl@{settings.VERSION}",
            send_default_pii=False,  # Don't send PII by default
        )

        logger.info(
            "Sentry error tracking initialized",
            extra={
                "environment": settings.SENTRY_ENVIRONMENT or settings.ENVIRONMENT,
                "traces_sample_rate": settings.SENTRY_TRACES_SAMPLE_RATE,
                "release": f"isl@{settings.VERSION}",
            }
        )

    except ImportError:
        logger.warning(
            "SENTRY_ENABLED=true but sentry-sdk not installed. "
            "Run: poetry add sentry-sdk[fastapi]"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")


# Initialize Sentry before app creation
_init_sentry()

# Validate production configuration
config_errors = settings.validate_production_config()
if config_errors:
    for error in config_errors:
        logger.error(f"Configuration error: {error}")
    if settings.is_production():
        logger.critical("Invalid configuration for production environment. Exiting.")
        sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Handles initialization on startup and cleanup on shutdown.
    """
    # Startup
    logger.info(
        "application_startup",
        extra={
            "version": settings.VERSION,
            "environment": "production" if not settings.RELOAD else "development",
        },
    )

    yield

    # Shutdown
    logger.info("application_shutdown")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Enable gzip compression (40-70% size reduction for JSON responses)
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # Only compress responses >1KB
    compresslevel=6     # Balance speed vs compression (1-9, 6 is good default)
)


# Security headers middleware (defense in depth)
@app.middleware("http")
async def add_security_headers(request: Request, call_next: Callable) -> Response:
    """
    Add security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff (prevent MIME sniffing)
    - X-Frame-Options: DENY (prevent clickjacking)
    - X-XSS-Protection: 1; mode=block (legacy XSS protection)
    - Strict-Transport-Security: HSTS for HTTPS (if applicable)
    - Content-Security-Policy: Restrictive CSP for API-only service
    """
    response = await call_next(request)

    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"

    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    # Legacy XSS protection (still useful for older browsers)
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # HSTS: Force HTTPS for 1 year
    # Check both direct HTTPS and X-Forwarded-Proto (for TLS-terminating proxies)
    is_https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto", "").lower() == "https"
    )
    if is_https:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    # CSP: API-only service shouldn't load any resources
    # This prevents any content injection attacks
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"

    return response


# Request size limit (DoS protection - reject oversized requests)
app.add_middleware(RequestSizeLimitMiddleware, max_size_mb=settings.MAX_REQUEST_SIZE_MB)

# Request timeout (resource protection - cancel long-running requests)
# Default: 60s, configurable via REQUEST_TIMEOUT_SECONDS
app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=settings.REQUEST_TIMEOUT_SECONDS)

# Memory circuit breaker (reject requests when memory >85%)
app.add_middleware(MemoryCircuitBreaker, threshold_percent=85.0)

# Observability (service headers, payload hashing, boundary logging)
# NOTE: Must be added BEFORE TracingMiddleware so TracingMiddleware runs first (LIFO)
app.add_middleware(ObservabilityMiddleware)

# Distributed tracing (adds X-Trace-Id to all requests/responses)
# Sets trace ID from X-Request-Id header or generates unique ID per request
app.add_middleware(TracingMiddleware)

# API Key Authentication (validates X-API-Key header)
# SECURITY: Authentication is REQUIRED by default. Explicit opt-out via ISL_AUTH_DISABLED=true.
# Supports both ISL_API_KEYS (preferred) and ISL_API_KEY (legacy) for backward compatibility
_api_keys_configured = bool(settings.ISL_API_KEYS or settings.ISL_API_KEY)

if not _api_keys_configured and not settings.ISL_AUTH_DISABLED:
    raise RuntimeError(
        "ISL_API_KEYS environment variable required. "
        "Provide comma-separated API keys, or set ISL_AUTH_DISABLED=true for local development only."
    )

app.add_middleware(
    APIKeyAuthMiddleware,
    api_keys=settings.ISL_API_KEYS or settings.ISL_API_KEY,
    auth_disabled=settings.ISL_AUTH_DISABLED
)

# Configure CORS middleware using settings
# SECURITY: No wildcard origins - explicit origins only
CORS_ORIGINS = settings.get_cors_origins_list()

# Log CORS configuration
logger.info(
    "CORS configured",
    extra={
        "origins": CORS_ORIGINS,
        "allow_credentials": settings.CORS_ALLOW_CREDENTIALS,
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    # Explicit headers only (no wildcard for security)
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-Id",
        "X-Correlation-Id",
        "Accept",
        "Accept-Language",
        "Content-Language",
        # Observability request headers (for caller-provided payload hash)
        "x-olumi-payload-hash",
        "x-olumi-caller-service",
        "x-olumi-trace-id",
    ],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-Request-Id",
        "X-Trace-Id",
        # Observability headers (Week 1)
        "x-olumi-service",
        "x-olumi-service-build",
        "x-olumi-response-hash",
        # Trace echo headers (for debugging)
        "x-olumi-trace-received",
        "x-olumi-downstream-calls",
    ],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Add rate limiting middleware (before request logging)
app.add_middleware(RateLimitMiddleware)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Response:
    """
    Log all incoming requests and responses with Prometheus metrics.

    Tracks:
    - Request method, path, and client
    - Response status and duration
    - Any errors that occur
    - Prometheus metrics (request counts, latencies, errors)
    """
    from .metrics import (
        http_requests_total,
        http_request_duration_seconds,
        http_errors_total,
        active_requests,
    )

    start_time = time.time()
    # Use canonical trace ID from TracingMiddleware for consistent correlation
    request_id = get_trace_id()

    # Track active requests
    active_requests.inc()

    # Extract endpoint pattern (strip query params)
    endpoint = request.url.path
    # Use unified IP extraction for consistency with auth/rate limiting
    client_ip = get_client_ip(request)

    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": endpoint,
            "client": client_ip,
        },
    )

    try:
        response = await call_next(request)
        duration_seconds = time.time() - start_time
        duration_ms = duration_seconds * 1000

        # Record metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration_seconds)

        # Track errors (4xx and 5xx)
        if response.status_code >= 400:
            http_errors_total.labels(
                method=request.method,
                endpoint=endpoint,
                error_code=response.status_code,
            ).inc()

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": endpoint,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response

    except Exception as exc:
        # Handle anyio.EndOfStream from async test clients (known Starlette bug)
        # https://github.com/encode/starlette/issues/1678
        # This occurs when validation errors are raised before middleware completes
        import anyio
        if isinstance(exc, (anyio.EndOfStream, anyio.WouldBlock)):
            # Decrement active requests before re-raising
            active_requests.dec()
            raise

        # Handle other exceptions
        duration_seconds = time.time() - start_time
        duration_ms = duration_seconds * 1000

        # Record error metrics
        http_errors_total.labels(
            method=request.method,
            endpoint=endpoint,
            error_code="500",
        ).inc()

        logger.error(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": endpoint,
                "error": str(exc),
                "duration_ms": round(duration_ms, 2),
            },
            exc_info=True,
        )

        raise

    finally:
        # Always decrement active requests
        active_requests.dec()


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTPException with Olumi Error Schema v1.0."""
    from src.models.responses import RecoveryHints
    from src.utils.tracing import get_trace_id

    # Extract request ID from headers (X-Request-Id or X-Trace-Id)
    request_id = request.headers.get("X-Request-Id") or request.headers.get("X-Trace-Id") or get_trace_id()

    # Map status codes to appropriate error codes
    if exc.status_code == 400:
        code = ErrorCode.INVALID_INPUT.value
        reason = "bad_request"
    elif exc.status_code == 404:
        code = ErrorCode.NODE_NOT_FOUND.value
        reason = "not_found"
    elif exc.status_code == 422:
        code = ErrorCode.VALIDATION_ERROR.value
        reason = "validation_failed"
    elif exc.status_code == 429:
        code = ErrorCode.RATE_LIMIT_EXCEEDED.value
        reason = "rate_limit"
    elif exc.status_code == 503:
        code = ErrorCode.SERVICE_UNAVAILABLE.value
        reason = "service_unavailable"
    elif exc.status_code == 504:
        code = ErrorCode.TIMEOUT.value
        reason = "timeout"
    else:
        code = ErrorCode.VALIDATION_ERROR.value
        reason = "http_error"

    error_response = ErrorResponse(
        code=code,
        message=str(exc.detail),
        reason=reason,
        recovery=RecoveryHints(
            hints=["Check your request parameters", "Refer to API documentation"],
            suggestion="Fix the input and retry",
        ),
        retryable=exc.status_code in [429, 503, 504],
        source="isl",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(exclude_none=True),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors with Olumi Error Schema v1.0."""
    from src.models.responses import RecoveryHints
    from src.utils.tracing import get_trace_id

    # Extract request ID
    request_id = request.headers.get("X-Request-Id") or request.headers.get("X-Trace-Id") or get_trace_id()

    # Extract validation failures
    errors = exc.errors()
    validation_failures = [
        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
        for err in errors
    ]

    # Create recovery hints
    hints = []
    if any("missing" in err["type"] for err in errors):
        hints.append("Ensure all required fields are provided")
    if any("type_error" in err["type"] for err in errors):
        hints.append("Check data types match the expected schema")
    if any("value_error" in err["type"] for err in errors):
        hints.append("Verify values are within valid ranges")
    if not hints:
        hints.append("Check the API documentation for correct request format")

    error_response = ErrorResponse(
        code=ErrorCode.VALIDATION_ERROR.value,
        message="Request validation failed",
        reason="invalid_schema",
        recovery=RecoveryHints(
            hints=hints,
            suggestion="Fix validation errors and retry",
            example="See validation_failures field for specific issues",
        ),
        validation_failures=validation_failures,
        retryable=False,
        source="isl",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=422,
        content=error_response.model_dump(exclude_none=True),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions with Olumi Error Schema v1.0."""
    from src.models.responses import RecoveryHints
    from src.utils.tracing import get_trace_id

    # Extract request ID
    request_id = request.headers.get("X-Request-Id") or request.headers.get("X-Trace-Id") or get_trace_id()

    logger.error(
        "unhandled_exception",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
            "request_id": request_id,
        },
        exc_info=True,
    )

    # Capture to Sentry with enriched context
    if settings.SENTRY_ENABLED:
        try:
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("request_id", request_id)
                scope.set_extra("path", request.url.path)
                scope.set_extra("method", request.method)
                scope.set_extra("query_params", str(request.query_params))
                sentry_sdk.capture_exception(exc)
        except ImportError:
            pass  # Sentry not installed

    # Determine error code based on exception type
    exc_type_name = type(exc).__name__
    if "timeout" in exc_type_name.lower() or "Timeout" in str(exc):
        code = ErrorCode.TIMEOUT.value
        reason = "computation_timeout"
        retryable = True
    elif "memory" in exc_type_name.lower() or "Memory" in str(exc):
        code = ErrorCode.MEMORY_LIMIT.value
        reason = "memory_exceeded"
        retryable = False
    else:
        code = ErrorCode.COMPUTATION_ERROR.value
        reason = "internal_error"
        retryable = True

    error_response = ErrorResponse(
        code=code,
        message="An unexpected error occurred during computation",
        reason=reason,
        recovery=RecoveryHints(
            hints=[
                "Check server logs for details",
                "Simplify your request if possible",
                "Contact support if the error persists"
            ],
            suggestion="Retry with the same input or contact support",
        ),
        retryable=retryable,
        source="isl",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(exclude_none=True),
    )


# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(metrics_router, tags=["Monitoring"])
app.include_router(
    causal_router,
    prefix=f"{settings.API_V1_PREFIX}/causal",
    tags=["Causal Inference"],
)
app.include_router(
    batch_router,
    prefix=f"{settings.API_V1_PREFIX}/batch",
    tags=["Batch Processing"],
)
# ARCHIVED: Preferences endpoint deferred to TAE PoC v02
# app.include_router(
#     preferences_router,
#     prefix=f"{settings.API_V1_PREFIX}/preferences",
#     tags=["Preference Learning"],
# )
app.include_router(
    teaching_router,
    prefix=f"{settings.API_V1_PREFIX}/teaching",
    tags=["Bayesian Teaching"],
)
app.include_router(
    validation_router,
    prefix=f"{settings.API_V1_PREFIX}/validation",
    tags=["Advanced Validation"],
)
app.include_router(
    utility_router,
    prefix=f"{settings.API_V1_PREFIX}/utility",
    tags=["Utility Functions"],
)
app.include_router(
    team_router,
    prefix=f"{settings.API_V1_PREFIX}/team",
    tags=["Team Alignment"],
)
app.include_router(
    analysis_router,
    prefix=f"{settings.API_V1_PREFIX}/analysis",
    tags=["Sensitivity Analysis"],
)
app.include_router(
    dominance_router,
    prefix=f"{settings.API_V1_PREFIX}/analysis",
    tags=["Multi-Criteria Analysis"],
)
app.include_router(
    risk_router,
    prefix=f"{settings.API_V1_PREFIX}/analysis",
    tags=["Multi-Criteria Analysis"],
)
app.include_router(
    threshold_router,
    prefix=f"{settings.API_V1_PREFIX}/analysis",
    tags=["Multi-Criteria Analysis"],
)
app.include_router(
    aggregation_router,
    prefix=f"{settings.API_V1_PREFIX}/aggregation",
    tags=["Multi-Criteria Analysis"],
)
app.include_router(
    robustness_router,
    prefix=f"{settings.API_V1_PREFIX}/robustness",
    tags=["FACET Robustness"],
)
app.include_router(
    explain_router,
    prefix=f"{settings.API_V1_PREFIX}/explain",
    tags=["Contrastive Explanations"],
)
app.include_router(
    cee_router,
    prefix=f"{settings.API_V1_PREFIX}",
    tags=["CEE Enhancement"],
)
app.include_router(
    phase4_router,
    prefix=f"{settings.API_V1_PREFIX}/analysis",
    tags=["Phase 4: Sequential Decisions"],
)
app.include_router(
    identifiability_router,
    prefix=f"{settings.API_V1_PREFIX}/analysis",
    tags=["Y₀ Identifiability"],
)
app.include_router(
    decision_robustness_router,
    prefix=f"{settings.API_V1_PREFIX}/analysis",
    tags=["Decision Robustness Suite"],
)
app.include_router(
    outcomes_router,
    prefix=f"{settings.API_V1_PREFIX}/outcomes",
    tags=["Outcome Logging"],
)
# ARCHIVED: Deliberation (Habermas Machine) deferred to TAE PoC v02
# app.include_router(
#     deliberation_router,
#     prefix=f"{settings.API_V1_PREFIX}/deliberation",
#     tags=["Habermas Machine"],
# )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
