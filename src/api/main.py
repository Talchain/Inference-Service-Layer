"""
Main FastAPI application for Inference Service Layer.

This module sets up the FastAPI app with all routers, middleware,
and exception handlers.
"""

import logging
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
from src.middleware.circuit_breaker import MemoryCircuitBreaker
from src.middleware.rate_limiting import RateLimitMiddleware
from src.models.responses import ErrorCode, ErrorResponse
from src.utils.tracing import TracingMiddleware

from .analysis import router as analysis_router
from .batch import router as batch_router
from .causal import router as causal_router
from .cee import router as cee_router
# ARCHIVED: Deliberation deferred to TAE PoC v02
# from .deliberation import router as deliberation_router
from .explain import router as explain_router
from .health import router as health_router
from .metrics import router as metrics_router
# ARCHIVED: Preferences deferred to TAE PoC v02
# from .preferences import router as preferences_router
from .robustness import router as robustness_router
from .teaching import router as teaching_router
from .team import router as team_router
from .validation import router as validation_router

# Setup logging
logger = setup_logging()
settings = get_settings()


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

# Memory circuit breaker (reject requests when memory >85%)
app.add_middleware(MemoryCircuitBreaker, threshold_percent=85.0)

# Distributed tracing (adds X-Trace-Id to all requests/responses)
app.add_middleware(TracingMiddleware)

# Configure CORS middleware
# For production: Set specific origins via environment variable
CORS_ORIGINS = [
    "http://localhost:3000",  # Local development
    "http://localhost:8080",  # Alternative dev port
]

# In development mode, allow all origins for easier testing
if settings.RELOAD:
    CORS_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-Request-Id", "X-Trace-Id"],
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
    request_id = f"{int(time.time() * 1000)}"

    # Track active requests
    active_requests.inc()

    # Extract endpoint pattern (strip query params)
    endpoint = request.url.path

    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": endpoint,
            "client": request.client.host if request.client else "unknown",
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
