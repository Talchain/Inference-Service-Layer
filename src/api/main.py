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

# Configure CORS - NEVER use wildcard in production
# Get allowed origins from environment variable or use safe defaults
CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "")
if CORS_ORIGINS_ENV:
    # Parse comma-separated list from environment
    CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_ENV.split(",")]
elif settings.RELOAD:
    # Development mode - localhost only (NOT wildcard for security)
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
    ]
else:
    # Keep production defaults
    CORS_ORIGINS = [
        "https://plot.olumi.ai",
        "https://tae.olumi.ai",
        "https://cee.olumi.ai",
        "http://localhost:3000",
        "http://localhost:8080",
    ]

# Security validation: prevent wildcard in production
if not settings.RELOAD and "*" in CORS_ORIGINS:
    raise ValueError("SECURITY: Wildcard CORS not allowed in production!")

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
    """Handle HTTPException with structured error response."""
    error_response = ErrorResponse(
        error_code=ErrorCode.VALIDATION_ERROR,
        message=exc.detail,
        retryable=False,
        suggested_action="fix_input",
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    error_response = ErrorResponse(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details={"errors": exc.errors()},
        retryable=False,
        suggested_action="fix_input",
    )

    return JSONResponse(
        status_code=422,
        content=error_response.model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    logger.error(
        "unhandled_exception",
        extra={
            "path": request.url.path,
            "method": request.method,
            "error": str(exc),
        },
        exc_info=True,
    )

    error_response = ErrorResponse(
        error_code=ErrorCode.COMPUTATION_ERROR,
        message="An unexpected error occurred",
        retryable=True,
        suggested_action="retry_with_same_input",
    )

    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
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
