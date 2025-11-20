"""
Main FastAPI application for Inference Service Layer.

This module sets up the FastAPI app with all routers, middleware,
and exception handlers.
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.config import get_settings, setup_logging
from src.models.responses import ErrorCode, ErrorResponse

from .analysis import router as analysis_router
from .causal import router as causal_router
from .health import router as health_router
from .preferences import router as preferences_router
from .teaching import router as teaching_router
from .team import router as team_router

# Setup logging
logger = setup_logging()
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Response:
    """
    Log all incoming requests and responses.

    Tracks:
    - Request method, path, and client
    - Response status and duration
    - Any errors that occur
    """
    start_time = time.time()
    request_id = f"{int(time.time() * 1000)}"

    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        },
    )

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return response

    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000

        logger.error(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error": str(exc),
                "duration_ms": round(duration_ms, 2),
            },
            exc_info=True,
        )

        raise


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
app.include_router(
    causal_router,
    prefix=f"{settings.API_V1_PREFIX}/causal",
    tags=["Causal Inference"],
)
app.include_router(
    preferences_router,
    prefix=f"{settings.API_V1_PREFIX}/preferences",
    tags=["Preference Learning"],
)
app.include_router(
    teaching_router,
    prefix=f"{settings.API_V1_PREFIX}/teaching",
    tags=["Bayesian Teaching"],
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


@app.on_event("startup")
async def startup_event() -> None:
    """Run on application startup."""
    logger.info(
        "application_startup",
        extra={
            "version": settings.VERSION,
            "environment": "production" if not settings.RELOAD else "development",
        },
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Run on application shutdown."""
    logger.info("application_shutdown")


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
