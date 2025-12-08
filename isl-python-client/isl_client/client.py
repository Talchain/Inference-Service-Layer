"""Core ISL client with retry logic and error handling."""

import asyncio
import logging
from typing import Any

import httpx

from .exceptions import (
    AuthenticationError,
    ISLException,
    NotFoundError,
    RateLimitError,
    ServiceUnavailable,
    TimeoutError,
    ValidationError,
)

logger = logging.getLogger(__name__)


class ISLClient:
    """
    Type-safe async client for Olumi Inference Service Layer.

    Usage:
        client = ISLClient(base_url="https://isl.olumi.ai", api_key="...")
        result = await client.causal.validate(dag, treatment, outcome)
        await client.close()

    Or use as context manager:
        async with ISLClient(...) as client:
            result = await client.causal.validate(...)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
    ):
        """
        Initialize ISL client.

        Args:
            base_url: Base URL of ISL service (e.g., "https://isl.olumi.ai")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retry attempts (default: 3)
            retry_backoff_factor: Exponential backoff multiplier (default: 2.0)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
        )

        # Lazy-load API modules to avoid circular imports
        self._causal_api = None
        self._explain_api = None
        self._optimize_api = None
        self._discovery_api = None

    @property
    def causal(self):
        """Access causal inference APIs."""
        if self._causal_api is None:
            from .causal import CausalAPI

            self._causal_api = CausalAPI(self)
        return self._causal_api

    @property
    def explain(self):
        """Access contrastive explanation APIs."""
        if self._explain_api is None:
            from .explain import ExplainAPI

            self._explain_api = ExplainAPI(self)
        return self._explain_api

    @property
    def optimize(self):
        """Access sequential optimization APIs."""
        if self._optimize_api is None:
            from .optimize import OptimizeAPI

            self._optimize_api = OptimizeAPI(self)
        return self._optimize_api

    @property
    def discovery(self):
        """Access causal discovery APIs."""
        if self._discovery_api is None:
            from .discovery import DiscoveryAPI

            self._discovery_api = DiscoveryAPI(self)
        return self._discovery_api

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Execute HTTP request with retry logic and error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., "/api/v1/causal/validate")
            **kwargs: Additional arguments for httpx request

        Returns:
            httpx.Response object

        Raises:
            ValidationError: For 400-level errors
            AuthenticationError: For 401 errors
            NotFoundError: For 404 errors
            RateLimitError: For 429 errors
            ServiceUnavailable: For 500-level errors after retries
            TimeoutError: For timeout errors
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Request attempt {attempt + 1}/{self.max_retries}: {method} {path}"
                )

                response = await self._client.request(method, path, **kwargs)

                # Handle specific status codes
                if response.status_code == 200:
                    return response

                elif response.status_code == 401:
                    raise AuthenticationError("Authentication failed. Check your API key.")

                elif response.status_code == 404:
                    raise NotFoundError(f"Endpoint not found: {path}")

                elif response.status_code == 429:
                    # Rate limit - get retry-after header if available
                    retry_after = response.headers.get("Retry-After")
                    retry_after_seconds = int(retry_after) if retry_after else None
                    raise RateLimitError(
                        "Rate limit exceeded. Try again later.",
                        retry_after=retry_after_seconds,
                    )

                elif 400 <= response.status_code < 500:
                    # Client error - parse error response
                    try:
                        error_data = response.json()
                        raise ValidationError(
                            error_data.get("message", "Validation failed"),
                            details=error_data.get("details", {}),
                        )
                    except (ValueError, KeyError):
                        raise ValidationError(
                            f"Request failed with status {response.status_code}: {response.text}"
                        )

                elif response.status_code >= 500:
                    # Server error - retry
                    logger.warning(
                        f"Server error {response.status_code} on attempt {attempt + 1}"
                    )

                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        delay = self.retry_backoff_factor**attempt
                        logger.debug(f"Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise ServiceUnavailable(
                            f"Service unavailable after {self.max_retries} retries. "
                            f"Status: {response.status_code}"
                        )

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(f"Request timeout on attempt {attempt + 1}: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.retry_backoff_factor**attempt
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise TimeoutError(
                        f"Request timed out after {self.max_retries} attempts"
                    ) from e

            except httpx.NetworkError as e:
                last_exception = e
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.retry_backoff_factor**attempt
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise ServiceUnavailable(
                        f"Network error after {self.max_retries} attempts: {e}"
                    ) from e

            except (AuthenticationError, NotFoundError, RateLimitError, ValidationError):
                # Don't retry these errors
                raise

        # Should never reach here, but just in case
        if last_exception:
            raise ServiceUnavailable(
                f"Request failed after {self.max_retries} attempts"
            ) from last_exception

        raise ISLException("Unexpected error in request handling")

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:
        """Execute GET request."""
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        """Execute POST request."""
        return await self._request("POST", path, **kwargs)

    async def put(self, path: str, **kwargs: Any) -> httpx.Response:
        """Execute PUT request."""
        return await self._request("PUT", path, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        """Execute DELETE request."""
        return await self._request("DELETE", path, **kwargs)

    async def health(self) -> dict[str, Any]:
        """
        Check service health.

        Returns:
            Health status dict

        Raises:
            ServiceUnavailable: If health check fails
        """
        response = await self.get("/health")
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
