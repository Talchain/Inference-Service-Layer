"""
PLoT Engine Client for ISL.

HTTP client for interfacing with the PLoT (Platform for Learning over Time) engine.
Handles requests, retries, error handling, and logging.
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import asdict

import httpx

from src.models.plot_engine import (
    ChangeAttribution,
    ChangeDriver,
    CompareOption,
    CompareRequest,
    CompareResponse,
    EvidenceFreshness,
    IdempotencyMismatchError,
    ModelCard,
    RunRequest,
    RunResponse,
)

logger = logging.getLogger(__name__)


class PLoTEngineClient:
    """
    HTTP client for PLoT Engine.

    Features:
    - Automatic retries for network errors (not 409)
    - Evidence quality logging
    - Change attribution logging
    - Idempotency mismatch handling
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize PLoT engine client.

        Args:
            base_url: PLoT engine base URL (e.g., "https://plot.olumi.ai")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for network errors
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Create httpx client
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

        logger.info(
            "plot_client_initialized",
            extra={
                "base_url": base_url,
                "timeout": timeout,
                "max_retries": max_retries,
            }
        )

    async def run(self, request: RunRequest) -> RunResponse:
        """
        Execute PLoT /v1/run endpoint.

        Args:
            request: Run request with graph and options

        Returns:
            RunResponse with results and model card

        Raises:
            IdempotencyMismatchError: If 409 conflict on idempotency key
            httpx.HTTPError: For other HTTP errors
        """
        logger.info(
            "plot_run_request",
            extra={
                "idempotency_key": request.idempotency_key,
                "timeout_ms": request.timeout_ms,
            }
        )

        try:
            response_data = await self._post(
                "/v1/run",
                data=asdict(request)
            )

            # Parse response
            run_response = self._parse_run_response(response_data)

            # Log evidence quality if available
            if run_response.model_card.evidence_freshness:
                self._log_evidence_quality(run_response.model_card.evidence_freshness)

            logger.info(
                "plot_run_completed",
                extra={
                    "run_id": run_response.run_id,
                    "status": run_response.status,
                }
            )

            return run_response

        except IdempotencyMismatchError:
            logger.warning(
                "plot_run_idempotency_mismatch",
                extra={"idempotency_key": request.idempotency_key}
            )
            raise
        except Exception as e:
            logger.error(
                "plot_run_error",
                extra={"error": str(e)},
                exc_info=True
            )
            raise

    async def compare(self, request: CompareRequest) -> CompareResponse:
        """
        Execute PLoT /v1/compare endpoint.

        Args:
            request: Compare request with graph and scenarios

        Returns:
            CompareResponse with comparison results and attributions

        Raises:
            IdempotencyMismatchError: If 409 conflict on idempotency key
            httpx.HTTPError: For other HTTP errors
        """
        logger.info(
            "plot_compare_request",
            extra={
                "idempotency_key": request.idempotency_key,
                "num_scenarios": len(request.scenarios),
                "timeout_ms": request.timeout_ms,
            }
        )

        try:
            response_data = await self._post(
                "/v1/compare",
                data=asdict(request)
            )

            # Parse response
            compare_response = self._parse_compare_response(response_data)

            # Log change attributions
            for option in compare_response.options:
                if option.change_attribution:
                    logger.info(
                        "plot_change_attribution",
                        extra={
                            "option_id": option.option_id,
                            "outcome_delta": option.change_attribution.outcome_delta,
                            "summary": option.change_attribution.summary,
                            "num_drivers": len(option.change_attribution.primary_drivers),
                        }
                    )

            # Log evidence quality if available
            if compare_response.model_card.evidence_freshness:
                self._log_evidence_quality(compare_response.model_card.evidence_freshness)

            logger.info(
                "plot_compare_completed",
                extra={
                    "compare_id": compare_response.compare_id,
                    "status": compare_response.status,
                    "num_options": len(compare_response.options),
                }
            )

            return compare_response

        except IdempotencyMismatchError:
            logger.warning(
                "plot_compare_idempotency_mismatch",
                extra={"idempotency_key": request.idempotency_key}
            )
            raise
        except Exception as e:
            logger.error(
                "plot_compare_error",
                extra={"error": str(e)},
                exc_info=True
            )
            raise

    async def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make POST request with retry logic.

        Args:
            endpoint: API endpoint path
            data: Request data

        Returns:
            Response JSON data

        Raises:
            IdempotencyMismatchError: If 409 conflict
            httpx.HTTPError: For other HTTP errors
        """
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1

            try:
                response = await self.client.post(endpoint, json=data)

                # Handle 409 Idempotency Mismatch - DON'T RETRY
                if response.status_code == 409:
                    error_data = response.json()
                    if error_data.get("code") == "IDEMPOTENCY_MISMATCH":
                        raise IdempotencyMismatchError(
                            error_data.get("message", "Idempotency key mismatch"),
                            idempotency_key=data.get("idempotency_key")
                        )

                # Raise for other HTTP errors
                response.raise_for_status()

                # Return successful response
                return response.json()

            except IdempotencyMismatchError:
                # Never retry 409 errors
                raise
            except httpx.HTTPError as e:
                if attempt >= self.max_retries:
                    logger.error(
                        "plot_request_failed_max_retries",
                        extra={
                            "endpoint": endpoint,
                            "attempt": attempt,
                            "error": str(e),
                        }
                    )
                    raise

                # Log and retry
                logger.warning(
                    "plot_request_retry",
                    extra={
                        "endpoint": endpoint,
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "error": str(e),
                    }
                )

                # Exponential backoff: 2s, 4s, 8s
                import asyncio
                await asyncio.sleep(2 ** attempt)

    def _parse_run_response(self, data: Dict[str, Any]) -> RunResponse:
        """Parse /v1/run response data."""
        model_card_data = data.get("model_card", {})
        model_card = self._parse_model_card(model_card_data)

        return RunResponse(
            run_id=data["run_id"],
            status=data["status"],
            result=data.get("result"),
            model_card=model_card,
            metadata=data.get("metadata", {})
        )

    def _parse_compare_response(self, data: Dict[str, Any]) -> CompareResponse:
        """Parse /v1/compare response data."""
        model_card_data = data.get("model_card", {})
        model_card = self._parse_model_card(model_card_data)

        # Parse options with change attribution
        options = []
        for opt_data in data.get("options", []):
            change_attr = None
            if "change_attribution" in opt_data and opt_data["change_attribution"]:
                change_attr = self._parse_change_attribution(opt_data["change_attribution"])

            option = CompareOption(
                option_id=opt_data["option_id"],
                label=opt_data["label"],
                outcome_value=opt_data["outcome_value"],
                change_attribution=change_attr,
                metadata=opt_data.get("metadata", {})
            )
            options.append(option)

        return CompareResponse(
            compare_id=data["compare_id"],
            status=data["status"],
            options=options,
            model_card=model_card,
            metadata=data.get("metadata", {})
        )

    def _parse_model_card(self, data: Dict[str, Any]) -> ModelCard:
        """Parse model card data."""
        evidence_fresh = None
        if "evidence_freshness" in data and data["evidence_freshness"]:
            fresh_data = data["evidence_freshness"]
            evidence_fresh = EvidenceFreshness(
                total=fresh_data["total"],
                with_timestamp=fresh_data["with_timestamp"],
                oldest_days=fresh_data.get("oldest_days"),
                newest_days=fresh_data.get("newest_days"),
                buckets=fresh_data.get("buckets", {})
            )

        return ModelCard(
            model_id=data.get("model_id"),
            version=data.get("version"),
            created_at=data.get("created_at"),
            evidence_freshness=evidence_fresh,
            metadata=data.get("metadata", {})
        )

    def _parse_change_attribution(self, data: Dict[str, Any]) -> ChangeAttribution:
        """Parse change attribution data."""
        drivers = []
        for driver_data in data.get("primary_drivers", []):
            driver = ChangeDriver(
                change_type=driver_data["change_type"],
                description=driver_data["description"],
                contribution_to_delta=driver_data["contribution_to_delta"],
                contribution_pct=driver_data["contribution_pct"],
                affected_nodes=driver_data.get("affected_nodes", [])
            )
            drivers.append(driver)

        return ChangeAttribution(
            outcome_delta=data["outcome_delta"],
            primary_drivers=drivers,
            summary=data.get("summary", "")
        )

    def _log_evidence_quality(self, evidence: EvidenceFreshness) -> None:
        """Log evidence freshness metrics."""
        freshness_pct = (evidence.with_timestamp / evidence.total * 100) if evidence.total > 0 else 0

        logger.info(
            "plot_evidence_quality",
            extra={
                "total_evidence": evidence.total,
                "with_timestamp": evidence.with_timestamp,
                "freshness_pct": round(freshness_pct, 1),
                "oldest_days": evidence.oldest_days,
                "newest_days": evidence.newest_days,
                "buckets": evidence.buckets,
            }
        )

        # Warn if evidence is stale
        if evidence.oldest_days and evidence.oldest_days > 365:
            logger.warning(
                "plot_stale_evidence",
                extra={"oldest_days": evidence.oldest_days}
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
