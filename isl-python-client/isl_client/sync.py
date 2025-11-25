"""Synchronous wrapper for ISL client."""

import asyncio
from typing import Any

from .client import ISLClient
from .models import (
    BatchCounterfactualResponse,
    ConformalResponse,
    ContrastiveResponse,
    CounterfactualResponse,
    DiscoveryResponse,
    OptimizationResponse,
    StrategiesResponse,
    TransportabilityResponse,
    ValidationResponse,
)


def _run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running - create new one
        return asyncio.run(coro)
    else:
        # Event loop already running - use it
        return loop.run_until_complete(coro)


class CausalAPISync:
    """Synchronous wrapper for CausalAPI."""

    def __init__(self, async_client: ISLClient):
        self._async_client = async_client

    def validate(
        self,
        dag: dict[str, Any],
        treatment: str,
        outcome: str,
    ) -> ValidationResponse:
        """Synchronous version of validate()."""
        return _run_async(
            self._async_client.causal.validate(dag, treatment, outcome)
        )

    def validate_with_strategies(
        self,
        dag: dict[str, Any],
        treatment: str,
        outcome: str,
    ) -> StrategiesResponse:
        """Synchronous version of validate_with_strategies()."""
        return _run_async(
            self._async_client.causal.validate_with_strategies(dag, treatment, outcome)
        )

    def counterfactual(
        self,
        model: dict[str, Any],
        intervention: dict[str, float],
        seed: int | None = None,
    ) -> CounterfactualResponse:
        """Synchronous version of counterfactual()."""
        return _run_async(
            self._async_client.causal.counterfactual(model, intervention, seed)
        )

    def counterfactual_conformal(
        self,
        model: dict[str, Any],
        intervention: dict[str, float],
        calibration_data: list[dict[str, Any]],
        confidence: float = 0.95,
        seed: int | None = None,
    ) -> ConformalResponse:
        """Synchronous version of counterfactual_conformal()."""
        return _run_async(
            self._async_client.causal.counterfactual_conformal(
                model, intervention, calibration_data, confidence, seed
            )
        )

    def batch_counterfactuals(
        self,
        model: dict[str, Any],
        scenarios: list[dict[str, Any]],
        analyze_interactions: bool = True,
        seed: int | None = None,
    ) -> BatchCounterfactualResponse:
        """Synchronous version of batch_counterfactuals()."""
        return _run_async(
            self._async_client.causal.batch_counterfactuals(
                model, scenarios, analyze_interactions, seed
            )
        )

    def transport(
        self,
        source_domain: dict[str, Any],
        target_domain: dict[str, Any],
        treatment: str,
        outcome: str,
    ) -> TransportabilityResponse:
        """Synchronous version of transport()."""
        return _run_async(
            self._async_client.causal.transport(
                source_domain, target_domain, treatment, outcome
            )
        )


class ExplainAPISync:
    """Synchronous wrapper for ExplainAPI."""

    def __init__(self, async_client: ISLClient):
        self._async_client = async_client

    def contrastive(
        self,
        model: dict[str, Any],
        scenario_a: dict[str, Any],
        scenario_b: dict[str, Any],
        top_k: int = 5,
    ) -> ContrastiveResponse:
        """Synchronous version of contrastive()."""
        return _run_async(
            self._async_client.explain.contrastive(model, scenario_a, scenario_b, top_k)
        )

    def batch_contrastive(
        self,
        model: dict[str, Any],
        scenarios: list[dict[str, Any]],
        baseline_id: str,
        top_k: int = 5,
    ) -> list[ContrastiveResponse]:
        """Synchronous version of batch_contrastive()."""
        return _run_async(
            self._async_client.explain.batch_contrastive(
                model, scenarios, baseline_id, top_k
            )
        )


class OptimizeAPISync:
    """Synchronous wrapper for OptimizeAPI."""

    def __init__(self, async_client: ISLClient):
        self._async_client = async_client

    def sequential(
        self,
        model: dict[str, Any],
        objective: str,
        constraints: dict[str, Any] | None = None,
        horizon: int = 5,
        initial_state: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> OptimizationResponse:
        """Synchronous version of sequential()."""
        return _run_async(
            self._async_client.optimize.sequential(
                model, objective, constraints, horizon, initial_state, seed
            )
        )

    def multi_objective(
        self,
        model: dict[str, Any],
        objectives: list[str],
        weights: list[float] | None = None,
        constraints: dict[str, Any] | None = None,
        horizon: int = 5,
        pareto_frontier: bool = False,
        seed: int | None = None,
    ) -> OptimizationResponse | list[OptimizationResponse]:
        """Synchronous version of multi_objective()."""
        return _run_async(
            self._async_client.optimize.multi_objective(
                model, objectives, weights, constraints, horizon, pareto_frontier, seed
            )
        )


class DiscoveryAPISync:
    """Synchronous wrapper for DiscoveryAPI."""

    def __init__(self, async_client: ISLClient):
        self._async_client = async_client

    def from_data(
        self,
        data: list[dict[str, float]],
        variable_names: list[str],
        algorithm: str = "notears",
        prior_knowledge: dict[str, Any] | None = None,
        threshold: float = 0.3,
        seed: int | None = None,
    ) -> DiscoveryResponse:
        """Synchronous version of from_data()."""
        return _run_async(
            self._async_client.discovery.from_data(
                data, variable_names, algorithm, prior_knowledge, threshold, seed
            )
        )

    def from_knowledge(
        self,
        domain_description: str,
        variable_names: list[str],
        prior_knowledge: dict[str, Any] | None = None,
    ) -> DiscoveryResponse:
        """Synchronous version of from_knowledge()."""
        return _run_async(
            self._async_client.discovery.from_knowledge(
                domain_description, variable_names, prior_knowledge
            )
        )

    def hybrid(
        self,
        data: list[dict[str, float]],
        domain_description: str,
        variable_names: list[str],
        algorithm: str = "auto",
        data_weight: float = 0.7,
        seed: int | None = None,
    ) -> DiscoveryResponse:
        """Synchronous version of hybrid()."""
        return _run_async(
            self._async_client.discovery.hybrid(
                data, domain_description, variable_names, algorithm, data_weight, seed
            )
        )


class ISLClientSync:
    """
    Synchronous wrapper for ISLClient.

    Usage:
        with ISLClientSync(base_url="https://isl.olumi.ai") as client:
            result = client.causal.validate(dag, treatment, outcome)
            print(result.status)
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
        Initialize synchronous ISL client.

        Args:
            base_url: Base URL of ISL service
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_backoff_factor: Exponential backoff multiplier
        """
        self._async_client = ISLClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
        )
        self.causal = CausalAPISync(self._async_client)
        self.explain = ExplainAPISync(self._async_client)
        self.optimize = OptimizeAPISync(self._async_client)
        self.discovery = DiscoveryAPISync(self._async_client)

    def health(self) -> dict[str, Any]:
        """Check service health (synchronous)."""
        return _run_async(self._async_client.health())

    def close(self) -> None:
        """Close the client."""
        _run_async(self._async_client.close())

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
