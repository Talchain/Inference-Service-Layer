"""Causal inference API endpoints."""

from typing import TYPE_CHECKING, Any

from .models import (
    BatchCounterfactualResponse,
    ConformalResponse,
    CounterfactualResponse,
    StrategiesResponse,
    TransportabilityResponse,
    ValidationResponse,
)

if TYPE_CHECKING:
    from .client import ISLClient


class CausalAPI:
    """Causal inference endpoints."""

    def __init__(self, client: "ISLClient"):
        self._client = client

    async def validate(
        self,
        dag: dict[str, Any],
        treatment: str,
        outcome: str,
    ) -> ValidationResponse:
        """
        Validate causal identifiability.

        Checks if the causal effect of treatment on outcome is identifiable
        given the DAG structure.

        Args:
            dag: DAG structure as dict with "nodes" and "edges" keys
            treatment: Treatment variable name
            outcome: Outcome variable name

        Returns:
            ValidationResponse with identifiability status and suggestions

        Example:
            result = await client.causal.validate(
                dag={"nodes": ["X", "Y", "Z"], "edges": [["X", "Y"], ["Z", "Y"]]},
                treatment="X",
                outcome="Y"
            )
            print(result.status)  # "identifiable", "uncertain", or "cannot_identify"
        """
        response = await self._client.post(
            "/api/v1/causal/validate",
            json={
                "dag_structure": dag,
                "treatment": treatment,
                "outcome": outcome,
            },
        )
        return ValidationResponse.model_validate(response.json())

    async def validate_with_strategies(
        self,
        dag: dict[str, Any],
        treatment: str,
        outcome: str,
    ) -> StrategiesResponse:
        """
        Get complete adjustment strategies for identifiability.

        Returns detailed strategies (backdoor, frontdoor, instrumental)
        for making the causal effect identifiable.

        Args:
            dag: DAG structure
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            StrategiesResponse with ranked adjustment strategies

        Example:
            result = await client.causal.validate_with_strategies(...)
            for strategy in result.strategies:
                print(f"{strategy.strategy_type}: {strategy.explanation}")
        """
        response = await self._client.post(
            "/api/v1/validation/strategies",
            json={
                "dag_structure": dag,
                "treatment": treatment,
                "outcome": outcome,
            },
        )
        return StrategiesResponse.model_validate(response.json())

    async def counterfactual(
        self,
        model: dict[str, Any],
        intervention: dict[str, float],
        seed: int | None = None,
    ) -> CounterfactualResponse:
        """
        Generate counterfactual prediction.

        Predicts what would happen if we intervene to set variables
        to specific values.

        Args:
            model: Structural causal model (SCM) specification
            intervention: Variables to intervene on and their values
            seed: Random seed for reproducibility

        Returns:
            CounterfactualResponse with prediction and explanation

        Example:
            result = await client.causal.counterfactual(
                model=scm,
                intervention={"Price": 45.0},
                seed=42
            )
            print(result.prediction.prediction)  # {"Revenue": 1250.0}
        """
        payload: dict[str, Any] = {
            "model": model,
            "intervention": intervention,
        }
        if seed is not None:
            payload["seed"] = seed

        response = await self._client.post(
            "/api/v1/causal/counterfactual",
            json=payload,
        )
        return CounterfactualResponse.model_validate(response.json())

    async def counterfactual_conformal(
        self,
        model: dict[str, Any],
        intervention: dict[str, float],
        calibration_data: list[dict[str, Any]],
        confidence: float = 0.95,
        seed: int | None = None,
    ) -> ConformalResponse:
        """
        Counterfactual prediction with conformal intervals.

        Provides finite-sample valid prediction intervals using
        conformal prediction.

        Args:
            model: Structural causal model
            intervention: Variables to intervene on
            calibration_data: Historical data for calibration
            confidence: Desired coverage level (default: 0.95)
            seed: Random seed

        Returns:
            ConformalResponse with guaranteed coverage intervals

        Example:
            result = await client.causal.counterfactual_conformal(
                model=scm,
                intervention={"Price": 45.0},
                calibration_data=historical_data,
                confidence=0.95
            )
            print(f"95% interval: [{result.conformal_interval.lower}, "
                  f"{result.conformal_interval.upper}]")
            print(f"Coverage guaranteed: {result.coverage_guarantee.guaranteed}")
        """
        payload: dict[str, Any] = {
            "model": model,
            "intervention": intervention,
            "calibration_data": calibration_data,
            "alpha": 1.0 - confidence,
        }
        if seed is not None:
            payload["seed"] = seed

        response = await self._client.post(
            "/api/v1/causal/conformal",
            json=payload,
        )
        return ConformalResponse.model_validate(response.json())

    async def batch_counterfactuals(
        self,
        model: dict[str, Any],
        scenarios: list[dict[str, Any]],
        analyze_interactions: bool = True,
        seed: int | None = None,
    ) -> BatchCounterfactualResponse:
        """
        Analyze multiple counterfactual scenarios.

        Evaluates multiple interventions and optionally analyzes
        interactions/synergies between them.

        Args:
            model: Structural causal model
            scenarios: List of scenarios, each with "id" and "intervention"
            analyze_interactions: Whether to analyze synergistic effects
            seed: Random seed

        Returns:
            BatchCounterfactualResponse with all scenarios and interaction analysis

        Example:
            result = await client.causal.batch_counterfactuals(
                model=scm,
                scenarios=[
                    {"id": "base", "intervention": {"Price": 40}},
                    {"id": "aggressive", "intervention": {"Price": 50}},
                ],
                analyze_interactions=True
            )
            for scenario in result.scenarios:
                print(f"{scenario.scenario_id}: {scenario.prediction}")
            if result.interactions:
                print(f"Synergy detected: {result.interactions.has_synergy}")
        """
        payload: dict[str, Any] = {
            "model": model,
            "scenarios": scenarios,
            "analyze_interactions": analyze_interactions,
        }
        if seed is not None:
            payload["seed"] = seed

        response = await self._client.post(
            "/api/v1/batch/counterfactuals",
            json=payload,
        )
        return BatchCounterfactualResponse.model_validate(response.json())

    async def transport(
        self,
        source_domain: dict[str, Any],
        target_domain: dict[str, Any],
        treatment: str,
        outcome: str,
    ) -> TransportabilityResponse:
        """
        Check causal effect transportability across domains.

        Validates whether a causal effect learned in one domain
        can be applied to another domain.

        Args:
            source_domain: Source domain description (DAG, data characteristics)
            target_domain: Target domain description
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            TransportabilityResponse with validity conditions

        Example:
            result = await client.causal.transport(
                source_domain={"dag": lab_dag, "population": "healthy_adults"},
                target_domain={"dag": field_dag, "population": "general"},
                treatment="Drug",
                outcome="Recovery"
            )
            print(f"Transportable: {result.transportable}")
            if result.adaptation_required:
                print("Adaptations needed:", result.suggestions)
        """
        response = await self._client.post(
            "/api/v1/causal/transport",
            json={
                "source_domain": source_domain,
                "target_domain": target_domain,
                "treatment": treatment,
                "outcome": outcome,
            },
        )
        return TransportabilityResponse.model_validate(response.json())
