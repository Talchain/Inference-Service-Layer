"""Contrastive explanation API endpoints."""

from typing import TYPE_CHECKING, Any

from .models import ContrastiveResponse

if TYPE_CHECKING:
    from .client import ISLClient


class ExplainAPI:
    """Contrastive explanation endpoints."""

    def __init__(self, client: "ISLClient"):
        self._client = client

    async def contrastive(
        self,
        model: dict[str, Any],
        scenario_a: dict[str, Any],
        scenario_b: dict[str, Any],
        top_k: int = 5,
    ) -> ContrastiveResponse:
        """
        Generate contrastive explanation between two scenarios.

        Explains why scenario A has a different outcome than scenario B
        by identifying key causal differences.

        Args:
            model: Structural causal model
            scenario_a: First scenario (with "id" and "state" or "intervention")
            scenario_b: Second scenario to compare against
            top_k: Number of top differences to return (default: 5)

        Returns:
            ContrastiveResponse with explanations and importance scores

        Example:
            result = await client.explain.contrastive(
                model=scm,
                scenario_a={"id": "success", "state": {"Price": 40, "Quality": 8}},
                scenario_b={"id": "failure", "state": {"Price": 60, "Quality": 6}},
                top_k=3
            )
            print(result.summary)
            for explanation in result.explanations:
                print(f"Key difference: {explanation.counterfactual_path}")
        """
        response = await self._client.post(
            "/api/v1/explain/contrastive",
            json={
                "model": model,
                "scenario_a": scenario_a,
                "scenario_b": scenario_b,
                "top_k": top_k,
            },
        )
        return ContrastiveResponse.model_validate(response.json())

    async def batch_contrastive(
        self,
        model: dict[str, Any],
        scenarios: list[dict[str, Any]],
        baseline_id: str,
        top_k: int = 5,
    ) -> list[ContrastiveResponse]:
        """
        Generate contrastive explanations for multiple scenarios against a baseline.

        Args:
            model: Structural causal model
            scenarios: List of scenarios to compare
            baseline_id: ID of baseline scenario to compare against
            top_k: Number of top differences per comparison

        Returns:
            List of ContrastiveResponse, one for each scenario vs. baseline

        Example:
            results = await client.explain.batch_contrastive(
                model=scm,
                scenarios=[
                    {"id": "base", "state": {...}},
                    {"id": "var1", "state": {...}},
                    {"id": "var2", "state": {...}},
                ],
                baseline_id="base"
            )
            for result in results:
                print(result.summary)
        """
        response = await self._client.post(
            "/api/v1/explain/batch_contrastive",
            json={
                "model": model,
                "scenarios": scenarios,
                "baseline_id": baseline_id,
                "top_k": top_k,
            },
        )
        data = response.json()
        return [ContrastiveResponse.model_validate(item) for item in data]
