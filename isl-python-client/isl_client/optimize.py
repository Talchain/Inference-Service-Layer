"""Sequential optimization API endpoints."""

from typing import TYPE_CHECKING, Any

from .models import OptimizationResponse

if TYPE_CHECKING:
    from .client import ISLClient


class OptimizeAPI:
    """Sequential optimization endpoints."""

    def __init__(self, client: "ISLClient"):
        self._client = client

    async def sequential(
        self,
        model: dict[str, Any],
        objective: str,
        constraints: dict[str, Any] | None = None,
        horizon: int = 5,
        initial_state: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> OptimizationResponse:
        """
        Find optimal sequential intervention policy.

        Optimizes a sequence of interventions over time to maximize
        an objective function while respecting constraints.

        Args:
            model: Structural causal model
            objective: Objective to maximize (variable name or expression)
            constraints: Optional constraints (budget, bounds, etc.)
            horizon: Planning horizon (number of steps, default: 5)
            initial_state: Initial state of system (if not default)
            seed: Random seed for reproducibility

        Returns:
            OptimizationResponse with optimal sequence and utility

        Example:
            result = await client.optimize.sequential(
                model=scm,
                objective="Revenue",
                constraints={
                    "budget": 1000,
                    "max_price_change": 10
                },
                horizon=5,
                initial_state={"Price": 40, "Inventory": 100}
            )
            print(f"Total utility: {result.total_utility}")
            for step in result.optimal_sequence:
                print(f"Step {step.step}: {step.intervention} -> {step.predicted_outcome}")
        """
        payload: dict[str, Any] = {
            "model": model,
            "objective": objective,
            "horizon": horizon,
        }
        if constraints:
            payload["constraints"] = constraints
        if initial_state:
            payload["initial_state"] = initial_state
        if seed is not None:
            payload["seed"] = seed

        response = await self._client.post(
            "/api/v1/optimize/sequential",
            json=payload,
        )
        return OptimizationResponse.model_validate(response.json())

    async def multi_objective(
        self,
        model: dict[str, Any],
        objectives: list[str],
        weights: list[float] | None = None,
        constraints: dict[str, Any] | None = None,
        horizon: int = 5,
        pareto_frontier: bool = False,
        seed: int | None = None,
    ) -> OptimizationResponse | list[OptimizationResponse]:
        """
        Multi-objective sequential optimization.

        Optimizes for multiple competing objectives simultaneously.

        Args:
            model: Structural causal model
            objectives: List of objectives to optimize
            weights: Weights for each objective (if not computing Pareto frontier)
            constraints: Optional constraints
            horizon: Planning horizon
            pareto_frontier: If True, return Pareto frontier instead of single solution
            seed: Random seed

        Returns:
            Single OptimizationResponse (if weights provided) or
            List of OptimizationResponse (if pareto_frontier=True)

        Example:
            # Weighted optimization
            result = await client.optimize.multi_objective(
                model=scm,
                objectives=["Revenue", "CustomerSatisfaction"],
                weights=[0.7, 0.3],
                horizon=5
            )

            # Pareto frontier
            frontier = await client.optimize.multi_objective(
                model=scm,
                objectives=["Revenue", "CustomerSatisfaction"],
                pareto_frontier=True,
                horizon=5
            )
            for solution in frontier:
                print(f"Utility: {solution.total_utility}")
        """
        payload: dict[str, Any] = {
            "model": model,
            "objectives": objectives,
            "horizon": horizon,
            "pareto_frontier": pareto_frontier,
        }
        if weights:
            payload["weights"] = weights
        if constraints:
            payload["constraints"] = constraints
        if seed is not None:
            payload["seed"] = seed

        response = await self._client.post(
            "/api/v1/optimize/multi_objective",
            json=payload,
        )

        data = response.json()
        if pareto_frontier:
            return [OptimizationResponse.model_validate(item) for item in data]
        else:
            return OptimizationResponse.model_validate(data)
