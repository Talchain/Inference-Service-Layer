"""Causal discovery API endpoints."""

from typing import TYPE_CHECKING, Any

from .models import DiscoveryResponse

if TYPE_CHECKING:
    from .client import ISLClient


class DiscoveryAPI:
    """Causal discovery endpoints."""

    def __init__(self, client: "ISLClient"):
        self._client = client

    async def from_data(
        self,
        data: list[dict[str, float]],
        variable_names: list[str],
        algorithm: str = "notears",
        prior_knowledge: dict[str, Any] | None = None,
        threshold: float = 0.3,
        seed: int | None = None,
    ) -> DiscoveryResponse:
        """
        Discover causal DAG structure from observational data.

        Uses advanced algorithms (NOTEARS, PC) to learn causal relationships
        from data.

        Args:
            data: Observational data as list of dicts (one per sample)
            variable_names: Names of variables in the data
            algorithm: Discovery algorithm ("notears", "pc", or "auto")
            prior_knowledge: Optional prior knowledge (forbidden/required edges)
            threshold: Edge significance threshold (default: 0.3)
            seed: Random seed

        Returns:
            DiscoveryResponse with discovered DAG and alternatives

        Example:
            result = await client.discovery.from_data(
                data=[
                    {"Price": 40, "Quality": 7, "Revenue": 1200},
                    {"Price": 45, "Quality": 8, "Revenue": 1350},
                    # ... more samples
                ],
                variable_names=["Price", "Quality", "Revenue"],
                algorithm="notears",
                prior_knowledge={
                    "forbidden_edges": [["Revenue", "Price"]],  # No reverse causation
                    "required_edges": [["Quality", "Revenue"]]
                }
            )
            print(f"Discovered {len(result.dag.edges)} edges")
            print(f"Confidence: {result.confidence}")
        """
        payload: dict[str, Any] = {
            "data": data,
            "variable_names": variable_names,
            "algorithm": algorithm,
            "threshold": threshold,
        }
        if prior_knowledge:
            payload["prior_knowledge"] = prior_knowledge
        if seed is not None:
            payload["seed"] = seed

        response = await self._client.post(
            "/api/v1/causal/discover",
            json=payload,
        )
        return DiscoveryResponse.model_validate(response.json())

    async def from_knowledge(
        self,
        domain_description: str,
        variable_names: list[str],
        prior_knowledge: dict[str, Any] | None = None,
    ) -> DiscoveryResponse:
        """
        Discover causal DAG from domain knowledge.

        Uses domain description (natural language) to infer plausible
        causal structures.

        Args:
            domain_description: Plain English description of the domain
            variable_names: Variables to include in the DAG
            prior_knowledge: Optional constraints on the structure

        Returns:
            DiscoveryResponse with inferred DAG structures

        Example:
            result = await client.discovery.from_knowledge(
                domain_description=
                    "In e-commerce, product price affects both purchase decision "
                    "and customer satisfaction. Quality also affects satisfaction. "
                    "Advertising spend influences traffic.",
                variable_names=["Price", "Quality", "Advertising", "Traffic", "Sales"],
                prior_knowledge={
                    "required_edges": [["Price", "Sales"]]
                }
            )
            print(f"Inferred structure: {result.dag.edges}")
        """
        payload: dict[str, Any] = {
            "domain_description": domain_description,
            "variable_names": variable_names,
        }
        if prior_knowledge:
            payload["prior_knowledge"] = prior_knowledge

        response = await self._client.post(
            "/api/v1/causal/discover/knowledge",
            json=payload,
        )
        return DiscoveryResponse.model_validate(response.json())

    async def hybrid(
        self,
        data: list[dict[str, float]],
        domain_description: str,
        variable_names: list[str],
        algorithm: str = "auto",
        data_weight: float = 0.7,
        seed: int | None = None,
    ) -> DiscoveryResponse:
        """
        Hybrid discovery combining data and domain knowledge.

        Combines statistical discovery from data with domain expertise
        for more robust structure learning.

        Args:
            data: Observational data
            domain_description: Domain knowledge description
            variable_names: Variable names
            algorithm: Discovery algorithm
            data_weight: Weight for data vs. knowledge (0.0-1.0, default: 0.7)
            seed: Random seed

        Returns:
            DiscoveryResponse with hybrid DAG

        Example:
            result = await client.discovery.hybrid(
                data=observational_data,
                domain_description="Price affects demand. Quality affects satisfaction.",
                variable_names=["Price", "Quality", "Demand", "Satisfaction"],
                data_weight=0.6  # Balance data and domain knowledge
            )
        """
        payload: dict[str, Any] = {
            "data": data,
            "domain_description": domain_description,
            "variable_names": variable_names,
            "algorithm": algorithm,
            "data_weight": data_weight,
        }
        if seed is not None:
            payload["seed"] = seed

        response = await self._client.post(
            "/api/v1/causal/discover/hybrid",
            json=payload,
        )
        return DiscoveryResponse.model_validate(response.json())
