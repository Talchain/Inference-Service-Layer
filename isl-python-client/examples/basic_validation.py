"""
Basic causal validation workflow.

This example demonstrates:
1. Checking if a causal effect is identifiable
2. Handling non-identifiable cases
3. Getting adjustment strategies
"""

import asyncio
from isl_client import ISLClient


async def main():
    """Run basic validation examples."""
    async with ISLClient(base_url="http://localhost:8000") as client:
        # Example 1: Simple identifiable DAG
        print("=" * 60)
        print("Example 1: Identifiable Effect")
        print("=" * 60)

        dag = {
            "nodes": ["Price", "Quality", "Revenue"],
            "edges": [["Price", "Revenue"], ["Quality", "Revenue"]],
        }

        result = await client.causal.validate(
            dag=dag, treatment="Price", outcome="Revenue"
        )

        print(f"Status: {result.status}")
        print(f"Method: {result.method}")
        print(f"Adjustment sets: {result.adjustment_sets}")
        print(f"\n{result.explanation.summary}")

        # Example 2: Confounded effect (not identifiable)
        print("\n" + "=" * 60)
        print("Example 2: Non-Identifiable Effect")
        print("=" * 60)

        confounded_dag = {
            "nodes": ["Price", "Revenue"],
            "edges": [["Price", "Revenue"]],
            # Hidden confounder U affects both Price and Revenue
        }

        result2 = await client.causal.validate(
            dag=confounded_dag, treatment="Price", outcome="Revenue"
        )

        print(f"Status: {result2.status}")

        if result2.suggestions:
            print("\nSuggestions to make it identifiable:")
            for suggestion in result2.suggestions:
                print(f"\n[{suggestion.priority.upper()}] {suggestion.description}")
                print(f"Action: {suggestion.action.action_type}")
                print(f"Details: {suggestion.technical_detail}")

        # Example 3: Get complete adjustment strategies
        print("\n" + "=" * 60)
        print("Example 3: Adjustment Strategies")
        print("=" * 60)

        complex_dag = {
            "nodes": ["X", "Y", "Z", "W"],
            "edges": [["X", "Y"], ["Z", "X"], ["Z", "Y"], ["W", "Z"]],
        }

        strategies = await client.causal.validate_with_strategies(
            dag=complex_dag, treatment="X", outcome="Y"
        )

        print(f"Found {len(strategies.strategies)} adjustment strategies:\n")

        for i, strategy in enumerate(strategies.strategies, 1):
            print(f"{i}. Type: {strategy.strategy_type}")
            print(f"   Explanation: {strategy.explanation}")
            print(f"   Expected identifiability: {strategy.expected_identifiability:.2%}")

            if strategy.nodes_to_add:
                print(f"   Nodes to add: {strategy.nodes_to_add}")

            print()


if __name__ == "__main__":
    asyncio.run(main())
