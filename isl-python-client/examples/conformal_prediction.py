"""
Conformal prediction example.

This example demonstrates:
1. Standard counterfactual prediction
2. Conformal prediction with coverage guarantees
3. Comparing conformal vs Monte Carlo intervals
"""

import asyncio
from isl_client import ISLClient


async def main():
    """Run conformal prediction examples."""
    async with ISLClient(base_url="http://localhost:8000") as client:
        # Define a simple structural causal model
        model = {
            "equations": {
                "Revenue": {
                    "formula": "2.5 * Price + 1.2 * Quality + noise",
                    "noise_dist": "normal(0, 50)",
                }
            },
            "variables": ["Price", "Quality", "Revenue"],
        }

        # Historical calibration data
        calibration_data = [
            {"Price": 40, "Quality": 7, "Revenue": 1184},
            {"Price": 42, "Quality": 8, "Revenue": 1289},
            {"Price": 38, "Quality": 6, "Revenue": 1102},
            {"Price": 45, "Quality": 9, "Revenue": 1423},
            {"Price": 41, "Quality": 7, "Revenue": 1211},
            {"Price": 43, "Quality": 8, "Revenue": 1317},
            {"Price": 39, "Quality": 6, "Revenue": 1124},
            {"Price": 44, "Quality": 9, "Revenue": 1391},
            {"Price": 40, "Quality": 7, "Revenue": 1201},
            {"Price": 42, "Quality": 8, "Revenue": 1278},
            {"Price": 41, "Quality": 7, "Revenue": 1195},
            {"Price": 43, "Quality": 8, "Revenue": 1302},
        ]

        # Example 1: Standard counterfactual
        print("=" * 60)
        print("Example 1: Standard Counterfactual Prediction")
        print("=" * 60)

        result1 = await client.causal.counterfactual(
            model=model, intervention={"Price": 45.0}, seed=42
        )

        print(f"Intervention: Price = ${result1.intervention['Price']:.2f}")
        print(f"Predicted Revenue: ${result1.prediction.prediction['Revenue']:.2f}")

        if result1.prediction.uncertainty:
            print(
                f"Uncertainty: ±${result1.prediction.uncertainty.get('Revenue', 0):.2f}"
            )

        print(f"\n{result1.explanation.summary}")

        # Example 2: Conformal prediction
        print("\n" + "=" * 60)
        print("Example 2: Conformal Prediction (95% Coverage)")
        print("=" * 60)

        result2 = await client.causal.counterfactual_conformal(
            model=model,
            intervention={"Price": 45.0},
            calibration_data=calibration_data,
            confidence=0.95,
            seed=42,
        )

        interval = result2.conformal_interval
        coverage = result2.coverage_guarantee

        print(f"Point estimate: ${interval.point_estimate:.2f}")
        print(f"95% Conformal interval: [${interval.lower:.2f}, ${interval.upper:.2f}]")
        print(f"Interval width: ${interval.width:.2f}")
        print(f"\nCoverage guaranteed: {coverage.guaranteed}")
        print(f"Theoretical coverage: {coverage.theoretical_coverage:.2%}")
        print(f"Finite-sample valid: {coverage.finite_sample_valid}")

        # Example 3: Comparison with Monte Carlo
        print("\n" + "=" * 60)
        print("Example 3: Conformal vs Monte Carlo Comparison")
        print("=" * 60)

        comparison = result2.comparison_to_monte_carlo

        print(f"Monte Carlo interval width: ${comparison.monte_carlo_width:.2f}")
        print(f"Conformal interval width: ${comparison.conformal_width:.2f}")
        print(f"Width ratio (conformal/MC): {comparison.width_ratio:.2f}")
        print(f"Relative efficiency: {comparison.relative_efficiency:.2f}x")

        if comparison.conformal_width < comparison.monte_carlo_width:
            print("\n✅ Conformal intervals are tighter (more efficient)!")
        else:
            print("\n⚠️  Monte Carlo intervals are tighter in this case")

        # Example 4: Different confidence levels
        print("\n" + "=" * 60)
        print("Example 4: Different Confidence Levels")
        print("=" * 60)

        for confidence_level in [0.90, 0.95, 0.99]:
            result = await client.causal.counterfactual_conformal(
                model=model,
                intervention={"Price": 45.0},
                calibration_data=calibration_data,
                confidence=confidence_level,
                seed=42,
            )

            interval = result.conformal_interval
            print(
                f"{confidence_level:.0%} interval: [${interval.lower:.2f}, ${interval.upper:.2f}] "
                f"(width: ${interval.width:.2f})"
            )


if __name__ == "__main__":
    asyncio.run(main())
