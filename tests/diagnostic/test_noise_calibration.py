"""
Diagnostic: auto-noise calibration comparison.

Runs the SaaS pricing graph at different noise multipliers (0x, 0.25x, 0.5x, 1.0x)
and prints a comparison table of outcome statistics. This is informational — it
does not assert correctness but provides data for calibrating the noise heuristic.

Run with: ISL_AUTH_DISABLED=true poetry run pytest tests/diagnostic/test_noise_calibration.py -s
"""

from collections import Counter
from typing import Dict, List

import numpy as np
import pytest

from src.models.robustness_v2 import (
    EdgeV2,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.utils.rng import SeededRNG


def _build_saas_pricing_graph() -> GraphV2:
    """Standard SaaS pricing graph: 15 nodes, 27 edges."""
    nodes = [
        NodeV2(
            id="price_level",
            kind="decision",
            label="Price Level",
            observed_state=ObservedState(value=0.6),
        ),
        NodeV2(
            id="brand_strength",
            kind="factor",
            label="Brand Strength",
            observed_state=ObservedState(value=0.7),
        ),
        NodeV2(
            id="market_size",
            kind="factor",
            label="Market Size",
            observed_state=ObservedState(value=0.5),
        ),
        NodeV2(
            id="competitor_price",
            kind="factor",
            label="Competitor Price",
            observed_state=ObservedState(value=0.5),
        ),
        NodeV2(
            id="feature_quality",
            kind="factor",
            label="Feature Quality",
            observed_state=ObservedState(value=0.65),
        ),
        NodeV2(id="perceived_value", kind="chance", label="Perceived Value"),
        NodeV2(id="price_sensitivity", kind="chance", label="Price Sensitivity"),
        NodeV2(id="switching_cost", kind="chance", label="Switching Cost"),
        NodeV2(id="customer_acquisition", kind="chance", label="Customer Acquisition"),
        NodeV2(id="customer_retention", kind="chance", label="Customer Retention"),
        NodeV2(id="churn_rate", kind="chance", label="Churn Rate"),
        NodeV2(id="arpu", kind="chance", label="ARPU"),
        NodeV2(id="ltv", kind="chance", label="LTV"),
        NodeV2(id="cac", kind="chance", label="CAC"),
        NodeV2(id="revenue", kind="outcome", label="Revenue"),
    ]

    edges = [
        # Price effects
        EdgeV2(
            **{"from": "price_level", "to": "perceived_value"},
            strength=StrengthDistribution(mean=-0.4, std=0.15),
            exists_probability=0.95,
        ),
        EdgeV2(
            **{"from": "price_level", "to": "price_sensitivity"},
            strength=StrengthDistribution(mean=0.5, std=0.12),
            exists_probability=0.9,
        ),
        EdgeV2(
            **{"from": "price_level", "to": "arpu"},
            strength=StrengthDistribution(mean=0.7, std=0.1),
            exists_probability=0.98,
        ),
        EdgeV2(
            **{"from": "price_level", "to": "churn_rate"},
            strength=StrengthDistribution(mean=0.3, std=0.15),
            exists_probability=0.8,
        ),
        # Brand effects
        EdgeV2(
            **{"from": "brand_strength", "to": "perceived_value"},
            strength=StrengthDistribution(mean=0.6, std=0.1),
            exists_probability=0.95,
        ),
        EdgeV2(
            **{"from": "brand_strength", "to": "customer_acquisition"},
            strength=StrengthDistribution(mean=0.4, std=0.12),
            exists_probability=0.85,
        ),
        EdgeV2(
            **{"from": "brand_strength", "to": "switching_cost"},
            strength=StrengthDistribution(mean=0.3, std=0.1),
            exists_probability=0.7,
        ),
        # Market effects
        EdgeV2(
            **{"from": "market_size", "to": "customer_acquisition"},
            strength=StrengthDistribution(mean=0.5, std=0.15),
            exists_probability=0.9,
        ),
        EdgeV2(
            **{"from": "market_size", "to": "cac"},
            strength=StrengthDistribution(mean=-0.2, std=0.1),
            exists_probability=0.75,
        ),
        # Competitor effects
        EdgeV2(
            **{"from": "competitor_price", "to": "price_sensitivity"},
            strength=StrengthDistribution(mean=-0.3, std=0.12),
            exists_probability=0.85,
        ),
        EdgeV2(
            **{"from": "competitor_price", "to": "churn_rate"},
            strength=StrengthDistribution(mean=-0.25, std=0.1),
            exists_probability=0.8,
        ),
        # Feature effects
        EdgeV2(
            **{"from": "feature_quality", "to": "perceived_value"},
            strength=StrengthDistribution(mean=0.5, std=0.1),
            exists_probability=0.95,
        ),
        EdgeV2(
            **{"from": "feature_quality", "to": "switching_cost"},
            strength=StrengthDistribution(mean=0.4, std=0.12),
            exists_probability=0.8,
        ),
        EdgeV2(
            **{"from": "feature_quality", "to": "customer_retention"},
            strength=StrengthDistribution(mean=0.35, std=0.1),
            exists_probability=0.85,
        ),
        # Internal chain
        EdgeV2(
            **{"from": "perceived_value", "to": "customer_acquisition"},
            strength=StrengthDistribution(mean=0.5, std=0.12),
            exists_probability=0.9,
        ),
        EdgeV2(
            **{"from": "perceived_value", "to": "customer_retention"},
            strength=StrengthDistribution(mean=0.45, std=0.1),
            exists_probability=0.9,
        ),
        EdgeV2(
            **{"from": "price_sensitivity", "to": "churn_rate"},
            strength=StrengthDistribution(mean=0.4, std=0.12),
            exists_probability=0.85,
        ),
        EdgeV2(
            **{"from": "price_sensitivity", "to": "customer_acquisition"},
            strength=StrengthDistribution(mean=-0.3, std=0.1),
            exists_probability=0.8,
        ),
        EdgeV2(
            **{"from": "switching_cost", "to": "churn_rate"},
            strength=StrengthDistribution(mean=-0.35, std=0.1),
            exists_probability=0.85,
        ),
        EdgeV2(
            **{"from": "switching_cost", "to": "customer_retention"},
            strength=StrengthDistribution(mean=0.3, std=0.1),
            exists_probability=0.8,
        ),
        # Outcome paths
        EdgeV2(
            **{"from": "customer_acquisition", "to": "revenue"},
            strength=StrengthDistribution(mean=0.4, std=0.1),
            exists_probability=0.95,
        ),
        EdgeV2(
            **{"from": "customer_retention", "to": "revenue"},
            strength=StrengthDistribution(mean=0.5, std=0.1),
            exists_probability=0.95,
        ),
        EdgeV2(
            **{"from": "churn_rate", "to": "revenue"},
            strength=StrengthDistribution(mean=-0.4, std=0.12),
            exists_probability=0.9,
        ),
        EdgeV2(
            **{"from": "arpu", "to": "revenue"},
            strength=StrengthDistribution(mean=0.6, std=0.1),
            exists_probability=0.95,
        ),
        EdgeV2(
            **{"from": "ltv", "to": "revenue"},
            strength=StrengthDistribution(mean=0.3, std=0.1),
            exists_probability=0.85,
        ),
        EdgeV2(
            **{"from": "arpu", "to": "ltv"},
            strength=StrengthDistribution(mean=0.7, std=0.1),
            exists_probability=0.9,
        ),
        EdgeV2(
            **{"from": "customer_retention", "to": "ltv"},
            strength=StrengthDistribution(mean=0.5, std=0.1),
            exists_probability=0.9,
        ),
    ]

    return GraphV2(nodes=nodes, edges=edges)


def _build_options() -> List[InterventionOption]:
    return [
        InterventionOption(
            id="keep_current", label="Keep Current Price", interventions={"price_level": 0.5}
        ),
        InterventionOption(
            id="increase_10pct", label="Increase 10%", interventions={"price_level": 0.6}
        ),
        InterventionOption(
            id="increase_25pct", label="Increase 25%", interventions={"price_level": 0.75}
        ),
    ]


@pytest.mark.slow
def test_noise_calibration_comparison() -> None:
    """Diagnostic: compare outcome distributions at different noise multipliers.

    Prints a formatted table — run with ``pytest -s`` to see output.
    This test always passes; it is purely informational.
    """
    graph = _build_saas_pricing_graph()
    options = _build_options()
    seed = 12345
    n_samples = 2000
    multipliers = [0.0, 0.25, 0.5, 1.0]

    print("\n" + "=" * 90)
    print("AUTO-NOISE CALIBRATION COMPARISON")
    print(f"Graph: SaaS Pricing ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")
    print(f"Seed: {seed}, Samples: {n_samples}")
    print("=" * 90)

    results_table: List[Dict] = []

    for mult in multipliers:
        request = RobustnessRequestV2(
            graph=graph,
            options=options,
            goal_node_id="revenue",
            n_samples=n_samples,
            seed=seed,
        )

        analyzer = RobustnessAnalyzerV2()
        response = analyzer.analyze(request)

        # Re-run MC to get raw outcomes, then apply noise with multiplier
        rng_edge = SeededRNG(seed)
        rng_factor = SeededRNG(seed + 1)

        from src.services.robustness_analyzer_v2 import (
            DualUncertaintySampler,
            FactorSampler,
            SCMEvaluatorV2,
        )

        sampler = DualUncertaintySampler(graph.edges, rng_edge)
        factor_sampler = FactorSampler(graph.nodes, None, rng_factor)
        evaluator = SCMEvaluatorV2(graph)

        option_outcomes: Dict[str, List[float]] = {opt.id: [] for opt in options}

        for _ in range(n_samples):
            edge_config = sampler.sample_edge_configuration()
            factor_values = factor_sampler.sample_factor_values()

            for opt in options:
                val = evaluator.evaluate(
                    edge_strengths=edge_config,
                    interventions=opt.interventions,
                    goal_node="revenue",
                    factor_values=factor_values,
                )
                option_outcomes[opt.id].append(val)

        # Apply noise with the specific multiplier
        rng_noise = SeededRNG(seed + 2)
        noised_outcomes, _ = analyzer._apply_auto_scaled_noise(
            option_outcomes, "revenue", graph.nodes, rng_noise, noise_multiplier=mult
        )

        # Compute stats per option
        for opt in options:
            arr = np.array(noised_outcomes[opt.id])
            results_table.append(
                {
                    "multiplier": mult,
                    "option": opt.label,
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "p10": float(np.percentile(arr, 10)),
                    "p50": float(np.percentile(arr, 50)),
                    "p90": float(np.percentile(arr, 90)),
                }
            )

        # Winner stats
        winners = []
        for i in range(n_samples):
            best_opt = max(options, key=lambda o: noised_outcomes[o.id][i])
            winners.append(best_opt.id)

        win_counts = Counter(winners)
        win_probs = {opt.id: win_counts.get(opt.id, 0) / n_samples for opt in options}

        # Recommendation stability: fraction of samples agreeing with overall winner
        overall_winner = max(win_probs, key=lambda k: win_probs[k])
        stability = win_probs[overall_winner]

        results_table.append(
            {
                "multiplier": mult,
                "option": "--- AGGREGATE ---",
                "mean": 0,
                "std": 0,
                "p10": 0,
                "p50": stability,  # reuse column for stability
                "p90": 0,
            }
        )

        for opt in options:
            results_table[-1][f"win_{opt.id}"] = win_probs[opt.id]

    # Print table
    print(
        f"\n{'Mult':>5} | {'Option':<22} | {'Mean':>8} | {'Std':>8} | {'P10':>8} | {'P50':>8} | {'P90':>8}"
    )
    print("-" * 90)

    for row in results_table:
        if row["option"] == "--- AGGREGATE ---":
            win_str = " | ".join(
                f"{k}={row.get(k, 0):.3f}" for k in sorted(row) if k.startswith("win_")
            )
            print(f"{row['multiplier']:>5.2f} | {'WINNER PROBS':<22} | {win_str}")
            print(f"{'':>5} | {'REC STABILITY':<22} | {row['p50']:.3f}")
            print("-" * 90)
        else:
            print(
                f"{row['multiplier']:>5.2f} | {row['option']:<22} | "
                f"{row['mean']:>8.5f} | {row['std']:>8.5f} | "
                f"{row['p10']:>8.5f} | {row['p50']:>8.5f} | {row['p90']:>8.5f}"
            )

    print("\nNOTE: This is a diagnostic — no assertions. Results inform noise calibration.")
