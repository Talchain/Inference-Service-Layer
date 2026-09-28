"""ONE influence algorithm — AIQ meaning ruling (#72 5875853496): influence is the EXPECTED NET effect.

    raw(factor) = | Σ_paths ∏_links (mean × exists_probability) |

- NET, not gross reach: signed path products are summed, THEN the magnitude is taken, so offsetting
  channels cancel. Every other readout the engine gives a user (option effects, sensitivity) is net.
- DISCOUNTED by existence: the Monte Carlo samples each link from Bernoulli(exists_probability), so for a
  linear path of distinct links E[∏ sᵢ·1{eᵢ}] = ∏ sᵢ·pᵢ exactly.
- Unchanged: identity partials at the centre (R3-5), max-normalisation, exact-or-null on truncation.

The graphs below are built so net and gross (and discounted and undiscounted) give DIFFERENT answers; each
row names the mutant it separates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from src.models.robustness_v2 import GraphV2, RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

Edge = Tuple[str, str, float, float]  # (from, to, mean, exists_probability)


def graph(edges: List[Edge], factors: List[str]) -> GraphV2:
    nodes: List[Dict[str, Any]] = [{"id": f, "kind": "factor", "label": f} for f in factors]
    known = set(factors) | {"goal"}
    for a, b, _, _ in edges:
        for n in (a, b):
            if n not in known:
                nodes.append({"id": n, "kind": "outcome", "label": n})
                known.add(n)
    nodes.append({"id": "goal", "kind": "outcome", "label": "Goal"})
    req = RobustnessRequestV2(
        graph={
            "nodes": nodes,
            "edges": [
                {"from": a, "to": b, "exists_probability": p, "strength": {"mean": m, "std": 0.1}}
                for a, b, m, p in edges
            ],
        },
        options=[{"id": "o1", "label": "O1", "interventions": {factors[0]: 0.5}}],
        goal_node_id="goal",
        n_samples=100,
        seed=7,
        analysis_types=["comparison", "robustness"],
    )
    return req.graph


def influence(edges: List[Edge], factors: List[str]) -> Dict[str, float]:
    scores, truncated = RobustnessAnalyzerV2()._compute_structural_influence(graph(edges, factors), factors, "goal")
    assert truncated == []
    return scores


class TestNetNotGross:
    def test_a_two_exactly_offsetting_paths_give_zero_influence(self):
        """(a) F: +0.5·+0.5 via A and +0.5·−0.5 via B → net 0. Gross would give F 0.5 → the top bar (1.0)."""
        s = influence(
            [("f", "a", 0.5, 1.0), ("a", "goal", 0.5, 1.0), ("f", "b", 0.5, 1.0), ("b", "goal", -0.5, 1.0),
             ("g", "goal", 0.1, 1.0)],
            ["f", "g"],
        )
        assert s["f"] == pytest.approx(0.0, abs=1e-12)
        assert s["g"] == 1.0

    def test_a2_a_partial_offset_scores_the_net_remainder(self):
        """F: +0.25 and −0.10 → net 0.15; H direct 0.3 → F = 0.5. Gross: F 0.35 → 1.0, H 0.857."""
        s = influence(
            [("f", "a", 0.5, 1.0), ("a", "goal", 0.5, 1.0), ("f", "b", 0.5, 1.0), ("b", "goal", -0.2, 1.0),
             ("h", "goal", 0.3, 1.0)],
            ["f", "h"],
        )
        assert s["f"] == pytest.approx(0.5, abs=1e-12)
        assert s["h"] == 1.0

    def test_the_census_eng_hiring_shape_is_net_and_discounted(self):
        """Census 5875292873, eng-hiring 183807Z-E: senior → delivery → goal (+0.5, +0.5), senior → delay →
        delivery → goal (+0.5, −0.5, +0.5), senior → spend → headroom → goal (+0.5, −0.5, +0.5), every link
        p = 0.8. Expected net: 0.25·0.64 − 0.125·0.512 − 0.125·0.512 = 0.032; spend: −0.25·0.64 → |0.16|.
        So spend LEADS (1.0) and senior is 0.2 — gross had senior 1.0 (0.288), spend 0.5556."""
        p = 0.8
        s = influence(
            [("senior", "delivery", 0.5, p), ("delivery", "goal", 0.5, p),
             ("senior", "delay", 0.5, p), ("delay", "delivery", -0.5, p),
             ("senior", "spend", 0.5, p), ("spend", "headroom", -0.5, p), ("headroom", "goal", 0.5, p)],
            ["senior", "spend"],
        )
        assert s["spend"] == 1.0
        assert s["senior"] == pytest.approx(0.032 / 0.16, abs=1e-12)


class TestDiscountedByExistence:
    def test_b_a_link_believed_at_half_gives_half_the_influence(self):
        """(b) F → goal (0.4, p 0.5) vs G → goal (0.4, p 1.0): F = 0.5. Dropping p gives F 1.0."""
        s = influence([("f", "goal", 0.4, 0.5), ("g", "goal", 0.4, 1.0)], ["f", "g"])
        assert s["f"] == pytest.approx(0.5, abs=1e-12)
        assert s["g"] == 1.0

    def test_b2_two_half_believed_links_in_series_give_a_quarter(self):
        s = influence(
            [("f", "m", 1.0, 0.5), ("m", "goal", 0.4, 0.5), ("g", "goal", 0.4, 1.0)],
            ["f", "g"],
        )
        assert s["f"] == pytest.approx(0.25, abs=1e-12)
        assert s["g"] == 1.0


class TestUnchanged:
    def test_a_single_positive_path_is_unchanged_by_net(self):
        """Contrast: with one path net == gross, so a chain's score does not move."""
        s = influence([("f", "m", 0.5, 1.0), ("m", "goal", 0.5, 1.0), ("g", "goal", 0.5, 1.0)], ["f", "g"])
        assert s["f"] == pytest.approx(0.5, abs=1e-12)
        assert s["g"] == 1.0

    def test_a_negative_single_path_scores_its_magnitude(self):
        s = influence([("f", "goal", -0.3, 1.0), ("g", "goal", 0.6, 1.0)], ["f", "g"])
        assert s["f"] == pytest.approx(0.5, abs=1e-12)
