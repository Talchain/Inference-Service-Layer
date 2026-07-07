"""Graph builders for the science-validation harness.

Three families:

1. Margin family (`margin_cases`) — single decisive edge whose flip probability
   under the marginal-switch estimator is analytically known (truncated-normal
   CDF at zero), spanning comfortable-margin -> knife-edge decisions, plus a
   structurally impossible flip (TRUE ZERO by construction).
2. Repo fixtures (`fixture_requests`) — the pinned-seed variant graphs from
   tests/benchmarks/sample_variants.json.
3. Random DAGs (`random_graph_payloads`) — deterministic diverse graphs for the
   determinism-at-scale experiment.

All payloads are wire-shaped dicts (edges keyed "from"/"to").
"""

from __future__ import annotations

import json
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy import stats  # noqa: E402

from src.utils.rng import SeededRNG  # noqa: E402

SAMPLE_VARIANTS = REPO_ROOT / "tests" / "benchmarks" / "sample_variants.json"

EDGE_LO, EDGE_HI = -1.0, 1.0


def truncnorm_cdf_at(x: float, mean: float, std: float) -> float:
    """CDF at x of Normal(mean, std) truncated to the schema bounds [-1, 1].

    This is exactly the distribution produced by SeededRNG.truncated_normal
    (rejection sampling within bounds), up to the negligible clip fallback
    (probability (1 - mass)^100 of 100 consecutive rejections).
    """
    a = (EDGE_LO - mean) / std
    b = (EDGE_HI - mean) / std
    return float(stats.truncnorm.cdf(x, a, b, loc=mean, scale=std))


@dataclass
class MarginCase:
    """A single-decisive-edge graph with analytically known flip probability."""

    name: str
    payload: Dict[str, Any]
    decisive_edge: Tuple[str, str]  # analytic flip probability applies to this edge
    structural_zero_edge: Optional[Tuple[str, str]]  # flip impossible by construction
    analytic_flip_probability: float
    regime: str  # comfortable | under_resolution | knife_edge | true_zero


def _margin_payload(mean: float, std: float) -> Dict[str, Any]:
    """Graph: upstream -> lever -> goal, where every option intervenes on `lever`.

    - Edge lever->goal is decisive: goal = lever_value * strength, so with the
      baseline strength mean > 0 the higher intervention wins; the winner flips
      exactly when the sampled strength is negative.
      P(flip) = P(TruncNormal(mean, std, [-1,1]) < 0) (exists_probability = 1).
      A zero sampled strength ties both options; the deterministic tie-break
      (option id sort) then picks opt_hi, the baseline winner — not a flip.
    - Edge upstream->lever can never flip anything: `lever` is intervened on by
      every option, and interventions override structural equations, so this
      edge is a structural TRUE ZERO.
    """
    return {
        "graph": {
            "nodes": [
                {"id": "upstream", "kind": "factor", "label": "Upstream driver"},
                {"id": "lever", "kind": "factor", "label": "Decision lever"},
                {"id": "goal", "kind": "outcome", "label": "Goal"},
            ],
            "edges": [
                {
                    "from": "upstream",
                    "to": "lever",
                    "exists_probability": 0.9,
                    "strength": {"mean": 0.5, "std": 0.2},
                },
                {
                    "from": "lever",
                    "to": "goal",
                    "exists_probability": 1.0,
                    "strength": {"mean": mean, "std": std},
                },
            ],
        },
        "options": [
            {"id": "opt_hi", "label": "High lever", "interventions": {"lever": 1.0}},
            {"id": "opt_lo", "label": "Low lever", "interventions": {"lever": 0.5}},
        ],
        "goal_node_id": "goal",
        "n_samples": 300,
    }


def margin_cases() -> List[MarginCase]:
    """Margin family spanning knife-edge -> comfortable -> effectively-zero.

    std is fixed at 0.15; the strength mean is chosen as z * std for z values
    that place the analytic flip probability across the K-resolution regimes:

    - p ~ 0.4, 0.05, 0.02: resolvable at K=100
    - p ~ 0.01: exactly at the K=100 resolution floor
    - p ~ 0.005 .. 0.0001: the under-resolution band (zeros at K=100)
    - p ~ 3e-6: below even K=100000 single-seed resolution
    - structural zero: flip impossible at any K
    """
    std = 0.15
    z_targets = {
        "knife_edge_p0.4": 0.2533,
        "moderate_p0.05": 1.6449,
        "moderate_p0.02": 2.0537,
        "floor_p0.01": 2.3263,
        "under_res_p0.005": 2.5758,
        "under_res_p0.002": 2.8782,
        "under_res_p0.001": 3.0902,
        "under_res_p0.0003": 3.4316,
        "under_res_p0.0001": 3.7190,
        "near_zero_p3e-6": 4.5,
    }
    cases: List[MarginCase] = []
    for name, z in z_targets.items():
        mean = round(z * std, 6)
        p = truncnorm_cdf_at(0.0, mean, std)
        if p >= 0.02:
            regime = "knife_edge" if p > 0.1 else "comfortable"
        elif p >= 0.01:
            regime = "floor"
        elif p > 1e-5:
            regime = "under_resolution"
        else:
            regime = "near_zero"
        cases.append(
            MarginCase(
                name=name,
                payload=_margin_payload(mean, std),
                decisive_edge=("lever", "goal"),
                structural_zero_edge=("upstream", "lever"),
                analytic_flip_probability=p,
                regime=regime,
            )
        )
    return cases


def fixture_requests() -> List[Dict[str, Any]]:
    """Wire-shaped payloads for the three pinned repo fixture graphs."""
    data = json.loads(SAMPLE_VARIANTS.read_text())
    payloads = []
    for variant in data["graphs"]:
        payloads.append(
            {
                "graph": variant["graph"] if "graph" in variant else variant,
                "options": data["options"],
                "goal_node_id": data["goal_node_id"],
                "n_samples": data.get("n_samples", 200),
                "seed": data.get("seed", 42),
            }
        )
    return payloads


# ---------------------------------------------------------------------------
# Random diverse DAGs for determinism-at-scale
# ---------------------------------------------------------------------------
_KINDS_INTERNAL = ["factor", "chance", "factor", "outcome"]


def random_graph_payloads(n_graphs: int, gen_seed: int, request_seed: int) -> List[Dict[str, Any]]:
    """Deterministically generate diverse valid v2 request payloads.

    Diversity dimensions: node count (2-30), edge density, node kinds, factor
    uncertainties, goal thresholds, goal constraints, option count, analysis
    flags, n_samples. All draws come from a single SeededRNG(gen_seed) so the
    graph set is fully pinned.
    """
    rng = SeededRNG(gen_seed)
    payloads: List[Dict[str, Any]] = []
    for gi in range(n_graphs):
        n_nodes = 2 + rng.integers(0, 29)  # 2..30
        node_ids = [f"n{j}" for j in range(n_nodes)]
        goal_id = node_ids[-1]

        nodes = []
        for j, nid in enumerate(node_ids):
            if nid == goal_id:
                kind = "outcome" if rng.random() < 0.6 else "factor"
            elif j == 0:
                kind = "factor"
            else:
                kind = _KINDS_INTERNAL[rng.integers(0, len(_KINDS_INTERNAL))]
            node: Dict[str, Any] = {"id": nid, "kind": kind, "label": f"Node {j}"}
            if j < n_nodes - 1 and rng.random() < 0.5:
                node["observed_state"] = {"value": round(rng.uniform(0.1, 0.9), 3)}
            nodes.append(node)

        density = 0.15 + rng.random() * 0.35
        max_edges = 190  # schema caps edges at 200; leave headroom for the goal edge
        edges = []
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                if len(edges) >= max_edges:
                    break
                if rng.random() < density:
                    edges.append(
                        {
                            "from": node_ids[i],
                            "to": node_ids[j],
                            "exists_probability": round(rng.uniform(0.5, 1.0), 3),
                            "strength": {
                                "mean": round(rng.uniform(-0.9, 0.9), 3),
                                "std": round(rng.uniform(0.05, 0.3), 3),
                            },
                        }
                    )
        # Guarantee the goal has at least one incoming edge
        if not any(e["to"] == goal_id for e in edges):
            src = node_ids[rng.integers(0, n_nodes - 1)]
            edges.append(
                {
                    "from": src,
                    "to": goal_id,
                    "exists_probability": 0.9,
                    "strength": {"mean": 0.5, "std": 0.15},
                }
            )

        # Ancestors of the goal — every option must intervene on at least one,
        # or the endpoint blocks the request with NO_EFFECTIVE_PATH_TO_GOAL.
        parents: Dict[str, List[str]] = {}
        for e in edges:
            parents.setdefault(e["to"], []).append(e["from"])
        ancestors: set = set()
        frontier = [goal_id]
        while frontier:
            node_id = frontier.pop()
            for p in parents.get(node_id, []):
                if p not in ancestors:
                    ancestors.add(p)
                    frontier.append(p)
        ancestor_targets = sorted(ancestors)

        n_options = 2 + rng.integers(0, 3)  # 2..4
        candidate_targets = node_ids[:-1]
        options = []
        for oi in range(n_options):
            n_iv = 1 + rng.integers(0, min(3, len(candidate_targets)))
            targets = {
                candidate_targets[rng.integers(0, len(candidate_targets))] for _ in range(n_iv)
            }
            targets.add(ancestor_targets[rng.integers(0, len(ancestor_targets))])
            options.append(
                {
                    "id": f"opt{oi}",
                    "label": f"Option {oi}",
                    "interventions": {t: round(rng.uniform(0.0, 1.0), 3) for t in sorted(targets)},
                }
            )

        payload: Dict[str, Any] = {
            "graph": {"nodes": nodes, "edges": edges},
            "options": options,
            "goal_node_id": goal_id,
            "n_samples": [100, 300, 500][rng.integers(0, 3)],
            "seed": request_seed,
            "request_id": f"scival-det-{gi:03d}",
        }
        # Factor uncertainties on ~30% of graphs (nodes that carry observed_state)
        observed = [n["id"] for n in nodes if "observed_state" in n]
        if observed and rng.random() < 0.3:
            payload["parameter_uncertainties"] = [
                {
                    "node_id": observed[rng.integers(0, len(observed))],
                    "distribution": "normal",
                    "std": round(rng.uniform(0.05, 0.2), 3),
                }
            ]
            payload["include_voi"] = bool(rng.random() < 0.5)
        if rng.random() < 0.4:
            payload["goal_threshold"] = round(rng.uniform(0.1, 0.6), 3)
        if rng.random() < 0.2:
            payload["goal_constraints"] = [
                {
                    "node_id": goal_id,
                    "operator": ">=" if rng.random() < 0.7 else "<=",
                    "value": round(rng.uniform(0.1, 0.5), 3),
                }
            ]
        if rng.random() < 0.3:
            payload["include_e_values"] = True
        if rng.random() < 0.3:
            payload["include_path_decomposition"] = True
        payloads.append(payload)
    return payloads


# ---------------------------------------------------------------------------
# EVPI floor-validation graphs (exp2)
# ---------------------------------------------------------------------------
def evpi_payload(knife_edge: bool) -> Dict[str, Any]:
    """Graph with four uncertain factors for EVPI floor validation.

    In ISL's linear SCM under the p_win_recommended metric, a factor's true
    EVPI is non-zero ONLY if the options intervene asymmetrically on its
    causal path: an intervention overrides the structural equation, severing
    the factor's influence for that option while other options still ride it.
    A factor that influences every option's outcome identically is common-mode
    — its contribution cancels in the winner comparison, so its true EVPI is
    exactly zero however strongly it drives the goal.

    Construction: opt_fix intervenes on `mid` (fixed outcome, insensitive to
    driver/weak); opt_ride does not (rides driver and weak through mid).

    - `driver`   : strong path driver->mid->goal, asymmetric => material EVPI
    - `weak`     : faint path weak->mid->goal, asymmetric => small true EVPI
    - `common`   : common->goal directly; affects both options identically
                   => true EVPI exactly 0 despite real goal influence
    - `stranded` : no causal path to goal => true EVPI exactly 0

    knife_edge=True sets opt_fix's intervention at the decision boundary
    (P(win) ~ 0.5, the worst case p = 0.5 the floor formula assumes);
    False gives a comfortable winner.
    """
    fix_value = 0.45 if knife_edge else 0.62
    return {
        "graph": {
            "nodes": [
                {
                    "id": "driver",
                    "kind": "factor",
                    "label": "Driver",
                    "observed_state": {"value": 0.5},
                },
                {
                    "id": "weak",
                    "kind": "factor",
                    "label": "Weak driver",
                    "observed_state": {"value": 0.5},
                },
                {
                    "id": "common",
                    "kind": "factor",
                    "label": "Common-mode factor",
                    "observed_state": {"value": 0.5},
                },
                {
                    "id": "stranded",
                    "kind": "factor",
                    "label": "Stranded factor",
                    "observed_state": {"value": 0.5},
                },
                {"id": "sink", "kind": "chance", "label": "Sink (absorbs stranded)"},
                {"id": "mid", "kind": "chance", "label": "Mediator (opt_fix intervenes here)"},
                {"id": "goal", "kind": "outcome", "label": "Goal"},
            ],
            "edges": [
                {
                    "from": "driver",
                    "to": "mid",
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.8, "std": 0.1},
                },
                {
                    "from": "weak",
                    "to": "mid",
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.1, "std": 0.05},
                },
                {
                    "from": "mid",
                    "to": "goal",
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.8, "std": 0.05},
                },
                {
                    "from": "common",
                    "to": "goal",
                    "exists_probability": 0.9,
                    "strength": {"mean": 0.6, "std": 0.2},
                },
                {
                    "from": "stranded",
                    "to": "sink",
                    "exists_probability": 0.9,
                    "strength": {"mean": 0.7, "std": 0.2},
                },
            ],
        },
        "options": [
            {"id": "opt_fix", "label": "Fix mediator", "interventions": {"mid": fix_value}},
            {"id": "opt_ride", "label": "Ride the driver", "interventions": {}},
        ],
        "goal_node_id": "goal",
        "n_samples": 1000,
        "include_voi": True,
        "parameter_uncertainties": [
            {"node_id": "driver", "distribution": "normal", "std": 0.2},
            {"node_id": "weak", "distribution": "normal", "std": 0.2},
            {"node_id": "common", "distribution": "normal", "std": 0.2},
            {"node_id": "stranded", "distribution": "normal", "std": 0.2},
        ],
    }


# ---------------------------------------------------------------------------
# Analytic calibration cases (exp4)
# ---------------------------------------------------------------------------
@dataclass
class CalibrationCase:
    """Single-edge graph whose P(goal >= threshold) is known in closed form."""

    name: str
    payload: Dict[str, Any]
    option_truths: Dict[str, float]  # option_id -> analytic probability_of_goal
    goal_kind: str  # "chance" (no auto-noise) or "outcome" (auto-noise applies)


def analytic_probability(
    x: float, mean: float, std: float, exists_p: float, threshold: float
) -> float:
    """Closed-form P(x * S >= threshold) for the single-edge SCM.

    S = B * T where B ~ Bernoulli(exists_p) gates existence and
    T ~ TruncNormal(mean, std, [-1, 1]). goal = x * S (x > 0).
    """
    p_exists = exists_p * (1.0 - truncnorm_cdf_at(threshold / x, mean, std))
    p_absent = (1.0 - exists_p) * (1.0 if 0.0 >= threshold else 0.0)
    return p_exists + p_absent


def calibration_cases(goal_kind: str) -> List[CalibrationCase]:
    """Grid of single-edge cases over (mean, std, exists_probability, threshold)."""
    cases = []
    grid = [
        # (mean, std, exists_p, threshold)
        (0.5, 0.15, 1.0, 0.3),
        (0.5, 0.15, 1.0, 0.5),
        (0.5, 0.3, 1.0, 0.5),
        (0.3, 0.2, 1.0, 0.2),
        (0.5, 0.15, 0.8, 0.3),
        (0.5, 0.3, 0.7, 0.4),
        (-0.2, 0.2, 0.9, 0.1),
        (0.7, 0.1, 0.9, 0.6),
        (0.2, 0.25, 0.6, 0.15),
    ]
    for mean, std, exists_p, threshold in grid:
        x_hi, x_lo = 1.0, 0.5
        payload = {
            "graph": {
                "nodes": [
                    {"id": "lever", "kind": "factor", "label": "Lever"},
                    {"id": "goal", "kind": goal_kind, "label": "Goal"},
                ],
                "edges": [
                    {
                        "from": "lever",
                        "to": "goal",
                        "exists_probability": exists_p,
                        "strength": {"mean": mean, "std": std},
                    }
                ],
            },
            "options": [
                {"id": "opt_hi", "label": "High", "interventions": {"lever": x_hi}},
                {"id": "opt_lo", "label": "Low", "interventions": {"lever": x_lo}},
            ],
            "goal_node_id": "goal",
            "goal_threshold": threshold,
            "n_samples": 10000,
        }
        truths = {
            "opt_hi": analytic_probability(x_hi, mean, std, exists_p, threshold),
            "opt_lo": analytic_probability(x_lo, mean, std, exists_p, threshold),
        }
        name = f"m{mean}_s{std}_e{exists_p}_t{threshold}_{goal_kind}"
        cases.append(
            CalibrationCase(name=name, payload=payload, option_truths=truths, goal_kind=goal_kind)
        )
    return cases
