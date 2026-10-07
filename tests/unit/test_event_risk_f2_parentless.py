"""F2: a parentless risk that may happen moves every option's goal probability.

Analytic answers, stated before the rows: p_mid = (0.10 + 0.30) / 2 = 0.20.
- Status quo meets the goal iff the developer stays: P(goal) = 0.80;
  mean level = 0.80 - 0.30 * 0.20 = 0.74.
- Contractors give levels 0.85 or 0.55: P(goal) = 0.80;
  mean level = 0.85 - 0.06 = 0.79.
- The legacy twin is the same graph with event_risk removed. Its risk is an
  inert root: P(goal) = 1.0 for both options. Opt-in preserves legacy behaviour.
- With p_low = p_high = 0.50, status quo P(goal) = 0.50, below the first row.

Monte Carlo tolerance is four standard errors: SE = sqrt(p * (1-p) / 10000).
Mean outcomes are in the level frame, with an absolute tolerance of 0.005.
"""

from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("ISL_AUTH_DISABLED", "true")

from src.models.robustness_v2 import RobustnessRequestV2  # noqa: E402
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2  # noqa: E402

N = 10_000
OPTION_IDS = ("status_quo", "contractors")


def _se(p: float) -> float:
    return math.sqrt(p * (1.0 - p) / N)


def _edge(source: str, target: str, mean: float) -> dict:
    return {
        "from": source,
        "to": target,
        "exists_probability": 1.0,
        "strength": {"mean": mean, "std": 0.002},
    }


def _request_body(*, event: bool = True, p_low: float = 0.10, p_high: float = 0.30) -> dict:
    risk = {"id": "key_dev_leaves", "kind": "risk", "label": "Key developer leaves"}
    if event:
        risk["event_risk"] = {
            "version": 1,
            "occurrence": {"p_low": p_low, "p_high": p_high, "basis": "user"},
            "horizon": {"months": 12},
        }
    return {
        "graph": {
            "nodes": [
                {
                    "id": "delivery_capacity",
                    "kind": "goal",
                    "label": "Delivery capacity",
                    "observed_state": {"value": 0.80, "baseline": 0.80, "source": "user"},
                },
                risk,
                {
                    "id": "contractor_budget",
                    "kind": "factor",
                    "label": "Contractor budget",
                    "observed_state": {"value": 0.0},
                },
            ],
            "edges": [
                _edge("key_dev_leaves", "delivery_capacity", -0.30),
                _edge("contractor_budget", "delivery_capacity", 0.05),
            ],
        },
        "options": [
            {
                "id": "status_quo",
                "label": "Carry on as now",
                "interventions": {"contractor_budget": 0.0},
            },
            {
                "id": "contractors",
                "label": "Bring in contractors",
                "interventions": {"contractor_budget": 1.0},
            },
        ],
        "goal_node_id": "delivery_capacity",
        "n_samples": N,
        "seed": 11,
        "goal_threshold": 0.70,
        "goal_threshold_frame": "level",
        "request_id": "event-risk-f2-key-dev",
    }


def _analyze(body: dict):
    return RobustnessAnalyzerV2().analyze(RobustnessRequestV2(**body))


def _result(response, option_id: str):
    matches = [r for r in response.results if r.option_id == option_id]
    assert len(matches) == 1, f"expected exactly one result for {option_id}"
    return matches[0]


@pytest.fixture(scope="module")
def event_response():
    return _analyze(_request_body())


@pytest.fixture(scope="module")
def legacy_response():
    return _analyze(_request_body(event=False))


def test_f2_status_quo_reads_p_no_occurrence(event_response):
    got = _result(event_response, "status_quo").probability_of_goal
    assert got is not None
    assert abs(got - 0.80) <= 4 * _se(0.80), got


def test_f2_every_option_moves(event_response):
    for option_id in OPTION_IDS:
        got = _result(event_response, option_id).probability_of_goal
        assert got is not None
        assert got < 0.95, (option_id, got)
        assert abs(got - 0.80) <= 4 * _se(0.80), (option_id, got)


def test_f2_impact_moves_the_mean(event_response):
    for option_id, want in (("status_quo", 0.74), ("contractors", 0.79)):
        got = _result(event_response, option_id).outcome_distribution.mean
        assert abs(got - want) <= 0.005, (option_id, got, want)


def test_f2_legacy_twin_is_unchanged(legacy_response):
    for option_id in OPTION_IDS:
        assert _result(legacy_response, option_id).probability_of_goal == 1.0


def test_f2_a_bigger_risk_moves_it_more(event_response):
    bigger_response = _analyze(_request_body(p_low=0.50, p_high=0.50))
    got = _result(bigger_response, "status_quo").probability_of_goal
    original = _result(event_response, "status_quo").probability_of_goal
    assert got is not None and original is not None
    assert abs(got - 0.50) <= 4 * _se(0.50), got
    assert got < original, (got, original)
