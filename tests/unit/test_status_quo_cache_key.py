"""ISL #187 follow-up 1 (MG's APPROVE 5861838085): the identity path's status-quo cache is keyed on ALL three inputs.

``SCMEvaluatorV2._status_quo`` keeps a one-entry cache keyed on the CONTENT of the edge strengths, the base values and
the factor values. The #187 rows only varied EDGES (the flip-threshold bisection), so the mutant "key on the edges
only" survived all 53 of them while changing 59/709 fields of the identity wire (``factor_sensitivity``,
``factor_flip_values``). These rows hold the edges FIXED and vary the other two inputs — as a new dict and mutated in
place — and require a fresh reading each time.
"""
from __future__ import annotations

import pytest

from src.models.robustness_v2 import GraphV2
from src.services.robustness_analyzer_v2 import SCMEvaluatorV2


def evaluator() -> SCMEvaluatorV2:
    return SCMEvaluatorV2(GraphV2.model_validate({
        "nodes": [
            {"id": "a", "kind": "factor", "label": "A"},
            {"id": "b", "kind": "factor", "label": "B"},
            {"id": "g", "kind": "goal", "label": "G"},
        ],
        "edges": [
            {"from": "a", "to": "b", "strength": {"mean": 0.5, "std": 0.0011}, "exists_probability": 1.0},
            {"from": "b", "to": "g", "strength": {"mean": 0.5, "std": 0.0011}, "exists_probability": 1.0},
        ],
    }))


EDGES = {("a", "b"): 0.5, ("b", "g"): 0.5}


class TestTheStatusQuoCacheKeysOnEveryInput:
    def test_control_the_same_content_is_one_reading(self):
        """Non-vacuity: the cache IS used — equal content in new objects returns the same reading object."""
        ev = evaluator()
        first = ev._status_quo(dict(EDGES), {"a": 0.2}, {"a": 0.2})
        again = ev._status_quo(dict(EDGES), {"a": 0.2}, {"a": 0.2})
        assert again is first

    def test_new_factor_values_at_fixed_edges_are_read_fresh(self):
        ev = evaluator()
        low = ev._status_quo(dict(EDGES), None, {"a": 0.2})["g"]
        high = ev._status_quo(dict(EDGES), None, {"a": 0.8})["g"]
        assert high != pytest.approx(low)
        assert high == pytest.approx(evaluator()._status_quo(dict(EDGES), None, {"a": 0.8})["g"])

    def test_factor_values_mutated_in_place_are_read_fresh(self):
        ev = evaluator()
        factors = {"a": 0.2}
        low = ev._status_quo(dict(EDGES), None, factors)["g"]
        factors["a"] = 0.8
        high = ev._status_quo(dict(EDGES), None, factors)["g"]
        assert high != pytest.approx(low)
        assert high == pytest.approx(evaluator()._status_quo(dict(EDGES), None, {"a": 0.8})["g"])

    def test_new_base_values_at_fixed_edges_are_read_fresh(self):
        ev = evaluator()
        low = ev._status_quo(dict(EDGES), {"a": 0.2}, None)["g"]
        high = ev._status_quo(dict(EDGES), {"a": 0.8}, None)["g"]
        assert high != pytest.approx(low)
        assert high == pytest.approx(evaluator()._status_quo(dict(EDGES), {"a": 0.8}, None)["g"])
