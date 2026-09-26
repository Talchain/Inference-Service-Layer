"""A limit "met" on levels the quantity cannot take is reported, not hidden (level_domain).

THE FINDING (#70 5844762506, LOCAL ISL on Paul's served churn shape). A drafted AI -> churn link of -0.5,
applied on churn's 0-100% scale, moves churn from 4% to about -46%. "Churn at most 10%" then reads as met
with P ~ 0.99, while ~82% of the draws put churn below 0%. The verdict is probably right; its confidence
rests on levels that cannot exist. ISL cannot know a quantity's physical range (the caller owns units), so
the caller states it (``GoalConstraint.level_domain``) and ISL reports, per option, the share of draws
whose LEVEL falls outside it. Report-only: no probability moves.

THE WITNESS GRAPH, every level by hand::

    a  root lever, default 0 (no observed level)      a -> c (-0.5),  c -> g (-0.5)
    c  NON-root rate, today's level (baseline) 0.04
    limit: c <= 0.10, level frame, domain [0, 1]

    hold  {}          level = 0.04 + (0 - 0)            =  0.04   inside
    ai    {a: 1.0}    level = 0.04 + (-0.5*1 - 0)       = -0.46   OUTSIDE, yet 'met' (-0.46 <= 0.10)
"""

from typing import Dict, Optional

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.models.robustness_v2 import (
    EdgeV2,
    GoalConstraint,
    GraphV2,
    InterventionOption,
    LevelDomain,
    NodeV2,
    ObservedState,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

EXACT = 1e-12
LIMIT = "lim-churn"
UNIT = LevelDomain(min=0.0, max=1.0)


def edge(src: str, dst: str, mean: float) -> EdgeV2:
    return EdgeV2(**{"from": src, "to": dst}, exists_probability=1.0, strength=StrengthDistribution(mean=mean, std=0.0011))


def analyse(
    options: Optional[Dict[str, Dict[str, float]]] = None,
    domain: Optional[LevelDomain] = UNIT,
    frame: str = "level",
    threshold: float = 0.10,
):
    return RobustnessAnalyzerV2().analyze(
        RobustnessRequestV2(
            request_id="level-domain",
            graph=GraphV2(
                nodes=[
                    NodeV2(id="a", kind="factor", label="AI availability"),
                    NodeV2(id="c", kind="factor", label="Churn", observed_state=ObservedState(value=0.04, baseline=0.04)),
                    NodeV2(id="g", kind="outcome", label="Goal"),
                ],
                edges=[edge("a", "c", -0.5), edge("c", "g", -0.5)],
            ),
            options=[
                InterventionOption(id=oid, label=oid, interventions=iv)
                for oid, iv in (options if options is not None else {"hold": {}, "ai": {"a": 1.0}}).items()
            ],
            goal_node_id="g",
            n_samples=500,
            seed=11,
            goal_constraints=[
                GoalConstraint(
                    constraint_id=LIMIT, node_id="c", operator="<=", value=threshold, value_frame=frame, level_domain=domain
                )
            ],
        )
    )


def row(response, option_id: str):
    results = [r for r in response.results if r.option_id == option_id]
    assert len(results) == 1
    assert results[0].constraint_analysis is not None, f"{option_id}: constraint_analysis omitted"
    rows = [c for c in results[0].constraint_analysis.constraints if c.constraint_id == LIMIT]
    assert len(rows) == 1
    return rows[0]


class TestAMetLimitOnImpossibleLevelsIsReported:
    def test_the_lever_option_meets_the_limit_on_levels_below_zero_on_every_draw(self):
        response = analyse()

        assert row(response, "ai").prob_satisfied == pytest.approx(1.0, abs=EXACT)
        assert row(response, "ai").level_out_of_domain_fraction == pytest.approx(1.0, abs=EXACT)

    def test_the_status_quo_sits_inside_the_domain(self):
        assert row(analyse(), "hold").level_out_of_domain_fraction == pytest.approx(0.0, abs=EXACT)

    def test_it_never_moves_a_probability(self):
        with_domain, without = analyse(), analyse(domain=None)
        for oid in ("hold", "ai"):
            assert row(with_domain, oid).prob_satisfied == row(without, oid).prob_satisfied

    def test_a_bound_is_inclusive(self):
        """hold's level is EXACTLY 0.04 on every draw (0.04 + (s - s)): a domain starting at 0.04 holds it."""
        assert row(analyse(domain=LevelDomain(min=0.04)), "hold").level_out_of_domain_fraction == pytest.approx(0.0, abs=EXACT)

    def test_the_upper_bound_counts(self):
        """The mirror: with the domain ending at 0.03, hold's 0.04 is outside on every draw."""
        assert row(analyse(domain=LevelDomain(max=0.03)), "hold").level_out_of_domain_fraction == pytest.approx(1.0, abs=EXACT)

    def test_judged_on_the_level_not_the_models_raw_samples(self):
        """hold's raw sample of c is 0.0 (a non-root's change-from-origin); its LEVEL is 0.04. With the domain
        starting at 0.02, only the level reading puts hold inside."""
        response = analyse(domain=LevelDomain(min=0.02, max=1.0))
        assert row(response, "hold").level_out_of_domain_fraction == pytest.approx(0.0, abs=EXACT)

    def test_one_bound_is_enough(self):
        response = analyse(domain=LevelDomain(min=0.0))
        assert row(response, "ai").level_out_of_domain_fraction == pytest.approx(1.0, abs=EXACT)


class TestAbsentUnlessStatedAndMeaningful:
    def test_no_domain_no_field(self):
        """CONTROL: the byte-identity path. With no domain the field is None, so absent on the wire."""
        response = analyse(domain=None)
        for oid in ("hold", "ai"):
            assert row(response, oid).level_out_of_domain_fraction is None

    def test_a_delta_limit_has_no_level_to_judge(self):
        """A 'delta' series is a CHANGE, which has no physical range: no fraction, even with a domain."""
        response = analyse(frame="delta", threshold=0.0)
        for oid in ("hold", "ai"):
            assert row(response, oid).level_out_of_domain_fraction is None


class TestTheDomainIsValidated:
    @pytest.mark.parametrize(
        "bounds",
        [{}, {"min": 1.0, "max": 0.0}, {"min": float("nan")}, {"max": float("inf")}],
        ids=["no-bound", "min-above-max", "nan", "inf"],
    )
    def test_refused(self, bounds):
        with pytest.raises(ValidationError):
            LevelDomain(**bounds)


class TestOnTheWire:
    """The V2 envelope is what PLoT reads: the field must survive every hop, and be ABSENT without a domain."""

    @staticmethod
    def post(domain):
        body = {
            "graph": {
                "nodes": [
                    {"id": "a", "kind": "factor", "label": "AI availability"},
                    {"id": "c", "kind": "factor", "label": "Churn", "observed_state": {"value": 0.04, "baseline": 0.04}},
                    {"id": "g", "kind": "outcome", "label": "Goal"},
                ],
                "edges": [
                    {"from": "a", "to": "c", "exists_probability": 1.0, "strength": {"mean": -0.5, "std": 0.0011}},
                    {"from": "c", "to": "g", "exists_probability": 1.0, "strength": {"mean": -0.5, "std": 0.0011}},
                ],
            },
            "options": [
                {"id": "hold", "label": "Hold", "interventions": {"a": 0.0}},
                {"id": "ai", "label": "AI", "interventions": {"a": 1.0}},
            ],
            "goal_node_id": "g",
            "n_samples": 300,
            "seed": 11,
            "goal_constraints": [
                {"constraint_id": LIMIT, "node_id": "c", "operator": "<=", "value": 0.10, "value_frame": "level",
                 **({} if domain is None else {"level_domain": domain})}
            ],
        }
        response = TestClient(app).post("/api/v1/robustness/analyze/v2?response_version=2", json=body)
        assert response.status_code == 200, response.text
        return response.json()

    @staticmethod
    def rows(body):
        found = {}

        def walk(o):
            if isinstance(o, dict):
                if o.get("constraint_id") == LIMIT and "prob_satisfied" in o:
                    found.setdefault("rows", []).append(o)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(body)
        assert found.get("rows"), "no constraint rows on the wire — the probe is blind"
        return found["rows"]

    def test_present_with_a_domain(self):
        fractions = sorted(r.get("level_out_of_domain_fraction") for r in self.rows(self.post({"min": 0, "max": 1})))
        assert fractions == [0.0, 1.0]

    def test_absent_without_one(self):
        assert all("level_out_of_domain_fraction" not in r for r in self.rows(self.post(None)))
