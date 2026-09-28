"""AIQ #72 5872386356 (a row, low urgency): the marginal switch sweep is capped at MARGINAL_MAX_EDGES.
A definitional link of an evaluated identity (R3-9) is never sampled, so a switch probability priced on
it measures nothing, and if it took a slot a real belief edge would go untested while the truncation
count misstated how much was tested. RULE: definitions leave BEFORE ranking and capping, and the
truncation count is over belief edges only.

It holds on staging since #197: ``_compute_sensitivity`` skips every definitional edge (the ``fixed``
set), and the capped sweep (``_compute_alternative_winners``) ranks only sensitivity rows. These rows
pin that at the point of the cap, bound by edge identity. Mutant: the sensitivity skip removed → a
definitional edge enters the capped set → RED.
"""

from typing import Any, Dict

import pytest

import src.services.robustness_analyzer_v2 as rav2
from src.models.robustness_v2 import RobustnessRequestV2
from tests.unit.test_r3_identity_evaluation import wire


def run_capturing_the_capped_set(monkeypatch: pytest.MonkeyPatch, d: Dict[str, Any], cap: int):
    seen: Dict[str, Any] = {}
    real = rav2.RobustnessAnalyzerV2._compute_alternative_winners

    def spy(self, fragile_edge_info, *args, **kwargs):
        seen["fragile"] = dict(fragile_edge_info)
        seen["ranking"] = dict(kwargs.get("edge_max_elasticity") or {})
        return real(self, fragile_edge_info, *args, **kwargs)

    monkeypatch.setattr(rav2.RobustnessAnalyzerV2, "_compute_alternative_winners", spy)
    monkeypatch.setattr(rav2, "MARGINAL_MAX_EDGES", cap)
    request = RobustnessRequestV2.model_validate(d)
    response = rav2.RobustnessAnalyzerV2().analyze(request)
    return seen, request, response


class TestTheMarginalCapReadsBeliefEdgesOnly:
    def test_no_definitional_link_reaches_the_ranking_the_cap_reads(self, monkeypatch):
        """The cap takes the top-K of ``edge_max_elasticity`` over ``fragile_edge_info``; both come
        from the sensitivity rows. On the served wire the 3 definitional links (price, subscribers,
        other growth -> MRR) must be absent from both, while the 7 belief edges are ranked. The
        truncation count (``len(fragile)``) is then over belief edges by construction."""
        seen, request, _ = run_capturing_the_capped_set(monkeypatch, wire(), cap=1)
        definitional = {f"{a}->{b}" for a, b in rav2.definitional_edges(request.graph)}
        belief = {f"{e.from_}->{e.to}" for e in request.graph.edges} - definitional
        # Fixture controls: the identity IS evaluated, and the ranking DOES see belief edges.
        assert len(definitional) == 3, sorted(definitional)
        assert "ranking" in seen, "fixture control: the capped sweep ran"
        assert set(seen["ranking"]) == belief, (sorted(seen["ranking"]), sorted(belief))
        # The rule, by identity.
        assert not (set(seen["ranking"]) & definitional)
        assert not (set(seen["fragile"]) & definitional)
