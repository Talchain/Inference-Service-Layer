"""The displayed summary is read from the SAME clamped draws as the band (DL ruling #72 5865702681; MG 5865668959).

B1a (AIQ 5855046894 (2)) reports an anchored goal's band as a LEVEL, clamped to the goal's domain — but it clamped only
p10/p50/p90, after ``np.percentile``, and left ``mean``, ``std`` and the downside block (``p05``, ``cvar_10``) on the
unclamped draws. Served journey A (CEE a4f4d2b · PLoT c0f0a9a · ISL 9b8aa34, Runtime's run 3): the £59 option's draws
reached 4.17 on the 0–1 level domain, so its card read "mean £126,363, 90th percentile £125,000" — a distribution no
reader can believe — and ``p05`` could sit below a ``p10`` clamped to 0.

THE RULE: when the band is clamped, every displayed summary is taken from the clamped draws. DISPLAY-ONLY: win shares,
differences, regret, EVPPI and the leader are the analyzer's, on the unclamped draws, and do not move.

THE FIXTURE is the ISL body PLoT c0f0a9a built for that run's first graph through CEE's real ``run_analysis`` path
(status quo held at today's levels, as CEE #2217 sends it).

NOT A THEOREM, SO NOT PINNED: ``p10 <= mean <= p90``. On shared draws a skewed population still breaks it (91 % of draws
at 0 and 9 % at 1: p90 = 0, mean = 0.09). What shared draws DO guarantee is pinned instead: the mean never leaves the
clamped domain, and ``p05 <= p10``.
"""

import json

from pathlib import Path
from typing import Any, Dict, List

import pytest

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "anchored_delta"
    / "journey-a-run3-status-quo-held-c0f0a9a.isl.json"
)
CAP_GBP = 125_000.0
P59 = "59_with_feature_release"


@pytest.fixture(scope="module")
def served() -> Dict[str, Any]:
    from fastapi.testclient import TestClient

    from src.api.main import app

    response = TestClient(app).post(
        "/api/v1/robustness/analyze/v2",
        json=json.loads(FIXTURE.read_text()),
        headers={"X-ISL-Response-Version": "2"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def clamped_options(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Options whose band is reported from samples (the goal here is anchored, so its band is clamped to [0, 1])."""
    rows = [
        o
        for o in body["options"]
        if (o.get("outcome") or {}).get("percentiles_source") == "samples"
    ]
    assert len(rows) == 3, [o["id"] for o in body["options"]]
    return rows


class TestTheServedOptionReadsAsOneDistribution:
    def test_the_59_option_mean_is_no_higher_than_the_cap(self, served):
        """RED before: mean 1.0109 (£126,363) beside p90 1.0 (£125,000)."""
        (p59,) = [o for o in served["options"] if o["id"] == P59]
        outcome = p59["outcome"]
        assert outcome["mean"] * CAP_GBP <= CAP_GBP + 1e-6, outcome
        assert outcome["p10"] <= outcome["mean"] <= outcome["p90"], outcome

    def test_the_59_option_std_is_that_of_draws_inside_the_domain(self, served):
        """A population inside [0, 1] has a standard deviation of at most 0.5 (RED before: ~18)."""
        (p59,) = [o for o in served["options"] if o["id"] == P59]
        assert 0.0 <= p59["outcome"]["std"] <= 0.5, p59["outcome"]


class TestInvariantsOfOneSetOfDraws:
    def test_every_clamped_mean_stays_inside_the_domain(self, served):
        for option in clamped_options(served):
            assert 0.0 <= option["outcome"]["mean"] <= 1.0, (option["id"], option["outcome"])

    def test_p05_is_never_above_p10(self, served):
        for option in clamped_options(served):
            downside = option.get("downside")
            if downside is None:
                continue
            assert downside["p05"] <= option["outcome"]["p10"] + 1e-12, (
                option["id"],
                downside,
                option["outcome"],
            )
            assert 0.0 <= downside["cvar_10"] <= 1.0, (option["id"], downside)


class TestTheStatusQuoIsUnchanged:
    def test_the_held_status_quo_still_reads_today(self, served):
        """CONTROL: a population already inside the domain is untouched — the status quo is £75,000 at every statistic."""
        (keep,) = [o for o in served["options"] if o["id"] == "keep_current_pricing"]
        for key in ("p10", "p50", "p90", "mean"):
            assert abs(keep["outcome"][key] - 0.6) <= 2e-4, (key, keep["outcome"])
