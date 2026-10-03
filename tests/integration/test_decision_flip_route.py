"""
SCIENCE ROBUSTNESS step 2 (EXPERIMENT): the on-demand decision-flip route through ISL's real app (in-process path).
"""

import copy
import json
import pathlib

import pytest

from src.api.main import app
from src.services.analysis_pool import create_analysis_pool
from src.services.compute_governor import ComputeGovernor

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "robustness" / "d1-isl-request.json"
ROUTE = "/api/v1/robustness/decision-flip/v2"
L1 = {"from_id": "sprint_capacity_for_ai_reporting", "to_id": "ai_reporting_module_availability"}


def body(links=(L1,), n=2000, replicates=2):
    q = json.loads(FIXTURE.read_text())
    q.update({"n_samples": n, "seed": "7"})
    return {"request": q, "links": list(links), "replicates": replicates}


@pytest.fixture
def real_pool():
    """A real one-worker pool, as startup installs it: the block never computes without one (review 5972444369).
    The test client does not run the app's startup, so the rows that compute a block install it themselves."""
    saved = {a: getattr(app.state, a) for a in ("analysis_pool", "governor", "analysis_workers") if hasattr(app.state, a)}
    app.state.analysis_workers = 1
    app.state.governor = ComputeGovernor(workers=1, queue_max=2)
    app.state.analysis_pool = create_analysis_pool(1)
    yield
    app.state.analysis_pool.shutdown(wait=True, cancel_futures=True)  # idle by now; a non-waiting shutdown races
    for a in ("analysis_pool", "governor", "analysis_workers"):
        if hasattr(app.state, a):
            delattr(app.state, a)
    for a, v in saved.items():
        setattr(app.state, a, v)


async def test_d1_returns_a_typed_block(client, real_pool):
    r = await client.post(ROUTE, json=body())
    assert r.status_code == 200, r.text[:400]
    b = r.json()
    assert b["method"] == "affine_crn_replicates_v1" and b["leader_option_id"] == "ai_reporting_module_sprint"
    (link,) = b["links"]
    assert link["status"] in ("quoted", "absent", "no_change") and link["current_mean"] == 0.25
    assert len(link["replicate_thresholds"]) == 2


async def test_unknown_link_is_a_client_error(client, real_pool):
    r = await client.post(ROUTE, json=body(links=({"from_id": "nope", "to_id": "quarterly_revenue"},)))
    assert r.status_code == 422 and "DECISION_FLIP_UNKNOWN_LINK" in r.text


async def test_a_deadline_breach_is_the_typed_504_never_a_hang(client, monkeypatch):
    import src.api.robustness as route
    from src.services.analysis_pool import AnalysisDeadlineExceeded

    async def too_slow(*a, **k):
        raise AnalysisDeadlineExceeded(80.0)

    monkeypatch.setattr(route, "run_decision_flip_offloaded", too_slow)
    r = await client.post(ROUTE, json=body())
    assert r.status_code == 504 and r.headers.get("Retry-After")


async def test_analyze_v2_is_unchanged_by_the_new_route(client):
    q = json.loads(FIXTURE.read_text())
    q.update({"n_samples": 500, "seed": "7"})
    a = (await client.post("/api/v1/robustness/analyze/v2", json=copy.deepcopy(q))).json()
    assert "horizon_view" not in a and "decision_flip" not in json.dumps(a)


async def test_a_worker_that_stays_dead_is_the_typed_503_with_retry_after(client, monkeypatch):
    # Review 5972142902 follow-up: the helper's second-failure Overload reaches the caller as the governor's own 503.
    import src.api.robustness as route
    from src.services.compute_governor import Overload

    async def dead(*a, **k):
        raise Overload(503, "analysis_worker_unavailable")

    monkeypatch.setattr(route, "run_decision_flip_offloaded", dead)
    r = await client.post(ROUTE, json=body())
    assert r.status_code == 503 and r.headers.get("Retry-After")
    assert "analysis_worker_unavailable" in r.text


async def test_a_pool_that_never_started_is_the_typed_503_never_an_inline_run(client):
    # Review 5972444369: startup failed, analysis_pool = None. Before: the whole block ran on the event loop.
    saved = getattr(app.state, "analysis_pool", "<unset>")
    app.state.analysis_pool = None
    try:
        r = await client.post(ROUTE, json=body())
    finally:
        if saved == "<unset>":
            delattr(app.state, "analysis_pool")
        else:
            app.state.analysis_pool = saved
    assert r.status_code == 503 and r.headers.get("Retry-After") and "analysis_worker_unavailable" in r.text
