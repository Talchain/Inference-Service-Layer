"""PC-A — event-loop responsiveness under CPU offload (Codex F15, the headline).

The whole point of F15: a heavy CPU analysis must run in a *worker process* so the
Uvicorn event loop stays free to serve ``/health`` (and everything else). Before
F15 the pure-Python, GIL-bound Monte-Carlo loops ran *inside* ``async def`` and
froze the loop for the whole analysis.

POSITIVE CONTROL (programme trap #13 — an absence test that never saw a presence
is vacuous): the same test first runs with the offload DISABLED (in-process
fallback, ``analysis_pool = None``) and PROVES the event loop is frozen for ~the
analysis duration — a heartbeat coroutine that ticks every 20 ms records a
multi-hundred-ms stall, and ``/health`` issued during the window is delayed. Only
then does it assert the offloaded path keeps the loop responsive.

Uses ``httpx.AsyncClient`` + ``ASGITransport`` (the repo's client fixture pattern),
which does NOT run the app lifespan, so the pool/governor are installed on
``app.state`` explicitly by the fixtures here.
"""

import asyncio
import time

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.services.analysis_pool import create_analysis_pool
from src.services.compute_governor import ComputeGovernor

pytestmark = pytest.mark.asyncio

HEALTH = "/health"
ANALYZE = "/api/v1/robustness/analyze/v2"


def heavy_request(n_chance: int = 20, n_samples: int = 10000, seed: int = 42) -> dict:
    """A DAG heavy enough that in-process analysis takes ~1.5 s locally (~2.3M cost
    units, far under the 24M admission ceiling), so a frozen loop is unmistakable
    even on faster hardware (>3x margin over the 400 ms control threshold).
    """
    nodes = [{"id": "price", "kind": "decision", "label": "Price"}]
    edges = []
    prev = "price"
    for i in range(n_chance):
        nid = f"c{i}"
        nodes.append({"id": nid, "kind": "chance", "label": nid})
        edges.append(
            {
                "from": prev,
                "to": nid,
                "exists_probability": 0.95,
                "strength": {"mean": 0.4, "std": 0.1},
            }
        )
        if i >= 2:
            edges.append(
                {
                    "from": f"c{i-2}",
                    "to": nid,
                    "exists_probability": 0.9,
                    "strength": {"mean": 0.3, "std": 0.08},
                }
            )
        prev = nid
    nodes.append({"id": "revenue", "kind": "outcome", "label": "Revenue"})
    edges.append(
        {
            "from": prev,
            "to": "revenue",
            "exists_probability": 1.0,
            "strength": {"mean": 0.7, "std": 0.05},
        }
    )
    edges.append(
        {
            "from": "price",
            "to": "revenue",
            "exists_probability": 1.0,
            "strength": {"mean": 0.45, "std": 0.12},
        }
    )
    return {
        "request_id": "pc-a-heavy",
        "graph": {"nodes": nodes, "edges": edges},
        "options": [
            {"id": "low", "label": "Low", "interventions": {"price": 0.3}},
            {"id": "high", "label": "High", "interventions": {"price": 0.7}},
        ],
        "goal_node_id": "revenue",
        "seed": seed,
        "n_samples": n_samples,
        "include_e_values": True,
        "include_voi": False,
        "include_path_decomposition": True,
    }


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _install_pool(workers: int = 1):
    app.state.analysis_workers = workers
    app.state.governor = ComputeGovernor(workers=workers, queue_max=2 * workers)
    app.state.analysis_pool = create_analysis_pool(workers)


def _install_no_pool(workers: int = 1):
    # Offload DISABLED: handler self-heals to the in-process path (blocks the loop).
    app.state.analysis_workers = workers
    app.state.governor = ComputeGovernor(workers=workers, queue_max=2 * workers)
    app.state.analysis_pool = None


def _teardown_pool():
    pool = getattr(app.state, "analysis_pool", None)
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    for attr in ("analysis_pool", "governor", "analysis_workers"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


async def _run_case(client: AsyncClient, *, offload: bool):
    """Fire one heavy analysis; measure (a) the max event-loop stall via a 20 ms
    heartbeat, (b) a /health latency sampled during the analysis, and (c) the
    analysis wall-clock (denominator for the self-normalising stall bound in
    ``test_offload_keeps_event_loop_responsive``). Returns
    ``(max_stall_s, health_latency_s, health_status, analyze_status, analyze_wall_s)``.
    """
    stalls = []
    health_latency = {"value": None, "status": None}
    stop = asyncio.Event()

    async def heartbeat():
        last = time.monotonic()
        fired_health = False
        while not stop.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            stalls.append(now - last)
            last = now
            # Once we've seen the loop is alive, sample /health mid-analysis.
            if not fired_health:
                fired_health = True
                t0 = time.monotonic()
                r = await client.get(HEALTH)
                health_latency["value"] = time.monotonic() - t0
                health_latency["status"] = r.status_code

    hb = asyncio.create_task(heartbeat())
    # Give the heartbeat one tick head-start so the analyze doesn't win the very
    # first scheduling slot.
    await asyncio.sleep(0.03)
    t_analyze = time.monotonic()
    resp = await client.post(ANALYZE, json=heavy_request(), headers={"X-ISL-Response-Version": "2"})
    analyze_wall = time.monotonic() - t_analyze
    stop.set()
    await hb
    max_stall = max(stalls) if stalls else 0.0
    return max_stall, health_latency["value"], health_latency["status"], resp.status_code, analyze_wall


class TestPCAResponsiveness:
    async def test_positive_control_in_process_blocks_the_loop(self, client):
        """POSITIVE CONTROL: with the offload OFF the loop IS frozen for ~the
        analysis duration — proving the responsiveness assertion below can see the
        blocking it claims F15 removes."""
        _install_no_pool(workers=1)
        try:
            max_stall, health_lat, health_status, analyze_status, _analyze_wall = await _run_case(
                client, offload=False
            )
        finally:
            _teardown_pool()
        assert analyze_status == 200, "heavy analysis should still succeed in-process"
        # The loop was frozen for a big chunk of the ~1s analysis: a single
        # heartbeat gap far exceeds the 20ms cadence (and the 100ms SLA).
        assert max_stall > 0.4, (
            f"expected a >400ms loop stall in-process, saw {max_stall*1000:.0f}ms — "
            "the blocking this test asserts F15 removes was not visible (vacuous)"
        )

    async def test_offload_keeps_event_loop_responsive(self, client):
        """THE FIX: with the offload ON the loop stays responsive — no multi-hundred
        -ms stall, and /health returns 200 fast while the heavy analysis runs."""
        _install_pool(workers=1)
        try:
            max_stall, health_lat, health_status, analyze_status, analyze_wall = await _run_case(
                client, offload=True
            )
        finally:
            _teardown_pool()
        assert analyze_status == 200, "heavy analysis should succeed via the pool"
        assert health_status == 200, "/health must be served during the analysis"
        # Discriminate a GENUINE event-loop block from mere runner jitter by
        # comparing the max stall to the analysis wall-clock, which self-normalises
        # to runner speed. A synchronous compute ON the loop stalls it for ~the
        # WHOLE analysis (empirically stall/wall ~1.0 and >1.5s absolute — see the
        # in-process positive control above and the #2 mutation proof); a FREE loop
        # under even the worst runner jitter stalls only a small fraction of it
        # (empirically ~0.01-0.02 ratio, ~30-80ms).
        #
        # This replaces a fixed ``max_stall < 0.2`` ceiling that false-RED'd at
        # 0.257s on a loaded GitHub runner (PR #94 first run; passed on re-run,
        # zero coupling to that diff). The ratio + 1s ceiling tolerate that jitter
        # (0.257s over a >1.5s analysis ⇒ ratio ~0.15, well under 0.5) while still
        # catching a real block. A ratio is chosen over a pre-analysis idle
        # baseline because jitter that strikes DURING the analysis is invisible in
        # a pre-analysis sample (so a baseline can itself false-RED); the wall-clock
        # inflates together with the stall on a loaded runner, a fixed baseline does
        # not. The absolute 1s ceiling backstops a pathologically inflated wall.
        assert max_stall < 0.5 * analyze_wall and max_stall < 1.0, (
            f"event loop stalled {max_stall*1000:.0f}ms during a {analyze_wall*1000:.0f}ms "
            f"analysis (ratio {max_stall / analyze_wall:.2f}) with offload on — expected the "
            "loop free during worker compute (stall < 0.5x wall AND < 1s)"
        )
        # /health was SERVED during the analysis window (status asserted above).
        #
        # THE LATENCY BOUND THAT USED TO BE HERE WAS VACUOUS, AND IS REMOVED.
        # It read ``assert health_lat < 0.1`` and looked like the test's second
        # line of defence. Measured against the pre-F15 defect (offload
        # disabled, loop fully blocked) with the stall assertion neutralised so
        # this one was reachable:
        #
        #     BLOCKED: max_stall=0.938s  health_lat=0.006s  wall=0.935s
        #
        # 6ms. It PASSED with the event loop frozen for the entire analysis, so
        # it never had the power to catch the defect it appeared to guard. The
        # reason is structural, not a tuning problem: the sampler is a coroutine
        # on the very loop it is measuring, and it only fires after its first
        # successful tick — so when the loop is blocked it cannot run, and the
        # sample is necessarily taken OUTSIDE the blocking window. Any threshold
        # would pass. Loosening it would have been theatre; tightening it would
        # only have produced more false REDs.
        #
        # Its one real-world effect was that false-RED: PR #115 disclosed this
        # assertion failing a --no-verify push on a loaded runner while passing
        # 3/3 in isolation. Zero detection value, non-zero false-RED rate.
        #
        # The stall/wall ratio assertion above is the instrument that genuinely
        # detects this defect — it goes RED at ratio ~1.01 under the same
        # mutation. Coverage is therefore not reduced by this removal.
        #
        # Restoring a REAL "served fast while blocked" probe needs an
        # out-of-loop sampler (separate thread or process); tracked separately
        # rather than faked here.
        assert health_lat is not None, "/health should have been sampled during the analysis"


class TestPCBWireOverload:
    """PC-B at the wire: a saturated governor makes the endpoint return a typed
    503 + Retry-After IMMEDIATELY (not a hung connection), on BOTH the v2-enhanced
    and v1-legacy handler paths (both were modified)."""

    async def _saturate_and_request(self, client, headers):
        # queue_max=0 → admission bound = workers(1). Occupy the single slot with a
        # held admit so the real HTTP request is rejected deterministically before
        # any offload — no real compute needed, so the 503 is instant.
        gov = ComputeGovernor(workers=1, queue_max=0)
        app.state.analysis_workers = 1
        app.state.governor = gov
        app.state.analysis_pool = None  # unused: we 503 before offload
        release = asyncio.Event()

        async def occupy():
            async with gov.admit(cost_units=1, api_key="occupier"):
                await release.wait()

        occ = asyncio.create_task(occupy())
        try:
            await asyncio.sleep(0.05)  # ensure the slot is held
            t0 = time.monotonic()
            resp = await client.post(
                ANALYZE, json=heavy_request(n_chance=4, n_samples=400), headers=headers
            )
            elapsed = time.monotonic() - t0
            return resp, elapsed
        finally:
            release.set()
            await occ
            _teardown_pool()

    async def test_v2_enhanced_returns_503_retry_after(self, client):
        resp, elapsed = await self._saturate_and_request(client, {"X-ISL-Response-Version": "2"})
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After"), "503 must carry Retry-After"
        assert elapsed < 1.0, "overload must be rejected immediately, not hung"
        body = resp.json()
        assert body["code"] == "ISL_SERVICE_UNAVAILABLE"
        assert body["retryable"] is True

    async def test_v1_legacy_returns_503_retry_after(self, client):
        resp, elapsed = await self._saturate_and_request(client, {"X-ISL-Response-Version": "1"})
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After"), "503 must carry Retry-After"
        assert elapsed < 1.0, "overload must be rejected immediately, not hung"
