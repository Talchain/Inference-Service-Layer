"""F15 offload + compute governor — PC-B..PC-F + governor gate tests.

Codex F15 offloads CPU-bound robustness analysis to a process pool behind a
weighted concurrency governor, UNCONDITIONALLY (Paul ruling 18 Jul — no env
kill-switch), with self-healing (worker crash / pool loss → in-process fallback)
as the safety mechanism in place of a rollback flag. Because it ships unconditional
these paths are tested thoroughly, each with a positive control where an absence is
asserted (programme trap #13) and each guarding a real failure mode.

Design references: acceptance-evidence/a3-verify-2026-07-16/codex-capacity-DESIGN.md
§F15.5 (PC-A..PC-E) + the PAUL-RULING PC-F (crash resilience).
"""

import asyncio
import json
import logging
import os
import time
import types
from pathlib import Path

import pytest

from src.models.robustness_v2 import RobustnessRequestV2, RobustnessResponseV2
from src.services import analysis_pool
from src.services.analysis_pool import (
    AnalysisDeadlineExceeded,
    _terminate_pool_processes,
    create_analysis_pool,
    run_offloaded,
)
from src.services.compute_governor import ComputeGovernor, Overload
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2, compute_weighted_cost
from src.services.robustness_worker import run_robustness_v2
from tests.unit._f15_workers import crash_worker, pid_probe, sleep_worker, slow_analysis_worker

pytestmark = pytest.mark.asyncio

_VARIANTS = json.loads(
    (Path(__file__).resolve().parents[1] / "benchmarks" / "sample_variants.json").read_text()
)


def make_request(
    *,
    seed: int = 42,
    n_samples: int = 400,
    voi: bool = False,
    path_decomp: bool = False,
    e_values: bool = True,
    constraints: bool = False,
    goal_threshold: bool = False,
) -> RobustnessRequestV2:
    """Build a valid v2 request over the 3-node sample graph (has ``demand`` for
    EVPI, ``revenue`` goal). Flags toggle the optional phases so a single request
    can exercise every one (used by PC-E)."""
    kwargs: dict = dict(
        request_id="f15-test",
        graph=_VARIANTS["graphs"][2],
        options=_VARIANTS["options"],
        goal_node_id=_VARIANTS["goal_node_id"],
        seed=seed,
        n_samples=n_samples,
        include_e_values=e_values,
        include_voi=voi,
        include_path_decomposition=path_decomp,
    )
    if voi:
        kwargs["parameter_uncertainties"] = [
            {"node_id": "demand", "distribution": "normal", "std": 0.1}
        ]
    if goal_threshold:
        kwargs["goal_threshold"] = 100.0
    if constraints:
        kwargs["goal_constraints"] = [
            {"node_id": "revenue", "operator": ">=", "value": 50.0, "label": "Revenue floor"}
        ]
    return RobustnessRequestV2(**kwargs)


def fake_app(workers: int = 1, queue_max: int | None = None, max_cost: int | None = None):
    """A minimal stand-in for the FastAPI app carrying the pool + governor on
    ``app.state`` (the ASGITransport test client does not run the lifespan)."""
    app = types.SimpleNamespace()
    app.state = types.SimpleNamespace()
    app.state.analysis_workers = workers
    app.state.governor = ComputeGovernor(
        workers=workers,
        queue_max=queue_max if queue_max is not None else 2 * workers,
        max_cost=max_cost,
    )
    app.state.analysis_pool = create_analysis_pool(workers)
    return app


def teardown_app(app) -> None:
    pool = getattr(app.state, "analysis_pool", None)
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _science(resp: RobustnessResponseV2) -> dict:
    """JSON view with the one volatile field (wall-clock execution_time_ms) zeroed,
    so two runs of the same seeded analysis compare byte-for-byte on the SCIENCE."""
    d = json.loads(resp.model_dump_json())
    if isinstance(d.get("metadata"), dict):
        d["metadata"]["execution_time_ms"] = 0
    return d


async def _await_pid_gone(pid: int, timeout: float = 4.0) -> bool:
    """Poll until ``pid`` no longer exists (killed AND reaped), or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.05)
    return False


# ===========================================================================
# Governor gate unit tests + PC-B (bounded queue)
# ===========================================================================
class TestGovernorGates:
    async def test_pc_b_bounded_queue_last_request_rejected_503(self):
        """PC-B: with W executing + QUEUE_MAX queued (bound admitted), the next
        admit is rejected EARLY + TYPED (503 + Retry-After), not left hanging; and
        in-flight never exceeds W."""
        gov = ComputeGovernor(workers=1, queue_max=2, max_cost=10**12)  # bound = 3
        release = asyncio.Event()
        entered = asyncio.Event()

        async def hold(i: int):
            async with gov.admit(cost_units=1, api_key=f"k{i}"):
                entered.set()
                await release.wait()

        tasks = [asyncio.create_task(hold(i)) for i in range(3)]
        # Let them settle: 1 executing (holds the sole worker slot), 2 queued.
        await asyncio.sleep(0.1)
        assert gov.admitted == 3, "all three should be admitted (1 running + 2 queued)"
        assert gov.inflight == 1, "in-flight must never exceed workers (=1)"

        # POSITIVE CONTROL: the 4th admit raises immediately, it does not hang.
        with pytest.raises(Overload) as ei:
            async with gov.admit(cost_units=1, api_key="k4"):
                pass  # pragma: no cover - never entered
        assert ei.value.status_code == 503
        assert ei.value.reason == "service_busy_queue_full"
        assert ei.value.retry_after > 0

        release.set()
        await asyncio.gather(*tasks)
        # Everything released cleanly — no slot leak.
        assert gov.admitted == 0 and gov.inflight == 0

    async def test_per_caller_concurrency_429(self):
        """A single caller holding W admitted gets 429 (its own concurrency) while
        the global bound still has room — one noisy caller cannot 503 the service."""
        gov = ComputeGovernor(workers=2, queue_max=4, max_cost=10**12)  # bound = 6
        release = asyncio.Event()

        async def hold():
            async with gov.admit(cost_units=1, api_key="same"):
                await release.wait()

        tasks = [asyncio.create_task(hold()) for _ in range(2)]  # key "same" -> 2 admitted
        await asyncio.sleep(0.1)
        assert gov.admitted == 2  # global bound (6) not reached

        with pytest.raises(Overload) as ei:
            async with gov.admit(cost_units=1, api_key="same"):
                pass  # pragma: no cover
        assert ei.value.status_code == 429
        assert ei.value.reason == "caller_concurrency_exceeded"

        # POSITIVE CONTROL: a DIFFERENT caller is NOT rejected by the per-key gate.
        # Both worker slots are occupied, so it legitimately QUEUES on the
        # semaphore (admitted, waiting) rather than raising 429 — proving the gate
        # is per-key, not global.
        other_ran = asyncio.Event()

        async def other():
            async with gov.admit(cost_units=1, api_key="other"):
                other_ran.set()

        other_task = asyncio.create_task(other())
        await asyncio.sleep(0.2)
        assert not other_task.done(), "different caller was rejected — gate not per-key"
        assert not other_ran.is_set(), "different caller should be queued, not running yet"

        # Freeing the slots lets the queued different-caller proceed to completion.
        release.set()
        await asyncio.wait_for(other_task, timeout=2.0)
        assert other_ran.is_set()
        await asyncio.gather(*tasks)

    async def test_weighted_cost_budget_503(self):
        """Gate 3: the weighted in-flight cost budget (= F8 cost_units) rejects with
        503 when Σ admitted cost would exceed it. Driven via a small max_cost
        override. (At the real ceiling budget = workers × max_cost, so gate 3 binds
        FIRST for near-max_cost bursts — capped at ≈workers — while the slot gate
        dominates for cheap jobs; see the corrected compute_governor gate-3 note.)"""
        gov = ComputeGovernor(workers=4, queue_max=4, max_cost=100)
        release = asyncio.Event()

        async def hold(cost: int, key: str):
            async with gov.admit(cost_units=cost, api_key=key):
                await release.wait()

        t = asyncio.create_task(hold(80, "a"))  # 80 <= 100 -> admitted
        await asyncio.sleep(0.05)

        # POSITIVE CONTROL that the gate can bind: a 30-unit job would push Σ to
        # 110 > 100 -> 503; a 15-unit job (Σ=95) is admitted.
        with pytest.raises(Overload) as ei:
            async with gov.admit(cost_units=30, api_key="b"):
                pass  # pragma: no cover
        assert ei.value.status_code == 503
        assert ei.value.reason == "service_busy_cost_budget"

        fits_ran = asyncio.Event()

        async def fits():
            async with gov.admit(cost_units=15, api_key="c"):  # 80 + 15 = 95 <= 100
                fits_ran.set()

        tf = asyncio.create_task(fits())
        await asyncio.wait_for(fits_ran.wait(), timeout=1.0)
        await tf
        release.set()
        await t

    async def test_slot_released_on_body_exception(self):
        """Robustness: an exception inside the admitted body must still release the
        slot (finally), or capacity silently shrinks forever."""
        gov = ComputeGovernor(workers=1, queue_max=1, max_cost=10**12)
        with pytest.raises(ValueError):
            async with gov.admit(cost_units=1, api_key="x"):
                raise ValueError("boom")
        assert gov.admitted == 0 and gov.inflight == 0
        # Capacity fully restored: a fresh admit succeeds.
        ran = asyncio.Event()
        async with gov.admit(cost_units=1, api_key="y"):
            ran.set()
        assert ran.is_set()


# ===========================================================================
# PC-C — the hard deadline actually stops the work (worker terminated)
# ===========================================================================
class TestPCCDeadlineTerminatesWorker:
    async def test_terminate_pool_processes_kills_running_worker(self):
        """Mechanism: a running worker's process is SIGKILLed (CPU freed). Positive
        control: the pids are proven ALIVE first, so the 'gone' assertion is not
        vacuous."""
        pool = create_analysis_pool(1)
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(pool, sleep_worker, 60.0)
        # Let the worker spawn and start sleeping.
        await asyncio.sleep(1.0)
        pids = [p.pid for p in pool._processes.values() if p.pid is not None]
        assert pids, "pool should have at least one worker process"

        # POSITIVE CONTROL: alive now.
        for pid in pids:
            os.kill(pid, 0)  # no exception == alive

        killed = _terminate_pool_processes(pool)
        assert set(killed) == set(pids), "every worker pid should be killed"

        for pid in pids:
            assert await _await_pid_gone(pid), f"worker {pid} still alive after SIGKILL"

        fut.cancel()
        fut.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        pool.shutdown(wait=False, cancel_futures=True)

    async def test_run_offloaded_deadline_raises_and_recreates_pool(self, monkeypatch):
        """Wire: an analysis that blows the hard deadline raises
        AnalysisDeadlineExceeded, its worker is terminated (pid gone), and the pool
        is recreated so the NEXT request is served."""
        # Submit a worker that sleeps past any deadline, with a tiny deadline.
        monkeypatch.setattr(analysis_pool, "run_robustness_v2", slow_analysis_worker)
        monkeypatch.setattr(analysis_pool, "ANALYSIS_HARD_DEADLINE_S", 0.75)

        killed_holder: dict = {}
        orig_terminate = analysis_pool._terminate_pool_processes

        def spy(pool):
            pids = orig_terminate(pool)
            killed_holder["pids"] = pids
            return pids

        monkeypatch.setattr(analysis_pool, "_terminate_pool_processes", spy)

        app = fake_app(workers=1)
        old_pool = app.state.analysis_pool
        try:
            with pytest.raises(AnalysisDeadlineExceeded):
                await run_offloaded(app, make_request(), "pc-c-wire")

            # Worker(s) terminated — pid gone within a bound.
            assert killed_holder.get("pids"), "the deadline path must terminate workers"
            for pid in killed_holder["pids"]:
                assert await _await_pid_gone(pid), f"deadline worker {pid} not terminated"

            # Pool recreated → recovery.
            assert app.state.analysis_pool is not old_pool, "pool must be recreated"

            # The recreated pool serves the next request (restore the real worker).
            monkeypatch.setattr(analysis_pool, "run_robustness_v2", run_robustness_v2)
            monkeypatch.setattr(analysis_pool, "ANALYSIS_HARD_DEADLINE_S", 80.0)
            resp = await run_offloaded(app, make_request(), "pc-c-recover")
            assert resp.results, "recreated pool should serve the next analysis"
        finally:
            teardown_app(app)


# ===========================================================================
# PC-D — determinism byte-parity (guards the refactor fold)
# ===========================================================================
class TestPCDDeterminism:
    async def test_offloaded_response_byte_identical_to_in_process(self):
        req = make_request(seed=42, n_samples=500, voi=True, path_decomp=True)
        ref = RobustnessAnalyzerV2().analyze(req)

        app = fake_app(workers=1)
        try:
            off = await run_offloaded(app, req, "pc-d")
        finally:
            teardown_app(app)

        ref_json = json.loads(ref.model_dump_json())
        off_json = json.loads(off.model_dump_json())

        # The offloaded and in-process runs may or may not differ on the ONE
        # volatile field, the wall-clock ``metadata.execution_time_ms``: a slow
        # runner records two different timings; a fast runner rounds them to the
        # same integer ms and the diff set is empty. BOTH are correct. The old
        # ``diffs == ["metadata.execution_time_ms"]`` REQUIRED the timings to
        # differ and inverted into a false-RED (``diffs == []``) whenever a fast
        # runner produced identical timings (22-23 Jul PR runs). Accept the
        # timing field as the ONLY PERMITTED difference (subset — empty is fine)
        # while still failing on ANY drift in a science path.
        diffs = _diff_paths(ref_json, off_json)
        assert set(diffs) <= {
            "metadata.execution_time_ms"
        }, f"offloaded science drifted from in-process: differing paths {diffs}"
        # Byte-identical on the science after zeroing that one volatile field.
        # This pair (subset check + science equality) is the positive control
        # that the comparison can SEE drift: a genuine offloaded-worker byte
        # change (the #1 mutation proof) adds a science path to ``diffs`` and
        # breaks this equality, turning BOTH assertions RED.
        assert _science(ref) == _science(off)


def _diff_paths(a, b, prefix=""):
    """Return the sorted list of dotted paths where a and b differ."""
    paths = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            child = f"{prefix}.{k}" if prefix else str(k)
            paths += _diff_paths(a.get(k), b.get(k), child)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            paths.append(prefix + "[len]")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                paths += _diff_paths(x, y, f"{prefix}[{i}]")
    else:
        if a != b:
            paths.append(prefix)
    return sorted(paths)


# ===========================================================================
# PC-E — pickle/JSON round-trip: no field loss across the process boundary
# ===========================================================================
class TestPCERoundTrip:
    async def test_every_optional_phase_and_constraints_survive_round_trip(self):
        req = make_request(
            seed=7,
            n_samples=500,
            voi=True,
            path_decomp=True,
            e_values=True,
            constraints=True,
            goal_threshold=True,
        )
        # The request survives model_dump_json -> model_validate_json unchanged.
        reparsed = RobustnessRequestV2.model_validate_json(req.model_dump_json())
        assert reparsed.model_dump() == req.model_dump(), "request lost a field on round-trip"

        # And the worker (fed the serialized request) returns a response carrying
        # every phase the request asked for — proving nothing was silently dropped
        # across the boundary.
        resp = RobustnessResponseV2.model_validate_json(run_robustness_v2(req.model_dump_json()))
        assert resp.results, "base MC results present"
        assert resp.edge_e_values, "e-values present (include_e_values)"
        assert resp.factor_evpi, "EVPI present (include_voi + parameter_uncertainties)"
        assert resp.path_decomposition is not None, "path decomposition present"


# ===========================================================================
# PC-F — crash resilience / self-healing (replaces the kill-switch)
# ===========================================================================
class TestPCFCrashResilience:
    async def test_worker_crash_falls_back_in_process_with_correct_science(
        self, monkeypatch, caplog
    ):
        """Kill a worker mid-analysis → the request STILL completes via the in-
        process fallback with byte-correct science + a degraded-mode log, and the
        pool recovers for the next request."""
        req = make_request(seed=42, n_samples=500, voi=True, path_decomp=True)
        reference = RobustnessAnalyzerV2().analyze(req)  # in-process ground truth

        # Make the offloaded worker die (os._exit) → BrokenProcessPool.
        monkeypatch.setattr(analysis_pool, "run_robustness_v2", crash_worker)

        app = fake_app(workers=1)
        old_pool = app.state.analysis_pool
        try:
            with caplog.at_level(logging.WARNING, logger="src.services.analysis_pool"):
                resp = await run_offloaded(app, req, "pc-f")

            # (1) Correct science — the fallback ran the real analyzer in-process.
            assert _science(resp) == _science(
                reference
            ), "self-healed response must be byte-identical to the in-process result"
            # (2) The degraded event was logged (paths/reason only).
            degraded = [r for r in caplog.records if r.msg == "isl_analysis_offload_degraded"]
            assert degraded, "self-heal must log isl_analysis_offload_degraded"
            assert getattr(degraded[0], "reason", None) == "broken_process_pool"
            # (3) Pool recovered — a fresh pool object is installed.
            assert app.state.analysis_pool is not old_pool, "pool must be recreated"

            # (4) The NEXT request succeeds via the recreated pool (real worker).
            monkeypatch.setattr(analysis_pool, "run_robustness_v2", run_robustness_v2)
            resp2 = await run_offloaded(app, make_request(seed=99), "pc-f-recover")
            assert _science(resp2) == _science(
                RobustnessAnalyzerV2().analyze(make_request(seed=99))
            ), "recreated pool must serve correct science on the next request"
        finally:
            teardown_app(app)

    async def test_missing_pool_runs_in_process_not_flagged_degraded(self, caplog):
        """A never-configured pool (e.g. a no-lifespan test) runs in-process with
        CORRECT science and is NOT mislabelled a degradation (that label is
        reserved for real self-heal events — keeps the alarm meaningful)."""
        app = types.SimpleNamespace()
        app.state = types.SimpleNamespace()
        app.state.analysis_pool = None
        req = make_request(seed=42)
        with caplog.at_level(logging.WARNING, logger="src.services.analysis_pool"):
            resp = await run_offloaded(app, req, "no-pool")
        assert _science(resp) == _science(RobustnessAnalyzerV2().analyze(req))
        assert not [r for r in caplog.records if r.msg == "isl_analysis_offload_degraded"]
