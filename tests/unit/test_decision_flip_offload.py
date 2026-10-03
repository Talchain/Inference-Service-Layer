"""
Review 5963778665 P1 (analysis_pool.py:249): a worker failure must never run the decision-flip block INLINE.

The block is up to ~80 s of Monte Carlo. Running it on the event loop after a ``BrokenProcessPool`` bypassed the hard
deadline and stalled every other request on the worker (the reviewer's crash harness returned after 45 ms against a
5 ms deadline). These rows drive the REAL helper, ``run_decision_flip_offloaded``, with stand-in pools.
"""

import asyncio
import concurrent.futures
import time
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import pytest

import src.services.analysis_pool as ap
import src.services.robustness_worker as worker
from src.models.robustness_v2 import DecisionFlipBlockV2
from src.services.compute_governor import Overload

BLOCK = DecisionFlipBlockV2(method="affine_crn_replicates_v1", leader_option_id=None, replicates=2, bound_abs=0.01,
                            bound_rel=0.15, grid_step=0.0025, links=[]).model_dump_json()


class Pool(concurrent.futures.Executor):
    """``broken``: the worker died. ``ok``: computes in the 'worker'. ``hang``: never answers."""

    def __init__(self, behaviour):
        self.behaviour, self.submits = behaviour, 0

    def submit(self, fn, *args, **kwargs):
        self.submits += 1
        fut = concurrent.futures.Future()
        if self.behaviour == "broken":
            fut.set_exception(BrokenProcessPool("worker died"))
        elif self.behaviour == "ok":
            Pool.in_worker = True
            try:
                fut.set_result(fn(*args))
            finally:
                Pool.in_worker = False
        return fut


Pool.in_worker = False


@pytest.fixture
def harness(monkeypatch):
    inline = []

    def compute(payload):
        if not Pool.in_worker:
            inline.append(payload)  # computed on the event loop: the defect
            time.sleep(0.05)
        return BLOCK

    monkeypatch.setattr(worker, "run_decision_flip_v2", compute)
    fresh = []

    def swap(app, stale):
        nxt = fresh.pop(0)
        app.state.analysis_pool = nxt
        return nxt

    monkeypatch.setattr(ap, "_swap_in_fresh_pool", swap)
    monkeypatch.setattr(ap, "_hard_kill_and_recreate", lambda app, pool: (pool, []))

    def make(first, *then):
        fresh.extend(then)
        app = SimpleNamespace(state=SimpleNamespace(analysis_pool=first))
        return app

    dreq = SimpleNamespace(model_dump_json=lambda: "{}")
    return SimpleNamespace(inline=inline, make=make, dreq=dreq)


async def test_a_broken_pool_resubmits_to_a_fresh_worker_never_inline(harness):
    first, fresh = Pool("broken"), Pool("ok")
    app = harness.make(first, fresh)
    block = await ap.run_decision_flip_offloaded(app, harness.dreq, "rid")
    assert block.method == "affine_crn_replicates_v1"
    assert harness.inline == [] and first.submits == 1 and fresh.submits == 1


async def test_a_pool_that_breaks_twice_is_a_typed_503_never_inline(harness):
    healed = Pool("ok")
    app = harness.make(Pool("broken"), Pool("broken"), healed)
    with pytest.raises(Overload) as err:
        await ap.run_decision_flip_offloaded(app, harness.dreq, "rid")
    assert (err.value.status_code, err.value.reason) == (503, "analysis_worker_unavailable")
    assert harness.inline == []
    assert app.state.analysis_pool is healed and healed.submits == 0  # left healthy for the NEXT request, not retried


async def test_the_resubmit_runs_inside_the_remaining_deadline(harness, monkeypatch):
    monkeypatch.setattr(ap, "ANALYSIS_HARD_DEADLINE_S", 0.005)
    app = harness.make(Pool("broken"), Pool("hang"))
    started = time.monotonic()
    with pytest.raises(ap.AnalysisDeadlineExceeded):
        await ap.run_decision_flip_offloaded(app, harness.dreq, "rid")
    assert time.monotonic() - started < 0.5 and harness.inline == []


async def test_control_a_healthy_pool_is_one_offloaded_call(harness):
    pool = Pool("ok")
    block = await ap.run_decision_flip_offloaded(harness.make(pool), harness.dreq, "rid")
    assert block.leader_option_id is None and pool.submits == 1 and harness.inline == []


def test_the_event_loop_is_never_handed_the_block_by_name():
    # Static belt to the rows above: the helper's body never calls the computation directly once a pool exists.
    import inspect

    src = inspect.getsource(ap.run_decision_flip_offloaded)
    assert "run_decision_flip_v2(payload)" not in src  # never inline, with or without a pool


async def test_a_pool_that_never_started_is_a_typed_503_never_inline(harness):
    # Review 5972444369: startup leaves analysis_pool = None when the pool fails to start (src/api/main.py).
    started = time.monotonic()
    with pytest.raises(Overload) as err:
        await ap.run_decision_flip_offloaded(harness.make(None), harness.dreq, "rid")
    assert (err.value.status_code, err.value.reason) == (503, "analysis_worker_unavailable")
    assert harness.inline == [] and time.monotonic() - started < 0.5


def test_asyncio_is_the_loop_under_test():
    assert asyncio.iscoroutinefunction(ap.run_decision_flip_offloaded)
