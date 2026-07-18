"""Process-pool worker entrypoint for CPU-bound robustness analysis (Codex F15).

This module is imported and executed **inside a separate worker process** of the
analysis ``ProcessPoolExecutor`` (see ``src/services/analysis_pool.py``). It runs
in another interpreter with its own GIL, which is the whole point of F15: the hot
Monte-Carlo loops in ``RobustnessAnalyzerV2.analyze`` are pure-Python, GIL-bound,
so a thread pool cannot relieve the event loop — only a *process* pool can.

Design invariants (must hold for the offload to be correct and picklable):

* The entrypoint is a **module-level pure function** taking a JSON string and
  returning a JSON string. Under the ``spawn`` start method (macOS local + the
  safe default) the worker re-imports this module, so the function must be
  importable by reference with no closure state.
* Only **pure data crosses the process boundary** — a serialized
  ``RobustnessRequestV2`` in, a serialized ``RobustnessResponseV2`` out. The
  un-picklable objects (``SeededRNG``/``PCG64``, the samplers, the
  ``SCMEvaluatorV2``) are all constructed *inside* ``analyze()`` and never
  marshalled; ``RobustnessAnalyzerV2`` is stateless, so the worker rebuilds a
  fresh instance per call.
* Imports are kept **minimal and free of ``src.api.main``** so a spawned worker
  never re-runs app/Sentry/middleware setup. (Verified: neither
  ``robustness_analyzer_v2`` nor ``robustness_v2`` imports the app module.)
"""

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


def run_robustness_v2(request_json: str) -> str:
    """Run one robustness analysis in the worker process.

    Args:
        request_json: ``RobustnessRequestV2`` serialized with ``model_dump_json``.

    Returns:
        ``RobustnessResponseV2`` serialized with ``model_dump_json`` — the exact
        object the in-process ``analyze()`` would have produced (byte-identical
        for a fixed seed; guarded by the determinism parity test PC-D).
    """
    request = RobustnessRequestV2.model_validate_json(request_json)
    response = RobustnessAnalyzerV2().analyze(request)
    return response.model_dump_json()
