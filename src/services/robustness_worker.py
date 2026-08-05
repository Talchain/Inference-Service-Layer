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

import json

from src.models.robustness_v2 import RobustnessRequestV2, RobustnessResponseV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


def encode_analysis_response(response: RobustnessResponseV2) -> str:
    """Serialize an analysis result for the worker->parent hop, NON-FINITE-SAFE.

    ROADMAP 2.477(i). This pairing exists because ``model_dump_json`` CANNOT
    round-trip this model: pydantic v2's JSON serializer writes ``null`` for
    every non-finite float, and the model's floats are required and
    non-Optional — so the parent's ``model_validate_json`` REJECTED ITS OWN
    WORKER'S OUTPUT. Measured on the deployed shape: a run with 267 of 500
    non-finite samples produced ``271 validation errors for
    RobustnessResponseV2`` (``outcome_distribution.mean/std/median/ci_upper``
    plus every non-finite ``samples[N]``: "Input should be a valid number
    [input_value=None]"), which the endpoint's catch-all mapped to a 500.

    Consequence before this fix: EVERY run with >=1 non-finite Monte Carlo
    sample died at the process-pool boundary — before ``validate_mc_samples``,
    before the v2 conversion loop, before any critique could be emitted. The
    entire 2.475 disclosure (honest validity counting, LOW_EFFECTIVE_SAMPLES,
    overflow-safe partial statistics) was unreachable on the deployed build,
    while passing every test, because ``TestClient(app)`` used bare does not
    run the app lifespan and therefore silently takes ``run_offloaded``'s
    ``pool is None`` IN-PROCESS branch, where no serialization boundary exists.

    ``json.dumps(..., allow_nan=True)`` emits the ``Infinity``/``-Infinity``/
    ``NaN`` tokens that ``json.loads`` parses back to the identical Python
    floats, so the hop is exact for finite AND non-finite values alike. This is
    an INTERNAL hop only: the public wire response is rendered separately by
    starlette with ``allow_nan=False`` and stays strictly JSON-compliant.

    Args:
        response: the analysis result produced in the worker process.

    Returns:
        A JSON string that ``decode_analysis_response`` restores exactly.
    """
    return json.dumps(response.model_dump(), allow_nan=True)


def decode_analysis_response(payload: str) -> RobustnessResponseV2:
    """Inverse of ``encode_analysis_response`` — the parent side of the hop.

    Kept adjacent to its encoder on purpose: the two are a single contract, and
    splitting them across modules is how the previous asymmetry (dump with one
    serializer, validate with another) survived unnoticed.
    """
    return RobustnessResponseV2.model_validate(json.loads(payload))


def run_robustness_v2(request_json: str) -> str:
    """Run one robustness analysis in the worker process.

    Args:
        request_json: ``RobustnessRequestV2`` serialized with ``model_dump_json``.

    Returns:
        ``RobustnessResponseV2`` serialized with ``encode_analysis_response`` —
        the exact object the in-process ``analyze()`` would have produced
        (byte-identical for a fixed seed; guarded by the determinism parity
        test PC-D), INCLUDING any non-finite floats it legitimately carries.
    """
    request = RobustnessRequestV2.model_validate_json(request_json)
    response = RobustnessAnalyzerV2().analyze(request)
    return encode_analysis_response(response)
