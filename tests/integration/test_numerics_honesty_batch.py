"""ISL numerics-honesty batch — ROADMAP 2.477(a)–(i).

Every test here pins a defect that was REPRODUCED BY EXECUTION at pristine
`c6a87cb5`, with its mechanism read from the server traceback rather than
inferred from the status code (a 500 is not evidence of its own cause):

  (i) process-pool boundary   ValidationError (271 errors) in
                              analysis_pool.run_offloaded — pydantic's JSON
                              serializer writes `null` for non-finite floats and
                              the parent then rejects its own worker's output.
                              THE deployed-path defect: it made 2.475 dark in
                              production while every test passed, because a bare
                              TestClient runs no lifespan and silently takes the
                              in-process branch instead.
  (a) all-non-finite option   ValueError: Out of range float values are not JSON
                              compliant (JSONResponse render) — outcome.mean=inf,
                              outcome.std=nan; MONTE_CARLO_FAILED died inside the
                              500 it caused.
  (b) absurd intervention     OverflowError: cannot convert float infinity to
                              integer at request_validator.py:108.
  (c) all-NaN winners loop    ZeroDivisionError: float division by zero at
                              robustness_analyzer_v2.py:2923.
  (f) 0.8 <= validity < 1.0   the same render ValueError — 2.475 gated its
                              overflow-safe statistics on `status == "partial"`,
                              leaving every option in that validity band on the
                              poisoned raw-array mean/std.
  (g) all-finite ~1e299       the same render ValueError — np.std's squared
                              deviations overflow on a fully-valid option.
  (e) decision_evpi           min over a PARTIAL downside population can only be
                              >= the true min_o, i.e. it OVERSTATES EVPI.

Assertions bind to their object by IDENTITY (option id, critique code) and never
by a value predicate a sibling could satisfy (trap #19), and bodies are parsed
with `parse_constant` because plain `json.loads` ACCEPTS `Infinity`/`NaN` tokens
and therefore cannot witness JSON compliance (trap #13).
"""

import json
import math
import os

import pytest

from fastapi.testclient import TestClient

from src.api.main import app
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_worker import run_robustness_v2

# NOTE: encode_analysis_response / decode_analysis_response are imported INSIDE
# the tests that use them, not at module scope. At pristine they do not exist,
# and a module-level import would abort COLLECTION of this whole file — so every
# other test in the batch would report as an error rather than as its own
# specific RED. Per-test imports keep each defect's RED signature legible.

# float64 max is ~1.797e308: one HUGE contribution stays finite, two sum to inf.
HUGE_FINITE = 1.7e308
# ~1e299: every sample stays FINITE, but np.std's squared deviations overflow.
EXTREME_FINITE = 1e299

N_SAMPLES = 200
SEED = 42
ENDPOINT = "/api/v1/robustness/analyze/v2?response_version=2"


@pytest.fixture
def client():
    """Bare client — no lifespan, so run_offloaded takes its in-process branch."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    if os.environ.get("ISL_AUTH_DISABLED", "").lower() == "true":
        return {}
    return {"X-API-Key": os.environ.get("ISL_API_KEY", "test_key")}


# ---------------------------------------------------------------- graph shapes


def _two_factor_graph(observed_value: float, coin_probability: float) -> dict:
    """Two root factors feeding one goal.

    `observed_value` sets the magnitude regime and `coin_probability` sets how
    often the second factor's edge exists, which together select the shape:
      (HUGE, 0.5) -> both factors present on ~half the draws, their sum
                     overflows to inf there -> PARTIAL validity;
      (HUGE, 1.0) -> both always present -> EVERY draw inf -> ZERO valid;
      (EXTREME, 1.0) -> every draw finite but near float64 max -> std overflow.
    """
    return {
        "nodes": [
            {
                "id": "f_certain",
                "kind": "factor",
                "label": "Certain factor",
                "observed_state": {"value": observed_value},
            },
            {
                "id": "f_coin",
                "kind": "factor",
                "label": "Coinflip factor",
                "observed_state": {"value": observed_value},
            },
            {"id": "d_lever", "kind": "decision", "label": "Small lever"},
            {"id": "goal_out", "kind": "outcome", "label": "Goal"},
        ],
        "edges": [
            {
                "from": "f_certain",
                "to": "goal_out",
                "exists_probability": 1.0,
                "strength": {"mean": 1.0, "std": 0.05},
            },
            {
                "from": "f_coin",
                "to": "goal_out",
                "exists_probability": coin_probability,
                "strength": {"mean": 1.0, "std": 0.05},
            },
            {
                "from": "d_lever",
                "to": "goal_out",
                "exists_probability": 1.0,
                "strength": {"mean": 0.9, "std": 0.05},
            },
        ],
    }


_TWO_OPTIONS = [
    {"id": "opt_degraded", "label": "Degraded option", "interventions": {"d_lever": 0.7}},
    {
        "id": "opt_healthy",
        "label": "Healthy option",
        "interventions": {"f_certain": 0.5, "f_coin": 0.5},
    },
]


def _request(observed_value: float, coin_probability: float, n_samples: int = N_SAMPLES) -> dict:
    return {
        "graph": _two_factor_graph(observed_value, coin_probability),
        "options": [dict(o) for o in _TWO_OPTIONS],
        "goal_node_id": "goal_out",
        "n_samples": n_samples,
        "seed": SEED,
    }


def _evpi_partial_population_request() -> dict:
    """A run whose downside population is PARTIAL — one option carries a joint
    regret, another SAMPLED option does not.

    Both factors sit at float-max on coin-flip edges, so roughly a quarter of
    draws overflow. `opt_a` moves the lever and ends up 'partial' but keeps a
    finite regret; `opt_b` pins the second factor, stays 'computed', and loses
    its downside because its own joint regret against opt_a's extreme draws is
    non-finite. That asymmetry is the whole point: it is the only shape in which
    `min` over the surviving regrets would OVERSTATE the true `min_o`.
    """
    return {
        "graph": {
            "nodes": [
                {
                    "id": "f0",
                    "kind": "factor",
                    "label": "F0",
                    "observed_state": {"value": HUGE_FINITE},
                },
                {
                    "id": "f1",
                    "kind": "factor",
                    "label": "F1",
                    "observed_state": {"value": HUGE_FINITE},
                },
                {"id": "d_lever", "kind": "decision", "label": "Lever"},
                {"id": "goal_out", "kind": "outcome", "label": "Goal"},
            ],
            "edges": [
                {
                    "from": "f0",
                    "to": "goal_out",
                    "exists_probability": 0.5,
                    "strength": {"mean": 1.0, "std": 0.05},
                },
                {
                    "from": "f1",
                    "to": "goal_out",
                    "exists_probability": 0.5,
                    "strength": {"mean": 1.0, "std": 0.05},
                },
                {
                    "from": "d_lever",
                    "to": "goal_out",
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.9, "std": 0.05},
                },
            ],
        },
        "options": [
            {"id": "opt_a", "label": "Lever option", "interventions": {"d_lever": 0.7}},
            {"id": "opt_b", "label": "Pinned option", "interventions": {"f1": 0.5}},
        ],
        "goal_node_id": "goal_out",
        "n_samples": N_SAMPLES,
        "seed": SEED,
    }


def _aggregator_all_nan_request() -> dict:
    """Every option NaN on every draw.

    An aggregator sums two HUGE factors to inf; the edge from it to the goal is a
    coin flip, and the evaluator multiplies an ABSENT edge by 0.0 — `0.0 * inf`
    is NaN. Both options therefore see NaN at every draw, `max()` finds no
    winner (NaN != NaN), and the winners loop divided by len([]).
    """
    return {
        "graph": {
            "nodes": [
                {
                    "id": "f_a",
                    "kind": "factor",
                    "label": "A",
                    "observed_state": {"value": HUGE_FINITE},
                },
                {
                    "id": "f_b",
                    "kind": "factor",
                    "label": "B",
                    "observed_state": {"value": HUGE_FINITE},
                },
                {"id": "agg", "kind": "factor", "label": "Aggregator"},
                {"id": "d_lever", "kind": "decision", "label": "Lever"},
                {"id": "goal_out", "kind": "outcome", "label": "Goal"},
            ],
            "edges": [
                {
                    "from": "f_a",
                    "to": "agg",
                    "exists_probability": 1.0,
                    "strength": {"mean": 1.0, "std": 0.01},
                },
                {
                    "from": "f_b",
                    "to": "agg",
                    "exists_probability": 1.0,
                    "strength": {"mean": 1.0, "std": 0.01},
                },
                {
                    "from": "agg",
                    "to": "goal_out",
                    "exists_probability": 0.5,
                    "strength": {"mean": 1.0, "std": 0.01},
                },
                {
                    "from": "d_lever",
                    "to": "goal_out",
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.9, "std": 0.05},
                },
            ],
        },
        "options": [
            {"id": "opt_a", "label": "A", "interventions": {"d_lever": 0.7}},
            {"id": "opt_b", "label": "B", "interventions": {"d_lever": 0.2}},
        ],
        "goal_node_id": "goal_out",
        "n_samples": N_SAMPLES,
        "seed": SEED,
    }


# ---------------------------------------------------------------- helpers


def _post(client, auth_headers, payload):
    return client.post(ENDPOINT, json=payload, headers=auth_headers)


def _strict_parse(resp):
    """Parse asserting JSON compliance. Python's json.loads ACCEPTS the
    Infinity/NaN tokens by default, so a plain .json() cannot prove the body is
    compliant; parse_constant fires exactly on those tokens."""

    def _reject(token):
        pytest.fail(f"non-JSON-compliant float token in response body: {token}")

    return json.loads(resp.text, parse_constant=_reject)


def _option(body, option_id):
    matches = [o for o in body["options"] if o["id"] == option_id]
    assert len(matches) == 1, f"expected exactly one option {option_id!r}"
    return matches[0]


def _critiques(body, code):
    return [c for c in body["critiques"] if c["code"] == code]


def _assert_all_finite(body):
    """Walk the whole body; no non-finite float may reach the wire anywhere."""
    bad = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append((path, node))

    walk(body)
    assert bad == [], f"non-finite floats reached the wire: {bad}"


# ================================================================ (i) the pool


class TestPoolBoundaryPreservesNonFiniteValues:
    """2.477(i) — the worker->parent hop must round-trip non-finite floats.

    This is the defect that made 2.475 unreachable on the deployed build.
    """

    def test_worker_hop_round_trips_a_non_finite_analysis_result(self):
        """Drives the REAL worker entrypoint on the REAL partial-validity
        request, then decodes it exactly as the parent process does."""
        from src.services.robustness_worker import decode_analysis_response

        request = RobustnessRequestV2.model_validate(_request(HUGE_FINITE, 0.5))
        payload = run_robustness_v2(request.model_dump_json())

        response = decode_analysis_response(payload)

        degraded = [r for r in response.results if r.option_id == "opt_degraded"]
        assert len(degraded) == 1, "fixture must produce the identified option"
        samples = degraded[0].outcome_distribution.samples
        assert samples is not None and len(samples) == N_SAMPLES

        # The point of the hop: the non-finite draws SURVIVE it. Before the fix
        # they became `null` and the parent's re-validation rejected them.
        n_non_finite = sum(1 for s in samples if not math.isfinite(s))
        assert 0 < n_non_finite < N_SAMPLES, (
            "fixture must carry a strict mix of finite and non-finite samples "
            f"across the hop; got {n_non_finite} non-finite of {len(samples)}"
        )

    def test_encoder_decoder_pair_is_exact_for_non_finite_floats(self):
        """The pairing itself, independent of any analysis: whatever the encoder
        writes, the decoder must restore bit-for-bit."""
        from src.services.robustness_worker import (
            decode_analysis_response,
            encode_analysis_response,
        )

        request = RobustnessRequestV2.model_validate(_request(HUGE_FINITE, 0.5))
        original = decode_analysis_response(run_robustness_v2(request.model_dump_json()))

        restored = decode_analysis_response(encode_analysis_response(original))

        for before, after in zip(original.results, restored.results):
            assert before.option_id == after.option_id
            b_samples = before.outcome_distribution.samples or []
            a_samples = after.outcome_distribution.samples or []
            assert len(b_samples) == len(a_samples)
            for i, (b, a) in enumerate(zip(b_samples, a_samples)):
                if math.isnan(b):
                    assert math.isnan(a), f"sample {i}: NaN became {a!r}"
                else:
                    assert b == a, f"sample {i}: {b!r} became {a!r}"

    async def test_run_offloaded_preserves_non_finite_samples(self):
        """THE WIRING TEST — `run_offloaded` itself, on its OFFLOAD branch.

        The two tests above pin the encoder/decoder contract; this one pins that
        `analysis_pool` actually USES it. A mutant restoring
        `RobustnessResponseV2.model_validate_json` in the parent survives them
        and dies here, which is the whole point: the defect was the PAIRING, not
        either half.

        Driven with a bare namespace as the "app" and a ThreadPoolExecutor as the
        pool. `run_in_executor` accepts either executor and `run_offloaded`'s code
        is identical for both — same `run_robustness_v2` call, same string, same
        `decode_analysis_response`. So this proves the serialization contract,
        which is where the defect lives; it does not prove process isolation, and
        it is deliberately not driven through the FastAPI app: doing that made
        five unrelated budget/golden-sensitive tests fail in the full-suite run
        while each passed in isolation (measured — the endpoint path drags in the
        compute governor and wall-clock budgets). The real ProcessPoolExecutor is
        what produced the `271 validation errors for RobustnessResponseV2`
        signature in the lane evidence, and the process boundary is witnessed
        again post-deploy by the live-witness recipe in the PR body.
        """
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        from src.services.analysis_pool import run_offloaded

        executor = ThreadPoolExecutor(max_workers=1)
        fake_app = SimpleNamespace(state=SimpleNamespace(analysis_pool=executor))
        try:
            request = RobustnessRequestV2.model_validate(_request(HUGE_FINITE, 0.5))
            response = await run_offloaded(fake_app, request, "test-2477i")
        finally:
            executor.shutdown(wait=True)

        degraded = [r for r in response.results if r.option_id == "opt_degraded"]
        assert len(degraded) == 1, "fixture must produce the identified option"
        samples = degraded[0].outcome_distribution.samples
        assert samples is not None and len(samples) == N_SAMPLES
        n_non_finite = sum(1 for s in samples if not math.isfinite(s))
        assert 0 < n_non_finite < N_SAMPLES, (
            "the non-finite draws must SURVIVE the offload hop; before the fix "
            "they became null and the parent rejected its own worker's output. "
            f"Got {n_non_finite} non-finite of {len(samples)}"
        )

    async def test_run_offloaded_matches_the_in_process_result_exactly(self):
        """Positive control + the offload's own contract: for a fixed seed the
        offloaded result must equal what the in-process path produces. Proves the
        hop is lossless in BOTH directions, not merely non-throwing."""
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        from src.services.analysis_pool import run_offloaded
        from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

        request = RobustnessRequestV2.model_validate(_request(HUGE_FINITE, 0.5))
        in_process = RobustnessAnalyzerV2().analyze(
            RobustnessRequestV2.model_validate(_request(HUGE_FINITE, 0.5))
        )

        executor = ThreadPoolExecutor(max_workers=1)
        fake_app = SimpleNamespace(state=SimpleNamespace(analysis_pool=executor))
        try:
            offloaded = await run_offloaded(fake_app, request, "test-2477i-parity")
        finally:
            executor.shutdown(wait=True)

        assert [r.option_id for r in offloaded.results] == [r.option_id for r in in_process.results]
        for expected, actual in zip(in_process.results, offloaded.results):
            exp_samples = expected.outcome_distribution.samples or []
            act_samples = actual.outcome_distribution.samples or []
            assert len(exp_samples) == len(act_samples)
            for i, (e, a) in enumerate(zip(exp_samples, act_samples)):
                if math.isnan(e):
                    assert math.isnan(a), f"{expected.option_id}[{i}]: NaN became {a!r}"
                else:
                    assert e == a, f"{expected.option_id}[{i}]: {e!r} became {a!r}"


# ================================================================ (a)


class TestAllNonFiniteOptionShipsOn200:
    """2.477(a) — an option whose every draw is non-finite must ship as a
    non-computed option CARRYING its critique, not die as a 500."""

    def test_zero_valid_samples_ships_with_monte_carlo_failed(self, client, auth_headers):
        resp = _post(client, auth_headers, _request(HUGE_FINITE, 1.0))

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        degraded = _option(body, "opt_degraded")
        assert degraded["status"] == "failed"
        outcome = degraded["outcome"]
        assert outcome["n_valid_samples"] == 0
        assert outcome["validity_ratio"] == 0.0
        assert outcome["percentiles_source"] == "unavailable"

        # Absent, never null and never a fabricated 0.0 — there is no honest
        # mean for a distribution with no draws.
        assert "mean" not in outcome, f"outcome.mean must be ABSENT, got {outcome.get('mean')!r}"
        assert "std" not in outcome, f"outcome.std must be ABSENT, got {outcome.get('std')!r}"
        assert "downside" not in degraded

        # The critique that used to die inside the 500 it caused.
        failed_rows = _critiques(body, "MONTE_CARLO_FAILED")
        assert len(failed_rows) == 1, f"expected one MONTE_CARLO_FAILED, got {failed_rows!r}"
        assert failed_rows[0]["affected_option_ids"] == ["opt_degraded"]
        assert failed_rows[0]["severity"] == "blocker"

        # The healthy sibling is untouched and still carries real numbers —
        # the in-run control proving the failed option's absences are specific.
        healthy = _option(body, "opt_healthy")
        assert healthy["status"] == "computed"
        assert healthy["outcome"]["n_valid_samples"] == N_SAMPLES
        assert math.isfinite(healthy["outcome"]["mean"])
        assert math.isfinite(healthy["outcome"]["std"])


# ================================================================ (b)


class TestAbsurdInterventionValueIsAnHonest422:
    """2.477(b) — a value the pipeline cannot canonicalise is the CALLER's
    fault: a 422 naming the field, not a 500 INTERNAL_ERROR."""

    def test_out_of_range_intervention_value_blocks_with_422(self, client, auth_headers):
        payload = _request(0.6, 0.5)
        payload["options"] = [
            {"id": "opt_absurd", "label": "Absurd", "interventions": {"d_lever": 1.9e299}},
            {"id": "opt_sane", "label": "Sane", "interventions": {"d_lever": 0.5}},
        ]

        resp = _post(client, auth_headers, payload)

        assert resp.status_code == 422, resp.text[:600]
        body = _strict_parse(resp)
        rows = _critiques(body, "INTERVENTION_VALUE_INVALID")
        assert len(rows) == 1, f"expected one INTERVENTION_VALUE_INVALID, got {rows!r}"
        assert rows[0]["affected_option_ids"] == ["opt_absurd"]
        assert rows[0]["affected_node_ids"] == ["d_lever"]
        assert rows[0]["severity"] == "blocker"
        assert "d_lever" in rows[0]["message"]
        assert "{" not in rows[0]["message"], "message template left unpopulated"

        assert _critiques(body, "INTERNAL_ERROR") == [], "must not be reported as a server error"

    def test_non_finite_intervention_value_blocks_with_422(self, client, auth_headers):
        """The originally-documented case for this critique code, which also had
        no emission site: an infinite value."""
        payload = _request(0.6, 0.5)
        payload["options"] = [
            {"id": "opt_inf", "label": "Inf", "interventions": {"d_lever": 1e400}},
            {"id": "opt_sane", "label": "Sane", "interventions": {"d_lever": 0.5}},
        ]

        resp = _post(client, auth_headers, payload)

        assert resp.status_code == 422, resp.text[:600]
        rows = _critiques(_strict_parse(resp), "INTERVENTION_VALUE_INVALID")
        assert len(rows) == 1
        assert rows[0]["affected_option_ids"] == ["opt_inf"]

    def test_realistic_intervention_values_are_not_blocked(self, client, auth_headers):
        """Absence control — the guard must not reject ordinary requests. Its
        ability to SEE a violation is proven by the two tests above."""
        resp = _post(client, auth_headers, _request(0.6, 0.5))

        assert resp.status_code == 200, resp.text[:600]
        assert _critiques(_strict_parse(resp), "INTERVENTION_VALUE_INVALID") == []


# ================================================================ (c)


class TestAllNaNSamplesDoNotCrashTheWinnersLoop:
    """2.477(c) — a draw where no option is finite has no winner; it must not
    divide by zero."""

    def test_all_nan_run_returns_a_response(self, client, auth_headers):
        resp = _post(client, auth_headers, _aggregator_all_nan_request())

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        for option_id in ("opt_a", "opt_b"):
            option = _option(body, option_id)
            assert option["status"] == "failed"
            assert option["outcome"]["n_valid_samples"] == 0

        assert len(_critiques(body, "MONTE_CARLO_FAILED")) == 2
        assert _critiques(body, "INTERNAL_ERROR") == []


# ================================================================ (f) and (g)


class TestOverflowSafeStatisticsAreGatedOnFinitenessNotStatus:
    """2.477(f)/(g) — 2.475 gated its overflow-safe mean/std on
    `status == "partial"`, which left two live 500 classes behind."""

    def test_partially_invalid_but_computed_option_still_ships(self, client, auth_headers):
        """(f) The validity band 0.8 <= ratio < 1.0: enough non-finite draws to
        poison the raw-array mean/std, too few to make the option 'partial'."""
        payload = _request(HUGE_FINITE, 0.5)
        # Nudge the coin so the degraded option lands ABOVE MIN_VALID_RATIO but
        # below 1.0 — the band the status gate could not see.
        payload["graph"]["edges"][1]["exists_probability"] = 0.1

        resp = _post(client, auth_headers, payload)

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        degraded = _option(body, "opt_degraded")
        outcome = degraded["outcome"]
        n_valid = outcome["n_valid_samples"]
        assert 0.8 <= outcome["validity_ratio"] < 1.0, (
            "fixture must land INSIDE the band the status gate missed; got "
            f"validity_ratio={outcome['validity_ratio']} ({n_valid}/{N_SAMPLES})"
        )
        assert degraded["status"] == "computed", "the band is 'computed' by definition"
        # The whole point: mean/std are present and finite despite the poison.
        assert math.isfinite(outcome["mean"])
        assert math.isfinite(outcome["std"])

    def test_all_finite_extreme_magnitude_option_still_ships(self, client, auth_headers):
        """(g) Every draw finite at ~1e299 — np.std's squared deviations
        overflow, so a FULLY VALID option shipped std=inf and 500'd."""
        resp = _post(client, auth_headers, _request(EXTREME_FINITE, 1.0))

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        degraded = _option(body, "opt_degraded")
        outcome = degraded["outcome"]
        assert outcome["n_valid_samples"] == N_SAMPLES, "every draw must be finite here"
        assert outcome["validity_ratio"] == 1.0
        assert degraded["status"] == "computed"
        assert math.isfinite(outcome["mean"])
        assert math.isfinite(outcome["std"])
        # Not a degenerate rescue: the magnitude is genuinely reported.
        assert outcome["mean"] > 1e298

    def test_ordinary_run_keeps_the_analyzer_values_verbatim(self, client, auth_headers):
        """No-regression control. For a run with no non-finite draw the new
        branch must never be taken — outcome.mean/std stay exactly the
        analyzer's np.mean/np.std over the raw array."""
        resp = _post(client, auth_headers, _request(0.6, 0.5))

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        for option in body["options"]:
            assert option["outcome"]["validity_ratio"] == 1.0
            assert math.isfinite(option["outcome"]["mean"])
            assert math.isfinite(option["outcome"]["std"])


# ================================================================ (e)


class TestDecisionEvpiIsHonestOrHonestlyAbsent:
    """2.477(e) — min over a PARTIAL regret population can only be >= the true
    min_o, so it OVERSTATES EVPI. It must be omitted, not approximated."""

    def test_evpi_absent_when_a_sampled_option_lacks_a_downside(self, client, auth_headers):
        """Executed shape producing a genuinely PARTIAL regret population: one
        option carries a downside and another SAMPLED option does not.

        ⚠ The fixture matters more than the assertion here, and the first one I
        wrote was vacuous. With both factors at -/+ float-max the joint regret is
        poisoned for EVERY option, so the regret population came out EMPTY — and
        `decision_evpi` is absent for an empty population whether or not the
        completeness guard exists. The mutant that drops the guard SURVIVED that
        fixture. This shape (both edges coin-flips at float-max, one option on
        the lever and one pinning a factor) leaves `opt_a` WITH a downside and
        `opt_b` sampled WITHOUT one — the only configuration in which `min` over
        the survivors would actually overstate, and therefore the only one that
        tests the guard. Found by sweeping 375 configurations for the shape; 104
        produce it.
        """
        payload = _evpi_partial_population_request()

        resp = _post(client, auth_headers, payload)

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        with_downside = [o["id"] for o in body["options"] if "downside" in o]
        sampled_without_downside = [
            o["id"]
            for o in body["options"]
            if o["outcome"]["percentiles_source"] == "samples" and "downside" not in o
        ]
        # BOTH halves are required for this test to mean anything: a survivor to
        # take the (overstating) minimum over, AND a gap that makes it wrong.
        assert with_downside == ["opt_a"], (
            "fixture must leave exactly opt_a carrying a regret — otherwise the "
            f"population is empty and absence proves nothing; got {with_downside}"
        )
        assert sampled_without_downside == ["opt_b"], (
            "fixture must leave opt_b SAMPLED but without a downside — that gap "
            f"is what makes min-over-survivors an overstatement; got {sampled_without_downside}"
        )
        assert "decision_evpi" not in body, (
            "decision_evpi must be ABSENT while the regret population is "
            f"incomplete; options missing a downside: {sampled_without_downside}"
        )

    def test_evpi_present_and_exact_when_the_population_is_complete(self, client, auth_headers):
        """Positive control — the emission is not simply switched off. On an
        ordinary run every sampled option carries a downside, and the value must
        equal the wire minimum exactly."""
        resp = _post(client, auth_headers, _request(0.6, 0.5))

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)

        regrets = [o["downside"]["expected_regret"] for o in body["options"] if "downside" in o]
        assert len(regrets) == len(body["options"]), "population must be complete here"
        assert "decision_evpi" in body, "a complete population must emit decision_evpi"
        assert body["decision_evpi"] == pytest.approx(min(regrets))


class TestDecisionEvpiValidatorAcceptsOnlyHonestAbsence:
    """The model-level half of (e): the emission-iff validator had to be taught
    that absence is legitimate — but ONLY when the wire itself shows the gap.
    Silent loss must still fail loud."""

    @staticmethod
    def _option(option_id, *, regret=None, source="samples"):
        from src.models.response_v2 import DownsideV2, OptionResultV2, OutcomeDistributionV2

        return OptionResultV2(
            id=option_id,
            outcome=OutcomeDistributionV2(
                mean=1.0 if source == "samples" else None,
                std=0.5 if source == "samples" else None,
                p10=0.5 if source == "samples" else None,
                p50=1.0 if source == "samples" else None,
                p90=1.5 if source == "samples" else None,
                n_samples=100,
                n_valid_samples=100 if source == "samples" else 0,
                validity_ratio=1.0 if source == "samples" else 0.0,
                percentiles_source=source,
            ),
            downside=(
                DownsideV2(cvar_10=0.1, p05=0.2, expected_regret=regret)
                if regret is not None
                else None
            ),
            status="computed" if source == "samples" else "failed",
        )

    def _build(self, options, evpi):
        from src.utils.response_builder import ResponseBuilder
        from src.models.response_v2 import RequestEchoV2

        builder = ResponseBuilder(
            request_id="test-evpi",
            request_echo=RequestEchoV2(
                graph_node_count=2,
                graph_edge_count=1,
                options_count=len(options),
                goal_node_id_hash="deadbeefcafe",
                n_samples=100,
                response_version_requested=2,
                include_diagnostics=False,
            ),
        )
        builder.set_results(options)
        builder.set_decision_evpi(evpi)
        return builder.build()

    def test_absence_is_rejected_when_every_sampled_option_has_a_downside(self):
        """The silent-loss direction still bites."""
        options = [
            self._option("opt_a", regret=2.0),
            self._option("opt_b", regret=5.0),
        ]
        with pytest.raises(ValueError, match="decision_evpi"):
            self._build(options, None)

    def test_absence_is_accepted_when_a_sampled_option_lacks_a_downside(self):
        """The honest-absence exemption — and it must be visible ON THE WIRE."""
        options = [
            self._option("opt_a", regret=2.0),
            self._option("opt_b", regret=None),  # sampled, no downside -> gap
        ]
        response = self._build(options, None)
        assert response.decision_evpi is None

    def test_an_unsampled_option_does_not_license_absence(self):
        """An option with no samples was never in the regret population; it must
        not be usable as an excuse to drop a perfectly good EVPI."""
        options = [
            self._option("opt_a", regret=2.0),
            self._option("opt_dead", regret=None, source="unavailable"),
        ]
        with pytest.raises(ValueError, match="decision_evpi"):
            self._build(options, None)

        # ...and with the number present it is accepted, on the same options.
        response = self._build(
            [
                self._option("opt_a", regret=2.0),
                self._option("opt_dead", regret=None, source="unavailable"),
            ],
            2.0,
        )
        assert response.decision_evpi == 2.0

    def test_fabrication_direction_still_bites(self):
        """evpi present with NO regret population at all."""
        options = [self._option("opt_dead", regret=None, source="unavailable")]
        with pytest.raises(ValueError, match="decision_evpi"):
            self._build(options, 1.0)


class TestOutcomeSummaryStatsValidator:
    """2.477(a)'s fail-loud half: mean/std may go absent only together, and only
    for an option with no usable sample population."""

    @staticmethod
    def _outcome(**overrides):
        from src.models.response_v2 import OutcomeDistributionV2

        kwargs = dict(
            mean=None,
            std=None,
            n_samples=100,
            n_valid_samples=0,
            validity_ratio=0.0,
            percentiles_source="unavailable",
        )
        kwargs.update(overrides)
        return OutcomeDistributionV2(**kwargs)

    def test_absent_together_on_an_unsampled_option_is_accepted(self):
        outcome = self._outcome()
        assert outcome.mean is None and outcome.std is None

    def test_half_absent_is_rejected(self):
        with pytest.raises(ValueError, match="half-summarised"):
            self._outcome(mean=1.0)
        with pytest.raises(ValueError, match="half-summarised"):
            self._outcome(std=1.0)

    def test_absent_on_a_sampled_option_is_rejected(self):
        with pytest.raises(ValueError, match="silent data loss"):
            self._outcome(
                percentiles_source="samples",
                n_valid_samples=100,
                validity_ratio=1.0,
                p10=0.5,
                p50=1.0,
                p90=1.5,
            )


# ================================================================ (j)
#
# ROADMAP 2.477(j) — probability_of_goal was the LAST statistic in the family
# still computed over the UNFILTERED sample array.
#
# `int(np.sum(samples_array >= threshold)) / len(samples)` has no finiteness
# gate, and `+inf >= anything` is True. So the one shape that makes ISL classify
# an option `status: "failed"` — every draw non-finite — made every draw "meet"
# the goal, and the option shipped:
#
#     status: "failed", n_valid_samples: 0, win_probability: 0.0,
#     probability_of_goal: 1.0          <-- fabricated
#
# REPRODUCED BY EXECUTION at pristine 28fe0c95 through the real endpoint, and
# the inversion is total: the option that wins NOTHING claims a 100% chance of
# hitting the goal while the option that wins EVERY draw claims 0%.
#
# The sibling metrics were all gated years apart and this one was missed:
# mean/std by 2.477(f)/(g) at the emission boundary, p05/p10/p50/p90 and the
# whole downside block by 2.475/2.477(f), the winners loop by 2.477(c), the
# auto-noise std by its own finite_mask, factor elasticities by 2.514(a).
#
# THE RULE APPLIED HERE is the one this field's own resolver already states
# (`_resolve_goal_threshold_in_sample_frame`): "FAIL CLOSED ... No fabricated
# number, no clamp, no silent default." A draw whose compared quantity is not a
# real number is UNINFORMATIVE — it is excluded from both the numerator and the
# denominator, exactly as `n_valid_samples` already counts and `validity_ratio`
# already discloses. With no informative draw at all there is no honest
# probability, so the field is OMITTED (`exclude_none` => absent, never null,
# never a fabricated 0.0 — 0.0 would assert a measured zero where nothing was
# measured).


def _mixed_finite_and_infinite_request() -> dict:
    """One option whose draws MIX small-finite, huge-finite and +inf.

    A small always-present factor, plus TWO float-max factors on independent
    coin-flip edges:
      * neither edge present  -> a tiny finite outcome that does NOT meet 1.0;
      * one edge present      -> ~1.7e308, finite, MEETS;
      * both edges present    -> their sum overflows to +inf, which "meets" only
                                 because `inf >= anything` is True.

    This is the shape that discriminates the ARITHMETIC from the empty-guard:
    the population is neither all-finite nor all-non-finite, so a fix that only
    special-cased "zero valid samples" would leave it reporting the raw figure.
    """
    return {
        "graph": {
            "nodes": [
                {
                    "id": "f_small",
                    "kind": "factor",
                    "label": "Small",
                    "observed_state": {"value": 0.2},
                },
                {
                    "id": "f_a",
                    "kind": "factor",
                    "label": "A",
                    "observed_state": {"value": HUGE_FINITE},
                },
                {
                    "id": "f_b",
                    "kind": "factor",
                    "label": "B",
                    "observed_state": {"value": HUGE_FINITE},
                },
                {"id": "d_lever", "kind": "decision", "label": "Lever"},
                {"id": "goal_out", "kind": "outcome", "label": "Goal"},
            ],
            "edges": [
                {
                    "from": "f_small",
                    "to": "goal_out",
                    "exists_probability": 1.0,
                    "strength": {"mean": 1.0, "std": 0.01},
                },
                {
                    "from": "f_a",
                    "to": "goal_out",
                    "exists_probability": 0.5,
                    "strength": {"mean": 1.0, "std": 0.05},
                },
                {
                    "from": "f_b",
                    "to": "goal_out",
                    "exists_probability": 0.5,
                    "strength": {"mean": 1.0, "std": 0.05},
                },
                {
                    "from": "d_lever",
                    "to": "goal_out",
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.9, "std": 0.05},
                },
            ],
        },
        "options": [
            {"id": "opt_mixed", "label": "Mixed", "interventions": {"d_lever": 0.7}},
            {"id": "opt_healthy", "label": "Healthy", "interventions": {"f_a": 0.4, "f_b": 0.4}},
        ],
        "goal_node_id": "goal_out",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "goal_threshold": 1.0,
        "goal_threshold_frame": "delta",
        "noise_multiplier": 0,
    }


def _goal_request(observed_value: float, coin_probability: float, threshold: float) -> dict:
    payload = _request(observed_value, coin_probability)
    payload["goal_threshold"] = threshold
    payload["goal_threshold_frame"] = "delta"
    return payload


class TestProbabilityOfGoalIsGatedOnFiniteness:
    """2.477(j) — probability_of_goal must be computed over the INFORMATIVE
    draws only, and omitted when there are none."""

    def test_all_non_finite_option_omits_probability_of_goal(self, client, auth_headers):
        """THE DEFECT. `opt_degraded` has zero valid samples; at pristine it
        shipped `probability_of_goal: 1.0`."""
        resp = _post(client, auth_headers, _goal_request(HUGE_FINITE, 1.0, 1.0))

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        degraded = _option(body, "opt_degraded")

        # PRECONDITION, pinned in-test (trap 13b): this assertion is only about
        # the fabricated-probability defect if the option really does have an
        # all-non-finite population. If the graph shape ever drifts, this fails
        # FIRST and names the drift instead of silently passing on a healthy
        # option that happens to have no goal probability.
        assert degraded["status"] == "failed"
        assert degraded["outcome"]["n_valid_samples"] == 0
        assert degraded["outcome"]["validity_ratio"] == 0.0

        assert "probability_of_goal" not in degraded, (
            "an option with ZERO informative draws must OMIT probability_of_goal; "
            f"got {degraded.get('probability_of_goal')!r} "
            "(1.0 is the pristine fabrication: inf >= threshold on every draw)"
        )

    def test_win_probability_and_goal_probability_cannot_contradict(self, client, auth_headers):
        """The user-visible shape of the defect, bound by identity to the two
        options: at pristine the option that won NOTHING claimed a 100% chance
        of hitting the goal while the option that won EVERY draw claimed 0%."""
        resp = _post(client, auth_headers, _goal_request(HUGE_FINITE, 1.0, 1.0))
        body = _strict_parse(resp)

        degraded = _option(body, "opt_degraded")
        healthy = _option(body, "opt_healthy")

        # Precondition: this really is the "wins nothing / wins everything" pair.
        assert degraded["win_probability"] == 0.0
        assert healthy["win_probability"] == 1.0

        assert degraded.get("probability_of_goal") is None, (
            "the option that wins no draw must not claim a goal probability at "
            f"all; got {degraded.get('probability_of_goal')!r}"
        )

    def test_mixed_population_counts_only_the_finite_draws(self, client, auth_headers):
        """The ARITHMETIC, not the empty-guard: 64 of `opt_mixed`'s 200 draws
        overflow to +inf and were being counted as goal-meeting."""
        resp = _post(client, auth_headers, _mixed_finite_and_infinite_request())

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        mixed = _option(body, "opt_mixed")
        outcome = mixed["outcome"]

        # PRECONDITION: the population must genuinely be MIXED, or this test is
        # measuring something else entirely.
        n_valid = outcome["n_valid_samples"]
        assert 0 < n_valid < N_SAMPLES, (
            f"fixture no longer produces a mixed population (n_valid={n_valid} "
            f"of {N_SAMPLES}) — the arithmetic claim below would be vacuous"
        )
        assert n_valid == 136, f"population shape drifted: n_valid={n_valid}, expected 136"

        pog = mixed.get("probability_of_goal")
        assert pog is not None, "a mixed population still has informative draws"

        # 92 of the 136 FINITE draws meet the threshold.
        assert pog == pytest.approx(92 / 136), (
            f"probability_of_goal must be taken over the {n_valid} informative "
            f"draws only; got {pog!r}"
        )
        # And it is NOT the pristine figure, which counted the 64 +inf draws as
        # goal-meeting. Asserting the target value alone would pass if the two
        # ever coincided.
        assert pog != pytest.approx(
            156 / N_SAMPLES
        ), "probability_of_goal is still being taken over the raw array"

    def test_healthy_option_keeps_its_genuine_probability_of_goal(self, client, auth_headers):
        """OPPOSITE-DIRECTION TWIN / no-regression control. A fully finite
        option must keep its real number — including a genuine 0.0, which is a
        MEASURED zero and must not be suppressed into an absence. Green at
        pristine by construction; its job is to RED if the gate over-suppresses,
        which is what the over-suppression mutants prove."""
        resp = _post(client, auth_headers, _goal_request(HUGE_FINITE, 1.0, 1.0))
        body = _strict_parse(resp)

        healthy = _option(body, "opt_healthy")

        # Precondition: fully valid population, so there is nothing to gate.
        assert healthy["status"] == "computed"
        assert healthy["outcome"]["n_valid_samples"] == N_SAMPLES
        assert healthy["outcome"]["validity_ratio"] == 1.0

        assert "probability_of_goal" in healthy, (
            "a fully-finite option must still report probability_of_goal — "
            "trading the lie for a gap is not a fix"
        )
        assert healthy["probability_of_goal"] == pytest.approx(
            0.0
        ), "a measured zero must survive the finiteness gate unchanged"

    def test_ordinary_run_probability_is_unchanged(self, client, auth_headers):
        """Second twin, on an ordinary all-finite request with a NON-degenerate
        probability — proves the gate is a no-op on the population every real
        user has, and that it does not merely preserve the 0.0/1.0 endpoints."""
        resp = _post(client, auth_headers, _goal_request(0.6, 0.5, 0.5))
        body = _strict_parse(resp)

        healthy = _option(body, "opt_healthy")
        assert healthy["outcome"]["n_valid_samples"] == N_SAMPLES
        assert healthy["probability_of_goal"] == pytest.approx(
            0.53
        ), "the all-finite path must be byte-identical to pristine"


def _level_frame_non_finite_status_quo_request() -> dict:
    """LEVEL frame, where the compared quantity is NOT the option's samples.

    `level_i = goal_baseline + (option_sample_i - status_quo_sample_i)`, so a
    non-finite STATUS-QUO draw makes the level non-finite even when the option's
    own sample is perfectly finite.

    `opt_pinned` pins both float-max factors, so ALL 200 of its own samples are
    finite and its status is "computed" with `n_valid_samples == 200`. The
    status quo pins nothing, so on 64 draws both coin edges fire and it
    overflows to +inf — and those 64 levels are -inf.

    This is the OPPOSITE-DIRECTION face of the same defect: `-inf >= threshold`
    is False, so the raw computation silently UNDERSTATES the probability
    instead of overstating it. It is also invisible to `n_valid_samples`, which
    counts the option's own samples and reports a fully-valid 200.
    """
    return {
        "graph": {
            "nodes": [
                {
                    "id": "f_small",
                    "kind": "factor",
                    "label": "Small",
                    "observed_state": {"value": 0.2},
                },
                {
                    "id": "f_a",
                    "kind": "factor",
                    "label": "A",
                    "observed_state": {"value": HUGE_FINITE},
                },
                {
                    "id": "f_b",
                    "kind": "factor",
                    "label": "B",
                    "observed_state": {"value": HUGE_FINITE},
                },
                {"id": "d_lever", "kind": "decision", "label": "Lever"},
                {
                    "id": "goal_out",
                    "kind": "outcome",
                    "label": "Goal",
                    "observed_state": {"value": 0.5, "baseline": 0.5},
                },
            ],
            "edges": [
                {
                    "from": "f_small",
                    "to": "goal_out",
                    "exists_probability": 1.0,
                    "strength": {"mean": 1.0, "std": 0.01},
                },
                {
                    "from": "f_a",
                    "to": "goal_out",
                    "exists_probability": 0.5,
                    "strength": {"mean": 1.0, "std": 0.05},
                },
                {
                    "from": "f_b",
                    "to": "goal_out",
                    "exists_probability": 0.5,
                    "strength": {"mean": 1.0, "std": 0.05},
                },
                {
                    "from": "d_lever",
                    "to": "goal_out",
                    "exists_probability": 1.0,
                    "strength": {"mean": 0.9, "std": 0.05},
                },
            ],
        },
        "options": [
            {"id": "opt_pinned", "label": "Pinned", "interventions": {"f_a": 0.4, "f_b": 0.4}},
            {"id": "opt_lever", "label": "Lever", "interventions": {"d_lever": 0.7}},
        ],
        "goal_node_id": "goal_out",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "goal_threshold": 0.5,
        "goal_threshold_frame": "level",
        "noise_multiplier": 0,
    }


class TestProbabilityOfGoalMasksTheComparedQuantityNotTheSamples:
    """2.477(j), level frame — the gate must mask what is COMPARED.

    Masking `samples_array` instead of the computed levels passes every
    delta-frame test (there the two arrays are the same object) while leaving
    this case wrong, which is exactly why this class exists: it is the
    discriminating mutant pair for the choice of masked array.
    """

    def test_non_finite_status_quo_draws_are_excluded_in_level_frame(self, client, auth_headers):
        resp = _post(client, auth_headers, _level_frame_non_finite_status_quo_request())

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        pinned = _option(body, "opt_pinned")

        # PRECONDITION, pinned in-test: the option's OWN samples must be fully
        # finite, or this test is just the delta-frame case again and proves
        # nothing about which array is masked.
        assert pinned["status"] == "computed"
        assert pinned["outcome"]["n_valid_samples"] == N_SAMPLES, (
            "opt_pinned must have a fully-finite sample population — the whole "
            "point is that n_valid_samples cannot see this defect"
        )
        assert pinned["outcome"]["validity_ratio"] == 1.0

        pog = pinned["probability_of_goal"]

        # 44 of the 136 draws whose LEVEL is finite meet the threshold. The
        # other 64 draws have a +inf status quo, so their level is -inf: not a
        # real number, and therefore not an observation either way.
        assert pog == pytest.approx(44 / 136), (
            "probability_of_goal must be taken over the draws whose COMPARED "
            f"quantity (the level) is finite; got {pog!r}"
        )
        # 44/200 is what masking the option's own samples — or masking nothing
        # at all, as at pristine — produces. Asserting only the target value
        # would pass if the two ever coincided.
        assert pog != pytest.approx(44 / N_SAMPLES), (
            "probability_of_goal is being masked on the option's samples "
            "instead of on the computed levels (or not masked at all)"
        )


def _computed_status_inflated_request() -> dict:
    """The case that defeats every DOWNSTREAM guard.

    PLoT's `isCrownableCandidate` and CEE's `isComparable` both fail CLOSED on
    an allowlist of `status === 'computed'` — so a "failed" or "partial" option
    is already excluded from crowning and from goal-attainment coaching.

    This option is `status: "computed"` with 91.5% validity. It passes both
    allowlists, is crownable, is comparable, looks entirely ordinary — and at
    pristine it still carried an inflated `probability_of_goal`, because 17 of
    its 200 draws overflowed to +inf and every one of them was counted as
    goal-meeting.

    Same shape as the mixed fixture with the coin edges lowered to 0.30, so
    P(both present) is small enough to leave validity above MIN_VALID_RATIO.
    This is 2.477(f)'s lesson repeating one field along: gating on
    `status == "partial"` was too narrow then, and keying a downstream guard on
    `status == "computed"` cannot see this now.
    """
    payload = _mixed_finite_and_infinite_request()
    payload["graph"]["edges"][1]["exists_probability"] = 0.30
    payload["graph"]["edges"][2]["exists_probability"] = 0.30
    payload["options"][0]["id"] = "opt_computed"
    payload["options"][0]["label"] = "Computed but inflated"
    return payload


class TestProbabilityOfGoalOnACrownableComputedOption:
    """2.477(j) — the defect is NOT confined to options a downstream allowlist
    already refuses. It reaches a fully 'computed', crownable, comparable one."""

    def test_computed_option_above_min_valid_ratio_is_still_gated(self, client, auth_headers):
        resp = _post(client, auth_headers, _computed_status_inflated_request())

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        option = _option(body, "opt_computed")
        outcome = option["outcome"]

        # PRECONDITION, pinned in-test: the option must be BOTH 'computed' (so
        # every downstream status allowlist admits it) AND carry non-finite
        # draws (so there is something to gate). Lose either and this test
        # stops being about the guard-defeating case.
        assert option["status"] == "computed", (
            "the whole point is an option the downstream allowlists ADMIT; "
            f"got status={option['status']!r}"
        )
        assert (
            outcome["n_valid_samples"] == 183
        ), f"population drifted: n_valid={outcome['n_valid_samples']}, expected 183"
        assert outcome["validity_ratio"] > 0.8, (
            "must sit ABOVE MIN_VALID_RATIO, or the option would be 'partial' "
            "and the downstream allowlists would already exclude it"
        )

        pog = option["probability_of_goal"]

        # 96 of the 183 informative draws meet the threshold.
        assert pog == pytest.approx(
            96 / 183
        ), f"probability_of_goal must be taken over the 183 informative draws; got {pog!r}"
        # 113/200 is the pristine figure: the same 96 genuine draws PLUS the 17
        # that merely overflowed. Asserting the target alone would pass if the
        # two ever coincided.
        assert pog != pytest.approx(
            113 / N_SAMPLES
        ), "probability_of_goal is still counting the +inf draws as goal-meeting"


# ================================================================ (k)
#
# ROADMAP 2.477(k) — the CONSTRAINT channel, the sibling the (j) sweep missed.
#
# WHY IT WAS MISSED, because the miss is the lesson: (j)'s enumeration swept
# NUMPY reductions and generalised that to "every statistic in this file". The
# constraint channel counts in PLAIN PYTHON (`sum(1 for …)`), so the probe was
# structurally unable to see it. A sweep that sees one SYNTAX is not a
# population. Re-derived across three syntaxes with cross-syntax contrast
# controls (numpy=25, plain-Python=8, bare predicates=2).
#
# THE DEFECT. `_check_constraint_satisfied` is a bare `value >= threshold` /
# `value <= threshold` with no finiteness gate, and every count divided by the
# FULL `n_samples`. It is SIGN-SYMMETRIC, unlike Channel A: `+inf >= t` is True
# AND `-inf <= t` is True, so it could fabricate in both directions.
#
# WHY THE PRODUCER IS THE ONLY PLACE TO FIX IT. Every guard on the whole chain
# tests the DELIVERED VALUE and none tests the SAMPLE POPULATION — PLoT
# `prob01`, CEE's identical predicate and `z.number().min(0).max(1)`, UI
# `safeFiniteNumber`. An inflated 0.565 is finite and inside [0, 1] and passes
# all of them. `validity_ratio`/`n_valid_samples` exist but sit on the OUTCOME
# block and the constraint mapping never consults them.
#
# THESE TESTS BIND THE CONSUMER-VISIBLE CLAIM, NOT THE FLOAT. The two predicates
# below are the consumer's own, evaluated here against ISL's output:
#   BADGE  — PLoT fires "met every limit you set, in all the scenarios we
#            tested" on `values.every(p => p === 1)`.
#   BREACH — the constraint-breach warning fires on `p === 0`.
# ISL cannot render the badge; what it CAN do — and what these assert — is that
# its output makes those predicates come out right, in BOTH directions. A guard
# that only proved suppression would suppress everything and pass, so every
# suppression case here is paired with a twin in which the badge must still be
# EARNABLE and the warning must still FIRE.


def _badge_earned(option: dict) -> bool:
    """PLoT's crown-badge predicate, applied to ISL's output."""
    ca = option.get("constraint_analysis")
    if ca is None:
        return False
    return all(c["prob_satisfied"] == 1 for c in ca["constraints"])


def _breach_flagged(option: dict) -> bool:
    """The constraint-breach warning predicate, applied to ISL's output."""
    ca = option.get("constraint_analysis")
    if ca is None:
        return False
    return any(c["prob_satisfied"] == 0 for c in ca["constraints"])


def _with_constraint(payload: dict, operator: str, threshold: float) -> dict:
    payload["goal_constraints"] = [
        {
            "node_id": "goal_out",
            "operator": operator,
            "threshold": threshold,
            "label": "Stated limit",
            "value_frame": "delta",
        }
    ]
    return payload


def _all_finite_constraint_request(operator: str, threshold: float) -> dict:
    """Healthy graph: every draw finite, so the gate is a pure no-op here."""
    payload = _mixed_finite_and_infinite_request()
    for node in payload["graph"]["nodes"]:
        if node["id"] in ("f_a", "f_b"):
            node["observed_state"]["value"] = 0.3
    payload["graph"]["edges"][1]["exists_probability"] = 0.30
    payload["graph"]["edges"][2]["exists_probability"] = 0.30
    payload["options"][0]["id"] = "opt_x"
    payload["options"][1]["id"] = "opt_y"
    return _with_constraint(payload, operator, threshold)


class TestConstraintChannelIsGatedOnFiniteness:
    """2.477(k) — the constraint channel over the informative population."""

    def test_all_non_finite_option_refuses_the_whole_constraint_block(self, client, auth_headers):
        """No draw is informative, so the conjunction was never evaluated once.
        `prob_satisfied`/`joint_probability` are REQUIRED wire floats — there is
        no field to null — so per 2.798 the refusal unit is the BLOCK."""
        payload = _with_constraint(_goal_request(HUGE_FINITE, 1.0, 1.0), ">=", 1.0)
        resp = _post(client, auth_headers, payload)

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        degraded = _option(body, "opt_degraded")
        assert degraded["status"] == "failed"
        assert degraded["outcome"]["n_valid_samples"] == 0

        assert "constraint_analysis" not in degraded, (
            "an option with zero informative draws must OMIT the constraint "
            f"block, not fabricate one: {degraded.get('constraint_analysis')!r}"
        )
        # CONSUMER-VISIBLE: with no block there is nothing to badge.
        assert _badge_earned(degraded) is False

        # In-run control: the healthy sibling still gets a real block, so the
        # refusal is specific to the option that earned it.
        healthy = _option(body, "opt_healthy")
        assert healthy["constraint_analysis"] is not None
        assert _breach_flagged(healthy) is True

    def test_crownable_computed_option_counts_only_informative_draws(self, client, auth_headers):
        """The guard-defeating case: `status: "computed"`, 91.5% validity, so
        every downstream status allowlist admits it."""
        payload = _with_constraint(_computed_status_inflated_request(), ">=", 1.0)
        resp = _post(client, auth_headers, payload)

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        option = _option(body, "opt_computed")

        # PRECONDITION: genuinely mixed, and genuinely 'computed'.
        assert option["status"] == "computed"
        assert option["outcome"]["n_valid_samples"] == 183
        assert option["outcome"]["validity_ratio"] > 0.8

        ca = option["constraint_analysis"]
        prob = ca["constraints"][0]["prob_satisfied"]

        assert prob == pytest.approx(
            96 / 183
        ), f"prob_satisfied must use the 183 informative draws; got {prob!r}"
        assert prob != pytest.approx(
            113 / N_SAMPLES
        ), "prob_satisfied is still counting the +inf draws as satisfied"
        assert ca["joint_probability"] == pytest.approx(96 / 183)

        # CONSUMER-VISIBLE: neither badged nor breached, which is the truth.
        assert _badge_earned(option) is False
        assert _breach_flagged(option) is False

    def test_constraint_and_goal_channels_agree_on_the_same_question(self, client, auth_headers):
        """Same node, same threshold, same frame — so the two channels are
        answering ONE question and must return ONE number. At pristine they
        disagreed (0.565 vs 0.5246) and only the goal channel was fixed by (j);
        this pins that (k) closed the gap rather than moving it."""
        payload = _with_constraint(_computed_status_inflated_request(), ">=", 1.0)
        body = _strict_parse(_post(client, auth_headers, payload))

        option = _option(body, "opt_computed")
        pog = option["probability_of_goal"]
        prob_satisfied = option["constraint_analysis"]["constraints"][0]["prob_satisfied"]

        assert prob_satisfied == pytest.approx(pog), (
            "the constraint and goal channels ask the same question of the same "
            f"samples and must agree; got {prob_satisfied!r} vs {pog!r}"
        )

    def test_badge_is_still_earnable_on_a_healthy_run(self, client, auth_headers):
        """OPPOSITE-DIRECTION TWIN. A guard that only proved suppression would
        suppress everything and pass. On an all-finite run whose every draw
        satisfies the limit, the badge predicate must come out TRUE."""
        resp = _post(client, auth_headers, _all_finite_constraint_request(">=", -100.0))

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)

        for option_id in ("opt_x", "opt_y"):
            option = _option(body, option_id)
            # PRECONDITION: fully finite, so the gate has nothing to remove.
            assert option["outcome"]["n_valid_samples"] == N_SAMPLES
            assert (
                option["constraint_analysis"] is not None
            ), "a healthy run must still get a constraint block"
            assert option["constraint_analysis"]["constraints"][0][
                "prob_satisfied"
            ] == pytest.approx(1.0)
            assert _badge_earned(option) is True, (
                "the crown badge must remain EARNABLE — trading the lie for a " "gap is not a fix"
            )
            assert _breach_flagged(option) is False

    def test_breach_warning_still_fires_on_a_healthy_run(self, client, auth_headers):
        """OPPOSITE-DIRECTION TWIN. An unreachable limit on an all-finite run
        must still drive `prob_satisfied` to exactly 0, so the breach warning
        fires. Inflation is what used to lift it off zero and silence this."""
        resp = _post(client, auth_headers, _all_finite_constraint_request(">=", 99.0))

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)

        option = _option(body, "opt_x")
        assert option["outcome"]["n_valid_samples"] == N_SAMPLES
        assert option["constraint_analysis"]["constraints"][0]["prob_satisfied"] == pytest.approx(
            0.0
        )
        assert (
            _breach_flagged(option) is True
        ), "the constraint-breach warning must still fire on a genuine breach"
        assert _badge_earned(option) is False

    def test_extreme_finite_failure_margins_do_not_kill_the_response(self, client, auth_headers):
        """2.477(k), the (f)/(g) family again. `np.median` AVERAGES the two
        middle elements of an EVEN-length population, so two finite margins near
        float64 max sum to inf before the division — measured: 92 margins, all
        finite, max 1.696e308, median inf, render 500. The margin is omitted;
        the probability is unaffected."""
        payload = _with_constraint(_computed_status_inflated_request(), "<=", 1e300)
        resp = _post(client, auth_headers, payload)

        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)
        _assert_all_finite(body)

        row = _option(body, "opt_computed")["constraint_analysis"]["constraints"][0]
        # The probability survives — only the unrepresentable diagnostic is dropped.
        assert row["prob_satisfied"] == pytest.approx(87 / 183)
        assert (
            row.get("failure_margin_median") is None
        ), "a non-finite failure margin must be OMITTED, never rendered"
