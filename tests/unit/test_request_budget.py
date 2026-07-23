"""Governing request budget + degrade-and-disclose + monotonic guards.

A3 ISL remediation (2026-07-18). Spec:
acceptance-evidence/a3-verify-2026-07-16/REMEDIATION-SPEC.md (ISL LANE + ALTITUDE).

Context problem (xhigh #1 / ALTITUDE hunt 1): ISL runs base MC + e-values(8s) +
bands(30s) + EVPI + path-decomposition SEQUENTIALLY with NO governing budget — only a
90s route guillotine (src/api/main.py / middleware/request_limits.py) that HARD-KILLS
and loses ALL results. PLoT's caller timeout (60s) sits BELOW ISL's 90s route cap, so
stacked optional phases orphan compute: the whole analysis is lost, silently.

FIX: a governing OVERALL_REQUEST_BUDGET_MS (~50s, below PLoT's 60s) that the OPTIONAL
phases (e-values, bands, EVPI, path-decomposition) check remaining-budget against and
SKIP-WITH-DISCLOSURE when insufficient. Base MC + core robustness are NEVER gated —
always returned. Degrade-within, never guillotine-lose.

HONESTY: every skip/trip rides the wire via inference_warnings
(E_VALUES_UNAVAILABLE / STABILITY_BANDS_UNAVAILABLE / EVPI_UNAVAILABLE /
PATH_DECOMPOSITION_UNAVAILABLE), each carrying elapsed_ms — mirroring the LOG-only
e_value_budget_exceeded / flip_stability_budget_exceeded events. This is the gap #226
closed for flip_thresholds, now closed for the whole optional-phase family.

RED-first: at base tip ad01c08c none of these exist — OVERALL_REQUEST_BUDGET_MS is
absent (monkeypatch raises) and no *_UNAVAILABLE wire code is emitted.
"""

import inspect
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS_PATH = REPO_ROOT / "tests" / "benchmarks" / "sample_variants.json"
INPLACE_GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "flip_stability" / "inplace_bg_golden.json"

ENDPOINT = "/api/v1/robustness/analyze/v2"
V2_HEADERS = {"X-ISL-Response-Version": "2"}

# PLoT's ISL_TIMEOUT_MS. ISL must return BELOW this so PLoT sees a clean 200-with-partial
# rather than aborting mid-compute (ALTITUDE hunt 1 — the inverted timeout order).
PLOT_ISL_TIMEOUT_MS = 60000

E_VALUES_UNAVAILABLE = "E_VALUES_UNAVAILABLE"
STABILITY_BANDS_UNAVAILABLE = "STABILITY_BANDS_UNAVAILABLE"
EVPI_UNAVAILABLE = "EVPI_UNAVAILABLE"
PATH_DECOMPOSITION_UNAVAILABLE = "PATH_DECOMPOSITION_UNAVAILABLE"


def _variants() -> dict:
    return json.loads(VARIANTS_PATH.read_text())


def _request(
    idx: int = 2,
    seed: int = 42,
    *,
    e_values: bool = True,
    voi: bool = False,
    path_decomp: bool = False,
) -> RobustnessRequestV2:
    v = _variants()
    kwargs: dict = dict(
        request_id="budget-test",
        graph=v["graphs"][idx],
        options=v["options"],
        goal_node_id=v["goal_node_id"],
        seed=seed,
        n_samples=v["n_samples"],
        include_e_values=e_values,
        include_voi=voi,
        include_path_decomposition=path_decomp,
    )
    if voi:
        # 'demand' is a non-root chance node — EVPI runs against it (probed 07-18).
        kwargs["parameter_uncertainties"] = [
            {"node_id": "demand", "distribution": "normal", "std": 0.1}
        ]
    return RobustnessRequestV2(**kwargs)


def _codes(resp) -> set:
    return {w.code for w in resp.inference_warnings}


def _warn(resp, code):
    for w in resp.inference_warnings:
        if w.code == code:
            return w
    return None


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Value pins — the governing budget itself
# ---------------------------------------------------------------------------


class TestBudgetValuePins:
    def test_overall_budget_below_plot_timeout(self):
        """The whole point: ISL self-limits BELOW PLoT's 60s so it returns a
        clean partial instead of being aborted mid-compute (ALTITUDE hunt 1)."""
        assert RobustnessAnalyzerV2.OVERALL_REQUEST_BUDGET_MS < PLOT_ISL_TIMEOUT_MS
        # A meaningful margin (>= ~5s) for the base-call round trip + PLoT overhead.
        assert RobustnessAnalyzerV2.OVERALL_REQUEST_BUDGET_MS <= 55000

    def test_overall_budget_pinned_at_50000(self):
        assert RobustnessAnalyzerV2.OVERALL_REQUEST_BUDGET_MS == 50000


# ---------------------------------------------------------------------------
# Item 1 + 2 — degrade-within + wire disclosure (behaviour-adjacent, RED-first)
# ---------------------------------------------------------------------------


class TestDegradeAndDisclose:
    def test_positive_control_full_budget_all_optional_phases_present(self):
        """POSITIVE CONTROL (trap-13): under the real budget every optional
        phase is present and NO *_UNAVAILABLE warning fires — so the absence
        assertions below are proven able to see a presence."""
        resp = RobustnessAnalyzerV2().analyze(_request(voi=True, path_decomp=True))
        assert resp.edge_e_values, "e-values should be present under full budget"
        assert any(e.get("stability") for e in resp.edge_e_values), "bands present"
        assert resp.p_win_sensitivity, "EVPI present under full budget"
        assert resp.path_decomposition is not None, "path decomposition present"
        assert not (
            _codes(resp)
            & {
                E_VALUES_UNAVAILABLE,
                STABILITY_BANDS_UNAVAILABLE,
                EVPI_UNAVAILABLE,
                PATH_DECOMPOSITION_UNAVAILABLE,
            }
        ), "no unavailability disclosed when nothing was squeezed"

    def test_budget_pressure_keeps_core_skips_and_discloses_optional(self, monkeypatch):
        """Item 1: induce budget pressure (budget exhausted before optional
        phases). (a) base + robustness PRESENT, (b) each optional phase ABSENT
        and DISCLOSED with elapsed_ms, (c) no exception (the analyzer returns)."""
        monkeypatch.setattr(RobustnessAnalyzerV2, "OVERALL_REQUEST_BUDGET_MS", 0)
        resp = RobustnessAnalyzerV2().analyze(_request(voi=True, path_decomp=True))

        # (a) Core ALWAYS survives — never guillotine-lost.
        assert resp.results, "base MC option results must survive budget pressure"
        assert resp.robustness is not None, "core robustness must survive budget pressure"
        assert resp.recommended_option_id, "recommendation must survive"

        # (b) Every optional phase is absent AND disclosed on the wire, each
        # carrying elapsed_ms. The reason is precise: e-values / EVPI / path are
        # skipped directly by the exhausted budget; bands ride on the e-value
        # sweep, so their absence is honestly attributed to e_values_unavailable.
        assert resp.edge_e_values is None
        assert resp.p_win_sensitivity is None
        assert resp.path_decomposition is None
        expected_reason = {
            E_VALUES_UNAVAILABLE: "request_budget_exhausted",
            STABILITY_BANDS_UNAVAILABLE: "e_values_unavailable",
            EVPI_UNAVAILABLE: "request_budget_exhausted",
            PATH_DECOMPOSITION_UNAVAILABLE: "request_budget_exhausted",
        }
        for code, reason in expected_reason.items():
            w = _warn(resp, code)
            assert w is not None, f"{code} must be disclosed on inference_warnings"
            assert isinstance(
                w.detail.get("elapsed_ms"), (int, float)
            ), f"{code} must carry elapsed_ms"
            assert w.detail.get("reason") == reason, f"{code} reason"

    def test_evalues_internal_budget_trip_rides_the_wire(self, monkeypatch):
        """Item 2: the e-value INTERNAL budget trip (formerly log-only) now
        also emits E_VALUES_UNAVAILABLE + STABILITY_BANDS_UNAVAILABLE (bands
        ride on e-values). Base results untouched."""
        monkeypatch.setattr(RobustnessAnalyzerV2, "E_VALUE_BUDGET_MS", -1)
        resp = RobustnessAnalyzerV2().analyze(_request())
        assert resp.results, "core survives"
        assert resp.edge_e_values is None, "e-values omitted on internal budget trip"
        assert E_VALUES_UNAVAILABLE in _codes(resp)
        assert STABILITY_BANDS_UNAVAILABLE in _codes(resp)
        assert isinstance(_warn(resp, E_VALUES_UNAVAILABLE).detail.get("elapsed_ms"), (int, float))

    def test_bands_internal_budget_trip_rides_the_wire(self, monkeypatch):
        """Item 2: the band INTERNAL budget trip (the #226 gap for
        flip_thresholds, now for bands) emits STABILITY_BANDS_UNAVAILABLE while
        the base edge_e_values survive untouched — all-or-nothing, disclosed."""
        monkeypatch.setattr(RobustnessAnalyzerV2, "FLIP_STABILITY_BUDGET_MS", -1)
        resp = RobustnessAnalyzerV2().analyze(_request())
        assert resp.edge_e_values, "base e-values survive a band-only trip"
        for entry in resp.edge_e_values:
            assert "stability" not in entry, "all-or-nothing: no partial bands"
        w = _warn(resp, STABILITY_BANDS_UNAVAILABLE)
        assert w is not None, "band trip must ride the wire"
        assert isinstance(w.detail.get("elapsed_ms"), (int, float))
        # e-values themselves are fine — no E_VALUES_UNAVAILABLE here.
        assert E_VALUES_UNAVAILABLE not in _codes(resp)

    def test_wire_200_with_disclosure_under_pressure(self, client, monkeypatch):
        """Item 1 (c) end-to-end: HTTP 200 (NOT a 504 guillotine) with the
        disclosure codes on the JSON wire under budget pressure."""
        monkeypatch.setattr(RobustnessAnalyzerV2, "OVERALL_REQUEST_BUDGET_MS", 0)
        v = _variants()
        body = {
            "request_id": "budget-wire",
            "graph": v["graphs"][2],
            "options": v["options"],
            "goal_node_id": v["goal_node_id"],
            "seed": 42,
            "n_samples": v["n_samples"],
            "include_e_values": True,
        }
        resp = client.post(ENDPOINT, json=body, headers=V2_HEADERS)
        assert resp.status_code == 200, resp.text
        blob = json.dumps(resp.json())
        assert E_VALUES_UNAVAILABLE in blob
        assert STABILITY_BANDS_UNAVAILABLE in blob


# ---------------------------------------------------------------------------
# Item 3 — monotonic clock in the budget guards
# ---------------------------------------------------------------------------


class TestMonotonicGuards:
    """The band/e-value budget guards must measure with time.monotonic(), not
    time.time() (an NTP step can move wall-clock backward and corrupt a
    wall-clock elapsed guard). Source-derived pin: the guard math in these two
    methods uses monotonic and does NOT use time.time(). Mutation-checked
    (revert monotonic->time.time turns this RED)."""

    @pytest.mark.parametrize(
        "method",
        [
            RobustnessAnalyzerV2._compute_edge_e_values,
            RobustnessAnalyzerV2._attach_flip_stability_bands,
        ],
    )
    def test_guard_uses_monotonic_not_walltime(self, method):
        src = inspect.getsource(method)
        assert "time.monotonic()" in src, f"{method.__name__} must use time.monotonic()"
        assert (
            "time.time()" not in src
        ), f"{method.__name__} budget guard must not use wall-clock time.time()"


# ---------------------------------------------------------------------------
# Item 4 — in-place background mutation stays byte-identical (determinism)
# ---------------------------------------------------------------------------


class TestInPlaceBackgroundDeterminism:
    """_flip_mean_under_background drops the per-bisection dict(baseline_config)
    copy in favour of mutate-single-key / evaluate / restore-in-finally. This
    MUST be byte-identical to the copy-based base code. The golden was captured
    from BASE (pre-change) code, so equality is a genuine before/after proof."""

    def test_bands_match_base_golden_byte_for_byte(self):
        golden = json.loads(INPLACE_GOLDEN_PATH.read_text())
        for idx in range(len(_variants()["graphs"])):
            resp = RobustnessAnalyzerV2().analyze(_request(idx=idx))
            blocks = [e.get("stability") for e in (resp.edge_e_values or [])]
            assert json.dumps(blocks, sort_keys=True) == json.dumps(
                golden[str(idx)], sort_keys=True
            ), f"in-place mutation changed band output for graph {idx}"

    def test_same_seed_twice_byte_identical(self):
        a = RobustnessAnalyzerV2().analyze(_request())
        b = RobustnessAnalyzerV2().analyze(_request())
        blocks_a = [e.get("stability") for e in (a.edge_e_values or [])]
        blocks_b = [e.get("stability") for e in (b.edge_e_values or [])]
        assert json.dumps(blocks_a, sort_keys=True) == json.dumps(blocks_b, sort_keys=True)

    def test_flip_search_does_not_mutate_background(self):
        """Restore-in-finally correctness: the shared background dict handed to
        _flip_mean_under_background must be unchanged after the search (CRN:
        the same background is reused across all edges within a seed)."""
        from src.services.robustness_analyzer_v2 import (
            DualUncertaintySampler,
            SCMEvaluatorV2,
            SeededRNG,
        )

        req = _request()
        evaluator = SCMEvaluatorV2(req.graph)
        evaluator._epsilon_rng = None
        sampler = DualUncertaintySampler(req.graph.edges, SeededRNG(123))
        background = sampler.sample_edge_configuration()
        before = dict(background)
        analyzer = RobustnessAnalyzerV2()
        for edge in req.graph.edges:
            analyzer._flip_mean_under_background(req, evaluator, edge, background)
        assert background == before, "background dict must be restored (not mutated)"
