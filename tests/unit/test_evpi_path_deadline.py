"""Codex F7 + F4 (ISL producer half): EVPI / path-decomposition INTERNAL
wall-clock deadline + typed degradation-warning severity + corrected field paths.

F7 — byte-confirmed defect at base 933c3404: `_compute_evpi` /
`_compute_evpi_metric` / `_compute_path_decomposition` were wall-clock ENTRY-gated
only. Once started they re-checked NOTHING, so EVPI could run (n_uncertainties+1)
full MC passes (<=2000 samples x N options) and path-decomposition could walk a
dense DAG UNBOUNDED — crossing OVERALL_REQUEST_BUDGET_MS (50s) and PLoT's 60s ISL
timeout and returning every row with NO disclosure. They now carry an INTERNAL
monotonic deadline (min(cap, remaining) measured against a phase t0, mirroring the
E-value / band sweeps in test_request_budget.py), degrade ALL-OR-NOTHING, and
disclose EVPI_UNAVAILABLE / PATH_DECOMPOSITION_UNAVAILABLE with elapsed_ms.

F4 (producer half): InferenceWarning now carries a typed `severity` (default
'warning') so PLoT/UI no longer default a MISSING severity to 'info' and hide the
disclosure; and the EVPI/path disclosures name the TOP-LEVEL envelope fields
`factor_evpi` / `path_decomposition` (were `robustness.factor_evpi` /
`robustness.path_decomposition`).

RED-first at base 933c3404: EVPI_BUDGET_MS / PATH_DECOMPOSITION_BUDGET_MS are
absent (monkeypatch.setattr raises), InferenceWarning has no `severity`
(AttributeError on `.severity`), and the entry-skip field paths are `robustness.*`.

Internal-trip idiom mirrors test_request_budget.py exactly: monkeypatch the phase
cap to -1 so the first internal `elapsed > budget` re-check trips immediately.

Positive control (trap-13): the SAME fixture under the real (ample) budget produces
EVPI rows and a path decomposition with NO *_UNAVAILABLE code — so the absence
assertions below are proven able to see a presence.
"""

import json
from pathlib import Path

import pytest

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS_PATH = REPO_ROOT / "tests" / "benchmarks" / "sample_variants.json"

EVPI_UNAVAILABLE = "EVPI_UNAVAILABLE"
PATH_DECOMPOSITION_UNAVAILABLE = "PATH_DECOMPOSITION_UNAVAILABLE"


def _variants() -> dict:
    return json.loads(VARIANTS_PATH.read_text())


def _request(idx: int = 2, seed: int = 42, *, voi: bool = True, path_decomp: bool = True):
    """Graph idx=2 = {price, demand, revenue}; options intervene on `price`
    (non-empty entry_nodes) and `demand` is a non-root chance node EVPI runs
    against (same fixture as tests/unit/test_request_budget.py)."""
    v = _variants()
    kwargs: dict = dict(
        request_id="evpi-path-deadline-test",
        graph=v["graphs"][idx],
        options=v["options"],
        goal_node_id=v["goal_node_id"],
        seed=seed,
        n_samples=v["n_samples"],
        include_e_values=False,
        include_voi=voi,
        include_path_decomposition=path_decomp,
    )
    if voi:
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


# ---------------------------------------------------------------------------
# Positive control — the absence assertions can see a presence
# ---------------------------------------------------------------------------


class TestEvpiPathPositiveControl:
    def test_ample_budget_phases_present_no_unavailability(self):
        resp = RobustnessAnalyzerV2().analyze(_request())
        assert resp.factor_evpi, "EVPI rows present under the full budget"
        assert resp.path_decomposition is not None, "path decomposition present under full budget"
        assert not (
            _codes(resp) & {EVPI_UNAVAILABLE, PATH_DECOMPOSITION_UNAVAILABLE}
        ), "no unavailability disclosed when nothing was squeezed"

    def test_deadline_guard_is_inert_when_not_tripped(self):
        """The periodic guard reads no RNG, so under the (untripped) default
        budget the EVPI rows are byte-identical across runs — the guard does not
        perturb output."""
        a = RobustnessAnalyzerV2().analyze(_request())
        b = RobustnessAnalyzerV2().analyze(_request())
        assert json.dumps(a.factor_evpi, sort_keys=True) == json.dumps(
            b.factor_evpi, sort_keys=True
        )


# ---------------------------------------------------------------------------
# F7 — EVPI internal deadline trip (all-or-nothing, disclosed)
# ---------------------------------------------------------------------------


class TestEvpiInternalDeadlineTrip:
    def test_evpi_internal_trip_all_or_nothing_and_discloses(self, monkeypatch):
        # Entry gate (remaining >= EVPI_MIN_BUDGET_MS) still passes: EVPI STARTS,
        # then the internal deadline (cap = -1) trips on the first re-check.
        monkeypatch.setattr(RobustnessAnalyzerV2, "EVPI_BUDGET_MS", -1)
        resp = RobustnessAnalyzerV2().analyze(_request())

        assert resp.results, "core MC results survive an EVPI-only internal trip"
        assert resp.robustness is not None, "core robustness survives"
        assert resp.factor_evpi is None, "all-or-nothing: NO partial EVPI rows on internal trip"

        w = _warn(resp, EVPI_UNAVAILABLE)
        assert w is not None, "EVPI_UNAVAILABLE must ride inference_warnings"
        assert isinstance(w.detail.get("elapsed_ms"), (int, float)), "carries elapsed_ms"
        assert w.detail.get("reason") == "evpi_budget_exceeded", "internal-trip reason"

    def test_evpi_disclosure_severity_and_top_level_field(self, monkeypatch):
        """F4: severity='warning' (was absent) + TOP-LEVEL field `factor_evpi`
        (was `robustness.factor_evpi`)."""
        monkeypatch.setattr(RobustnessAnalyzerV2, "EVPI_BUDGET_MS", -1)
        resp = RobustnessAnalyzerV2().analyze(_request())
        w = _warn(resp, EVPI_UNAVAILABLE)
        assert w is not None
        assert w.severity == "warning"
        assert w.field == "factor_evpi"


# ---------------------------------------------------------------------------
# F7 — path-decomposition internal deadline trip (all-or-nothing, disclosed)
# ---------------------------------------------------------------------------


class TestPathDecompositionInternalDeadlineTrip:
    def test_path_internal_trip_all_or_nothing_and_discloses(self, monkeypatch):
        monkeypatch.setattr(RobustnessAnalyzerV2, "PATH_DECOMPOSITION_BUDGET_MS", -1)
        resp = RobustnessAnalyzerV2().analyze(_request())

        assert resp.results, "core survives a path-only internal trip"
        assert resp.path_decomposition is None, "all-or-nothing: NO partial path decomposition"

        w = _warn(resp, PATH_DECOMPOSITION_UNAVAILABLE)
        assert w is not None, "PATH_DECOMPOSITION_UNAVAILABLE must ride the wire"
        assert isinstance(w.detail.get("elapsed_ms"), (int, float))
        assert w.detail.get("reason") == "path_decomposition_budget_exceeded"

    def test_path_disclosure_severity_and_top_level_field(self, monkeypatch):
        monkeypatch.setattr(RobustnessAnalyzerV2, "PATH_DECOMPOSITION_BUDGET_MS", -1)
        resp = RobustnessAnalyzerV2().analyze(_request())
        w = _warn(resp, PATH_DECOMPOSITION_UNAVAILABLE)
        assert w is not None
        assert w.severity == "warning"
        assert w.field == "path_decomposition"


# ---------------------------------------------------------------------------
# F4 — entry-skip disclosures also corrected (field + severity)
# ---------------------------------------------------------------------------


class TestEntrySkipDisclosuresCorrected:
    def test_entry_skip_fields_top_level_and_severity(self, monkeypatch):
        """Budget exhausted BEFORE the phases start (OVERALL=0): the entry-skip
        EVPI/path disclosures also name the TOP-LEVEL fields and carry
        severity='warning'."""
        monkeypatch.setattr(RobustnessAnalyzerV2, "OVERALL_REQUEST_BUDGET_MS", 0)
        resp = RobustnessAnalyzerV2().analyze(_request())
        we = _warn(resp, EVPI_UNAVAILABLE)
        wp = _warn(resp, PATH_DECOMPOSITION_UNAVAILABLE)
        assert we is not None and wp is not None
        assert we.field == "factor_evpi", "F4: top-level, not robustness.factor_evpi"
        assert wp.field == "path_decomposition", "F4: top-level, not robustness.path_decomposition"
        assert we.severity == "warning"
        assert wp.severity == "warning"


# ---------------------------------------------------------------------------
# F4 — the severity default is additive on the wire (survives model_dump)
# ---------------------------------------------------------------------------


class TestSeverityIsAdditiveOnWire:
    def test_severity_serialises_by_alias_exclude_none(self):
        """The new field has a non-None default so it always survives the
        envelope's model_dump(by_alias=True, exclude_none=True); older consumers
        ignore it (extra='ignore'). Proven on a real disclosure warning."""
        from src.models.response_v2 import InferenceWarning

        # Directly-constructed warnings default to the QUIET 'info' (the benign
        # diagnostics path). The non-None default always rides the wire.
        w_default = InferenceWarning(code="STRENGTH_MEAN_CLAMPED", field="edges[a→b]", detail={})
        assert w_default.severity == "info", "quiet default for benign diagnostics"
        dumped_default = w_default.model_dump(by_alias=True, exclude_none=True)
        assert dumped_default.get("severity") == "info", "severity rides the wire, not dropped"

        # An explicit 'warning' (the degradation-code path) also serialises.
        w_warn = InferenceWarning(
            code="EVPI_UNAVAILABLE", field="factor_evpi", detail={}, severity="warning"
        )
        dumped_warn = w_warn.model_dump(by_alias=True, exclude_none=True)
        assert dumped_warn.get("severity") == "warning"


# ---------------------------------------------------------------------------
# F4 — the severity MAPPING: exactly the 4 degradation codes are 'warning';
# the benign input-adjustment diagnostics stay 'info' (coordinator delta).
# ---------------------------------------------------------------------------

import copy  # noqa: E402

DEGRADATION_CODES = (
    "E_VALUES_UNAVAILABLE",
    "STABILITY_BANDS_UNAVAILABLE",
    "EVPI_UNAVAILABLE",
    "PATH_DECOMPOSITION_UNAVAILABLE",
)


def _request_clamped_edge_all_phases():
    """Graph idx=2 with edges[0].strength.mean forced out of [-1, 1] so parsing
    emits the benign STRENGTH_MEAN_CLAMPED, plus every optional phase enabled so a
    budget-exhausted run also emits all four degradation codes."""
    v = _variants()
    graph = copy.deepcopy(v["graphs"][2])
    graph["edges"][0]["strength"]["mean"] = 1000.0  # clamped -> STRENGTH_MEAN_CLAMPED
    return RobustnessRequestV2(
        request_id="severity-mapping-test",
        graph=graph,
        options=v["options"],
        goal_node_id=v["goal_node_id"],
        seed=42,
        n_samples=v["n_samples"],
        include_e_values=True,
        include_voi=True,
        include_path_decomposition=True,
        parameter_uncertainties=[{"node_id": "demand", "distribution": "normal", "std": 0.1}],
    )


class TestSeverityMappingDegradationVsBenign:
    def test_only_degradation_codes_are_warning_benign_stays_info(self, monkeypatch):
        # Budget exhausted -> all four optional phases entry-skip and disclose.
        monkeypatch.setattr(RobustnessAnalyzerV2, "OVERALL_REQUEST_BUDGET_MS", 0)
        resp = RobustnessAnalyzerV2().analyze(_request_clamped_edge_all_phases())

        # POSITIVE CONTROL: the four degradation codes AND the benign
        # STRENGTH_MEAN_CLAMPED are all present, so the severity assertions below
        # can see a presence (trap-13).
        codes = _codes(resp)
        for c in DEGRADATION_CODES:
            assert c in codes, f"positive control: {c} must be present"
        assert "STRENGTH_MEAN_CLAMPED" in codes, "positive control: benign code must be present"

        # The four degradation codes opt UP to 'warning'.
        for c in DEGRADATION_CODES:
            assert _warn(resp, c).severity == "warning", f"{c} must be 'warning'"

        # The benign input-adjustment diagnostic STAYS 'info' (the inversion the
        # default-'warning' code caused). RED-first: fails while the default is
        # 'warning'.
        assert (
            _warn(resp, "STRENGTH_MEAN_CLAMPED").severity == "info"
        ), "benign STRENGTH_MEAN_CLAMPED must stay 'info', not be stamped 'warning'"

        # Pin the mapping on the SERIALISED wire too.
        by_code = {
            w["code"]: w
            for w in resp.model_dump(by_alias=True, exclude_none=True)["inference_warnings"]
        }
        for c in DEGRADATION_CODES:
            assert by_code[c]["severity"] == "warning"
        assert by_code["STRENGTH_MEAN_CLAMPED"]["severity"] == "info"
