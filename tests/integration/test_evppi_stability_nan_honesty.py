"""Two non-finite-float defects that each destroy a whole 200 response (2.514).

Found in passing by the B2 lane (PR #126) and re-derived at the branch tip before
building. Both are the 2.477 family: a non-finite float reaching a validated or
serialised field kills the entire response — including every critique it was
carrying — instead of degrading the one statistic that could not be computed.

(a) ``elasticity_std = nan`` -> HTTP 500.
    ``_compute_bootstrap_stability`` reports ``np.std`` over a bootstrap
    elasticity population that can contain non-finite values. The field is
    ``ge=0``, so pydantic raises and the run 500s at
    ``robustness_analyzer_v2.py:4583``. Worse than the 500: ``nan`` also flows
    into ``classify_attribution_stability``, where every comparison against
    ``nan`` is False and the factor falls through to the confident label
    **"low"** — a fabricated classification, not a measurement.

(b) ``evppi_raw = nan`` ships, and ``evppi`` becomes a fabricated ``0.0``.
    ``evppi = max(0.0, est.evppi_raw)`` returns **0.0** for a nan raw because
    every nan comparison is False — indistinguishable on the wire from a real
    "learning this factor is worth nothing", with ``clamped_low=False``. And
    ``round(nan, 6)`` is still nan, so ``evppi_raw`` reaches the body and the
    JSONResponse render dies with ``ValueError: Out of range float values are
    not JSON compliant`` (``api/robustness.py:1556``).

Both degrade to ABSENCE, never to a different plausible number. (b) joins the
EVPPI block's EXISTING in-loop drop machinery (``FACTOR_EVPPI_PARTIAL``) rather
than inventing a parallel disclosure.

Every absence assertion is preceded by a PRESENCE the same harness can see
(trap #13) — for (b) the control's EVPPI is a real NON-ZERO number, so a harness
that could only ever produce zero would be caught. Assertions bind by IDENTITY
(option id, factor id, critique/warning code), never by a value predicate a
sibling could satisfy (trap #19). Bodies are parsed with ``parse_constant``
because plain ``json.loads`` ACCEPTS ``NaN``/``Infinity`` tokens and therefore
cannot witness JSON compliance.
"""

import json
import math
import os

import pytest

from fastapi.testclient import TestClient

from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

from src.api.main import app

HUGE_FINITE = 1.7e308
N_SAMPLES = 200
SEED = 42
ENDPOINT = "/api/v1/robustness/analyze/v2?response_version=2"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    if os.environ.get("ISL_AUTH_DISABLED", "").lower() == "true":
        return {}
    return {"X-API-Key": os.environ.get("ISL_API_KEY", "test_key")}


def _strict_parse(response):
    """json.loads ACCEPTS NaN/Infinity tokens; this refuses them, so the test can
    actually witness JSON compliance instead of assuming it."""

    def _reject(token):
        raise AssertionError(f"non-JSON-compliant float token in body: {token}")

    return json.loads(response.text, parse_constant=_reject)


def _graph(include_dead_option: bool, dead_first: bool = False):
    """`f_huge_a` + `f_huge_b` both land on the goal at ~float64 max, so their sum
    overflows on every draw. An option that does NOT intervene on both is dead
    (every draw non-finite). `f_info` is the non-lever uncertain factor that EVPPI
    and factor-sensitivity are computed for; no option intervenes on it.
    """
    nodes = [
        {"id": "f_huge_a", "kind": "factor", "label": "HA", "observed_state": {"value": HUGE_FINITE}},
        {"id": "f_huge_b", "kind": "factor", "label": "HB", "observed_state": {"value": HUGE_FINITE}},
        {"id": "f_x", "kind": "factor", "label": "X", "observed_state": {"value": 1.0}},
        {"id": "f_y", "kind": "factor", "label": "Y", "observed_state": {"value": 1.0}},
        {"id": "f_info", "kind": "factor", "label": "INFO", "observed_state": {"value": 1.0}},
        {"id": "goal_out", "kind": "outcome", "label": "Goal"},
    ]
    edges = [
        {"from": "f_huge_a", "to": "goal_out", "exists_probability": 1.0, "strength": {"mean": 1.0, "std": 0.05}},
        {"from": "f_huge_b", "to": "goal_out", "exists_probability": 1.0, "strength": {"mean": 1.0, "std": 0.05}},
        {"from": "f_x", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": 3.0, "std": 0.5}},
        {"from": "f_y", "to": "goal_out", "exists_probability": 0.5, "strength": {"mean": 3.0, "std": 0.5}},
        {"from": "f_info", "to": "goal_out", "exists_probability": 1.0, "strength": {"mean": 2.0, "std": 0.3}},
    ]
    healthy = [
        {"id": "opt_x", "label": "Push X",
         "interventions": {"f_huge_a": 0.5, "f_huge_b": 0.5, "f_x": 5.0, "f_y": 0.0}},
        {"id": "opt_y", "label": "Push Y",
         "interventions": {"f_huge_a": 0.5, "f_huge_b": 0.5, "f_x": 0.0, "f_y": 5.0}},
    ]
    dead = {"id": "opt_dead", "label": "Dead", "interventions": {"f_x": 1.0}}
    if not include_dead_option:
        options = healthy
    elif dead_first:
        # The dead option is the factor-sensitivity REFERENCE option — this is
        # what drives defect (a).
        options = [dead] + healthy
    else:
        options = healthy + [dead]

    return {
        "graph": {"nodes": nodes, "edges": edges},
        "options": options,
        "goal_node_id": "goal_out",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "parameter_uncertainties": [{"node_id": "f_info", "distribution": "normal", "std": 2.0}],
        "include_voi": True,
    }


def _post(client, headers, payload):
    return client.post(ENDPOINT, json=payload, headers=headers)


def _factor_sensitivity_row(body, node_id):
    for row in body.get("factor_sensitivity") or []:
        if row.get("node_id") == node_id:
            return row
    return None


# ============================================================ shared control


class TestTheHarnessCanSeeAPresence:
    """THE positive control for every absence assertion in this file (trap #13).

    The control request is the degraded one MINUS the all-non-finite option. If
    this harness cannot reproduce these two numbers, it cannot see a presence and
    nothing it says about an absence counts. Both are pinned to the values
    measured on the untouched control path, so a harness that silently stopped
    computing EVPI/EVPPI would fail here rather than passing every absence test.
    """

    def test_control_reproduces_the_reference_decision_evpi_and_evppi(
        self, client, auth_headers
    ):
        resp = _post(client, auth_headers, _graph(include_dead_option=False))
        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)

        assert body.get("decision_evpi") == pytest.approx(0.9908, abs=1e-4), (
            f"control decision_evpi drifted from the reference 0.9908; "
            f"got {body.get('decision_evpi')!r}"
        )
        rows = {r["factor_id"]: r["evppi"] for r in (body.get("factor_evppi") or [])}
        assert rows.get("f_info") == pytest.approx(0.0497, abs=1e-4), (
            f"control factor_evppi[f_info] drifted from the reference 0.0497; "
            f"got {rows!r}"
        )


# ============================================================ (a)


class TestNonFiniteBootstrapStabilityDegradesInsteadOf500:
    """2.514(a) — a non-finite bootstrap std must make the STABILITY SUMMARY
    absent, not kill the response."""

    def test_healthy_run_reports_a_present_stability_summary(self, client, auth_headers):
        """POSITIVE CONTROL: the harness can see a PRESENT stability summary.
        Without this, the absence assertions below would pass on a harness that
        never produces one."""
        resp = _post(client, auth_headers, _graph(include_dead_option=False))
        assert resp.status_code == 200, resp.text[:600]
        body = _strict_parse(resp)

        row = _factor_sensitivity_row(body, "f_info")
        assert row is not None, "control must produce a factor_sensitivity row for f_info"
        assert row.get("elasticity_std") is not None, (
            "positive control failed: a healthy run must report elasticity_std"
        )
        assert math.isfinite(row["elasticity_std"])
        assert row.get("attribution_stability") in {"high", "moderate", "low", "negligible"}

    def test_dead_reference_option_ships_200_with_stability_absent(self, client, auth_headers):
        """The defect: this request used to be an HTTP 500 that took every
        critique with it."""
        resp = _post(client, auth_headers, _graph(include_dead_option=True, dead_first=True))

        assert resp.status_code == 200, (
            f"a degraded option must not 500 the whole analysis; got "
            f"{resp.status_code}: {resp.text[:400]}"
        )
        body = _strict_parse(resp)  # also witnesses JSON compliance

        # `elasticity` is a REQUIRED float and could not be computed here (the
        # reference option's outcomes are non-finite), so the honest shape is NO
        # ROW — there is no field to null. Asserted unconditionally and bound to
        # f_info BY ID: an `if row is not None:` wrapper would let this pass
        # vacuously the moment the row disappeared, which is exactly what the fix
        # makes happen (trap 13b — a guard agreeing with itself).
        assert _factor_sensitivity_row(body, "f_info") is None, (
            "a factor whose elasticity is non-finite must be OMITTED, not emitted "
            "with a nan that kills the render"
        )
        # Whatever rows DO survive must carry only real numbers — no part-null,
        # no fabricated 'low' stability label from a nan CV.
        for row in body.get("factor_sensitivity") or []:
            assert math.isfinite(row["elasticity"]), f"{row['node_id']}: nan elasticity survived"
            if row.get("elasticity_std") is not None:
                assert math.isfinite(row["elasticity_std"])
                assert row.get("attribution_stability") in {
                    "high", "moderate", "low", "negligible",
                }, "a stability CLASS may only accompany a real std"
            else:
                assert row.get("attribution_stability") is None, (
                    f"{row['node_id']}: stability CLASS present without a std — a "
                    f"nan CV falls through to the confident label 'low', which is a "
                    f"fabricated classification, not a measurement"
                )

        # The critique that used to die inside the 500 it caused — bound to the
        # dead option BY ID.
        failed = [c for c in body.get("critiques", []) if c["code"] == "MONTE_CARLO_FAILED"]
        assert len(failed) == 1, f"expected one MONTE_CARLO_FAILED, got {failed!r}"
        assert failed[0]["affected_option_ids"] == ["opt_dead"]


# ============================================================ (b)


class TestNonFiniteEvppiIsDroppedNotFabricatedZero:
    """2.514(b) — a nan EVPPI estimate is ABSENT (with disclosure), never 0.0."""

    def test_control_produces_a_real_NON_ZERO_evppi(self):
        """POSITIVE CONTROL, and the one that matters most here: the harness can
        produce a genuinely non-zero EVPPI. An absence/zero assertion measured on
        a harness that only ever yields 0.0 would be testing nothing."""
        response = RobustnessAnalyzerV2().analyze(
            RobustnessRequestV2.model_validate(_graph(include_dead_option=False))
        )
        rows = {r["factor_id"]: r for r in (response.factor_evppi or [])}
        assert "f_info" in rows, "control must compute an EVPPI row for f_info"
        assert math.isfinite(rows["f_info"]["evppi_raw"])
        assert rows["f_info"]["evppi"] > 0.0, (
            f"positive control failed: the control EVPPI must be strictly "
            f"positive, else this harness cannot distinguish absence from zero; "
            f"got {rows['f_info']['evppi']!r}"
        )

    def test_non_finite_estimate_omits_the_factor_instead_of_emitting_zero(self):
        response = RobustnessAnalyzerV2().analyze(
            RobustnessRequestV2.model_validate(_graph(include_dead_option=True))
        )
        rows = {r["factor_id"]: r for r in (response.factor_evppi or [])}

        assert "f_info" not in rows, (
            f"a non-finite EVPPI estimate must OMIT the factor (missing != zero); "
            f"got {rows.get('f_info')!r}"
        )
        # And no surviving row anywhere carries a non-finite number.
        for fid, row in rows.items():
            assert math.isfinite(row["evppi"]), f"{fid}: evppi non-finite"
            assert math.isfinite(row["evppi_raw"]), f"{fid}: evppi_raw non-finite"

    def test_the_drop_is_DISCLOSED_through_the_existing_partial_warning(self):
        """Absence must be visible, not silent — and must reuse the block's own
        disclosure rather than a parallel scheme."""
        response = RobustnessAnalyzerV2().analyze(
            RobustnessRequestV2.model_validate(_graph(include_dead_option=True))
        )
        warnings = [
            w for w in (response.inference_warnings or [])
            if getattr(w, "code", None) == "FACTOR_EVPPI_PARTIAL"
        ]
        assert len(warnings) == 1, (
            f"the dropped factor must be disclosed via FACTOR_EVPPI_PARTIAL; "
            f"got {[getattr(w, 'code', None) for w in (response.inference_warnings or [])]!r}"
        )
        detail = warnings[0].detail or {}
        failed_ids = detail.get("failed_factor_ids") or []
        assert "f_info" in failed_ids, (
            f"the disclosure must name the dropped factor BY ID; got {failed_ids!r}"
        )
        categories = {
            f["factor_id"]: f["category"] for f in (detail.get("failures") or [])
        }
        assert categories.get("f_info") == "non_finite_estimate", (
            f"the drop must be categorised as a non-finite ESTIMATE (not conflated "
            f"with estimator_error or non_finite_theta); got {categories!r}"
        )

    def test_endpoint_ships_200_and_json_compliant_with_a_dead_option(
        self, client, auth_headers
    ):
        """The render-level half: `evppi_raw = nan` used to reach the body and
        kill the response at JSONResponse render."""
        resp = _post(client, auth_headers, _graph(include_dead_option=True))
        assert resp.status_code == 200, (
            f"a nan EVPPI estimate must not 500 the response; got "
            f"{resp.status_code}: {resp.text[:400]}"
        )
        body = _strict_parse(resp)  # raises if any NaN/Infinity token is present
        for row in body.get("factor_evppi") or []:
            assert row["factor_id"] != "f_info"
