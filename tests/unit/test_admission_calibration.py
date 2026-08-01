"""Weighted compute-admission cost model — pins + positive controls (Codex F8).

Covers the F8.5 test plan for the cost model half:
  * base-term correctness against the design's worked table,
  * PC-2 (option repricing must flip): dense-mid admits at 1 option, rejects at
    10 options + EVPI (the abusive multi-option x multi-EVPI combo),
  * PC-3 (shape fix must NOT regress): a deep single graph the OLD scalar 422'd
    (48M > 30M) now ADMITS,
  * ceiling + weight pins so a silent revert turns RED (mirrors
    test_lenient_limits.py's silent-revert guard doctrine).

The CEILING is PROVISIONAL (local-hardware calibration; staging recalibration
owed). This file pins the SHIPPED provisional value: changing it must be a
conscious edit, surfaced in review, not a silent drift.
"""
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import (
    BASE_COST_COEF,
    COMPLEXITY_FORMULA_VERSION,
    DEFAULT_MAX_COST_UNITS,
    EVPI_SAMPLE_CAP,
    FLIP_STABILITY_N_SEEDS,
    PHASE_COST_ATTRIBUTION,
    SENSITIVITY_SUBSAMPLE_CAP,
    SENSITIVITY_SUBSAMPLE_DIVISOR,
    RobustnessAnalyzerV2,
    W_BANDS_COEF,
    W_EVAL_COEF,
    W_EVPC_COEF,
    W_EVPPI_COEF,
    W_PATH_COEF,
    W_SENS_COEF,
    build_compute_admission,
    compute_weighted_cost,
    get_max_cost_units,
)
from src.utils.evppi import REGRESSION_EVPPI_NULL_PERMUTATIONS

ENDPOINT = "/api/v1/robustness/analyze/v2"


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------
def _graph(n_nodes: int, n_edges: int, evpi_factors: int = 0) -> dict:
    """A forward DAG with a reachable goal: a n0->...->n_last chain (so the goal
    is reachable) plus forward skip edges up to n_edges. Requires n_edges >=
    n_nodes-1. Valid: no cycles/self-loops/duplicate edges."""
    nodes = [{"id": f"n{i}", "kind": "factor", "label": f"N{i}"} for i in range(n_nodes)]
    nodes[-1]["kind"] = "outcome"
    for k in range(min(evpi_factors, n_nodes - 1)):
        nodes[k]["observed_state"] = {"value": 50.0, "std": 5.0}

    def _edge(i: int, j: int) -> dict:
        return {
            "from": f"n{i}",
            "to": f"n{j}",
            "exists_probability": 0.9,
            "strength": {"mean": 0.3, "std": 0.1},
        }

    edges: list = []
    seen: set = set()
    for i in range(n_nodes - 1):  # reachability chain
        if len(edges) >= n_edges:
            break
        edges.append(_edge(i, i + 1))
        seen.add((i, i + 1))
    for j in range(2, n_nodes):  # fill remaining budget
        for i in range(j - 1):
            if len(edges) >= n_edges:
                break
            if (i, j) not in seen:
                edges.append(_edge(i, j))
                seen.add((i, j))
        if len(edges) >= n_edges:
            break
    return {"nodes": nodes, "edges": edges}


def _request_dict(
    n_nodes, n_edges, n_samples, n_options, *, evpi_factors=0, sensitivity=True
) -> dict:
    g = _graph(n_nodes, n_edges, evpi_factors)
    opts = [
        {"id": f"o{k}", "label": f"O{k}", "interventions": {"n0": 0.1 * (k + 1)}}
        for k in range(n_options)
    ]
    types = ["comparison", "robustness"] + (["sensitivity"] if sensitivity else [])
    body = {
        "graph": g,
        "options": opts,
        "goal_node_id": f"n{n_nodes - 1}",
        "n_samples": n_samples,
        "seed": 7,
        "analysis_types": types,
    }
    if evpi_factors:
        body["include_voi"] = True
        body["parameter_uncertainties"] = [
            {"node_id": f"n{k}", "distribution": "normal", "std": 5.0} for k in range(evpi_factors)
        ]
    return body


def _cost(n_nodes, n_edges, n_samples, n_options, **kw) -> int:
    req = RobustnessRequestV2(**_request_dict(n_nodes, n_edges, n_samples, n_options, **kw))
    return compute_weighted_cost(req).total


# ---------------------------------------------------------------------------
# Base-term + shape correctness
# ---------------------------------------------------------------------------
class TestWeightedCostShape:
    def test_base_term_is_S_times_O_times_W(self):
        """base_mc = n_samples * n_options * (n_nodes + n_edges) — sensitivity off."""
        req = RobustnessRequestV2(**_request_dict(5, 8, 1000, 2, sensitivity=False))
        wc = compute_weighted_cost(req)
        assert wc.terms["base_mc"] == 1000 * 2 * (5 + 8)  # = 26,000
        assert "sensitivity" not in wc.terms  # sensitivity not requested
        assert wc.total == 26000

    def test_option_multiplier_present(self):
        """The pre-F8 defect: cost must scale ~linearly with option count."""
        one = _cost(12, 30, 5000, 1, sensitivity=False)
        ten = _cost(12, 30, 5000, 10, sensitivity=False)
        assert ten == 10 * one  # base term is exactly O-linear

    def test_shape_uses_N_plus_E_not_N_times_E(self):
        """Deep single graph is NOT priced by n_nodes*n_edges (the old shape)."""
        req = RobustnessRequestV2(**_request_dict(40, 120, 10000, 1, sensitivity=False))
        wc = compute_weighted_cost(req)
        # correct shape: S*O*(N+E) = 10000*1*160 = 1.6M  (old scalar was 48M)
        assert wc.terms["base_mc"] == 10000 * 1 * (40 + 120)

    def test_evpi_term_uses_unique_factor_count_and_sample_cap(self):
        req = RobustnessRequestV2(**_request_dict(40, 120, 5000, 10, evpi_factors=5))
        wc = compute_weighted_cost(req)
        # (U+1) * min(S, cap) * O * W = 6 * 2000 * 10 * 160 (p_win term, sample-capped).
        assert wc.terms["evpi"] == 6 * min(5000, EVPI_SAMPLE_CAP) * 10 * 160
        # D-23.12: the S2 EVPPI regression is a SEPARATE term on the FULL S (not the
        # 2000 cap). v4 (OC-1, D-23.17): recalibrated to U * (1+K) * S — no O factor
        # (one multi-RHS SVD shared across options), no (deg+1): 1 * 5 * 17 * 5000 = 425k.
        assert wc.terms["evppi_full"] == (
            W_EVPPI_COEF * 5 * (1 + REGRESSION_EVPPI_NULL_PERMUTATIONS) * 5000
        )
        # v4 GOLDEN CHANGED (OC-1): the recalibrated evppi_full (425k) no longer
        # dwarfs the sample-capped p_win term (19.2M) — 'evpi' is dominant again
        # (as it was pre-D-23.12).
        assert wc.dominant_term == "evpi"

    def test_evalues_add_evalue_and_bands_terms(self):
        base = _request_dict(12, 40, 3000, 3, sensitivity=False)
        base["include_e_values"] = True
        wc = compute_weighted_cost(RobustnessRequestV2(**base))
        assert wc.terms["e_values"] == W_EVAL_COEF * 40 * 3
        assert wc.terms["bands"] == W_BANDS_COEF * 40 * 3


# ---------------------------------------------------------------------------
# PC-2 — option repricing must flip
# ---------------------------------------------------------------------------
class TestPC2OptionRepricing:
    def test_dense_mid_one_option_admits(self):
        cost = _cost(40, 120, 5000, 1)
        assert cost <= get_max_cost_units(), f"1-option dense-mid must admit (cost={cost:,})"

    def test_dense_mid_ten_options_plus_evpi_rejects(self):
        cost = _cost(40, 120, 5000, 10, evpi_factors=5)
        assert (
            cost > get_max_cost_units()
        ), f"10-option + EVPI dense-mid must reject (cost={cost:,})"

    def test_ten_option_evpi_endpoint_returns_422_with_cost_body(self):
        """Endpoint wiring: the enhanced (v2) handler surfaces the admission 422."""
        client = TestClient(app)
        body = _request_dict(40, 120, 5000, 10, evpi_factors=5)
        resp = client.post(ENDPOINT, json=body, headers={"X-ISL-Response-Version": "2"})
        assert resp.status_code == 422
        data = resp.json()
        assert data["cost_units"] > data["limit"]
        # v4 (OC-1): the recalibrated evppi_full no longer dominates; 'evpi' does.
        assert data["dominant_term"] == "evpi"
        assert data["complexity_formula_version"] == COMPLEXITY_FORMULA_VERSION

    def test_ten_option_evpi_legacy_endpoint_returns_422(self):
        """Legacy (v1) handler returns the SAME normalised admission 422 as v2.

        /simplify (19 Jul): the v1 path formerly raised HTTPException, which the
        app's custom handler reshaped into the Olumi Error Schema (stringifying the
        structured admission body into `message`). Both v2 handlers now go through
        the shared _admission_cost_guard helper and return the flat structured
        body directly — so cost_units / limit / dominant_term are top-level JSON
        numbers, not buried in a stringified `message`.
        """
        client = TestClient(app)
        body = _request_dict(40, 120, 5000, 10, evpi_factors=5)
        resp = client.post(ENDPOINT, json=body)  # default response_version=1
        assert resp.status_code == 422
        data = resp.json()
        # Flat structured body (the shape the enhanced/v2 handler serves), NOT the
        # Olumi Error Schema wrapper — data["cost_units"] would KeyError on the old
        # stringified-into-message form, so this discriminates the normalisation.
        assert data["detail"] == "Request compute cost exceeds limit"
        assert data["cost_units"] > data["limit"]
        # v4 (OC-1): the recalibrated evppi_full no longer dominates; 'evpi' does.
        assert data["dominant_term"] == "evpi"

    def test_admission_422_preserves_x_request_id_both_handlers(self):
        """Both v2 handlers' compute-cost 422 echo X-Request-Id + serve the flat
        structured body (regression guard for the /simplify legacy↔enhanced fix).

        Before the fix the legacy (v1) path raised HTTPException and the global
        handler rebuilt the response into the Olumi Error Schema; both paths now go
        through the shared _admission_cost_guard. (X-Request-Id is additionally
        backstopped by TracingMiddleware; the sharper discriminator here is the
        flat cost body, which the old wrapped shape lacked.)
        """
        client = TestClient(app)
        body = _request_dict(40, 120, 5000, 10, evpi_factors=5)  # over the ceiling
        rid = "req-simplify-xrid-probe"

        # Legacy (response_version=1, default)
        legacy = client.post(ENDPOINT, json=body, headers={"X-Request-Id": rid})
        assert legacy.status_code == 422
        assert legacy.headers.get("X-Request-Id") == rid
        assert legacy.json()["cost_units"] > legacy.json()["limit"]

        # Enhanced (response_version=2)
        enhanced = client.post(
            ENDPOINT,
            json=body,
            headers={"X-Request-Id": rid, "X-ISL-Response-Version": "2"},
        )
        assert enhanced.status_code == 422
        assert enhanced.headers.get("X-Request-Id") == rid
        assert enhanced.json()["cost_units"] > enhanced.json()["limit"]


# ---------------------------------------------------------------------------
# D-23.12 (Codex-fix-A F2) — the EVPC + full-pop EVPPI free-ride must be priced
# ---------------------------------------------------------------------------
def _max_controls(n_factor_ids: int = 5, n_values: int = 7, start: int = 1) -> list:
    """MAX_CONTROL_CANDIDATES(5) x MAX_CONTROL_VALUES(7) control candidates on the
    factor nodes n{start}..; distinct ids, finite values."""
    return [
        {"factor_id": f"n{start + k}", "values": [0.1 * (v + 1) for v in range(n_values)]}
        for k in range(n_factor_ids)
    ]


class TestEvpcAdmission:
    """Codex F2 repro (D-23.12): a 50n/200e/10000s request charged the SAME with vs
    without max 5x7 control_candidates and admitted ~90M work against the 24M ceiling.
    The EVPC term must now change the charge and flip admit->reject."""

    def test_max_controls_changes_the_charge(self):
        """RED pre-fix: EVPC unpriced -> with-controls total == without-controls total."""
        without = _cost(50, 200, 10000, 1)
        body = _request_dict(50, 200, 10000, 1)
        body["control_candidates"] = _max_controls()
        with_controls = compute_weighted_cost(RobustnessRequestV2(**body)).total
        # The charge must strictly INCREASE by exactly the EVPC term (D-23.12).
        S, W, grid = 10000, 50 + 200, 5 * 7
        assert with_controls == without + W_EVPC_COEF * S * W * grid
        assert with_controls > without

    def test_evpc_term_at_least_S_W_sum_values(self):
        """The D-23.12 floor: EVPC term >= S * W * sum_candidates(len(values))."""
        body = _request_dict(50, 200, 10000, 1)
        body["control_candidates"] = _max_controls()
        wc = compute_weighted_cost(RobustnessRequestV2(**body))
        S, W, grid = 10000, 50 + 200, 5 * 7
        assert wc.terms["evpc"] >= S * W * grid
        assert wc.terms["evpc"] == 87_500_000  # 10000 * 250 * 35 (pinned exact)

    def test_max_controls_flips_admit_to_reject(self):
        """The load-bearing flip (Codex's exact matrix): a near-ceiling request that
        ADMITS without controls REJECTS once max controls are added.

        MUTATION ANCHOR: reverting the EVPC term in compute_weighted_cost makes this
        request admit again (with_controls collapses back to ~2.5M)."""
        ceiling = get_max_cost_units()
        without = _cost(50, 200, 10000, 1)
        assert without <= ceiling, f"baseline must admit (cost={without:,})"

        body = _request_dict(50, 200, 10000, 1)
        body["control_candidates"] = _max_controls()
        with_controls = compute_weighted_cost(RobustnessRequestV2(**body)).total
        assert with_controls > ceiling, f"max controls must reject (cost={with_controls:,})"

    def test_max_controls_endpoint_returns_422_evpc_dominant(self):
        client = TestClient(app)
        body = _request_dict(50, 200, 10000, 1)
        body["control_candidates"] = _max_controls()
        resp = client.post(ENDPOINT, json=body, headers={"X-ISL-Response-Version": "2"})
        assert resp.status_code == 422
        data = resp.json()
        assert data["cost_units"] > data["limit"]
        assert data["dominant_term"] == "evpc"
        assert "control_candidates" in data["suggestion"]


class TestFullPopulationEvppiTerm:
    """D-23.12: the S2 EVPPI regression runs on the FULL S, not the EVPI_SAMPLE_CAP
    subsample the p_win 'evpi' term prices, so it needs its own full-S term."""

    def test_evppi_full_uses_full_S_not_sample_cap(self):
        # At S=10000 (>> cap 2000) the term must scale with the FULL S.
        s_small = compute_weighted_cost(
            RobustnessRequestV2(**_request_dict(20, 40, 2000, 3, evpi_factors=4))
        ).terms["evppi_full"]
        s_big = compute_weighted_cost(
            RobustnessRequestV2(**_request_dict(20, 40, 10000, 3, evpi_factors=4))
        ).terms["evppi_full"]
        # 5x the samples -> 5x the term (linear in FULL S, unlike the capped 'evpi').
        assert s_big == 5 * s_small

    def test_evppi_full_formula(self):
        wc = compute_weighted_cost(
            RobustnessRequestV2(**_request_dict(20, 40, 10000, 3, evpi_factors=4))
        )
        # v4 (OC-1): U * (1+K) * S — no O factor (shared multi-RHS SVD), no (deg+1).
        expected = W_EVPPI_COEF * 4 * (1 + REGRESSION_EVPPI_NULL_PERMUTATIONS) * 10000
        assert wc.terms["evppi_full"] == expected

    def test_evppi_full_is_O_flat(self):
        """v4 invariant (OC-1): the EVPPI term must NOT scale with option count —
        the estimator solves all options in one multi-RHS SVD (evppi.py
        _inner_expected_max), so charging *O was measured 42-192x over.

        MUTATION ANCHOR: restoring the v3 `* O` factor makes these differ 5x."""
        o2 = compute_weighted_cost(
            RobustnessRequestV2(**_request_dict(20, 40, 10000, 2, evpi_factors=4))
        ).terms["evppi_full"]
        o10 = compute_weighted_cost(
            RobustnessRequestV2(**_request_dict(20, 40, 10000, 10, evpi_factors=4))
        ).terms["evppi_full"]
        assert o2 == o10

    def test_evppi_full_absent_without_voi(self):
        wc = compute_weighted_cost(RobustnessRequestV2(**_request_dict(20, 40, 5000, 3)))
        assert "evppi_full" not in wc.terms


# ---------------------------------------------------------------------------
# F2 CLASS-closer — no analyzer phase may ship unpriced-and-unregistered
# ---------------------------------------------------------------------------
class TestPhasePricingInventory:
    """Codex F2's ROOT CAUSE was not the missing EVPC term — it was that a new
    compute phase could ship with NOBODY forced to answer 'who prices this?'.
    This guard makes that question fail loud (trap #12: a mirror must FAIL
    LOUD on drift). See PHASE_COST_ATTRIBUTION in robustness_analyzer_v2."""

    def test_every_compute_phase_is_registered(self):
        from src.services.robustness_analyzer_v2 import (
            PHASE_COST_ATTRIBUTION,
            RobustnessAnalyzerV2,
        )

        # N3 (D-23.19): the inventory is a NAME-PREFIX tripwire, not a proof —
        # widened to _run_ after Codex showed a `_run_unpriced_probe` evaded the
        # _compute_-only sweep. A phase named outside these prefixes still
        # evades it; the registry header says so honestly.
        methods = {
            m
            for m in dir(RobustnessAnalyzerV2)
            if m.startswith("_compute_") or m.startswith("_run_")
        }
        unregistered = methods - set(PHASE_COST_ATTRIBUTION)
        stale = set(PHASE_COST_ATTRIBUTION) - methods
        assert not unregistered, (
            f"New analyzer phase(s) {sorted(unregistered)} are not in "
            "PHASE_COST_ATTRIBUTION. Before merging: either add a pricing term to "
            "compute_weighted_cost and register 'priced:<term>', or register an "
            "honest 'subsumed:'/'bounded:' entry with the justification. A phase "
            "that ships unpriced is the Codex-F2 defect (EVPC ran 87.5M units of "
            "work charged at 0)."
        )
        assert not stale, (
            f"PHASE_COST_ATTRIBUTION entries {sorted(stale)} no longer exist on "
            "RobustnessAnalyzerV2 — remove them so the registry stays derived-true."
        )

    def test_priced_terms_exist_and_formula_terms_are_claimed(self):
        """Bidirectional: every 'priced:<term>' names a real formula term, and
        every term the formula can emit is claimed by at least one phase."""
        from src.services.robustness_analyzer_v2 import PHASE_COST_ATTRIBUTION

        # Derive the full term set from the formula itself: a request with every
        # optional phase enabled (VOI + controls + sensitivity + e-values + paths).
        body = _request_dict(20, 40, 5000, 3, evpi_factors=4)
        body["include_e_values"] = True
        body["include_path_decomposition"] = True
        # ROADMAP 2.228-F3: "every optional phase enabled" must MEAN every one —
        # omitting a flag here silently shrinks the derived term set and turns the
        # bidirectional check into a one-directional one.
        body["include_factor_flips"] = True
        body["control_candidates"] = [{"factor_id": "n5", "values": [0.1, 0.2]}]
        formula_terms = set(
            compute_weighted_cost(RobustnessRequestV2(**body)).terms
        ) | {"bands"}  # bands ride e_values; both emitted together

        priced_terms = {
            v.split(":", 1)[1]
            for v in PHASE_COST_ATTRIBUTION.values()
            if v.startswith("priced:")
        }
        unknown = priced_terms - formula_terms
        assert not unknown, (
            f"Registry prices phase(s) against non-existent formula term(s) {sorted(unknown)}"
        )
        # bands ride the e_values sweep; every OTHER term needs a claiming phase.
        unclaimed = formula_terms - priced_terms - {"bands"}
        assert not unclaimed, (
            f"Formula term(s) {sorted(unclaimed)} are claimed by no registered phase — "
            "either dead pricing or an unregistered phase."
        )

    def test_subsumed_targets_are_registered_and_bounded_reasons_nonempty(self):
        from src.services.robustness_analyzer_v2 import PHASE_COST_ATTRIBUTION

        for phase, disposition in PHASE_COST_ATTRIBUTION.items():
            kind, _, rest = disposition.partition(":")
            assert kind in ("priced", "subsumed", "bounded"), (
                f"{phase}: unknown disposition kind {kind!r}"
            )
            if kind == "subsumed":
                assert rest in PHASE_COST_ATTRIBUTION, (
                    f"{phase} is subsumed by unregistered {rest!r}"
                )
            if kind == "bounded":
                assert len(rest.strip()) >= 10, (
                    f"{phase}: 'bounded' needs a real justification, not {rest!r}"
                )


# ---------------------------------------------------------------------------
# OC-1 (D-23.17) — the v3 over-charge must stop 422-ing legal requests
# ---------------------------------------------------------------------------
class TestOC1RecalAdmission:
    """OC-1 repro: S=10000/U=20/O=2 (a legal, ~2s request) charged 34M EVPPI units
    under v3 (5*20*17*2*10000) against the 24M ceiling -> 422. Under the measured
    v4 term (1*20*17*10000 = 3.4M) it must ADMIT, while the F2 free-ride stays
    closed (TestEvpcAdmission pins that side, unchanged)."""

    def test_oc1_shape_now_admits(self):
        """MUTATION ANCHOR: restoring the v3 term ((deg+1)*U*(1+K)*O*S) charges
        this request 34M for EVPPI alone -> total > 24M -> this assert flips RED."""
        body = _request_dict(22, 40, 10000, 2, evpi_factors=20)
        wc = compute_weighted_cost(RobustnessRequestV2(**body))
        # v4 charge: base 1.24M + evpi 5.208M + evppi_full 3.4M + sensitivity 992k
        assert wc.terms["evppi_full"] == 20 * (1 + REGRESSION_EVPPI_NULL_PERMUTATIONS) * 10000
        assert wc.total <= get_max_cost_units(), (
            f"OC-1 shape must admit under v4 (cost={wc.total:,})"
        )

    def test_oc1_shape_endpoint_200(self):
        """End-to-end: the exact OC-1 shape passes admission AND completes.
        (sensitivity omitted — not part of the OC-1 claim; keeps the test fast)"""
        client = TestClient(app)
        body = _request_dict(22, 40, 10000, 2, evpi_factors=20, sensitivity=False)
        resp = client.post(ENDPOINT, json=body, headers={"X-ISL-Response-Version": "2"})
        assert resp.status_code == 200, f"OC-1 shape must run (got {resp.status_code})"


# ---------------------------------------------------------------------------
# PC-3 — shape fix must NOT regress (deep single graph)
# ---------------------------------------------------------------------------
class TestPC3ShapeFix:
    def test_deep_single_graph_now_admits(self):
        """40n/120e/10000s/1opt: OLD scalar 422'd (48M > 30M). Must now admit."""
        old_scalar = 10000 * 40 * 120
        assert old_scalar == 48_000_000  # what the old formula computed
        cost = _cost(40, 120, 10000, 1)
        assert cost <= get_max_cost_units(), f"deep single graph must admit (cost={cost:,})"

    def test_deep_single_graph_endpoint_admits(self):
        client = TestClient(app)
        body = _request_dict(40, 120, 10000, 1)
        resp = client.post(ENDPOINT, json=body, headers={"X-ISL-Response-Version": "2"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Ceiling + weight pins (silent-revert guards)
# ---------------------------------------------------------------------------
class TestCalibrationPins:
    def test_provisional_ceiling_pinned(self):
        """PROVISIONAL — Paul-DIRECTED 24M (2026-07-18, the more-lenient choice),
        staging recalibration still owed. A silent change must turn this RED so the
        number is never altered unnoticed."""
        assert DEFAULT_MAX_COST_UNITS == 24_000_000

    def test_weight_coefficients_pinned(self):
        assert BASE_COST_COEF == 1
        assert W_SENS_COEF == 4
        assert W_EVAL_COEF == 20
        assert W_BANDS_COEF == 200
        assert W_PATH_COEF == 1
        assert EVPI_SAMPLE_CAP == 2000
        # D-23.12 EVPC + full-pop EVPPI coefficients (silent-revert guard).
        assert W_EVPC_COEF == 1
        # v4 (OC-1, D-23.17): measurement-recalibrated from (deg+1)=5 with *O to a
        # flat 1 per (factor-fit x permutation x sample) — see W_EVPPI_COEF comment.
        assert W_EVPPI_COEF == 1
        assert REGRESSION_EVPPI_NULL_PERMUTATIONS == 16

    def test_formula_version_pinned(self):
        # Bumped v4 -> v5 by ROADMAP 2.228-F3, which added the `factor_flips`
        # term. This pin is doing its job: the formula cannot change without an
        # explicit edit here and a new version string on /health.
        assert COMPLEXITY_FORMULA_VERSION == "v5-factor-flips-2026-08-01"

    def test_env_override_resolves_new_var_only(self, monkeypatch):
        """ISL_MAX_COST_UNITS overrides; the OLD ISL_MAX_COMPUTE_COMPLEXITY does NOT."""
        monkeypatch.setenv("ISL_MAX_COST_UNITS", "12345")
        assert get_max_cost_units() == 12345
        monkeypatch.delenv("ISL_MAX_COST_UNITS", raising=False)
        # Old env name must be ignored (would be in wrong units).
        monkeypatch.setenv("ISL_MAX_COMPUTE_COMPLEXITY", "999")
        assert get_max_cost_units() == DEFAULT_MAX_COST_UNITS

    def test_env_override_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("ISL_MAX_COST_UNITS", "not-an-int")
        assert get_max_cost_units() == DEFAULT_MAX_COST_UNITS


# ---------------------------------------------------------------------------
# ROADMAP 2.260 step 3 — the advertisement must be SUFFICIENT to price a request
# ---------------------------------------------------------------------------
#
# The 12 weight keys as advertised at 29cb4e27, BEFORE this change. Pinned as a
# LITERAL on purpose — the one place in this file where restating rather than
# deriving is correct. PLoT couples the `weights` KEY SET exactly to the formula
# version (it requires every expected key AND rejects any unexpected one,
# compute-admission.ts:125-130 and :141-145 at #302 head 5fff2253), so growing
# `weights` forces a lockstep consumer release or degrades PLoT to the
# conservative fallback this change exists to lift. New per-term parameters go in
# the `formula_parameters` sibling instead. If a future change genuinely must add
# a weight key, this pin is the place that makes the consumer cost visible.
_WEIGHT_KEYS_AT_29CB4E27 = {
    "base_per_sample_per_option_per_struct",
    "evpi_sample_cap",
    "evpc_coef",
    "evppi_full_coef",
    "evppi_null_permutations",
    "factor_flip_coef",
    "influence_walk_pool",
    "sensitivity_coef",
    "evalue_coef",
    "bands_coef",
    "path_coef",
    "max_decomposition_paths",
}


def _reference_terms_from_advertisement(req: RobustnessRequestV2, admission: dict) -> dict:
    """Reimplement compute_weighted_cost using ONLY the advertised block.

    Deliberately written the way a CONSUMER must write it (a direct transcription
    of PLoT's `estimateWeightedCostV2` style): the structural SHAPE is hard-coded
    here, and every NUMBER comes from the advertisement. Not a single literal
    from the cost model is permitted below — if a term needs a constant this
    function cannot reach through `weights` / `formula_parameters` / `caps`, the
    term is unpriceable from the advertisement, and that is exactly the defect
    under test.

    ⚠ HONEST SCOPE. This proves NUMERIC sufficiency, not shape-freeness. The
    formula's SHAPE is versioned by `complexity_formula_version` and a consumer
    is expected to hold code per shape; what it must NOT have to hold is ISL's
    numbers. A future term that changes the shape is caught by the term-set
    assertion in the test below (it appears in ISL's terms and not here), not by
    this function silently adapting to it.
    """
    w = admission["weights"]
    params = admission["formula_parameters"]
    S = req.n_samples
    O = len(req.options)
    N = len(req.graph.nodes)
    E = len(req.graph.edges)
    W = N + E

    terms = {"base_mc": w["base_per_sample_per_option_per_struct"] * S * O * W}

    if req.include_voi and req.parameter_uncertainties:
        u = len({pu.node_id for pu in req.parameter_uncertainties})
        if u > 0:
            terms["evpi"] = (u + 1) * min(S, w["evpi_sample_cap"]) * O * W
            terms["evppi_full"] = (
                w["evppi_full_coef"] * u * (1 + w["evppi_null_permutations"]) * S
            )

    if req.control_candidates:
        grid_points = sum(len(c.values) for c in req.control_candidates)
        if grid_points > 0:
            terms["evpc"] = w["evpc_coef"] * S * W * grid_points

    if "sensitivity" in req.analysis_types:
        sens = params["sensitivity"]
        terms["sensitivity"] = (
            w["sensitivity_coef"]
            * E
            * min(sens["subsample_cap"], S // sens["subsample_divisor"])
            * W
        )
        if req.parameter_uncertainties:
            terms["structural_influence"] = w["influence_walk_pool"]

    if req.include_e_values:
        terms["e_values"] = w["evalue_coef"] * E * O
        terms["bands"] = w["bands_coef"] * E * O

    if req.include_factor_flips:
        flips = params["factor_flips"]
        candidate_cap = flips["max_candidates"]
        seeds = flips["stability_seeds"]
        evaluates = O * (1 + 2 * N + 2 * candidate_cap * (max(O - 1, 0) + seeds))
        terms["factor_flips"] = w["factor_flip_coef"] * evaluates * W

    if req.include_path_decomposition:
        terms["path_decomposition"] = w["path_coef"] * min(w["max_decomposition_paths"], E * E)

    return terms


def _all_terms_request() -> dict:
    """A request that switches on EVERY optional phase, so the shape grid below
    exercises every term the formula can emit."""
    body = _request_dict(12, 40, 3000, 3, evpi_factors=4, sensitivity=True)
    body["control_candidates"] = _max_controls(n_factor_ids=3, n_values=4, start=6)
    body["include_e_values"] = True
    body["include_factor_flips"] = True
    body["include_path_decomposition"] = True
    return body


def _sufficiency_shape_grid() -> dict:
    """Request shapes covering every term in both its present and absent state."""
    return {
        "all_phases": _all_terms_request(),
        "base_only": _request_dict(8, 20, 1000, 2, sensitivity=False),
        "sensitivity_only": _request_dict(12, 40, 3000, 3),
        "voi_and_sensitivity": _request_dict(12, 40, 3000, 3, evpi_factors=4),
        "flips_only": {
            **_request_dict(10, 30, 2000, 4, sensitivity=False),
            "include_factor_flips": True,
        },
        "flips_single_option": {
            **_request_dict(10, 30, 2000, 1, sensitivity=False),
            "include_factor_flips": True,
        },
        "evalues_and_paths": {
            **_request_dict(15, 50, 5000, 2, sensitivity=False),
            "include_e_values": True,
            "include_path_decomposition": True,
        },
        # Straddle the sensitivity sub-sweep min(): 10000//10 = 1000 -> CAP binds;
        # 200//10 = 20 -> DIVISOR binds. Both parameters must be advertised for
        # these two to agree with ISL.
        "deep_samples_sensitivity_cap_binds": _request_dict(12, 40, 10000, 1),
        "shallow_samples_sensitivity_divisor_binds": _request_dict(12, 40, 200, 1),
        # ⚠ BOTH SHAPES BELOW EXIST BECAUSE THE GRID WAS BRANCH-BLIND (adversarial
        # review of #119). Every term was exercised, but two min() calls were only
        # ever evaluated on ONE side, so mutants that DELETED the min survived the
        # entire 2494-test gate. Covering a term is not covering its branches.
        #
        # min(S, evpi_sample_cap): every other VOI shape uses S=3000 > 2000, so the
        # CAP always bound. S=1500 < 2000 binds the S side. (Reviewer mutant M6.)
        "voi_shallow_samples_S_branch_binds": _request_dict(12, 40, 1500, 2, evpi_factors=3),
        # min(max_decomposition_paths, E*E): the other path shape has E=50, so
        # E*E=2500 < 20000 and E*E always bound. E=145 gives E*E=21025 > 20000,
        # binding the CAP side. (Reviewer mutant M7.)
        "paths_cap_branch_binds": {
            **_request_dict(50, 145, 1000, 2, sensitivity=False),
            "include_path_decomposition": True,
        },
    }


class TestAdvertisementSufficiency:
    """ROADMAP 2.260 step 3 — /health must advertise EVERY number the price uses.

    THE DEFECT THIS CLOSES. PLoT prices ISL's advertised complexity-formula
    parameters to admit a formula version. The v5 `factor_flips` term is
    parameterised by FACTOR_FLIP_MAX_CANDIDATES and FLIP_STABILITY_N_SEEDS, and
    NEITHER was advertised — while PLoT sends `include_factor_flips: true` on
    every base call, so the term is always in ISL's real price. PLoT could not
    price v5, fell back conservatively, and silently capped production depth
    10,000 -> 4,000 (made loud by PLoT #302). The completeness audit for this
    lane found the same class of gap in the `sensitivity` term, whose sub-sweep
    sizing (min(CAP, S//DIVISOR)) was a bare literal PLoT hard-codes on its side.

    Refusing to hard-code ISL's constants cross-repo was the CORRECT call
    (programme trap 12 — the hand-maintained mirror). The fix belongs here.
    """

    def test_factor_flip_parameters_are_advertised(self):
        """RED pre-fix: the two constants the v5 factor_flips term reads are absent."""
        params = build_compute_admission().get("formula_parameters", {})
        flips = params.get("factor_flips", {})
        missing = [k for k in ("max_candidates", "stability_seeds") if k not in flips]
        assert not missing, (
            f"factor_flips term parameters not advertised: {missing}. PLoT sends "
            f"include_factor_flips=true on every base call, so this term is always "
            f"priced; without these it cannot reproduce the price. "
            f"formula_parameters={params}"
        )

    def test_sensitivity_subsample_parameters_are_advertised(self):
        """RED pre-fix: found by the completeness audit, not by the original report."""
        params = build_compute_admission().get("formula_parameters", {})
        sens = params.get("sensitivity", {})
        missing = [k for k in ("subsample_cap", "subsample_divisor") if k not in sens]
        assert not missing, (
            f"sensitivity term parameters not advertised: {missing}. The term is "
            f"sensitivity_coef*E*min(CAP, S//DIVISOR)*W; a consumer that cannot read "
            f"CAP and DIVISOR must hard-code them. formula_parameters={params}"
        )

    def test_advertised_parameter_values_derive_from_the_constants(self):
        """CONTRACT PIN, DERIVED not restated (trap 12).

        Compares the advertised values against the SAME symbols the cost model
        reads, never against literals. Retuning any constant flows into the
        advertisement with no edit here; hard-coding a value in
        build_compute_admission instead of deriving it turns this RED.
        """
        params = build_compute_admission()["formula_parameters"]
        assert (
            params["factor_flips"]["max_candidates"]
            == RobustnessAnalyzerV2.FACTOR_FLIP_MAX_CANDIDATES
        )
        assert params["factor_flips"]["stability_seeds"] == FLIP_STABILITY_N_SEEDS
        assert params["sensitivity"]["subsample_cap"] == SENSITIVITY_SUBSAMPLE_CAP
        assert params["sensitivity"]["subsample_divisor"] == SENSITIVITY_SUBSAMPLE_DIVISOR

    def test_formula_parameters_are_keyed_by_real_term_names(self):
        """The association a consumer relies on: every formula_parameters key names
        a term compute_weighted_cost actually emits, so a parameter can be matched
        to the term it prices without a naming convention to remember. Derived from
        the phase-cost attribution registry, not a hand-listed set."""
        priced_terms = {
            v.split("priced:", 1)[1]
            for v in PHASE_COST_ATTRIBUTION.values()
            if v.startswith("priced:")
        }
        advertised = set(build_compute_admission()["formula_parameters"])
        assert advertised <= priced_terms, (
            f"formula_parameters keys that are not priced term names: "
            f"{sorted(advertised - priced_terms)}. Known priced terms: "
            f"{sorted(priced_terms)}"
        )

    def test_weights_key_set_is_unchanged(self):
        """CONSUMER-COUPLING GUARD (the reason formula_parameters is a sibling).

        PLoT couples the `weights` key set EXACTLY to the formula version — every
        expected key must be present AND any unexpected key is treated as skew
        (compute-admission.ts:125-130, :141-145 at #302 head 5fff2253). So adding
        a weight key is not additive at the seam: it forces a lockstep PLoT
        release or drops PLoT to the conservative fallback. This asserts the key
        set is byte-for-byte what 29cb4e27 advertised, so that cost can never be
        incurred by accident — a new per-term parameter belongs in
        `formula_parameters`, which PLoT's shape check ignores (:110-122).
        """
        advertised = set(build_compute_admission()["weights"])
        assert advertised == _WEIGHT_KEYS_AT_29CB4E27, (
            f"the `weights` key set changed — added={sorted(advertised - _WEIGHT_KEYS_AT_29CB4E27)}, "
            f"removed={sorted(_WEIGHT_KEYS_AT_29CB4E27 - advertised)}. This is a "
            f"BREAKING change for consumers that couple to the key set; put new "
            f"per-term parameters in `formula_parameters` instead, or bump "
            f"complexity_formula_version and coordinate the consumer release."
        )

    def test_health_endpoint_serves_the_parameters(self):
        """End-to-end through health.py's import-time precomputed static block —
        the advertisement a consumer actually reads off the wire, not just the
        builder's return value."""
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        block = resp.json()["compute_admission"]
        params = block["formula_parameters"]
        assert (
            params["factor_flips"]["max_candidates"]
            == RobustnessAnalyzerV2.FACTOR_FLIP_MAX_CANDIDATES
        )
        assert params["factor_flips"]["stability_seeds"] == FLIP_STABILITY_N_SEEDS
        assert params["sensitivity"]["subsample_cap"] == SENSITIVITY_SUBSAMPLE_CAP
        assert params["sensitivity"]["subsample_divisor"] == SENSITIVITY_SUBSAMPLE_DIVISOR
        # The sibling placement must hold ON THE WIRE too, not just in the builder:
        # these must NOT have leaked into `weights`, or PLoT sees skew.
        assert set(block["weights"]) == _WEIGHT_KEYS_AT_29CB4E27

    def test_advertisement_reproduces_every_term_exactly(self):
        """THE SUFFICIENCY GUARANTEE — the point of the exercise.

        A consumer holding ONLY the advertised block and its own request must
        reproduce compute_weighted_cost term-for-term. Asserted across a shape
        grid that switches every optional phase on and off, so every term the
        formula can emit is covered in both its present and absent state.

        This is what stops the next silent fallback: a new term, or a new
        constant inside an existing term, that is not advertised cannot satisfy
        this equality, so it fails CI at the moment it is added rather than
        surfacing months later as an unexplained depth cut in production.
        """
        admission = build_compute_admission()
        for name, body in _sufficiency_shape_grid().items():
            req = RobustnessRequestV2(**body)
            actual = compute_weighted_cost(req)
            reference = _reference_terms_from_advertisement(req, admission)
            assert set(actual.terms) == set(reference), (
                f"[{name}] term SET differs — ISL prices a term the advertisement "
                f"cannot express. ISL-only={sorted(set(actual.terms) - set(reference))}, "
                f"reference-only={sorted(set(reference) - set(actual.terms))}"
            )
            for term in sorted(actual.terms):
                assert actual.terms[term] == reference[term], (
                    f"[{name}] term '{term}' not reproducible from the advertisement: "
                    f"ISL={actual.terms[term]:,} vs advertised-only={reference[term]:,}"
                )
            assert actual.total == sum(reference.values())

    def test_sufficiency_grid_actually_exercises_every_term(self):
        """POSITIVE CONTROL for the test above (trap 13 — an equality that never
        sees a term proves nothing about it).

        Asserts the shape grid collectively emits EVERY term compute_weighted_cost
        can produce, derived from the phase-cost attribution registry rather than
        a hand-listed set, so a newly-priced phase joins this expectation
        automatically and the grid must be extended to cover it.
        """
        priced_terms = {
            v.split("priced:", 1)[1]
            for v in PHASE_COST_ATTRIBUTION.values()
            if v.startswith("priced:")
        }
        seen: set = set()
        for body in _sufficiency_shape_grid().values():
            seen |= set(compute_weighted_cost(RobustnessRequestV2(**body)).terms)
        # 'bands' rides on e_values and is not itself a registry 'priced:' entry.
        seen.discard("bands")
        assert priced_terms <= seen, (
            f"the sufficiency grid never exercises {sorted(priced_terms - seen)} — "
            f"the equality assertion is vacuous for those terms. Extend the grid."
        )
