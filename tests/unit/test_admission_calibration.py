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
    W_BANDS_COEF,
    W_EVAL_COEF,
    W_EVPC_COEF,
    W_EVPPI_DEGREE_COEF,
    W_PATH_COEF,
    W_SENS_COEF,
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
        # 2000 cap): (deg+1) * U * (1+K) * O * S = 5 * 5 * 17 * 10 * 5000 = 21.25M.
        assert wc.terms["evppi_full"] == (
            W_EVPPI_DEGREE_COEF * 5 * (1 + REGRESSION_EVPPI_NULL_PERMUTATIONS) * 10 * 5000
        )
        # Because the full-S EVPPI term (21.25M) exceeds the sample-capped p_win term
        # (19.2M) at 5000 samples, evppi_full is now the dominant term for VOI-heavy
        # requests (GOLDEN CHANGED by D-23.12; previously "evpi").
        assert wc.dominant_term == "evppi_full"

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
        # D-23.12: full-S EVPPI term now dominates over the sample-capped p_win term.
        assert data["dominant_term"] == "evppi_full"
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
        # D-23.12: full-S EVPPI term now dominates over the sample-capped p_win term.
        assert data["dominant_term"] == "evppi_full"

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
        expected = W_EVPPI_DEGREE_COEF * 4 * (1 + REGRESSION_EVPPI_NULL_PERMUTATIONS) * 3 * 10000
        assert wc.terms["evppi_full"] == expected

    def test_evppi_full_absent_without_voi(self):
        wc = compute_weighted_cost(RobustnessRequestV2(**_request_dict(20, 40, 5000, 3)))
        assert "evppi_full" not in wc.terms


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
        assert W_EVPPI_DEGREE_COEF == 5  # REGRESSION_EVPPI_POLY_DEGREE (4) + 1
        assert REGRESSION_EVPPI_NULL_PERMUTATIONS == 16

    def test_formula_version_pinned(self):
        assert COMPLEXITY_FORMULA_VERSION == "v3-evpc-evppi-2026-07-24"

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
