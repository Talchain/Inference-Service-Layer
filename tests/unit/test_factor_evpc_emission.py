"""S4 (A3 value-of-control, D-23.8) — analyzer + model tests for ``factor_evpc``.

EVPC(lever) = max_x E[U | do(lever=x)] − max_a E[U_a] on the retained joint CRN
samples (grid do(), no nested MC). Covers:

* linear hand-derivation (exact best_candidate + CRN-EXACT EVPC ratios for a 3-value
  grid: baseline/best_do/evpc all scale with the shared sampled edge strength S, so
  the RATIOS are seed-invariant and exact),
* the zero-case (a lever with no path to goal → EVPC=0 with clamped_low, and
  best_candidate_value STILL reported),
* request-driven gating (absent control_candidates → factor_evpc is None; present →
  emitted) and byte-additivity (presence perturbs no other field),
* NO include_voi coupling (control is a distinct capability from information),
* NO D-U lever suppression (an option-controlled factor still gets an EVPC entry —
  the mirror-image of factor_evppi, which OMITS levers),
* emission + correlation_active under active factor correlation (seeded pin),
* determinism, grid lower-bound monotonicity, sort order,
* request validation 422s (unknown factor / goal-node / duplicate / non-finite / caps),
* the ISLResponseV2 value-integrity validator (clamp identity + best_value present +
  method/units), asserted BOTH ways.
"""

from __future__ import annotations

import math

import pytest

from pydantic import ValidationError

from src.models.response_v2 import ISLResponseV2
from src.models.robustness_v2 import (
    ControlCandidate,
    EdgeV2,
    FactorCorrelation,
    GraphV2,
    InterventionOption,
    NodeV2,
    ObservedState,
    ParameterUncertainty,
    RobustnessRequestV2,
    StrengthDistribution,
)
from src.services.robustness_analyzer_v2 import GRID_DO_EVPC_METHOD, RobustnessAnalyzerV2


# ---------------------------------------------------------------------------
# Graph builders (epsilon_std defaults to 0 → evaluate() is deterministic; the
# EVPC evaluator additionally runs with epsilon disabled, analyze() ~line 1524).
# ---------------------------------------------------------------------------
def _linear_grid_request(m=0.5, values=None, seed=42, n_samples=8000, **overrides):
    """goal = strength(mean=m) * lever. Two options pin lever ∈ {0.2, 0.8}; a
    'zed' factor has NO path to the goal (base 0). control candidates on both."""
    if values is None:
        values = [0.2, 0.5, 0.9]
    nodes = [
        NodeV2(id="lever", kind="factor", label="Lever", observed_state=ObservedState(value=0.0)),
        NodeV2(id="zed", kind="factor", label="No-path", observed_state=ObservedState(value=0.0)),
        NodeV2(id="goal", kind="outcome", label="Goal", observed_state=ObservedState(value=0.0)),
    ]
    edges = [
        EdgeV2(
            **{"from": "lever"},
            to="goal",
            strength=StrengthDistribution(mean=m, std=0.002),
            exists_probability=1.0,
        ),
    ]
    req = dict(
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[
            InterventionOption(id="lo", label="Lever 0.2", interventions={"lever": 0.2}),
            InterventionOption(id="hi", label="Lever 0.8", interventions={"lever": 0.8}),
        ],
        goal_node_id="goal",
        seed=seed,
        n_samples=n_samples,
        control_candidates=[
            ControlCandidate(factor_id="lever", values=values),
            ControlCandidate(factor_id="zed", values=[0.1, 0.4]),
        ],
    )
    req.update(overrides)
    return RobustnessRequestV2(**req)


def _by_factor(response):
    return {e["factor_id"]: e for e in (response.factor_evpc or [])}


# ===========================================================================
# 1. Linear hand-derivation (exact) + zero-case
# ===========================================================================
class TestEvpcLinearHandDerivation:
    def test_lever_grid_best_candidate_and_evpc(self):
        """m=0.5, options pin lever∈{0.2,0.8} ⇒ baseline = 0.8·S; grid {0.2,0.5,0.9}
        ⇒ best_do = 0.9·S at x=0.9 ⇒ EVPC_raw = 0.1·S, EVPC = 0.05.

        The CRN ratios are EXACT (S — the shared sampled edge strength — cancels):
            best_do/baseline = 0.9/0.8 = 1.125
            evpc_raw/baseline = 0.1/0.8 = 0.125
        """
        resp = RobustnessAnalyzerV2().analyze(_linear_grid_request(m=0.5))
        lever = _by_factor(resp)["lever"]

        assert lever["best_candidate_value"] == 0.9  # exact argmax over the grid
        assert lever["clamped_low"] is False
        assert lever["units"] == "outcome"
        assert lever["method"] == GRID_DO_EVPC_METHOD
        assert lever["n_candidate_values"] == 3

        # Value ≈ 0.1·m = 0.05 (S ≈ m for std 0.002); tolerance dwarfs the ~1e-5
        # sampling noise but is far tighter than any real mutation (~0.05 shift).
        assert lever["evpc"] == pytest.approx(0.05, abs=1e-3)
        assert lever["evpc"] == max(0.0, lever["evpc_raw"])  # clamp identity

        # CRN-EXACT ratios (seed-invariant; only 6-dp rounding-limited).
        base = lever["baseline_max_expected_utility"]
        assert lever["best_do_expected_utility"] / base == pytest.approx(1.125, abs=1e-5)
        assert lever["evpc_raw"] / base == pytest.approx(0.125, abs=1e-5)

    def test_do_shift_scales_with_edge_strength(self):
        """do(x) shifts the outcome by m·(x−x_baseline): doubling m doubles EVPC."""
        e1 = _by_factor(RobustnessAnalyzerV2().analyze(_linear_grid_request(m=0.4)))["lever"]
        e2 = _by_factor(RobustnessAnalyzerV2().analyze(_linear_grid_request(m=0.8)))["lever"]
        # EVPC = 0.1·m ⇒ ratio 2.0 (exact up to shared sampling noise).
        assert e2["evpc"] / e1["evpc"] == pytest.approx(2.0, abs=1e-2)

    def test_zero_case_lever_with_no_path_to_goal(self):
        """'zed' has no edge to the goal ⇒ do(zed=x) cannot beat the best option ⇒
        EVPC=0 (clamped_low True), and best_candidate_value is STILL reported
        (honest: 'controlling this adds nothing, and this was the best value tried')."""
        resp = RobustnessAnalyzerV2().analyze(_linear_grid_request())
        zed = _by_factor(resp)["zed"]
        assert zed["evpc"] == 0.0
        assert zed["evpc_raw"] < 0.0
        assert zed["clamped_low"] is True
        # best_candidate_value present + finite even though EVPC == 0.
        assert zed["best_candidate_value"] == 0.1  # first value (strict '>' keeps first)
        assert math.isfinite(zed["best_do_expected_utility"])


# ===========================================================================
# 2. Gating, coupling, suppression, additivity
# ===========================================================================
class TestEvpcGating:
    def test_absent_control_candidates_omits_factor_evpc(self):
        resp = RobustnessAnalyzerV2().analyze(_linear_grid_request(control_candidates=None))
        assert resp.factor_evpc is None

    def test_present_control_candidates_emits_factor_evpc(self):
        resp = RobustnessAnalyzerV2().analyze(_linear_grid_request())
        assert resp.factor_evpc is not None
        assert {e["factor_id"] for e in resp.factor_evpc} == {"lever", "zed"}

    def test_not_coupled_to_include_voi(self):
        """EVPC emits with include_voi=False (default): control ≠ information."""
        req = _linear_grid_request()
        assert req.include_voi is False
        resp = RobustnessAnalyzerV2().analyze(req)
        assert resp.factor_evpc is not None

    def test_no_lever_suppression_option_controlled_factor_still_scored(self):
        """'lever' IS intervened by both options (a D-U lever). factor_evppi OMITS
        such levers; factor_evpc must NOT — control is the point."""
        resp = RobustnessAnalyzerV2().analyze(_linear_grid_request())
        assert "lever" in _by_factor(resp)

    def test_presence_is_byte_additive(self):
        """Adding control_candidates perturbs NO other wire field (additive-only):
        the two responses are identical except for factor_evpc. request_id is pinned
        and the wall-clock execution_time_ms is stripped (both inherently vary and are
        not part of the analysis payload)."""
        with_cc = RobustnessAnalyzerV2().analyze(_linear_grid_request(request_id="evpc-additive"))
        without_cc = RobustnessAnalyzerV2().analyze(
            _linear_grid_request(request_id="evpc-additive", control_candidates=None)
        )
        a = with_cc.model_dump(by_alias=True, exclude_none=True)
        b = without_cc.model_dump(by_alias=True, exclude_none=True)
        a.pop("factor_evpc", None)
        b.pop("factor_evpc", None)
        a["_metadata"].pop("execution_time_ms", None)
        b["_metadata"].pop("execution_time_ms", None)
        assert a == b


# ===========================================================================
# 3. Correlation, determinism, monotonicity, sort
# ===========================================================================
def _corr_request(rho=0.7, seed=2024, n_samples=4000, include_voi=False):
    """factor_a & factor_b are correlated (Gaussian copula); both feed the goal.
    Options pin factor_a ∈ {0.3, 0.6}; control candidate grids factor_a. factor_b is a
    NON-lever uncertain factor, so with include_voi it yields a factor_evppi entry."""
    nodes = [
        NodeV2(id="factor_a", kind="factor", label="A", observed_state=ObservedState(value=0.5)),
        NodeV2(id="factor_b", kind="factor", label="B", observed_state=ObservedState(value=0.5)),
        NodeV2(id="goal", kind="outcome", label="G", observed_state=ObservedState(value=0.0)),
    ]
    edges = [
        EdgeV2(
            **{"from": "factor_a"},
            to="goal",
            strength=StrengthDistribution(mean=0.6, std=0.002),
            exists_probability=1.0,
        ),
        EdgeV2(
            **{"from": "factor_b"},
            to="goal",
            strength=StrengthDistribution(mean=0.4, std=0.002),
            exists_probability=1.0,
        ),
    ]
    return RobustnessRequestV2(
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[
            InterventionOption(id="a_lo", label="a lo", interventions={"factor_a": 0.3}),
            InterventionOption(id="a_hi", label="a hi", interventions={"factor_a": 0.6}),
        ],
        goal_node_id="goal",
        seed=seed,
        n_samples=n_samples,
        parameter_uncertainties=[
            ParameterUncertainty(node_id="factor_a", distribution="normal", std=0.2),
            ParameterUncertainty(node_id="factor_b", distribution="normal", std=0.2),
        ],
        factor_correlations=[FactorCorrelation(factor_a="factor_a", factor_b="factor_b", rho=rho)],
        control_candidates=[ControlCandidate(factor_id="factor_a", values=[0.3, 0.6, 0.9])],
        include_voi=include_voi,
    )


class TestEvpcCorrelationAndDeterminism:
    def test_emitted_under_active_correlation_with_disclosure(self):
        """EVPC do() runs on the joint copula draws (partner keeps its draw) ⇒ it is
        honest under correlation and EMITTED (like factor_evppi), tagged
        correlation_active=True. Seeded pin (seed=2024)."""
        resp = RobustnessAnalyzerV2().analyze(_corr_request())
        entry = _by_factor(resp)["factor_a"]
        assert entry["correlation_active"] is True
        assert entry["best_candidate_value"] == 0.9
        assert entry["clamped_low"] is False
        # Seeded pin: do(a=0.9) ⇒ 0.6·0.9 + 0.4·E[b≈0.5] ≈ 0.74; baseline a=0.6 ⇒
        # 0.6·0.6 + 0.4·E[b≈0.5] ≈ 0.56 ⇒ EVPC ≈ 0.18.
        assert entry["evpc"] == pytest.approx(0.179994, abs=1e-4)
        assert entry["baseline_max_expected_utility"] == pytest.approx(0.560575, abs=1e-4)
        assert entry["best_do_expected_utility"] == pytest.approx(0.74057, abs=1e-4)

    def test_deterministic_same_seed(self):
        r1 = RobustnessAnalyzerV2().analyze(_corr_request())
        r2 = RobustnessAnalyzerV2().analyze(_corr_request())
        assert r1.factor_evpc == r2.factor_evpc

    def test_composition_correlation_voi_control_suppression_record(self):
        """Rebase composition onto #104 (record-at-skip suppression + typed VOI blocks):
        correlation-active + control_candidates + include_voi ⇒ factor_evpc AND
        factor_evppi are BOTH present (both are honest joint-copula-sample quantities,
        EMITTED under correlation), and the correlation suppression RECORD
        (correlation_model.suppressed_attributions) contains NEITHER — only the
        independence-assuming attributions are recorded. Non-vacuous: the record IS
        populated (positive control below), so 'not in' is a real absence."""
        resp = RobustnessAnalyzerV2().analyze(_corr_request(include_voi=True))

        # Both VOI-family blocks emit under active correlation.
        assert resp.factor_evpc is not None
        assert {e["factor_id"] for e in resp.factor_evpc} == {"factor_a"}
        assert resp.factor_evppi is not None
        assert {e["factor_id"] for e in resp.factor_evppi} == {"factor_b"}

        # The suppression record is REAL (positive control), and captures ONLY the
        # independence-assuming attributions — never EVPC or EVPPI.
        assert resp.correlation_model is not None
        suppressed = set(resp.correlation_model.suppressed_attributions)
        assert "p_win_sensitivity" in suppressed  # positive control: record populated
        assert "factor_sensitivity" in suppressed
        assert "factor_evpc" not in suppressed
        assert "factor_evppi" not in suppressed

    def test_grid_is_a_lower_bound_more_values_never_lower_evpc(self):
        """More candidate values can only find an equal-or-better do() point ⇒ the
        reported EVPC (a grid lower bound on the true continuous EVPC) is monotone
        non-decreasing as the grid is refined."""
        coarse = _by_factor(
            RobustnessAnalyzerV2().analyze(_linear_grid_request(values=[0.2, 0.5]))
        )["lever"]
        fine = _by_factor(
            RobustnessAnalyzerV2().analyze(_linear_grid_request(values=[0.2, 0.5, 0.9]))
        )["lever"]
        assert fine["evpc"] >= coarse["evpc"] - 1e-9

    def test_sorted_by_evpc_descending(self):
        """A strong lever (steep edge to goal) sorts before a weak one."""
        nodes = [
            NodeV2(id="strong", kind="factor", label="S", observed_state=ObservedState(value=0.0)),
            NodeV2(id="weak", kind="factor", label="W", observed_state=ObservedState(value=0.0)),
            NodeV2(id="goal", kind="outcome", label="G", observed_state=ObservedState(value=0.0)),
        ]
        edges = [
            EdgeV2(
                **{"from": "strong"},
                to="goal",
                strength=StrengthDistribution(mean=0.9, std=0.002),
                exists_probability=1.0,
            ),
            EdgeV2(
                **{"from": "weak"},
                to="goal",
                strength=StrengthDistribution(mean=0.1, std=0.002),
                exists_probability=1.0,
            ),
        ]
        req = RobustnessRequestV2(
            graph=GraphV2(nodes=nodes, edges=edges),
            options=[
                InterventionOption(id="o", label="o", interventions={"strong": 0.2, "weak": 0.2})
            ],
            goal_node_id="goal",
            seed=7,
            n_samples=2000,
            control_candidates=[
                ControlCandidate(factor_id="weak", values=[0.2, 0.9]),
                ControlCandidate(factor_id="strong", values=[0.2, 0.9]),
            ],
        )
        resp = RobustnessAnalyzerV2().analyze(req)
        order = [e["factor_id"] for e in resp.factor_evpc]
        assert order == ["strong", "weak"]
        assert resp.factor_evpc[0]["evpc"] >= resp.factor_evpc[1]["evpc"]


# ===========================================================================
# 4. Request-validation 422s (fail-closed at parse time, factor-named)
# ===========================================================================
def _valid_request_kwargs():
    nodes = [
        NodeV2(id="lever", kind="factor", label="L", observed_state=ObservedState(value=0.0)),
        NodeV2(id="goal", kind="outcome", label="G", observed_state=ObservedState(value=0.0)),
    ]
    edges = [
        EdgeV2(
            **{"from": "lever"},
            to="goal",
            strength=StrengthDistribution(mean=0.5, std=0.01),
            exists_probability=1.0,
        )
    ]
    return dict(
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[InterventionOption(id="o", label="o", interventions={"lever": 0.5})],
        goal_node_id="goal",
        n_samples=200,
    )


class TestControlCandidateValidation:
    def test_unknown_factor_rejected(self):
        with pytest.raises(ValidationError, match="non-existent factor node: ghost"):
            RobustnessRequestV2(
                **_valid_request_kwargs(),
                control_candidates=[ControlCandidate(factor_id="ghost", values=[0.1])],
            )

    def test_goal_node_rejected(self):
        with pytest.raises(ValidationError, match="may not target the goal node"):
            RobustnessRequestV2(
                **_valid_request_kwargs(),
                control_candidates=[ControlCandidate(factor_id="goal", values=[0.1])],
            )

    def test_duplicate_factor_id_rejected(self):
        with pytest.raises(ValidationError, match="duplicate factor_id 'lever'"):
            RobustnessRequestV2(
                **_valid_request_kwargs(),
                control_candidates=[
                    ControlCandidate(factor_id="lever", values=[0.1]),
                    ControlCandidate(factor_id="lever", values=[0.2]),
                ],
            )

    def test_non_finite_value_rejected_factor_named(self):
        with pytest.raises(ValidationError, match="values must all be finite"):
            ControlCandidate(factor_id="lever", values=[0.1, math.inf])
        with pytest.raises(ValidationError, match="values must all be finite"):
            ControlCandidate(factor_id="lever", values=[math.nan])

    def test_too_many_candidates_rejected(self):
        # 6 > MAX_CONTROL_CANDIDATES (5) — distinct ids so the count cap trips.
        with pytest.raises(ValidationError):
            RobustnessRequestV2(
                **_valid_request_kwargs(),
                control_candidates=[
                    ControlCandidate(factor_id=f"c{i}", values=[0.1]) for i in range(6)
                ],
            )

    def test_too_many_values_rejected(self):
        # 8 > MAX_CONTROL_VALUES (7).
        with pytest.raises(ValidationError):
            ControlCandidate(factor_id="lever", values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    def test_empty_values_rejected(self):
        with pytest.raises(ValidationError):
            ControlCandidate(factor_id="lever", values=[])

    def test_absent_control_candidates_is_valid_and_inert(self):
        req = RobustnessRequestV2(**_valid_request_kwargs())
        assert req.control_candidates is None


def _kwargs_with_extra_node(node_id: str, kind: str) -> dict:
    """_valid_request_kwargs plus an isolated extra node of the given kind, so a
    control candidate can target a non-factor kind."""
    nodes = [
        NodeV2(id="lever", kind="factor", label="L", observed_state=ObservedState(value=0.0)),
        NodeV2(id="goal", kind="outcome", label="G", observed_state=ObservedState(value=0.0)),
        NodeV2(id=node_id, kind=kind, label=node_id, observed_state=ObservedState(value=0.0)),
    ]
    edges = [
        EdgeV2(
            **{"from": "lever"},
            to="goal",
            strength=StrengthDistribution(mean=0.5, std=0.01),
            exists_probability=1.0,
        )
    ]
    return dict(
        graph=GraphV2(nodes=nodes, edges=edges),
        options=[InterventionOption(id="o", label="o", interventions={"lever": 0.5})],
        goal_node_id="goal",
        n_samples=200,
    )


class TestControlCandidateKindValidation:
    """Codex F5 (D-23.14): control_candidates accepted a FILTERED node kind
    (decision/option/constraint), which filter_inference_graph then removes -> do()
    no-ops -> EVPC clamps to a FALSE 0. A filtered kind must 422 BEFORE compute,
    naming only the offending factor. Kinds that SURVIVE the filter (factor / chance /
    non-goal outcome) compute a real EVPC and are accepted (S4 design)."""

    @pytest.mark.parametrize("kind", ["decision", "option", "constraint"])
    def test_filtered_kind_rejected_before_compute(self, kind):
        # RED pre-fix: these validated and reached analyze() with EVPC = 0.0 (clamped).
        with pytest.raises(ValidationError, match=rf"factor 'x' has kind '{kind}'"):
            RobustnessRequestV2(
                **_kwargs_with_extra_node("x", kind),
                control_candidates=[ControlCandidate(factor_id="x", values=[0.1, 0.5])],
            )

    def test_filtered_kind_message_names_filter_reason(self):
        with pytest.raises(ValidationError, match="removed by the inference-node filter"):
            RobustnessRequestV2(
                **_kwargs_with_extra_node("d", "decision"),
                control_candidates=[ControlCandidate(factor_id="d", values=[0.1])],
            )

    @pytest.mark.parametrize("kind", ["chance", "outcome"])
    def test_filter_surviving_non_factor_kind_accepted(self, kind):
        # A node that SURVIVES the filter (chance / non-goal outcome mediator) computes
        # a REAL, non-clamped EVPC — the S4 design controls intermediate chance nodes
        # (tests/integration/test_factor_evpc_wire.py) — so it is ACCEPTED. Only the
        # FILTERED kinds carry the Codex false-0 defect. (Deviation from D-23.14's
        # literal "kind==factor allowlist", per its own "reject if it won't reach the
        # evaluator" intent — see the validator docstring; flagged for A3 review.)
        req = RobustnessRequestV2(
            **_kwargs_with_extra_node("c", kind),
            control_candidates=[ControlCandidate(factor_id="c", values=[0.1, 0.5])],
        )
        assert req.control_candidates[0].factor_id == "c"  # validated, not rejected

    def test_message_does_not_echo_other_request_values(self):
        # The rejection MESSAGE (error["msg"], the text the API 422 handler surfaces to
        # the client — src/api/main.py builds the body from msg, never pydantic's raw
        # input) names only the offending factor id + its kind, never a candidate value
        # (mirrors the correlation validator's disclosure discipline).
        with pytest.raises(ValidationError) as exc:
            RobustnessRequestV2(
                **_kwargs_with_extra_node("bad", "decision"),
                control_candidates=[ControlCandidate(factor_id="bad", values=[0.31337, 0.42424])],
            )
        msgs = " || ".join(err["msg"] for err in exc.value.errors())
        assert "bad" in msgs  # names the offending factor
        assert "0.31337" not in msgs and "0.42424" not in msgs

    def test_valid_factor_candidate_reaches_evaluator(self):
        """Positive control: a valid factor candidate passes validation AND emits a
        real factor_evpc row (reaches the evaluator, not clamped by a silent filter)."""
        resp = RobustnessAnalyzerV2().analyze(
            RobustnessRequestV2(
                **_valid_request_kwargs(),
                control_candidates=[ControlCandidate(factor_id="lever", values=[0.2, 0.8])],
            )
        )
        rows = {e["factor_id"]: e for e in (resp.factor_evpc or [])}
        assert "lever" in rows
        assert rows["lever"]["method"] == GRID_DO_EVPC_METHOD


# ===========================================================================
# 5. ISLResponseV2 value-integrity validator (both ways)
# ===========================================================================
def _isl_response(entry):
    echo = {
        "n_samples": 10,
        "seed_used": "1",
        "graph_node_count": 2,
        "graph_edge_count": 1,
        "options_count": 2,
        "goal_node_id_hash": "h",
        "response_version_requested": 2,
        "include_diagnostics": False,
    }
    return dict(
        endpoint_version="analyze/v2",
        engine_version="x",
        analysis_status="computed",
        robustness_status="computed",
        factor_sensitivity_status="skipped",
        request_echo=echo,
        request_id="r",
        processing_time_ms=1,
        factor_evpc=[entry],
    )


_GOOD_ENTRY = {
    "factor_id": "lever",
    "evpc": 0.05,
    "evpc_raw": 0.05,
    "best_candidate_value": 0.9,
    "baseline_max_expected_utility": 0.4,
    "best_do_expected_utility": 0.45,
    "units": "outcome",
    "method": "grid_do_v1",
    "n_samples": 10,
    "n_candidate_values": 3,
    "clamped_low": False,
    "correlation_active": False,
}


class TestFactorEvpcValidator:
    def test_valid_entry_passes(self):
        assert ISLResponseV2(**_isl_response(_GOOD_ENTRY))

    def test_zero_case_entry_passes(self):
        zero = {
            **_GOOD_ENTRY,
            "evpc": 0.0,
            "evpc_raw": -0.4,
            "best_candidate_value": 0.1,
            "clamped_low": True,
        }
        assert ISLResponseV2(**_isl_response(zero))

    def test_absent_factor_evpc_passes(self):
        body = _isl_response(_GOOD_ENTRY)
        body.pop("factor_evpc")
        assert ISLResponseV2(**body)

    def test_clamp_forgotten_rejected(self):
        """evpc == evpc_raw when raw is NEGATIVE (clamp dropped) must fail loud."""
        bad = {**_GOOD_ENTRY, "evpc": -0.4, "evpc_raw": -0.4}
        with pytest.raises(ValidationError, match="evpc must equal max"):
            ISLResponseV2(**_isl_response(bad))

    def test_evpc_not_equal_clamped_raw_rejected(self):
        """A positive raw shipped mislabelled (evpc != max(0,raw)) must fail loud."""
        bad = {**_GOOD_ENTRY, "evpc": 0.9}  # != max(0, 0.05)
        with pytest.raises(ValidationError, match="evpc must equal max"):
            ISLResponseV2(**_isl_response(bad))

    def test_missing_best_candidate_value_rejected(self):
        # best_candidate_value is a REQUIRED float on FactorEvpcEntryV2 (C3 typed
        # house style), so a dropped argmax now fails loud at the Pydantic type layer
        # (the field name appears in the error), not a custom message.
        bad = {**_GOOD_ENTRY, "best_candidate_value": None}
        with pytest.raises(ValidationError, match="best_candidate_value"):
            ISLResponseV2(**_isl_response(bad))

    def test_non_finite_evpc_raw_rejected(self):
        bad = {**_GOOD_ENTRY, "evpc_raw": math.inf, "evpc": 0.05}
        with pytest.raises(ValidationError, match="must be finite"):
            ISLResponseV2(**_isl_response(bad))

    def test_wrong_method_rejected(self):
        bad = {**_GOOD_ENTRY, "method": "not_grid_do"}
        with pytest.raises(ValidationError, match="must be method-tagged"):
            ISLResponseV2(**_isl_response(bad))

    def test_wrong_units_rejected(self):
        bad = {**_GOOD_ENTRY, "units": "probability"}
        with pytest.raises(ValidationError, match="must be in outcome units"):
            ISLResponseV2(**_isl_response(bad))
