"""F4 (Codex) — hard-invalid correlation matrices are REJECTED (typed 422), not
silently Higham-projected. Doctrine D-23.13 (enforcing D-23.4 +
PARAMETER-RESEARCH-2026-07-23.md:49-59): "reject hard-invalid, ELSE Higham-project
near-PSD with disclosure". Before this fix the sampler Higham-projected EVERYTHING —
a logically contradictory matrix (rho(a,b)=1, rho(a,c)=1, rho(b,c)=-1 → eigenvalues
[-1,2,2]) was accepted and silently moved to effective off-diagonals ~±0.5, so the
analysis ran under materially different assumptions and disclosed only an aggregate
distance.

Each mechanism carries a mutation note in NOTES.md: reverting the reject band turns
the contradiction test RED (200 again); removing the effective-matrix disclosure turns
the disclosure test RED.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from src.models.robustness_v2 import (
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
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2


def _three_factor_graph():
    nodes = [
        NodeV2(id=f, kind="factor", label=f, observed_state=ObservedState(value=1.0))
        for f in ("fa", "fb", "fc")
    ]
    nodes.append(NodeV2(id="rev", kind="outcome", label="Rev"))
    edges = [
        EdgeV2(
            **{"from": f, "to": "rev"},
            exists_probability=1.0,
            strength=StrengthDistribution(mean=1.0, std=0.05),
        )
        for f in ("fa", "fb", "fc")
    ]
    return GraphV2(nodes=nodes, edges=edges)


_UNC3 = [
    ParameterUncertainty(node_id="fa", distribution="normal", std=0.5),
    ParameterUncertainty(node_id="fb", distribution="normal", std=0.5),
    ParameterUncertainty(node_id="fc", distribution="normal", std=0.5),
]
_OPTS3 = [
    InterventionOption(id="o1", label="O1", interventions={}),
    InterventionOption(id="o2", label="O2", interventions={"fa": 1.0}),
]


def _request3(correlations):
    return RobustnessRequestV2(
        request_id="corr-hard-invalid",
        graph=_three_factor_graph(),
        options=_OPTS3,
        goal_node_id="rev",
        n_samples=2000,
        seed=42,
        analysis_types=["comparison", "sensitivity", "robustness"],
        parameter_uncertainties=_UNC3,
        include_voi=True,
        factor_correlations=correlations,
    )


# Codex's EXACT repro: 3 normal factors, rho(a,b)=1, rho(a,c)=1, rho(b,c)=-1.
_CONTRADICTION = [
    FactorCorrelation(factor_a="fa", factor_b="fb", rho=1.0),
    FactorCorrelation(factor_a="fa", factor_b="fc", rho=1.0),
    FactorCorrelation(factor_a="fb", factor_b="fc", rho=-1.0),
]
# Genuinely near-PSD (frustrated 0.51/0.51/-0.51): lambda_min=-0.02, max adj=0.01 —
# non-PSD but well inside the near-PSD repair band → must STILL project (200).
_NEAR_PSD = [
    FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.51),
    FactorCorrelation(factor_a="fa", factor_b="fc", rho=-0.51),
    FactorCorrelation(factor_a="fb", factor_b="fc", rho=0.51),
]
# Strongly inconsistent (the OLD showcase input): lambda_min=-0.8 — NOT noise.
_STRONGLY_INCONSISTENT = [
    FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.9),
    FactorCorrelation(factor_a="fb", factor_b="fc", rho=0.9),
    FactorCorrelation(factor_a="fa", factor_b="fc", rho=-0.9),
]


@pytest.fixture(scope="module")
def analyzer():
    return RobustnessAnalyzerV2()


class TestHardInvalidRejected:
    def test_codex_contradiction_1_1_neg1_rejected_422(self):
        # THE headline: [-1,2,2] contradiction must be rejected at parse time (typed
        # 422), never reach the sampler. Mutation anchor: reverting the reject band
        # makes this a 200.
        with pytest.raises(ValidationError) as exc:
            _request3(_CONTRADICTION)
        msg = str(exc.value)
        assert "factor_correlations" in msg

    def test_contradiction_message_names_offending_pairs(self):
        try:
            _request3(_CONTRADICTION)
            assert False, "contradiction should have been rejected"
        except ValidationError as exc:
            # Assert against the CLEAN validator message (errors()[0]["msg"]) — the
            # str(ValidationError) wrapper additionally appends pydantic's own truncated
            # input_value= echo, a framework artifact present on every pydantic error;
            # the privacy contract is about OUR message content.
            msg = exc.errors()[0]["msg"]
            # Names the offending factors (its own correlation ids) ...
            assert "fa" in msg and "fb" in msg and "fc" in msg
            # ... and the scalar spectral/adjustment metrics, not the raw rho values.
            assert "smallest eigenvalue" in msg
            assert "corr_admission_v1" in msg

    def test_contradiction_message_no_other_request_values(self):
        # Privacy: OUR reject message may name the correlation factor ids + derived
        # scalar metrics, but must NOT echo other request values.
        try:
            _request3(_CONTRADICTION)
            assert False, "should have raised"
        except ValidationError as exc:
            msg = exc.errors()[0]["msg"]
            assert "observed_state" not in msg
            assert "strength" not in msg
            assert "n_samples" not in msg
            assert "seed" not in msg

    def test_strongly_inconsistent_09_09_neg09_rejected(self):
        # lambda_min=-0.8: strongly inconsistent, not float noise → rejected.
        with pytest.raises(ValidationError):
            _request3(_STRONGLY_INCONSISTENT)


class TestNearPsdStillProjects:
    def test_near_psd_projects_200(self, analyzer):
        # frustrated 0.51 (lambda_min=-0.02) is non-PSD but inside the repair band →
        # analysis completes (200) and Higham projection is disclosed.
        resp = analyzer.analyze(_request3(_NEAR_PSD))
        proj = resp.correlation_model.psd_projection
        assert proj is not None
        assert proj.applied is True
        assert proj.frobenius_distance > 0.0
        assert proj.iterations > 0

    def test_near_psd_discloses_effective_matrix(self, analyzer):
        # Part 2: the projection must disclose the EFFECTIVE adjusted correlations,
        # not only the aggregate distance — so a caller can reconstruct what actually
        # drove the numbers. (RED pre-fix: field does not exist.)
        resp = analyzer.analyze(_request3(_NEAR_PSD))
        proj = resp.correlation_model.psd_projection
        eff = proj.effective_correlations
        assert eff is not None
        assert len(eff) == 3  # one per supplied pair
        # Reconstruct the effective matrix: each |effective_rho| < the stated 0.51
        # (projection pulled the frustrated triangle toward consistency).
        for e in eff:
            assert abs(e.effective_rho) <= 0.51
            assert e.adjustment == pytest.approx(e.effective_rho - _stated_rho(e), abs=1e-9)


def _stated_rho(e):
    stated = {("fa", "fb"): 0.51, ("fa", "fc"): -0.51, ("fb", "fc"): 0.51}
    key = (e.factor_a, e.factor_b)
    return stated.get(key, stated.get((e.factor_b, e.factor_a)))


class TestValidPathUnchanged:
    def test_psd_pair_still_accepted(self, analyzer):
        # A normal valid correlation (rho=0.9 pair, PSD) is untouched — no reject,
        # no projection.
        resp = analyzer.analyze(
            _request3([FactorCorrelation(factor_a="fa", factor_b="fb", rho=0.9)])
        )
        assert resp.correlation_model.psd_projection is None


# =============================================================================
# The admissibility band as pure math (evaluate_correlation_admissibility)
# =============================================================================

from src.utils.correlation import (  # noqa: E402
    CORRELATION_ADMISSION_METHOD_VERSION,
    CORRELATION_REJECT_MAX_ADJUSTMENT,
    CORRELATION_REJECT_MIN_EIGENVALUE,
    assemble_correlation_matrix,
    evaluate_correlation_admissibility,
)


def _frustrated(rho):
    return assemble_correlation_matrix(
        ["a", "b", "c"], [("a", "b", rho), ("b", "c", rho), ("a", "c", -rho)]
    )


class TestAdmissibilityBand:
    def test_contradiction_inadmissible_both_reasons(self):
        v = evaluate_correlation_admissibility(_frustrated(1.0))  # [-1, 2, 2]
        assert v.admissible is False
        assert "min_eigenvalue" in v.reasons
        assert "max_adjustment" in v.reasons
        assert v.min_eigenvalue == pytest.approx(-1.0, abs=1e-9)
        assert v.max_abs_off_diagonal_adjustment == pytest.approx(0.5, abs=1e-9)

    def test_near_psd_admissible(self):
        v = evaluate_correlation_admissibility(_frustrated(0.51))  # lambda_min = -0.02
        assert v.admissible is True
        assert v.reasons == ()
        assert -0.05 <= v.min_eigenvalue < 0.0  # genuinely non-PSD but inside the band

    def test_already_psd_admissible_no_projection(self):
        # rho=0.3 frustrated is PSD (lambda_min = +0.4): admissible, no projection metrics.
        v = evaluate_correlation_admissibility(_frustrated(0.3))
        assert v.admissible is True
        assert v.max_abs_off_diagonal_adjustment == 0.0
        assert v.frobenius_distance == 0.0

    def test_boundary_just_inside_projects_just_outside_rejects(self):
        # 0.52 -> lambda_min = -0.04 (admissible); 0.55 -> lambda_min = -0.10 (reject).
        assert evaluate_correlation_admissibility(_frustrated(0.52)).admissible is True
        assert evaluate_correlation_admissibility(_frustrated(0.55)).admissible is False

    def test_verdict_is_permutation_invariant(self):
        # Relabeling the factors (row/col permutation) must not change the verdict —
        # eigenvalues and max|off-diag adjustment| are permutation-invariant.
        m1 = assemble_correlation_matrix(
            ["a", "b", "c"], [("a", "b", 1.0), ("a", "c", 1.0), ("b", "c", -1.0)]
        )
        m2 = assemble_correlation_matrix(
            ["c", "a", "b"], [("a", "b", 1.0), ("a", "c", 1.0), ("b", "c", -1.0)]
        )
        v1 = evaluate_correlation_admissibility(m1)
        v2 = evaluate_correlation_admissibility(m2)
        assert v1.admissible == v2.admissible
        assert v1.min_eigenvalue == pytest.approx(v2.min_eigenvalue, abs=1e-9)
        assert v1.max_abs_off_diagonal_adjustment == pytest.approx(
            v2.max_abs_off_diagonal_adjustment, abs=1e-9
        )


# =============================================================================
# Fingerprint guard — the Neil-parameters are versioned; a silent retune fails loud
# (derive-don't-mirror, CLAUDE.md #12; same pattern as CONFIDENCE_METHOD_VERSION).
# =============================================================================

from src.utils.canonical_hash import canonical_json_hash  # noqa: E402

# PINNED PAIR (update BOTH together, and only as a deliberate, disclosed change):
#   fingerprint  <->  version
# When you intentionally change a reject-band constant:
#   1. bump CORRELATION_ADMISSION_METHOD_VERSION in src/utils/correlation.py, and
#   2. update PINNED_ADMISSION_FINGERPRINT + PINNED_ADMISSION_VERSION below to the new
#      values (the failure message prints the fresh fingerprint to paste).
# Do NOT regenerate the pin blindly to make this test pass — that defeats the
# fail-loud-on-drift contract.
PINNED_ADMISSION_VERSION = "corr_admission_v1"
PINNED_ADMISSION_FINGERPRINT = "92307365c71bd35fe57c8c669a6b275a24cc01ed3f3610769d9d332f962f69d8"


def _current_admission_fingerprint() -> str:
    """sha256 over a canonical repr of the reject-band constants, computed LIVE from
    source. The pin is a hardcoded literal, so this never self-heals."""
    return canonical_json_hash(
        {
            "CORRELATION_REJECT_MIN_EIGENVALUE": CORRELATION_REJECT_MIN_EIGENVALUE,
            "CORRELATION_REJECT_MAX_ADJUSTMENT": CORRELATION_REJECT_MAX_ADJUSTMENT,
        }
    )


class TestAdmissionFingerprintGuard:
    def test_version_matches_pin(self):
        assert CORRELATION_ADMISSION_METHOD_VERSION == PINNED_ADMISSION_VERSION

    def test_constants_fingerprint_pinned_to_version(self):
        current = _current_admission_fingerprint()
        assert current == PINNED_ADMISSION_FINGERPRINT, (
            "The correlation hard-invalid reject band "
            "(CORRELATION_REJECT_MIN_EIGENVALUE / CORRELATION_REJECT_MAX_ADJUSTMENT) "
            "changed but CORRELATION_ADMISSION_METHOD_VERSION is still "
            f"'{CORRELATION_ADMISSION_METHOD_VERSION}'. These are Neil-parameters; a "
            "retune MUST be disclosed:\n"
            "  1. bump CORRELATION_ADMISSION_METHOD_VERSION in src/utils/correlation.py, and\n"
            "  2. update PINNED_ADMISSION_FINGERPRINT + PINNED_ADMISSION_VERSION in this test.\n"
            f"Current fingerprint to pin: {current}\n"
            f"Old pinned fingerprint:     {PINNED_ADMISSION_FINGERPRINT}"
        )
