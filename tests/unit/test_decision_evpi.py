"""S1 (A3 VOI honesty, D-23.8) — decision-level EVPI: identity + validator.

decision_evpi is the HONEST decision-value EVPI in outcome units:

    decision_evpi = E[max_o U] − max_o E[U]
                  = min_o expected_regret[o]           (exact identity)

on the JOINT pre-noise Common-Random-Numbers population (the same population as
win_probability / downside.expected_regret). It is a READ of the regret already
computed by ``expected_regret_per_option`` — no new sampling.

This module covers:
  * the mathematical identity (min_o regret == E[max]−max E) across shapes,
    INCLUDING a point-mass shape where EVPI must be exactly 0; and
  * the ISLResponseV2 emission-iff + value validator, BOTH directions.
The end-to-end wire pins live in tests/integration/test_decision_evpi_wire.py.
"""

import numpy as np
import pytest

from pydantic import ValidationError

from src.models.response_v2 import (
    DownsideV2,
    ISLResponseV2,
    OptionResultV2,
    OutcomeDistributionV2,
)
from src.utils.downside import expected_regret_per_option
from src.utils.response_builder import build_request_echo


# ---------------------------------------------------------------------------
# The identity: decision_evpi == min_o expected_regret[o] == E[max] − max_o E[o]
# ---------------------------------------------------------------------------


def _evpi_from_identity(samples):
    """E[max_o U] − max_o E[U] computed directly (the definition)."""
    arrs = {k: np.asarray(v, dtype=float) for k, v in samples.items()}
    mat = np.vstack([arrs[k] for k in samples])
    e_max = float(mat.max(axis=0).mean())          # E[max_o U]
    max_e = max(float(a.mean()) for a in arrs.values())  # max_o E[U]
    return e_max - max_e


def _decision_evpi(samples):
    """The production quantity: min over per-option expected regret."""
    return min(expected_regret_per_option(samples).values())


def test_identity_three_separated_options():
    rng = np.random.default_rng(1)
    n = 5000
    base = rng.normal(0, 1, n)
    samples = {
        "a": (base + rng.normal(0, 0.3, n) + 2.0).tolist(),
        "b": (base + rng.normal(0, 0.3, n) + 0.5).tolist(),
        "c": (base + rng.normal(0, 0.3, n) - 1.0).tolist(),
    }
    assert _decision_evpi(samples) == pytest.approx(_evpi_from_identity(samples), abs=1e-9)


def test_identity_two_near_equivalent_options():
    rng = np.random.default_rng(2)
    n = 5000
    base = rng.normal(0, 1, n)
    samples = {
        "x": (base + rng.normal(0, 0.1, n)).tolist(),
        "y": (base + rng.normal(0, 0.1, n) + 0.01).tolist(),
    }
    assert _decision_evpi(samples) == pytest.approx(_evpi_from_identity(samples), abs=1e-9)


def test_identity_point_mass_shape_evpi_is_exactly_zero():
    """THEOREM case: degenerate (constant) options carry ZERO uncertainty, so
    perfect information has no value — decision_evpi must be EXACTLY 0.

    This pins the point-mass corner the brief calls out: min_o regret == 0 and
    E[max]−max E == 0 when every option is a point mass.
    """
    n = 500
    samples = {"p": [3.0] * n, "q": [1.0] * n, "r": [2.0] * n}
    assert _decision_evpi(samples) == 0.0
    assert _evpi_from_identity(samples) == 0.0


def test_identity_single_dominating_option_evpi_zero():
    """If one option is best on EVERY draw, you'd never switch — EVPI == 0
    (min_o regret is 0 for the dominator)."""
    rng = np.random.default_rng(3)
    n = 4000
    base = rng.normal(0, 1, n)
    samples = {"win": (base + 5.0).tolist(), "lose": base.tolist()}
    assert _decision_evpi(samples) == pytest.approx(0.0, abs=1e-9)
    assert _evpi_from_identity(samples) == pytest.approx(0.0, abs=1e-9)


def test_identity_point_mass_single_sample():
    samples = {"p": [3.0], "q": [1.0]}
    assert _decision_evpi(samples) == 0.0
    assert _evpi_from_identity(samples) == 0.0


# ---------------------------------------------------------------------------
# ISLResponseV2 validator — emission-iff + value, BOTH directions
# ---------------------------------------------------------------------------


def _echo():
    return build_request_echo(
        graph_node_count=3, graph_edge_count=2, options_count=2,
        goal_node_id="goal", n_samples=400, response_version=2,
        include_diagnostics=False,
    )


def _outcome(source="samples", **kw):
    base = {
        "mean": 100.0, "std": 10.0, "p10": 85.0, "p50": 100.0, "p90": 115.0,
        "n_samples": 400, "n_valid_samples": 400, "validity_ratio": 1.0,
        "percentiles_source": source,
    }
    base.update(kw)
    return OutcomeDistributionV2(**base)


def _opt(oid, regret, with_downside=True):
    return OptionResultV2(
        id=oid,
        outcome=_outcome(),
        downside=(
            DownsideV2(cvar_10=70.0, p05=78.0, expected_regret=regret)
            if with_downside else None
        ),
        status="computed",
    )


def _resp(options, decision_evpi):
    return ISLResponseV2(
        endpoint_version="analyze/v2",
        engine_version="1.0.0",
        analysis_status="computed",
        robustness_status="computed",
        factor_sensitivity_status="computed",
        critiques=[],
        request_echo=_echo(),
        options=options,
        decision_evpi=decision_evpi,
        request_id="t-1",
        processing_time_ms=100,
    )


def test_validator_present_ok_when_downside_and_value_is_min():
    """POSITIVE CONTROL: population present + decision_evpi == min(regret) is VALID."""
    r = _resp([_opt("low", 0.044), _opt("high", 0.054)], decision_evpi=0.044)
    assert r.decision_evpi == 0.044


def test_validator_absent_ok_when_no_downside_population():
    """ABSENT-WHEN-REGRET-ABSENT: no downside anywhere + decision_evpi None is VALID."""
    r = _resp(
        [_opt("low", 0.0, with_downside=False), _opt("high", 0.0, with_downside=False)],
        decision_evpi=None,
    )
    assert r.decision_evpi is None


def test_validator_absent_ok_when_options_none():
    r = _resp(None, decision_evpi=None)
    assert r.decision_evpi is None


def test_validator_rejects_present_without_population():
    """FABRICATION direction: decision_evpi present but NO regret population -> reject."""
    with pytest.raises(ValidationError, match="emission-iff"):
        _resp(
            [_opt("low", 0.0, with_downside=False)],
            decision_evpi=0.044,
        )


def test_validator_rejects_absent_with_population():
    """SILENT-LOSS direction: regret population present but decision_evpi dropped -> reject."""
    with pytest.raises(ValidationError, match="emission-iff"):
        _resp([_opt("low", 0.044), _opt("high", 0.054)], decision_evpi=None)


def test_validator_rejects_wrong_value_not_the_minimum():
    """VALUE direction (min->max / wrong-field mutation catcher): decision_evpi must
    equal the MINIMUM regret. Emitting the max (0.054) instead of the min (0.044)
    is rejected."""
    with pytest.raises(ValidationError, match="MINIMUM per-option expected regret"):
        _resp([_opt("low", 0.044), _opt("high", 0.054)], decision_evpi=0.054)


def test_validator_zero_evpi_is_present_not_absent():
    """0.0 is a REAL present value (dominated / degenerate decision), NOT 'absent'.
    A point-mass wire has all regrets 0 -> decision_evpi 0.0 and MUST validate as
    present (guards a `if decision_evpi:` truthiness bug conflating 0.0 with None)."""
    r = _resp([_opt("low", 0.0), _opt("high", 0.0)], decision_evpi=0.0)
    assert r.decision_evpi == 0.0


def test_validator_rejects_zero_when_population_absent():
    """The 0.0/None distinction the OTHER way: decision_evpi=0.0 with no population
    is still a fabrication (0.0 is present)."""
    with pytest.raises(ValidationError, match="emission-iff"):
        _resp([_opt("low", 0.0, with_downside=False)], decision_evpi=0.0)


def test_decision_evpi_omitted_not_null_on_wire_when_absent():
    """exclude_none=True (the emission path) drops decision_evpi entirely when
    absent — absent, never a JSON null."""
    r = _resp(
        [_opt("low", 0.0, with_downside=False)],
        decision_evpi=None,
    )
    dumped = r.model_dump(by_alias=True, exclude_none=True)
    assert "decision_evpi" not in dumped


def test_decision_evpi_present_serialises_on_wire():
    r = _resp([_opt("low", 0.044), _opt("high", 0.054)], decision_evpi=0.044)
    dumped = r.model_dump(by_alias=True, exclude_none=True)
    assert dumped["decision_evpi"] == 0.044


def test_decision_evpi_rejects_negative():
    """ge=0 field constraint: EVPI is non-negative by construction."""
    with pytest.raises(ValidationError):
        _resp([_opt("low", 0.044), _opt("high", 0.054)], decision_evpi=-0.01)
