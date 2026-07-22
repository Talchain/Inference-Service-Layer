"""Model-level tests for the B2 ``DownsideV2`` object and its emission invariant.

The invariant guards the FABRICATION direction: a downside (tail-risk) object may
ride only alongside real MC samples (outcome.percentiles_source == 'samples').
Emitting tail-risk metrics with no valid samples would invent a distribution.

The forward direction (samples ⟹ downside) is a completeness property of the
emission locus (src/api/robustness.py), asserted end-to-end in the API tests —
NOT a construction invariant, because many legitimate constructors build
OptionResultV2 with percentiles_source defaulting to 'samples' and no enrichment.
"""

import pytest
from pydantic import ValidationError

from src.models.response_v2 import DownsideV2, OptionResultV2, OutcomeDistributionV2


def _outcome(source="samples", **kw):
    base = dict(
        mean=100.0, std=10.0, p10=85.0, p50=100.0, p90=115.0,
        n_samples=1000, n_valid_samples=1000, validity_ratio=1.0,
        percentiles_source=source,
    )
    base.update(kw)
    return OutcomeDistributionV2(**base)


def test_downside_constructs_with_three_fields():
    d = DownsideV2(cvar_10=70.0, p05=78.0, expected_regret=12.5)
    assert d.cvar_10 == 70.0
    assert d.p05 == 78.0
    assert d.expected_regret == 12.5


def test_expected_regret_must_be_nonnegative():
    """expected_regret >= 0 by construction — the model enforces it (ge=0)."""
    with pytest.raises(ValidationError):
        DownsideV2(cvar_10=70.0, p05=78.0, expected_regret=-0.01)


def test_cvar_and_p05_may_be_negative():
    """cvar_10 / p05 are outcome values (same units as mean) — unbounded."""
    d = DownsideV2(cvar_10=-50.0, p05=-30.0, expected_regret=0.0)
    assert d.cvar_10 == -50.0


def test_downside_allowed_when_samples_present():
    r = OptionResultV2(
        id="a",
        outcome=_outcome(source="samples"),
        downside=DownsideV2(cvar_10=70.0, p05=78.0, expected_regret=5.0),
        status="computed",
    )
    assert r.downside is not None
    assert r.downside.cvar_10 == 70.0


def test_downside_forbidden_when_samples_unavailable():
    """FABRICATION guard: downside present but percentiles_source != 'samples'
    (i.e. no valid samples) must fail loud at construction."""
    with pytest.raises(ValidationError, match="B2 invariant"):
        OptionResultV2(
            id="a",
            outcome=_outcome(source="unavailable", p10=None, p50=None, p90=None),
            downside=DownsideV2(cvar_10=70.0, p05=78.0, expected_regret=5.0),
            status="computed",
        )


def test_absent_downside_with_samples_is_allowed_backward_compat():
    """The reverse is NOT a construction invariant: an option with
    percentiles_source == 'samples' and no downside is legitimate (every
    pre-existing OptionResultV2 construction is exactly this shape)."""
    r = OptionResultV2(id="a", outcome=_outcome(source="samples"), status="computed")
    assert r.downside is None


def test_absent_downside_with_unavailable_samples_is_allowed():
    r = OptionResultV2(
        id="a",
        outcome=_outcome(source="unavailable", p10=None, p50=None, p90=None),
        status="failed",
    )
    assert r.downside is None


def test_absent_downside_omitted_not_null_on_wire():
    """exclude_none=True (as the emission path uses) drops downside entirely when
    absent — absent, never a JSON null (mirrors confidence_provenance)."""
    r = OptionResultV2(id="a", outcome=_outcome(source="samples"), status="computed")
    dumped = r.model_dump(by_alias=True, exclude_none=True)
    assert "downside" not in dumped


def test_present_downside_serialises_all_three_fields():
    r = OptionResultV2(
        id="a",
        outcome=_outcome(source="samples"),
        downside=DownsideV2(cvar_10=70.0, p05=78.0, expected_regret=5.0),
        status="computed",
    )
    dumped = r.model_dump(by_alias=True, exclude_none=True)
    assert dumped["downside"] == {"cvar_10": 70.0, "p05": 78.0, "expected_regret": 5.0}
