"""SBC ground truth must retain the analyser's unavailable-outcome contract."""

from types import SimpleNamespace

import pytest

from src.models.robustness_v2 import GraphV2, InterventionOption
from src.services.robustness_analyzer_v2 import AccumulationDrawRefusedError
from src.services.sbc_validator import _compute_ground_truth


def _response(monkeypatch, mean, option_id="reference"):
    result = SimpleNamespace(
        option_id=option_id,
        outcome_distribution=SimpleNamespace(mean=mean),
    )
    monkeypatch.setattr(
        "src.services.sbc_validator.RobustnessAnalyzerV2.analyze",
        lambda self, request: SimpleNamespace(results=[result]),
    )


def _ground_truth():
    return _compute_ground_truth(
        GraphV2.model_construct(nodes=[], edges=[]),
        [InterventionOption(id="reference", label="Reference", interventions={})],
        "goal",
        "reference",
        42,
    )


def test_withheld_reference_outcome_raises_typed_refusal(monkeypatch):
    _response(monkeypatch, None)

    with pytest.raises(AccumulationDrawRefusedError, match="ground-truth outcome is unavailable"):
        _ground_truth()


def test_numeric_reference_outcome_is_unchanged(monkeypatch):
    _response(monkeypatch, 12.25)

    assert _ground_truth() == 12.25


def test_missing_reference_option_retains_existing_error(monkeypatch):
    _response(monkeypatch, 12.25, option_id="other")

    with pytest.raises(ValueError, match="Option 'reference' not found"):
        _ground_truth()
